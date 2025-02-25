import gym
import numpy as np
import cv2
import torch
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import socket
import struct
import math
import json
import threading
import signal
import sys
import time
import asyncio
from reward.Reward2 import RewardFunction  # 改用 Reward2
from DataHandler import DataHandler
from torchvision import transforms 
from logger import TrainLog

# 影像標準化轉換
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class CrawlerEnv(gym.Env):
    def __init__(self, show, epoch=0, test_mode=False, feature_save_interval=100, image_save_interval=50, reward_save_interval=1):
        super(CrawlerEnv, self).__init__()
        
        # 基本設置
        self.show = show
        self.test_mode = test_mode
        self.feature_save_interval = feature_save_interval
        self.image_save_interval = image_save_interval
        self.reward_save_interval = reward_save_interval
        self.should_save = False
        self.last_reward_list = None
        self.last_update_time = time.time()
        self.step_count = 0
        self.fps_counter = 0
        self.fps = 0
        self.done = False
        self.min_distance = float('inf')  # 追蹤最小距離
        
        # 任務完成記錄
        self.success_log_file = 'E:/train_log0118/training_results.csv'
        self.found_target = False
        self.success_step = 0
        self.start_time = None  # 初始化世代開始時間變數
        
        # 如果記錄檔案不存在，則建立並寫入標題列
        if not os.path.exists(self.success_log_file):
            with open(self.success_log_file, 'w', encoding='utf-8') as f:
                f.write("世代,是否成功,總步數,成功步數,執行時間,最小距離\n")
        
        # 動作空間: 0=左轉(-45°), 1=直走(0°), 2=右轉(45°)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(3, 224, 224), 
            dtype=np.float32
        )

        # 控制相關參數
        self.relative_angles = {
            0: -45,  # 左轉
            1: 0,    # 直走
            2: 45    # 右轉
        }
        
        # 模型與獎勵函數初始化
        self.YoloModel = YOLO('yolo/1220gray.pt', verbose=False)
        self.reward_function = RewardFunction()
        self.layer_outputs = None
        
        # Socket 伺服器相關變量初始化
        self._init_socket_vars()
        
        # 資料處理相關設置
        base_dir = "test_logs" if test_mode else "E:/train_log0118/train_log"
        self.logger = TrainLog()
        self.data_handler = DataHandler(
            base_dir=base_dir,
            logger=self.logger,
            feature_save_interval=self.feature_save_interval,
            image_save_interval=self.image_save_interval,
            reward_save_interval=self.reward_save_interval
        )
        self.epoch = epoch
        self.step_count = 0
        
        if self.epoch == 0:
            self.data_handler.create_epoch_file(self.epoch)
        
        # 重置事件相關設置
        self.reset_event = threading.Event()
        self.reset_thread = None
        self.reset_thread_stop = threading.Event()

        try:
            self.setup_all_servers()
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")

    def _init_socket_vars(self):
        self.control_socket = None
        self.control_conn = None
        self.info_socket = None
        self.info_conn = None
        self.obs_socket = None
        self.obs_conn = None
        self.reset_socket = None
        self.reset_conn = None
        self.top_camera_socket = None
        self.top_camera_conn = None
        
        # 非同步相關變量
        self.loop = None
        self.tasks = []  # 存放所有非同步任務

    def setup_all_servers(self):
        # 創建或獲取事件循環
        self.loop = asyncio.get_event_loop()
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # 使用run_until_complete運行所有伺服器設置任務
        self.loop.run_until_complete(asyncio.gather(
            self.setup_control_server(),
            self.setup_info_server(),
            self.setup_obs_server(),
            self.setup_reset_server(),
            self.setup_top_camera_server()
        ))

    async def setup_top_camera_server(self):
        """設置頂部攝影機伺服器"""
        self.top_camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.top_camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.top_camera_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.top_camera_address = ('localhost', 9000)
        self.top_camera_socket.setblocking(False)
        
        try:
            self.top_camera_socket.bind(self.top_camera_address)
            self.top_camera_socket.listen(5)
            print("頂部攝影機伺服器已啟動，等待連接...")
            
            # 使用非同步方式接受連接
            conn, addr = await self.loop.sock_accept(self.top_camera_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 將非阻塞socket轉換為阻塞模式進行後續操作
            conn.setblocking(True)
            
            self.top_camera_conn = conn
            self.top_camera_addr = addr
            print("已連接到頂部攝影機:", addr)
        except Exception as e:
            raise Exception(f"頂部攝影機伺服器設置失敗: {e}")

    def receive_top_camera_image(self):
        """接收頂部攝影機的影像"""
        try:
            image_len_bytes = self.recv_all(self.top_camera_conn, 4)
            if not image_len_bytes:
                return None

            image_len = int.from_bytes(image_len_bytes, byteorder='little')
            image_data = self.recv_all(self.top_camera_conn, image_len)
            if not image_data:
                return None

            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if self.show:
                cv2.imshow("頂部視角", image)
                cv2.waitKey(1)
                
            return image

        except Exception as e:
            print(f"接收頂部攝影機影像時發生錯誤: {e}")
            self.top_camera_conn, self.top_camera_addr = self.reconnect_socket(
                self.top_camera_socket,
                self.top_camera_address,
                '頂部攝影機'
            )
            return None

    async def setup_control_server(self):
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control_address = ('localhost', 5000)
        self.control_socket.setblocking(False)
        
        try:
            self.control_socket.bind(self.control_address)
            self.control_socket.listen(5)
            print('控制伺服器已啟動，等待連接...')
            
            # 使用非同步方式接受連接
            conn, addr = await self.loop.sock_accept(self.control_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 將非阻塞socket轉換為阻塞模式進行後續操作
            conn.setblocking(True)
            
            self.control_conn = conn
            self.control_addr = addr
            print("已連接到控制伺服器:", addr)
        except Exception as e:
            raise Exception(f"控制伺服器設置失敗: {e}")

    async def setup_info_server(self):
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_address = ('localhost', 8000)
        self.info_socket.setblocking(False)
        
        try:
            self.info_socket.bind(self.info_address)
            self.info_socket.listen(5)
            print('資訊伺服器已啟動，等待連接...')
            
            # 使用非同步方式接受連接
            conn, addr = await self.loop.sock_accept(self.info_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 將非阻塞socket轉換為阻塞模式進行後續操作
            conn.setblocking(True)
            conn.settimeout(1)  # 降低超時時間以更快檢測連接問題
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)  # 設置接收緩衝區大小
            
            self.info_conn = conn
            self.info_addr = addr
            print("已連接到資訊伺服器:", addr)
        except Exception as e:
            raise Exception(f"資訊伺服器設置失敗: {e}")

    async def setup_obs_server(self):
        self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.obs_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.obs_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.obs_address = ('localhost', 6000)
        self.obs_socket.setblocking(False)
        
        try:
            self.obs_socket.bind(self.obs_address)
            self.obs_socket.listen(5)
            print("影像接收伺服器已啟動，等待連接...")
            
            # 使用非同步方式接受連接
            conn, addr = await self.loop.sock_accept(self.obs_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 將非阻塞socket轉換為阻塞模式進行後續操作
            conn.setblocking(True)
            
            self.obs_conn = conn
            self.obs_addr = addr
            print("已連接到影像接收伺服器:", addr)
        except Exception as e:
            raise Exception(f"影像接收伺服器設置失敗: {e}")

    async def setup_reset_server(self):
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_address = ('localhost', 7000)
        self.reset_socket.setblocking(False)
        
        try:
            self.reset_socket.bind(self.reset_address)
            self.reset_socket.listen(5)
            print("重置訊號伺服器已啟動，等待連接...")
            
            self.reset_event = asyncio.Event()
            
            # 啟動非同步重置信號處理任務
            self.reset_task = asyncio.create_task(self.accept_reset_connections())
            # 將任務添加到列表中以防止垃圾回收
            self.tasks.append(self.reset_task)
            
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")

    async def accept_reset_connections(self):
        """非同步版本的重置連接處理"""
        while True:  # 持續監聽重置連接
            try:
                # 非同步接受連接
                reset_conn, reset_addr = await self.loop.sock_accept(self.reset_socket)
                print("已連接到重置訊號發送端:", reset_addr)
                
                # 設置為非阻塞模式以進行非同步操作
                reset_conn.setblocking(False)
                
                while True:
                    try:
                        # 非同步接收數據
                        signal_data = await self.loop.sock_recv(reset_conn, 4)
                        if not signal_data:
                            break
                            
                        signal = int.from_bytes(signal_data, byteorder='little')
                        
                        if signal == 1:
                            epoch_data = self.epoch.to_bytes(4, byteorder='little')
                            try:
                                # 非同步發送
                                await self.loop.sock_sendall(reset_conn, epoch_data)
                                print(f"向 Unity 發送當前 epoch: {self.epoch}")
                            except Exception as e:
                                print(f"發送 epoch 到 Unity 時發生錯誤: {e}")
                            
                            # 使用非同步事件
                            self.reset_event.set()
                            print(f"收到重置訊號，當前 epoch: {self.epoch}")
                            
                    except Exception as e:
                        print(f"接收重置訊號時發生錯誤: {e}")
                        break
                        
                reset_conn.close()
            except Exception as e:
                print(f"重置連接發生錯誤: {e}")
                # 短暫休眠以防止CPU佔用過高
                await asyncio.sleep(0.1)

    def step(self, action):
        try:
            self.step_count += 1
            
            current_time = time.time()
            dt = current_time - self.last_update_time
            if dt >= 1.0:
                self.fps = self.fps_counter / dt
                self.fps_counter = 0
                self.last_update_time = current_time
            self.fps_counter += 1
            
            reward_data = self.receive_data()
            if reward_data is None:
                return None, 0, True, {}
            
            relative_angle = self.relative_angles[action]
            
            # 發送相對角度到Unity
            self.send_control_signal(relative_angle)
            
            results, obs, origin_image = self.receive_image()
            if obs is None:
                return None, 0, True, {}
            
            # 接收頂部攝影機影像
            top_view_image = self.receive_top_camera_image()
                
            reward, reward_list = self.reward_function.get_reward(
                detection_results=results,
                reward_data=reward_data,
            )
            
            # 檢查是否找到目標
            if reward_list[10] > 0:
                self.found_target = True
                self.success_step = self.step_count
            
            # 更新最小距離
            crawler_pos = reward_data['position']
            target_pos, _ = self.reward_function.find_nearest_target(reward_data)
            current_distance = np.sqrt((crawler_pos['x'] - target_pos[0])**2 + (crawler_pos['z'] - target_pos[1])**2)
            self.min_distance = min(self.min_distance, current_distance)
            
            self.last_reward_list = reward_list.copy() if reward_list is not None else None
            
            if self.should_save:
                try:
                    self.data_handler.save_step_data(
                        self.step_count,
                        obs,
                        relative_angle,
                        reward,
                        reward_list,
                        origin_image,  
                        results,
                        self.layer_outputs,
                        top_view_image  
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(e)
            
            processed_obs = self.preprocess_observation(obs)
            self.done = self.reset_event.is_set()
            
            return processed_obs, reward, self.done, {
                'reward_list': reward_list,
                'fps': self.fps,
                'step': self.step_count,
                'relative_angle': relative_angle
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            return None, 0, True, {}

    def _log_epoch_result(self):
        """記錄世代結果"""
        with open(self.success_log_file, 'a', encoding='utf-8') as f:
            success_str = "1" if self.found_target else "0"
            success_step_str = str(self.success_step) if self.found_target else ""
            duration = time.time() - self.start_time if self.start_time else 0
            f.write(f"{self.epoch},{success_str},{self.step_count},{success_step_str},{duration:.1f},{self.min_distance:.1f}\n")

    def reset(self):
        try:
            # 在世代結束時記錄結果
            if self.epoch > 0:
                self._log_epoch_result()
                if self.should_save:
                    self.data_handler.close_epoch_file()
            self.epoch += 1
            self.step_count = 0
            self.fps_counter = 0
            self.fps = 0
            self.last_update_time = time.time()
            
            # 重置任務完成狀態
            self.found_target = False
            self.success_step = 0
            self.start_time = time.time()  # 記錄世代開始時間
            self.min_distance = float('inf')  # 重置最小距離
            
            # 重置獎勵函數的狀態
            self.reward_function.reset()
            
            self.should_save = True  # 移除儲存間隔檢查，由DataHandler控制儲存頻率
            
            if self.should_save:
                self.data_handler.create_epoch_file(self.epoch)
                if self.logger:
                    self.logger.log_info(f"將儲存第 {self.epoch} 個世代的資料")
            else:
                if self.logger:
                    self.logger.log_info(f"跳過第 {self.epoch} 個世代的資料儲存")

            # 處理非同步事件和傳統事件
            if isinstance(self.reset_event, threading.Event):
                self.reset_event.wait()
                self.reset_event.clear()
            elif isinstance(self.reset_event, asyncio.Event):
                # 在同步環境中等待非同步事件
                fut = asyncio.run_coroutine_threadsafe(self._wait_for_reset_event(), self.loop)
                fut.result()  # 等待非同步事件完成
            
            results, obs, origin_image = self.receive_image()
            if self.logger:
                self.logger.log_info("環境重置完成")
            
            return self.preprocess_observation(obs)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            return None
    
    async def _wait_for_reset_event(self):
        """等待非同步重置事件"""
        await self.reset_event.wait()
        self.reset_event.clear()

    def preprocess_observation(self, obs):
        obs = torch.from_numpy(obs).float() 
        obs = obs / 255.0
        obs = obs.permute(2, 0, 1)
        obs = normalize(obs)
        resize = transforms.Resize((224, 224))
        obs = resize(obs)
        return obs

    def send_control_signal(self, relative_angle):
        """
        向 Unity 發送控制訊號
        只發送相對角度，讓Unity端計算實際位置
        
        參數:
            relative_angle: 相對角度 (-45°, 0°, 45°)
        """
        try:
            # 將角度轉換為弧度並打包
            angle_rad = math.radians(relative_angle)
            buffer = struct.pack('f', angle_rad)
            self.control_conn.sendall(buffer)

        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            self.control_conn.close()
            self.control_conn, self.control_addr = self.reconnect_socket(
                self.control_socket, 
                self.control_address, 
                '控制'
            )
            print("重新連接到控制伺服器:", self.control_addr)

    def receive_image(self):
        try:
            image_len_bytes = self.recv_all(self.obs_conn, 4)
            if not image_len_bytes:
                return None, None, None
            image_len = int.from_bytes(image_len_bytes, byteorder='little')

            image_data = self.recv_all(self.obs_conn, image_len)
            if not image_data:
                return None, None, None

            nparr = np.frombuffer(image_data, np.uint8)
            origin_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            obs = origin_image.copy()
            
            if obs is not None:
                results = self.YoloModel(
                    obs, 
                    imgsz=(640, 384), 
                    device='cuda:0',
                    conf=0.75, 
                    classes=[0],
                    show_boxes=False, 
                    show=False
                )[0]

                try:
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                    obs = cv2.rectangle(obs, (x1, y1), (x2, y2), (0, 255, 0), -1)
                except Exception:
                    pass

                if self.show:
                    cv2.imshow("觀察空間", obs)
                    cv2.waitKey(1)

                results = results.cpu().numpy()
                return results, obs, origin_image
            
            return None, None, None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            self.obs_conn, self.obs_addr = self.reconnect_socket(
                self.obs_socket, 
                self.obs_address, 
                '影像接收'
            )
            return None, None, None

    def receive_data(self):
        max_retries = 3
        retry_delay = 0.01  # 10ms 延遲
        
        for retry_count in range(max_retries):
            try:
                length_bytes = self.recv_all(self.info_conn, 4)
                if not length_bytes:
                    if retry_count < max_retries - 1:
                        print(f"未接收到資料長度 (重試 {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return None

                length = int.from_bytes(length_bytes, byteorder='big', signed=True)
                
                if length <= 0 or length > 10**6:
                    print(f"接收到無效的資料長度: {length}")
                    if retry_count < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

                data_bytes = self.recv_all(self.info_conn, length)
                if not data_bytes:
                    if retry_count < max_retries - 1:
                        print(f"未接收到完整資料 (重試 {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return None
                
                try:
                    data_str = data_bytes.decode('utf-8')
                    return json.loads(data_str)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失敗: {e}")
                    if retry_count < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

            except socket.timeout:
                if retry_count < max_retries - 1:
                    print(f"接收資料超時 (重試 {retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return None
            except ConnectionResetError:
                print("連接被重置，嘗試重新連接...")
                self.info_conn, self.info_addr = self.reconnect_socket(
                    self.info_socket,
                    self.info_address,
                    '資訊'
                )
                if retry_count < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            except Exception as e:
                print(f"接收資料時發生錯誤: {e}")
                if retry_count < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None  # 如果所有重試都失敗，返回None

    def reconnect_socket(self, socket_obj, address, conn_type):
        try:
            new_conn, new_addr = socket_obj.accept()
            new_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"重新連接到{conn_type}伺服器:", new_addr)
            return new_conn, new_addr
        except Exception as e:
            print(f"重新連接{conn_type}伺服器時發生錯誤: {e}")
            return None, None

    def recv_all(self, conn, length):
        data = bytearray()
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def get_layer_outputs(self):
        return self.layer_outputs

    def set_layer_outputs(self, outputs):
        self.layer_outputs = outputs

    def close(self):
        print("正在關閉環境...")
        
        # 關閉非同步任務
        if hasattr(self, 'tasks') and self.tasks:
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # 如果有事件循環，執行所有未完成的任務
            if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
                try:
                    # 等待所有任務完成或被取消
                    pending = asyncio.all_tasks(self.loop)
                    if pending:
                        self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception as e:
                    print(f"關閉非同步任務時發生錯誤: {e}")
        
        # 關閉傳統線程
        if hasattr(self, 'reset_thread_stop'):
            self.reset_thread_stop.set()
        
        if hasattr(self, 'reset_thread') and self.reset_thread is not None:
            self.reset_thread.join(timeout=1)
        
        # 關閉所有連接和套接字
        connections = [
            ('control_conn', 'control_socket'),
            ('info_conn', 'info_socket'),
            ('obs_conn', 'obs_socket'),
            ('reset_conn', 'reset_socket'),
            ('top_camera_conn', 'top_camera_socket')
        ]
        
        for conn_attr, socket_attr in connections:
            if hasattr(self, conn_attr) and getattr(self, conn_attr) is not None:
                try:
                    getattr(self, conn_attr).close()
                    print(f"{conn_attr} 已關閉")
                except Exception as e:
                    print(f"關閉 {conn_attr} 時發生錯誤: {e}")
                setattr(self, conn_attr, None)
            
            if hasattr(self, socket_attr) and getattr(self, socket_attr) is not None:
                try:
                    getattr(self, socket_attr).close()
                    print(f"{socket_attr} 已關閉")
                except Exception as e:
                    print(f"關閉 {socket_attr} 時發生錯誤: {e}")
                setattr(self, socket_attr, None)
        
        # 關閉事件循環
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            try:
                self.loop.close()
                print("事件循環已關閉")
            except Exception as e:
                print(f"關閉事件循環時發生錯誤: {e}")
        
        # 關閉資料處理器
        if hasattr(self, 'data_handler'):
            try:
                self.data_handler.close_epoch_file()
                print("資料處理器已關閉")
            except Exception as e:
                print(f"關閉資料處理器時發生錯誤: {e}")
        
        print("環境關閉完成")

def signal_handler(sig, frame):
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    env.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        current_epoch = 1
        env = CrawlerEnv(show=True, epoch=current_epoch)
        
        while True:
            obs = env.reset()
            env.done = False
            while not env.done:
                action = env.action_space.sample()
                obs, reward, env.done, info = env.step(action)
    finally:
        env.close()
