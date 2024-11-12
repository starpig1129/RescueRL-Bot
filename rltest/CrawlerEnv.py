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
from reward.Reward import RewardFunction
from DataHandler import DataHandler
from torchvision import transforms 
from logger import TrainLog
# 影像標準化轉換
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class CrawlerEnv(gym.Env):
    """
    Crawler 環境
    實現了與 Unity 端的通信、影像處理、獎勵計算等功能
    """
    def __init__(self, show, epoch=0, test_mode=False, save_interval=1):
        """
        初始化 Crawler 環境
        
        參數:
            show (bool): 是否顯示即時畫面
            epoch (int): 當前訓練世代
            test_mode (bool): 是否為測試模式
            save_interval (int): 資料儲存間隔
        """
        super(CrawlerEnv, self).__init__()
        
        # 基本設置
        self.show = show
        self.test_mode = test_mode
        self.save_interval = save_interval
        self.should_save = False  # 是否儲存當前 epoch 的資料
        self.last_reward_list = None
        self.last_update_time = time.time()
        self.step_count = 0
        self.fps_counter = 0
        self.fps = 0
        
        # 動作與觀察空間定義
        self.action_space = gym.spaces.Discrete(9)  # 9個離散動作
        self.observation_space = gym.spaces.Box(    # 224x224 RGB 影像
            low=0.0, 
            high=1.0, 
            shape=(3, 224, 224), 
            dtype=np.float32
        )

        # 控制相關參數
        self.magnitude = 5.0        # 移動幅度
        self.angle_degrees = 90     # 初始角度
        
        # 模型與獎勵函數初始化
        self.YoloModel = YOLO('yolo/1110_skew.pt', verbose=False)  # YOLO 物件偵測模型
        self.reward_function = RewardFunction()               # 獎勵計算器
        self.layer_outputs = None                            # 神經網路層輸出暫存
        
        # Socket 伺服器相關變量初始化
        self._init_socket_vars()
        
        # 資料處理相關設置
        base_dir = "test_logs" if test_mode else "train_logs"
        self.logger = TrainLog()
        self.data_handler = DataHandler(base_dir=base_dir,logger=self.logger)
        self.epoch = epoch
        self.step_count = 0
        
        # 重置事件相關設置
        self.reset_event = threading.Event()
        self.reset_thread = None
        self.reset_thread_stop = threading.Event()

        # 建立所有伺服器連接
        try:
            self.setup_all_servers()
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")

    def _init_socket_vars(self):
        """初始化所有 Socket 相關變量"""
        self.control_socket = None  # 控制指令伺服器
        self.control_conn = None
        self.info_socket = None     # 資訊交換伺服器
        self.info_conn = None
        self.obs_socket = None      # 觀察資料伺服器
        self.obs_conn = None
        self.reset_socket = None    # 重置訊號伺服器
        self.reset_conn = None

    def setup_all_servers(self):
        """設置並啟動所有必要的伺服器"""
        self.setup_control_server()  # 控制伺服器 (埠 5000)
        self.setup_info_server()     # 資訊伺服器 (埠 8000)
        self.setup_obs_server()      # 觀察伺服器 (埠 6000)
        self.setup_reset_server()    # 重置伺服器 (埠 7000)

    def setup_control_server(self):
        """設置控制指令伺服器，接收動作控制指令"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control_address = ('localhost', 5000)
        
        try:
            self.control_socket.bind(self.control_address)
            self.control_socket.listen(5)
            print('控制伺服器已啟動，等待連接...')
            
            self.control_socket.settimeout(20)
            self.control_conn, self.control_addr = self.control_socket.accept()
            self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("已連接到控制伺服器:", self.control_addr)
        except Exception as e:
            raise Exception(f"控制伺服器設置失敗: {e}")

    def setup_info_server(self):
        """設置資訊交換伺服器，用於接收狀態資訊"""
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_address = ('localhost', 8000)
        
        try:
            self.info_socket.bind(self.info_address)
            self.info_socket.listen(5)
            print('資訊伺服器已啟動，等待連接...')
            
            self.info_socket.settimeout(10)
            self.info_conn, self.info_addr = self.info_socket.accept()
            self.info_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.info_conn.settimeout(5)
            print("已連接到資訊伺服器:", self.info_addr)
        except Exception as e:
            raise Exception(f"資訊伺服器設置失敗: {e}")

    def setup_obs_server(self):
        """設置觀察資料伺服器，接收影像資料"""
        self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.obs_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.obs_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.obs_address = ('localhost', 6000)
        
        try:
            self.obs_socket.bind(self.obs_address)
            self.obs_socket.listen(5)
            print("影像接收伺服器已啟動，等待連接...")
            
            self.obs_socket.settimeout(10)
            self.obs_conn, self.obs_addr = self.obs_socket.accept()
            self.obs_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("已連接到影像接收伺服器:", self.obs_addr)
        except Exception as e:
            raise Exception(f"影像接收伺服器設置失敗: {e}")

    def setup_reset_server(self):
        """設置重置訊號伺服器，處理環境重置"""
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_address = ('localhost', 7000)
        
        try:
            self.reset_socket.bind(self.reset_address)
            self.reset_socket.listen(5)
            print("重置訊號伺服器已啟動，等待連接...")
            self.reset_socket.settimeout(10)
            
            # 初始化重置相關事件
            self.reset_event = threading.Event()
            self.reset_thread_stop = threading.Event()
            
            # 啟動重置訊號處理線程
            self.reset_thread = threading.Thread(target=self.accept_reset_connections)
            self.reset_thread.daemon = True
            self.reset_thread.start()
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")

    def accept_reset_connections(self):
        """重置訊號接收線程函數"""
        while not self.reset_thread_stop.is_set():
            try:
                self.reset_socket.settimeout(1)
                reset_conn, reset_addr = self.reset_socket.accept()
                print("已連接到重置訊號發送端:", reset_addr)
                
                while not self.reset_thread_stop.is_set():
                    try:
                        # 接收重置訊號
                        signal_data = reset_conn.recv(4)
                        if not signal_data:
                            break
                        signal = int.from_bytes(signal_data, byteorder='little')
                        
                        if signal == 1:
                            # 發送當前訓練世代回 Unity
                            epoch_data = self.epoch.to_bytes(4, byteorder='little')
                            try:
                                reset_conn.send(epoch_data)
                                print(f"向 Unity 發送當前 epoch: {self.epoch}")
                            except Exception as e:
                                print(f"發送 epoch 到 Unity 時發生錯誤: {e}")
                            
                            self.reset_event.set()
                            print(f"收到重置訊號，當前 epoch: {self.epoch}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"接收重置訊號時發生錯誤: {e}")
                        break
                        
                reset_conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                if not self.reset_thread_stop.is_set():
                    print(f"重置連接發生錯誤: {e}")
                continue

    def step(self, action):
        """
        執行一個環境步驟
        
        參數:
            action (int): 要執行的動作索引 (0-8)
            
        返回:
            tuple: (observation, reward, done, info)
        """
        try:
            # 更新步數計數
            self.step_count += 1
            
            # 計算FPS
            current_time = time.time()
            dt = current_time - self.last_update_time
            if dt >= 1.0:  # 每秒更新一次FPS
                self.fps = self.fps_counter / dt
                self.fps_counter = 0
                self.last_update_time = current_time
            self.fps_counter += 1
            
            # 執行動作並獲取結果
            self.angle_degrees = action * 40
            self.send_control_signal()
            
            results, obs, origin_image = self.receive_image()
            if obs is None:
                return None, 0, True, {}
                
            reward_data = self.receive_data()
            if reward_data is None:
                return None, 0, True, {}
                
            # 計算獎勵
            reward, reward_list = self.reward_function.get_reward(
                results=results,
                reward_data=reward_data
            )
            
            # 更新獎勵列表
            self.last_reward_list = reward_list.copy() if reward_list is not None else None
            
            # 檢查是否需要保存數據
            if self.should_save:
                try:
                    self.data_handler.save_step_data(
                        self.step_count,
                        obs,
                        self.angle_degrees,
                        reward,
                        reward_list,
                        origin_image,
                        results,
                        self.layer_outputs
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(e)
            
            # 處理觀察數據
            processed_obs = self.preprocess_observation(obs)
            done = self.reset_event.is_set()
            
            return processed_obs, reward, done, {
                'reward_list': reward_list,
                'fps': self.fps,
                'step': self.step_count
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            return None, 0, True, {}

    def reset(self):
        try:
            # 關閉上一個世代的數據文件
            if self.epoch > 0 and self.should_save:
                self.data_handler.close_epoch_file()
            
            # 更新計數器
            self.epoch += 1
            self.step_count = 0
            self.fps_counter = 0
            self.fps = 0
            self.last_update_time = time.time()
            
            # 設置數據保存標誌
            self.should_save = (self.epoch % self.save_interval) == 0
            
            # 創建新的數據文件
            if self.should_save:
                self.data_handler.create_epoch_file(self.epoch)
                if self.logger:
                    self.logger.log_info(f"將儲存第 {self.epoch} 個世代的資料")
            else:
                if self.logger:
                    self.logger.log_info(f"跳過第 {self.epoch} 個世代的資料儲存")
            
            # 等待重置信號
            self.reset_event.wait()
            self.reset_event.clear()
            
            # 獲取初始觀察
            results, obs, origin_image = self.receive_image()
            if self.logger:
                self.logger.log_info("環境重置完成")
            
            return self.preprocess_observation(obs)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            return None

    def preprocess_observation(self, obs):
        """
        對觀察資料進行前處理
        
        參數:
            obs (ndarray): 原始觀察資料 (RGB 影像)
            
        返回:
            Tensor: 標準化後的 224x224 張量
        """
        # 轉換為 PyTorch 張量並正規化到 0-1 區間
        obs = torch.from_numpy(obs).float() 
        obs = obs / 255.0
        
        # 調整維度順序 (HWC -> CHW)
        obs = obs.permute(2, 0, 1)
        
        # 標準化
        obs = normalize(obs)
        
        # 調整大小至 224x224
        resize = transforms.Resize((224, 224))
        obs = resize(obs)
        
        return obs

    def send_control_signal(self):
        """
        向 Unity 發送控制訊號
        將角度轉換為 x,z 平面上的方向向量
        """
        try:
            # 確保角度在 0-360 度範圍內
            if self.angle_degrees >= 360:
                self.angle_degrees = 0

            # 將角度轉換為弧度並計算方向向量
            angle_radians = math.radians(self.angle_degrees)
            target_ford_x = self.magnitude * math.cos(angle_radians)
            target_ford_z = self.magnitude * math.sin(angle_radians)

            # 打包並發送資料
            buffer = struct.pack('ff', target_ford_x, target_ford_z)
            self.control_conn.sendall(buffer)

        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            self.control_conn.close()
            # 嘗試重新連接
            self.control_conn, self.control_addr = self.reconnect_socket(
                self.control_socket, 
                self.control_address, 
                '控制'
            )
            print("重新連接到控制伺服器:", self.control_addr)

    def receive_image(self):
        """
        從 Unity 接收影像資料並進行 YOLO 物件偵測
        
        返回:
            tuple: (YOLO 結果, 處理後影像, 原始影像)
        """
        try:
            # 接收影像長度
            image_len_bytes = self.recv_all(self.obs_conn, 4)
            if not image_len_bytes:
                return None, None, None
            image_len = int.from_bytes(image_len_bytes, byteorder='little')

            # 接收影像資料
            image_data = self.recv_all(self.obs_conn, image_len)
            if not image_data:
                return None, None, None

            # 將資料轉換為影像
            nparr = np.frombuffer(image_data, np.uint8)
            origin_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            obs = origin_image.copy()
            
            if obs is not None:
                # 使用 YOLO 模型進行物件偵測
                results = self.YoloModel(
                    obs, 
                    imgsz=(640, 384), 
                    device='cuda:0',
                    conf=0.75, 
                    classes=[0],  # 只偵測人類
                    show_boxes=False, 
                    show=False
                )[0]

                try:
                    # 在偵測到的人物位置繪製綠色遮罩
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                    obs = cv2.rectangle(obs, (x1, y1), (x2, y2), (0, 255, 0), -1)
                except Exception:
                    pass  # 忽略繪製失敗的錯誤

                # 顯示觀察畫面 (如果啟用)
                if self.show:
                    cv2.imshow("觀察空間", obs)
                    cv2.waitKey(1)

                # 將 YOLO 結果轉移到 CPU
                results = results.cpu().numpy()
                return results, obs, origin_image
            
            return None, None, None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            # 嘗試重新連接
            self.obs_conn, self.obs_addr = self.reconnect_socket(
                self.obs_socket, 
                self.obs_address, 
                '影像接收'
            )
            return None, None, None

    def receive_data(self):
        """
        從 Unity 接收狀態資料 (JSON 格式)
        
        返回:
            dict: 解析後的 JSON 資料
        """
        try:
            # 接收資料長度 (4 bytes, big-endian)
            length_bytes = self.recv_all(self.info_conn, 4)
            if not length_bytes:
                print("未接收到資料長度")
                return None

            length = int.from_bytes(length_bytes, byteorder='big', signed=True)
            
            # 檢查資料長度是否合理 (限制在 1MB 以內)
            if length <= 0 or length > 10**6:
                print(f"接收到無效的資料長度: {length}")
                return None

            # 接收完整資料
            data_bytes = self.recv_all(self.info_conn, length)
            if not data_bytes:
                print("未接收到完整資料")
                return None
            
            # 解析 JSON 資料
            try:
                data_str = data_bytes.decode('utf-8')
                return json.loads(data_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失敗: {e}")
                return None

        except socket.timeout:
            print("接收資料超時")
            return None
        except ConnectionResetError:
            print("連接被重置")
            self.info_conn, self.info_addr = self.reconnect_socket(
                self.info_socket, 
                self.info_address, 
                '資料'
            )
            return None
        except Exception as e:
            print(f"接收資料時發生錯誤: {e}")
            self.info_conn, self.info_addr = self.reconnect_socket(
                self.info_socket, 
                self.info_address, 
                '資料'
            )
            return None

    def reconnect_socket(self, socket_obj, address, conn_type):
        """
        重新建立 Socket 連接
        
        參數:
            socket_obj: Socket 物件
            address: 連接地址
            conn_type: 連接類型描述
            
        返回:
            tuple: (新連接, 新地址)
        """
        try:
            new_conn, new_addr = socket_obj.accept()
            new_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"重新連接到{conn_type}伺服器:", new_addr)
            return new_conn, new_addr
        except Exception as e:
            print(f"重新連接{conn_type}伺服器時發生錯誤: {e}")
            return None, None

    def recv_all(self, conn, length):
        """
        確保接收指定長度的完整資料
        
        參數:
            conn: Socket 連接
            length: 要接收的資料長度
            
        返回:
            bytearray: 接收到的資料
        """
        data = bytearray()
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def get_layer_outputs(self):
        """取得目前的神經網路層輸出"""
        return self.layer_outputs

    def set_layer_outputs(self, outputs):
        """設置神經網路層輸出"""
        self.layer_outputs = outputs

    def close(self):
        """
        關閉環境和所有連接
        清理所有資源
        """
        print("正在關閉環境...")
        
        # 停止重置訊號線程
        if hasattr(self, 'reset_thread_stop'):
            self.reset_thread_stop.set()
        
        # 等待重置線程結束
        if hasattr(self, 'reset_thread') and self.reset_thread is not None:
            self.reset_thread.join(timeout=1)
        
        # 關閉所有連接
        connections = [
            ('control_conn', 'control_socket'),
            ('info_conn', 'info_socket'),
            ('obs_conn', 'obs_socket'),
            ('reset_conn', 'reset_socket')
        ]
        
        for conn_attr, socket_attr in connections:
            # 關閉連接
            if hasattr(self, conn_attr) and getattr(self, conn_attr) is not None:
                try:
                    getattr(self, conn_attr).close()
                    print(f"{conn_attr} 已關閉")
                except Exception as e:
                    print(f"關閉 {conn_attr} 時發生錯誤: {e}")
                setattr(self, conn_attr, None)
            
            # 關閉 socket
            if hasattr(self, socket_attr) and getattr(self, socket_attr) is not None:
                try:
                    getattr(self, socket_attr).close()
                    print(f"{socket_attr} 已關閉")
                except Exception as e:
                    print(f"關閉 {socket_attr} 時發生錯誤: {e}")
                setattr(self, socket_attr, None)
        
        # 關閉資料處理器
        if hasattr(self, 'data_handler'):
            try:
                self.data_handler.close_epoch_file()
                print("資料處理器已關閉")
            except Exception as e:
                print(f"關閉資料處理器時發生錯誤: {e}")
        
        print("環境關閉完成")

# Ctrl+C 信號處理器
def signal_handler(sig, frame):
    """處理 Ctrl+C 中斷信號"""
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    env.close()
    sys.exit(0)

# 設置信號處理器
signal.signal(signal.SIGINT, signal_handler)

# 主程式進入點
if __name__ == "__main__":
    try:
        # 初始化環境
        current_epoch = 1
        env = CrawlerEnv(show=True, epoch=current_epoch)
        
        # 主迴圈
        while True:
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # 隨機動作測試
                obs, reward, done, info = env.step(action)
    finally:
        env.close()  # 確保資源被正確釋放