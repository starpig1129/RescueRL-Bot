import gym
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import socket
import struct
import math
import json
import threading
import signal
import sys
import time
from reward.Reward_3 import RewardFunction
from DataHandler import DataHandler
from torchvision import transforms 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
class CrawlerEnv(gym.Env):
    def __init__(self, show, epoch=0, test_mode=False):
        super(CrawlerEnv, self).__init__()
        self.show = show
        self.test_mode = test_mode
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32)

        self.magnitude = 5.0
        self.angle_degrees = 90
        self.YoloModel = YOLO('yolo/best_1.pt', verbose=False)
        self.reward_function = RewardFunction()

        # 初始化 socket 相關變量
        self.control_socket = None
        self.control_conn = None
        self.info_socket = None
        self.info_conn = None
        self.obs_socket = None
        self.obs_conn = None
        self.reset_socket = None
        self.reset_conn = None
        
        # 根據測試模式選擇不同的儲存目錄
        base_dir = "test_logs" if test_mode else "train_logs"
        self.data_handler = DataHandler(base_dir=base_dir)
        self.epoch = epoch
        self.step_counter = 0
        self.reset_event = threading.Event()
        self.reset_thread = None
        self.reset_thread_stop = threading.Event()

        # 建立伺服器連接
        try:
            self.setup_all_servers()
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")
        
    def setup_all_servers(self):
        """設置所有伺服器"""
        self.setup_control_server()
        self.setup_info_server()
        self.setup_obs_server()
        self.setup_reset_server()
        
    def setup_control_server(self):
        """設置控制伺服器"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control_address = ('localhost', 5000)
        try:
            self.control_socket.bind(self.control_address)
            self.control_socket.listen(5)
            print('控制伺服器已啟動，等待連接...')
            self.control_socket.settimeout(10)  # 設置超時
            self.control_conn, self.control_addr = self.control_socket.accept()
            self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("已連接到控制伺服器:", self.control_addr)
        except Exception as e:
            raise Exception(f"控制伺服器設置失敗: {e}")

    def setup_info_server(self):
        """設置數據伺服器"""
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_address = ('localhost', 8000)
        try:
            self.info_socket.bind(self.info_address)
            self.info_socket.listen(5)
            print('數據伺服器已啟動，等待連接...')
            self.info_socket.settimeout(10)  # 設置超時
            self.info_conn, self.info_addr = self.info_socket.accept()
            self.info_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.info_conn.settimeout(5)
            print("已連接到數據伺服器:", self.info_addr)
        except Exception as e:
            raise Exception(f"數據伺服器設置失敗: {e}")

    def setup_obs_server(self):
        """設置觀察伺服器"""
        self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.obs_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.obs_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.obs_address = ('localhost', 6000)
        try:
            self.obs_socket.bind(self.obs_address)
            self.obs_socket.listen(5)
            print("影像接收伺服器已啟動，等待連接...")
            self.obs_socket.settimeout(10)  # 設置超時
            self.obs_conn, self.obs_addr = self.obs_socket.accept()
            self.obs_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("已連接到影像接收伺服器:", self.obs_addr)
        except Exception as e:
            raise Exception(f"影像接收伺服器設置失敗: {e}")

    def setup_reset_server(self):
        """設置重置伺服器"""
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_address = ('localhost', 7000)
        try:
            self.reset_socket.bind(self.reset_address)
            self.reset_socket.listen(5)
            print("重置訊號接收伺服器已啟動，等待連接...")
            self.reset_socket.settimeout(10)  # 設置超時
            
            # 重置相關的事件標誌
            self.reset_event = threading.Event()
            self.reset_thread_stop = threading.Event()
            
            # 啟動重置訊號接收線程
            self.reset_thread = threading.Thread(target=self.accept_reset_connections)
            self.reset_thread.daemon = True
            self.reset_thread.start()
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")

    def accept_reset_connections(self):
        """處理重置連接的線程函數"""
        while not self.reset_thread_stop.is_set():
            try:
                print('連線重置訊號發送端中')
                self.reset_socket.settimeout(1)  # 短暫的超時，以便能夠檢查停止標誌
                reset_conn, reset_addr = self.reset_socket.accept()
                print("已連接到重置訊號發送端:", reset_addr)
                
                while not self.reset_thread_stop.is_set():
                    try:
                        data = reset_conn.recv(4)
                        if not data:
                            break
                        signal = int.from_bytes(data, byteorder='little')
                        if signal == 1:
                            self.reset_event.set()
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                        
                reset_conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                if not self.reset_thread_stop.is_set():
                    print(f"重置連接發生錯誤: {e}")
                continue
            
    def preprocess_observation(self, obs):
        # Convert from numpy array to torch tensor
        obs = torch.from_numpy(obs).float()
        # Normalize the image
        obs = obs / 255.0
        # Permute dimensions from HWC to CHW
        obs = obs.permute(2, 0, 1)
        # Apply normalization
        obs = normalize(obs)
        resize = transforms.Resize((224, 224))
        obs = resize(obs)
        return obs

    def step(self, action):
        try:
            self.angle_degrees = action * 40
            self.send_control_signal()

            # 接收影像並進行 YOLO 處理
            results, obs, origin_image = self.receive_image()
            if obs is None:
                print("未能接收到有效的觀察數據，結束當前 episode。")
                done = True
                reward = 0
                return obs, reward, done, {}

            reward_data = self.receive_data()
            if reward_data is None:
                print("未能接收到有效的獎勵數據，結束當前 episode。")
                done = True
                reward = 0
                return obs, reward, done, {}

            # 計算獎勵
            reward, reward_list = self.reward_function.get_reward(results=results, reward_data=reward_data)
            done = self.reset_event.is_set()

            # 儲存每一代的每一步數據
            self.data_handler.save_step_data(self.step_counter, obs, self.angle_degrees, reward, reward_list, origin_image, results)
            self.step_counter += 1
            
            # In step and reset methods
            obs = self.preprocess_observation(obs)
                
            return obs, reward, done, {}

        except Exception as e:
            print(f"步驟執行時發生錯誤: {e}")
            return None, 0, True, {}

    def reset(self):
        if self.epoch > 0:
            self.data_handler.close_epoch_file()

        self.epoch += 1
        self.step_counter = 0
        # 創建一個帶有當前 epoch 的 HDF5 檔案
        self.data_handler.create_epoch_file(self.epoch)

        # 等待重置訊號
        self.reset_event.wait()
        self.reset_event.clear()

        # 接收初始觀察
        results, obs, origin_image = self.receive_image()
        print("環境重置完成")
        obs = self.preprocess_observation(obs)
        return obs

    def send_control_signal(self):
        try:
            if self.angle_degrees >= 360:
                self.angle_degrees = 0

            angle_radians = math.radians(self.angle_degrees)
            target_ford_x = self.magnitude * math.cos(angle_radians)
            target_ford_z = self.magnitude * math.sin(angle_radians)

            buffer = struct.pack('ff', target_ford_x, target_ford_z)
            self.control_conn.sendall(buffer)

        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            self.control_conn.close()
            self.control_conn, self.control_addr = self.reconnect_socket(self.control_socket, self.control_address, '控制')
            print("重新連接到控制伺服器:", self.control_addr)

    def reconnect_socket(self, socket_obj, address, conn_type):
        """ 重新連接伺服器 """
        try:
            new_conn, new_addr = socket_obj.accept()
            new_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"重新連接到 {conn_type} 伺服器:", new_addr)
            return new_conn, new_addr
        except Exception as e:
            print(f"重新連接 {conn_type} 伺服器時發生錯誤: {e}")
            return None, None

    def recv_all(self, conn, length):
        """ 確保接收指定長度的數據 """
        data = bytearray()
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive_image(self):
        try:
            # 接收影像長度資料
            image_len_bytes = self.recv_all(self.obs_conn, 4)
            if not image_len_bytes:
                return None, None, None
            image_len = int.from_bytes(image_len_bytes, byteorder='little')

            # 接收影像資料
            image_data = self.recv_all(self.obs_conn, image_len)
            if not image_data:
                return None, None, None

            # 將資料轉換為圖像
            nparr = np.frombuffer(image_data, np.uint8)
            origin_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 複製圖像作為 YOLO 的輸入
            obs = origin_image.copy()
            
            if obs is not None:
                # 使用 YOLO 模型進行推理 (使用 CUDA)
                results = self.YoloModel(obs, imgsz=(640, 384), device='cuda:0',
                                        conf=0.7, classes=[0], show_boxes=False)[0]
                try:
                    # 獲取 YOLO 模型的檢測框並確保範圍正確
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                    obs = cv2.rectangle(obs, (x1, y1), (x2, y2), (0, 255, 0), -1)  # 填滿矩形
                except Exception as e:
                    pass

                # 如果開啟顯示，則顯示觀察空間
                if obs is not None and self.show:
                    cv2.imshow("觀察空間", obs)
                    cv2.waitKey(1)
                results = results.cpu().numpy()
                # 返回 YOLO 結果、處理過的圖像與原始圖像
                return results, obs, origin_image
            else:
                return None, None, None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            # 重新連接影像接收的 Socket
            self.obs_conn, self.obs_addr = self.reconnect_socket(self.obs_socket, self.obs_address, '影像接收')
            return None, None, None


    def receive_data(self):
        try:
            # 接收 4 個位元組的資料長度，使用大端序
            length_bytes = self.recv_all(self.info_conn, 4)
            if not length_bytes:
                print("未接收到資料長度位元組")
                return None

            # 將位元組轉換為整數（大端序）
            length = int.from_bytes(length_bytes, byteorder='big', signed=True)
            print(f"接收到的資料長度：{length} 位元組")
            
            # 檢查資料長度是否在合理範圍內，避免無效或過大的資料長度
            if length <= 0 or length > 10**6:  # 假設資料長度限制為 1MB
                print(f"接收到的長度無效或過大: {length}")
                return None

            # 接收實際的資料
            data_bytes = self.recv_all(self.info_conn, length)
            if not data_bytes:
                print("未接收到完整的資料")
                return None

            print(f"實際接收到的資料長度：{len(data_bytes)} 位元組")
            
            try:
                # 解碼並解析 JSON 資料
                data_str = data_bytes.decode('utf-8')
                json_data = json.loads(data_str)
                return json_data
            except json.JSONDecodeError as e:
                print(f"JSON 解析失敗: {e}")
                return None

        except socket.timeout:
            print("接收數據超時")
            return None
        except ConnectionResetError:
            print("連接被重置")
            self.info_conn, self.info_addr = self.reconnect_socket(self.info_socket, self.info_address, '數據')
            return None
        except Exception as e:
            print(f"接收數據時發生錯誤: {e}")
            self.info_conn, self.info_addr = self.reconnect_socket(self.info_socket, self.info_address, '數據')
            return None

    def render(self, mode='human'):
        pass

    def close(self):
        """關閉環境和所有連接"""
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
        
        # 關閉數據處理器
        if hasattr(self, 'data_handler'):
            try:
                self.data_handler.close_epoch_file()
                print("數據處理器已關閉")
            except Exception as e:
                print(f"關閉數據處理器時發生錯誤: {e}")
        
        print("環境關閉完成")


def signal_handler(sig, frame):
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    env.close()
    sys.exit(0)


# 設置信號處理器來處理 Ctrl+C 中斷
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        # 初始化環境時傳入當前世代(epoch)
        current_epoch = 1  # 這裡可以設置為當前的 epoch，或通過訓練腳本動態傳入
        env = CrawlerEnv(show=True, epoch=current_epoch)
        # 主迴圈，可以隨時按下 Ctrl+C 結束
        while True:
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # 隨機動作
                obs, reward, done, info = env.step(action)
    finally:
        env.close()  # 確保資源被正確釋放
