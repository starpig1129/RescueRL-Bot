import gym
import numpy as np
import cv2
from ultralytics import YOLO
import socket
import struct
import math
import json
import threading
from reward.Reward_3 import RewardFunction
from DataHandler import DataHandler

class CrawlerEnv(gym.Env):
    def __init__(self, show):
        super(CrawlerEnv, self).__init__()
        self.show = show
        self.action_space = gym.spaces.Discrete(9)  # 定義動作空間，動作為 0 到 8 的整數
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(384, 640, 3), dtype=np.uint8)  # 定義觀察空間，影像大小為 (384, 640, 3)，像素值範圍為 0-255

        self.magnitude = 5.0  # 控制向量的長度（幅度）
        self.angle_degrees = 90  # 控制角度值（以度為單位）
        self.YoloModel = YOLO('yolo/best_1.pt', verbose=False)  # 載入訓練好的 YOLO 模型
        self.reward_function = RewardFunction()  # 建立獎勵函數物件

        self.data_handler = DataHandler(base_dir="train_logs")
        self.episode_counter = 0
        self.step_counter = 0

        # 建立控制訊號伺服器
        self.setup_control_server()

        # 建立數據回傳伺服器
        self.setup_info_server()

        # 建立影像接收伺服器
        self.setup_obs_server()

        # 建立重置訊號接收伺服器
        self.setup_reset_server()

    def setup_control_server(self):
        # 建立控制訊號伺服器
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用 Nagle 算法
        self.control_address = ('localhost', 5000)
        self.control_socket.bind(self.control_address)
        self.control_socket.listen(5)
        print('控制伺服器已啟動，等待連接...')
        self.control_conn, self.control_addr = self.control_socket.accept()
        self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("已連接到控制伺服器:", self.control_addr)

    def setup_info_server(self):
        # 建立數據回傳伺服器
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_address = ('localhost', 8000)
        self.info_socket.bind(self.info_address)
        self.info_socket.listen(5)
        print('數據伺服器已啟動，等待連接...')
        self.info_conn, self.info_addr = self.info_socket.accept()
        self.info_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("已連接到數據伺服器:", self.info_addr)

    def setup_obs_server(self):
        # 建立影像接收伺服器
        self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.obs_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.obs_address = ('localhost', 6000)
        self.obs_socket.bind(self.obs_address)
        self.obs_socket.listen(5)
        print("影像接收伺服器已啟動，等待連接...")
        self.obs_conn, self.obs_addr = self.obs_socket.accept()
        self.obs_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("已連接到影像接收伺服器:", self.obs_addr)

    def setup_reset_server(self):
        # 建立重置訊號接收伺服器
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_address = ('localhost', 7000)
        self.reset_socket.bind(self.reset_address)
        self.reset_socket.listen(5)
        print("重置訊號接收伺服器已啟動，等待連接...")

        self.reset_event = threading.Event()  # 建立事件物件用於同步重置訊號

        # 在一個新的執行緒中啟動重置訊號接收伺服器
        reset_thread = threading.Thread(target=self.accept_reset_connections)
        reset_thread.daemon = True
        reset_thread.start()

    def accept_reset_connections(self):
        while True:
            try:
                reset_conn, reset_addr = self.reset_socket.accept()
                print("已連接到重置訊號發送端:", reset_addr)
                while True:
                    data = reset_conn.recv(4)
                    if not data:
                        break
                    signal = int.from_bytes(data, byteorder='little')
                    print(f"接收到的重置訊號值: {signal}")
                    if signal == 1:
                        self.reset_event.set()  # 設置重置訊號
            except Exception as e:
                print(f"接收重置訊號時發生錯誤: {e}")
                break  # 出現異常時退出循環

    def step(self, action):
        # 將動作轉換為角度
        self.angle_degrees = action * 40

        # 發送控制訊號
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

        done = self.reset_event.is_set()  # 檢查是否接收到重置訊號

        # 儲存每一代的每一步數據
        self.data_handler.save_step_data(self.step_counter, obs, self.angle_degrees, reward, reward_list, origin_image, results)

        self.step_counter += 1

        print('reward', reward, done)
        return obs, reward, done, {}

    def reset(self):
        # 每次 reset 時創建新的 HDF5 檔案來儲存該世代數據
        self.episode_counter += 1
        self.step_counter = 0
        self.data_handler.create_epoch_file(self.episode_counter)

        # 等待重置訊號
        self.reset_event.wait()
        self.reset_event.clear()

        # 接收初始觀察
        results, obs, origin_image = self.receive_image()
        print("環境重置完成")
        return obs

    def send_control_signal(self):
        try:
            if self.angle_degrees >= 360:
                self.angle_degrees = 0

            angle_radians = math.radians(self.angle_degrees)  # 將角度轉換為弧度

            # 根據角度計算目標點的 x 和 z 坐標
            target_ford_x = self.magnitude * math.cos(angle_radians)
            target_ford_z = self.magnitude * math.sin(angle_radians)

            # 將浮點數轉換為位元組流並發送
            buffer = struct.pack('ff', target_ford_x, target_ford_z)
            self.control_conn.sendall(buffer)

        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            # 嘗試重新建立控制連接
            self.control_conn.close()
            self.control_conn, self.control_addr = self.control_socket.accept()
            self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("重新連接到控制伺服器:", self.control_addr)

    def receive_image(self):
        try:
            # 接收影像長度（前 4 個位元組）
            image_len_bytes = self.obs_conn.recv(4)
            if not image_len_bytes:
                return None, None, None
            image_len = int.from_bytes(image_len_bytes, byteorder='little')

            # 循環讀取影像資料直到接收到指定長度
            image_data = b''
            while len(image_data) < image_len:
                packet = self.obs_conn.recv(min(image_len - len(image_data), 4096))
                if not packet:
                    return None, None, None
                image_data += packet

            if not image_data:
                return None, None, None

            nparr = np.frombuffer(image_data, np.uint8)
            origin_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if origin_image is not None:
                results = self.YoloModel(origin_image, imgsz=(640, 384), device='cpu',
                            conf=0.7,
                            classes=[0],  # 只偵測人的類別編號
                            show_boxes=False,  # 不顯示偵測框
                            show_labels=False,  # 不顯示標籤
                            show_conf=False,  # 不顯示置信度
                            )[0] 
                try:
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                    obs = cv2.rectangle(results.orig_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                except:
                    obs = origin_image
                # 顯示接收到的影像
                if obs is not None and self.show:
                    cv2.imshow("觀察空間", obs)
                    cv2.waitKey(1)
                return results, obs ,origin_image
            else:
                return None, None, None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            # 嘗試重新建立影像連接
            self.obs_conn.close()
            self.obs_conn, self.obs_addr = self.obs_socket.accept()
            self.obs_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("重新連接到影像接收伺服器:", self.obs_addr)
            return None, None, None

    def receive_data(self):
        try:
            # 讀取消息長度（前 4 個位元組）
            length_bytes = self.info_conn.recv(4)
            if not length_bytes:
                return None

            # 確定消息的長度
            length = int.from_bytes(length_bytes, byteorder='little')

            # 循環讀取資料直到接收到指定長度
            data = b''
            while len(data) < length:
                packet = self.info_conn.recv(length - len(data))
                if not packet:
                    return None
                data += packet

            data = data.decode('utf-8')

            # 反序列化 JSON 資料
            json_data = json.loads(data)

            # 處理目標資訊...
            return json_data

        except Exception as e:
            print(f"接收數據時發生錯誤: {e}")
            # 嘗試重新建立數據連接
            self.info_conn.close()
            self.info_conn, self.info_addr = self.info_socket.accept()
            self.info_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("重新連接到數據伺服器:", self.info_addr)
            return None

    def render(self, mode='human'):
        pass

    def close(self):
        # 關閉重置訊號接收伺服器
        self.reset_socket.close()
        # 清理資源，關閉連線
        self.control_conn.close()
        self.control_socket.close()
        self.obs_conn.close()
        self.obs_socket.close()
        self.info_conn.close()
        self.info_socket.close()
        
        self.data_handler.close_epoch_file()
