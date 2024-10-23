import gym
import numpy as np
import cv2
from ultralytics import YOLO
import socket
import struct
import math
import json
import threading
import signal
import sys
from reward.Reward_3 import RewardFunction
from DataHandler import DataHandler


class CrawlerEnv(gym.Env):
    def __init__(self, show):
        super(CrawlerEnv, self).__init__()
        self.show = show
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(384, 640, 3), dtype=np.uint8)

        self.magnitude = 5.0
        self.angle_degrees = 90
        self.YoloModel = YOLO('yolo/best_1.pt', verbose=False)
        self.reward_function = RewardFunction()

        self.data_handler = DataHandler(base_dir="train_logs")
        self.episode_counter = 0
        self.step_counter = 0

        # 建立伺服器
        self.setup_control_server()
        self.setup_info_server()
        self.setup_obs_server()
        self.setup_reset_server()

    def setup_control_server(self):
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control_address = ('localhost', 5000)
        self.control_socket.bind(self.control_address)
        self.control_socket.listen(5)
        print('控制伺服器已啟動，等待連接...')
        self.control_conn, self.control_addr = self.control_socket.accept()
        self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("已連接到控制伺服器:", self.control_addr)

    def setup_info_server(self):
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_address = ('localhost', 8000)
        self.info_socket.bind(self.info_address)
        self.info_socket.listen(5)
        print('數據伺服器已啟動，等待連接...')
        self.info_conn, self.info_addr = self.info_socket.accept()
        self.info_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("已連接到數據伺服器:", self.info_addr)
        # 設置接收超時時間，例如 5 秒
        self.info_conn.settimeout(5)

    def setup_obs_server(self):
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
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_address = ('localhost', 7000)
        self.reset_socket.bind(self.reset_address)
        self.reset_socket.listen(5)
        print("重置訊號接收伺服器已啟動，等待連接...")

        self.reset_event = threading.Event()  # 用於同步重置訊號

        # 在一個新的執行緒中啟動重置訊號接收伺服器
        reset_thread = threading.Thread(target=self.accept_reset_connections)
        reset_thread.daemon = True
        reset_thread.start()

    def accept_reset_connections(self):
        while True:
            try:
                print('連線重置訊號發送端中')
                reset_conn, reset_addr = self.reset_socket.accept()
                print("已連接到重置訊號發送端:", reset_addr)
                while True:
                    data = reset_conn.recv(4)
                    if not data:
                        break
                    signal = int.from_bytes(data, byteorder='little')
                    print(f"接收到的重置訊號值: {signal}")
                    if signal == 1:
                        self.reset_event.set()
            except Exception as e:
                print(f"接收重置訊號時發生錯誤: {e}")
                break

    def step(self, action):
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
        done = self.reset_event.is_set()

        # 儲存每一代的每一步數據
        self.data_handler.save_step_data(self.step_counter, obs, self.angle_degrees, reward, reward_list, origin_image, results)
        self.step_counter += 1

        return obs, reward, done, {}

    def reset(self):
        if self.episode_counter > 0:
            self.data_handler.close_epoch_file()

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

            angle_radians = math.radians(self.angle_degrees)
            target_ford_x = self.magnitude * math.cos(angle_radians)
            target_ford_z = self.magnitude * math.sin(angle_radians)

            buffer = struct.pack('ff', target_ford_x, target_ford_z)
            self.control_conn.sendall(buffer)

        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            self.control_conn.close()
            self.control_conn, self.control_addr = self.control_socket.accept()
            self.control_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("重新連接到控制伺服器:", self.control_addr)

    def recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

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

            if origin_image is not None:
                results = self.YoloModel(origin_image, imgsz=(640, 384), device='cpu',
                                         conf=0.7,
                                         classes=[0],
                                         show_boxes=False)[0]
                try:
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                    obs = cv2.rectangle(results.orig_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                except:
                    obs = origin_image.copy()

                if obs is not None and self.show:
                    cv2.imshow("觀察空間", obs)
                    cv2.waitKey(1)

                return results, obs, origin_image
            else:
                return None, None, None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            self.obs_conn.close()
            self.obs_conn, self.obs_addr = self.obs_socket.accept()
            self.obs_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("重新連接到影像接收伺服器:", self.obs_addr)
            return None, None, None

    def receive_data(self):
        try:
            # 接收 4 個位元組的資料長度，使用大端序
            length_bytes = self.recv_all(self.info_conn, 4)
            if not length_bytes:
                return None

            # 將位元組轉換為整數（大端序）
            length = int.from_bytes(length_bytes, byteorder='big', signed=True)
            print(f"接收到的資料長度：{length} 位元組")
            if length <= 0:
                print(f"接收到的長度無效: {length}")
                return None

            # 接收實際的資料
            data_bytes = self.recv_all(self.info_conn, length)
            if not data_bytes:
                print("未接收到完整的資料")
                return None

            print(f"實際接收到的資料長度：{len(data_bytes)} 位元組")
            data_str = data_bytes.decode('utf-8')
            json_data = json.loads(data_str)
            return json_data

        except socket.timeout:
            print("接收數據超時")
            return None
        except Exception as e:
            print(f"接收數據時發生錯誤: {e}")
            self.info_conn.close()
            return None


    def render(self, mode='human'):
        pass

    def close(self):
        # 關閉所有連接
        if hasattr(self, 'reset_socket') and self.reset_socket:
            self.reset_socket.close()
        if hasattr(self, 'control_conn') and self.control_conn:
            self.control_conn.close()
        if hasattr(self, 'control_socket') and self.control_socket:
            self.control_socket.close()
        if hasattr(self, 'obs_conn') and self.obs_conn:
            self.obs_conn.close()
        if hasattr(self, 'obs_socket') and self.obs_socket:
            self.obs_socket.close()
        if hasattr(self, 'info_conn') and self.info_conn:
            self.info_conn.close()
        if hasattr(self, 'info_socket') and self.info_socket:
            self.info_socket.close()

        # 關閉 HDF5 檔案
        self.data_handler.close_epoch_file()

        print("所有伺服器與檔案已關閉")


def signal_handler(sig, frame):
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    env.close()
    sys.exit(0)


# 設置信號處理器來處理 Ctrl+C 中斷
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        env = CrawlerEnv(show=True)
        # 主迴圈，可以隨時按下 Ctrl+C 結束
        while True:
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # 隨機動作
                obs, reward, done, info = env.step(action)
    finally:
        env.close()  # 確保資源被正確釋放
