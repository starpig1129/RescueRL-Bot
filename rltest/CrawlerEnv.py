import gym
import numpy as np
import cv2
import torch
import os
import math
import signal
import sys
import time
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from reward.Reward2 import RewardFunction
from DataHandler import DataHandler
from torchvision import transforms 
from logger import TrainLog
from server_manager import ServerManager

# 影像標準化轉換
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class CrawlerEnv(gym.Env):
    def __init__(self, show, epoch=0, test_mode=False, feature_save_interval=100, image_save_interval=50, reward_save_interval=1):
        super(CrawlerEnv, self).__init__()
        
        # 步驟記錄相關
        self.steps_log_file = 'E:/train_log0118/debug_steps.txt'
        self.should_log_steps = False
        self.current_steps = []
        
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

        # 初始化伺服器管理器
        self.server_manager = ServerManager(logger=self.logger)
        try:
            self.server_manager.setup_all_servers()
            self.server_manager.set_epoch(self.epoch)
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")

    def step(self, action):
        try:
            self.step_count += 1
            step_log = []
            
            # 檢查是否需要記錄
            epoch_mod = self.epoch % 16
            self.should_log_steps = epoch_mod in [15, 0, 1]
            
            if self.should_log_steps:
                step_log.append(f"=== 世代 {self.epoch} 步驟 {self.step_count} ===")
                step_log.append(f"開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            current_time = time.time()
            dt = current_time - self.last_update_time
            if dt >= 1.0:
                self.fps = self.fps_counter / dt
                self.fps_counter = 0
                self.last_update_time = current_time
            self.fps_counter += 1
            
            # 接收獎勵資料
            max_retries = 3
            for retry in range(max_retries):
                try:
                    start_time = time.time()
                    reward_data = self.server_manager.receive_info()
                    if self.should_log_steps:
                        step_log.append(f"receive_info: 成功 ({(time.time() - start_time):.3f}秒)")
                    if reward_data is not None:
                        break
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_info: 重試 {retry + 1}/{max_retries}")
                        time.sleep(0.1)  # 短暫等待後重試
                        continue
                except Exception as e:
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_info: 錯誤重試 {retry + 1}/{max_retries} - {str(e)}")
                        time.sleep(0.1)
                        continue
                    if self.should_log_steps:
                        step_log.append(f"receive_info: 錯誤 - {str(e)}")
                        self._save_step_log(step_log)
                    return None, 0, True, {}
            
            if reward_data is None:
                if self.should_log_steps:
                    step_log.append("receive_info: 重試後仍然失敗，提前結束")
                    self._save_step_log(step_log)
                return None, 0, True, {}
            
            # 發送控制信號
            try:
                start_time = time.time()
                relative_angle = self.relative_angles[action]
                angle_rad = math.radians(relative_angle)
                self.server_manager.send_control_signal(angle_rad)
                if self.should_log_steps:
                    step_log.append(f"send_control_signal: 成功 ({(time.time() - start_time):.3f}秒)")
            except Exception as e:
                if self.should_log_steps:
                    step_log.append(f"send_control_signal: 錯誤 - {str(e)}")
                    self._save_step_log(step_log)
                return None, 0, True, {}
            
            # 接收和處理影像
            max_retries = 3
            for retry in range(max_retries):
                try:
                    start_time = time.time()
                    obs, origin_image = self.server_manager.receive_image(show=self.show)
                    if self.should_log_steps:
                        step_log.append(f"receive_image: 成功 ({(time.time() - start_time):.3f}秒)")
                    if obs is not None and origin_image is not None:
                        break
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_image: 重試 {retry + 1}/{max_retries}")
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_image: 錯誤重試 {retry + 1}/{max_retries} - {str(e)}")
                        time.sleep(0.1)
                        continue
                    if self.should_log_steps:
                        step_log.append(f"receive_image: 錯誤 - {str(e)}")
                        self._save_step_log(step_log)
                    return None, 0, True, {}
            
            if obs is None or origin_image is None:
                if self.should_log_steps:
                    step_log.append("receive_image: 重試後仍然失敗，提前結束")
                    self._save_step_log(step_log)
                return None, 0, True, {}
                
            # YOLO檢測
            results = self.YoloModel(
                obs, 
                imgsz=(640, 384), 
                device='cuda:0',
                conf=0.75, 
                classes=[0],
                show_boxes=False, 
                show=False
            )[0]

            # 繪製檢測框
            try:
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[0][:4])
                obs = cv2.rectangle(obs, (x1, y1), (x2, y2), (0, 255, 0), -1)
            except Exception:
                pass

            results = results.cpu().numpy()
            
            # 接收頂部攝影機影像
            max_retries = 3
            for retry in range(max_retries):
                try:
                    start_time = time.time()
                    top_view_image = self.server_manager.receive_top_camera_image(show=self.show)
                    if self.should_log_steps:
                        step_log.append(f"receive_top_camera_image: 成功 ({(time.time() - start_time):.3f}秒)")
                    if top_view_image is not None:
                        break
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_top_camera_image: 重試 {retry + 1}/{max_retries}")
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    if retry < max_retries - 1:
                        if self.should_log_steps:
                            step_log.append(f"receive_top_camera_image: 錯誤重試 {retry + 1}/{max_retries} - {str(e)}")
                        time.sleep(0.1)
                        continue
                    if self.should_log_steps:
                        step_log.append(f"receive_top_camera_image: 錯誤 - {str(e)}")
                        self._save_step_log(step_log)
                    return None, 0, True, {}
            
            if top_view_image is None:
                if self.should_log_steps:
                    step_log.append("receive_top_camera_image: 重試後仍然失敗，提前結束")
                    self._save_step_log(step_log)
                return None, 0, True, {}
                
            # 計算獎勵
            try:
                start_time = time.time()
                reward, reward_list = self.reward_function.get_reward(
                    detection_results=results,
                    reward_data=reward_data
                )
                if self.should_log_steps:
                    step_log.append(f"get_reward: 成功 ({(time.time() - start_time):.3f}秒)")
            except Exception as e:
                if self.should_log_steps:
                    step_log.append(f"get_reward: 錯誤 - {str(e)}")
                    self._save_step_log(step_log)
                return None, 0, True, {}
            
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
            
            # 儲存資料
            if self.should_save:
                try:
                    self.data_handler.save_step_data(
                        self.step_count,
                        self.epoch,
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
            
            # 預處理觀察
            try:
                start_time = time.time()
                processed_obs = self.preprocess_observation(obs)
                if self.should_log_steps:
                    step_log.append(f"preprocess_observation: 成功 ({(time.time() - start_time):.3f}秒)")
            except Exception as e:
                if self.should_log_steps:
                    step_log.append(f"preprocess_observation: 錯誤 - {str(e)}")
                    self._save_step_log(step_log)
                return None, 0, True, {}
            
            # 檢查是否需要重置
            try:
                start_time = time.time()
                self.done = self.server_manager.is_reset_triggered()
                if self.should_log_steps:
                    step_log.append(f"is_reset_triggered: 成功 ({(time.time() - start_time):.3f}秒)")
            except Exception as e:
                if self.should_log_steps:
                    step_log.append(f"is_reset_triggered: 錯誤 - {str(e)}")
                    self._save_step_log(step_log)
                return None, 0, True, {}
            
            # 如果需要記錄，保存這個步驟的日誌
            if self.should_log_steps:
                step_log.append(f"結束時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self._save_step_log(step_log)
            
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

    def preprocess_observation(self, obs):
        """處理觀察空間的數據"""
        obs = torch.from_numpy(obs).float() 
        obs = obs / 255.0
        obs = obs.permute(2, 0, 1)
        obs = normalize(obs)
        resize = transforms.Resize((224, 224))
        obs = resize(obs)
        return obs

    def get_layer_outputs(self):
        return self.layer_outputs

    def set_layer_outputs(self, outputs):
        self.layer_outputs = outputs

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
            self.start_time = time.time()
            self.min_distance = float('inf')
            
            # 重置獎勵函數
            self.reward_function.reset()
            
            self.should_save = True
            
            # 更新伺服器管理器的世代
            self.server_manager.set_epoch(self.epoch)
            
            if self.should_save:
                self.data_handler.create_epoch_file(self.epoch)
                if self.logger:
                    self.logger.log_info(f"將儲存第 {self.epoch} 個世代的資料")
            
            # 等待重置信號
            if self.server_manager.is_reset_triggered():
                self.server_manager.clear_reset_event()
            
            # 接收初始觀察
            obs, origin_image = self.server_manager.receive_image(show=self.show)
            if self.logger:
                self.logger.log_info("環境重置完成")
            
            return self.preprocess_observation(obs)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            return None

    def _save_step_log(self, log_lines):
        """保存步驟日誌"""
        with open(self.steps_log_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(log_lines) + '\n\n')
    
    def close(self):
        """關閉環境並釋放所有資源"""
        print("正在關閉環境...")
        
        try:
            # 儲存當前世代的訓練結果
            if hasattr(self, 'epoch') and self.epoch > 0:
                self._log_epoch_result()
        except Exception as e:
            print(f"儲存訓練結果時發生錯誤: {e}")

        # 關閉伺服器管理器
        if hasattr(self, 'server_manager'):
            self.server_manager.close()
            
        # 關閉資料處理器
        if hasattr(self, 'data_handler'):
            try:
                self.data_handler.close_epoch_file()
                print("資料處理器已關閉")
            except Exception as e:
                print(f"關閉資料處理器時發生錯誤: {e}")
            finally:
                self.data_handler = None
        
        # 關閉日誌系統
        if hasattr(self, 'logger'):
            try:
                self.logger.cleanup()
                print("日誌系統已關閉")
            except Exception as e:
                print(f"關閉日誌系統時發生錯誤: {e}")
            finally:
                self.logger = None
                
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
