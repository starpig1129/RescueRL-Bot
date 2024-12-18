import os
import signal
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy
from logger import TrainLog

class EnhancedEpisodeCallback(BaseCallback):
    def __init__(self, train_logger, save_freq=1, verbose=1):
        super(EnhancedEpisodeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.last_epoch = 0
        self._logger = train_logger
        self.episode_rewards = []
        self.recent_rewards = []
        self.n_calls = 0
        self.update_interval = 5  # 每5步更新一次顯示
        self.last_update_time = time.time()
        self.last_step = 0
    def _on_step(self) -> bool:
        self.n_calls += 1
        
        try:
            # 獲取當前訓練狀態
            current_epoch = self.training_env.get_attr('epoch')[0]
            current_step = self.num_timesteps
            current_time = time.time()
            
            # 更新獎勵歷史
            if self.locals.get('rewards') is not None:
                current_reward = self.locals['rewards'][0]
                self.recent_rewards.append(current_reward)
                if len(self.recent_rewards) > 1000:
                    self.recent_rewards.pop(0)
            
            # 計算平均獎勵
            mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
            
            # 計算FPS
            if (current_time - self.last_update_time) > 0:
                current_fps = (current_step - self.last_step) / (current_time - self.last_update_time)
            else:
                current_fps = 0
                
            # 更新訓練信息
            train_info = {
                'fps': current_fps,
                'total_timesteps': self.model.num_timesteps,
                'mean_reward': float(mean_reward),
                'step': current_step,
                'max_steps': self.model._total_timesteps
            }
            
            # 更新環境信息
            reward_list = (self.training_env.get_attr('last_reward_list')[0] 
                         if hasattr(self.training_env, 'get_attr') else None)
            
            # 更新logger
            self._logger.update_training_info(train_info)
            self._logger.update_env_info(current_epoch, current_step, reward_list)
            
            # 定期更新顯示
            if self.n_calls % self.update_interval == 0:
                self._logger.display()
                self.last_update_time = current_time
                self.last_step = current_step
            
            # 檢查是否需要保存模型
            if current_epoch != self.last_epoch and current_epoch % self.save_freq == 0:
                self._save_model(current_epoch)
                self.last_epoch = current_epoch
            
            return True
            
        except Exception as e:
            self._logger.log_error(e)
            return False

    def _on_rollout_end(self) -> None:
        """每個rollout結束時更新獎勵統計"""
        try:
            ep_info_buf = self.model.ep_info_buffer
            if ep_info_buf is not None and len(ep_info_buf) > 0:
                # 取得最近完成的回合資訊
                for ep_info in ep_info_buf:
                    if 'r' in ep_info:  # 'r' 是回合總獎勵
                        self.episode_rewards.append(ep_info['r'])
                
                # 保持最近100個回合的記錄
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
        except Exception as e:
            self._logger.log_error(e)
    
    def _save_model(self, epoch):
        """安全地將模型保存到檔案"""
        try:
            path = f"models/ppo_crawler_ep{epoch:03d}.zip"
            self.model.save(path)
            print(f"\n模型已保存: {path}")
        except Exception as e:
            self._logger.log_error(e)

def get_latest_epoch(model_dir="models"):
    """獲取最新的訓練世代號碼"""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return 0
            
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith("ppo_crawler_ep") and f.endswith(".zip")]
        if not model_files:
            return 0
            
        epochs = [int(f.split('_ep')[1].split('.')[0]) for f in model_files]
        return max(epochs)
    except Exception as e:
        print(f"讀取模型文件時發生錯誤: {e}")
        return 0

# 全局變量用於資源管理
env = None      # 訓練環境實例
model = None    # PPO模型實例
callback = None # 訓練回調實例
logger = None   # 日誌實例

def signal_handler(sig, frame):
    """處理中斷信號"""
    print('\n收到中斷信號 (Ctrl+C)，正在關閉...')
    try:
        if env is not None:
            current_epoch = env.epoch
            # 在callback存在且epoch有效時保存模型
            if callback is not None and current_epoch > 0 and hasattr(callback, '_save_model'):
                callback._save_model(current_epoch)
            env.close()
    except Exception as e:
        if logger is not None:
            logger.log_error(e)
        else:
            print(f"關閉時發生錯誤: {e}")
    finally:
        sys.exit(0)

# 註冊信號處理器
signal.signal(signal.SIGINT, signal_handler)

def main():
    """主訓練函數"""
    global env, model, callback, logger
    
    try:
        # 初始化日誌系統
        logger = TrainLog()
        
        # 獲取最新的訓練世代
        current_epoch = get_latest_epoch()
        print(f"從 epoch {current_epoch} 開始訓練")
        
        # 初始化訓練環境
        env = CrawlerEnv(
            show=False,              # 是否顯示視覺化界面
            epoch=current_epoch,     # 當前訓練世代
            test_mode=False,         # 是否為測試模式
            save_interval=50         # 數據保存間隔
        )
        
        # 設置PPO模型參數
        model_params = {
            "policy": CustomPolicy,         # 使用自定義策略網絡
            "env": env,                     # 訓練環境
            "verbose": 0,                   # 設為0以禁用進度條
            "learning_rate": 3e-4,          # 學習率
            "n_steps": 2048,               # 每次更新的步數
            "batch_size": 64,              # 批次大小
            "n_epochs": 10,                # 每次更新的訓練輪數
            "gamma": 0.99,                 # 折扣因子
            "gae_lambda": 0.95,            # GAE參數
            "clip_range": 0.2,             # PPO裁剪範圍
            "ent_coef": 0.01,              # 熵係數
            "vf_coef": 0.5,                # 價值函數係數
            "max_grad_norm": 0.5,          # 梯度裁剪閾值
            "use_sde": False,              # 是否使用狀態依賴探索
            "sde_sample_freq": 4,          # SDE採樣頻率
            "target_kl": 0.03,             # 目標KL散度
            "tensorboard_log": "./logs/",   # TensorBoard日誌目錄
        }
        
        # 檢查並載入現有模型或創建新模型
        latest_model_path = f"models/ppo_crawler_ep{current_epoch:03d}.zip"
        if os.path.exists(latest_model_path):
            print(f"載入之前的模型: {latest_model_path}")
            try:
                model = PPO.load(latest_model_path, env=env, verbose=0)  # 這裡也要設置 verbose=0
                model.policy.env = env
            except Exception as e:
                logger.log_error(e)
                print("創建新模型...")
                model = PPO(**model_params)
        else:
            print("創建新模型...")
            model = PPO(**model_params)
            model.policy.env = env

        # 初始化訓練回調
        callback = EnhancedEpisodeCallback(train_logger=logger, save_freq=1)
        
        # 設置總訓練步數
        total_timesteps = 10_00_000
        
        # 開始訓練循環
        print("\n開始訓練...")
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,      # 不重置步數計數器
            tb_log_name="PPO",             # TensorBoard日誌名稱
            callback=callback,              # 使用自定義回調
            progress_bar=False              # 禁用進度條
        )
        
        print("\n訓練完成！")
        
    except Exception as e:
        if logger is not None:
            logger.log_error(e)
        else:
            print(f"訓練過程中發生錯誤: {e}")
    finally:
        # 確保環境資源被正確釋放
        try:
            if env is not None:
                env.close()
        except Exception as e:
            if logger is not None:
                logger.log_error(e)
            else:
                print(f"清理資源時發生錯誤: {e}")

if __name__ == "__main__":
    main()