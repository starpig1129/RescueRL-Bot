import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy

class EpisodeCallback(BaseCallback):
    """
    在每個回合結束時進行檢查和更新的回調
    """
    def __init__(self, save_freq=1, verbose=1):
        super(EpisodeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.last_epoch = 0
        
    def _on_step(self) -> bool:
        # 直接從環境獲取當前epoch
        current_epoch = self.training_env.get_attr('epoch')[0]
        
        # 如果epoch發生變化，表示新的回合開始
        if current_epoch != self.last_epoch:
            print(f"\n新回合開始: Epoch {current_epoch}")
            
            # 根據實際epoch決定是否保存
            if current_epoch % self.save_freq == 0:
                self.save_model(current_epoch)
            
            self.last_epoch = current_epoch
            
        return True
    
    def save_model(self, epoch):
        """安全地保存模型"""
        try:
            path = f"models/ppo_crawler_ep{epoch:03d}.zip"
            self.model.save(path)
            print(f"模型已保存: {path}")
        except Exception as e:
            print(f"保存模型時發生錯誤: {e}")
            
    def get_last_epoch(self):
        """獲取最後記錄的epoch"""
        return self.last_epoch

def get_latest_epoch(model_dir="models"):
    """獲取當前最新的世代號碼"""
    try:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("ppo_crawler_ep") and f.endswith(".zip")]
        if not model_files:
            return 0
        epochs = [int(f.split('_ep')[1].split('.')[0]) for f in model_files]
        return max(epochs)
    except Exception as e:
        print(f"讀取模型文件時發生錯誤: {e}")
        return 0

# 全局變量
env = None
model = None
callback = None

def signal_handler(sig, frame):
    """處理 Ctrl+C 信號"""
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    try:
        if env is not None:
            current_epoch = env.epoch
            # 只在callback存在且epoch有效時保存
            if callback is not None and current_epoch > 0:
                callback.save_model(current_epoch)
            env.close()
    except Exception as e:
        print(f"關閉時發生錯誤: {e}")
    finally:
        sys.exit(0)

# 設置信號處理器
signal.signal(signal.SIGINT, signal_handler)

def main():
    global env, model, callback
    
    try:
        # 從最新的epoch開始
        current_epoch = get_latest_epoch()
        print(f"從 epoch {current_epoch} 開始訓練")
        
        # 初始化環境
        env = CrawlerEnv(show=False, epoch=current_epoch, test_mode=False, save_interval=10)
        
        # 修改模型參數確保適當的訓練步驟
        model_params = {
            "policy": CustomPolicy,
            "env": env,
            "verbose": 1,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": 4,
            "target_kl": 0.03,
            "tensorboard_log": "./logs/",
        }
        
        # 檢查是否有之前的模型
        latest_model_path = f"models/ppo_crawler_ep{current_epoch:03d}.zip"
        if os.path.exists(latest_model_path):
            print(f"載入之前的模型: {latest_model_path}")
            try:
                model = PPO.load(latest_model_path, env=env)
                model.policy.env = env
            except Exception as e:
                print(f"載入模型失敗: {e}")
                print("創建新模型...")
                model = PPO(**model_params)
        else:
            print("創建新模型...")
            model = PPO(**model_params)
            model.policy.env = env

        # 創建回調
        callback = EpisodeCallback(save_freq=1)
        
        # 訓練參數
        total_timesteps = 1_000_000
        
        # 訓練循環
        print("\n開始訓練...")
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            callback=callback,
            progress_bar=True
        )
        
        print("訓練完成！")
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保資源正確釋放
        try:
            if env is not None:
                env.close()
        except Exception as e:
            print(f"清理資源時發生錯誤: {e}")

if __name__ == "__main__":
    main()