import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy

class EpisodeCallback(BaseCallback):
    """
    回調類：用於在訓練過程中監控和保存模型
    
    功能：
    - 追踪訓練的epoch
    - 在特定間隔保存模型檢查點
    - 提供訓練進度的日誌記錄
    """
    def __init__(self, save_freq=1, verbose=1):
        super(EpisodeCallback, self).__init__(verbose)
        self.save_freq = save_freq          # 保存模型的頻率（每N個epoch）
        self.last_epoch = 0                 # 記錄上一個處理的epoch
        
    def _on_step(self) -> bool:
        # 從訓練環境獲取當前epoch
        current_epoch = self.training_env.get_attr('epoch')[0]
        
        # 當進入新的epoch時執行保存檢查
        if current_epoch != self.last_epoch:
            print(f"\n新回合開始: Epoch {current_epoch}")
            
            # 根據保存頻率決定是否保存模型
            if current_epoch % self.save_freq == 0:
                self.save_model(current_epoch)
            
            self.last_epoch = current_epoch
        return True
    
    def save_model(self, epoch):
        """安全地將模型保存到檔案"""
        try:
            path = f"models/ppo_crawler_ep{epoch:03d}.zip"
            self.model.save(path)
            print(f"模型已保存: {path}")
        except Exception as e:
            print(f"保存模型時發生錯誤: {e}")
            
    def get_last_epoch(self):
        """獲取最後記錄的epoch編號"""
        return self.last_epoch

def get_latest_epoch(model_dir="models"):
    """
    從模型目錄中獲取最新的訓練世代號碼
    
    Args:
        model_dir: 模型存儲目錄路徑
    Returns:
        int: 最新的世代編號，如果沒有找到則返回0
    """
    try:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("ppo_crawler_ep") and f.endswith(".zip")]
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

def signal_handler(sig, frame):
    """
    處理中斷信號的處理器（如Ctrl+C）
    確保程序正確關閉並保存進度
    """
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    try:
        if env is not None:
            current_epoch = env.epoch
            # 在callback存在且epoch有效時保存模型
            if callback is not None and current_epoch > 0:
                callback.save_model(current_epoch)
            env.close()
    except Exception as e:
        print(f"關閉時發生錯誤: {e}")
    finally:
        sys.exit(0)

# 註冊信號處理器
signal.signal(signal.SIGINT, signal_handler)

def main():
    """
    主訓練循環
    
    功能：
    - 初始化訓練環境和模型
    - 設置訓練參數
    - 執行訓練循環
    - 處理異常情況並確保資源正確釋放
    """
    global env, model, callback
    
    try:
        # 獲取最新的訓練世代
        current_epoch = get_latest_epoch()
        print(f"從 epoch {current_epoch} 開始訓練")
        
        # 初始化訓練環境
        env = CrawlerEnv(
            show=False,              # 是否顯示視覺化界面
            epoch=current_epoch,     # 當前訓練世代
            test_mode=False,         # 是否為測試模式
            save_interval=10         # 數據保存間隔
        )
        
        # 設置PPO模型參數
        model_params = {
            "policy": CustomPolicy,         # 使用自定義策略網絡
            "env": env,                     # 訓練環境
            "verbose": 1,                   # 輸出詳細程度
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

        # 初始化訓練回調
        callback = EpisodeCallback(save_freq=1)
        
        # 設置總訓練步數
        total_timesteps = 1_000_000
        
        # 開始訓練循環
        print("\n開始訓練...")
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,      # 不重置步數計數器
            tb_log_name="PPO",             # TensorBoard日誌名稱
            callback=callback,              # 使用自定義回調
            progress_bar=True               # 顯示進度條
        )
        
        print("訓練完成！")
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保環境資源被正確釋放
        try:
            if env is not None:
                env.close()
        except Exception as e:
            print(f"清理資源時發生錯誤: {e}")

if __name__ == "__main__":
    main()