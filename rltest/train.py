import os
import signal
import sys
from stable_baselines3 import PPO
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy

# 全局變量 env 和 model，這樣可以在 signal_handler 函數中訪問
env = None
model = None

# 確保 'logs' 和 'models' 目錄存在
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 設置模型參數
model_params = {
    "policy": CustomPolicy,
    "env": None,  # 暫時設為 None，稍後會設置為 env
    "verbose": 2,
    "learning_rate": 2.5e-4,
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

def signal_handler(sig, frame):
    """處理 Ctrl+C 信號，確保環境關閉"""
    print('收到中斷信號 (Ctrl+C)，正在關閉...')
    if env is not None:
        env.close()  # 確保調用環境的關閉函數
    if model is not None:
        model.save("models/ppo_crawler_last_checkpoint")  # 保存最後的檢查點
    sys.exit(0)  # 正常退出程式

# 設置信號處理器來處理 Ctrl+C 中斷
signal.signal(signal.SIGINT, signal_handler)

def main():
    global env, model
    env = CrawlerEnv(show=False)  # 創建環境
    model_params['env'] = env  # 將環境傳遞給模型

    # 檢查是否有之前保存的模型
    model_path = "models/ppo_crawler_last_checkpoint.zip"
    if os.path.exists(model_path):
        print("發現之前的模型，正在從檢查點加載...")
        model = PPO.load(model_path, env=env)  # 加載之前的模型，並繼續訓練
    else:
        print("未找到之前的模型，從頭開始訓練...")
        model = PPO(**model_params)

    total_timesteps = 1_000_000
    checkpoint_interval = 2048
    
    # 訓練模型並保存檢查點
    for i in range(int(total_timesteps / checkpoint_interval)):
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"models/ppo_crawler_{(i+1)*checkpoint_interval}")
    
    print("訓練完成！")

if __name__ == "__main__":
    try:
        main()
    finally:
        if env is not None:
            env.close()  # 確保無論如何都關閉環境
        if model is not None:
            model.save("models/ppo_crawler_last_checkpoint")  # 最後保存檢查點
