from stable_baselines3 import PPO
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy
# 創建環境實例
env = CrawlerEnv(show=False)

# 設定模型超參數
model_params = {
    "policy": CustomPolicy,  # 使用卷積神經網路(CNN)作為策略網路
    "env": env,  # 指定要訓練的環境
    "verbose": 2,  # 設定訊息輸出級別為1,輸出基本訊息
    "learning_rate": 2.5e-4,  # 設定學習率為2.5e-4
    "n_steps": 2048,  # 每次更新使用的步數設為2048
    "batch_size": 64,  # 設定批次大小為64
    "n_epochs": 10,  # 每次更新的訓練迭代次數設為10
    "gamma": 0.99,  # 設定折扣因子為0.99
    "gae_lambda": 0.95,  # 設定GAE參數為0.95
    "clip_range": 0.2,  # 設定PPO的限制範圍為0.2
    "ent_coef": 0.01,  # 設定交叉熵損失係數為0.01
    "vf_coef": 0.5,  # 設定值函數損失係數為0.5
    "max_grad_norm": 0.5,  # 設定梯度剪裁的最大範數為0.5
    "use_sde": False,  # 啟用隨機微分方程(SDE)
    "sde_sample_freq": 4,  # 設定SDE的採樣頻率為4
    "target_kl": 0.03,  # 設定目標KL散度為0.03
    "tensorboard_log": "./logs/",  # 設定TensorBoard日誌的儲存路徑
}

# 創建PPO模型,並傳入設定的超參數
model = PPO(**model_params)

# 設定訓練的總時間步數
total_timesteps = 1000000

# 設定檢查點的間隔時間步數
checkpoint_interval = 50000

# 設定評估的頻率(以時間步數為單位)
eval_freq = 10000

def main():
    # 開始訓練
    for i in range(int(total_timesteps / checkpoint_interval)):
        # 訓練一個checkpoint_interval的時間步數
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False, tb_log_name="PPO")
        
        # 儲存訓練得到的模型
        model.save(f"models/ppo_crawler_{(i+1)*checkpoint_interval}")
    # 訓練完成
    print("Training completed!")
    
if __name__ == "__main__":
    main()