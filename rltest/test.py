import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import zipfile
import io
import sys
from policy import PretrainedResNet, CustomActor
from CrawlerEnv import CrawlerEnv
from DataHandler import DataHandler

class TestPolicy:
    """測試策略類別，用於載入和執行訓練好的模型"""
    
    def __init__(self, state_dim=(3, 224, 224), action_dim=9, features_dim=512):
        """
        初始化測試策略
        
        參數:
            state_dim (tuple): 輸入狀態的維度，預設為 (3, 224, 224)
            action_dim (int): 動作空間的維度，預設為 9
            features_dim (int): 特徵維度，預設為 512
        """
        # 初始化特徵提取器（使用預訓練的ResNet模型）
        self.features_extractor = PretrainedResNet(observation_space=None, features_dim=512)
        
        # 初始化演員網路（用於決策動作）
        self.actor = CustomActor(features_dim, action_dim)
        
        # 儲存最新的層輸出
        self.layer_outputs = None
    
    def load_state_dict(self, state_dict):
        """
        從狀態字典中載入模型權重
        
        參數:
            state_dict (dict): 包含模型權重的狀態字典
        """
        try:
            # 載入特徵提取器權重
            extractor_dict = {
                k.replace('features_extractor.extractor.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('features_extractor.extractor.')
            }
            
            if extractor_dict:
                print("正在載入特徵提取器權重...")
                self.features_extractor.extractor.load_state_dict(extractor_dict)
            
            # 載入演員網路權重
            actor_dict = {
                k.replace('action_net.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('action_net.')
            }
            
            if actor_dict:
                print("正在載入演員網路權重...")
                self.actor.load_state_dict(actor_dict)
            
            print("所有權重載入成功")
            
        except Exception as e:
            print(f"載入權重時發生錯誤: {e}")
            print("狀態字典的鍵:", state_dict.keys())
            raise
    
    def to(self, device):
        """將模型移動到指定設備（CPU/GPU）"""
        self.features_extractor = self.features_extractor.to(device)
        self.actor = self.actor.to(device)
        return self
    
    def eval(self):
        """將模型設置為評估模式"""
        self.features_extractor.eval()
        self.actor.eval()
    
    def predict(self, observation, device):
        """
        根據觀察預測動作
        
        參數:
            observation: 環境觀察值
            device: 計算設備（CPU/GPU）
        
        返回:
            預測的動作值
        """
        with torch.no_grad():
            # 確保觀察值是正確設備上的張量
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, dtype=torch.float32)
            observation = observation.to(device)
            
            # 增加批次維度（如果需要）
            if observation.dim() == 3:
                observation = observation.unsqueeze(0)
            
            # 提取特徵並儲存層輸出
            features = self.features_extractor(observation)
            
            # 獲取動作邏輯
            action_logits = self.actor(features)
            
            # 更新層輸出
            self.layer_outputs = {
                'input': self.features_extractor.layer_outputs['input'],
                'conv1_output': self.features_extractor.layer_outputs['conv1_output'],
                'final_residual_output': self.features_extractor.layer_outputs['final_residual_output'],
                'features_output': self.features_extractor.layer_outputs['features_output'],
                'actor_output': action_logits.detach().cpu().numpy()
            }
            
            # 選擇最高機率的動作
            action = torch.argmax(action_logits, dim=1)
            return action.item()

def test_episode(env, policy, episode_num, device, render=False):
    """
    測試單個回合
    
    參數:
        env: 環境實例
        policy: 策略實例
        episode_num: 回合編號
        device: 計算設備
        render: 是否渲染環境
    
    返回:
        total_reward: 總獎勵
        steps: 總步數
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = policy.predict(obs, device)
        env.set_layer_outputs(policy.layer_outputs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
            
    return total_reward, steps

def load_model_from_zip(model_path, device):
    """
    從ZIP檔案中載入模型權重
    
    參數:
        model_path: ZIP檔案路徑
        device: 計算設備
    
    返回:
        載入權重後的策略實例
    """
    try:
        print(f"正在從 {model_path} 載入模型...")
        
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            # 檢查ZIP檔案內容
            file_list = zip_ref.namelist()
            print(f"ZIP檔案中的內容: {file_list}")
            
            # 尋找權重檔案
            pytorch_vars = [f for f in file_list if 'pytorch_variables.pth' in f]
            if not pytorch_vars:
                raise ValueError(f"在 {model_path} 中找不到權重檔案")
            
            print(f"找到權重檔案: {pytorch_vars[0]}")
            
            # 讀取權重檔案
            with zip_ref.open(pytorch_vars[0]) as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer, map_location=device)
            
            print("權重檔案內容的鍵:", state_dict.keys())
            
            # 創建並配置策略實例
            policy = TestPolicy()
            policy.to(device)
            policy.load_state_dict(state_dict)
            policy.eval()
            
            print("模型載入成功")
            return policy
            
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        raise

def find_model_files(models_dir="models"):
    """
    找到所有模型檔案並按照世代排序
    
    參數:
        models_dir: 模型目錄路徑
    
    返回:
        排序後的模型檔案路徑列表
    """
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    model_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
    return [os.path.join(models_dir, f) for f in model_files]

def main():
    """主程式入口"""
    # 設置命令列參數解析
    parser = argparse.ArgumentParser(description='Crawler 模型測試程式')
    parser.add_argument('--episodes', type=int, default=5,
                      help='每個模型要運行的測試回合數 (預設: 5)')
    parser.add_argument('--model', type=str, default=None,
                      help='指定的模型路徑 (預設: 測試所有模型)')
    parser.add_argument('--render', type=bool, default=False,
                      help='是否顯示視覺化界面')
    parser.add_argument('--device', type=str, default='cuda',
                      help='使用的設備 (預設: cuda)')
    
    args = parser.parse_args()
    
    # 設置計算設備
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用設備: {device}")
    
    # 確定要測試的模型路徑
    if args.model:
        model_paths = [args.model]
    else:
        try:
            model_paths = find_model_files()
        except Exception as e:
            print(f"尋找模型檔案時發生錯誤: {e}")
            sys.exit(1)
    
    if not model_paths:
        print("沒有找到要測試的模型")
        sys.exit(1)
    
    print(f"將要測試的模型數量: {len(model_paths)}")
    
    # 初始化環境
    try:
        env = CrawlerEnv(show=args.render, test_mode=True)
    except Exception as e:
        print(f"初始化環境時發生錯誤: {e}")
        sys.exit(1)
    
    try:
        # 測試每個模型
        for model_path in model_paths:
            print(f"\n測試模型: {model_path}")
            model_name = os.path.basename(model_path).split('.')[0]
            
            try:
                # 載入模型
                policy = load_model_from_zip(model_path, device)
                
                # 運行測試回合
                episode_rewards = []
                episode_steps = []
                
                for episode in range(args.episodes):
                    reward, steps = test_episode(env, policy, episode, device, args.render)
                    episode_rewards.append(reward)
                    episode_steps.append(steps)
                    print(f"回合 {episode + 1}: 獎勵 = {reward:.2f}, 步數 = {steps}")
                    
                # 計算並輸出統計數據
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_steps = np.mean(episode_steps)
                std_steps = np.std(episode_steps)
                
                print(f"\n模型 {model_path} 的測試結果:")
                print(f"平均獎勵: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"平均步數: {mean_steps:.2f} ± {std_steps:.2f}")
                
            except Exception as e:
                print(f"測試模型 {model_path} 時發生錯誤: {e}")
                continue
        
        print("\n所有模型測試完成！")
        
    except Exception as e:
        print(f"執行測試時發生錯誤: {e}")
    finally:
        # 確保環境被正確關閉
        try:
            env.close()
            print("環境已關閉")
        except Exception as e:
            print(f"關閉環境時發生錯誤: {e}")
        
        sys.exit(0)

if __name__ == "__main__":
    main()