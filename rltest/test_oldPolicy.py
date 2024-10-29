import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import zipfile
import io
from policy import CustomResNet, CustomActor
from CrawlerEnv import CrawlerEnv
from DataHandler import DataHandler

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class TestResNet(CustomResNet):
    def __init__(self, observation_space=None, features_dim=512):
        super(TestResNet, self).__init__(observation_space, features_dim)
        self.layer_outputs = {}

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 儲存輸入
        self.layer_outputs['input'] = observations.detach().cpu().numpy()

        # 前向傳播並儲存中間層輸出
        x = self.conv1(observations)
        x = self.bn1(x)
        x = self.relu(x)
        self.layer_outputs['conv1_output'] = x.detach().cpu().numpy()
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self.layer_outputs['final_residual_output'] = x.detach().cpu().numpy()

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        self.layer_outputs['feature_output'] = features.detach().cpu().numpy()

        return features

class TestPolicy:
    def __init__(self, state_dim=(3, 224, 224), action_dim=9, features_dim=512):
        # 初始化特徵提取器，保持與原始架構一致
        self.features_extractor = TestResNet(features_dim=features_dim)
        
        # 動作網路
        self.actor = CustomActor(features_dim, action_dim)
        
    def load_state_dict(self, state_dict):
        try:
            # 載入特徵提取器的權重
            extractor_dict = {}
            for k, v in state_dict.items():
                if k.startswith('features_extractor.extractor.'):
                    new_key = k.replace('features_extractor.extractor.', '')
                    extractor_dict[new_key] = v
            
            if extractor_dict:
                print("載入特徵提取器權重...")
                self.features_extractor.load_state_dict(extractor_dict)
            
            # 載入 actor 網路的權重
            actor_dict = {}
            for k, v in state_dict.items():
                if k.startswith('action_net.'):
                    new_key = k.replace('action_net.', '')
                    actor_dict[new_key] = v
            
            if actor_dict:
                print("載入 actor 網路權重...")
                self.actor.load_state_dict(actor_dict)
            
            print("成功載入所有權重")
            
        except Exception as e:
            print(f"載入權重時發生錯誤: {e}")
            print("State dict keys:", state_dict.keys())
            raise
        
    def to(self, device):
        self.features_extractor = self.features_extractor.to(device)
        self.actor = self.actor.to(device)
        return self
        
    def eval(self):
        self.features_extractor.eval()
        self.actor.eval()
        
    def predict(self, observation, device):
        with torch.no_grad():
            # 確保觀察值是tensor且在正確的設備上
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, dtype=torch.float32)
            observation = observation.to(device)
            
            if observation.dim() == 3:
                observation = observation.unsqueeze(0)
                
            # 使用特徵提取器並獲取中間層輸出
            features = self.features_extractor(observation)
            layer_outputs = self.features_extractor.layer_outputs
            
            # 通過動作網路
            action_logits = self.actor(features)
            # 保存 actor 輸出
            layer_outputs['actor_output'] = action_logits.detach().cpu().numpy()
            
            action = torch.argmax(action_logits, dim=1)
            return action.item(), layer_outputs

def load_model_from_zip(model_path, device):
    """
    從zip檔案中載入模型權重
    """
    try:
        print(f"正在從 {model_path} 載入模型...")
        
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            # 檢查zip檔案中的內容
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
            
            # 創建策略實例
            policy = TestPolicy()
            policy.to(device)
            policy.load_state_dict(state_dict)
            policy.eval()
            
            print("模型載入成功")
            return policy
            
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        raise

def test_episode(env, policy, test_data_handler, episode_num, step_counter, device, render=False):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # 獲取動作和層輸出
        action, layer_outputs = policy.predict(obs, device)
        obs, reward, done, info = env.step(action)
        
        # 記錄數據
        test_data_handler.save_step_data(
            step=step_counter,
            obs=obs,
            angle_degrees=info.get('angle_degrees', 0),
            reward=reward,
            reward_list=[0] * 12,  # 或從 info 中獲取實際的 reward_list
            origin_image=info.get('origin_image', obs),
            results=info.get('results', None),
            layer_outputs=layer_outputs
        )
        
        total_reward += reward
        steps += 1
        step_counter += 1
        
        if render:
            env.render()
            
    return total_reward, steps, step_counter

def main():
    parser = argparse.ArgumentParser(description='Crawler 模型測試程式')
    parser.add_argument('--episodes', type=int, default=5,
                      help='每個模型要運行的測試集數 (預設: 5)')
    parser.add_argument('--model', type=str, default=None,
                      help='指定的模型路徑 (預設: 測試所有模型)')
    parser.add_argument('--render', type=bool, default=False,
                      help='是否顯示視覺化界面')
    parser.add_argument('--device', type=str, default='cuda',
                      help='使用的設備 (預設: cuda)')
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用設備: {device}")
    
    # 確定要測試的模型
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
        for model_path in model_paths:
            print(f"\n測試模型: {model_path}")
            model_name = os.path.basename(model_path).split('.')[0]
            
            try:
                # 載入模型
                policy = load_model_from_zip(model_path, device)
                
                # 初始化該模型的測試資料記錄
                test_data_handler = DataHandler(
                    base_dir=os.path.join("test_results", model_name)
                )
                test_data_handler.create_epoch_file(epoch=0)
                
                # 運行測試episodes
                episode_rewards = []
                episode_steps = []
                step_counter = 1  # 從1開始計數
                
                for episode in range(args.episodes):
                    reward, steps, step_counter = test_episode(
                        env, policy, test_data_handler, 
                        episode, step_counter, device, args.render
                    )
                    episode_rewards.append(reward)
                    episode_steps.append(steps)
                    print(f"Episode {episode + 1}: Reward = {reward:.2f}, Steps = {steps}")
                
                # 計算並輸出統計數據
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_steps = np.mean(episode_steps)
                std_steps = np.std(episode_steps)
                
                print(f"\n模型 {model_path} 的測試結果:")
                print(f"平均獎勵: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"平均步數: {mean_steps:.2f} ± {std_steps:.2f}")
                
                # 關閉測試資料記錄
                test_data_handler.close_epoch_file()
                
            except Exception as e:
                print(f"測試模型 {model_path} 時發生錯誤: {e}")
                continue
        
        print("\n所有模型測試完成！")
        
    finally:
        env.close()
        print("環境已關閉")

if __name__ == "__main__":
    main()