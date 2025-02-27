import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models

class PretrainedResNet(BaseFeaturesExtractor):
    """
    預訓練的 ResNet 特徵提取器
    使用 ResNet18 作為基礎模型，移除最後的全連接層，用於提取圖像特徵
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        try:
            super(PretrainedResNet, self).__init__(observation_space, features_dim)
            
            # 初始化預訓練的 ResNet 模型
            resnet = models.resnet18(pretrained=True)
            num_features = resnet.fc.in_features
            # 移除原始的全連接層，替換為恆等映射
            resnet.fc = nn.Identity()
            
            # 確保所有參數可訓練
            for param in resnet.parameters():
                param.requires_grad = True
                
            self.extractor = resnet
            self._features_dim = num_features
            
            # 打印參數狀態
            print("\n特徵提取器參數狀態:")
            for name, param in self.extractor.named_parameters():
                print(f"{name}: requires_grad = {param.requires_grad}")
            
            # 初始化用於存儲層輸出和梯度信息的字典
            self.layer_outputs = {}
            
            # 註冊前向傳播的 hook 捕捉中間層輸出
            self.extractor.conv1.register_forward_hook(self.get_activation('conv1_output'))
            self.extractor.layer4.register_forward_hook(self.get_activation('final_residual_output'))
            
            # 註冊梯度 hook（只為需要梯度的參數註冊）
            if self.extractor.conv1.weight.requires_grad:
                self.extractor.conv1.weight.register_hook(self._get_gradient_hook('conv1'))
            for i, layer in enumerate(self.extractor.layer4):
                if layer.conv1.weight.requires_grad:
                    layer.conv1.weight.register_hook(self._get_gradient_hook(f'layer4_{i}_conv1'))
                if layer.conv2.weight.requires_grad:
                    layer.conv2.weight.register_hook(self._get_gradient_hook(f'layer4_{i}_conv2'))
                    
        except Exception as e:
            print(f"初始化特徵提取器時發生錯誤: {e}")
            raise
    
    def get_activation(self, name):
        """創建一個 hook 函數來捕獲並存儲指定層的輸出"""
        def hook(model, input, output):
            with torch.no_grad():
                try:
                    self.layer_outputs[name] = output.cpu().numpy()
                except Exception as e:
                    print(f"存儲層輸出時發生錯誤: {e}")
        return hook
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向傳播函數"""
        try:
            # 強制切換到訓練模式，確保 BatchNorm 等層能更新
            self.extractor.train()
            
            # 儲存輸入（僅用於可視化）
            with torch.no_grad():
                self.layer_outputs['input'] = observations.cpu().numpy()
            
            # 通過特徵提取器並保持梯度流
            features = self.extractor(observations)
            
            # 儲存最終特徵輸出（僅用於可視化）
            with torch.no_grad():
                self.layer_outputs['features_output'] = features.cpu().numpy()
            
            # 若為訓練模式，為需要梯度的張量註冊 hook
            if self.training:
                if observations.requires_grad:
                    observations.register_hook(self._get_gradient_hook('input'))
                if features.requires_grad:
                    features.register_hook(self._get_gradient_hook('final_features'))
                
            return features
            
        except Exception as e:
            print(f"前向傳播時發生錯誤: {e}")
            # 如果出錯，盡可能返回一個有效的張量
            if torch.is_tensor(observations):
                return torch.zeros(observations.shape[0], self._features_dim,
                                device=observations.device)
            raise
        
    def _get_gradient_hook(self, name):
        """創建一個梯度 hook 函數"""
        def hook(grad):
            try:
                self._log_gradient(grad, name)
            except Exception as e:
                print(f"記錄梯度時發生錯誤: {e}")
            return grad
        return hook
        
    def _log_gradient(self, grad, name):
        """記錄梯度信息"""
        try:
            if grad is not None and self.training:
                # 計算梯度統計
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_max = grad.max().item()
                grad_min = grad.min().item()
                
                print(f"\n層 {name} 的梯度統計:")
                print(f"  範數: {grad_norm:.6f}")
                print(f"  平均值: {grad_mean:.6f}")
                print(f"  標準差: {grad_std:.6f}")
                print(f"  最大值: {grad_max:.6f}")
                print(f"  最小值: {grad_min:.6f}")
                
                if grad_norm < 1e-8:
                    print(f"警告: {name} 層的梯度可能消失")
        except Exception as e:
            print(f"計算梯度統計時發生錯誤: {e}")

    def cleanup(self):
        """清理資源"""
        try:
            # 清理 hook
            for handle in self._forward_hooks.values():
                handle.remove()
            for handle in self._backward_hooks.values():
                handle.remove()
            
            # 清理輸出緩存
            if hasattr(self, 'layer_outputs'):
                self.layer_outputs.clear()
            
            # 清理 CUDA 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"清理特徵提取器資源時發生錯誤: {e}")

class TemporalModule(nn.Module):
    """時序處理模組"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=1):
        try:
            super(TemporalModule, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        except Exception as e:
            print(f"初始化時序處理模組時發生錯誤: {e}")
            raise
        
    def forward(self, x):
        try:
            output, (hidden, _) = self.lstm(x)
            return output[:, -1, :]
        except Exception as e:
            print(f"時序處理模組前向傳播時發生錯誤: {e}")
            # 返回零張量作為應急處理
            return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

    def cleanup(self):
        """清理LSTM資源"""
        try:
            # 清理隱藏狀態
            if hasattr(self, 'lstm'):
                for param in self.lstm.parameters():
                    if hasattr(param, 'data'):
                        param.data = None
        except Exception as e:
            print(f"清理LSTM資源時發生錯誤: {e}")

class CustomActor(nn.Module):
    """自定義演員網絡"""
    def __init__(self, features_dim, action_dim):
        try:
            super(CustomActor, self).__init__()
            self.temporal = TemporalModule(features_dim)
            self.fc1 = nn.Linear(self.temporal.hidden_dim, 128)
            self.fc2 = nn.Linear(128, action_dim)
            self.dropout = nn.Dropout(0.3)
        except Exception as e:
            print(f"初始化演員網絡時發生錯誤: {e}")
            raise

    def forward(self, x):
        try:
            x = self.temporal(x)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
        except Exception as e:
            print(f"演員網絡前向傳播時發生錯誤: {e}")
            # 返回零張量作為應急處理
            return torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)

    def cleanup(self):
        """清理演員網絡資源"""
        try:
            self.temporal.cleanup()
            # 清理所有參數
            for param in self.parameters():
                if hasattr(param, 'data'):
                    param.data = None
        except Exception as e:
            print(f"清理演員網絡資源時發生錯誤: {e}")

class CustomCritic(nn.Module):
    """自定義評論家網絡"""
    def __init__(self, features_dim):
        try:
            super(CustomCritic, self).__init__()
            self.temporal = TemporalModule(features_dim)
            self.fc1 = nn.Linear(self.temporal.hidden_dim, 128)
            self.fc2 = nn.Linear(128, 1)
            self.dropout = nn.Dropout(0.3)
        except Exception as e:
            print(f"初始化評論家網絡時發生錯誤: {e}")
            raise

    def forward(self, x):
        try:
            x = self.temporal(x)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
        except Exception as e:
            print(f"評論家網絡前向傳播時發生錯誤: {e}")
            # 返回零張量作為應急處理
            return torch.zeros(x.shape[0], 1, device=x.device)

    def cleanup(self):
        """清理評論家網絡資源"""
        try:
            self.temporal.cleanup()
            # 清理所有參數
            for param in self.parameters():
                if hasattr(param, 'data'):
                    param.data = None
        except Exception as e:
            print(f"清理評論家網絡資源時發生錯誤: {e}")

class CustomPolicy(ActorCriticPolicy):
    """自定義策略類"""
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        try:
            super(CustomPolicy, self).__init__(
                observation_space, 
                action_space, 
                lr_schedule,
                *args, 
                **kwargs, 
                features_extractor_class=PretrainedResNet,
                features_extractor_kwargs={'features_dim': 512}
            )
            
            # 強制將特徵擷取器設為訓練模式
            self.features_extractor.train()
            
            # 初始化網絡組件
            self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
            self.value_net = CustomCritic(self.features_extractor.features_dim)
            
            # 設置優化器
            try:
                self.optimizer = torch.optim.Adam([
                    {'params': self.features_extractor.parameters(), 'lr': lr_schedule(1) * 1.0},
                    {'params': self.action_net.parameters()},
                    {'params': self.value_net.parameters()}
                ], lr=lr_schedule(1))
                
                for param_group in self.optimizer.param_groups:
                    print(f"參數組學習率: {param_group['lr']}")
            except Exception as e:
                print(f"初始化優化器時發生錯誤: {e}")
                raise
            
            self.action_logits = None
            self.layer_outputs = None
            
            # 初始化時序特徵相關參數
            self.buffer_size = 60
            self.sample_interval = 1
            self.temporal_size = 60
            
            # 初始化緩衝區
            self.feature_buffer_tensor = None
            self.temporal_indices = None
            
        except Exception as e:
            print(f"初始化策略類時發生錯誤: {e}")
            self.cleanup()
            raise
    def _get_env(self):
        """
        獲取環境引用的改進方法，支援多種環境配置方式
        """
        if hasattr(self, 'env'):
            return self.env
        if hasattr(self, 'policy_parent'):
            if hasattr(self.policy_parent, 'env'):
                return self.policy_parent.env
            if hasattr(self.policy_parent, 'venv'):
                return self.policy_parent.venv.envs[0]
        return None

    def _build(self, lr_schedule) -> None:
        """
        重建網絡組件
        """
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)

    def predict_values(self, obs):
        """
        預測給定觀察的價值，包含時序特徵處理
        """
        features = self.extract_features(obs)
        temporal_features = self._get_temporal_features(features)
        return self.value_net(temporal_features)
    def cleanup(self):
        """清理所有資源"""
        try:
            # 清理特徵提取器
            if hasattr(self, 'features_extractor'):
                self.features_extractor.cleanup()
            
            # 清理演員網絡
            if hasattr(self, 'action_net'):
                self.action_net.cleanup()
            
            # 清理評論家網絡
            if hasattr(self, 'value_net'):
                self.value_net.cleanup()
            
            # 清理優化器
            if hasattr(self, 'optimizer'):
                self.optimizer = None
            
            # 清理緩衝區
            if hasattr(self, 'feature_buffer_tensor'):
                del self.feature_buffer_tensor
            if hasattr(self, 'temporal_indices'):
                del self.temporal_indices
            
            # 清理輸出
            if hasattr(self, 'action_logits'):
                del self.action_logits
            if hasattr(self, 'layer_outputs'):
                del self.layer_outputs
            
            # 清理 CUDA 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"清理策略類資源時發生錯誤: {e}")

    def _initialize_buffers(self, features, batch_size, feature_dim):
        """初始化特徵緩衝區和時序索引，包含錯誤處理和資源管理"""
        try:
            # 釋放舊的緩衝區（如果存在）
            if hasattr(self, 'feature_buffer_tensor'):
                del self.feature_buffer_tensor
            if hasattr(self, 'temporal_indices'):
                del self.temporal_indices
            
            # 清理 CUDA 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                try:
                    self.feature_buffer_tensor = torch.zeros(
                        (batch_size, self.buffer_size, feature_dim),
                        dtype=features.dtype,
                        device=features.device,
                        requires_grad=self.training
                    )
                except RuntimeError as e:
                    print(f"建立特徵緩衝區時發生錯誤: {e}")
                    # 如果 CUDA 內存不足，嘗試在 CPU 上創建
                    self.feature_buffer_tensor = torch.zeros(
                        (batch_size, self.buffer_size, feature_dim),
                        dtype=features.dtype,
                        device='cpu',
                        requires_grad=self.training
                    )
                
                try:
                    self.temporal_indices = torch.arange(
                        self.buffer_size - 1, -1, -self.sample_interval,
                        device=features.device
                    )[:self.temporal_size]
                except RuntimeError as e:
                    print(f"建立時序索引時發生錯誤: {e}")
                    # 如果 CUDA 內存不足，嘗試在 CPU 上創建
                    self.temporal_indices = torch.arange(
                        self.buffer_size - 1, -1, -self.sample_interval,
                        device='cpu'
                    )[:self.temporal_size]
                
        except Exception as e:
            print(f"初始化緩衝區時發生錯誤: {e}")
            self.cleanup()
            raise

    def _get_temporal_features(self, features):
        """處理特徵緩衝區並返回時序特徵"""
        try:
            batch_size = features.shape[0]
            feature_dim = features.shape[1]
            
            # 如果需要重置緩衝區
            if (self.feature_buffer_tensor is None or 
                self.feature_buffer_tensor.shape[0] != batch_size or
                self.feature_buffer_tensor.device != features.device):
                with torch.no_grad():
                    self.feature_buffer_tensor = torch.zeros(
                        (batch_size, self.buffer_size, feature_dim),
                        dtype=features.dtype,
                        device=features.device,
                        requires_grad=self.training
                    )
            
            # 更新特徵緩衝區
            with torch.set_grad_enabled(self.training):
                # 移動舊特徵並添加新特徵
                self.feature_buffer_tensor = torch.cat([
                    self.feature_buffer_tensor[:, 1:],
                    features.unsqueeze(1)
                ], dim=1)
                
            return self.feature_buffer_tensor
            
        except Exception as e:
            print(f"處理時序特徵時發生錯誤: {e}")
            # 返回安全的默認值
            return features.unsqueeze(1).repeat(1, self.buffer_size, 1)
    
    def forward(self, obs, deterministic=False):
        """前向傳播函數"""
        try:
            # 確保特徵擷取器為訓練模式
            self.features_extractor.train()
            
            # 提取特徵
            features = self.extract_features(obs)
            self.layer_outputs = self.features_extractor.layer_outputs
            
            # 取得時序特徵
            temporal_features = self._get_temporal_features(features)
            
            # 演員與評論家網路計算
            self.action_logits = self.action_net(temporal_features)
            value = self.value_net(temporal_features)
            
            # 計算動作分布
            try:
                action_dist = torch.distributions.Categorical(logits=self.action_logits)
                if deterministic:
                    actions = torch.argmax(self.action_logits, dim=1)
                else:
                    actions = action_dist.sample()
                    
                log_probs = action_dist.log_prob(actions)
            except Exception as e:
                print(f"計算動作分布時發生錯誤: {e}")
                device = self.action_logits.device if hasattr(self, 'action_logits') else obs.device
                actions = torch.zeros(obs.shape[0], dtype=torch.long, device=device)
                log_probs = torch.zeros_like(actions, dtype=torch.float)
            
            # 收集各層輸出
            with torch.no_grad():
                try:
                    layer_outputs = {
                        'input': self.features_extractor.layer_outputs.get('input', None),
                        'conv1_output': self.features_extractor.layer_outputs.get('conv1_output', None),
                        'final_residual_output': self.features_extractor.layer_outputs.get('final_residual_output', None),
                        'features_output': self.features_extractor.layer_outputs.get('features_output', None),
                        'actor_output': self.action_logits.cpu().numpy()
                    }
                    
                    env = self._get_env()
                    if env is not None and hasattr(env, 'set_layer_outputs'):
                        env.set_layer_outputs(layer_outputs)
                except Exception as e:
                    print(f"處理層輸出時發生錯誤: {e}")
            
            return actions, value, log_probs
            
        except Exception as e:
            print(f"前向傳播時發生錯誤: {e}")
            # 返回默認值
            batch_size = obs.shape[0] if torch.is_tensor(obs) else 1
            device = self.action_logits.device if hasattr(self, 'action_logits') else 'cpu'
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            value = torch.zeros(batch_size, 1, device=device)
            log_probs = torch.zeros(batch_size, device=device)
            return actions, value, log_probs
        
    def evaluate_actions(self, obs, actions):
        """
        評估給定觀察與行動的對數機率、熵及狀態價值
        """
        features = self.extract_features(obs)
        temporal_features = self._get_temporal_features(features)
        
        action_logits = self.action_net(temporal_features)
        value = self.value_net(temporal_features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value
    
    def __del__(self):
        """解構函數：確保資源被正確釋放"""
        try:
            self.cleanup()
        except Exception as e:
            print(f"解構時發生錯誤: {e}")

if __name__ == "__main__":
    try:
        # 假設有一個 gym 環境，其 observation 為圖像
        env = gym.make("CartPole-v1")
        observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)
        
        # 定義學習率排程函數
        lr_schedule = lambda _: 1e-4
        
        # 建立 PPO 模型
        model = PPO(
            CustomPolicy,
            env,
            verbose=1,
            learning_rate=lr_schedule,
            policy_kwargs={"share_features_extractor": False}
        )
        
        # 開始訓練
        model.learn(total_timesteps=1000)
        
    except Exception as e:
        print(f"執行示例時發生錯誤: {e}")
    finally:
        if 'env' in locals():
            env.close()
