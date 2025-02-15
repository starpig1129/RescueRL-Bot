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
    
    def get_activation(self, name):
        """
        創建一個 hook 函數來捕獲並存儲指定層的輸出
        """
        def hook(model, input, output):
            with torch.no_grad():
                self.layer_outputs[name] = output.cpu().numpy()
        return hook
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數
        儲存輸入和所有中間層的輸出用於可視化，同時保持梯度流
        """
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
        
    def _get_gradient_hook(self, name):
        """
        創建一個梯度 hook 函數，僅記錄梯度而不修改它
        """
        def hook(grad):
            self._log_gradient(grad, name)
            return grad  # 返回原始梯度，不做修改
        return hook
        
    def _log_gradient(self, grad, name):
        """記錄梯度信息"""
        if grad is not None and self.training:
            # 計算梯度統計
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            grad_max = grad.max().item()
            grad_min = grad.min().item()
            
            # 輸出詳細的梯度信息
            print(f"\n層 {name} 的梯度統計:")
            print(f"  範數: {grad_norm:.6f}")
            print(f"  平均值: {grad_mean:.6f}")
            print(f"  標準差: {grad_std:.6f}")
            print(f"  最大值: {grad_max:.6f}")
            print(f"  最小值: {grad_min:.6f}")
            
            # 檢查梯度是否過小或消失
            if grad_norm < 1e-8:
                print(f"警告: {name} 層的梯度可能消失")

class TemporalModule(nn.Module):
    """
    時序處理模組
    使用 LSTM 處理連續50幀的特徵序列，從300幀中每6幀採樣一次
    用於捕捉智能體行為的時序依賴關係
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=1):
        super(TemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        output, (hidden, _) = self.lstm(x)
        # 使用最後一個時間步的輸出
        return output[:, -1, :]  # shape: (batch_size, hidden_dim)

class CustomActor(nn.Module):
    """
    自定義演員網絡
    整合時序特徵處理
    """
    def __init__(self, features_dim, action_dim):
        super(CustomActor, self).__init__()
        self.temporal = TemporalModule(features_dim)
        self.fc1 = nn.Linear(self.temporal.hidden_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features_dim)
        x = self.temporal(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CustomCritic(nn.Module):
    """
    自定義評論家網絡
    整合時序特徵處理
    """
    def __init__(self, features_dim):
        super(CustomCritic, self).__init__()
        self.temporal = TemporalModule(features_dim)
        self.fc1 = nn.Linear(self.temporal.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features_dim)
        x = self.temporal(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CustomPolicy(ActorCriticPolicy):
    """
    自定義策略類
    整合特徵提取器、時序處理、演員網絡和評論家網絡
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
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
        
        # 設置不同的學習率
        self.optimizer = torch.optim.Adam([
            {'params': self.features_extractor.parameters(), 'lr': lr_schedule(1) * 1.0},
            {'params': self.action_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr_schedule(1))
        for param_group in self.optimizer.param_groups:
            print(f"參數組學習率: {param_group['lr']}")
        self.action_logits = None
        self.layer_outputs = None
        
        # 初始化時序特徵相關參數
        self.buffer_size = 300  # 存儲最近300幀的特徵
        self.sample_interval = 1  # 每1幀取1幀
        self.temporal_size = 300  # 使用300幀作為時序輸入
    
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

    def _get_temporal_features(self, features):
        """
        處理特徵緩衝區並返回時序特徵
        使用純向量操作每6幀取1幀，總共取50幀來觀察前300幀的狀態
        在新世代開始時會重置特徵緩衝區，保持梯度流以支持反向傳播
        """
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        if self._need_buffer_reset(features, batch_size):
            self._reset_buffers(features, batch_size, feature_dim)
        
        temporal_features = self._update_feature_buffer(features)
        
        return temporal_features

    def _reset_buffers(self, features, batch_size, feature_dim):
        """
        重置所有緩存，清理舊的並初始化新的
        """
        for attr in ['feature_buffer_tensor', 'temporal_indices']:
            if hasattr(self, attr):
                delattr(self, attr)
        self._initialize_buffers(features, batch_size, feature_dim)
        
    def _update_feature_buffer(self, features):
        """
        更新特徵緩衝區並返回時序特徵
        根據訓練模式決定是否保持梯度流
        """
        if self.training:
            self.feature_buffer_tensor = torch.cat([
                self.feature_buffer_tensor[:, 1:],
                features.unsqueeze(1)
            ], dim=1)
            temporal_features = self.feature_buffer_tensor[:, self.temporal_indices]
        else:
            with torch.no_grad():
                self.feature_buffer_tensor = torch.cat([
                    self.feature_buffer_tensor[:, 1:],
                    features.unsqueeze(1)
                ], dim=1)
                temporal_features = self.feature_buffer_tensor[:, self.temporal_indices]
        return temporal_features
        
    def _initialize_buffers(self, features, batch_size, feature_dim):
        """
        初始化特徵緩衝區和時序索引
        """
        with torch.no_grad():
            self.feature_buffer_tensor = torch.zeros(
                (batch_size, self.buffer_size, feature_dim),
                dtype=features.dtype,
                device=features.device,
                requires_grad=self.training
            )
            self.temporal_indices = torch.arange(
                self.buffer_size - 1, -1, -self.sample_interval,
                device=features.device
            )[:self.temporal_size]
            
    def _need_buffer_reset(self, features, batch_size):
        """
        檢查是否需要重置特徵緩衝區：
        1. 新世代開始
        2. 首次調用（緩衝區不存在）
        3. batch size 改變
        4. 設備改變
        """
        return (
            self._is_new_epoch() or
            not hasattr(self, 'feature_buffer_tensor') or
            self.feature_buffer_tensor.shape[0] != batch_size or
            not hasattr(self, 'temporal_indices') or
            (hasattr(self, 'temporal_indices') and self.temporal_indices.device != features.device)
        )
        
    def _is_new_epoch(self):
        """
        判斷是否為新世代開始（步數為0時）
        """
        env = self._get_env()
        if env is not None and hasattr(env, 'step_count'):
            return env.step_count == 0
        return False

    def forward(self, obs, deterministic=False):
        """
        前向傳播函數：處理觀察並生成行動、狀態價值及對數機率
        """
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
        
        action_dist = torch.distributions.Categorical(logits=self.action_logits)
        if deterministic:
            actions = torch.argmax(self.action_logits, dim=1)
        else:
            actions = action_dist.sample()
            
        log_probs = action_dist.log_prob(actions)
        
        # 收集各層輸出以便可視化（不影響梯度流）
        with torch.no_grad():
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

# 使用範例：
if __name__ == "__main__":
    # 假設有一個 gym 環境，其 observation 為圖像（例如 shape=(3, 224, 224)）
    env = gym.make("CartPole-v1")  # 此處請換成適合圖像觀察的環境
    observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)
    
    # 定義一個簡單的學習率排程函數
    lr_schedule = lambda _: 1e-4
    
    # 建立 PPO 模型，並使用自定義策略
    model = PPO(
        CustomPolicy,
        env,
        verbose=1,
        learning_rate=lr_schedule,
        policy_kwargs={"share_features_extractor": False}  # 如有需要可將共享參數關閉
    )
    
    # 開始訓練（此處僅為示意）
    model.learn(total_timesteps=1000)
