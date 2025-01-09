import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.extractor = resnet
        self._features_dim = num_features
        
        # 初始化用於存儲層輸出的字典
        self.layer_outputs = {}
        
        # 註冊前向傳播鉤子，用於捕獲中間層的輸出
        self.extractor.conv1.register_forward_hook(self.get_activation('conv1_output'))
        self.extractor.layer4.register_forward_hook(self.get_activation('final_residual_output'))
    
    def get_activation(self, name):
        """
        創建一個鉤子函數來捕獲並存儲指定層的輸出
        """
        def hook(model, input, output):
            self.layer_outputs[name] = output.detach().cpu().numpy()
        return hook
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數
        儲存輸入和所有中間層的輸出用於可視化
        """
        # 儲存輸入
        self.layer_outputs['input'] = observations.detach().cpu().numpy()
        
        # 通過特徵提取器
        features = self.extractor(observations)
        
        # 儲存最終特徵輸出
        self.layer_outputs['features_output'] = features.detach().cpu().numpy()
        
        return features

class TemporalModule(nn.Module):
    """
    時序處理模組
    使用LSTM處理連續10幀的特徵序列
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

        # 初始化網絡組件
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        self.action_logits = None
        self.layer_outputs = None
        
        # 初始化特徵緩衝區，用於存儲最近60幀的特徵
        self.feature_buffer = []
        self.buffer_size = 60  # 存儲60幀
        self.sample_interval = 6  # 每6幀取1幀
        self.temporal_size = 10  # 時序輸入使用10幀
    
    def _get_env(self):
        """
        獲取環境引用的改進方法
        支援多種環境配置方式
        """
        # 方法1：通過 self 查找
        if hasattr(self, 'env'):
            return self.env
            
        # 方法2：通過 policy_parent (PPO實例) 查找
        if hasattr(self, 'policy_parent'):
            if hasattr(self.policy_parent, 'env'):
                return self.policy_parent.env
            # 對於向量化環境的情況
            if hasattr(self.policy_parent, 'venv'):
                return self.policy_parent.venv.envs[0]

    def _build(self, lr_schedule) -> None:
        """
        建構網絡組件
        """
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)

    def predict_values(self, obs):
        """
        預測給定觀察的價值
        包含時序特徵處理
        """
        features = self.extract_features(obs)
        temporal_features = self._get_temporal_features(features)
        return self.value_net(temporal_features)

    def _get_temporal_features(self, features):
        """
        處理特徵緩衝區並返回時序特徵
        使用純向量操作每6幀取1幀，總共取10幀來觀察前60幀的狀態
        """
        # 獲取當前特徵的batch size和特徵維度
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        # 創建零填充的特徵張量
        if not hasattr(self, 'feature_buffer_tensor') or self.feature_buffer_tensor.shape[0] != batch_size:
            self.feature_buffer_tensor = torch.zeros((batch_size, self.buffer_size, feature_dim),
                                                   dtype=features.dtype,
                                                   device=features.device)
        
        # 向左移動特徵緩衝區並添加新特徵
        self.feature_buffer_tensor = torch.cat([
            self.feature_buffer_tensor[:, 1:],
            features.unsqueeze(1)
        ], dim=1)
        
        # 使用步長為6的切片操作選取10幀
        indices = torch.arange(self.buffer_size - 1, -1, -self.sample_interval, device=features.device)[:self.temporal_size]
        temporal_features = self.feature_buffer_tensor[:, indices]
        
        return temporal_features

    def forward(self, obs, deterministic=False):
        """
        前向傳播函數
        處理觀察並生成行動、價值和對數概率
        """
        # 提取特徵
        features = self.extract_features(obs)
        self.layer_outputs = self.features_extractor.layer_outputs
        
        # 獲取時序特徵
        temporal_features = self._get_temporal_features(features)
        
        # 獲取行動邏輯值和狀態價值
        self.action_logits = self.action_net(temporal_features)
        value = self.value_net(temporal_features)
        
        # 創建行動分佈
        action_dist = torch.distributions.Categorical(logits=self.action_logits)
        
        # 根據是否確定性選擇行動
        if deterministic:
            actions = torch.argmax(self.action_logits, dim=1)
        else:
            actions = action_dist.sample()
        
        # 計算對數概率
        log_probs = action_dist.log_prob(actions)
        
        # 收集所有層輸出用於可視化
        layer_outputs = {
            'input': self.features_extractor.layer_outputs['input'],
            'conv1_output': self.features_extractor.layer_outputs['conv1_output'],
            'final_residual_output': self.features_extractor.layer_outputs['final_residual_output'],
            'features_output': self.features_extractor.layer_outputs['features_output'],
            'actor_output': self.action_logits.detach().cpu().numpy()
        }
        
        # 更新環境中的層輸出
        env = self._get_env()
        if env is not None and hasattr(env, 'set_layer_outputs'):
            env.set_layer_outputs(layer_outputs)
        return actions, value, log_probs

    def evaluate_actions(self, obs, actions):
        """
        評估給定觀察和行動的價值
        返回對數概率、熵和價值
        """
        features = self.extract_features(obs)
        temporal_features = self._get_temporal_features(features)
        
        action_logits = self.action_net(temporal_features)
        value = self.value_net(temporal_features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value
