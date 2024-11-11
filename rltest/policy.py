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

class CustomActor(nn.Module):
    """
    自定義演員網絡
    用於根據提取的特徵決定行動
    包含三個全連接層和 dropout 層以防止過擬合
    """
    def __init__(self, features_dim, action_dim):
        super(CustomActor, self).__init__()
        self.fc1 = nn.Linear(features_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))  
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x) 
        return x

class CustomCritic(nn.Module):
    """
    自定義評論家網絡
    用於評估當前狀態的價值
    架構與演員網絡類似，但輸出為單一值
    """
    def __init__(self, features_dim):
        super(CustomCritic, self).__init__()
        self.fc1 = nn.Linear(features_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  
        return x

class CustomPolicy(ActorCriticPolicy):
    """
    自定義策略類
    整合特徵提取器、演員網絡和評論家網絡
    實現完整的策略網絡功能
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
        """
        features = self.extract_features(obs)
        return self.value_net(features)

    def forward(self, obs, deterministic=False):
        """
        前向傳播函數
        處理觀察並生成行動、價值和對數概率
        """
        # 提取特徵並儲存層輸出
        features = self.extract_features(obs)
        self.layer_outputs = self.features_extractor.layer_outputs
        
        # 獲取行動邏輯值和狀態價值
        self.action_logits = self.action_net(features)
        value = self.value_net(features)
        
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
        
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value