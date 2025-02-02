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
        
        # 註冊前向傳播和梯度鉤子
        self.extractor.conv1.register_forward_hook(self.get_activation('conv1_output'))
        self.extractor.layer4.register_forward_hook(self.get_activation('final_residual_output'))
        
        # 註冊梯度鉤子（只為需要梯度的參數註冊）
        if self.extractor.conv1.weight.requires_grad:
            self.extractor.conv1.weight.register_hook(self._get_gradient_hook('conv1'))
        for i, layer in enumerate(self.extractor.layer4):
            if layer.conv1.weight.requires_grad:
                layer.conv1.weight.register_hook(self._get_gradient_hook(f'layer4_{i}_conv1'))
            if layer.conv2.weight.requires_grad:
                layer.conv2.weight.register_hook(self._get_gradient_hook(f'layer4_{i}_conv2'))
    
    def get_activation(self, name):
        """
        創建一個鉤子函數來捕獲並存儲指定層的輸出
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
        # 儲存輸入（僅用於可視化）
        with torch.no_grad():
            self.layer_outputs['input'] = observations.cpu().numpy()
        
        # 通過特徵提取器並保持梯度流
        features = self.extractor(observations)
        
        # 儲存最終特徵輸出（僅用於可視化）
        with torch.no_grad():
            self.layer_outputs['features_output'] = features.cpu().numpy()
        
        # 在訓練模式下註冊梯度鉤子
        if self.training:
            # 只為需要梯度的張量註冊鉤子
            if observations.requires_grad:
                observations.register_hook(self._get_gradient_hook('input'))
            if features.requires_grad:
                features.register_hook(self._get_gradient_hook('final_features'))
            
        return features
        
    def _get_gradient_hook(self, name):
        """
        創建一個梯度鉤子函數
        只記錄梯度而不修改它
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
    使用LSTM處理連續50幀的特徵序列，從300幀中每6幀採樣一次
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
        
        # 初始化網絡組件
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)
        
        # 設置不同的學習率
        self.optimizer = torch.optim.Adam([
            {'params': self.features_extractor.parameters(), 'lr': lr_schedule(1) * 1.0},  # 特徵提取器使用標準學習率
            {'params': self.action_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr_schedule(1))
        for param_group in self.optimizer.param_groups:
            print(f"參數組學習率: {param_group['lr']}")
        self.action_logits = None
        self.layer_outputs = None
        
        # 初始化時序特徵相關參數
        self.buffer_size = 60  # 存儲最近60幀的特徵
        self.sample_interval = 6  # 每6幀取1幀
        self.temporal_size = 10  # 使用10幀作為時序輸入
    
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
        使用純向量操作每6幀取1幀，總共取50幀來觀察前300幀的狀態
        在新世代開始時會重置特徵緩衝區
        保持梯度流以支持反向傳播
        """
        # 獲取當前特徵的batch size和特徵維度
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        # 如果需要重置緩存，則進行重置
        if self._need_buffer_reset(features, batch_size):
            self._reset_buffers(features, batch_size, feature_dim)
        
        # 更新特徵緩衝區並獲取時序特徵
        temporal_features = self._update_feature_buffer(features)
        
        return temporal_features

    def _reset_buffers(self, features, batch_size, feature_dim):
        """
        重置所有緩存
        清理舊的緩存並初始化新的緩存
        """
        # 清理舊的緩存
        for attr in ['feature_buffer_tensor', 'temporal_indices']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # 初始化新的緩存
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
        根據當前特徵的屬性設置緩存參數
        """
        with torch.no_grad():
            # 創建特徵緩衝區
            self.feature_buffer_tensor = torch.zeros(
                (batch_size, self.buffer_size, feature_dim),
                dtype=features.dtype,
                device=features.device,
                requires_grad=self.training
            )
            
            # 創建時序採樣索引
            self.temporal_indices = torch.arange(
                self.buffer_size - 1, -1, -self.sample_interval,
                device=features.device
            )[:self.temporal_size]
            
    def _need_buffer_reset(self, features, batch_size):
        """
        檢查是否需要重置特徵緩衝區
        在以下情況下需要重置：
        1. 新世代開始
        2. 首次調用（緩衝區不存在）
        3. batch size改變
        4. 設備改變
        """
        return (
            self._is_new_epoch() or  # 新世代開始
            not hasattr(self, 'feature_buffer_tensor') or  # 首次調用
            self.feature_buffer_tensor.shape[0] != batch_size or  # batch size改變
            not hasattr(self, 'temporal_indices') or  # 首次調用
            (hasattr(self, 'temporal_indices') and  # 設備改變
             self.temporal_indices.device != features.device)
        )
        
    def _is_new_epoch(self):
        """
        檢查是否為新的世代開始
        通過環境的 step_count 來判斷
        """
        env = self._get_env()
        if env is not None:
            # 如果步數為0，表示是新世代的開始
            return env.step_count == 0
        return False

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
        
        # 收集所有層輸出用於可視化（不影響梯度流）
        with torch.no_grad():
            layer_outputs = {
                'input': self.features_extractor.layer_outputs['input'],
                'conv1_output': self.features_extractor.layer_outputs['conv1_output'],
                'final_residual_output': self.features_extractor.layer_outputs['final_residual_output'],
                'features_output': self.features_extractor.layer_outputs['features_output'],
                'actor_output': self.action_logits.cpu().numpy()
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
