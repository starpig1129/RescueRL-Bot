import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

class CustomResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomResNet, self).__init__(observation_space, features_dim)
        
        # 確保輸入維度正確
        self._features_dim = features_dim
        input_channels = observation_space.shape[0]
        
        # 第一層卷積和池化
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 層
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, features_dim)
        
        # 初始化用於存儲中間層輸出的字典
        self.layer_outputs = {}
        
        # 註冊 hooks
        self.register_hooks()

    def register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                if name == 'input':
                    # 保存輸入，確保是正確的形狀
                    self.layer_outputs[name] = input[0].detach().cpu().numpy()
                else:
                    # 保存輸出，根據層的不同可能需要調整形狀
                    self.layer_outputs[name] = output.detach().cpu().numpy()
            return hook

        # 註冊各層的 hooks
        self.conv1.register_forward_hook(get_hook('input'))
        self.conv1.register_forward_hook(get_hook('conv1_output'))
        self.layer4.register_forward_hook(get_hook('final_residual_output'))
        self.fc.register_forward_hook(get_hook('feature_output'))

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 確保輸入形狀正確 [batch_size, channels, height, width]
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
            
        x = self.conv1(observations)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        
        return features

class CustomActor(nn.Module):
    def __init__(self, features_dim, action_dim):
        super(CustomActor, self).__init__()
        self.fc1 = nn.Linear(features_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        # 用於存儲最後的輸出
        self.last_output = None
        
        # 註冊 forward hook
        self.register_forward_hook(self._save_output)

    def _save_output(self, module, input, output):
        self.last_output = output.detach().cpu().numpy()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CustomCritic(nn.Module):
    def __init__(self, features_dim):
        super(CustomCritic, self).__init__()
        self.fc1 = nn.Linear(features_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # 確保觀察空間的形狀正確 (C, H, W)
        if len(observation_space.shape) != 3 or observation_space.shape[0] != 3:
            raise ValueError(f"預期觀察空間形狀為 (C, H, W)，實際為 {observation_space.shape}")
            
        super(CustomPolicy, self).__init__(
            observation_space, 
            action_space, 
            lr_schedule,
            *args, 
            **kwargs,
            features_extractor_class=CustomResNet,
            features_extractor_kwargs={'features_dim': 512}
        )

        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def get_layer_outputs(self):
        """獲取所有層的輸出"""
        layer_outputs = {}
        
        # 從特徵提取器獲取輸出
        if hasattr(self.features_extractor, 'layer_outputs'):
            layer_outputs.update(self.features_extractor.layer_outputs)
        
        # 從 actor 網絡獲取輸出
        if hasattr(self.action_net, 'last_output'):
            layer_outputs['actor_output'] = self.action_net.last_output
            
        return layer_outputs

    def forward(self, obs, deterministic=False):
        """
        前向傳播，確保輸入維度正確
        """
        # 確保輸入是正確的形狀
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            actions = action_dist.sample()
        
        log_probs = action_dist.log_prob(actions)
        
        return actions, value, log_probs

    def evaluate_actions(self, obs, actions):
        """
        評估動作，確保輸入維度正確
        """
        # 確保輸入是正確的形狀
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value