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
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(PretrainedResNet, self).__init__(observation_space, features_dim)
        
        # Initialize the pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        
        # Get the number of features from the last linear layer (fc)
        num_features = resnet.fc.in_features
        
        # Replace the fully connected layer with an identity mapping
        resnet.fc = nn.Identity()
        
        self.extractor = resnet
        self._features_dim = num_features  # Set the features dimension correctly

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class CustomActor(nn.Module):
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


# Custom policy class overriding the ActorCriticPolicy
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                           *args, **kwargs, 
                                           features_extractor_class=PretrainedResNet,
                                           features_extractor_kwargs={'features_dim': 512})

        # Custom Actor-Critic network
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
    def _build(self, lr_schedule) -> None:
        # Construct action and value networks using custom architectures
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)

    def predict_values(self, obs):
        # 提取特徵
        features = self.extract_features(obs)
        # 使用自訂的 value_net 來預測價值函數
        return self.value_net(features)

    def forward(self, obs, deterministic=False):
        # Extract features using the custom ResNet extractor
        features = self.extract_features(obs)
        
        # Get the action logits from the actor and the value from the critic
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        # Create a distribution for the action (Categorical for discrete actions)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            actions = action_dist.sample()
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(actions)
        
        return actions, value, log_probs

    def evaluate_actions(self, obs, actions):
        # Extract features and process through the actor-critic networks
        features = self.extract_features(obs)
        
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        # Create a distribution for the action (Categorical for discrete actions)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value