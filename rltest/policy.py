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
        
        # Initialize pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        
        self.extractor = resnet
        self._features_dim = num_features
        
        # Initialize dictionary for storing layer outputs
        self.layer_outputs = {}
        
        # Register forward hooks
        self.extractor.conv1.register_forward_hook(self.get_activation('conv1_output'))
        self.extractor.layer4.register_forward_hook(self.get_activation('final_residual_output'))
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.layer_outputs[name] = output.detach().cpu().numpy()
        return hook
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Store input
        self.layer_outputs['input'] = observations.detach().cpu().numpy()
        
        # Pass through feature extractor
        features = self.extractor(observations)
        
        # Store final features output
        self.layer_outputs['features_output'] = features.detach().cpu().numpy()
        
        return features

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

class CustomPolicy(ActorCriticPolicy):
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

        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        self.action_logits = None
        self.layer_outputs = None
        
    def _get_env(self):
        """改进的环境引用获取方法"""
        # 方法1：通过 self 查找
        if hasattr(self, 'env'):
            return self.env
            
        # 方法2：通过 policy_parent (PPO实例) 查找
        if hasattr(self, 'policy_parent'):
            if hasattr(self.policy_parent, 'env'):
                return self.policy_parent.env
            # 对于vec_env的情况
            if hasattr(self.policy_parent, 'venv'):
                return self.policy_parent.venv.envs[0]

    def _build(self, lr_schedule) -> None:
        self.action_net = CustomActor(self.features_extractor.features_dim, self.action_space.n)
        self.value_net = CustomCritic(self.features_extractor.features_dim)

    def predict_values(self, obs):
        features = self.extract_features(obs)
        return self.value_net(features)

    def forward(self, obs, deterministic=False):
        # Extract features and store layer outputs
        features = self.extract_features(obs)
        self.layer_outputs = self.features_extractor.layer_outputs
        
        # Get the action logits and value
        self.action_logits = self.action_net(features)
        value = self.value_net(features)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=self.action_logits)
        
        if deterministic:
            actions = torch.argmax(self.action_logits, dim=1)
        else:
            actions = action_dist.sample()
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(actions)
        
        # Collect all layer outputs
        layer_outputs = {
            'input': self.features_extractor.layer_outputs['input'],
            'conv1_output': self.features_extractor.layer_outputs['conv1_output'],
            'final_residual_output': self.features_extractor.layer_outputs['final_residual_output'],
            'features_output': self.features_extractor.layer_outputs['features_output'],
            'actor_output': self.action_logits.detach().cpu().numpy()
        }
        
        # Get environment reference and set layer outputs if available
        env = self._get_env()
        if env is not None and hasattr(env, 'set_layer_outputs'):
            env.set_layer_outputs(layer_outputs)
            print('儲存成功')
        return actions, value, log_probs

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        return log_prob, entropy, value