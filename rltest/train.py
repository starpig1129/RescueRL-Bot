import os
from stable_baselines3 import PPO
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy

# Create environment instance
env = CrawlerEnv(show=False)

# Ensure the 'logs' and 'models' directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Set up model hyperparameters
model_params = {
    "policy": CustomPolicy,  # Custom policy (ensure it's correctly implemented)
    "env": env,  # The environment to train on
    "verbose": 2,  # Output verbosity
    "learning_rate": 2.5e-4,  # Learning rate
    "n_steps": 2048,  # Number of steps per update
    "batch_size": 64,  # Batch size
    "n_epochs": 10,  # Number of training iterations per update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # Clip range for PPO
    "ent_coef": 0.01,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Maximum gradient norm
    "use_sde": False,  # Stochastic differential equations usage
    "sde_sample_freq": 4,  # SDE sample frequency
    "target_kl": 0.03,  # Target KL divergence
    "tensorboard_log": "./logs/",  # TensorBoard log directory
}

# Create PPO model using the specified hyperparameters
model = PPO(**model_params)

# Training configuration
total_timesteps = 1_000_000  # Total number of timesteps to train
checkpoint_interval = 50_000  # Save model every 50,000 steps
eval_freq = 10_000  # Evaluate the model every 10,000 steps (if necessary)

def main():
    # Train and save checkpoints
    for i in range(int(total_timesteps / checkpoint_interval)):
        # Train for the checkpoint interval
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False, tb_log_name="PPO")
        
        # Save the model at each checkpoint
        model.save(f"models/ppo_crawler_{(i+1)*checkpoint_interval}")
    
    # Training complete
    print("Training completed!")
    
if __name__ == "__main__":
    main()
