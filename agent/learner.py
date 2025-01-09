import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
import sys
sys.path.append("/gpfsnyu/home/lm5489/Protein-Design-RL/src/protein_design_env")
import protein_design_env
import os
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
import torch.nn as nn




class Agent():
    def __init__(self, args):
        """ Initialize the agent with the command line arguments. """
        self.args = args
        self.env = gym.make(
            args.env_name,
            change_motif_at_each_episode=args.variable_motif,
            change_sequence_length_at_each_episode=args.variable_length,
        )
        self.initialize_model()

    def initialize_model(self):
        """ Initialize the model based on the algorithm specified in the command line arguments. """
        if self.args.algo == "PPO":
            if self.args.manual:
                """ Defined a speicifed model if required. We use that to experiment new settings """
                activation_fn = nn.ReLU
                lr_schedule = get_schedule_fn(3e-4)
                # Define PPO hyperparameters
                ppo_params = {
                    "learning_rate": lr_schedule,         # lr
                    "n_steps": 1024,              # Steps per update
                    "batch_size": 64,             # Batch size
                    "clip_range": 0.2,            # Clipping range
                    "ent_coef": 0.01,             # Entropy coefficient
                    "gamma": 0.99,                # Discount factor
                    "gae_lambda": 0.95,           # GAE lambda
                    "n_epochs": 10,               # Number of epochs
                    "policy_kwargs": {            # Policy network architecture
                        "net_arch": [128, 128],   # 2-layer MLP with 128 units each
                        "activation_fn": activation_fn,  # Activation function
                    }
                }
                # Create the PPO agent
                self.model = PPO("MlpPolicy", self.env, verbose=1,tensorboard_log=f"./saved-model/{self.args.algo}_Protein_Design_manual",  **ppo_params)

            else: self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=f"./saved-model/{self.args.algo}_Protein_Design")
        elif self.args.algo == "DQN":
            self.model = DQN("MlpPolicy", self.env, verbose=1, tensorboard_log=f"./saved-model/{self.args.algo}_Protein_Design")
        elif self.args.algo == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=1, tensorboard_log=f"./saved-model/{self.args.algo}_Protein_Design")
        else:
            raise ValueError(f"Unsupported algorithm: {self.args.algo}")

    def callback(self):
        """ Evaluation callback. """
        if self.args.variable_motif and self.args.variable_length:
            best_model_save_path = f"./saved-model/{self.args.algo}_Protein_Design_rng_motif_length"
        elif self.args.variable_length and not self.args.variable_motif:
            best_model_save_path = f"./saved-model/{self.args.algo}_Protein_Design_rng_length"
        else: 
            best_model_save_path = f"./saved-model/{self.args.algo}_Protein_Design"
       
        os.makedirs(best_model_save_path, exist_ok=True)
        env = Monitor(self.env, best_model_save_path)

        eval_env = gym.make(
            self.args.env_name,
            change_motif_at_each_episode = self.args.variable_motif,
            change_sequence_length_at_each_episode = self.args.variable_length,
        )
        eval_env = Monitor(eval_env, os.path.join(best_model_save_path, "eval"))
        eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                     log_path=best_model_save_path, eval_freq=5000,
                                     deterministic=True, render=False)
        return eval_callback
            
    def train(self):
        """
        Train a reinforcement learning agent to solve Problem 1, 2, 3"""

        # Train the model
        print(f"Training {self.args.algo} on {self.args.env_name} for {self.args.timesteps} timesteps...")
        self.model.learn(total_timesteps=self.args.timesteps, callback=self.callback(), progress_bar = True)


        if self.args.model_save_bool: 
            self.save_model()
        
    def save_model(self):
        """ Save the model """
        if self.args.variable_motif and self.args.variable_length:
            name = f"{self.args.algo}_Protein_Design_rng_motif_length"
        elif self.args.variable_length and not self.args.variable_motif:
            name = f"{self.args.algo}_Protein_Design_rng_length"
        else: 
            name = f"{self.args.algo}_Protein_Design"
        model_path = os.path.join(self.args.dir, name)
        self.model.save(model_path)
        print(f"Model saved at {model_path}")