import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
import sys
sys.path.append("/gpfsnyu/home/lm5489/Protein-Design-RL/src/protein_design_env")
import protein_design_env
import os


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
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        elif self.args.algo == "DQN":
            self.model = DQN("MlpPolicy", self.env, verbose=1)
        elif self.args.algo == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=1)
        else:
            raise ValueError(f"Unsupported algorithm: {self.args.algo}")

    def callback(self):
        """ Evaluation callback. """
        eval_env = gym.make(
            self.args.env_name,
            change_motif_at_each_episode=self.args.variable_motif,
            change_sequence_length_at_each_episode=self.args.variable_length,
        )
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                     log_path="./logs/", eval_freq=5000,
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
        name = f"{self.args.algo}_Protein_Design"
        model_path = os.path.join(self.args.dir, name)
        self.model.save(model_path)
        print(f"Model saved at {model_path}")