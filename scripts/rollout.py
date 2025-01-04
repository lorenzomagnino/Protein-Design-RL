"""This script demonstrates a simple rollout of the "Protein-Design-v0" Gymnasium environment.

The script initializes the environment, resets it, and performs a loop of random actions
until the episode terminates. It then prints the terminal state and the final reward.
"""
import gymnasium as gym
from protein_design_env.amino_acids import AminoAcids
from protein_design_env.environment import Environment

change_motif_at_each_episode: bool = False
change_sequence_length_at_each_episode: bool = False

env = gym.make(
    "Protein-Design-v0",
    change_motif_at_each_episode=change_motif_at_each_episode,
    change_sequence_length_at_each_episode=change_sequence_length_at_each_episode,
)

observations, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observations, reward, done, _, _ = env.step(action)

base_env: Environment = env.env.env
print(f"Terminal state : {[AminoAcids(idx).name for idx in base_env.state]}")
print(f"Reward of : {reward}")
