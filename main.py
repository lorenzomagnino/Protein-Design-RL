import argparse
from agent.learner import Agent
import os
from test.tester import Tester
import tensorboard
import logging
from utility.logging_config import configure_logging
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym 
def parse_args():
    parser = argparse.ArgumentParser(description="RL for Protein Design")
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO", "DQN", "A2C"],
                        help="The RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total training timesteps")
    parser.add_argument("--variable_motif", action="store_true",
                        help="Enable variable motif (Problem 3)")
    parser.add_argument("--variable_length", action="store_true",
                        help="Enable variable sequence length (Problems 2 and 3)")
    parser.add_argument("--env_name", type=str, default="Protein-Design-v0", help="select the environment") 
    parser.add_argument("--mode", type=int, default=1, help="1:train 2:test")

    # save and load
    parsed_args, remaining_argv = parser.parse_known_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    tensorboard_dir = os.path.join(BASE_DIR,"tensorboard/")
    model_dir = os.path.join(BASE_DIR, "saved-results", "model_saved")
    parser.add_argument("--model_save_bool", action="store_false", default=True, help="if save model after training")
    parser.add_argument("--tb_save_dir", type=str, default=tensorboard_dir, help="directory for tensorboard")
    parser.add_argument("--dir", type=str, default=model_dir, help="directory for saved model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    configure_logging(level = logging.DEBUG)
    args = parse_args()
    env = gym.make(
        args.env_name,
        change_motif_at_each_episode = args.variable_motif,
        change_sequence_length_at_each_episode=args.variable_length,
        )

    agent = Agent(args)
    if args.mode==1: 
        logging.info(f"-----------Start Training with {args.algo}-----------")
        agent.train()
    if args.mode==2: 
        name = f"{args.algo}_Protein_Design.zip"
        model_path = os.path.join(args.dir, name)
        logging.debug(f"the path of the saved model is {model_path}")
        if args.algo=="PPO":
            agent = PPO.load(model_path, env)
        elif args.algo=="DQN":
            agent = DQN.load(model_path, env)
        elif args.algo=="A2C":
            agent = A2C.load(model_path, env)
        else: 
            raise FileNotFoundError("No  model file found.")
        tester = Tester(agent, env)
        logging.info(f"-----------Start Testing with {args.algo}-----------")
        tester.test()


