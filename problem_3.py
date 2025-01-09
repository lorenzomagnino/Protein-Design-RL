import argparse
from agent.learner import Agent
import os
from test.test_algo import Tester
import tensorboard
import logging
from utility.logging_config import configure_logging
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym 
import numpy as np
from parse_args import *


if __name__ == "__main__":
    configure_logging(level = logging.DEBUG)
    algo_classes = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
    }
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # main directory
    # defining the hyperparameters
    args = parse_args()

    # Problem 3 uses variable target motif and variable sequence length 
    args.variable_length = True
    args.variable_motif = True

    #making the environment
    env = gym.make(
        args.env_name,
        change_motif_at_each_episode = args.variable_motif,
        change_sequence_length_at_each_episode=args.variable_length,
        seed = args.seed,
        )
    # deifining the agent (policy model)
    agent = Agent(args)

    #TRAINING 
    if args.mode==1: 
        logging.info(f"-----------Start Training with {args.algo}-----------")
        agent.train()

    #TESTING
    if args.mode==2: 
        name = f"{args.algo}_Protein_Design_rng_motif_length.zip"
        model_path = os.path.join(args.dir, name)
        best_model_path = f"{args.algo}_Protein_Design_rng_motif_length"
        best_model_file_path=os.path.join(BASE_DIR, "saved-model", best_model_path, "best_model.zip")
        evaluation_file_path = os.path.join(BASE_DIR, "saved-model", best_model_path, "evaluations.npz")
        evaluations = np.load(evaluation_file_path)

        if args.take_best_model: model_path = best_model_file_path
        logging.debug(f"the path of the saved model is {model_path}")

        if args.take_best_model: model_path = best_model_file_path
        logging.debug(f"the path of the saved model is {model_path}")

        # Check if the chosen algorithm is supported
        if args.algo not in algo_classes:
            raise ValueError(f"Unsupported algorithm '{args.algo}'. Choose from {list(algo_classes.keys())}.")
        #loading the model
        agent = algo_classes[args.algo].load(model_path, env)

        # creating the setting for the test
        tester = Tester(agent, env, args)
        logging.info(f"-----------Start Testing with {args.algo}-----------")
        tester.test()


        logging.info(f"""
        For the algorithm: {args.algo} in the problem_3 the evaluation is:
        timesteps: {evaluations["timesteps"]}
        results: {evaluations["results"]}
        ep_lenghts: {evaluations["ep_lengths"]}""")
