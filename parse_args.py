import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="RL for Protein Design")
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO", "DQN", "A2C"],
                        help="The RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps")
    parser.add_argument("--variable_motif", default=False,
                        help="Enable variable motif (Problem 3)")
    parser.add_argument("--variable_length", default=False,
                        help="Enable variable sequence length (Problems 2 and 3)")
    parser.add_argument("--env_name", type=str, default="Protein-Design-v0", help="select the environment") 
    parser.add_argument("--mode", type=int, default=1, help="1:train 2:test")
    parser.add_argument("--manual", type=bool, default=False, help="If True the model is createed with specified paramters (Use only in Problem 3!)")
    # save and load
    parsed_args, remaining_argv = parser.parse_known_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "saved-results", "model_saved")

    parser.add_argument("--model_save_bool", action="store_false", default=True, help="if save model after training")
    parser.add_argument("--dir", type=str, default=model_dir, help="directory for saved model")
    parser.add_argument("--seed", type=int, default = 0, help = "seed for the random number generator")
    parser.add_argument("--test_episodes", type=int, default=2, help = "how many episodes you want to run to test your policy")
    parser.add_argument("--take_best_model", type=bool, default= False, help = "decide to take the best model evaluated with EvalCallback or not")
    args = parser.parse_args()

    return args