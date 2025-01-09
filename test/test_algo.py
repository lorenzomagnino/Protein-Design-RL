import logging
from src.protein_design_env.constants import *
from src.protein_design_env.amino_acids import AminoAcids

class Tester():
    """
    Initialize the tester with the trained agent.
    Args:
        model (Agent): The trained agent instance.
        env (Environment) : The protein design environment
        args : command line arguments
    """
    def __init__(self, model, env, args):
        self.model = model
        self.env = env
        self.args = args

    def convert_to_amino_acids(self, sequence):
        """Convert a sequence of numbers to amino acid names."""
        return [AminoAcids(int(num)).name if num in AminoAcids._value2member_map_ else "UNKNOWN" for num in sequence]

    def test(self):
        """
        Test the loaded model.

        Returns:
            total_reward (float): Total reward accumulated during testing.
        """
        print("Testing the loaded model...")
        for it in range(self.args.test_episodes):

            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Use the trained model to predict actions
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward

            # Extract the padded sequence
            current_sequence = obs[:MAX_SEQUENCE_LENGTH]
            current_sequence = current_sequence[current_sequence!=0]

            current_sequence = self.convert_to_amino_acids(current_sequence)
            # Extract the actual sequence length
            sequence_length = int(obs[MAX_SEQUENCE_LENGTH])

            # Extract the padded target motif
            target_motif = obs[MAX_SEQUENCE_LENGTH + 1:MAX_SEQUENCE_LENGTH + 1 + MAX_MOTIF_LENGTH]
            target_motif = target_motif[target_motif!=0]
            target_motif = self.convert_to_amino_acids(target_motif)
            # Extract the target sequence length
            target_sequence_length = int(obs[MAX_SEQUENCE_LENGTH + 1 + MAX_MOTIF_LENGTH])

            # Extract the sequence charge
            sequence_charge = int(obs[MAX_SEQUENCE_LENGTH + 1 + MAX_MOTIF_LENGTH + 1]) 

            logging.info(f"""
            ---------Episode: {it}
            Current Sequence (unpadded): {current_sequence}
            Target Motif: {target_motif}
            Sequence Length: {sequence_length}
            Target Sequence Length: {sequence_length}
            Sequence Charge: {sequence_charge}
            Total Reward: {total_reward}""")
        return current_sequence, target_motif, sequence_length, target_sequence_length, sequence_charge, total_reward