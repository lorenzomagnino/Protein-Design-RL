from typing import Any
import gymnasium as gym
import numpy as np
from numpy._typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
import logging

from protein_design_env.amino_acids import AMINO_ACIDS_TO_CHARGES_DICT, AminoAcids
from protein_design_env.constants import (
    AMINO_ACIDS_VALUES,
    DEFAULT_MOTIF,
    DEFAULT_SEQUENCE_LENGTH,
    MAX_MOTIF_LENGTH,
    MAX_SEQUENCE_LENGTH,
    MIN_MOTIF_LENGTH,
    MIN_SEQUENCE_LENGTH,
    NUM_AMINO_ACIDS,
    REWARD_PER_MOTIF,
)
from protein_design_env.constants import CHARGE_PENALTY


class Environment(gym.Env):
    """This class encapsulates all the logic of the protein design environment.

    The goal of the agent is to design amino acids sequences of neutral charge containing patterns.

    The initial state of the environment is: [].
    The actions are adding an amino acid to the sequence.
    The reward is CHARGE_PENALTY if the charge is not neutral and REWARD_PER_MOTIF per motif
    present in the sequence.
    The target motif to be present in the sequence is either DEFAULT_MOTIF if the flag
    "change_motif_at_each_episode" is False or a random motif of length between 2 and 4.
    The length of the episode is either DEFAULT_SEQUENCE_LENGTH if the flag
    "change_sequence_length_at_each_episode" is False or a random number between 15 and 25
    otherwise.
    """

    def __init__(
        self,
        change_motif_at_each_episode: bool = False,
        change_sequence_length_at_each_episode: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.change_motif_at_each_episode = change_motif_at_each_episode
        self.change_sequence_length_at_each_episode = change_sequence_length_at_each_episode
        self.rng = np.random.default_rng(seed)

        self.motif = DEFAULT_MOTIF
        self.sequence_length = DEFAULT_SEQUENCE_LENGTH

        self.state: list[int] = []
        #self.action_space = gym.spaces.Discrete(start=1, n=NUM_AMINO_ACIDS)
        self.action_space = gym.spaces.Discrete(NUM_AMINO_ACIDS)
        highest_value_possible_in_obs = max(MAX_SEQUENCE_LENGTH, MAX_MOTIF_LENGTH, NUM_AMINO_ACIDS)
        self.observation_space = gym.spaces.Box(
            low=-highest_value_possible_in_obs,
            high=highest_value_possible_in_obs,
            shape=(
                MAX_SEQUENCE_LENGTH  # padded state.
                + 1  # sequence length.
                + MAX_MOTIF_LENGTH  # padded motif.
                + 1  # target sequence length.
                + 1,  # charge
            ),
            dtype=float,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed, options=options)

        self.motif = self._generate_motif()
        self.sequence_length = self._generate_sequence_length()
        self.state.clear()
        obs = self._get_observation()
        #logging.debug(f"motif: {self.motif}; sequence_lenght: {self.sequence_length}")
        return obs, {}

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict[str, Any]]:
        """Adds an amino acid, compute the reward and the termination condition."""
        # Since algorithms use 0 starting then we map zero-based action to one-based action
        one_based_action = action + 1
        if one_based_action not in range(1, NUM_AMINO_ACIDS + 1):
            raise ValueError(f"Invalid action: {one_based_action}")
            
        self.state.append(AminoAcids(one_based_action).value)

        reward = self._get_reward()
        terminated = truncated = len(self.state) >= self.sequence_length
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self) -> NDArray:
        """Returns the observation of the current state."""
        charge = self._get_charge()
        flattened_obs = np.hstack(
            [self._pad_state(), len(self.state), self._pad_motif(), self.sequence_length, charge]
        ).astype(np.float64)
        return flattened_obs

    def _get_reward(self) -> int:
        """Compute the reward of a sequence."""
        motif_coeff = 0.0
        bonus_per_amino_acid_of_the_motif_in_state = 0.0
        if len(self.state) >= len(self.motif):
            possible_motifs = sliding_window_view(self.state, len(self.motif))
            for motif in possible_motifs:
                if all(motif == self.motif):
                    motif_coeff = 1.0

        if motif_coeff == 0.0:
            for amino_acid_id in self.motif:
                if amino_acid_id in self.state:
                    bonus_per_amino_acid_of_the_motif_in_state += 1 / (5 * len(self.motif))

        if self._get_charge() != 0 and len(self.state) >= self.sequence_length:
            charge_penalty = CHARGE_PENALTY if self._get_charge() != 0 else 0
        else:
            charge_penalty = 0

        reward: int = (
            charge_penalty
            + REWARD_PER_MOTIF * motif_coeff
            + bonus_per_amino_acid_of_the_motif_in_state
        )
        return reward

    def _get_charge(self) -> int:
        """Compute the charge of a sequence."""
        return sum(AMINO_ACIDS_TO_CHARGES_DICT[amino_acid] for amino_acid in self.state)

    def _generate_motif(self) -> list[int]:
        """Generate a random motif of amino acids and update the observation space.

        The motif length is between MIN_MOTIF_LENGTH and MAX_MOTIF_LENGTH.
        """
        if self.change_motif_at_each_episode:
            motif_length = self.rng.integers(
                low=MIN_MOTIF_LENGTH, high=MAX_MOTIF_LENGTH + 1, size=1
            ).item()
            self.motif: list[int] = self.rng.choice(  # type: ignore[no-redef]
                AMINO_ACIDS_VALUES, replace=True, size=motif_length
            ).tolist()
        return self.motif  # type: ignore[no-any-return]

    def _generate_sequence_length(self) -> int:
        """Generate a random sequence length and update the observation space."""
        if self.change_sequence_length_at_each_episode:
            self.sequence_length = self.rng.integers(
                low=MIN_SEQUENCE_LENGTH, high=MAX_SEQUENCE_LENGTH + 1, size=1
            ).item()
        return self.sequence_length  # type: ignore[no-any-return]

    def _pad_state(self) -> NDArray:
        """Return the padded state with zeros if no amino acids are present."""
        n_zeros_to_add = MAX_SEQUENCE_LENGTH - len(self.state)
        return np.hstack([self.state, np.zeros(n_zeros_to_add)]).astype(np.int64)

    def _pad_motif(self) -> NDArray:
        """Return the padded motif with zeros if no amino acids are present."""
        n_zeros_to_add = MAX_MOTIF_LENGTH - len(self.motif)
        return np.hstack([self.motif, np.zeros(n_zeros_to_add)]).astype(np.int64)
