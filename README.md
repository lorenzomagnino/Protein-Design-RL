# Protein Design Environment

The **Protein Design Environment** is a reinforcement learning (RL) environment designed as a toy protein design
environment.

## Environment Description

In this environment, the agent acts by sequentially building a protein sequence.

### Key Features

- **Dynamic Motifs**:
    - A fixed target motif, such as `ARGININE, ISOLEUCINE`, can be used.
    - Alternatively, a random motif of length between 2 and 4 is generated for each episode.
- **Flexible Sequence Lengths**:
    - Sequence lengths can be fixed at 15.
    - The lengths can vary randomly between 15 and 25 for each episode.
- **Reward Structure**:
    - **Motif Rewards**: Bonus for an occurrence of the target motif in the sequence. Smaller bonus added for each amino acids of the target motif is in the sequence.
    - **Charge Penalty**: Penalty if the sequence is not neutral in the end of the episode.

### Environment Mechanics

- **Actions**: The agent chooses an amino acid to append to the sequence.
- **Observation**:
    - The current sequence (padded to a maximum length).
    - The length of the sequence.
    - The target motif (padded to a maximum length).
    - The sequence's charge and the target sequence length.
- **Termination**: An episode ends when the sequence reaches the pre-defined or random length for that episode.

## Goal of the Agent

The agent's goal is to optimize amino acid sequences that:

1. **Design a sequence containing the pattern**: Include an instance of the target motif within the sequence.
2. **Achieve Charge Neutrality**: Ensure the sequence has a net charge of zero to avoid penalties.

## Getting Started

- Clone this repository:

```
git clone https://github.com/lorenzomagnino/Protein-Design-RL.git
cd Protein-Design-RL
```

- Install uv with curl (or the latest version with pip): ```curl -LsSf https://astral.sh/uv/install.sh | sh```
- Install python 3.10.15: ```uv python install 3.10.15```
- Create virtual environment: ```uv venv --python=3.10.15```
- Activate virtual environment: ```source .venv/bin/activate```
- Install all requirements: ```uv sync```
- Install the protein-design-env package: ```pip install -e .```
- Install other important libraries: ```pip install colorlog, tensorboard, torch```
- Launch the rollout script: ```uv run python scripts/rollout.py```
- To train the agent for Problem 1 with PPO (similar for Problem 2 and 3): ```python problem_3.py --mode 1 --algo PPO --timesteps 100000```
- To visualize training metrics (port 6008 is arbitrary, you can use default 6006): ```tensorboard --logdir ./saved-model --port 6008```
- To test with the trained model: ```python problem_3.py --mode 2 --algo PPO --take_best_model True```