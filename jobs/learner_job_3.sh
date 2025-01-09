#!/bin/bash
#SBATCH --time=0-01:00:00           # 1 day wall-time
#SBATCH --mem=16G                   # Total memory
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --partition=sfscai          # Partition name
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --output=protein_design_%j.out    # Standard output log
#SBATCH --error=protein_design%j.err      # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lm5489@nyu.edu

# Load modules
module purge 
module load anaconda3/2022.10

cd Protein-Design-RL

source .venv/bin/activate

python problem_2.py --mode 1 --algo PPO --timesteps 70000

