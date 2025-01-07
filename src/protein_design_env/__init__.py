from gymnasium.envs.registration import register
from protein_design_env.environment import Environment

register(id="Protein-Design-v0", entry_point="protein_design_env.environment:Environment")
