import pandas as pd
import matplotlib.pyplot as plt


""" 
--------------------------------
        Plotting from CSV
--------------------------------
"""


# Load the CSV files
# Each CSV file contains the training steps and corresponding mean episode lengths for different algorithms

csv_file_A2C = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_2/mean_ep_length_A2C_2.csv"
data_A2C = pd.read_csv(csv_file_A2C)

csv_file_DQN = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_2/mean_ep_length_DQN_2.csv"
data_DQN = pd.read_csv(csv_file_DQN)


csv_file_PPO = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_2/mean_ep_length_PPO_2.csv"
data_PPO = pd.read_csv(csv_file_PPO)

# Extract Steps and Values for each algorithm
# 'Step' column contains the training steps
# 'Value' column contains the mean episode length
steps_A2C = data_A2C['Step']
values_A2C = data_A2C['Value']

steps_DQN = data_DQN['Step']
values_DQN = data_DQN['Value']

steps_PPO = data_PPO['Step']
values_PPO = data_PPO['Value']

# Create the plot
plt.figure(figsize=(10, 6))


# Set background color to floral white
plt.gca().set_facecolor("floralwhite")
plt.gcf().set_facecolor("floralwhite")

# Plot each algorithm
plt.plot(steps_A2C, values_A2C, label="A2C", color="coral", linewidth=2)
plt.plot(steps_DQN, values_DQN, label="DQN", color="gold", linewidth=2)
plt.plot(steps_PPO, values_PPO, label="PPO", color="crimson", linewidth=2)
plt.grid(color="white", linestyle="-", linewidth=1)

# Add labels, title, and legend
plt.xlabel("Training Steps")
plt.ylabel("Length")
plt.title("Mean Episode Length")
plt.legend()

name = "mean_episode_length_2"
# Save the plot as an image
plt.savefig(f"/gpfsnyu/home/lm5489/Protein-Design-RL/figures/problem_2/{name}.pdf")  # Save as PDF

# Show the plot
plt.show()
