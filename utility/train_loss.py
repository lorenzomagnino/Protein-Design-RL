import pandas as pd
import matplotlib.pyplot as plt




""" ---- Train_loss ---- """
# Load the CSV files
csv_file_A2C = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_1.py/train_entropy_loss_A2C.csv"
data_A2C = pd.read_csv(csv_file_A2C)

csv_file_DQN = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_1.py/train_loss_DQN.csv"
data_DQN = pd.read_csv(csv_file_DQN)

csv_file_PPO = "/gpfsnyu/home/lm5489/Protein-Design-RL/utility/results_csv/problem_1.py/train_entropy_loss_PPO.csv"
data_PPO = pd.read_csv(csv_file_PPO)

# Extract Steps and Values for each algorithm
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
#plt.plot(steps_DQN, values_DQN, label="DQN", color="gold", linewidth=2)
#plt.plot(steps_PPO, values_PPO, label="PPO", color="crimson", linewidth=2)
plt.grid(color="white", linestyle="-", linewidth=1)

# Add labels, title, and legend
plt.xlabel("Training Steps")
plt.ylabel("Entropy Loss")
plt.title("Train Entropy Loss A2C")
plt.legend()

# Save the plot as an image
plt.savefig("/gpfsnyu/home/lm5489/Protein-Design-RL/figures/problem_1/train_entropy_loss_A2C.pdf")  # Save as PDF

# Show the plot
plt.show()
