import torch 
import torch.nn as nn

# Define the Q_net class
class Q_net(nn.Module):
    # Initialise the network using the size of the observation space and the number of actions
    def __init__(self, obs_size, n_actions):
        # Use the nn.Module's __init__ method to ensure that the parameters can be updated during training
        super().__init__()

        # Define the layers of the network
        self.Network = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softplus()
        )


    # Define the forward method
    def forward(self, x):
        return self.Network(x)
    
    
class Game_agent:
    def __init__(self, obs_size, n_actions, size, symbol, loadFile):

        self.symbol = symbol
        self.size = size
        # Save the number of actions and the observation size
        self.n_actions = n_actions
        self.obs_size = obs_size

        # Define the value network for the agent.
        self.value_network = Q_net(obs_size, n_actions)
        self.value_network.load_state_dict(torch.load(loadFile))

    def play(self, observation):
        input = torch.tensor(observation, dtype=torch.float32)
        action_values = self.value_network(input)
        action = action_values.argmax().item()
        row = action // self.size
        col = action % self.size
        return row, col

