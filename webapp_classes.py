import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class TictacEnv:
    """
    The class representing the game environment.
    """
    def __init__(self, size=3, randomOpponent=False, visualise = False):
        self.size = size
        self.board = np.zeros((size,size), np.int8)
        self.actionSpaceSize = size**2
        self.observationSpaceSize = size**2
        self.randomOpponent = randomOpponent
        if randomOpponent:
            self.rewardValues = {'undecided': 0, 'win_1': 50, 'win_-1': -5, 'illegal': -10, 'draw': 10}
        else:
            self.rewardValues = {'undecided': 0, 'win_1': 50, 'win_-1': 5, 'illegal': -10, 'draw': 10}
        self.visualise = visualise
        self.round = 0

    def _getObs(self):
        return self.board.flatten()

    def _updateBoard(self, row, column, symbol):
        if self.board[row, column] != 0:
            raise ValueError("Playing in an occupied field.")
        if symbol != 1 | symbol != -1:
            raise ValueError("Symbol to play not valid - must be 1 or -1.")
        self.board[row, column] = symbol

    def _evalAndUpdate(self, row, column, symbol):
        if self.board[row, column] != 0:
            return "illegal"
        self._updateBoard(row, column, symbol)
        rows = np.array(np.vsplit(np.transpose(self.board), self.size))[:,0,:]
        columns = np.array(np.vsplit(self.board, self.size))[:,0,:]
        diagonals = np.array([[self.board[0,0],self.board[1,1],self.board[2,2]], [self.board[0,2],self.board[1,1],self.board[2,0]]])
        lines = np.vstack([rows,columns,diagonals])
        for line in lines:
            if np.all(line == line[0]) and line[0] != 0:
                if line[0] == 1:
                    return 'win_1'
                else:
                    return 'win_-1'
        if not np.any(self.board == 0):
            return 'draw'
        return 'undecided'

    def _terminated(self, eval):
        return eval == 'win_1' or eval == 'win_-1' or eval == 'draw'

    def step(self, action):
        if action.field >= self.size**2 or action.field < 0:
            raise IndexError("Updating a nonexistent field.")
        row = action.field // self.size
        column = action.field % self.size

        eval = self._evalAndUpdate(row, column, action.symbol)

        if self.randomOpponent and eval == 'undecided':
            randomRow, randomColumn = self.randomPlay()
            eval = self._evalAndUpdate(randomRow, randomColumn, -1)
        
        reward = self.rewardValues[eval] #self._getReward(eval)
        observation = self._getObs()
        terminated = self._terminated(eval)
        truncated = self.round >= 20
        info = None
        #self.state = self.board.flatten()
        if self.visualise:
            self.printBoard()
            print(observation, reward, terminated, truncated, info)
        
        self.round += 1
        return observation, reward, terminated, truncated, info

    # def _getReward(self, eval):
    #     if eval == 'undecided':
    #         reward = len(np.where(self.board > 0))
    #     else:
    #         reward = self.rewardValues[eval]
    #     return reward

    def randomPlay(self):
        emptyFields = np.where(self.board == 0)
        emptyCoords = list(zip(emptyFields[0], emptyFields[1]))
        randomField = emptyCoords[np.random.choice(len(emptyCoords))]
        return randomField[0], randomField[1]

    def reset(self):
        self.board = np.zeros((self.size,self.size), np.int8)
        self.round = 0
        observation = self._getObs()
        return observation
    
    def printBoard(self):
        print("\n---------\n".join([" | ".join(row) for row in self.board.astype(str)]) + "\n")

class Action:
    def __init__(self, field, symbol):
        self.field = field
        if symbol != 1 | symbol != -1:
            raise ValueError("Symbol to play not valid - must be 1 or -1.")
        self.symbol = symbol

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
    def __init__(self, obs_size, n_actions, symbol, loadFile):

        self.symbol = symbol

        # Save the number of actions and the observation size
        self.n_actions = n_actions
        self.obs_size = obs_size

        # Define the value network for the agent.
        self.value_network = Q_net(9,9)
        self.value_network.load_state_dict(torch.load(loadFile))

    def play(self, observation):
        input = torch.tensor(observation, dtype=torch.float32)
        action_values = self.value_network(input)
        action = action_values.argmax().item()
        row = action // 3
        col = action % 3
        return row, col

