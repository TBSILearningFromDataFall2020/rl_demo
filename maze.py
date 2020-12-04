import numpy as np

from util import Discrete, Box

"define the environment"
class MazeEnv:
    def __init__(self, reward_matrix):
        # set the reward matrix
        self.reward = reward_matrix
        self.state = [0, 0]

        self.max_row, self.max_col = reward_matrix.shape
        self.action_map = {0: [0, -1], # left
                           1: [0, 1],  # right
                           2: [-1, 0], # up
                           3: [1, 0],  # down
                           }
        self.action_space = Discrete(4)
        self.observation_space = Box([0, 0], [self.max_row - 1, self.max_col - 1])

    def step(self, action):
        """Given an action, outputs the reward.
        Args:
            action: int, an action taken by the agent
            state: str, the current state of the agent

        Outputs:
            next_state: list, next state of the agent
            reward: float, the reward at the next state
            done: bool, stop or continue learning
        """

        done = False
        state_shift = self.action_map[action]

        next_state = [0, 0]
        next_state[0] = self.state[0] + state_shift[0]
        next_state[1] = self.state[1] + state_shift[1]

        if next_state[0] < 0 or next_state[0] >= self.max_row or \
            next_state[1] < 0 or next_state[1] >= self.max_col:
            next_state = self.state # stay at current position

        reward = self.reward[next_state[0], next_state[1]]

        # fixed condition adviced by Shi Mao
        if reward < 0 or reward > 0:
            done = True
        self.state = next_state

        return next_state, reward, done, None

    def reset(self):
        """Reset the agent state
        """

        self.state = [0, 0]
        return self.state

class MazeEnvSample3x3(MazeEnv):
    def __init__(self):
        reward_matrix = np.zeros([3, 3], dtype=int)
        reward_matrix[2, 2] = 1
        super(MazeEnvSample3x3, self).__init__(reward_matrix)

class MazeEnvSpecial4x4(MazeEnv):
    def __init__(self):
        reward_matrix = np.zeros([4, 4], dtype=int)
        reward_matrix[3, 3] = 4
        reward_matrix[1, 1] = -5 # block which could not be reached
        reward_matrix[1, 3] = -5
        reward_matrix[3, 1] = -5
        super(MazeEnvSpecial4x4, self).__init__(reward_matrix)