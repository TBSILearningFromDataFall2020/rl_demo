import numpy as np


class ValueIteration:
    '''
    solve discrete 2D reinforcement learning problem
    '''
    def __init__(self, env, gamma=0.8, max_iter=100):
        self.env = env
        self.gamma = gamma # discounted rate
        self.max_iter = max_iter
        # number of action space
        self.num_of_actions = self.env.action_space.n
        # number of state
        state_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))

        self.values = np.zeros(state_size, dtype=float)
        self.actions = np.zeros(state_size, dtype=int)

    def _select_action(self, state):
        max_value = -1.0 * np.Inf
        optimal_action = 0
        for action in range(self.num_of_actions):
            next_state = self.env.peek_step(state, action)
            next_state_tuple = tuple(np.asarray(next_state, dtype=int))
            if self.values[next_state_tuple] > max_value:
                max_value = self.values[next_state_tuple]
                optimal_action = action
        return optimal_action, max_value

    def train(self):
        for _ in range(self.max_iter):
            for j_1 in range(self.values.shape[0]):
                for j_2 in range(self.values.shape[1]):
                    _, next_value = self._select_action([j_1, j_2])
                    self.values[(j_1, j_2)] = self.env.reward[(j_1, j_2)] + self.gamma * next_value
        # update action matrix
        for j_1 in range(self.values.shape[0]):
            for j_2 in range(self.values.shape[1]):
                action, _ = self._select_action([j_1, j_2])
                self.actions[(j_1, j_2)] = action


    def predict(self, state):
        # given the current state, select the best action
        state_tuple = tuple(np.asarray(state, dtype=int))
        return self.actions[state_tuple]
