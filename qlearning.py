import numpy as np


class QTableLearning:
    '''
    solve discrete reinforcement learning
    '''
    def __init__(self, env, gamma=0.8, alpha=0.1, eps=0.9, max_iter=10000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        # number of action space
        num_of_actions = self.env.action_space.n
        # number of state
        state_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))

        self.q_table = np.zeros(state_size + (num_of_actions,), dtype=float)

    def select_action(self, state):
        # epsilon greedy learning
        if np.random.rand() < self.eps:
            # random action
            action = self.env.action_space.sample()
        else:
            # select from Q table
            action = int(np.argmax(self.q_table[state]))

        return action

    def learn(self, state, action, reward, next_state):
        """Learn from the environment.
        Args:
            state: str, the current state
            action: int, the action taken by itself
            reward: float, the reward after taking the action
            next_state: str, the next state

        Outputs:
            None
        """
        # update Q table with Q-Table Learning
        current_q = self.q_table[state][action]

        # set feasible action space
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max(self.q_table[next_state]) - current_q)
        return
  
    def train(self):
        for i in range(self.max_iter):
            current_state = self.env.reset()
            current_state_tuple = tuple(current_state)
            while True:
                # act
                action = self.select_action(current_state_tuple)

                # get reward from the environment
                next_state, reward, done, _ = self.env.step(action)

                next_state_tuple = tuple(next_state)
                # learn from the reward
                self.learn(current_state_tuple, action, reward, next_state_tuple)

                current_state_tuple = next_state_tuple
                if done:
                    break
            # epsilon decay
            eps = self.eps
            if i % 100 == 0 and i > 0:
                eps = - 0.99 * i / self.max_iter + 1.0
                eps = np.maximum(0.1, eps)
                self.eps = eps

    def predict(self, state):
        # given the current state, select the best action
        state_tuple = tuple(np.asarray(state, dtype=int))
        return self.select_action(state_tuple)
