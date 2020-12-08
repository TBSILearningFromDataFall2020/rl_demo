import numpy as np

from util import Discrete, Box



"define the environment"
class InvertedPendulumEnv:
    def __init__(self):
        self.tau = 0.02 # time delta between two consecutive steps
        self.delta_x = 4.8
        self.delta_theta = 12 * np.pi / 180
        self.delta_theta_prime = 45 * np.pi / 180
        self.F = 10
        self.m = 0.1
        self.M = 1
        self.L = 0.5
        self.g = 9.8
        self.action_space = Discrete(2)
        lower_bound = [-1.0 * self.delta_theta, -1.0 * self.delta_theta, -1.0 * np.Inf, -1.0 * np.Inf]
        upper_bound = [self.delta_theta, self.delta_theta, np.Inf, np.Inf]
        self.observation_space = Box(lower_bound, upper_bound)
        self.reset()
    def peek_step(self, state, action):
        x = state[0]
        x_dot = state[2]
        theta = state[1]
        theta_dot = state[3]        
        if action:
            F = self.F
        else:
            F = -1.0 * self.F
        # compute acceleration of x and theta
        _matrix_22 = [[self.m + self.M, -1.0 * self.m * self.L * np.cos(theta)],
                     [np.cos(theta), -1.0 * self.L]]
        _vector_2 = [F - self.m * self.L * theta_dot ** 2 * np.sin(theta),
                    -1.0 * self.g * np.sin(theta)]
        x_double_dot, theta_double_dot = np.linalg.solve(_matrix_22, _vector_2)
        # set new state
        x_prime = x + self.tau * x_dot
        theta_prime = theta + self.tau * theta_dot
        x_prime_dot = x_dot + self.tau * x_double_dot
        theta_prime_dot = theta_dot + self.tau * theta_double_dot
        return np.array([x_prime, theta_prime, x_prime_dot,
                      theta_prime_dot])

    def step(self, action):
        """Given an action, outputs the reward.
        Args:
            action: boolean, an action taken by the agent

        Outputs:
            next_state: 1 x 4 array, the next state of the agent
            reward: float, the reward at the next state
            done: bool, stop or continue learning
        """
        

        next_state = self.peek_step(self.state, action)


        reward = 1
        # check termination condition
        if np.abs(next_state[1]) > self.delta_theta_prime or \
            np.abs(next_state[0]) > self.delta_x:
            done = True
        else:
            done = False
        self.state = next_state

        return self.state, reward, done, None

    def reset(self):
        """Reset the agent state

        Outputs:
            state: 1 x 4 array, the initial state.
        """
        self.state = np.array([0, 13 * np.pi / 180, 0, 0])

        return self.state
