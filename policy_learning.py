import gym
from gym import wrappers, logger
import numpy as np
import pickle
import json, sys, os
from os import path

import argparse
'''
solve continuous binary decision problem
'''
class CELearning:
    def __init__(self, env, elite_frac=0.2, batch_size=25, max_iter=10):
        self.env = env
        self.elite_frac = elite_frac
        self.batch_size = batch_size
        self.max_iter = max_iter
        state_dim = env.observation_space.shape[0]
        self.theta = np.zeros(state_dim + 1) # plus bias

    def select_action(self, state, theta):
        w = theta[:-1]
        b = theta[-1]
        y = state.dot(w) + b
        a = int(y < 0)
        return a # 0 or 1

    def get_reward(self, theta, num_steps=200):
        total_reward = 0
        state = self.env.reset()
        for t in range(num_steps):
            action = self.select_action(state, theta)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def train(self):
        n_elite = int(self.batch_size * self.elite_frac)
        theta_std = np.ones(self.theta.shape)
        theta_size = self.theta.size
        for i in range(self.max_iter):
            theta_list = np.array([self.theta + dth for dth in theta_std[None,:] * \
                                   np.random.randn(self.batch_size, theta_size)])
            reward_list = np.array([self.get_reward(theta) for theta in theta_list])
            elite_indexes = reward_list.argsort()[::-1][:n_elite]
            elite_theta_list = theta_list[elite_indexes]
            self.theta = elite_theta_list.mean(axis=0)
            theta_std = elite_theta_list.std(axis=0)

    def predict(self, state):
        # given the current state, select the best action
        return self.select_action(state, self.theta)


