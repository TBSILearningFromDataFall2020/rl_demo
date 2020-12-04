import unittest
import numpy as np
import gym

from maze import MazeEnvSample3x3, MazeEnvSpecial4x4
from ipendulum import InvertedPendulumEnv

from qlearning import QTableLearning
from policy_learning import CELearning

class TestMaze(unittest.TestCase):
    def test_3x3_maze(self):
        env = MazeEnvSample3x3()
        current_state, _, _, _ = env.step(1) # right
        self.assertEqual(current_state, [0, 1])
        current_state, _, _, _ = env.step(3) # down
        self.assertEqual(current_state, [1, 1])
        current_state, _, _, _ = env.step(3) # down
        self.assertEqual(current_state, [2, 1])
        current_state, reward, done, _ = env.step(1) # right
        self.assertEqual(current_state, [2, 2])
        self.assertTrue(done)
        self.assertTrue(reward > 0)
        env.reset()
        self.assertEqual(env.state, [0, 0])

    def test_4x4_maze(self):
        env = MazeEnvSpecial4x4()
        env.step(1) # right
        current_state, reward, done, _ = env.step(3)
        self.assertTrue(reward < 0)
        self.assertTrue(done)

    def test_3x3_maze_q_table_learning(self):
        env = MazeEnvSample3x3()
        alg = QTableLearning(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertTrue(done_cnt < 10)

    def test_4x4_maze_q_table_learning(self):
        env = MazeEnvSpecial4x4()
        alg = QTableLearning(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertTrue(done_cnt < 10)

class TestPendulum(unittest.TestCase):
    def test_model(self):
        env = InvertedPendulumEnv()
        env.step(1) # right force
        state, _, _, _ = env.step(1)

        self.assertTrue(state[0] > 0)
        self.assertTrue(state[2] > 0)
        self.assertTrue(state[3] > 0)
        env.step(0)
        env.step(0)
        env.step(0)
        state, _, done, _ = env.step(0)
        self.assertTrue(state[2] < 0)
        self.assertTrue(state[3] < 0)
        self.assertFalse(done)

    def test_cross_entropy_learning(self):
        env = InvertedPendulumEnv()
        alg = CELearning(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            # action = env.action_space.sample()
            current_state, reward, done, _ = env.step(action)
            if done or done_cnt == 1000:
                break
            done_cnt += 1
        self.assertTrue(done_cnt > 25)


if __name__ == '__main__':
    unittest.main()