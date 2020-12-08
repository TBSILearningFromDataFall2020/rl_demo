import unittest
import numpy as np
from sklearn.utils._testing import assert_array_almost_equal

from maze import MazeEnvSample3x3, MazeEnvSpecial4x4, MazeEnvSpecial5x5
from ipendulum import InvertedPendulumEnv

from qlearning import QTableLearning
from value_iteration import ValueIteration
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

    def test_3x3_maze_value_iteration(self):
        env = MazeEnvSample3x3()
        alg = ValueIteration(env, max_iter=90)
        alg.train()
        expected_values = np.array([[2.048, 2.56, 3.2], [2.56, 3.2, 4], [3.2, 4, 5]])
        # expected values are solved by Bell equation x = 1 + 0.8 * x for V[2, 2] = 5, etc..
        assert_array_almost_equal(alg.values, expected_values)
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertEqual(done_cnt, 3)
        self.assertEqual(reward, 1)

    def test_4x4_maze_value_iteration(self):
        env = MazeEnvSpecial4x4()
        alg = ValueIteration(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertEqual(done_cnt, 5)
        self.assertEqual(reward, 4)

    def test_5x5_maze_value_iteration(self):
        env = MazeEnvSpecial5x5()
        alg = ValueIteration(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertEqual(done_cnt, 15)
        self.assertEqual(reward, 1)

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
