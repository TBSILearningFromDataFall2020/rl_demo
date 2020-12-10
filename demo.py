'''
value iteration to solve the maze
'''

import matplotlib.pyplot as plt

from maze import MazeEnvSpecial4x4
from value_iteration import ValueIteration

plt.ion()

env = MazeEnvSpecial4x4()
alg = ValueIteration(env)
alg.train()
done_cnt = 0
current_state = env.reset()
trajectory_mat = env.reward.copy()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
while True:
    action = alg.predict(current_state)
    trajectory_mat[current_state[0], current_state[1]] = 5
    ax.matshow(trajectory_mat, cmap="coolwarm")
    plt.pause(0.5)
    current_state, reward, done, _ = env.step(action)
    if done:
        break
    done_cnt += 1
trajectory_mat[current_state[0], current_state[1]] = 5
ax.matshow(trajectory_mat, cmap="coolwarm")
plt.pause(0.5)