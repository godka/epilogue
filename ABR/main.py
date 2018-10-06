import gym
import gym_abr
import numpy as np
from ddpg import DDPG
import os
import time

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MAX_EPISODES = 200
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000

env = gym.make('ABR-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)
_file = open('test.csv', 'w')
var = 3.  # control exploration
t1 = time.time()
i = 0
while True:
    s = env.reset()
    ep_reward = 0
    j = 0
    while True:
        # Add exploration noise
        a = ddpg.choose_action(s)
        # add randomness to action selection for exploration
        a = np.clip(np.random.normal(a, var), 0., 60.)
        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            ddpg.learn()

        s = s_
        ep_reward += r
        j += 1
        if done:
            break
    print('Episode:', i, ' Reward: %.2f' %
          float(ep_reward / j), 'Explore: %.2f' % var, )
    i += 1
    _file.write(str(ep_reward / j) + '\n')
    _file.flush()
    var *= 0.999999    # decay the action randomness
_file.close()
print('Running time: ', time.time() - t1)
