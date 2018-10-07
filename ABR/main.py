import gym
import gym_abr
import numpy as np
from ddpg import DDPG
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MEMORY_CAPACITY = 10000

env = gym.make('ABR-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
os.system('mkdir results')
ddpg = DDPG(a_dim, s_dim, a_bound)
var = 3.  # control exploration
t1 = time.time()
i = 1
#we don't record first 200 steps
log_flag = False
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
            log_flag = True

        s = s_
        ep_reward += r
        j += 1
        if done:
            break
    if i % 100 == 0:
        print('Episode:', i, ' Reward: %.2f' %
          float(ep_reward / j), 'Explore: %.2f' % var, )
    if i % 1000 == 0:
        ddpg.save_model()
    if log_flag:
        ddpg.store_summaries(ep_reward / j, i)
        i += 1
    var *= (1. - 1e-4)    # decay the action randomness
print('Running time: ', time.time() - t1)
