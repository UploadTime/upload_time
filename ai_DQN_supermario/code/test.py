import numpy as np
import torch
import config
from model import DQN
from wrappers import wrapEnvironment

def test(environment, action_space, iteration, test_reward):
    flag = False
    env = wrapEnvironment(environment, action_space, monitor=True, iteration=iteration)
    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(f'models/{environment}.dat'))
    
    totalReward = 0.0
    state = env.reset()
    while True:
        state_v = torch.tensor(np.array([state], copy=False))
        q_values = net(state_v).data.numpy()[0]
        action = np.argmax(q_values)
        state, reward, done, info = env.step(action)
        totalReward += reward
        
        if config.render:
            env.render()
        
        if info['flag_get']:
            flag = True
            print('Congrates.')
        
        if done:
            print(f'Test reward: {totalReward}')

            test_reward.append(totalReward)

            break

    # env.close()
    return flag