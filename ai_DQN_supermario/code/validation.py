import numpy as np
import torch
import config
from model import DQN
from wrappers import wrapEnvironment

def test(environment, action_space):
    flag = False
    env = wrapEnvironment(environment, action_space, monitor=True, iteration=0)
    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(f'SuperMarioBros-1-1-v0.dat'))
    
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

            break

    # env.close()
    return flag

if __name__ == '__main__':
    test(config.environment, config.action_space)