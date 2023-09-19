import config
from replayBuffer import PrioritizedBuffer
from wrappers import wrapEnvironment
import torch
from torch.optim import Adam
from train import train
from model import DQN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc

def main():
    train_reward = []
    test_reward = []
    x_axis = []
    env = wrapEnvironment(config.environment, config.action_space)
    device = torch.device('cuda')

    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    targetModel = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if config.transfer:
        model.load_state_dict(torch.load(f'models/{config.environment}.dat'))
        targetModel.load_state_dict(model.state_dict())
    
    optimizer = Adam(model.parameters(), lr=config.lr)
    replayBuffer = PrioritizedBuffer(config.buffer_capacity)
    
    train(env, model, targetModel, optimizer, replayBuffer, device, train_reward, test_reward, x_axis)
    env.close()

    # print(train_reward)
    # print(test_reward)

    # plot
    plt.figure(2)
    plt.clf()
    plt.plot(x_axis, train_reward, 'r', label='train')
    plt.plot(x_axis, test_reward, 'b', label='test')
    plt.title('Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(["train", "test"])
    # plt.show()
    plt.savefig('result_pic.png')

    # x_axis.append(1)
    # x_axis.append(2)
    # x_axis.append(3)
    # x_axis.append(4)
    # x_axis.append(5)
    # x_axis.append(6)
    # train_reward.append(1)
    # train_reward.append(2)
    # train_reward.append(3)
    # train_reward.append(4)
    # train_reward.append(5)
    # train_reward.append(1)
    # test_reward.append(1)
    # test_reward.append(2)
    # test_reward.append(4)
    # test_reward.append(45)
    # test_reward.append(1)
    # test_reward.append(2)

    # store to excel
    gc.disable()
    num = len(x_axis)
    # arr = x_axis
    # arr.extend(train_reward)
    # arr.extend(test_reward)
    x = np.array(x_axis)
    tra = np.array(train_reward)
    tes = np.array(test_reward)
    data1 = x.reshape(num, 1)
    data2 = tra.reshape(num, 1)
    data3 = tes.reshape(num, 1)
    # data1 = x
    # data2 = tra
    # data3 = tes
    data_df1 = pd.DataFrame(data1, columns=['x_axis'])
    data_df2 = pd.DataFrame(data2, columns=['train reward'])
    data_df3 = pd.DataFrame(data3, columns=['test reward'])
    # data_df = pd.DataFrame(np.array(arr).reshape(3, num), index=['x', 'train', 'test'])
    with pd.ExcelWriter('data.xlsx') as writer:
        data_df1.to_excel(writer, sheet_name='x_axis', float_format='%.6f')
        data_df2.to_excel(writer, sheet_name='train reward', float_format='%.6f')
        data_df3.to_excel(writer, sheet_name='test reward', float_format='%.6f')
        # data_df.to_excel(writer, sheet_name='data', float_format='%.6f')
    gc.enable()




if __name__ == '__main__':
    main()
