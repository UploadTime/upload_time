import pandas as pd
import matplotlib.pyplot as plt
import random

def plot_u():
    # read
    data_x = pd.read_excel('data.xlsx', sheet_name='x_axis')
    x = data_x['x_axis']
    # print(x)
    # print(type(x))

    data_tra = pd.read_excel('data.xlsx', sheet_name='train reward')
    train = data_tra['train reward']
    # print(train)

    data_tes = pd.read_excel('data.xlsx', sheet_name='test reward')
    test = data_tes['test reward']
    # print(test)
    
    # plot
    plt.figure(2)
    plt.clf()
    plt.plot(x, train, 'r', label='train')
    plt.plot(x, test, 'b', label='test')
    plt.title('Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(["train", "test"])
    # plt.show()
    plt.savefig('result_pic_final.png')

if __name__ == '__main__':
    plot_u()