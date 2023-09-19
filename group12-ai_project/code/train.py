import math
import numpy as np
from shutil import copyfile
import config
from test import test
# from torch import save
from torch import FloatTensor, LongTensor, save
from torch.autograd import Variable


def compute_td_loss(model, targetModel, replayBuffer, gamma, device, batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replayBuffer.sample(batch_size, beta)

    state = Variable(FloatTensor(np.float32(state))).to(device)
    next_state = Variable(FloatTensor(np.float32(next_state))).to(device)
    action = Variable(LongTensor(action)).to(device)
    reward = Variable(FloatTensor(reward)).to(device)
    done = Variable(FloatTensor(done)).to(device)
    weights = Variable(FloatTensor(weights)).to(device)

    q_value = model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = targetModel(next_state).max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    # q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    # next_q_value = next_q_values.max(1)[0]
    
    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
    loss.backward()
    replayBuffer.updatePriorities(indices, prios.data.cpu().numpy())

# model training
def train(env, model, targetModel, optimizer, replayBuffer, device, train_reward, test_reward, x_axis):
    index = 0
    averageRange = 100
    newBestCounter = 0
    bestReward = -float('inf')
    bestAverage = -float('inf')
    rewards = []

    for i in range(config.num_epochs):
        
        epochReward = 0.0
        state = env.reset()

        while True:
            epsilon = config.epsFinal + (config.epsStart - config.epsFinal) * math.exp(-1 * ((index + 1) / config.decay))
            
            if len(replayBuffer) > config.batch_size:
                beta = min(1.0, config.start + index * (1.0 - config.start) / config.frames)
            else:
                beta = 0.4
            
            action = model.getAction(state, epsilon, device)
            
            if config.render:
                env.render()
            
            next_state, reward, done, info = env.step(action)
            
            replayBuffer.push(state, action, reward, next_state, done)
            
            state = next_state
            epochReward += reward
            index += 1

            if len(replayBuffer) > config.initialLearning:
                if not index % config.targetUpdateFrequency:
                    targetModel.load_state_dict(model.state_dict())
                optimizer.zero_grad()
                compute_td_loss(model, targetModel, replayBuffer, config.gamma, device, config.batch_size, beta)
                optimizer.step()
            
            if done:
                rewards.append(epochReward)
                x = False
                if epochReward > bestReward:
                    bestReward = epochReward
                    x = True
                y = False
                if sum(rewards[(-1*averageRange):]) / len(rewards[(-1*averageRange):]) > bestAverage:
                    bestAverage = sum(rewards[(-1*averageRange):]) / len(rewards[(-1*averageRange):])
                    y = True
                
                target = x or y
                if target:
                    newBestCounter += 1 
                    print(f'We get a new best average reward: {round(bestAverage, 3)}!')
                    save(model.state_dict(), f'models/{config.environment}.dat')
                    flag = test(config.environment, config.action_space, newBestCounter, test_reward)
                    if flag:
                        copyfile(f'models/{config.environment}.dat', f'runs/run{newBestCounter}/{config.environment}.dat')
                    train_reward.append(epochReward)
                    x_axis.append(i+1)

                elif info['flag_get']:
                    newBestCounter += 1
                    save(model.state_dict(), f'models/{config.environment}.dat')
                    flag = test(config.environment, config.action_space, newBestCounter, test_reward)
                    if flag:
                        copyfile(f'models/{config.environment}.dat', f'runs/run{newBestCounter}/{config.environment}.dat')
                    train_reward.append(epochReward)
                    x_axis.append(i+1)
                
                print(f'Epoch {i+1} - '
                      f'Reward: {round(epochReward, 3)}, '
                      f'Best: {round(bestReward, 3)}, '
                      f'Average: {round(sum(rewards[(-1*averageRange):]) / len(rewards[(-1*averageRange):]), 3)} '
                      f'Epsilon: {round(epsilon, 4)}')

                break