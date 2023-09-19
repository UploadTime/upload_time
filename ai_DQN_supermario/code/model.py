import numpy as np
from random import random, randrange

import torch
import torch.nn as nn   


class DQN(nn.Module):
    def __init__(self, inputShape, numActions):
        super(DQN, self).__init__()
        self._inputShape = inputShape
        self._numActions = numActions

        self.features = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(),
            nn.Linear(512, numActions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def featureSize(self):
        x = self.features(torch.zeros(1, *self._inputShape))
        return x.view(1, -1).size(1)

    def getAction(self, state, epsilon, device):
        if random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self._numActions)
        return action
