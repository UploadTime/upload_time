import numpy as np

# import random
# from collections import namedtuple

# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))



class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._alpha = alpha
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        # max_prio = self._priorities.max() if self._buffer else 1.0
        if self._buffer:
            max_prio = self._priorities.max()
        else:
            max_prio = 1.0
        
        batch = (state, action, reward, next_state, done)
        
        if len(self._buffer) < self._capacity:
            self._buffer.append(batch)
        else:
            self._buffer[self._position] = batch

        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4):
        if len(self._buffer) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]

        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def updatePriorities(self, batchIndices, batchPriorities):
        for idx, prio in zip(batchIndices, batchPriorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._buffer)

    # def __init__(self, capacity):
    #     self.capacity = capacity
    #     self.memory = []
    #     self.position = 0

    # def push(self, state, action, reward, next_state, done):
    #     batch = (state, action, reward, next_state, done)
    #     if len(self.memory) < self.capacity:
    #         self.memory.append(None)
    #     self.memory[self.position] = batch
    #     self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    
    # def updatePriorities(self, batchIndices, batchPriorities):
    #     for idx, prio in zip(batchIndices, batchPriorities):
    #         self._priorities[idx] = prio

    # def __len__(self):
    #     return len(self.memory)
