import numpy as np
import random


class ReplayMemory:
    def __init__(self, config):
        self.cnn_format = config.cnn_format

        self.memory_size = config.memory_size
        self.hist_len, self.screen_h, self.screen_w = config.hist_len, config.screen_h, config.screen_w
        self.screens = np.empty((self.memory_size, self.screen_h, self.screen_w), dtype=np.float16)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.terms = np.empty(self.memory_size, dtype=np.bool)
        self.batch_size = config.batch_size
        self.current = 0
        self.count = 0

        self.prestates = np.empty((self.batch_size, self.hist_len, self.screen_h, self.screen_w), dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.hist_len, self.screen_h, self.screen_w), dtype=np.float16)

    def add(self, screen, reward, action, term):
        self.screens[self.current] = screen
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terms[self.current] = term
        self.current = (self.current + 1) % self.memory_size
        self.count = min(self.memory_size, self.count + 1)

    def getState(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1!"
        #index is included
        if index - self.hist_len + 1 >= 0:
            return self.screens[(index-self.hist_len+1):(index+1), ...]
        else:
            indexes = [(index-i) for i in reversed(range(self.hist_len))]
            return self.screens[indexes, ...]
        pass

    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.hist_len, self.count-1)
                if index >= self.current and index - self.hist_len < self.current:
                    continue
                if self.terms[(index-self.hist_len):index].any():
                    continue
                break
            self.prestates[len(indexes), ...] = self.getState(index-1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)
        rewards = self.rewards[indexes]
        actions = self.actions[indexes]
        terms = self.terms[indexes]
        if self.cnn_format == 'NHWC':
            prestates = np.transpose(self.prestates, (0, 2, 3, 1))
            poststates = np.transpose(self.poststates, (0, 2, 3, 1))
            return prestates, actions, rewards, poststates, terms
        else:
            return self.prestates, actions, rewards, self.poststates, terms




