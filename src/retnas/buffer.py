import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class ReplayBuffer:
    DEFAULT_CAPACITY = 50000

    def __init__(self, max_size=DEFAULT_CAPACITY, seed=42):
        self.storage = [None] * max_size
        self.max_size = max_size
        self.current_size = 0
        self.ptr = int(0)
        random.seed(seed)

    def push(self, data):
        self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size):
        if self.current_size < batch_size:
            batch_data = random.choices(self.storage[:self.current_size], k=batch_size)
        else:
            batch_data = random.sample(self.storage[:self.current_size], k=batch_size)
        (state, next_state, action, reward, done) = tuple(zip(*batch_data))
        num_sjs = np.fromiter(map(len, state), dtype=np.int64)
        next_num_sjs = np.fromiter(map(len, next_state), dtype=np.int64)
        padded_state = pad_sequence([torch.from_numpy(s) for s in state], batch_first=True, padding_value=0).numpy()
        padded_next_state = pad_sequence([torch.from_numpy(s) for s in next_state], batch_first=True, padding_value=0).numpy()
        padded_action = pad_sequence([torch.from_numpy(s) for s in action], batch_first=True, padding_value=0).numpy()
        return (padded_state, padded_next_state, padded_action, np.array(reward).astype(float).reshape(-1, 1), num_sjs, next_num_sjs, np.array(done, dtype=int).reshape(-1, 1))
