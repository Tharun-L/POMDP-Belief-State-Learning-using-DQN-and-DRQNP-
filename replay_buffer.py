import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, episode):
        self.buffer.append(episode)

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size, seq_len):
        batch = []

        while len(batch) < batch_size:
            episode = random.choice(self.buffer)

            if len(episode) < seq_len:
                continue

            start = random.randint(0, len(episode) - seq_len)
            batch.append(episode[start:start + seq_len])

        return batch

    def __len__(self):
        return len(self.buffer)
