import numpy as np

class Discrete:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(0, self.n)

class Box:
    def __init__(self, low, high):
        self.shape = (len(low),)
        self.low = low
        self.high = high