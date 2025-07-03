import numpy as np
import random

# Ambiente fake para teste (sem visual)
class DinoEnv:
    def __init__(self):
        self.state_size = 4
        self.action_size = 2

    def reset(self):
        self.state = np.zeros(self.state_size)
        return self.state

    def step(self, action):
        reward = 1.0
        done = random.random() < 0.01  # Fim aleatório
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done
