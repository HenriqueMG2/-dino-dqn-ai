import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Rede Neural que será usada para estimar os Q-valores (função Q) PyTorch e PyGame Deep Q learning 
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Rede simples com 1 camada oculta de 64 neurônios
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  # Saída com tamanho igual ao número de ações possíveis
        )

    def forward(self, x):
        return self.fc(x)

# Agente DQN que interage com o ambiente
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)  # Rede principal
        self.memory = deque(maxlen=2000)  # Memória de experiências
        self.gamma = 0.95  # Fator de desconto (prioriza recompensas futuras)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Escolhe uma ação (exploração ou exploração)
    def act(self, state):
        if random.random() < 0.1:  # 10% das vezes, escolhe aleatoriamente (exploração)
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)  # Previsão da rede
        return torch.argmax(q_values).item()  # Retorna a ação com maior Q-valor

    # Armazena uma experiência no replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Re-treinamento da rede com experiências passadas (aprendizado por repetição)
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                # Atualiza o valor alvo com base na estimativa do próximo estado
                target += self.gamma * torch.max(self.model(next_state)).item()

            current_q = self.model(state)[0, action]
            target_tensor = torch.tensor(target, dtype=torch.float32)

            loss = nn.functional.mse_loss(current_q, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()