from agent.dqn import DQNAgent
from env.dino_game import DinoGame
import matplotlib.pyplot as plt
import os

def train():
    env = DinoGame()
    agent = DQNAgent(env.get_state().shape[0], 2)

    scores = []
    for episode in range(200):  # Episódios de treino
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            env.render()  # Mostra a simulação em tempo real
            agent.remember(state, action, reward, next_state, done)
            agent.replay()  # Treina com batch aleatório da memória
            state = next_state
            total_reward += reward
        scores.append(total_reward)
        print(f"Episode {episode+1}: Score {total_reward}")

    # Cria pasta de resultados
    os.makedirs("results", exist_ok=True)

    # Salva gráfico de desempenho
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Treinamento do Dino DQN")
    plt.savefig("results/training_plot.png")

    # Salva modelo treinado
    agent.save("results/dqn_model.pth")
    env.close()

if __name__ == "__main__":
    train()
