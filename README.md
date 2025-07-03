# 🦖 Dino AI – Aprendizado por Reforço com Deep Q-Learning (DQN)

Este projeto implementa um **agente inteligente** utilizando **Deep Q-Learning (DQN)** para jogar uma versão simplificada do famoso jogo do dinossauro offline do Google Chrome.

O objetivo é demonstrar, de forma didática, como um agente pode aprender a jogar **apenas por tentativa e erro**, utilizando técnicas de **Aprendizado por Reforço** com redes neurais.


![Demo do Jogo](DinoGame.gif)


---

## 📁 Estrutura do Projeto

```
.
├── main.py                  # Script principal: executa o treinamento
├── agent/
│   └── dqn.py              # Definição do agente DQN e da rede neural (PyTorch)
├── env/
│   ├── dino_game.py        # Ambiente visual com Pygame (dinossauro + obstáculo)
│   └── dino_env.py         # Ambiente simulado simplificado (opcional)
├── results/
│   ├── training_plot.png   # Gráfico de desempenho gerado ao final do treino
│   └── dqn_model.pth       # Modelo salvo após o treinamento
└── requirements.txt        # Lista de dependências (PyTorch, numpy, pygame etc)
```

---

## 🧠 Técnicas Utilizadas

- **Deep Q-Learning (DQN)**
- **Redes neurais com PyTorch**
- **Replay Buffer (memória de experiências)**
- **Epsilon-greedy (exploração x exploração)**
- **Ambiente próprio com Pygame**

---

## ✅ Pré-requisitos

Você precisa do Python 3.8+ e pip instalado. Para instalar as dependências:

```bash
pip install -r requirements.txt
```

---

## ▶️ Como Executar

```bash
python main.py
```

Durante a execução, o agente será treinado em tempo real, com visualização do jogo (usando `pygame`).

---

## 📊 Saídas do Projeto

- 📈 **`results/training_plot.png`**  
  Gráfico com a pontuação acumulada por episódio, mostrando a evolução do agente.

- 💾 **`results/dqn_model.pth`**  
  Arquivo contendo os pesos da rede neural treinada.

---

## 💡 Possíveis melhorias

- Adicionar **rede-alvo** (target network)
- Implementar **decaimento do epsilon**
- Incluir **múltiplos obstáculos e variáveis de velocidade**

---

## 📚 Referências

- [DQN - DeepMind 2015](https://www.nature.com/articles/nature14236)
- PyTorch: https://pytorch.org

---

## 🧪 Propósito acadêmico

Este projeto foi desenvolvido como parte de uma disciplina de Inteligência Artificial no curso de **Jogos Digitais** da **Unisinos**, com foco em aprendizado por reforço aplicado a jogos.