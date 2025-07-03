import pygame
import random
import numpy as np

class DinoGame:
    def __init__(self, width=600, height=200):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.state_size = 4
        self.action_size = 2
        self.reset()

    # Reseta o jogo
    def reset(self):
        self.dino_y = self.height - 40
        self.jump_speed = 0
        self.gravity = 1
        self.is_jumping = False
        self.obstacle_x = self.width
        self.obstacle_width = 20
        self.obstacle_height = 40
        self.done = False
        self.score = 0
        return self.get_state()

    # Constrói o vetor de estado com informações essenciais (altura do dino, velocidade de pulo, distância até obstáculo e largura do obstáculo)
    def get_state(self):
        distance = self.obstacle_x
        return np.array([self.dino_y, self.jump_speed, distance, self.obstacle_width])

    # Atualiza lógica do jogo - +1 por sobreviver, -100 ao colidir.
    def step(self, action):
        reward = 1.0 

        if action == 1 and not self.is_jumping: # 2 ações: pular ou não fazer nada, 0 nao faz nada, 1 pula
            self.jump_speed = -15
            self.is_jumping = True

        self.jump_speed += self.gravity
        self.dino_y += self.jump_speed

        if self.dino_y >= self.height - 40:
            self.dino_y = self.height - 40
            self.is_jumping = False

        self.obstacle_x -= 10
        if self.obstacle_x < -self.obstacle_width:
            self.obstacle_x = self.width + random.randint(0, 100)

        if self.check_collision():# -100 ao colidir.
            reward = -100
            self.done = True

        self.score += 1
        return self.get_state(), reward, self.done

    # Verifica colisão com obstáculo
    def check_collision(self):
        if (self.obstacle_x < 40 and self.obstacle_x + self.obstacle_width > 0 and
            self.dino_y + 40 > self.height - self.obstacle_height):
            return True
        return False

    # Desenha a cena com pygame
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))  # Fundo branco
        pygame.draw.rect(self.screen, (0, 0, 0), (20, self.dino_y, 20, 40))  # Dino
        pygame.draw.rect(self.screen, (255, 0, 0), (self.obstacle_x, self.height - self.obstacle_height, self.obstacle_width, self.obstacle_height))  # Obstáculo
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
