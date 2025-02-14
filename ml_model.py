import pygame
import random
import numpy as np
import sys
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================
# Global Constants & Settings
# ============================
BLOCK_SIZE   = 20      # Size of one grid block in pixels
SPEED        = 40      # Game speed (frames per second)
OBSTACLE_FREQ = SPEED * 2  # Every 2 seconds, spawn an obstacle

# Colors (R,G,B)
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE  = (0, 0, 255)   # Color for obstacles
BLACK = (0, 0, 0)

# ============================
# Snake Game Environment Class
# ============================
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # window width in pixels
        self.h = h  # window height in pixels
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI with Obstacles')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Initialize snake at the center of the grid
        self.direction = (1, 0)  # Initially moving right
        x = (self.w // BLOCK_SIZE) // 2
        y = (self.h // BLOCK_SIZE) // 2
        self.head = (x, y)
        self.snake = [self.head,
                      (self.head[0] - 1, self.head[1]),
                      (self.head[0] - 2, self.head[1])]
        self.score = 0
        self.food = None
        self.obstacles = set()  # Store obstacle positions as a set of (x,y) tuples
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        grid_width  = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        self.food = (x, y)
        # Ensure food is not placed on the snake or on an obstacle
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def spawn_obstacle(self):
        """Spawn a new obstacle at a random free location."""
        grid_width  = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        # List of available positions (not occupied by snake, food, or other obstacles)
        available = [(x, y) for x in range(grid_width) for y in range(grid_height)
                     if (x, y) not in self.snake and (x, y) != self.food and (x, y) not in self.obstacles]
        if available:
            pos = random.choice(available)
            self.obstacles.add(pos)

    def play_step(self, action):
        """
        Executes one step of the game.
          - action: list of three integers [1,0,0] (go straight), [0,1,0] (turn right) or [0,0,1] (turn left)
        Returns: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # Spawn obstacles every OBSTACLE_FREQ frames
        if self.frame_iteration % OBSTACLE_FREQ == 0:
            self.spawn_obstacle()

        # 1. Handle pygame events (e.g., window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Move the snake according to the action
        self._move(action)  # updates self.head
        self.snake.insert(0, self.head)

        # 3. Check for collisions (walls, self, obstacles)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # If no food is eaten, remove the tail
            self.snake.pop()

        # 5. Update UI and tick the clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Return True if point 'pt' collides with wall, snake body, or an obstacle."""
        if pt is None:
            pt = self.head
        x, y = pt
        grid_width  = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # Draw the snake
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, GREEN, rect)
        # Draw the food (apple)
        food_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, RED, food_rect)
        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[0]*BLOCK_SIZE, obs[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, BLUE, obs_rect)
        # Draw the score text
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Updates the snake's direction and head based on the action.
          - action: [straight, right, left]
        The action is interpreted relative to the current direction.
        """
        # Define clockwise order: right, down, left, up
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # No change: go straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1] turn left
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head
        dx, dy = self.direction
        self.head = (x + dx, y + dy)

    def get_state(self):
        """
        Returns an 11-dimensional state vector:
          1-3. Danger straight, right, left (1 or 0)
          4-7. Current direction (one-hot: [right, down, left, up])
          8-11. Food location relative to head (food left, food right, food up, food down)
        Note: Danger now includes collisions with walls, snake body, or obstacles.
        """
        head = self.snake[0]
        # Points one block ahead in each direction
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        # Determine danger based on current direction
        if self.direction == (1, 0):  # moving right
            danger_straight = self.is_collision((head[0] + 1, head[1]))
            danger_right    = self.is_collision((head[0], head[1] + 1))
            danger_left     = self.is_collision((head[0], head[1] - 1))
        elif self.direction == (-1, 0):  # moving left
            danger_straight = self.is_collision((head[0] - 1, head[1]))
            danger_right    = self.is_collision((head[0], head[1] - 1))
            danger_left     = self.is_collision((head[0], head[1] + 1))
        elif self.direction == (0, 1):  # moving down
            danger_straight = self.is_collision((head[0], head[1] + 1))
            danger_right    = self.is_collision((head[0] - 1, head[1]))
            danger_left     = self.is_collision((head[0] + 1, head[1]))
        elif self.direction == (0, -1):  # moving up
            danger_straight = self.is_collision((head[0], head[1] - 1))
            danger_right    = self.is_collision((head[0] + 1, head[1]))
            danger_left     = self.is_collision((head[0] - 1, head[1]))

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(self.direction == (1, 0)),   # moving right
            int(self.direction == (0, 1)),   # moving down
            int(self.direction == (-1, 0)),  # moving left
            int(self.direction == (0, -1)),  # moving up
            int(self.food[0] < head[0]),       # food left
            int(self.food[0] > head[0]),       # food right
            int(self.food[1] < head[1]),       # food up
            int(self.food[1] > head[1])        # food down
        ]
        return np.array(state, dtype=int)

# ============================
# DQN Agent & Helper Classes
# ============================

# --- Neural Network Model ---
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

# --- Trainer Class ---
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model   = model
        self.lr      = lr
        self.gamma   = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state      = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action     = torch.tensor(np.array(action), dtype=torch.long)
        reward     = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state      = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = (done,)

        # Predict Q values for current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# --- DQN Agent ---
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0      # controls randomness (exploration)
        self.gamma   = 0.9    # discount rate
        self.memory  = deque(maxlen=100_000)  # experience replay buffer
        self.batch_size = 1000

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_action(self, state):
        # Increase exploration when fewer games have been played.
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

# ============================
# Training Loop
# ============================
def train():
    pygame.init()
    agent = Agent()
    game  = SnakeGameAI()
    record = 0
    n_episodes = 1000  # Number of episodes to train

    for episode in range(n_episodes):
        game.reset()
        state_old = game.get_state()
        done = False

        while not done:
            # Get action from agent (based on current state)
            final_move = agent.get_action(state_old)
            # Execute move, get new state, reward, and game over flag
            reward, done, score = game.play_step(final_move)
            state_new = game.get_state()

            # Train on this move (short memory) and remember experience (for replay)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            state_old = state_new

        # Train long memory (experience replay) after each episode
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()  # Save model if a new record is reached

        print('Episode', episode, 'Score', score, 'Record:', record)
        agent.n_games += 1

    pygame.quit()

if __name__ == '__main__':
    train()
