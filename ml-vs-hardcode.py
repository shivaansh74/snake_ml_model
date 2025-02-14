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
BLOCK_SIZE = 20      # Size of one grid block in pixels
SPEED = 40           # Game speed (frames per second)

# Colors (R, G, B)
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)

# ----------------------------
# Choose which control mode to run:
#   "ml"         : use the trained DQN model
#   "hardcoded"  : use the heuristic (hard-coded) AI
# ----------------------------
CONTROL_MODE = "ml"  # change to "hardcoded" to compare with the hard-coded AI

# ============================
# Snake Game Environment Class
# ============================
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # window width in pixels
        self.h = h  # window height in pixels
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Comparison')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Start with the snake in the center of the grid
        self.direction = (1, 0)  # initially moving right
        x = (self.w // BLOCK_SIZE) // 2
        y = (self.h // BLOCK_SIZE) // 2
        self.head = (x, y)
        # The snake is a list of grid positions (head is first)
        self.snake = [self.head,
                      (self.head[0]-1, self.head[1]),
                      (self.head[0]-2, self.head[1])]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        grid_width = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        self.food = (x, y)
        # Ensure food is not placed on the snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Executes one step of the game.
          - action: list of three integers [1,0,0] (go straight), [0,1,0] (turn right), or [0,0,1] (turn left)
        Returns: (reward, game_over, score)
        """
        self.frame_iteration += 1
        # 1. Check for pygame events (e.g. closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Move the snake based on action
        self._move(action)  # update self.head
        self.snake.insert(0, self.head)

        # 3. Check if game over
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
            self.snake.pop()  # remove tail if no food eaten

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Return True if point 'pt' collides with wall or snake body."""
        if pt is None:
            pt = self.head
        x, y = pt
        grid_width = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # Draw snake
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, GREEN, rect)
        # Draw food
        rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, RED, rect)
        # Draw score text
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Update the snake's direction and head based on the action.
          - action: [straight, right, left]
        The action is interpreted relative to the current direction.
        """
        # Define clockwise order: right, down, left, up
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]          # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4             # right turn
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1] turn left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

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
        """
        head = self.snake[0]
        # Define points one block in each direction from the head:
        point_l = (head[0]-1, head[1])
        point_r = (head[0]+1, head[1])
        point_u = (head[0], head[1]-1)
        point_d = (head[0], head[1]+1)

        # Determine danger based on current direction:
        if self.direction == (1, 0):  # moving right
            danger_straight = self.is_collision((head[0]+1, head[1]))
            danger_right = self.is_collision((head[0], head[1]+1))
            danger_left = self.is_collision((head[0], head[1]-1))
        elif self.direction == (-1, 0):  # moving left
            danger_straight = self.is_collision((head[0]-1, head[1]))
            danger_right = self.is_collision((head[0], head[1]-1))
            danger_left = self.is_collision((head[0], head[1]+1))
        elif self.direction == (0, 1):  # moving down
            danger_straight = self.is_collision((head[0], head[1]+1))
            danger_right = self.is_collision((head[0]-1, head[1]))
            danger_left = self.is_collision((head[0]+1, head[1]))
        elif self.direction == (0, -1):  # moving up
            danger_straight = self.is_collision((head[0], head[1]-1))
            danger_right = self.is_collision((head[0]+1, head[1]))
            danger_left = self.is_collision((head[0]-1, head[1]))

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(self.direction == (1, 0)),   # moving right
            int(self.direction == (0, 1)),   # moving down
            int(self.direction == (-1, 0)),  # moving left
            int(self.direction == (0, -1)),  # moving up

            int(self.food[0] < head[0]),     # food left
            int(self.food[0] > head[0]),     # food right
            int(self.food[1] < head[1]),     # food up
            int(self.food[1] > head[1])      # food down
        ]
        return np.array(state, dtype=int)

# ============================
# DQN Agent and Neural Network
# ============================
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

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

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

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0      # controls exploration
        self.gamma = 0.9      # discount rate
        self.memory = deque(maxlen=100_000)
        self.batch_size = 1000

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_action(self, state):
        # For inference, we want a greedy policy (set epsilon to 0)
        self.epsilon = 0  
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

    # The following methods were used during training:
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
# Hard-Coded (Heuristic) AI
# ============================
def get_hardcoded_action(game):
    """
    A simple heuristic:
      - Compute the relative position of food versus the snake's head.
      - Choose the direction (right, left, up, down) that moves closer.
      - Convert the chosen absolute direction into a relative action
        (based on the snake's current direction: [straight, right, left]).
    """
    head = game.snake[0]
    food = game.food
    current_direction = game.direction

    dx = food[0] - head[0]
    dy = food[1] - head[1]

    # Define absolute directions
    # right, down, left, up (clockwise order)
    clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    current_idx = clock_wise.index(current_direction)

    # Determine desired absolute direction based on food location:
    candidate = None
    if abs(dx) > abs(dy):
        candidate = (1, 0) if dx > 0 else (-1, 0)
    else:
        candidate = (0, 1) if dy > 0 else (0, -1)

    candidate_idx = clock_wise.index(candidate)

    # Determine relative action:
    if candidate_idx == current_idx:
        return [1, 0, 0]  # go straight
    elif candidate_idx == (current_idx + 1) % 4:
        return [0, 1, 0]  # turn right
    elif candidate_idx == (current_idx - 1) % 4:
        return [0, 0, 1]  # turn left
    else:
        # If candidate is not immediately left/right, try going straight if safe
        next_pos = (head[0] + current_direction[0], head[1] + current_direction[1])
        if not game.is_collision(next_pos):
            return [1, 0, 0]
        else:
            return [0, 1, 0]

# ============================
# Play Mode (Inference)
# ============================
def play(control_mode="ml"):
    pygame.init()
    game = SnakeGameAI()
    agent = Agent()
    record = 0

    # If using the ML agent, try to load the trained model.
    if control_mode == "ml":
        try:
            agent.model.load_state_dict(torch.load("model.pth"))
            agent.model.eval()
            print("Loaded trained model.")
        except Exception as e:
            print("Could not load trained model. Error:", e)
            return

    while True:
        game.reset()
        state_old = game.get_state()
        done = False
        while not done:
            if control_mode == "ml":
                action = agent.get_action(state_old)
            else:
                action = get_hardcoded_action(game)

            reward, done, score = game.play_step(action)
            state_old = game.get_state()

        if score > record:
            record = score
        print("Score:", score, "Record:", record)

# ============================
# Main Entry Point
# ============================
if __name__ == '__main__':
    # Change the argument to "hardcoded" to use the heuristic agent.
    play(CONTROL_MODE)
