import pygame
import random
import numpy as np
import sys
from collections import deque

# ----------------------------
# Global Constants & Settings
# ----------------------------
BLOCK_SIZE   = 20                # Size of one grid block in pixels
SPEED        = 15                # Lower FPS to make transitions more interpretable
OBSTACLE_FREQ = SPEED * 3        # Spawn an obstacle every 3 seconds

# Colors (R,G,B)
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE  = (0, 0, 255)              # Obstacles color
BLACK = (0, 0, 0)

# ----------------------------
# Snake Game Environment Class
# ----------------------------
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # window width in pixels
        self.h = h  # window height in pixels
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Simulated Human Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Initialize the snake in the center of the grid
        self.direction = (1, 0)  # Initially moving right
        x = (self.w // BLOCK_SIZE) // 2
        y = (self.h // BLOCK_SIZE) // 2
        self.head = (x, y)
        self.snake = [self.head,
                      (self.head[0] - 1, self.head[1]),
                      (self.head[0] - 2, self.head[1])]
        self.score = 0
        self.food = None
        self.obstacles = set()  # Obstacles stored as (x, y) tuples
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        grid_width  = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        self.food = (x, y)
        # Ensure food is not on the snake or on an obstacle
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def spawn_obstacle(self):
        """Spawn a new obstacle at a random free location."""
        grid_width  = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        available = [(x, y) for x in range(grid_width) for y in range(grid_height)
                     if (x, y) not in self.snake and (x, y) != self.food and (x, y) not in self.obstacles]
        if available:
            pos = random.choice(available)
            self.obstacles.add(pos)

    def play_step(self, action):
        """
        Executes one step of the game.
          - action: a list of three integers:
            [1, 0, 0] => go straight
            [0, 1, 0] => turn right
            [0, 0, 1] => turn left
        Returns: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # Spawn obstacles every OBSTACLE_FREQ frames
        if self.frame_iteration % OBSTACLE_FREQ == 0:
            self.spawn_obstacle()

        # Handle Pygame events (e.g. window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Move the snake according to the action
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        # Check for collisions with walls, snake body, or obstacles
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

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
        # Draw snake
        for pt in self.snake:
            rect = pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, GREEN, rect)
        # Draw food (apple)
        food_rect = pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, RED, food_rect)
        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[0]*BLOCK_SIZE, obs[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, BLUE, obs_rect)
        # Draw score
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, (0, 0))
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
            new_dir = clock_wise[idx]  # Go straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # Turn right
        else:  # Turn left
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.head
        dx, dy = self.direction
        self.head = (x + dx, y + dy)

    def get_state(self):
        """
        Returns an 11-dimensional state vector:
          1-3. Danger straight, right, left (1 if collision is imminent, else 0)
          4-7. Current direction (one-hot: [right, down, left, up])
          8-11. Food location relative to head (food left, food right, food up, food down)
        Danger includes collisions with walls, the snake's body, or obstacles.
        """
        head = self.snake[0]
        if self.direction == (1, 0):
            danger_straight = self.is_collision((head[0] + 1, head[1]))
            danger_right    = self.is_collision((head[0], head[1] + 1))
            danger_left     = self.is_collision((head[0], head[1] - 1))
        elif self.direction == (-1, 0):
            danger_straight = self.is_collision((head[0] - 1, head[1]))
            danger_right    = self.is_collision((head[0], head[1] - 1))
            danger_left     = self.is_collision((head[0], head[1] + 1))
        elif self.direction == (0, 1):
            danger_straight = self.is_collision((head[0], head[1] + 1))
            danger_right    = self.is_collision((head[0] - 1, head[1]))
            danger_left     = self.is_collision((head[0] + 1, head[1]))
        elif self.direction == (0, -1):
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
            int(self.food[0] < head[0]),       # food is left
            int(self.food[0] > head[0]),       # food is right
            int(self.food[1] < head[1]),       # food is up
            int(self.food[1] > head[1])        # food is down
        ]
        return np.array(state, dtype=int)

# ----------------------------
# Human-Like Decision Function
# ----------------------------
def get_human_like_action(game):
    """
    Simulate an average human decision for playing Snake.
    The function computes a candidate action (based on Manhattan distance to the food)
    but with a chance to choose a random safe move, simulating human error.
    """
    error_probability = 0.2  # 20% chance to make an error (choose a random safe move)
    clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    current_direction = game.direction
    head = game.head
    food = game.food
    dx = food[0] - head[0]
    dy = food[1] - head[1]
    # Determine the candidate absolute direction (prefer the axis with greater distance)
    if abs(dx) >= abs(dy):
        candidate_abs_dir = (1, 0) if dx > 0 else (-1, 0)
    else:
        candidate_abs_dir = (0, 1) if dy > 0 else (0, -1)
    current_idx = clock_wise.index(current_direction)
    candidate_idx = clock_wise.index(candidate_abs_dir)
    if candidate_idx == current_idx:
        candidate_action = [1, 0, 0]
    elif candidate_idx == (current_idx + 1) % 4:
        candidate_action = [0, 1, 0]
    elif candidate_idx == (current_idx - 1) % 4:
        candidate_action = [0, 0, 1]
    else:
        candidate_action = [1, 0, 0]  # fallback
    # Compute safe moves (moves that don't lead to an immediate collision)
    safe_moves = []
    for action in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        idx = clock_wise.index(current_direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        if not game.is_collision(new_head):
            safe_moves.append(action)
    # With a chance to err, choose a random safe move (if available)
    if safe_moves and random.random() < error_probability:
        return random.choice(safe_moves)
    # Otherwise, if the candidate action is safe, return it; else choose a random safe move
    if candidate_action in safe_moves:
        return candidate_action
    elif safe_moves:
        return random.choice(safe_moves)
    else:
        return candidate_action

# ----------------------------
# Simulation Loop (Human-Like Play)
# ----------------------------
def simulate_human_game():
    pygame.init()
    game = SnakeGameAI()
    record = 0
    while True:
        game.reset()
        done = False
        while not done:
            action = get_human_like_action(game)
            reward, done, score = game.play_step(action)
        if score > record:
            record = score
        print("Human Simulation Score:", score, "Record:", record)

if __name__ == '__main__':
    simulate_human_game()
