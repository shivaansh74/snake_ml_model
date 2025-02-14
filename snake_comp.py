import pygame, sys, random
from collections import deque

# --- Game Configuration ---
CELL_SIZE   = 20                      # Size of each cell (in pixels)
GRID_WIDTH  = 30                      # Number of cells horizontally
GRID_HEIGHT = 20                      # Number of cells vertically

WINDOW_WIDTH  = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
GRAY  = (40, 40, 40)
BLUE  = (0, 0, 255)   # Color for obstacles

# Directions (dx, dy)
UP    = (0, -1)
DOWN  = (0, 1)
LEFT  = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# --- Helper Functions ---

def get_random_food_position(snake, obstacles):
    """
    Return a random grid cell that is not occupied by the snake or obstacles.
    """
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake and pos not in obstacles:
            return pos

def get_random_obstacle_position(snake, food, obstacles):
    """
    Return a random grid cell that is not occupied by the snake, food, or any existing obstacles.
    """
    occupied = set(snake)
    occupied.add(food)
    occupied = occupied.union(obstacles)
    available = [(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x, y) not in occupied]
    if not available:
        return None
    return random.choice(available)

def bfs(start, goal, snake, obstacles, allow_tail=True):
    """
    Perform a breadth-first search (BFS) from start to goal.
    
    Snake’s body cells (except optionally the tail) and obstacles are treated as blocked.
    Returns a list of positions (excluding the start) leading to the goal,
    or None if no such path exists.
    """
    queue = deque()
    queue.append(start)
    came_from = {start: None}

    # Create a set of obstacles from the snake's body and the fixed obstacles.
    snake_obstacles = set(snake[:-1]) if allow_tail and len(snake) > 0 else set(snake)
    total_obstacles = obstacles.union(snake_obstacles)

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for d in DIRECTIONS:
            next_cell = (current[0] + d[0], current[1] + d[1])
            if (0 <= next_cell[0] < GRID_WIDTH and
                0 <= next_cell[1] < GRID_HEIGHT and
                next_cell not in total_obstacles and
                next_cell not in came_from):
                came_from[next_cell] = current
                queue.append(next_cell)

    if goal not in came_from:
        return None

    # Reconstruct path (excluding the start)
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def compute_next_direction(snake, food, obstacles):
    """
    Compute the next direction for the snake.
    
    The function uses BFS to find a path from the snake’s head to the food.
    If a path exists, the snake moves toward the first step.
    If not, the function searches for any safe move.
    """
    head = snake[0]
    path = bfs(head, food, snake, obstacles)
    if path and len(path) > 0:
        next_cell = path[0]
        dx = next_cell[0] - head[0]
        dy = next_cell[1] - head[1]
        return (dx, dy)
    else:
        # No path found; try any safe direction.
        for d in DIRECTIONS:
            next_cell = (head[0] + d[0], head[1] + d[1])
            if (0 <= next_cell[0] < GRID_WIDTH and
                0 <= next_cell[1] < GRID_HEIGHT and
                next_cell not in snake and
                next_cell not in obstacles):
                return d
        # No safe move exists.
        return None

# --- Main Game Function ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI Game with Obstacles")
    clock = pygame.time.Clock()

    # Initialize the snake, obstacles, and food.
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    obstacles = set()
    food = get_random_food_position(snake, obstacles)
    score = 0

    # Set up a timer event to spawn an obstacle every 2 seconds (2000 milliseconds).
    SPAWN_OBSTACLE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_OBSTACLE_EVENT, 2000)

    game_over = False

    while not game_over:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == SPAWN_OBSTACLE_EVENT:
                pos = get_random_obstacle_position(snake, food, obstacles)
                if pos:
                    obstacles.add(pos)

        # --- AI Decision and Movement ---
        direction = compute_next_direction(snake, food, obstacles)
        if direction is None:
            game_over = True
            print("Game Over! Score:", score)
            continue

        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        # Check collisions with walls, the snake itself, or obstacles.
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
            new_head in snake or new_head in obstacles):
            game_over = True
            print("Game Over! Score:", score)
            continue

        snake.insert(0, new_head)

        # If the snake eats the food, increase score and spawn new food.
        if new_head == food:
            score += 1
            food = get_random_food_position(snake, obstacles)
        else:
            snake.pop()  # Remove the tail if no food is eaten.

        # --- Drawing ---
        screen.fill(BLACK)

        # (Optional) Draw grid lines for reference.
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (WINDOW_WIDTH, y))

        # Draw the snake.
        for cell in snake:
            rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

        # Draw the food.
        food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, food_rect)

        # Draw the obstacles.
        for cell in obstacles:
            rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLUE, rect)

        pygame.display.flip()
        clock.tick(10)  # Adjust game speed (frames per second).

    pygame.quit()

if __name__ == "__main__":
    main()
