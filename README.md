# 🐍 Snake Game AI Comparison

## Project Overview

This project implements the classic Snake game with various artificial intelligence control methods, enabling a comprehensive comparison between different approaches to automated gameplay. The implementation showcases the strengths and weaknesses of multiple AI techniques in solving this classic problem.

## 🧠 AI Approaches Implemented

### 1. Reinforcement Learning (Double DQN)
A machine learning approach where the AI learns optimal strategies through trial and error. The implementation uses:
- **Double Deep Q-Network** architecture to reduce overestimation bias
- **Experience replay** to efficiently use past experiences
- **Target networks** to stabilize training
- **Epsilon-greedy exploration** strategy

### 2. Deterministic Pathfinding (BFS)
A classic algorithmic solution that:
- Uses **Breadth-First Search** to find the shortest path to food
- Implements **intelligent fallback strategies** when direct paths are blocked
- Efficiently navigates around obstacles and the snake's own body

### 3. Heuristic-Based Control
A rule-based approach using predefined decision logic:
- Makes decisions based on relative positions of food, obstacles, and snake
- Demonstrates how simple rules can lead to reasonably good performance
- Serves as a baseline for comparison with more complex approaches

### 4. Human-Like Simulation
An AI that mimics human gameplay, including:
- **Intentional error probability** to simulate human mistakes
- Basic pathfinding with occasional suboptimal moves
- Decision-making based on Manhattan distance calculations

## 📁 Project Structure

### Source Files

- **ml_model_optimal.py** - Snake game with Double DQN implementation and obstacle avoidance
  - Features a complete reinforcement learning solution with target networks
  - Implements dynamic obstacle generation for increased difficulty
  - Provides visual feedback on the learning process

- **snake_comp.py** - Snake game using BFS pathfinding algorithm
  - Demonstrates efficient pathfinding in a dynamic environment
  - Shows how to implement graph search algorithms in the context of games
  - Includes obstacle handling and fallback strategies

- **ml-vs-hardcode.py** - Comparison between ML and hardcoded approaches
  - Directly compares reinforcement learning vs. heuristic methods
  - Allows switching between control methods for side-by-side comparison
  - Shows the strengths and weaknesses of each approach

- **human_simulated_game.py** - Snake game with human-like AI
  - Models human error probabilities and decision-making
  - Demonstrates a more realistic gameplay pattern
  - Useful for benchmarking other AI approaches against human-like play

## ⚙️ Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/snake-game-ai-comparison.git
```

2. Navigate to the project directory:
```
cd snake-game-ai-comparison
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the desired game script:
```
python ml_model_optimal.py
```
or
```
python snake_comp.py
```
or
```
python ml-vs-hardcode.py
```
or
```
python human_simulated_game.py
```

## 📝 Usage

- Modify the parameters in the respective script files to experiment with different AI settings.
- Observe the performance of each AI approach and compare their effectiveness.
- Use the visual feedback to understand the decision-making process of each AI.

## 📊 Results

- The results of the AI comparisons can be found in the `results` directory.
- Detailed performance metrics and analysis are provided for each AI approach.
- Graphs and charts illustrate the strengths and weaknesses of each method.

## 🤝 Contributing

- Contributions are welcome! Please fork the repository and submit a pull request.
- For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

- This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📧 Contact

- For any questions or inquiries, please contact [dhingrashivaansh@gmail.com](mailto:dhingrashivaansh@gmail.com).

