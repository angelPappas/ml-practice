# Connect 4 Reinforcement Learning

A Python implementation of the classic Connect 4 game with reinforcement learning (RL) agents, including neural network-based models for training and evaluation.

## Features

- **Game Logic**: Full Connect 4 game implementation with win detection, valid moves, and board management.
- **Agents**: Multiple agent types, including random, heuristic, minimax, and neural network-based agents.
- **Training**: Support for training RL agents using neural networks (e.g., policy gradients).
- **Evaluation**: Custom evaluation functions to run episodes and collect rewards.
- **Visualization**: Render games in terminal or Jupyter notebooks with animations.
- **Gym Compatibility**: Integration with OpenAI Gym for standardized RL environments.

## Installation

1. **Prerequisites**: Python 3.8+ and `uv` (for dependency management).
2. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd connect_4_RL
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt  # If you export from pyproject.toml
   ```

## Usage

### Running the Game
- **Have two agents play against each other and watch the match**:
 ```python
 from agents.env import ConnectXEnv

env = ConnectXEnv()

agents = ["minimax", "random"]

env.run(agents=agents)

env.render()
 ```
- **One can also play Connect4 with an opponent in real time**, either in terminal by:

   `python game/game/play_terminal.py` , or with pygame

### Training an Agent
```python
from models import build_model, load_model
from game.game import ROWS, COLS, ConnectXEnv
from agents.env import BUILTIN_AGENTS, make_ppo_agent

agent2 = "random"

# Option 1: Train only against random agent
model = build_model(agent2=agent2, verbose=0)
model.learn(total_timesteps=60000)
# model.save("ppo_connect4")

# Option 2: Train against random and then against heuristic agents.
# model = build_model(agent2="random", verbose=0)
# model.learn(total_timesteps=200_000)

# # Hot-swap the environment, keep the weights
# model.set_env(ConnectFourGym(agent2="heuristic"))
# model.learn(total_timesteps=500_000, reset_num_timesteps=False)
# model.save("ppo_connect4")

agent1 = make_ppo_agent(model)

# # Evaluate
env = ConnectXEnv()
env.run_n_times([agent1, "random"], num_games=100)
```

### Project Structure
```
connect_4_RL/
├── agents/              # Agent implementations (random, heuristic, neural, etc.)
├── game/                # Core game logic (Connect4 class, utilities)
├── models.py            # Neural network models and training code
├── evaluation.py        # Custom evaluate and make functions
├── pyproject.toml       # Project configuration and dependencies
└── README.md            # This file
```

## Acknowledgments

- Inspired by [Intro to Game AI and Reinforcement Learning from Kaggle. ](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning)
