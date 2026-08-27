# AGENTS.md

## Overview

Snake AI using a DQN (Deep Q-Network) agent trained via reinforcement learning.

## Quick Start

```bash
source venv/bin/activate
python agent.py
```

This launches training. It opens a Pygame window and plots scores via IPython/matplotlib.

## Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install torch pygame matplotlib numpy ipython
```

## Architecture

- `game.py` — Snake game engine (Pygame). `SnakeGameAI` class with `play_step(action)` interface.
- `agent.py` — DQN agent. Entry point: `train()` at bottom, run via `__main__`.
- `model.py` — `Linear_QNet` (11 inputs, 256 hidden, 3 outputs) and `QTrainer` (MSE loss, Adam optimizer).
- `helper.py` — Live score plotting (requires IPython `display`).
- `model/model.pth` — Saved model weights (auto-created on new record score).

## Key Constants

- `BLOCK_SIZE = 20` — grid unit size (pixels)
- `SPEED = 40` — FPS during gameplay
- State vector: 11 booleans (danger straight/right/left, direction, food relative position)
- Action space: 3 discrete actions — [straight, turn right, turn left]

## Gotchas

- `helper.py` uses `IPython.display` — plotting only works in IPython/Jupyter or an environment with a display backend. Will crash in headless/SSH sessions.
- `game.py` calls `pygame.init()` at import time — importing `game` without a display available will fail.
- No tests, no linting, no CI. Verify changes by running `python agent.py` and watching the score improve over ~50+ games.
- Model saves to `./model/model.pth` only when a new high score is reached.
