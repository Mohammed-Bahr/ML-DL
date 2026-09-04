import random
from collections import deque

import numpy as np
import torch

from game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from helper import plot
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000  # this determines how many states are stored in memory if it exceeds it the oldest states will be discarded removed
BATCH_SIZE = 1000  # this determines how many states are used in each training batch
LR = 0.0001  # this determines the learning rate of the optimizer

# CAUSE (efficiency): the previous code always ran on CPU even when a GPU is
# available, wasting the big speedup CUDA gives for the per-step network calls.
# WHY: torch can transparently pick the best device; on machines without a GPU
# this falls back to CPU so behaviour is unchanged.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # randomness control (probability of a random move)
        self.decay = 0.995  # to decay epsilon over time
        # CAUSE (bug): there was no epsilon floor. With multiplicative decay the
        # probability of exploring decays to ~0, so the agent can lock into a
        # suboptimal loop it foundecayd early and never try better paths again.
        # WHY: a small floor keeps a minimum amount of exploration forever.
        self.epsilon_min = 0.02
        self.gamma = 0.9  # discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)  # pops left automatically
        self.model = Linear_QNet(11, 256, 256, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        # CAUSE (bug): the offsets were hardcoded as literal `20`. If BLOCK_SIZE
        # in game.py is ever changed the state silently becomes wrong (danger
        # flags computed for a different grid than the game actually uses).
        # WHY: deriving it from BLOCK_SIZE keeps state and game grid in sync.
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # pops left if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # trade-off between exploration / exploitation
        # self.epsilon = 80 - self.n_games
        # CAUSE (bug): the old check was `random.randint(0, 200) < self.epsilon`
        # with epsilon in (0, 1]. randint only returns 0..200, so a random move
        # happened ONLY when the roll was exactly 0 (~0.5% of the time) — the
        # agent barely explored from the start and learned very slowly.
        # WHY: `random.random()` is uniform in [0, 1), so the comparison now
        # gives the intended "explore with probability epsilon" behaviour.
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            with torch.no_grad():
                prediction = self.model(state0)  # executes the forward function
            # CAUSE (efficiency): inference doesn't need gradients; wrapping it
            # in no_grad avoids building an autograd graph every step, saving
            # memory and CPU/GPU time.
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []  # keep track of the scores (for plotting)
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old / current state
        state_old = agent.get_state(game)

        # get move based on the current state
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (only one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember (store in the deque)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (replay memory / experience replay),
            # trains again on all previous moves and games
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
