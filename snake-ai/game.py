import random
from collections import namedtuple
from enum import Enum

import pygame


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # CAUSE (bug): `pygame.init()` used to run at module import time, so
        # merely `import game` in a headless/SSH session crashed before any
        # code could run (pygame fails without a display for set_mode).
        # WHY: initializing here means the import is side-effect free and only
        # actually creating a game window requires a display.
        pygame.init()
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        # CAUSE (bug): the font was created at module import time too, which
        # requires pygame to be initialized first.
        # WHY: creating it here (after pygame.init()) keeps the same font
        # fallback logic but at a safe point in time.
        import os
        if os.path.exists("arial.ttf"):
            self.font = pygame.font.Font("arial.ttf", 25)
        else:
            self.font = pygame.font.Font(None, 30)
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # CAUSE (bug + efficiency): the old version used recursion — for every
        # failed placement it called itself again. With a long snake this can
        # recurse hundreds of times (slow, random retries) and with a nearly
        # full board it can even hit Python's recursion limit and crash.
        # WHY: build the set of free cells once and pick uniformly from it —
        # always O(free cells), no retries, no recursion, and a truly uniform
        # distribution over free positions.
        free_cells = []
        occupied = set(self.snake)
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                if (x, y) not in occupied:
                    free_cells.append((x, y))
        if not free_cells:
            return  # board is full (perfect game) — no space left for food
        x, y = random.choice(free_cells)
        self.food = Point(x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        # CAUSE (efficiency): `pt in self.snake[1:]` slices the whole list
        # (allocating a new list of up to len(snake)-1 Points) on EVERY call,
        # and this runs 4x per step from get_state plus once in play_step.
        # WHY: comparing indices directly avoids the list copy with identical
        # semantics — the head itself is at index 0.
        for i in range(1, len(self.snake)):
            if self.snake[i] == pt:
                return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right turn, left turn]
        # CAUSE (efficiency): the old code used np.array_equal(action, [...]),
        # which converts Python lists to numpy arrays (allocation + comparison
        # overhead) for EVERY move. `action` here is just a plain 3-element
        # list, so direct comparison is equivalent and much cheaper.
        # WHY: this function runs on every single game frame; removing numpy
        # from the hot path speeds up the whole training loop.
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clockwise[idx]  # no change
        elif action == [0, 1, 0]:
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]  # right turn -> clockwise
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]  # left turn -> counter-clockwise

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
WHITE = (255, 255, 255)
