import os

import numpy as np  # needed to stack the batch of states/actions into arrays


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)   # new hidden layer
        self.linear3 = nn.Linear(hidden_size2, output_size)    # output layer

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))   # apply ReLU after the new layer
        x = self.linear3(x)           # no activation on output (for regression)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # CAUSE (stability): MSE squared-ly penalizes large TD errors. With
        # rewards of +10/-10 the early batches produce huge error values that
        # dominate the gradient and destabilize learning.
        # WHY: SmoothL1 (Huber) behaves like MSE for small errors but becomes
        # linear for large ones, capping the gradient from outlier samples.
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        # CAUSE (efficiency): tensors are built in one pass on the model's
        # device (CPU or GPU) instead of relying on implicit per-element moves.
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.model.linear1.weight.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.model.linear1.weight.device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=self.model.linear1.weight.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=self.model.linear1.weight.device)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x) -- append one dimension at the beginning
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. predicted Q values with current state
        prediction = self.model(state)

        # 2. Q_new = reward + gamma * max(next_predicted Q value)
        # only do this if not done -> otherwise take just the reward
        #
        # CAUSE (bug): the old code did `target = prediction.clone()` and then
        # assigned Q-values computed from `self.model(next_state[idx])` INTO
        # that clone. `clone()` keeps the autograd graph, so gradients flowed
        # through the target as well (double backprop through the next-state
        # network) — this makes the bootstrapping biased/unstable and can even
        # raise autograd errors with in-place assignment.
        # WHY: targets must be treated as constants (like in standard DQN /
        # target networks), so we compute them under torch.no_grad().
        with torch.no_grad():
            target = prediction.clone()
            # CAUSE (efficiency): the old code looped over the batch in Python
            # and called self.model() once PER SAMPLE, i.e. up to 1000 tiny
            # forward passes per training step.
            # WHY: one batched forward pass on next_state gives all next-Q
            # values in a single (vectorized) call — orders of magnitude less
            # overhead and it also uses the GPU efficiently.
            next_q = self.model(next_state).max(dim=1).values
            q_new = reward + self.gamma * next_q * (1 - torch.tensor(done, dtype=torch.float, device=reward.device))
            # scatter the Q_new value into the column of the action taken
            target[torch.arange(len(done)), torch.argmax(action, dim=1)] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
