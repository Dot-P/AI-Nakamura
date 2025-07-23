import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

def make_cnn(input_channels, architecture='small'):
    layers = []
    if architecture == 'small':
        # Conv32-64-64 + FC512
        layers += [nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU()]
        layers += [nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()]
        fc_input = 7 * 7 * 64
        fc_size = 512
    else:
        # Conv64-128-128 + FC1024
        layers += [nn.Conv2d(input_channels, 64, kernel_size=8, stride=4), nn.ReLU()]
        layers += [nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.ReLU()]
        fc_input = 7 * 7 * 128
        fc_size = 1024

    cnn = nn.Sequential(*layers)
    return cnn, fc_input, fc_size

class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, architecture='small'):
        super(QNetwork, self).__init__()
        self.cnn, fc_input, fc_size = make_cnn(input_channels, architecture)
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_size), nn.ReLU(),
            nn.Linear(fc_size, num_actions)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_channels, num_actions, gamma, lr, architecture, device):
        self.device = device
        self.policy_net = QNetwork(input_channels, num_actions, architecture).to(device)
        self.target_net = QNetwork(input_channels, num_actions, architecture).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.gamma = gamma
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0

    def select_action(self, state, eps_threshold):
        # state: numpy array (4,84,84)
        if np.random.rand() < eps_threshold:
            return np.random.randint(0, self.policy_net.fc[-1].out_features)
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()

    def optimize(self, memory, batch_size):
        if len(memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
