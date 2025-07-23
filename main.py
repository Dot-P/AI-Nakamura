import os
import csv
import random
import argparse
from collections import deque, namedtuple
from itertools import product

import gym
import numpy as np
# Gym passive_env_checker may expect np.bool8
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Data structures ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --- Neural network ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Utility functions ---
def normalize_state(s, low, high):
    return 2.0 * (s - low) / (high - low) - 1.0

def select_action(state, policy_net, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return policy_net(st).argmax().item()

def optimize(buffer, policy_net, target_net, optimizer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None, None

    trans = buffer.sample(batch_size)
    states = torch.tensor(np.array(trans.state), dtype=torch.float32, device=device)
    actions = torch.tensor(trans.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(trans.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(trans.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(trans.done, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q)
    td_errors = (target_q - q_values).abs().detach().cpu().numpy().flatten()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), td_errors

# --- Experiment loop ---
def run_experiment(params, device, max_episodes):
    gamma, lr, hidden_sizes, batch_size, memory_size, seed = params

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make('MountainCar-v0')
    low, high = env.observation_space.low, env.observation_space.high
    n_actions = env.action_space.n

    policy_net = QNetwork(low.shape[0], hidden_sizes, n_actions).to(device)
    target_net = QNetwork(low.shape[0], hidden_sizes, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    buffer = ReplayBuffer(memory_size)
    eps_start, eps_end, eps_decay = 1.0, 0.01, 500.0
    steps_done = 0

    os.makedirs('logs', exist_ok=True)
    fname = (
        f"logs/g{gamma}_lr{lr}_h{'-'.join(map(str,hidden_sizes))}"
        f"_b{batch_size}_m{memory_size}_s{seed}.csv"
    )
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'cum_reward', 'steps_to_goal',
                         'epsilon', 'mean_td_error', 'mean_loss'])

        consec_success = deque(maxlen=10)
        episode = 0

        pbar = tqdm(desc=f"Run γ={gamma}, lr={lr}, h={hidden_sizes}, b={batch_size}, m={memory_size}, s={seed}",
                    unit="ep", leave=False)
        while True:
            episode += 1
            if episode > max_episodes:
                print(f"[Max episodes reached] γ={gamma}, lr={lr}, hidden={hidden_sizes}, "
                      f"batch={batch_size}, mem={memory_size}, seed={seed}, episodes={episode-1}")
                pbar.close()
                break

            pbar.update(1)
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            state = normalize_state(obs, low, high)

            cum_reward = 0
            losses, td_errors = [], []

            for t in range(1, 201):
                epsilon = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay)
                steps_done += 1

                action = select_action(state, policy_net, epsilon, n_actions, device)
                step_out = env.step(action)
                if len(step_out) == 5:
                    nxt_obs, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    nxt_obs, reward, done, _ = step_out

                next_state = normalize_state(nxt_obs, low, high)
                cum_reward += reward

                buffer.push(state, action, reward, next_state, done)
                result = optimize(buffer, policy_net, target_net, optimizer,
                                  batch_size, gamma, device)
                if result[0] is not None:
                    losses.append(result[0])
                    td_errors.extend(result[1].tolist())

                state = next_state
                if done or nxt_obs[0] >= env.goal_position:
                    success = nxt_obs[0] >= env.goal_position
                    consec_success.append(success)
                    steps_to_goal = t if success else 200
                    break

            if episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            writer.writerow([
                episode,
                cum_reward,
                steps_to_goal,
                epsilon,
                np.mean(td_errors) if td_errors else 0.0,
                np.mean(losses) if losses else 0.0
            ])

            if len(consec_success) == 10 and all(consec_success):
                print(f"[Solved] γ={gamma}, lr={lr}, hidden={hidden_sizes}, "
                      f"batch={batch_size}, mem={memory_size}, seed={seed}, episodes={episode}")
                pbar.close()
                break

        env.close()

# --- Argument parsing & entry point ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','gpu'], default='cpu',
                        help='Compute device: cpu or gpu')
    parser.add_argument('--max_episodes', type=int, default=1000,
                        help='Maximum episodes per experiment (default: 1000)')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("CUDA not available, using CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    gammas = [0.95, 0.99]
    lrs = [1e-3, 5e-4]
    hidden_structures = [[32, 32], [64, 64], [128, 64]]
    batch_sizes = [32, 64]
    memory_sizes = [int(1e5), int(2e5)]
    seeds = list(range(5))

    experiments = list(product(gammas, lrs, hidden_structures, batch_sizes, memory_sizes, seeds))
    for params in tqdm(experiments, desc="All experiments", unit="run"):
        run_experiment(params, device, args.max_episodes)

if __name__ == '__main__':
    main()
