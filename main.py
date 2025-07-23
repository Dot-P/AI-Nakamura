import gym
import argparse
import os
import csv
import itertools
import numpy as np
import torch
from utils import set_seed, ReplayBuffer, DiscreteActionWrapper, FrameStack
from agent import DQNAgent

def train_one(config, seed, output_dir):
    # Unpack config\ n    gamma, lr, arch, batch_size, mem_size = config
    set_seed(seed)

    env = gym.make('CarRacing-v2')
    env = DiscreteActionWrapper(env)
    stacker = FrameStack(k=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DQNAgent(input_channels=4, num_actions=env.action_space.n,
                     gamma=gamma, lr=lr, architecture=arch, device=device)
    buffer = ReplayBuffer(capacity=mem_size)

    # CSV logging
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir,
        f"exp_g{gamma}_lr{lr}_a{arch}_b{batch_size}_m{mem_size}_s{seed}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'cum_reward', 'avg_score900_ep', 'avg100', 'epsilon', 'td_error'])

        # Training loop
        eps_start, eps_end, eps_decay = 1.0, 0.01, 100000
        target_update = 1000
        max_episodes = 10000
        consecutive = deque(maxlen=100)

        for ep in range(1, max_episodes+1):
            raw = env.reset()
            state = stacker.reset(raw)
            cum_reward = 0
            losses = []

            done = False
            while not done:
                eps = eps_end + (eps_start - eps_end) * np.exp(-1. * agent.steps_done / eps_decay)
                action = agent.select_action(state, eps)
                nxt_raw, reward, done, _ = env.step(action)
                next_state = stacker.step(nxt_raw)
                buffer.push(state, action, reward, next_state, done)
                loss = agent.optimize(buffer, batch_size)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                cum_reward += reward
                agent.steps_done += 1

                if agent.steps_done % target_update == 0:
                    agent.update_target()

            # metrics
            consecutive.append(cum_reward)
            avg100 = np.mean(consecutive)
            first900 = (ep if avg100 >= 900 and len(consecutive) == 100 else '')
            td_err = np.mean(losses) if losses else ''
            writer.writerow([ep, cum_reward, first900, avg100, eps, td_err])

            # early stop
            if len(consecutive) == 100 and avg100 >= 900:
                break

    env.close()

if __name__ == '__main__':
    # grid search parameters
    gammas = [0.95, 0.99]
    lrs = [1e-4, 5e-5]
    archs = ['small', 'large']
    b_sizes = [32, 64]
    mems = [100000, 200000]
    seeds = [0, 1, 2, 3, 4]
    output_dir = 'results'

    for config in itertools.product(gammas, lrs, ['small', 'large'], b_sizes, mems):
        for seed in seeds:
            train_one(config, seed, output_dir)