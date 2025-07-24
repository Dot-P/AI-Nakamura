import os
import argparse
import gym
import torch
import numpy as np
from model import QNetwork

# --- Utility ---
def normalize_state(s, low, high):
    return 2.0 * (s - low) / (high - low) - 1.0

def record_video(model_path, hidden_sizes, video_dir, seed=0, device='cpu'):
    # 環境セットアップ
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_dir=video_dir, name_prefix='dqn_demo')
    env.reset(seed=seed)

    low, high = env.observation_space.low, env.observation_space.high
    n_actions = env.action_space.n
    input_dim = low.shape[0]

    # モデル構築＆読み込み
    policy_net = QNetwork(input_dim, hidden_sizes, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    # 初期状態
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    state = normalize_state(obs, low, high)

    for _ in range(200):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(st).argmax().item()

        step_out = env.step(action)
        if len(step_out) == 5:
            nxt_obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            nxt_obs, reward, done, _ = step_out

        state = normalize_state(nxt_obs, low, high)
        if done:
            break

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.pt)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=True, help='Hidden layer sizes (e.g. 64 64)')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory to save the video')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Computation device')
    parser.add_argument('--seed', type=int, default=0, help='Environment seed')
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    record_video(args.model_path, args.hidden_sizes, args.video_dir, args.seed, args.device)
