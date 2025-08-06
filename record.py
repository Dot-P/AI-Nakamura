import os
import random  # 追加: 評価時のε-greedy用
import argparse
import gym
import torch

import numpy as np
# --- ここを追加 ---
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import imageio  # pip install imageio imageio-ffmpeg
from model import QNetwork

# --- Utility ---
def normalize_state(s, low, high):
    return 2.0 * (s - low) / (high - low) - 1.0

# --- 動画記録関数（ε-greedy 評価対応） ---
def record_video_manual(
    model_path,
    hidden_sizes,
    video_path,
    seed=0,
    device='cpu',
    fps=30,
    epsilon=0.,  # 評価時のランダム行動率
):
    # 環境セットアップ
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env.reset(seed=seed)

    low, high = env.observation_space.low, env.observation_space.high
    n_actions = env.action_space.n
    input_dim = low.shape[0]

    # モデル構築＆読み込み
    policy_net = QNetwork(input_dim, hidden_sizes, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    frames = []
    obs = env.reset()[0]
    frames.append(env.render())
    state = normalize_state(obs, low, high)

    for _ in range(1000):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            # ε-greedy によるアクション選択
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                action = policy_net(st).argmax().item()

        out = env.step(action)
        if len(out) == 5:
            nxt_obs, reward, terminated, truncated, _ = out
            done = terminated or truncated
        else:
            nxt_obs, reward, done, _ = out

        state = normalize_state(nxt_obs, low, high)
        frames.append(env.render())
        if done:
            break

    env.close()

    # 動画書き出し (.mp4)
    os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved video to {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   required=True)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', required=True)
    parser.add_argument('--video_path',    required=True, help="例: output.mp4")
    parser.add_argument('--device',        choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--seed',          type=int, default=0)
    parser.add_argument('--fps',           type=int, default=30)
    parser.add_argument('--epsilon',       type=float, default=0.05,
                        help='評価時のε-greedy割合（0.0〜1.0）')
    args = parser.parse_args()

    record_video_manual(
        args.model_path,
        args.hidden_sizes,
        args.video_path,
        seed=args.seed,
        device=args.device,
        fps=args.fps,
        epsilon=args.epsilon,
    )