# DQN for MountainCar-v0

This project implements a Deep Q-Network (DQN) agent for solving the `MountainCar-v0` environment from OpenAI Gym using PyTorch.  
It includes training with configurable hyperparameters and recording final episodes as video using a separately saved model.

---

## 📁 Project Structure

```

├── train.py              # Training script (called via main.py)
├── record_video.py       # Script to load a saved model and record a video
├── model.py              # QNetwork model definition
├── logs/                 # Training logs (CSV)
├── models/               # Trained model checkpoints (per experiment)
├── demo_videos/          # Output directory for recorded videos
└── README.md             # This file

```

---

## 🚀 How to Train the Model

Use the following command to start training. It will run all experiment combinations defined in `train.py`:

```bash
uv run train.py
```

* The training logs will be saved in the `logs/` directory.
* Trained models will be saved in the `models/` directory under a subfolder named by hyperparameters.

---

## 🎬 How to Record a Video from a Trained Model

To evaluate a trained agent and save a video of its behavior:

```bash
uv run record.py --model_path models/g0.99_lr0.001_h64-64_b64_m200000_s0/policy_net.pt --hidden_sizes 64 64  --video_path ./video/demo.mp4
```

* `--model_path`: Path to the trained `.pt` file saved from training.
* `--hidden_sizes`: Hidden layer sizes used in the network (must match the trained model).
* `--video_dir`: Directory to save the output video (`.mp4`).
* `--device`: `cpu` or `cuda` depending on your environment.

The output video will be saved inside the specified directory with the prefix `dqn_demo`.

---

## 📝 Notes

* Ensure `ffmpeg` is installed for video recording to work (required by Gym's `RecordVideo` wrapper).
* If using GPU, make sure CUDA is available via `torch.cuda.is_available()`.

---

## 📦 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.12
* Gym ≥ 0.26
* tqdm
* numpy

Install dependencies:

```bash
pip install torch gym tqdm numpy
```

---

