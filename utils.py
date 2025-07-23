import random
import numpy as np
torch = __import__('torch')
from collections import deque
import cv2

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DiscreteActionWrapper:
    """Convert continuous CarRacing action space to discrete actions."""
    ACTIONS = [
        np.array([0.0, 1.0, 0.0]),  # forward
        np.array([-1.0, 1.0, 0.0]), # left + forward
        np.array([ 1.0, 1.0, 0.0]), # right + forward
        np.array([0.0, 0.0, 0.8]),  # brake
        np.array([0.0, 0.0, 0.0]),  # no-op
    ]
    def __init__(self, env):
        self.env = env
        self.action_space = type('A', (), {'n': len(self.ACTIONS)})

    def reset(self):
        return self.env.reset()

    def step(self, action_idx: int):
        cont_action = self.ACTIONS[action_idx]
        return self.env.step(cont_action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

# preprocessing: grayscale, resize, normalize, stack
from collections import deque
def preprocess_frame(frame):
    # RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # normalize
    norm = resized.astype(np.float32) / 255.0
    return norm

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def step(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)