import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

SEQ_LEN = 6
PREDICT_STEPS = 12   # how many future frames
IMG_SIZE = 128

# -----------------------
# ConvLSTM model (same as training)
# -----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv = self.conv(combined)
        i, f, o, g = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class RadarPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = ConvLSTMCell(1, 32)
        self.conv_out = nn.Conv2d(32, 1, 1)

    def forward(self, x, h_t=None, c_t=None):
        b, t, c, h, w = x.shape

        if h_t is None:
            h_t = torch.zeros(b, 32, h, w).to(DEVICE)
            c_t = torch.zeros(b, 32, h, w).to(DEVICE)

        for i in range(t):
            h_t, c_t = self.lstm(x[:, i], h_t, c_t)

        out = torch.sigmoid(self.conv_out(h_t))
        return out, h_t, c_t

# -----------------------
# Find latest radar folder
# -----------------------
def get_latest_folder():
    base = "./images"
    folders = [
        int(f)
        for f in os.listdir(base)
        if f.isdigit() and os.path.isdir(os.path.join(base, f))
    ]
    latest = str(max(folders))
    return os.path.join(base, latest)

# -----------------------
# Load last sequence
# -----------------------
def load_last_sequence(folder):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".png") and f.split(".")[0].isdigit()
    ]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    last_files = files[-SEQ_LEN:]

    imgs = []
    for f in last_files:
        img = Image.open(os.path.join(folder, f)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img) / 255.0
        imgs.append(arr)

    return np.array(imgs)


def frame_to_uint8(frame):
    frame = np.clip(frame, 0.0, 1.0).astype(np.float32)
    raw = (frame * 255.0).astype(np.uint8)

    # Stretch low-contrast predictions so saved PNGs are visually inspectable.
    lo = float(np.percentile(frame, 1))
    hi = float(np.percentile(frame, 99))

    if hi > lo + 1e-6:
        vis = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    else:
        max_val = float(frame.max())
        vis = (frame / max_val) if max_val > 1e-6 else np.zeros_like(frame)

    vis = (vis * 255.0).astype(np.uint8)
    return raw, vis

# -----------------------
# Load model
# -----------------------
model = RadarPredictor().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
print("Loaded trained model")

# -----------------------
# Get latest radar
# -----------------------
folder = get_latest_folder()
print("Using latest radar folder:", folder)

seq = load_last_sequence(folder)

frames = []
input_seq = torch.tensor(seq).float().unsqueeze(0).unsqueeze(2).to(DEVICE)
raw_frames = []
h_t, c_t = None, None

# -----------------------
# Predict future frames
# -----------------------
for step in range(PREDICT_STEPS):
    with torch.no_grad():
        if step == 0:
            pred, h_t, c_t = model(input_seq, h_t, c_t)
        else:
            pred, h_t, c_t = model(new_frame, h_t, c_t)

    frame = pred.cpu().squeeze().numpy()
    raw, vis = frame_to_uint8(frame)
    raw_frames.append(raw)
    frames.append(vis)

    # feed prediction back in
    new_frame = pred.unsqueeze(1)

    print(f"Predicted future frame {step+1}/{PREDICT_STEPS}")

# -----------------------
# Save frames
# -----------------------
os.makedirs("forecast", exist_ok=True)
os.makedirs("forecast/raw", exist_ok=True)

for i, f in enumerate(frames):
    Image.fromarray(f).save(f"forecast/frame_{i:02d}.png")
    Image.fromarray(raw_frames[i]).save(f"forecast/raw/frame_{i:02d}.png")

print("Saved visualization frames to ./forecast")
print("Saved raw frames to ./forecast/raw")

# -----------------------
# Create GIF animation
# -----------------------
gif_path = "forecast/forecast.gif"
imageio.mimsave(gif_path, frames, duration=0.4)

print("Saved forecast animation:", gif_path)
