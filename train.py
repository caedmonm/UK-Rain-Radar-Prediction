import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

SEQ_LEN = 6
IMG_SIZE = 128

# -----------------------
# Load one folder sequence
# -----------------------
def load_folder_sequence(folder):
    files = []

    for f in os.listdir(folder):
        if not f.lower().endswith(".png"):
            continue
        name = f.split(".")[0]
        if not name.isdigit():
            continue
        files.append(f)

    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    imgs = []
    for f in files:
        img = Image.open(os.path.join(folder, f)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img) / 255.0
        imgs.append(arr)

    return np.array(imgs)

# -----------------------
# Load all folders safely
# -----------------------
def load_all_sequences(base="./images"):
    all_X = []
    all_Y = []

    folders = sorted(os.listdir(base))

    for folder in folders:
        path = os.path.join(base, folder)
        if not os.path.isdir(path):
            continue

        print("Loading folder:", folder)
        images = load_folder_sequence(path)

        if len(images) <= SEQ_LEN:
            print("Skipping (too few frames)")
            continue

        # create sequences ONLY within folder
        for i in range(len(images) - SEQ_LEN):
            seq = images[i:i+SEQ_LEN]
            target = images[i+SEQ_LEN]

            all_X.append(seq)
            all_Y.append(target)

    return np.array(all_X), np.array(all_Y)

X, Y = load_all_sequences()

print("Total training samples:", len(X))

X = torch.tensor(X).float().unsqueeze(2)
Y = torch.tensor(Y).float().unsqueeze(1)

# -----------------------
# ConvLSTM
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
        self.hidden_channels = hidden_channels

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

    def forward(self, x):
        b, t, c, h, w = x.shape
        h_t = torch.zeros(b, 32, h, w).to(DEVICE)
        c_t = torch.zeros(b, 32, h, w).to(DEVICE)

        for i in range(t):
            h_t, c_t = self.lstm(x[:, i], h_t, c_t)

        out = self.conv_out(h_t)
        return torch.sigmoid(out)

model = RadarPredictor().to(DEVICE)

# load existing weights if present
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    print("Loaded existing model weights")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X = X.to(DEVICE)
Y = Y.to(DEVICE)

# -----------------------
# Train
# -----------------------
best_loss = float("inf")
EPOCHS = 10

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} loss:", loss_val)

    if loss_val < best_loss:
        best_loss = loss_val
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Saved best model")
