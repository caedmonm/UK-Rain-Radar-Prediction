# UK Rain Radar Prediction

Predicts future UK rainfall radar frames using a ConvLSTM neural network trained on Met Office radar imagery.

Takes the last 90 minutes of radar data (6 frames at 15-minute intervals) and autoregressively predicts the next 3 hours (12 frames).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy requests Pillow imageio
```

## Usage

### 1. Download radar data

```bash
python3 scrape.py
```

Downloads the last 48 hours of UK rainfall radar images from the Met Office API into `images/{N}/`, where `{N}` auto-increments with each run.

### 2. Train the model

```bash
python3 train.py
```

Trains the ConvLSTM on all collected image sequences. Saves the best weights to `best_model.pth`. Resumes from existing weights if present.

### 3. Generate a forecast

```bash
python3 forecast.py
```

Loads the trained model, takes the last 6 frames from the most recent data folder, and predicts 12 future frames. Outputs individual PNGs and an animated GIF to `forecast/`.

## Model

- **Architecture:** ConvLSTMCell (1 → 32 hidden channels, 3×3 kernel) wrapped in a RadarPredictor with a 1×1 conv output layer
- **Input:** 6-frame sequences of 128×128 grayscale images
- **Training:** MSE loss, Adam optimizer (lr=0.001), 10 epochs
- **Prediction:** Autoregressive — each predicted frame is fed back as input for the next step
