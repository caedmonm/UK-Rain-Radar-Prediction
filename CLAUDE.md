# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UK rainfall radar prediction system using ConvLSTM neural networks. Downloads Met Office radar imagery, trains a ConvLSTM model on 6-frame sequences (90 minutes at 15-min intervals), and autoregressively predicts the next 12 frames (3 hours).

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Download latest 48 hours of radar data into images/{N}/
python3 scrape.py

# Train the ConvLSTM model (saves best weights to best_model.pth)
python3 train.py

# Generate 12-frame forecast from latest data (outputs to forecast/)
python3 forecast.py
```

Typical workflow: scrape → train → forecast.

## Architecture

**Pipeline:** `scrape.py` → `images/{0,1,...}/` → `train.py` → `best_model.pth` → `forecast.py` → `forecast/`

**Model (defined in both train.py and forecast.py):**
- `ConvLSTMCell` — Conv2d-based LSTM cell with 32 hidden channels, kernel size 3
- `RadarPredictor` — Wraps ConvLSTMCell, processes input sequences over time, final Conv2d produces single-frame prediction with sigmoid output

**Key parameters:** `SEQ_LEN=6`, `IMG_SIZE=128`, `PREDICT_STEPS=12`, `lr=0.001`, `EPOCHS=10`. Device auto-selects CUDA if available.

**Data format:** Grayscale 128×128 PNGs normalized to [0,1]. Training creates overlapping windows of 6 input frames → 1 target frame from each numbered image folder. Forecast feeds predictions back as input autoregressively.

## Dependencies

Python 3 with torch, torchvision, numpy, requests, Pillow, imageio. Virtual environment in `.venv/`.

## Notes

- `best_model.pth` is gitignored — regenerate by training
- The ConvLSTMCell/RadarPredictor classes are duplicated across train.py and forecast.py
- Radar data source: UK Met Office WMS rainfall radar API
