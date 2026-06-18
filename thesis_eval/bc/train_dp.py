"""
Train lightweight Diffusion Policy locally (8GB VRAM).

Usage:
    python train_dp.py --task PickPlaceCounterToCabinet --epochs 30
"""

import sys, os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from dp_model import DiffusionPolicy
from dataset import RoboCasaDataset

DATASET_BASE = "/media/razor/Razer/HKU_Dissertation/robocasa/datasets/v1.0"


class DPSequenceDataset(torch.utils.data.Dataset):
    """Wraps RoboCasaDataset to return (obs_seq, action_seq) for DP training."""

    def __init__(self, base_ds: RoboCasaDataset, obs_horizon=2, pred_horizon=16):
        self.ds = base_ds
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        # Valid start indices: need obs_horizon past frames and pred_horizon future frames
        self.valid_starts = list(range(obs_horizon - 1, len(base_ds) - pred_horizon - 1))
        print(f"DP dataset: {len(self.valid_starts)} valid sequences")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]

        # Get observation history
        images = []
        states = []
        for i in range(start - self.obs_horizon + 1, start + 1):
            img, state, _ = self.ds[i]
            images.append(img)
            states.append(state)

        # Get action sequence to predict
        actions = []
        for i in range(start + 1, start + 1 + self.pred_horizon):
            _, _, action = self.ds[i]
            actions.append(action)

        return (
            torch.stack(images),          # (obs_horizon, 3, H, W)
            torch.stack(states),           # (obs_horizon, state_dim)
            torch.stack(actions),          # (pred_horizon, action_dim)
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────
    data_dir = os.path.join(DATASET_BASE, args.split, "atomic", args.task)
    base_ds = RoboCasaDataset(
        data_dir,
        split=args.split,
        image_size=args.image_size,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames,
    )
    ds = DPSequenceDataset(base_ds, obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # ── Model ─────────────────────────────────────────────
    model = DiffusionPolicy(
        image_size=args.image_size,
        state_dim=16,
        action_dim=12,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Train ─────────────────────────────────────────────
    accumulation_steps = args.grad_accum
    print(f"Training: {len(ds)} seqs, {args.epochs} epochs, "
          f"batch={args.batch_size}, grad_accum={accumulation_steps}")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (images, states, actions) in enumerate(loader):
            images = images.to(device)      # (B, obs_horizon, 3, H, W)
            states = states.to(device)      # (B, obs_horizon, state_dim)
            actions = actions.to(device)    # (B, pred_horizon, action_dim)

            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                loss = model(images, states, actions)
                loss = loss / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.6f} | {elapsed:.0f}s")

        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"dp_{args.task}_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")

    final = os.path.join(args.output_dir, f"dp_{args.task}_final.pt")
    torch.save(model.state_dict(), final)
    print(f"Done! Model saved to {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickPlaceCounterToCabinet")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str,
                        default="/media/razor/Razer/HKU_Dissertation/results/checkpoints")
    train(parser.parse_args())
