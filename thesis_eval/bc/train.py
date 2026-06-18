"""
Train Behavior Cloning model on RoboCasa data.

Usage:
    python -m thesis_eval.bc.train --task PickPlaceCounterToCabinet --epochs 50
"""

import sys, os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from model import BCPolicy
from dataset import RoboCasaDataset

DATASET_BASE = "/media/razor/Razer/HKU_Dissertation/robocasa/datasets/v1.0"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────
    data_dir = os.path.join(DATASET_BASE, args.split, "atomic", args.task)

    train_ds = RoboCasaDataset(
        data_dir,
        split=args.split,
        image_size=args.image_size,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────
    model = BCPolicy(image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # ── Train ─────────────────────────────────────────────
    print(f"Training on {len(train_ds)} frames, {args.epochs} epochs")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        t0 = time.time()

        for images, states, actions in train_loader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)

            pred = model(images, states)
            loss = criterion(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.6f} | {elapsed:.0f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"bc_{args.task}_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Save final
    final_path = os.path.join(args.output_dir, f"bc_{args.task}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Done! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickPlaceCounterToCabinet")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str,
                        default="/media/razor/Razer/HKU_Dissertation/results/checkpoints")
    train(parser.parse_args())
