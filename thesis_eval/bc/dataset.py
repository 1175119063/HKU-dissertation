"""
LeRobot Dataset loader for Behavior Cloning.

Loads parquet (states, actions) + mp4 videos (camera images).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import av  # pyav for fast video decoding


class RoboCasaDataset(Dataset):
    """Load LeRobot-format RoboCasa data for BC training.

    Each episode = 1 parquet file (states + actions) + 3 mp4 videos (cameras).
    We use the 'agentview_left' camera as the primary view.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "pretrain",
        image_size: int = 128,
        max_episodes: int = None,
        max_frames_per_episode: int = 200,
    ):
        self.image_size = image_size
        self.max_frames = max_frames_per_episode

        # Find all parquet files
        data_path = Path(data_dir)
        parquet_files = sorted(data_path.rglob("episode_*.parquet"))
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        # Load all frames into memory (small dataset: ~108 episodes × ~230 frames)
        self.images = []
        self.states = []
        self.actions = []

        for pf in parquet_files:
            # Load parquet
            df = pd.read_parquet(pf)
            states = np.stack(df["observation.state"].values).astype(np.float32)
            actions = np.stack(df["action"].values).astype(np.float32)

            # Load video frames for agentview_left
            video_dir = pf.parent.parent.parent / "videos" / pf.parent.name
            video_path = video_dir / "observation.images.robot0_agentview_left" / f"{pf.stem}.mp4"

            frames = self._load_video(video_path, len(states))

            # Clip to max frames
            n = min(len(frames), len(states), len(actions), self.max_frames)
            self.images.extend(frames[:n])
            self.states.extend(states[:n])
            self.actions.extend(actions[:n])

        print(f"Loaded {len(self.images)} frames from {len(parquet_files)} episodes")

    def _load_video(self, path: Path, expected_frames: int):
        """Load video frames as numpy arrays, resized."""
        frames = []
        try:
            container = av.open(str(path))
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                img = np.array(img, dtype=np.float32) / 255.0
                # Resize: simple center crop + resize
                h, w = img.shape[:2]
                sz = min(h, w)
                crop = img[(h - sz) // 2:(h + sz) // 2, (w - sz) // 2:(w + sz) // 2]
                # Downsample to target size
                from PIL import Image
                pil_img = Image.fromarray((crop * 255).astype(np.uint8))
                pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
                img = np.array(pil_img, dtype=np.float32) / 255.0
                frames.append(img)
                if len(frames) >= expected_frames:
                    break
            container.close()
        except Exception as e:
            # Video missing — fill with zeros
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)] * expected_frames
        return frames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # (3, H, W)
        state = torch.from_numpy(self.states[idx])
        action = torch.from_numpy(self.actions[idx])
        return image, state, action
