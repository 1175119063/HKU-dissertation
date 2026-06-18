"""
Lightweight Diffusion Policy for 8GB GPUs.

Architecture:
  - CNN vision encoder (ResNet-18) → image features (512-dim)
  - State encoder (MLP) → state features (128-dim)
  - Diffusion UNet (1D CNN) → denoises action sequence
  - DDPM training: add noise → predict noise → MSE loss
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class DiffusionPolicy(nn.Module):
    """Lightweight Diffusion Policy for RoboCasa (8GB-friendly).

    Input: image (B, 3, H, W) + state (B, state_dim)
    Output: action sequence (B, pred_horizon, action_dim)
    """

    def __init__(
        self,
        image_size=128,
        state_dim=16,
        action_dim=12,
        pred_horizon=16,   # predict 16 future actions
        obs_horizon=2,     # use 2 past observations
        diffusion_steps=100,  # DDPM steps (fewer = faster training)
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps

        # Vision encoder
        self.vision = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision.fc = nn.Identity()
        img_feat_dim = 512

        # State encoder
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Noise prediction network (deeper, wider)
        cond_dim = img_feat_dim + 128  # image + state condition
        self.noise_pred = nn.Sequential(
            nn.Linear(action_dim * pred_horizon + cond_dim + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * pred_horizon),
        )

        # DDPM noise schedule
        self._build_schedule()

    def _build_schedule(self):
        """Linear beta schedule for DDPM."""
        betas = torch.linspace(0.0001, 0.02, self.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_obs(self, image, state):
        """Encode image + state observations into conditioning vector."""
        B = image.shape[0]
        img_feat = self.vision(image)                   # (B, 512)
        state_flat = state.reshape(B, -1)               # (B, state_dim * obs_horizon)
        state_feat = self.state_enc(state_flat)          # (B, 128)
        return torch.cat([img_feat, state_feat], dim=-1)  # (B, 640)

    def forward(self, image, state, action_seq):
        """Training step: add noise to action, predict noise, return loss.

        Args:
            image: (B, obs_horizon, 3, H, W)
            state: (B, obs_horizon, state_dim)
            action_seq: (B, pred_horizon, action_dim)
        """
        B = image.shape[0]

        # Flatten obs horizon into batch dimension for encoding
        image_flat = image.reshape(B * self.obs_horizon, *image.shape[-3:])
        state_flat = state.reshape(B, self.obs_horizon * state.shape[-1])

        # Encode
        cond = self.encode_obs(
            image.reshape(B, self.obs_horizon, 3, image.shape[-2], image.shape[-1])[:, -1],
            state_flat
        )  # (B, 640) — use last frame for conditioning

        # Flatten action
        x0 = action_seq.reshape(B, -1)  # (B, action_dim * pred_horizon)

        # Sample noise and timestep
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.diffusion_steps, (B,), device=x0.device)

        # Add noise: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        alpha_cumprod_t = self.alphas_cumprod[t].view(B, 1)
        x_t = torch.sqrt(alpha_cumprod_t) * x0 + torch.sqrt(1.0 - alpha_cumprod_t) * noise

        # Predict noise
        t_emb = t.float().unsqueeze(-1) / self.diffusion_steps  # (B, 1)
        noise_input = torch.cat([x_t, cond, t_emb], dim=-1)
        noise_pred = self.noise_pred(noise_input)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, image, state):
        """Inference: denoise from random noise to action sequence.

        Args:
            image: (1, 3, H, W) — single image
            state: (1, state_dim) — single state
        Returns:
            action: (action_dim,) — next action to take
        """
        B = 1
        device = image.device

        # Encode
        cond = self.encode_obs(image, state.reshape(B, -1).repeat(1, self.obs_horizon))
        cond = cond.reshape(B, -1)

        # Start from random noise
        x = torch.randn(B, self.action_dim * self.pred_horizon, device=device)

        # DDPM reverse process
        for t_idx in reversed(range(self.diffusion_steps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            t_emb = t.float().unsqueeze(-1) / self.diffusion_steps

            # Predict noise
            noise_input = torch.cat([x, cond, t_emb], dim=-1)
            noise_pred = self.noise_pred(noise_input)

            # DDPM step
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise

        # Return first action of the predicted sequence
        actions = x.reshape(B, self.pred_horizon, self.action_dim)
        return actions[0, 0]  # (action_dim,)
