"""
Behavior Cloning Model: ResNet-18 + MLP.

Input:  image (128x128x3) + robot_state (16-dim)
Output: action (12-dim)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BCPolicy(nn.Module):
    """Simple BC model: CNN encodes image, MLP predicts action."""

    def __init__(self, image_size=128, state_dim=16, action_dim=12):
        super().__init__()

        # Image encoder: ResNet-18 (pretrained)
        self.vision = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision.fc = nn.Identity()  # remove classifier, keep 512-dim features

        # State encoder
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, image, state):
        """
        Args:
            image: (B, 3, H, W) float32 in [0, 1]
            state: (B, state_dim) float32
        Returns:
            action: (B, action_dim) float32
        """
        img_feat = self.vision(image)       # (B, 512)
        state_feat = self.state_enc(state)  # (B, 128)
        fused = torch.cat([img_feat, state_feat], dim=-1)
        return self.action_head(fused)
