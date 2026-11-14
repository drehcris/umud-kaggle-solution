from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, base)
        self.down2 = ConvBlock(base, base * 2)
        self.down3 = ConvBlock(base * 2, base * 4)
        self.down4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        mid = self.mid(self.pool(d4))
        x = self.up4(mid)
        x = self.dec4(torch.cat([x, d4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, d1], dim=1))
        return self.head(x)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, target.ndim))
        intersection = (probs * target).sum(dims)
        union = probs.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()
        return 0.5 * bce + 0.5 * dice


class ThinStructureLoss(nn.Module):
    """Weighted BCE plus Tversky loss for highly sparse line annotations."""

    def __init__(
        self,
        pos_weight: float,
        alpha: float = 0.3,
        beta: float = 0.7,
        bce_weight: float = 0.35,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, target.ndim))
        true_positive = (probs * target).sum(dims)
        false_positive = (probs * (1 - target)).sum(dims)
        false_negative = ((1 - probs) * target).sum(dims)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - tversky.mean())
