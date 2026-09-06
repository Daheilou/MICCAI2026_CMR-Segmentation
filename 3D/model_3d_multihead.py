from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = x.shape
        pooled = self.pool(x).view(batch_size, channels)
        weights = torch.relu(self.fc1(pooled))
        weights = torch.sigmoid(self.fc2(weights)).view(batch_size, channels, 1, 1, 1)
        return x * weights


class StemBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels),
        )
        self.se = SqueezeExcitation3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_path(x) + self.skip_path(x))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels),
        )
        self.se = SqueezeExcitation3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_path(x) + self.skip_path(x))


class ASPP3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: Sequence[int] = (1, 2, 4, 6)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.proj = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = None
        for branch in self.branches:
            out = branch(x)
            fused = out if fused is None else fused + out
        return self.proj(fused)


class AttentionGate3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.g_proj = nn.Sequential(
            nn.BatchNorm3d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(gate_channels, skip_channels, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.x_proj = nn.Sequential(
            nn.BatchNorm3d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels, skip_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            nn.BatchNorm3d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels, skip_channels, kernel_size=3, padding=1),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gated = self.g_proj(gate)
        skip_proj = self.x_proj(skip)
        if gated.shape[2:] != skip_proj.shape[2:]:
            gated = F.interpolate(gated, size=skip_proj.shape[2:], mode="trilinear", align_corners=False)
        out = self.out_conv(gated + skip_proj)
        return out * skip


class DecoderBlock3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = AttentionGate3D(gate_channels, skip_channels)
        self.res = ResBlock3D(gate_channels + skip_channels, out_channels, stride=1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attn_skip = self.attn(gate, skip)
        gate_up = F.interpolate(gate, size=attn_skip.shape[2:], mode="trilinear", align_corners=False)
        out = torch.cat([attn_skip, gate_up], dim=1)
        return self.res(out)


class ResUNetPPDecoder3D(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.d1 = DecoderBlock3D(gate_channels=256, skip_channels=64, out_channels=128)
        self.d2 = DecoderBlock3D(gate_channels=128, skip_channels=32, out_channels=64)
        self.d3 = DecoderBlock3D(gate_channels=64, skip_channels=16, out_channels=32)
        self.aspp = ASPP3D(32, 16)
        self.head = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, skip1: torch.Tensor, skip2: torch.Tensor, skip3: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.d1(bottleneck, skip3)
        x = self.d2(x, skip2)
        x = self.d3(x, skip1)
        x = self.aspp(x)
        x = self.head(x)
        return x


class ResUNetPP3DMultiHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        source_order: Sequence[str] = ("2ch", "4ch", "sa"),
        num_classes_by_source: Dict[str, int] = None,
    ) -> None:
        super().__init__()
        self.source_order = list(source_order)
        self.num_classes_by_source = num_classes_by_source or {"2ch": 3, "4ch": 5, "sa": 4}

        self.c1 = StemBlock3D(in_channels, 16, stride=1)
        self.c2 = ResBlock3D(16, 32, stride=2)
        self.c3 = ResBlock3D(32, 64, stride=2)
        self.c4 = ResBlock3D(64, 128, stride=2)
        self.b1 = ASPP3D(128, 256)

        self.decoders = nn.ModuleDict()
        for source in self.source_order:
            num_classes = self.num_classes_by_source[source]
            self.decoders[source] = ResUNetPPDecoder3D(num_classes=num_classes)

    def encode(self, x: torch.Tensor):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        s4 = self.c4(s3)
        b1 = self.b1(s4)
        return s1, s2, s3, b1

    def forward_source(self, x: torch.Tensor, source: str) -> torch.Tensor:
        s1, s2, s3, b1 = self.encode(x)
        return self.decoders[source](s1, s2, s3, b1)

    def forward(self, x: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        max_classes = max(self.num_classes_by_source.values())
        batch_size, _, depth, height, width = x.shape
        output = x.new_full((batch_size, max_classes, depth, height, width), fill_value=-1e4)

        for index, source in enumerate(self.source_order):
            mask = source_id == index
            if not torch.any(mask):
                continue
            logits = self.forward_source(x[mask], source)
            channels = logits.shape[1]
            output[mask, :channels] = logits
        return output
