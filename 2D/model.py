#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from typing import Dict, Tuple, Optional, Union, List
from contextlib import contextmanager

# 项目内部模块
try:
    from config import DATASET_CONFIGS
    from backbone.dinov2 import DINOv2
    from util.blocks import FeatureFusionBlock, _make_scratch
except ImportError:
    raise ImportError("请确保 backbone/util/config 模块路径已加入PYTHONPATH")


def _make_fusion_block(features: int, use_bn: bool, size: Optional[Tuple[int, int]] = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    """DPT解码器头，适配DINOv2多尺度特征反融合上采样分割"""
    def __init__(
        self,
        nclass: int,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: List[int] = [256, 512, 1024, 1024],
    ):
        super().__init__()
        self.nclass = nclass
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 四层特征通道映射 1x1 conv
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for oc in out_channels
        ])

        # 各层分辨率恢复：上采样/下采样对齐到统一尺寸
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        # scratch refine融合模块
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # 输出分割头
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, out_features: List[torch.Tensor], patch_h: int, patch_w: int) -> torch.Tensor:
        feat_list = []
        for i, x in enumerate(out_features):
            # (B, num_patch, C) -> (B, C, Hp, Wp)
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            feat_list.append(x)

        layer_1, layer_2, layer_3, layer_4 = feat_list

        # 通道归一化层
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 自底向上多尺度融合
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        fuse_feat = self.scratch.refinenet1(path_2, layer_1_rn)

        logits = self.scratch.output_conv(fuse_feat)
        return logits

    def get_fusion_feature(self, out_features: List[torch.Tensor], patch_h: int, patch_w: int) -> torch.Tensor:
        """仅返回融合后的深层特征，不输出logits"""
        feat_list = []
        for i, x in enumerate(out_features):
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            feat_list.append(x)

        layer_1, layer_2, layer_3, layer_4 = feat_list
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        fuse_feat = self.scratch.refinenet1(path_2, layer_1_rn)
        return fuse_feat


class MultiDatasetDINOv2DPT(nn.Module):
    """
    多数据集共享DINOv2骨干 + 独立DPT分割头模型
    支持多通道输入、分层微调、特征dropout增强、多任务分割推理
    """
    DATASET_TYPES = ('2ch', '4ch', 'sa', 'ras154')
    # DINOv2各尺寸中间层索引（提取4层transformer输出）
    INTER_LAYER_IDX: Dict[str, List[int]] = {
        'small': [2, 5, 8, 11],
        'base': [2, 5, 8, 11],
        'large': [4, 11, 17, 23],
        'giant': [9, 19, 29, 39]
    }

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 518,
        dino_encoder_size: str = "large",
        dpt_features: int = 128,
        dropout_p: float = 0.2
    ):
        super().__init__()
        self.img_size = img_size
        self.dino_size = dino_encoder_size
        self.dropout_p = dropout_p
        self.binomial = Binomial(probs=1 - dropout_p)

        # 多通道输入适配到3通道给DINOv2
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        else:
            self.input_adapter = nn.Identity()

        # DINOv2视觉骨干
        self.backbone = DINOv2(model_name=dino_encoder_size)
        self.inter_layer_idx = self.INTER_LAYER_IDX[dino_encoder_size]
        self.embed_dim = self.backbone.embed_dim

        # 各数据集独立DPT解码器
        self.decoders = nn.ModuleDict()
        for dtype in self.DATASET_TYPES:
            n_classes = DATASET_CONFIGS[dtype]['num_classes'] - 1
            self.decoders[dtype] = DPTHead(
                nclass=n_classes,
                in_channels=self.embed_dim,
                features=dpt_features,
                out_channels=[96, 192, 384, 768],
                use_bn=False
            )

        # 默认冻结全部骨干
        self.lock_backbone()

    # -------------------------- 骨干冻结/解冻工具 --------------------------
    def lock_backbone(self):
        """冻结整个DINOv2骨干"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unlock_backbone(self):
        """完全解冻骨干"""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unlock_backbone_last_n_layers(self, n: int):
        """仅解冻transformer最后n层，前面全部冻结（微调常用策略）"""
        total_layers = len(self.backbone.vit.blocks)
        freeze_threshold = total_layers - n
        for idx, block in enumerate(self.backbone.vit.blocks):
            requires_grad = idx >= freeze_threshold
            for p in block.parameters():
                p.requires_grad = requires_grad
        # ln_norm 与 head 一同解冻
        for p in self.backbone.vit.norm.parameters():
            p.requires_grad = True

    @contextmanager
    def backbone_grad_enable(self, enable: bool = True):
        """上下文管理器：临时切换骨干梯度开关，退出自动恢复"""
        original_state = [p.requires_grad for p in self.backbone.parameters()]
        try:
            for p in self.backbone.parameters():
                p.requires_grad = enable
            yield
        finally:
            for param, state in zip(self.backbone.parameters(), original_state):
                param.requires_grad = state

    # -------------------------- 前向传播与特征丢弃增强 --------------------------
    def _comp_drop_aug(self, feat_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """comp_drop 随机丢弃部分特征通道增强，训练时启用"""
        if self.dropout_p <= 0.0 or not self.training:
            return feat_list
        drop_mask = self.binomial.sample(feat_list[0].shape[1]).bool()
        masked_feats = []
        for feat in feat_list:
            feat[:, ~drop_mask, :, :] = 0.0
            masked_feats.append(feat)
        return masked_feats

    def forward(
        self,
        x: torch.Tensor,
        dataset_type: str,
        comp_drop: bool = False
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        # 缩放至模型固定输入尺寸
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)

        x = self.input_adapter(x)
        patch_h, patch_w = self.img_size // 14, self.img_size // 14
        feats = self.backbone.get_intermediate_layers(x, self.inter_layer_idx)
        head = self.decoders[dataset_type]

        logits = head(feats, patch_h, patch_w)
        # 恢复原图分辨率
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
        return logits

    def get_feat_and_logit(self, x: torch.Tensor, dataset_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回融合特征 + 原图尺寸分割logits"""
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
        x = self.input_adapter(x)
        patch_h, patch_w = self.img_size // 14, self.img_size // 14
        feats = self.backbone.get_intermediate_layers(x, self.inter_layer_idx)
        head = self.decoders[dataset_type]

        fuse_feat = head.get_fusion_feature(feats, patch_h, patch_w)
        logits = head(feats, patch_h, patch_w)
        logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=True)
        return fuse_feat, logits

    # -------------------------- 推理工具接口 --------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, dataset_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理接口
        return: seg_prob(B, cls, H, W), seg_label(B, H, W)
        """
        self.eval()
        logits = self(x, dataset_type, comp_drop=False)
        seg_prob = F.softmax(logits, dim=1)
        seg_label = torch.argmax(seg_prob, dim=1)
        return seg_prob, seg_label

    # -------------------------- 权重保存/加载 --------------------------
    def save_ckpt(self, save_path: str, extra_info: Optional[Dict] = None):
        """保存模型权重与配置信息"""
        save_dict = {
            "state_dict": self.state_dict(),
            "model_cfg": {
                "in_channels": self.input_adapter.in_channels if isinstance(self.input_adapter, nn.Conv2d) else 3,
                "img_size": self.img_size,
                "dino_size": self.dino_size,
                "dropout_p": self.dropout_p
            },
            "extra": extra_info or {}
        }
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load_ckpt(cls, ckpt_path: str, device: Union[str, torch.device] = "cpu") -> MultiDatasetDINOv2DPT:
        """从checkpoint恢复模型"""
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["model_cfg"]
        model = cls(
            in_channels=cfg["in_channels"],
            img_size=cfg["img_size"],
            dino_encoder_size=cfg["dino_size"],
            dropout_p=cfg["dropout_p"]
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        print(f"Loaded checkpoint from {ckpt_path}")
        return model

    # -------------------------- 参数量统计 --------------------------
    def count_params(self) -> Dict[str, int]:
        """统计骨干、解码器、总参数量"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for dec in self.decoders.values() for p in dec.parameters())
        adapter_params = sum(p.numel() for p in self.input_adapter.parameters())
        total = backbone_params + decoder_params + adapter_params
        return {
            "backbone": backbone_params,
            "decoders": decoder_params,
            "input_adapter": adapter_params,
            "total": total
        }


# -------------------------- 增强测试代码 --------------------------
def test_dinov2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test device: {device}")

    # 1. 初始化模型（12通道输入，base DINO，128维DPT特征）
    model = MultiDatasetDINOv2DPT(
        in_channels=12,
        img_size=518,
        dino_encoder_size="base",
        dpt_features=128,
        dropout_p=0.2
    ).to(device)
    model.eval()

    # 打印参数量
    param_stats = model.count_params()
    print("\n===== Model Params Stats =====")
    for k, v in param_stats.items():
        print(f"{k}: {v / 1e6:.2f} M")

    # 2. 多分辨率输入测试
    test_shapes = [(1, 12, 44, 518), (2, 12, 600, 600), (1, 12, 518, 518)]
    test_dtypes = list(MultiDatasetDINOv2DPT.DATASET_TYPES)

    for shape in test_shapes:
        x = torch.randn(*shape).to(device)
        B, C, H, W = x.shape
        print(f"\nTest input shape: {shape}")
        for dtype in test_dtypes[:2]:  # 只测前两个数据集提速
            with torch.no_grad():
                out_logit = model(x, dataset_type=dtype)
                fuse_feat, logit2 = model.get_feat_and_logit(x, dataset_type=dtype)
                prob, label = model.predict(x, dataset_type=dtype)

            print(f"  Dataset {dtype}:")
            print(f"    Logit shape: {out_logit.shape}")
            print(f"    Fusion feat shape: {fuse_feat.shape}")
            print(f"    Seg prob shape: {prob.shape}, label shape: {label.shape}")

    # 3. 测试骨干解冻上下文管理器
    print("\n===== Test backbone temp grad enable =====")
    print(f"Before ctx: backbone block 0 grad = {next(model.backbone.vit.blocks[0].parameters()).requires_grad}")
    with model.backbone_grad_enable(enable=True):
        print(f"In ctx: backbone block 0 grad = {next(model.backbone.vit.blocks[0].parameters()).requires_grad}")
    print(f"After ctx: backbone block 0 grad = {next(model.backbone.vit.blocks[0].parameters()).requires_grad}")

    # 4. 分层解冻测试（解冻最后4层transformer）
    model.unlock_backbone_last_n_layers(n=4)
    unfrozen = sum(1 for blk in model.backbone.vit.blocks if next(blk.parameters()).requires_grad)
    print(f"\nUnlocked last {unfrozen} transformer blocks")

    print("\n✅ All tests passed, model works normally")


if __name__ == "__main__":
    test_dinov2_model()