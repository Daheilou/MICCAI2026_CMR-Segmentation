from typing import Sequence, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import vista3d132


class VistaSegWrapper(nn.Module):
    """单分支VISTA封装（仅兼容单数据源场景）"""
    def __init__(self, vista_model, num_classes):
        super().__init__()
        self.vista = vista_model
        self.num_classes = num_classes
        self.max_cls = num_classes
        self.register_buffer("cls_vecs", torch.arange(self.max_cls).unsqueeze(1))

    def forward(self, x):
        def enc_fn(img):
            return self.vista.image_encoder(img, with_point=False, with_label=True)
        _, out_auto = checkpoint(enc_fn, x, use_reentrant=False)
        mask_logits_all, _ = self.vista.class_head(out_auto, class_vector=self.cls_vecs)
        return mask_logits_all.permute(1, 0, 2, 3, 4)


class MultiHeadSegWrapper(torch.nn.Module):
    """
    共享骨干多数据源分割封装
    - VISTA/UNet双兼容
    - SAX回归头严格使用你指定的AvgPool极简结构
    - 特征只提取一次，分割+回归共享，节省显存
    - 梯度检查点 + 批量class_head切片优化显存速度
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
        source_order: Sequence[str],
        num_classes_by_source: Dict[str, int],
        enable_reg_head: bool = False,
        vista_embed_dim: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.source_order = source_order
        self.num_classes_by_source = num_classes_by_source
        self.enable_reg_head = enable_reg_head
        self.vista_embed_dim = vista_embed_dim

        # 判断骨干类型
        from monai.networks.nets import vista3d132
        self.is_vista = isinstance(backbone, type(vista3d132()))
        self.seg_heads = nn.ModuleDict()

        # UNet分支：独立1x1卷积分割头
        if not self.is_vista:
            feat_dim = backbone.out_channels
            for src, n_cls in num_classes_by_source.items():
                self.seg_heads[src] = nn.Conv3d(feat_dim, n_cls, kernel_size=1, bias=True)
        else:
            # VISTA预缓存全局最大类别向量，一次性推理所有类别
            self.max_global_cls = max(num_classes_by_source.values())
            self.register_buffer("global_cls_vecs", torch.arange(self.max_global_cls).unsqueeze(1))

        # 严格采用你指定的回归头结构，无额外注意力
        if self.enable_reg_head and self.is_vista:
            reg_in_dim = self.vista_embed_dim
            self.sax_reg_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(reg_in_dim, reg_in_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(reg_in_dim // 2, 1)
            )

    def extract_shared_feature(self, x: torch.Tensor) -> torch.Tensor:
        """梯度检查点提取共享特征，降低encoder激活显存"""
        if self.is_vista:
            def encoder_forward(img):
                return self.backbone.image_encoder(img, with_point=False, with_label=True)
            _, feat = checkpoint(encoder_forward, x, use_reentrant=False)
            return feat
        return self.backbone(x)

    def _vista_seg_logit(self, feat: torch.Tensor, n_cls: int) -> torch.Tensor:
        """VISTA批量推理全部类别再切片，消除for循环提速省显存"""
        all_logits, _ = self.backbone.class_head(feat, class_vector=self.global_cls_vecs)
        seg_logits = all_logits[:n_cls].permute(1, 0, 2, 3, 4)
        return seg_logits

    def freeze_backbone_low_level(self, freeze_up_to_layer:int=1):
        """
        VISTA3D 骨干冻结策略消融接口
        freeze_up_to_layer: 冻结 0 ... freeze_up_to_layer，剩余高层开放梯度
            -1: full finetune，全部encoder开放
             0: freeze layer0, open 1,2,3,4
             1: freeze layer0‑1, open 2,3,4 (default，当前实验配置)
             2: freeze layer0‑2, open3‑4
             3: freeze layer0‑3, open4
             4: freeze all encoder layers0‑4，仅class_head可训练
        point_head 永久全部冻结，本任务不使用。
        """
        # if not self.is_vista:
        #     self.unfreeze_backbone()
        #     return


    
        raw_bb = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        # step1: 骨干全部置为不需要梯度
        # for p in raw_bb.parameters():
        #     p.requires_grad = False
    
        for name, p in raw_bb.named_parameters():
            # point_head 永远冻结
            if "point_head." in name:
                p.requires_grad = True
                continue
    
            # class_head 永远打开
            if "class_head." in name:
                p.requires_grad = True
                continue
    
            # encoder layers逻辑
            if "image_encoder.encoder.layers." in name:
                # 提取layer index: image_encoder.encoder.layers.X.xxx
                layer_str = name.split("image_encoder.encoder.layers.")[1].split(".")[0]
                layer_idx = int(layer_str)
                if freeze_up_to_layer == -1:
                    p.requires_grad = True
                else:
                    if layer_idx > freeze_up_to_layer:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False




    def show_vista_layer_grad(self):
        if not self.is_vista:
            return
        raw_bb = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
    
        stats = {
            "ly0":0, "ly0_train":0,
            "ly1":0, "ly1_train":0,
            "ly2":0, "ly2_train":0,
            "ly3":0, "ly3_train":0,
            "ly4":0, "ly4_train":0,
            "class_head":0, "class_head_train":0,
            "point_head":0, "point_head_train":0,
        }
        for name, p in raw_bb.named_parameters():
            if "image_encoder.encoder.layers.0." in name:
                stats["ly0"] +=1
                if p.requires_grad: stats["ly0_train"] +=1
            elif "image_encoder.encoder.layers.1." in name:
                stats["ly1"] +=1
                if p.requires_grad: stats["ly1_train"] +=1
            elif "image_encoder.encoder.layers.2." in name:
                stats["ly2"] +=1
                if p.requires_grad: stats["ly2_train"] +=1
            elif "image_encoder.encoder.layers.3." in name:
                stats["ly3"] +=1
                if p.requires_grad: stats["ly3_train"] +=1
            elif "image_encoder.encoder.layers.4." in name:
                stats["ly4"] +=1
                if p.requires_grad: stats["ly4_train"] +=1
            elif "class_head." in name:
                stats["class_head"] +=1
                if p.requires_grad: stats["class_head_train"] +=1
            elif "point_head." in name:
                stats["point_head"] +=1
                if p.requires_grad: stats["point_head_train"] +=1
    
        print("\n==== VISTA grad status ====")
        print(f"  layers[0] shallow: {stats['ly0_train']}/{stats['ly0']} trainable")
        print(f"  layers[1]         : {stats['ly1_train']}/{stats['ly1']} trainable")
        print(f"  layers[2]         : {stats['ly2_train']}/{stats['ly2']} trainable")
        print(f"  layers[3]         : {stats['ly3_train']}/{stats['ly3']} trainable")
        print(f"  layers[4] deepest : {stats['ly4_train']}/{stats['ly4']} trainable")
        print(f"  class_head        : {stats['class_head_train']}/{stats['class_head']} trainable")
        print(f"  point_head        : {stats['point_head_train']}/{stats['point_head']} trainable")

    def forward_source(self, x: torch.Tensor, src_name: str) -> torch.Tensor:
        """普通切面：仅输出分割logits"""
        n_cls = self.num_classes_by_source[src_name]
        feat = self.extract_shared_feature(x)
        if not self.is_vista:
            return self.seg_heads[src_name](feat)
        return self._vista_seg_logit(feat, n_cls)

    def forward_source_reg(self, x: torch.Tensor, src_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAX专用：一次提取特征，同时计算分割+回归，不重复走encoder"""
        assert self.enable_reg_head, "Model init must set enable_reg_head=True"
        assert self.is_vista, "Regression only support VISTA backbone"
        assert src_name == "sa", "Regression head only available for source 'sa'"

        n_cls = self.num_classes_by_source[src_name]
        feat = self.extract_shared_feature(x)
        # 分割输出
        seg_logits = self._vista_seg_logit(feat, n_cls)
        # LVEF回归输出，使用你定义的sax_reg_head
        reg_pred = self.sax_reg_head(feat)
        return seg_logits, reg_pred

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """默认前向：输出所有切面分割结果字典"""
        out_dict = {}
        for src in self.source_order:
            out_dict[src] = self.forward_source(x, src)
        return out_dict

    def freeze_backbone(self):
        """冻结骨干，仅训练分割/回归头，大幅减少梯度显存占用"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻骨干全部参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_backbone(
    name: str,
    model_size: str,
    in_channels: int,
    ckpt_path: Optional[str] = None
) -> Tuple[nn.Module, int]:
    """加载VISTA骨干，支持自定义权重路径、兼容DDP module.前缀"""
    name = name.lower()
    if name in ("vast3d", "vista3d"):
        name = "vista3d"

    if name == "vista3d":
        size_cfg = {
            "small": {"channels": (16, 32, 64, 128)},
            "base": {"channels": (48, 96, 192, 384)},
            "large": {"channels": (64, 128, 256, 512)},
        }[model_size]
        channels = size_cfg["channels"]
        embed_dim = channels[0]
        base_vista = vista3d132(encoder_embed_dim=embed_dim, in_channels=in_channels)

        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                fixed_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
                load_res = base_vista.load_state_dict(fixed_ckpt, strict=False)
                print(f"[VISTA Checkpoint] Load done | Matched layers: {len(fixed_ckpt)-len(load_res.unexpected_keys)} | Missing layers: {len(load_res.missing_keys)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Warning] Fail to load ckpt {ckpt_path}, skip loading")
        return base_vista, embed_dim

    raise ValueError(f"Unsupported backbone name {name}, only vista3d available")


def build_multihead_model(
    backbone_name: str = "vista3d",
    model_size: str = "large",
    in_channels: int = 1,
    source_order: Sequence[str] = ("2ch", "4ch", "sa"),
    num_classes_by_source: Optional[Dict[str, int]] = None,
    enable_reg_head: bool = False,
    ckpt_path: Optional[str] = "./model.pt",
) -> MultiHeadSegWrapper:
    """统一模型构建入口"""
    if num_classes_by_source is None:
        num_classes_by_source = {"2ch": 3, "4ch": 6, "sa": 4}

    backbone, vista_embed_dim = get_backbone(
        name=backbone_name,
        model_size=model_size,
        in_channels=in_channels,
        ckpt_path=ckpt_path
    )
    model = MultiHeadSegWrapper(
        backbone=backbone,
        source_order=source_order,
        num_classes_by_source=num_classes_by_source,
        enable_reg_head=enable_reg_head,
        vista_embed_dim=vista_embed_dim
    )
    return model


# 测试示例
if __name__ == "__main__":
    from torch.cuda.amp import autocast, GradScaler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建带回归头模型
    model = build_multihead_model(enable_reg_head=True, ckpt_path=None).to(device)
    # model.freeze_backbone()  # 可选冻结骨干省显存

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    accum_steps = 2

    # 模拟输入 B,C,D,H,W
    dummy_img = torch.randn(1, 1, 32, 128, 128).to(device)

    # 1. 全部切面分割推理
    with autocast(), torch.no_grad():
        out_all = model(dummy_img)
        print("All source seg keys:", list(out_all.keys()))

    # 2. SA分割+LVEF回归推理（核心）
    with autocast(), torch.no_grad():
        sa_seg, lvef_pred = model.forward_source_reg(dummy_img, src_name="sa")
    print("SA seg logits shape:", sa_seg.shape)    # [B,4,D,H,W]
    print("LVEF pred shape:", lvef_pred.shape)    # [B,1]

    # 训练单步示范
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        seg_logits, reg_out = model.forward_source_reg(dummy_img, "sa")
        loss_seg = seg_logits.mean()
        loss_reg = reg_out.abs().mean()
        total_loss = (loss_seg + 0.1 * loss_reg) / accum_steps

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)