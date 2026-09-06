# train.py
# Joint training loop: all 4 datasets are trained simultaneously in round-robin
# order.  One shared encoder + 4 dataset-specific decoder heads.
# Loss = bce_weight * BCEWithLogits + dice_weight * binary SoftDiceLoss
#
# Usage (on server):
#   python train.py \
#       --epochs 100 \
#       --batch_size 4 \
#       --lr 1e-4 \
#       --image_size 256 \
#       --num_workers 4 \
#       --checkpoint_dir checkpoints
#
# Checkpoint saved to:  checkpoints/best_model.pth
# Training log saved to: checkpoints/train_log.csv

import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MultiDatasetDINOv2DPT
from dataset import MultiDatasetLoader_1
from metrics import dice_score, mean_dice, OneVsRestCombinedLoss
from config  import DATASET_CONFIGS, DEFAULT_FILTERS
from postprocess import decode_with_rules, masks_to_one_vs_rest
from copy import deepcopy

# ─── Argument parser ─────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='Joint training of multi-dataset ResUNet++')
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--lr',              type=float, default=5e-6)
    p.add_argument('--image_size',      type=int,   default=448)
    p.add_argument('--val_split',       type=float, default=0.1)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--checkpoint_dir',  type=str,   default='checkpoints6')
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--device',          type=str,   default='cuda:0',
                   help='cuda/cpu or specific like cuda:0')
    p.add_argument('--use_weighted_sampler', action='store_true',
                   help='Use per-slice weighted sampler for rare classes')
    p.add_argument('--bce_weight',        type=float, default=0.5,
                   help='Weight for BCEWithLogits term')
    p.add_argument('--dice_weight',      type=float, default=0.5,
                   help='Weight for SoftDiceLoss term')
    p.add_argument('--focal_weight',     type=float, default=0.4,
                   help='Interpolation weight to add focal modulation to BCE term')
    p.add_argument('--focal_gamma',      type=float, default=2.0,
                   help='Focal gamma for hard-example emphasis')
    return p.parse_args()


def _get_loss_tensors(dtype: str, device: torch.device):
    cfg = DATASET_CONFIGS[dtype].get('loss', {})
    pos_weight = cfg.get('pos_weight_fg', None)
    class_weights = cfg.get('class_weights_fg', None)

    pos_tensor = None
    cls_tensor = None
    if pos_weight is not None:
        pos_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    if class_weights is not None:
        cls_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return pos_tensor, cls_tensor


# ─── Validation helper ────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Run one validation pass over all datasets.

    Returns:
        avg_loss  : mean combined loss across all batches/datasets
        dice_map  : dict {dataset_type: mean_dice (ignoring background)}
    """
    model.eval()
    total_loss  = 0.0
    total_count = 0
    # Accumulate per-class dice sums per dataset
    dice_sums   = {k: [] for k in DATASET_CONFIGS}

    for images, masks, dtype in val_loader:
        images = images.to(device)
        masks  = masks.to(device)

        fg_logits = model(images, dtype)
        num_cls   = DATASET_CONFIGS[dtype]['num_classes']
        fg_target = masks_to_one_vs_rest(masks, num_cls)
        pos_weight, class_weights = _get_loss_tensors(dtype, device)
        loss      = criterion(
            fg_logits,
            fg_target,
            pos_weight=pos_weight,
            class_weights=class_weights,
        )
        total_loss  += loss.item()
        total_count += 1

        rules = DATASET_CONFIGS[dtype]['postprocess_rules']
        preds = decode_with_rules(fg_logits, rules)
        dc    = dice_score(preds, masks, num_cls)
        dice_sums[dtype].append(dc)

    avg_loss = total_loss / max(total_count, 1)
    dice_map = {}
    for dtype, records in dice_sums.items():
        if records:
            arr  = [sum(cls_vals) / len(cls_vals)
                    for cls_vals in zip(*records)]   # average over batches
            dice_map[dtype] = mean_dice(arr, ignore_background=True)
        else:
            dice_map[dtype] = 0.0

    return avg_loss, dice_map


# ─── Main training loop ───────────────────────────────────────────────────────
def main():
    args   = get_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Data loaders ─────────────────────────────────────────────────────────
    print("Building data loaders ...")


    
    train_loader = MultiDatasetLoader_1(
        DATASET_CONFIGS,
        split='train',
        batch_size=args.batch_size,
        # val_split=args.val_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=args.use_weighted_sampler,
    )
    val_loader = MultiDatasetLoader_1(
        DATASET_CONFIGS,
        split='val',
        batch_size=args.batch_size,
        # val_split=args.val_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=False,
    )
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val   batches/epoch: {len(val_loader)}")


    model = MultiDatasetDINOv2DPT(img_size=args.image_size, dino_encoder_size="large").to(device)
    # print(model)

    state_dict = torch.load(f'./dinov2_vitl14_pretrain.pth')

    # missing, unexpected = model.backbone.load_state_dict(state_dict)
    # print(missing)


    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False





    optimizer = AdamW([{'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': args.lr},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': args.lr * 40}], 
                        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)


    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,   # 整个训练周期余弦下降
        eta_min=1e-7         # 最低学习率
    )
    

    criterion = OneVsRestCombinedLoss(
        bce_weight=0.5,      # 降低BCE，避免压制小目标
        dice_weight=0.5,     # 提高Dice，小目标核心！
        focal_weight=0.75,    # 强烈加强困难样本
        focal_gamma=2.5,     # 困难样本挖掘更强
    )
    print(
        f"Loss: {args.bce_weight}×BCE + {args.dice_weight}×Dice "
        f"(focal_weight={args.focal_weight}, gamma={args.focal_gamma})"
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.checkpoint_dir, 'train_log.csv')
    dtype_list = list(DATASET_CONFIGS.keys())
    csv_header = (
        ['epoch', 'train_loss', 'val_loss'] +
        [f'val_dice_{d}' for d in dtype_list] +
        ['val_mean_dice', 'lr']
    )
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    # ── Training ─────────────────────────────────────────────────────────────
    best_mean_dice = -1.0
    ckpt_path      = os.path.join(args.checkpoint_dir, 'best_model.pth')

    total_iters = args.epochs * len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss   = 0.0
        train_count  = 0

        for i, (images, masks, dtype) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            fg_logits  = model(images, dtype)
            num_cls    = DATASET_CONFIGS[dtype]['num_classes']
            fg_target  = masks_to_one_vs_rest(masks, num_cls)
            pos_weight, class_weights = _get_loss_tensors(dtype, device)
            loss       = criterion(
                fg_logits,
                fg_target,
                pos_weight=pos_weight,
                class_weights=class_weights,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss  += loss.item()
            train_count += 1

            iters = (epoch - 1) * len(train_loader) + i
            # 多项式衰减lr，限制最低0
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # lr = max(lr, 1e-7)
            # optimizer.param_groups[0]["lr"] = lr
    
            # EMA 更新
            ema_ratio = min(1 - 1 / (iters + 1), 0.99)
            # 只同步参与梯度更新的参数，过滤冻结骨干，提速
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                if param.requires_grad:
                    param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            # buffer 全部同步
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

        scheduler.step()
        avg_train_loss = train_loss / max(train_count, 1)

        # ── Validation ────────────────────────────────────────────────────
        avg_val_loss, dice_map = validate(model_ema, val_loader, criterion, device)
        val_mean = sum(dice_map.values()) / max(len(dice_map), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────
        dice_strs = ' | '.join(
            f'{d}: {dice_map.get(d, 0.0):.4f}' for d in dtype_list
        )
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  "
            f"Val Dice ({dice_strs})  "
            f"Mean: {val_mean:.4f}  "
            f"LR: {current_lr:.2e}"
        )

        row = (
            [epoch, avg_train_loss, avg_val_loss] +
            [dice_map.get(d, 0.0) for d in dtype_list] +
            [val_mean, current_lr]
        )
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_mean > best_mean_dice:
            best_mean_dice = val_mean
            torch.save(
                {
                    'epoch':      epoch,
                    'model_state_dict': model_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mean_dice': val_mean,
                    'dice_per_dataset': dice_map,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (mean Dice={val_mean:.4f})")

    print(f"\nTraining complete. Best mean Dice: {best_mean_dice:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log:        {log_path}")


if __name__ == '__main__':
    main()
