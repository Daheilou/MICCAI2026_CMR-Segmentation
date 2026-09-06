# 放在所有import最顶部，消除OMP警告
import os

import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from model_factory import build_multihead_model

def _load_local_module(module_filename: str, module_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, module_filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dataset_module = _load_local_module("dataset_3d_multihead.py", "dataset_3d_multihead")

MultiSourceNiftiDataset3D = _dataset_module.MultiSourceNiftiDataset3D
collect_cases = _dataset_module.collect_cases
collect_cases_with_lvef = _dataset_module.collect_cases_with_lvef


# -------------------------- Excel LVEF读取工具（适配patient_id，统一去前导0） --------------------------
def load_lvef_from_excel(excel_path: str, case_id_col: str = "patient_id", lvef_col: str = "LVEF") -> dict:
    if not os.path.exists(excel_path):
        print(f"Warning: Excel file not found: {excel_path}, return empty LVEF dict")
        return {}
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        print(f"Excel ID column name: {case_id_col}")
        df = df.dropna(subset=[case_id_col, lvef_col])
        lvef_dict = {}
        # 判断：train文件带%，需要/100；valid文件本身是小数，不除
        is_train_set = "train" in excel_path
        for _, row in df.iterrows():
            pid_raw = str(row[case_id_col]).strip()
            pid = str(int(pid_raw)) if pid_raw.isdigit() else pid_raw
            lvef_raw = str(row[lvef_col]).strip()
            lvef_val = float(lvef_raw.replace("%", ""))
            # 训练集才归一化，验证集保持原值
            if is_train_set:
                lvef_val /= 100
            lvef_dict[pid] = lvef_val
        print(f"Excel LVEF key sample (first 10): {list(lvef_dict.keys())[:10]}")
        print(f"Loaded {len(lvef_dict)} valid LVEF labels from {excel_path}")
        return lvef_dict
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading Excel file: {e}, return empty LVEF dict")
        return {}

# -------------------------- 固定种子 --------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_num_classes(config: str):
    if config:
        return json.loads(config)
    return {"2ch": 3, "4ch": 5, "sa": 4}


# -------------------------- 分割指标计算 --------------------------
def multiclass_dice_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    pred = torch.argmax(logits, dim=1)
    eps = 1e-6
    dice_scores = []
    iou_scores = []
    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()
        intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
        pred_sum = pred_mask.sum(dim=(1, 2, 3))
        target_sum = target_mask.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)
        dice_scores.append(dice.mean())
        iou_scores.append(iou.mean())
    if not dice_scores:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    return torch.stack(dice_scores).mean(), torch.stack(iou_scores).mean()


# -------------------------- 分割损失 CrossEntropy + DiceLoss --------------------------
def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    if logits.shape[2:] != target.shape[1:]:
        logits = F.interpolate(logits, size=target.shape[1:], mode="trilinear", align_corners=False)
    ce = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    probs_fg = probs[:, 1:]
    target_fg = target_onehot[:, 1:]
    dims = (0, 2, 3, 4)
    intersection = (probs_fg * target_fg).sum(dims)
    denominator = probs_fg.sum(dims) + target_fg.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1.0 - dice.mean()
    return ce + dice_loss


# -------------------------- 回归损失 & MAE指标 --------------------------
reg_criterion = nn.SmoothL1Loss()
def reg_mae(pred_reg: torch.Tensor, gt_reg: torch.Tensor):
    return torch.mean(torch.abs(pred_reg - gt_reg))


def run_one_epoch(model, loader, optimizer, device, source_order, num_classes_by_source, is_train, reg_weight=0.3):
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_seg_loss = 0.0
    epoch_reg_loss = 0.0
    source_stats = defaultdict(lambda: {"dice": [], "iou": []})
    reg_mae_list = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        source_id = batch["source_id"].to(device)
        reg_label = batch["reg_label"].to(device)

        if is_train:
            optimizer.zero_grad()
        batch_seg_loss = 0.0
        batch_reg_loss = 0.0
        used_sources = 0

        with torch.set_grad_enabled(is_train):
            for src_index, src_name in enumerate(source_order):
                selected = source_id == src_index
                if not torch.any(selected):
                    continue
                src_images = images[selected]
                src_masks = masks[selected]
                src_reg_gt = reg_label[selected]
                num_classes = num_classes_by_source[src_name]

                if src_name == "sa":
                    # 调试打印当前批次LVEF真值，确认是否正常加载
                    # print(f"SAX batch LVEF gt: {src_reg_gt.detach().cpu().numpy()}")
                    logits, reg_pred = model.forward_source_reg(src_images, src_name)
                else:
                    logits = model.forward_source(src_images, src_name)
                    reg_pred = None

                if logits.shape[2:] != src_masks.shape[1:]:
                    logits = F.interpolate(logits, size=src_masks.shape[1:], mode="trilinear", align_corners=False)
                seg_loss = dice_ce_loss(logits, src_masks, num_classes)
                batch_seg_loss += seg_loss

                if src_name == "sa" and reg_pred is not None:
                    r_loss = reg_criterion(reg_pred.squeeze(), src_reg_gt)
                    batch_reg_loss += reg_weight * r_loss
                    mae_val = reg_mae(reg_pred.squeeze(), src_reg_gt)
                    reg_mae_list.append(float(mae_val.detach().cpu().item()))

                dice, iou = multiclass_dice_iou(logits, src_masks, num_classes)
                source_stats[src_name]["dice"].append(float(dice.detach().cpu().item()))
                source_stats[src_name]["iou"].append(float(iou.detach().cpu().item()))
                used_sources += 1

            if used_sources == 0:
                continue
            total_loss = (batch_seg_loss + batch_reg_loss) / used_sources
            if is_train:
                total_loss.backward()
                optimizer.step()

        # 修复：判断是否为tensor再执行detach，避免纯float报错
        epoch_seg_loss += float(batch_seg_loss.detach().cpu().item())
        if isinstance(batch_reg_loss, torch.Tensor):
            epoch_reg_loss += float(batch_reg_loss.detach().cpu().item())
        else:
            epoch_reg_loss += batch_reg_loss

    num_steps = max(len(loader), 1)
    epoch_seg_loss /= num_steps
    epoch_reg_loss /= num_steps

    summarized = {}
    for src in source_order:
        dice_values = source_stats[src]["dice"]
        iou_values = source_stats[src]["iou"]
        summarized[src] = {
            "dice": float(np.mean(dice_values)) if dice_values else 0.0,
            "iou": float(np.mean(iou_values)) if iou_values else 0.0,
        }
    avg_reg_mae = float(np.mean(reg_mae_list)) if reg_mae_list else 0.0
    return epoch_seg_loss, epoch_reg_loss, avg_reg_mae, summarized


def main():
    parser = argparse.ArgumentParser(description="3D Multi-task Train: seg for all, LVEF regression only for SAX")
    parser.add_argument("--train-root", type=str, required=True)
    parser.add_argument("--val-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="files_3d")
    parser.add_argument("--source-order", nargs="+", default=["2ch", "4ch", "sa"])
    parser.add_argument("--image-dirname", type=str, default="image")
    parser.add_argument("--label-dirname", type=str, default="seg")
    parser.add_argument("--num-classes-json", type=str, default='{"2ch":3,"4ch":5,"sa":4}')
    parser.add_argument("--train-lvef-xlsx", type=str, default="dataset_train.xlsx")
    parser.add_argument("--val-lvef-xlsx", type=str, default="dataset_val.xlsx")
    # 修复：默认列名改为patient_id，匹配你的Excel
    parser.add_argument("--excel-case-id-col", type=str, default="patient_id", help="Excel样本ID列名")
    parser.add_argument("--excel-lvef-col", type=str, default="LVEF", help="Excel LVEF数值列名")
    parser.add_argument("--input-size", nargs=3, type=int, default=[64, 160, 160])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    num_classes_by_source = parse_num_classes(args.num_classes_json)

    # 加载Excel LVEF映射
    train_lvef_dict = load_lvef_from_excel(args.train_lvef_xlsx, args.excel_case_id_col, args.excel_lvef_col)
    val_lvef_dict = load_lvef_from_excel(args.val_lvef_xlsx, args.excel_case_id_col, args.excel_lvef_col)

    # 使用dataset内置采集函数，输出标准CaseItem对象，无dict冲突
    train_cases = collect_cases_with_lvef(
        data_root=args.train_root,
        source_names=args.source_order,
        lvef_dict=train_lvef_dict,
        image_dirname=args.image_dirname,
        label_dirname=args.label_dirname,
    )
    val_cases = collect_cases_with_lvef(
        data_root=args.val_root,
        source_names=args.source_order,
        lvef_dict=val_lvef_dict,
        image_dirname=args.image_dirname,
        label_dirname=args.label_dirname,
    )
    print(f"Train cases total: {len(train_cases)}, Val cases total: {len(val_cases)}")

    # 原生数据集，内置返回reg_label，无需自定义子类
    train_dataset = MultiSourceNiftiDataset3D(
        cases=train_cases,
        output_size=args.input_size,
        source_num_classes=num_classes_by_source,
        augment=True,
    )
    val_dataset = MultiSourceNiftiDataset3D(
        cases=val_cases,
        output_size=args.input_size,
        source_num_classes=num_classes_by_source,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_multihead_model(
        backbone_name="vista3d",
        model_size="base",
        in_channels=1,
        source_order=args.source_order,
        num_classes_by_source=num_classes_by_source,
        enable_reg_head=True,
    ).to(device)



    model.freeze_backbone_low_level(freeze_up_to_layer=-1)
    model.show_vista_layer_grad()

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 3轮warmup，匹配你原来warmup_steps=3、0.1倍率预热
    warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
    # 50epoch完整余弦周期
    cos_sch = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cos_sch], milestones=[3])
    best_val_dice = -1.0
    best_ckpt = os.path.join(args.output_dir, "best_model_3d_lvef_multitask.pth")
    log_path = os.path.join(args.output_dir, "train_log_3d_lvef_multitask.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("3D Multi-Task Train Log: Segmentation + SAX LVEF Regression\n")
        f.write(f"reg_weight={args.reg_weight}\n")
        f.write(f"train_xlsx={args.train_lvef_xlsx}, val_xlsx={args.val_lvef_xlsx}\n\n")

    # -------------------------- 训练循环 --------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_seg_loss, train_reg_loss, train_reg_mae, train_stats = run_one_epoch(
            model, train_loader, optimizer, device, args.source_order, num_classes_by_source, True, args.reg_weight
        )
        val_seg_loss, val_reg_loss, val_reg_mae, val_stats = run_one_epoch(
            model, val_loader, optimizer, device, args.source_order, num_classes_by_source, False, args.reg_weight
        )

        val_dice_list = [val_stats[s]["dice"] for s in args.source_order]
        mean_val_dice = float(np.mean(val_dice_list)) if val_dice_list else 0.0
        total_val_loss = val_seg_loss + val_reg_loss
        scheduler.step()

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save({
                "model": model.state_dict(),
                "source_order": args.source_order,
                "num_classes_by_source": num_classes_by_source,
                "input_size": args.input_size,
                "reg_weight": args.reg_weight,
                "best_mean_val_dice": best_val_dice,
            }, best_ckpt)

        cost = time.time() - t0
        log_line = (
            f"Epoch {epoch:03d} | Time {cost:.1f}s | "
            f"TrainSeg={train_seg_loss:.4f} TrainReg={train_reg_loss:.4f} TrainMAE={train_reg_mae:.3f} | "
            f"ValSeg={val_seg_loss:.4f} ValReg={val_reg_loss:.4f} ValMAE={val_reg_mae:.3f} | "
            f"MeanValDice={mean_val_dice:.4f}\n"
        )
        for src in args.source_order:
            log_line += (
                f"  {src}: tr_dice={train_stats[src]['dice']:.4f} tr_iou={train_stats[src]['iou']:.4f}, "
                f"val_dice={val_stats[src]['dice']:.4f} val_iou={val_stats[src]['iou']:.4f}\n"
            )
        print(log_line, end="")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
    print(f"Train finished, best ckpt saved to {best_ckpt}")


if __name__ == "__main__":
    main()
