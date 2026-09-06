# test.py
# Evaluation script: loads the best joint checkpoint and reports per-class
# Dice, IoU, and 95% Hausdorff Distance (HD95) for every dataset type.
# Auto generate official submission nii + mass json after evaluation
#
# Usage (on server):
#   python test.py \
#       --checkpoint checkpoints/best_model.pth \
#       --image_size 256 \
#       --batch_size 4  \
#       --num_workers 4 \
#       --submission_dir submission \
#       --visualize --vis_samples 4 --vis_output_dir vis_results

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
import nibabel as nib
import json
import shutil

from dataset import NIfTISegDataset, _minmax_norm_slice
from metrics import dice_score, iou_score, hausdorff_distance_95, mean_dice, mean_iou, mean_hd95
from config  import DATASET_CONFIGS, DEFAULT_FILTERS
from postprocess import decode_with_rules
from model import MultiDatasetDINOv2DPT

# -------------------------- 官方统一mass计算常量 --------------------------
LGE_LABEL3_ID = 3
LGE_TISSUE_DENSITY = 1.05
# -------------------------------------------------------------------------

# ─── Argument parser ─────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='Evaluate multi-dataset ResUNet++ & export submission')
    p.add_argument('--checkpoint',  type=str, default='checkpoints1/best_model.pth')
    p.add_argument('--image_size',  type=int, default=448)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--val_split',   type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--voxel_spacing', type=float, default=1.0,
                   help='Isotropic pixel spacing in mm for HD95 (default 1.0)')
    p.add_argument('--visualize', action='store_true',
                   help='Save random case visualization for 2ch/4ch/sa/ras154')
    p.add_argument('--vis_samples', type=int, default=4,
                   help='Number of random cases per dataset to visualize')
    p.add_argument('--vis_output_dir', type=str, default='vis_results2',
                   help='Output directory for visualization images')
    # 新增提交输出目录参数
    p.add_argument('--submission_dir', type=str, default='submission', help='Submission root folder')
    return p.parse_args()


# ─── 对齐训练输入：三层邻片3通道构造 ─────────────────────────────────
@torch.no_grad()
def get_three_channel_input(vol_3d: np.ndarray, z: int, target_size: int):
    depth = vol_3d.shape[-1]
    idx_list = [max(0, z - 1), z, min(depth - 1, z + 1)]
    channels = []
    for i in idx_list:
        slice_2d = vol_3d[..., i]
        norm_slice = _minmax_norm_slice(slice_2d)
        channels.append(norm_slice)
    img_np = np.stack(channels, axis=0)
    img_t = torch.from_numpy(img_np).float().unsqueeze(0)
    img_t = F.interpolate(img_t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return img_t


# ─── 单病例完整3D推理函数 ─────────────────────────────────────────────
@torch.no_grad()
def predict_volume_mask(model, img_vol: np.ndarray, dtype, orig_hw, image_size, device):
    H_orig, W_orig = orig_hw
    D_orig = img_vol.shape[-1]
    pred_vol = np.zeros_like(img_vol, dtype=np.uint8)
    for z in range(D_orig):
        img_t = get_three_channel_input(img_vol, z, image_size).to(device)
        logits = model(img_t, dtype)
        pred_448 = decode_with_rules(logits, DATASET_CONFIGS[dtype]["postprocess_rules"])[0].cpu().float()
        # 还原原图尺寸，nearest插值与原版逻辑一致
        pred_orig = F.interpolate(
            pred_448.unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig),
            mode='nearest'
        ).squeeze().round().numpy().astype(np.uint8)
        pred_vol[..., z] = pred_orig
    # 复刻官方读取mask时的round取整
    pred_vol = np.round(pred_vol).astype(np.uint8)
    return pred_vol


# ─── 批量导出提交文件 + mass json ─────────────────────────────────────
@torch.no_grad()
def predict_val_to_submission(model, val_folder_map, submission_dir, device, image_size):
    model.eval()
    val_to_sub = [("2ch", "2CH"), ("4ch", "4CH"), ("sa", "SAX"), ("ras154", "RAS")]

    task_dir = os.path.join(submission_dir, "task2_lge")
    # 清空旧提交文件夹
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    for _, out_folder in val_to_sub:
        os.makedirs(os.path.join(task_dir, out_folder), exist_ok=True)

    mass_dict = {}
    print(f"\n===== Start generating official submission package =====")
    for dtype, out_folder in val_to_sub:
        val_root = val_folder_map[dtype]
        img_dir = os.path.join(val_root, "image")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        if len(img_paths) == 0:
            print(f"Warning: No images found for {dtype}, skip")
            continue
        print(f"\nProcess {out_folder}, total {len(img_paths)} volumes")

        for idx, img_path in enumerate(img_paths):
            file_idx = idx + 1
            base_name = f"LGE_{out_folder}_{file_idx:03d}"
            nii_save_name = f"{base_name}.nii.gz"
            try:
                nii_obj = nib.load(img_path)
                img_vol = nii_obj.get_fdata()
                affine = nii_obj.affine
                dx, dy, dz = nii_obj.header.get_zooms()
                H, W = img_vol.shape[:2]

                # 推理完整分割mask
                pred_mask = predict_volume_mask(model, img_vol, dtype, (H, W), image_size, device)
                # 保存nii
                save_full_path = os.path.join(task_dir, out_folder, nii_save_name)
                nib.save(nib.Nifti1Image(pred_mask, affine), save_full_path)

                # 官方标准mass计算逻辑
                if out_folder == "RAS":
                    scar_mass = 0.0
                else:
                    target_pixels = np.sum(np.round(pred_mask) == LGE_LABEL3_ID)
                    total_mm3 = target_pixels * dx * dy * dz
                    total_cm3 = total_mm3 / 1000.0
                    scar_mass = round(total_cm3 * LGE_TISSUE_DENSITY, 1)
                mass_dict[base_name] = scar_mass
                print(f"  [{file_idx:03d}] Saved: {nii_save_name}, Scar Mass={scar_mass}g")
            except Exception as e:
                print(f"  Error processing {img_path}, skip. Err: {str(e)}")
                base_name = f"LGE_{out_folder}_{file_idx:03d}"
                mass_dict[base_name] = 0.0
    # 输出mass json
    json_path = os.path.join(task_dir, "mass_predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mass_dict, f, indent=4, ensure_ascii=False)
    print(f"\nSubmission generate done!")
    print(f"NII Folder: {task_dir}")
    print(f"Mass Json: {json_path}")
    print(f"Total case count: {len(mass_dict)}")


# ─── Per-dataset evaluation ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate_dataset(model, dataloader, dtype, num_classes, device,
                     voxel_spacing: float = 1.0):
    """
    Run inference on a single dataset's test loader.

    Returns:
        dice_per_class : List[float], length = num_classes
        iou_per_class  : List[float], length = num_classes
        hd95_per_class : List[float], length = num_classes
    """
    model.eval()
    dice_accum = [0.0] * num_classes
    iou_accum  = [0.0] * num_classes
    hd95_accum = [0.0] * num_classes
    n_batches  = 0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks  = masks.to(device)

        fg_logits = model(images, dtype)
        preds  = decode_with_rules(fg_logits, DATASET_CONFIGS[dtype]['postprocess_rules'])

        dc   = dice_score(preds, masks, num_classes)
        iou  = iou_score(preds,  masks, num_classes)
        hd95 = hausdorff_distance_95(preds, masks, num_classes,
                                     voxel_spacing=voxel_spacing)

        for c in range(num_classes):
            dice_accum[c] += dc[c]
            iou_accum[c]  += iou[c]
            hd95_accum[c] += hd95[c]
        n_batches += 1

    if n_batches == 0:
        return dice_accum, iou_accum, hd95_accum

    dice_per_class = [v / n_batches for v in dice_accum]
    iou_per_class  = [v / n_batches for v in iou_accum]
    hd95_per_class = [v / n_batches for v in hd95_accum]
    return dice_per_class, iou_per_class, hd95_per_class


def _label_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 102, 255),
        4: (255, 215, 0),
        5: (255, 0, 255),
    }
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb[mask_np == class_id] = color
    return rgb


def _overlay_label(base_gray: np.ndarray, label_np: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.clip(base_gray, 0.0, 1.0)
    base_rgb = np.stack([base, base, base], axis=-1)
    label_rgb = _label_to_rgb(label_np).astype(np.float32) / 255.0
    fg_mask = (label_np > 0)[..., None].astype(np.float32)
    out = base_rgb * (1.0 - fg_mask * alpha) + label_rgb * (fg_mask * alpha)
    return np.clip(out, 0.0, 1.0)


@torch.no_grad()
def visualize_random_cases(model, dataset, dtype, device, out_dir, n_samples=4, seed=42):
    if len(dataset) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    random.seed(seed)
    indices = list(range(len(dataset)))
    chosen = random.sample(indices, k=min(n_samples, len(indices)))

    for i, data_idx in enumerate(chosen):
        image_t, mask_t, _ = dataset[data_idx]
        image = image_t.unsqueeze(0).to(device)
        mask = mask_t.cpu().numpy().astype(np.int64)

        fg_logits = model(image, dtype)
        probs = torch.sigmoid(fg_logits)[0].cpu().numpy()
        pred = decode_with_rules(fg_logits, DATASET_CONFIGS[dtype]['postprocess_rules'])[0]
        pred = pred.cpu().numpy().astype(np.int64)

        base_gray = image_t[1].cpu().numpy()
        gt_overlay = _overlay_label(base_gray, mask)
        pred_overlay = _overlay_label(base_gray, pred)

        class2_map = probs[1] if probs.shape[0] >= 2 else np.zeros_like(base_gray)
        class3_map = probs[2] if probs.shape[0] >= 3 else np.zeros_like(base_gray)

        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        axes[0].imshow(base_gray, cmap='gray')
        axes[0].set_title('Input (middle slice)')
        axes[0].axis('off')

        axes[1].imshow(gt_overlay)
        axes[1].set_title('GT overlay')
        axes[1].axis('off')

        axes[2].imshow(pred_overlay)
        axes[2].set_title('Pred overlay')
        axes[2].axis('off')

        im3 = axes[3].imshow(class2_map, cmap='magma', vmin=0.0, vmax=1.0)
        axes[3].set_title('Score: class_2')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        im4 = axes[4].imshow(class3_map, cmap='magma', vmin=0.0, vmax=1.0)
        axes[4].set_title('Score: class_3')
        axes[4].axis('off')
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

        fig.suptitle(f"{dtype} | sample_idx={data_idx}", fontsize=12)
        save_path = os.path.join(out_dir, f"{dtype}_sample_{i+1:02d}_idx_{data_idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# ─── Pretty-print results table ───────────────────────────────────────────────
def print_results(dtype, num_classes, dice_list, iou_list, hd95_list):
    sep = '─' * 62
    print(f"\n{'═' * 62}")
    print(f"  Dataset: {dtype}   (num_classes={num_classes})")
    print(sep)
    print(f"  {'Class':<12} {'Dice':>10} {'IoU':>10} {'HD95 (mm)':>12}")
    print(sep)
    for c in range(num_classes):
        label   = f"class_{c}" + (" (bg)" if c == 0 else "")
        hd_str  = f"{hd95_list[c]:>12.2f}" if hd95_list[c] < 999.0 else "      N/A"
        print(f"  {label:<12} {dice_list[c]:>10.4f} {iou_list[c]:>10.4f}{hd_str}")
    print(sep)
    print(f"  {'Mean (fg)':<12} {mean_dice(dice_list):>10.4f} "
          f"{mean_iou(iou_list):>10.4f} {mean_hd95(hd95_list):>12.2f}")
    print(f"{'═' * 62}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = MultiDatasetDINOv2DPT(img_size=args.image_size, dino_encoder_size="large").to(device)
    # model = MultiDatasetDINOv2UNet().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    # 固定BN、冻结参数，对齐训练EMA推理
    model.eval()



    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}  "
          f"Val mean Dice at save: {ckpt.get('val_mean_dice', 'N/A'):.4f}")

    # ── Evaluate each dataset ─────────────────────────────────────────────────
    all_mean_dice = []
    all_mean_iou  = []
    all_mean_hd95 = []

    vis_types = {'2ch', '4ch', 'sa', 'ras154'}
    val_folder_map = {
        "2ch": "/root/autodl-tmp/CMR_multi_baseline-main/2D_seg/CMR-MULTI/LGE_MULTI/2CH_VAL",
        "4ch": "/root/autodl-tmp/CMR_multi_baseline-main/2D_seg/CMR-MULTI/LGE_MULTI/4CH_VAL",
        "sa":  "/root/autodl-tmp/CMR_multi_baseline-main/2D_seg/CMR-MULTI/LGE_MULTI/SAX_VAL",
        "ras154":  "/root/autodl-tmp/CMR_multi_baseline-main/2D_seg/CMR-MULTI/LGE_MULTI/RAS_VAL",
    }
    # 循环四个数据集评测
    for dtype in ["2ch", "4ch", "sa", "ras154"]:
        cfg = DATASET_CONFIGS[dtype]
        num_classes = cfg['num_classes']
        val_root = val_folder_map[dtype]

        img_dir = os.path.join(val_root, "image")
        ann_dir = os.path.join(val_root, "anno")
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        volume_pairs = []
        for img_path in img_files:
            fname = os.path.basename(img_path)
            seg_path = os.path.join(ann_dir, fname)
            if os.path.exists(seg_path):
                volume_pairs.append((img_path, seg_path))
        
        val_ds = NIfTISegDataset(
            root_dir=val_root,
            dataset_type=dtype,
            image_size=args.image_size,
            volume_pairs=volume_pairs,
            use_all_slices=True
        )
        loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"\nEvaluating '{dtype}' ({len(val_ds)} samples) ...")
        dice_list, iou_list, hd95_list = evaluate_dataset(
            model, loader, dtype, num_classes, device,
            voxel_spacing=args.voxel_spacing,
        )
        print_results(dtype, num_classes, dice_list, iou_list, hd95_list)

        all_mean_dice.append(mean_dice(dice_list))
        all_mean_iou.append(mean_iou(iou_list))
        all_mean_hd95.append(mean_hd95(hd95_list))

        if args.visualize and dtype in vis_types:
            vis_dir = os.path.join(args.vis_output_dir, dtype)
            visualize_random_cases(
                model=model,
                dataset=val_ds,
                dtype=dtype,
                device=device,
                out_dir=vis_dir,
                n_samples=args.vis_samples,
                seed=args.seed,
            )
            print(f"  Saved visualization to: {vis_dir}")



    # ── Overall summary ───────────────────────────────────────────────────────
    overall_dice = sum(all_mean_dice) / len(all_mean_dice)
    overall_iou  = sum(all_mean_iou)  / len(all_mean_iou)
    valid_hd95   = [v for v in all_mean_hd95 if v < 999.0]
    overall_hd95 = sum(valid_hd95) / len(valid_hd95) if valid_hd95 else float('inf')
    print(f"\n{'═' * 62}")
    print(f"  Overall across all datasets:")
    print(f"    Mean Dice (fg):     {overall_dice:.4f}")
    print(f"    Mean IoU  (fg):     {overall_iou:.4f}")
    print(f"    Mean HD95 (fg, mm): {overall_hd95:.2f}")
    print(f"{'═' * 62}\n")

    # ========== 评测完成后自动导出提交文件 ==========
    print("\n===== Start generate submission nii & mass json =====")
    predict_val_to_submission(
        model=model,
        val_folder_map=val_folder_map,
        submission_dir=args.submission_dir,
        device=device,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()