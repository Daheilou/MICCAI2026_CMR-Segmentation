import argparse
import json
import os
import re
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import zoom
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

# ==================== 本地模型/数据集加载模块（完整保留） ====================
def _load_local_module(module_filename: str, module_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, module_filename)
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_dataset_module = _load_local_module("dataset_3d_multihead.py", "dataset_3d_multihead")
MultiSourceNiftiDataset3D = _dataset_module.MultiSourceNiftiDataset3D
collect_cases = _dataset_module.collect_cases

# ==================== 工具函数（保留，不再用容积算EF） ====================
def read_id_slice_sax(json_path):
    mapping_path2 = json_path
    if not os.path.exists(mapping_path2):
        print(f"Warning: {mapping_path2} not found.")
        return {}
    with open(mapping_path2, "r") as f:
        id_slice = json.load(f)
    return mapping_path

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

# ==================== 分割指标计算（完整模型评测） ====================
def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    if not mask_a.any() and not mask_b.any():
        return np.asarray([0.0], dtype=np.float32)
    if not mask_a.any() or not mask_b.any():
        max_dist = float(np.linalg.norm(mask_a.shape))
        return np.asarray([max_dist], dtype=np.float32)
    structure = generate_binary_structure(mask_a.ndim, 1)
    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a, structure=structure, border_value=0))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b, structure=structure, border_value=0))
    if not surface_a.any() or not surface_b.any():
        max_dist = float(np.linalg.norm(mask_a.shape))
        return np.asarray([max_dist], dtype=np.float32)
    dt_b = distance_transform_edt(~surface_b)
    dt_a = distance_transform_edt(~surface_a)
    dist_a_to_b = dt_b[surface_a]
    dist_b_to_a = dt_a[surface_b]
    return np.concatenate([dist_a_to_b, dist_b_to_a]).astype(np.float32)

def hd95_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    distances = _surface_distances(pred_mask, target_mask)
    return float(np.percentile(distances, 95))

def multiclass_dice_iou_hd95(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    eps = 1e-6
    dice_scores = []
    iou_scores = []
    hd95_scores = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()
        intersection = (pred_mask * target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)
        dice_scores.append(float(dice.item()))
        iou_scores.append(float(iou.item()))
        pred_mask_np = pred_np == class_id
        target_mask_np = target_np == class_id
        if pred_mask_np.any() or target_mask_np.any():
            hd95_scores.append(hd95_binary(pred_mask_np, target_mask_np))
    if not dice_scores:
        return 0.0, 0.0, 0.0
    mean_hd95 = float(np.mean(hd95_scores)) if hd95_scores else 0.0
    return float(np.mean(dice_scores)), float(np.mean(iou_scores)), mean_hd95

# ==================== 通用工具 ====================
def parse_num_classes(config: str):
    if config:
        return json.loads(config)
    return {"2ch": 3, "4ch": 5, "sa": 4}

def extract_file_number(file_path: str) -> int:
    fname = os.path.basename(file_path)
    nums = re.findall(r"\d+", fname)
    return int(nums[0]) if nums else 99999

def scale_label_array(arr: np.ndarray, target_shape):
    scale_factors = [t / s for s, t in zip(arr.shape, target_shape)]
    return zoom(arr, scale_factors, order=0)

# ==================== 主推理入口：SAX调用forward_source_reg获取模型回归EF ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="task1_cine")
    parser.add_argument("--source-order", nargs="+", default=["sa", "2ch", "4ch"])
    parser.add_argument("--image-dirname", type=str, default="image")
    parser.add_argument("--label-dirname", type=str, default="anno")
    parser.add_argument("--num-classes-json", type=str, default='{"2ch":3,"4ch":6,"sa":4}')
    parser.add_argument("--input-size", nargs=3, type=int, default=[144, 144, 144])
    args = parser.parse_args()

    view_map = {"sa": "SAX", "2ch": "2CH", "4ch": "4CH"}
    ef_json_save_path = os.path.join(args.output_root, "ef_predictions.json")
    # slice_mapping = read_id_slice_sax(args.slice_json_path)

    # 创建输出文件夹
    for src in args.source_order:
        out_dir = os.path.join(args.output_root, view_map[src])
        os.makedirs(out_dir, exist_ok=True)

    num_classes_by_source = parse_num_classes(args.num_classes_json)
    all_cases = collect_cases(
        data_root=args.data_root,
        source_names=args.source_order,
        image_dirname=args.image_dirname,
        label_dirname=args.label_dirname,
    )
    all_cases.sort(key=lambda x: extract_file_number(x.image_path))

    # 3D数据集构建
    dataset = MultiSourceNiftiDataset3D(
        cases=all_cases,
        output_size=args.input_size,
        source_num_classes=num_classes_by_source,
        augment=False,
    )

    # 加载3D多头模型，开启回归头
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from model_factory import build_multihead_model
    model = build_multihead_model(
        backbone_name="vast3d",
        model_size="base",
        in_channels=1,
        source_order=args.source_order,
        num_classes_by_source=num_classes_by_source,
        enable_reg_head=True,  # 关键：必须开启回归头，和训练一致
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = defaultdict(lambda: {"dice": [], "iou": [], "hd95": []})
    ef_dict = {}
    idx_counter = {"SAX": 1, "2CH": 1, "4CH": 1}

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            image_tensor = item["image"].unsqueeze(0).to(device)
            mask_gt = item["mask"].to(device)
            source = item["source"]
            raw_img_path = item["image_path"]

            # ========== 分支推理区分：SAX走带回归的forward ==========
            if source == "sa":
                logits, reg_pred = model.forward_source_reg(image_tensor, source)
                # 训练时LVEF标签除以100归一化，推理乘100恢复百分比
                ef_percent = float(reg_pred.squeeze().detach().cpu().item()) * 100
            else:
                logits = model.forward_source(image_tensor, source)
                ef_percent = None

            # 分割后处理不变
            logits = F.interpolate(logits, size=mask_gt.shape, mode="trilinear", align_corners=False)
            pred_3d = torch.argmax(logits, dim=1).squeeze(0)
            pred_3d_np = pred_3d.cpu().numpy().astype(np.uint8)

            # 读取原始影像shape与affine
            raw_nii = nib.load(raw_img_path)
            raw_shape = raw_nii.shape
            raw_affine = raw_nii.affine
            print(f"\n[{idx}] 原始文件: {raw_img_path}")
            print(f"模型输出尺寸: {pred_3d_np.shape} | 原图尺寸: {raw_shape}")

            # 缩放预测图至原图分辨率
            pred_raw_np = scale_label_array(pred_3d_np, raw_shape)
            print(f"缩放后预测尺寸: {pred_raw_np.shape}")

            # 计算分割指标
            dice, iou, hd95 = multiclass_dice_iou_hd95(pred_3d, mask_gt, num_classes_by_source[source])
            metrics[source]["dice"].append(dice)
            metrics[source]["iou"].append(iou)
            metrics[source]["hd95"].append(hd95)

            # 保存原图尺寸预测nii（复用原始affine）
            view_name = view_map[source]
            cur_num = idx_counter[view_name]
            base_key = f"CINE_{view_name}_{cur_num:03d}"
            save_path = os.path.join(args.output_root, view_name, f"{base_key}.nii.gz")
            out_nii = nib.Nifti1Image(pred_raw_np, raw_affine)
            nib.save(out_nii, save_path)

            # ========== 填充EF预测字典：SAX存模型回归输出，其余None ==========
            if source == "sa":
                ef_dict[base_key] = round(ef_percent, 2)
                print(f"✅ {base_key} 模型回归预测 EF={ef_percent:.2f}%")
            else:
                ef_dict[base_key] = None
                print(f"  ⏭️ {source}视图无回归头，跳过EF")

            idx_counter[view_name] += 1

    # 导出EF预测json
    with open(ef_json_save_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(ef_dict), f, ensure_ascii=False, indent=2)
    print(f"\n✅ 模型回归预测EF结果已保存至 {ef_json_save_path}")

    # 打印所有视图分割指标汇总
    print("\n===== 分割指标汇总 =====")
    for src in args.source_order:
        avg_dice = np.mean(metrics[src]["dice"])
        avg_iou = np.mean(metrics[src]["iou"])
        avg_hd95 = np.mean(metrics[src]["hd95"])
        print(f"{src:4s} | Dice:{avg_dice:.4f} IoU:{avg_iou:.4f} HD95:{avg_hd95:.4f}")

if __name__ == "__main__":
    main()