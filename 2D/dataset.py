# dataset.py
# NIfTI-based dataset loader for multi-dataset cardiac MRI segmentation.
#
# Each dataset directory is expected to contain:
#   img_resized/   ← *.nii.gz  (3D image volumes)
#   seg_resized/   ← *.nii.gz  (3D label volumes, same filename)
#
# For each volume the **center slice ± 1** is extracted and stacked into a
# 3-channel pseudo-RGB image.  Masks are taken from the **center** slice only.

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, Dict, List


# ─── Helper utilities ────────────────────────────────────────────────────────
def _load_volume(path: str) -> np.ndarray:
    """Load a .nii.gz file and return numpy array (H, W, D) or (H, W, D, ...)."""
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


def _minmax_norm_slice(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D slice to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        return (arr - lo) / (hi - lo)
    return arr - lo          # all-zero if constant


def _extract_pseudo_rgb_at_slice(vol: np.ndarray, center_idx: int) -> np.ndarray:
    """Build pseudo-RGB from [idx-1, idx, idx+1] around target slice."""
    if vol.ndim == 4:
        vol = vol[..., 0]
    depth = vol.shape[-1]
    idx = [max(0, center_idx - 1), center_idx, min(depth - 1, center_idx + 1)]
    channels = []
    for i in idx:
        channels.append(_minmax_norm_slice(vol[..., i]))
    return np.stack(channels, axis=0)


def _extract_mask_at_slice(seg_vol: np.ndarray, center_idx: int) -> np.ndarray:
    if seg_vol.ndim == 4:
        seg_vol = seg_vol[..., 0]
    return seg_vol[..., center_idx].astype(np.int64)


def _resize_pair(img_t: torch.Tensor, mask_t: torch.Tensor, image_size: int):
    img_t = F.interpolate(
        img_t.unsqueeze(0),
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    mask_t = F.interpolate(
        mask_t.unsqueeze(0).unsqueeze(0).float(),
        size=(image_size, image_size),
        mode='nearest',
    ).squeeze(0).squeeze(0).long()
    return img_t, mask_t


class NIfTITestDataset(Dataset):
    """纯推理数据集：只有image，无anno掩码"""
    def __init__(
        self,
        root_dir: str,
        dataset_type: str,
        image_size: int = 256,
        use_all_slices: bool = True,
        cache_size: int = 4,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.use_all_slices = use_all_slices
        self.cache_size = int(max(cache_size, 1))
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_keys: List[str] = []

        img_dir = os.path.join(root_dir, 'image')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
        if not img_files:
            raise FileNotFoundError(f"No image nii.gz in {img_dir}")
        self.img_paths = img_files

        self.samples = []
        self._build_test_samples()

    def _get_cached_volume(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        arr = _load_volume(path)
        self._cache[path] = arr
        self._cache_keys.append(path)
        if len(self._cache_keys) > self.cache_size:
            self._cache.pop(self._cache_keys.pop(0))
        return arr

    def _build_test_samples(self):
        for vol_idx, img_path in enumerate(self.img_paths):
            img_vol = _load_volume(img_path)
            depth = img_vol.shape[-1]
            slice_indices = list(range(depth)) if self.use_all_slices else [depth // 2]
            for s in slice_indices:
                self.samples.append((vol_idx, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.samples[idx]
        img_path = self.img_paths[vol_idx]
        img_vol = self._get_cached_volume(img_path)
        img_arr = _extract_pseudo_rgb_at_slice(img_vol, slice_idx)
        img_t = torch.from_numpy(img_arr).float()
        img_t, _ = _resize_pair(img_t, torch.zeros_like(img_t[0:1]), self.image_size)
        return img_t, self.dataset_type, img_path, slice_idx


# ─── Dataset class ───────────────────────────────────────────────────────────
class NIfTISegDataset(Dataset):
    """
    Loads paired (image, mask) samples from a NIfTI dataset directory.

    Args:
        root_dir     : directory containing img_resized/ and seg_resized/
        dataset_type : key in config.DATASET_CONFIGS
        image_size   : target (H, W) after resizing (both image and mask)
        volume_pairs : optional list of (img_path, seg_path) after split
    """

    def __init__(
        self,
        root_dir: str,
        dataset_type: str,
        image_size: int = 256,
        volume_pairs: Optional[List[Tuple[str, str]]] = None,
        use_all_slices: bool = True,
        rare_classes: Optional[List[int]] = None,
        rare_boost: float = 4.0,
        foreground_boost: float = 1.5,
        cache_size: int = 4,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.use_all_slices = use_all_slices
        self.rare_classes = set(rare_classes or [])
        self.rare_boost = float(rare_boost)
        self.foreground_boost = float(foreground_boost)
        self.cache_size = int(max(cache_size, 1))
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_keys: List[str] = []

        img_dir = os.path.join(root_dir, 'image')
        seg_dir = os.path.join(root_dir, 'anno')

        if volume_pairs is None:
            img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
            if not img_files:
                raise FileNotFoundError(f"No .nii.gz files found in '{img_dir}'.")

            pairs = []
            for img_path in img_files:
                fname = os.path.basename(img_path)
                seg_path = os.path.join(seg_dir, fname)
                if os.path.isfile(seg_path):
                    pairs.append((img_path, seg_path))
            if not pairs:
                raise FileNotFoundError(f"No paired files found in '{root_dir}'.")
            self.volume_pairs = pairs
        else:
            self.volume_pairs = volume_pairs

        self.samples: List[Tuple[int, int]] = []
        self.sample_weights: List[float] = []
        self._build_samples()

    def _get_cached_volume(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        arr = _load_volume(path)
        self._cache[path] = arr
        self._cache_keys.append(path)
        if len(self._cache_keys) > self.cache_size:
            old = self._cache_keys.pop(0)
            self._cache.pop(old, None)
        return arr

    def _build_samples(self):
        for vol_idx, (_, seg_path) in enumerate(self.volume_pairs):
            seg_vol = _load_volume(seg_path)
            depth = seg_vol.shape[-1]
            slice_indices = list(range(depth)) if self.use_all_slices else [depth // 2]

            for slice_idx in slice_indices:
                mask = _extract_mask_at_slice(seg_vol, slice_idx)
                classes_present = set(np.unique(mask).astype(int).tolist())
                has_foreground = any(c > 0 for c in classes_present)
                rare_hits = sum(1 for class_id in self.rare_classes if class_id in classes_present)

                weight = 1.0
                if has_foreground:
                    weight *= self.foreground_boost
                if rare_hits > 0:
                    weight *= (1.0 + self.rare_boost * rare_hits)

                self.samples.append((vol_idx, slice_idx))
                self.sample_weights.append(weight)

        if not self.samples:
            raise RuntimeError(f"No slice samples built for dataset '{self.dataset_type}'.")
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        vol_idx, slice_idx = self.samples[idx]
        img_path, seg_path = self.volume_pairs[vol_idx]

        img_vol = self._get_cached_volume(img_path)
        seg_vol = self._get_cached_volume(seg_path)

        img_arr = _extract_pseudo_rgb_at_slice(img_vol, slice_idx)
        mask_arr = _extract_mask_at_slice(seg_vol, slice_idx)

        img_t = torch.from_numpy(img_arr).float()
        mask_t = torch.from_numpy(mask_arr).long()
        img_t, mask_t = _resize_pair(img_t, mask_t, self.image_size)
        return img_t, mask_t, self.dataset_type

def create_standalone_dataset(
    root_dir: str,
    dataset_type: str,
    image_size: int = 256,
    use_all_slices: bool = True,
    rare_classes: Optional[List[int]] = None,
    rare_boost: float = 4.0,
    foreground_boost: float = 1.5,
) -> NIfTISegDataset:
    """
    直接读取已有划分好的文件夹（train文件夹 / val文件夹单独调用）
    目录结构：root_dir/img_resized、root_dir/seg_resized
    """
    # 自动读取文件夹内所有配对nii，不再切分
    return NIfTISegDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        image_size=image_size,
        volume_pairs=None,
        use_all_slices=use_all_slices,
        rare_classes=rare_classes,
        rare_boost=rare_boost,
        foreground_boost=foreground_boost,
    )


# ─── Split helpers ───────────────────────────────────────────────────────────
def make_split_datasets(
    root_dir: str,
    dataset_type: str,
    val_split: float = 0.2,
    image_size: int = 256,
    seed: int = 42,
    use_all_slices: bool = True,
    rare_classes: Optional[List[int]] = None,
    rare_boost: float = 4.0,
    foreground_boost: float = 1.5,
) -> Tuple[NIfTISegDataset, NIfTISegDataset]:
    """
    Create train/val datasets from a single root directory using a random split.

    Returns:
        (train_dataset, val_dataset)
    """
    img_dir = os.path.join(root_dir, 'image')
    seg_dir = os.path.join(root_dir, 'anno')

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    pairs = []
    for img_path in img_files:
        fname = os.path.basename(img_path)
        seg_path = os.path.join(seg_dir, fname)
        if os.path.isfile(seg_path):
            pairs.append((img_path, seg_path))
    if not pairs:
        raise FileNotFoundError(f"No paired files found in '{root_dir}'.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()
    n_val = max(1, int(len(pairs) * val_split))
    val_idx = set(indices[:n_val])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    train_ds = NIfTISegDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        image_size=image_size,
        volume_pairs=train_pairs,
        use_all_slices=use_all_slices,
        rare_classes=rare_classes,
        rare_boost=rare_boost,
        foreground_boost=foreground_boost,
    )
    val_ds = NIfTISegDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        image_size=image_size,
        volume_pairs=val_pairs,
        use_all_slices=use_all_slices,
        rare_classes=rare_classes,
        rare_boost=rare_boost,
        foreground_boost=foreground_boost,
    )
    return train_ds, val_ds


# ─── Multi-dataset DataLoader factory ────────────────────────────────────────

class MultiDatasetLoader_1:
    def __init__(
        self,
        dataset_configs: dict,
        split: str = 'train',
        batch_size: int = 4,
        image_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        use_weighted_sampler: bool = True,
    ):
        self.loaders   = {}
        self.iterators = {}
        self.split     = split
        is_train       = (split == 'train')

        for dtype, cfg in dataset_configs.items():
            sampler_cfg = cfg.get('sampler', {})
            rare_classes = sampler_cfg.get('rare_classes', [])
            rare_boost = sampler_cfg.get('rare_boost', 4.0)
            foreground_boost = sampler_cfg.get('foreground_boost', 1.5)
            use_all_slices = sampler_cfg.get('use_all_slices', True)

            # 直接读取现成划分好的文件夹，不切分
            data_root = cfg["data_root"] if is_train else cfg["data_root_val"]
            ds = create_standalone_dataset(
                root_dir=data_root,
                dataset_type=dtype,
                image_size=image_size,
                use_all_slices=use_all_slices,
                rare_classes=rare_classes,
                rare_boost=rare_boost,
                foreground_boost=foreground_boost,
            )

            sampler = None
            shuffle = is_train
            if is_train and use_weighted_sampler:
                sampler = WeightedRandomSampler(
                    weights=ds.sample_weights,
                    num_samples=len(ds),
                    replacement=True,
                )
                shuffle = False

            self.loaders[dtype] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=is_train,
            )

        self.lengths = {k: len(v) for k, v in self.loaders.items()}
        self.max_len = max(self.lengths.values()) if self.lengths else 0
        self.dtype_order = list(dataset_configs.keys())

    # __len__ / __iter__ / __next__ 原有代码不变
    def __len__(self) -> int:
        return self.max_len * len(self.dtype_order)

    def __iter__(self):
        self.iterators = {k: iter(v) for k, v in self.loaders.items()}
        self._step = 0
        return self

    def __next__(self):
        if self._step >= len(self):
            raise StopIteration
        dtype = self.dtype_order[self._step % len(self.dtype_order)]
        self._step += 1
        try:
            batch = next(self.iterators[dtype])
        except StopIteration:
            self.iterators[dtype] = iter(self.loaders[dtype])
            batch = next(self.iterators[dtype])
        images, masks, _ = batch
        return images, masks, dtype

        
class MultiDatasetLoader:
    """
    Wraps multiple per-dataset DataLoaders and returns batches in round-robin
    order.  Each batch is (images, masks, dataset_type_string).

    Usage:
        loader = MultiDatasetLoader(configs, split='train', batch_size=4)
        for images, masks, dtype in loader:
            logits = model(images, dtype)
    """

    def __init__(
        self,
        dataset_configs: dict,
        split: str = 'train',
        batch_size: int = 4,
        val_split: float = 0.2,
        image_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        use_weighted_sampler: bool = True,
    ):
        self.loaders   = {}
        self.iterators = {}
        self.split     = split
        is_train       = (split == 'train')

        for dtype, cfg in dataset_configs.items():
            sampler_cfg = cfg.get('sampler', {})
            rare_classes = sampler_cfg.get('rare_classes', [])
            rare_boost = sampler_cfg.get('rare_boost', 4.0)
            foreground_boost = sampler_cfg.get('foreground_boost', 1.5)
            use_all_slices = sampler_cfg.get('use_all_slices', True)

            train_ds, val_ds = make_split_datasets(
                cfg['data_root'], dtype,
                val_split=val_split,
                image_size=image_size,
                seed=seed,
                use_all_slices=use_all_slices,
                rare_classes=rare_classes,
                rare_boost=rare_boost,
                foreground_boost=foreground_boost,
            )
            ds = train_ds if is_train else val_ds

            sampler = None
            shuffle = is_train
            if is_train and use_weighted_sampler:
                sampler = WeightedRandomSampler(
                    weights=ds.sample_weights,
                    num_samples=len(ds),
                    replacement=True,
                )
                shuffle = False

            self.loaders[dtype] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=is_train,
            )

        # Number of batches = max across all loaders (larger datasets cycle round)
        self.lengths = {k: len(v) for k, v in self.loaders.items()}
        self.max_len = max(self.lengths.values()) if self.lengths else 0
        self.dtype_order = list(dataset_configs.keys())

    def __len__(self) -> int:
        """Total number of (dtype, batch) pairs in one epoch."""
        return self.max_len * len(self.dtype_order)

    def __iter__(self):
        self.iterators = {k: iter(v) for k, v in self.loaders.items()}
        self._step = 0
        return self

    def __next__(self):
        if self._step >= len(self):
            raise StopIteration
        # Round-robin over dataset types
        dtype = self.dtype_order[self._step % len(self.dtype_order)]
        self._step += 1
        try:
            batch = next(self.iterators[dtype])
        except StopIteration:
            # Re-start exhausted iterator
            self.iterators[dtype] = iter(self.loaders[dtype])
            batch = next(self.iterators[dtype])
        images, masks, _ = batch
        return images, masks, dtype
