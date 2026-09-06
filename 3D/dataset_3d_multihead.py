import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError as error:
    raise ImportError(
        "nibabel is required for NIfTI loading. Please install via `pip install nibabel`."
    ) from error


NII_EXTENSIONS = (".nii", ".nii.gz")


@dataclass
class CaseItem:
    source: str
    source_id: int
    image_path: str
    label_path: str


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _load_nii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine
    return data, affine


def _normalize_image(image: np.ndarray) -> np.ndarray:
    non_zero = image[np.abs(image) > 1e-8]
    if non_zero.size == 0:
        return image
    mean = non_zero.mean()
    std = non_zero.std()
    if std < 1e-8:
        return image - mean
    return (image - mean) / std


def _resize_volume(image: np.ndarray, output_size: Sequence[int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    if mode == "trilinear":
        resized = F.interpolate(tensor, size=tuple(output_size), mode=mode, align_corners=False)
    else:
        resized = F.interpolate(tensor, size=tuple(output_size), mode=mode)
    return resized.squeeze(0).squeeze(0).numpy()


def _random_flip_3d(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axes = [0, 1, 2]
    for axis in axes:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
    return image, label


def collect_cases(
    data_root: str,
    source_names: Sequence[str],
    image_dirname: str = "image",
    label_dirname: str = "seg",
) -> List[CaseItem]:
    cases: List[CaseItem] = []
    for source_id, source in enumerate(source_names):
        source_root = os.path.join(data_root, source)
        image_root = os.path.join(source_root, image_dirname)
        label_root = os.path.join(source_root, label_dirname)

        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not os.path.isdir(label_root):
            raise FileNotFoundError(f"Label directory not found: {label_root}")

        image_paths = []
        label_paths = []
        for extension in NII_EXTENSIONS:
            image_paths.extend(glob(os.path.join(image_root, "**", f"*{extension}"), recursive=True))
            label_paths.extend(glob(os.path.join(label_root, "**", f"*{extension}"), recursive=True))

        image_map = {_strip_nii_suffix(os.path.basename(path)): path for path in sorted(image_paths)}
        label_map = {_strip_nii_suffix(os.path.basename(path)): path for path in sorted(label_paths)}

        shared_names = sorted(set(image_map) & set(label_map))
        if not shared_names:
            raise RuntimeError(
                f"No paired NIfTI found for source={source}. "
                f"Checked image_dir={image_root}, label_dir={label_root}."
            )

        for name in shared_names:
            cases.append(
                CaseItem(
                    source=source,
                    source_id=source_id,
                    image_path=image_map[name],
                    label_path=label_map[name],
                )
            )
    return cases


class MultiSourceNiftiDataset3D(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseItem],
        output_size: Sequence[int] = (64, 160, 160),
        source_num_classes: Dict[str, int] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.cases = list(cases)
        self.output_size = tuple(output_size)
        self.source_num_classes = source_num_classes or {"2ch": 3, "4ch": 5, "sa": 4}
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        item = self.cases[index]
        image, _ = _load_nii(item.image_path)
        label, affine = _load_nii(item.label_path)

        image = _normalize_image(image)
        label = np.rint(label).astype(np.int64)

        expected_classes = self.source_num_classes.get(item.source)
        if expected_classes is None:
            raise KeyError(f"Missing class count for source: {item.source}")

        valid_min = int(label.min())
        valid_max = int(label.max())
        if valid_min < 0 or valid_max >= expected_classes:
            raise ValueError(
                f"Label range out of bounds for source={item.source}. "
                f"Got [{valid_min}, {valid_max}] but expected [0, {expected_classes - 1}]"
            )

        image = _resize_volume(image, self.output_size, mode="trilinear")
        label = _resize_volume(label.astype(np.float32), self.output_size, mode="nearest").astype(np.int64)

        if self.augment:
            image, label = _random_flip_3d(image, label)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).long()

        return {
            "image": image_tensor,
            "mask": label_tensor,
            "source": item.source,
            "source_id": torch.tensor(item.source_id, dtype=torch.long),
            "image_path": item.image_path,
            "label_path": item.label_path,
            "affine": torch.from_numpy(affine).float(),
            "original_shape": torch.tensor(label.shape, dtype=torch.long),
        }


def split_cases_by_source(
    cases: Sequence[CaseItem],
    source_names: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[CaseItem], List[CaseItem]]:
    train_cases: List[CaseItem] = []
    val_cases: List[CaseItem] = []

    rng = np.random.default_rng(seed)
    for source in source_names:
        source_cases = [item for item in cases if item.source == source]
        rng.shuffle(source_cases)

        if len(source_cases) < 2:
            train_cases.extend(source_cases)
            continue

        split_index = max(1, int(round(len(source_cases) * (1.0 - val_ratio))))
        split_index = min(split_index, len(source_cases) - 1)

        train_cases.extend(source_cases[:split_index])
        val_cases.extend(source_cases[split_index:])

    return train_cases, val_cases
