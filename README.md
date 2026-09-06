# MICCAI2026_CMR-Segmentation

## Training Commands

### Task 1
```bash
cd 3D

python train.py \
  --train-root ../2D_seg/CMR-MULTI/CINE_MULTI/train \
  --val-root ../2D_seg/CMR-MULTI/CINE_MULTI/val \
    --train-lvef-xlsx ../2D_seg/CMR-MULTI/CINE_MULTI/dataset_train.xlsx \
    --val-lvef-xlsx ../2D_seg/CMR-MULTI/CINE_MULTI/dataset_valid.xlsx \
  --output-dir checkpoints_3d_12 \
  --source-order 2ch 4ch sa \
  --image-dirname image \
  --label-dirname anno \
  --num-classes-json '{"2ch":3,"4ch":6,"sa":4}' \
  --excel-case-id-col patient_id \
  --excel-lvef-col LVEF \
  --input-size 144 144 144 \
  --batch-size 1 \
  --epochs 100 \
  --lr 1e-4 \
  --reg-weight 1

```

### Task 2
```bash
cd ../2D
python train.py \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-5 \
  --image_size 448 \
  --num_workers 4 \
  --device cuda:0 \
  --use_weighted_sampler \
  --checkpoint_dir checkpoints


```

## Citation
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qi2026hybridsupervised,
  title={Multi-View CMR Segmentation with Clinical Quantification for Cine and LGE: Vista3D and DINOv2-DPT},
  author={Yue Qi and Junxuan Yu and Zheng Yu and Li Yang},
  booktitle={The 1st MICCAI Workshop on Medical World Models},
  year={2026},
  url={https://openreview.net/forum?id=zi2K25F8zI}
}
