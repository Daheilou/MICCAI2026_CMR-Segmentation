# config.py
# Dataset-specific configurations for multi-dataset cardiac MRI segmentation.

DATASET_CONFIGS = {
    '2ch': {
        'num_classes': 4,
        'class_labels': [0, 1, 2, 3],
        'data_root': './CMR-MULTI/LGE_MULTI/2CH_TR',
        'data_root_val': './CMR-MULTI/LGE_MULTI/2CH_VAL',
        'description': '2-chamber view (AZLGE), 4 classes: background + 3 structures',
        'sampler': {
            'use_all_slices': True,
            'rare_classes': [3, 2],
            'rare_boost': 5.0,
            'foreground_boost': 1.8,
        },
        'loss': {
            'pos_weight_fg': [2.0, 3.5, 6.5],
            'class_weights_fg': [1.0, 1.6, 2.4],
        },
        'postprocess_rules': [
            {'class_id': 1, 'threshold': 0.50, 'terms': [{'channel': 0, 'weight': 1.0}], 'priority': 1},
            {'class_id': 2, 'threshold': 0.52, 'terms': [{'channel': 0, 'weight': 0.25}, {'channel': 1, 'weight': 1.0}], 'priority': 2},
            {'class_id': 3, 'threshold': 0.48, 'terms': [{'channel': 1, 'weight': 0.20}, {'channel': 2, 'weight': 1.0}], 'priority': 3},
        ],
    },
    '4ch': {
        'num_classes': 5,
        'class_labels': [0, 1, 2, 3, 4],
        'data_root': './CMR-MULTI/LGE_MULTI/4CH_TR',
        'data_root_val': './CMR-MULTI/LGE_MULTI/4CH_VAL',
        'description': '4-chamber view (AZLGE), 5 classes: background + 4 structures',
        'sampler': {
            'use_all_slices': True,
            'rare_classes': [3, 4],
            'rare_boost': 4.0,
            'foreground_boost': 1.6,
        },
        'loss': {
            'pos_weight_fg': [1.8, 2.2, 3.2, 3.6],
            'class_weights_fg': [1.0, 1.2, 1.5, 1.7],
        },
        'postprocess_rules': [
            {'class_id': 1, 'threshold': 0.50, 'terms': [{'channel': 0, 'weight': 1.0}], 'priority': 1},
            {'class_id': 2, 'threshold': 0.54, 'terms': [{'channel': 0, 'weight': 0.25}, {'channel': 1, 'weight': 1.0}], 'priority': 2},
            {'class_id': 3, 'threshold': 0.52, 'terms': [{'channel': 1, 'weight': 0.25}, {'channel': 2, 'weight': 1.0}], 'priority': 3},
            {'class_id': 4, 'threshold': 0.52, 'terms': [{'channel': 2, 'weight': 0.25}, {'channel': 3, 'weight': 1.0}], 'priority': 4},
        ],
    },
    'sa': {
        'num_classes': 5,
        'class_labels': [0, 1, 2, 3, 4],
        'data_root': './CMR-MULTI/LGE_MULTI/SAX_TR',
        'data_root_val': './CMR-MULTI/LGE_MULTI/SAX_VAL',
        "excel_label_path_train": "./CMR-MULTI/LGE_MULTI/dataset_lge.xlsx",
        "excel_label_path_val": "./CMR-MULTI/LGE_MULTI/dataset_lge_valid.xlsx",
        'description': 'Short-axis view (AZLGE), 5 classes: background + 4 structures',
        'sampler': {
            'use_all_slices': True,
            'rare_classes': [3, 4],
            'rare_boost': 3.5,
            'foreground_boost': 1.5,
        },
        'loss': {
            'pos_weight_fg': [1.6, 2.0, 2.8, 3.0],
            'class_weights_fg': [1.0, 1.2, 1.4, 1.6],
        },
        'postprocess_rules': [
            {'class_id': 1, 'threshold': 0.50, 'terms': [{'channel': 0, 'weight': 1.0}], 'priority': 1},
            {'class_id': 2, 'threshold': 0.54, 'terms': [{'channel': 0, 'weight': 0.25}, {'channel': 1, 'weight': 1.0}], 'priority': 2},
            {'class_id': 3, 'threshold': 0.52, 'terms': [{'channel': 1, 'weight': 0.25}, {'channel': 2, 'weight': 1.0}], 'priority': 3},
            {'class_id': 4, 'threshold': 0.52, 'terms': [{'channel': 2, 'weight': 0.25}, {'channel': 3, 'weight': 1.0}], 'priority': 4},
        ],
    },
    'ras154': {
        'num_classes': 2,
        'class_labels': [0, 1],
        'data_root': './CMR-MULTI/LGE_MULTI/RAS_TR',
        'data_root_val': './CMR-MULTI/LGE_MULTI/RAS_VAL',
        'description': 'RAS-154 dataset, 2 classes: background + foreground',
        'sampler': {
            'use_all_slices': True,
            'rare_classes': [1],
            'rare_boost': 2.5,
            'foreground_boost': 1.4,
        },
        'loss': {
            'pos_weight_fg': [2.0],
            'class_weights_fg': [1.0],
        },
        'postprocess_rules': [
            {'class_id': 1, 'threshold': 0.50, 'terms': [{'channel': 0, 'weight': 1.0}], 'priority': 1},
        ],
    },
}

# ─── Training hyper-parameters (defaults, overridable via argparse) ───────────
DEFAULT_IMAGE_SIZE  = 448        # H × W after resize
DEFAULT_IN_CHANNELS = 3          # 3 consecutive slices → pseudo-RGB
DEFAULT_FILTERS     = [32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZE  = 4
DEFAULT_LR          = 1e-4
DEFAULT_EPOCHS      = 100
DEFAULT_VAL_SPLIT   = 0.2        # fraction of data used for validation
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
