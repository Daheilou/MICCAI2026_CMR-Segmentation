import torch
from typing import Dict, List


def _score_from_terms(prob_maps: torch.Tensor, terms: List[dict]) -> torch.Tensor:
    score = torch.zeros_like(prob_maps[:, 0])
    for term in terms:
        channel = int(term['channel'])
        weight = float(term.get('weight', 1.0))
        score = score + weight * prob_maps[:, channel]
    return score


def decode_with_rules(fg_logits: torch.Tensor, rules: List[dict]) -> torch.Tensor:
    """
    Convert independent foreground logits into final multi-class labels via
    weighted-threshold post-processing rules.

    Args:
        fg_logits: (B, K, H, W), K = num_fg_classes = num_classes - 1.
        rules: list of dicts, each item format:
            {
              'class_id': int,              # final class id in mask space
              'threshold': float,
              'terms': [                    # weighted score terms
                 {'channel': int, 'weight': float},
                 ...
              ],
              'priority': int               # lower first; later rules can override
            }

    Returns:
        pred: (B, H, W) int64 class map.
    """
    prob_maps = torch.sigmoid(fg_logits)
    pred = torch.zeros(
        (prob_maps.shape[0], prob_maps.shape[2], prob_maps.shape[3]),
        dtype=torch.long,
        device=prob_maps.device,
    )

    ordered = sorted(rules, key=lambda x: int(x.get('priority', 999)))
    for rule in ordered:
        class_id = int(rule['class_id'])
        threshold = float(rule['threshold'])
        terms = rule.get('terms', [])
        if not terms:
            continue
        score = _score_from_terms(prob_maps, terms)
        pred = torch.where(score > threshold, torch.full_like(pred, class_id), pred)

    return pred


def masks_to_one_vs_rest(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer mask (B,H,W) to foreground one-vs-rest targets (B,K,H,W)
    where K = num_classes - 1 and channel k corresponds to class_id=k+1.
    """
    targets = []
    for class_id in range(1, num_classes):
        targets.append((mask == class_id).float())
    if not targets:
        return torch.zeros(mask.shape[0], 0, mask.shape[1], mask.shape[2], device=mask.device)
    return torch.stack(targets, dim=1)
