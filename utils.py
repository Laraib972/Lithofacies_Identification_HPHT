import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# Import the LTN loss function from your separated LTN script
from ltn_module import ltn_canonical_loss

# ==========================================
# 1. Reproducibility Setup
# ==========================================
def set_seed(seed=42):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. PyTorch Datasets
# ==========================================
class LithoDatasetTrain(Dataset):
    """Dataset with environment tags (for training with LTN)."""
    def __init__(self, x, y, env):
        if torch.is_tensor(x):
            self.x = x
        else:
            self.x = torch.tensor(x, dtype=torch.float32)

        if torch.is_tensor(y):
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

        if torch.is_tensor(env):
            self.env = env.long()
        else:
            self.env = torch.tensor(env, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.env[idx]


class LithoDatasetVal(Dataset):
    """Dataset without environment tags (for validation/testing)."""
    def __init__(self, x, y):
        if torch.is_tensor(x):
            self.x = x
        else:
            self.x = torch.tensor(x, dtype=torch.float32)

        if torch.is_tensor(y):
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==========================================
# 3. Loss Functions
# ==========================================
def dual_focal_loss(logits, targets, alpha=0.35, gamma=2.77, reduction='mean', eps=1e-9):
    """
    Computes the Dual Focal Loss for handling class imbalances in lithofacies.
    """
    probs = torch.softmax(logits, dim=-1)
    ce_loss = -(targets * torch.log(probs + eps)).sum(dim=-1)
    pt = (probs * targets).sum(dim=-1)
    focal_weight = alpha * ((1 - pt) ** gamma)
    loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def total_loss_function(logits, y_true, env_labels, class_names, rules, lambda_ltn=0.1, p_agg=2.0):
    """
    Combines the Dual Focal Loss with the Canonical LTN Loss penalty.
    """
    # 1. Base Data-Driven Loss
    focal = dual_focal_loss(logits, y_true)

    # Optimization: Skip LTN computation entirely if lambda is 0 (Baseline models)
    if lambda_ltn == 0.0:
        return focal

    # 2. Logic Tensor Network Penalty
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C) 
    env_flat = env_labels.reshape(B * T, -1) 

    # Calculate canonical LTN loss utilizing the rules and real logic
    ltn_penalty = ltn_canonical_loss(logits_flat, env_flat, class_names, rules, p_agg=p_agg)

    # Combine losses
    return focal + (lambda_ltn * ltn_penalty)