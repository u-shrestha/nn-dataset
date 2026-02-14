import torch
import numpy as np


def batch_transform(inputs, labels, alpha=1.0, augment=None, probs=None):
    """
    Randomly applies augmentation based on provided augment and probabilities
    """
    augment= np.random.choice(augment, p=probs)
    if augment == 'cutmix':
        return apply_cutmix(inputs, labels, alpha)
    elif augment == 'mixup':
        return apply_mixup(inputs, labels, alpha)
    elif augment == 'cutout':
        return apply_cutout(inputs, labels)
    else:
        return inputs, labels, labels, 1.0




def apply_cutmix(inputs, labels, alpha):

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size()[0]).to(inputs.device)
    
    target_a = labels
    target_b = labels[rand_index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    
    # Paste patches
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exact pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    
    return inputs, target_a, target_b, lam



def apply_mixup(inputs, labels, alpha):

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size()[0]).to(inputs.device)
    
    target_a = labels
    target_b = labels[rand_index]
    
    # Global weighted average
    inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
    
    return inputs, target_a, target_b, lam



def apply_cutout(inputs, labels, length=16):
    """
    Zeros out a random NxN square in each image
    """
    h, w = inputs.shape[2], inputs.shape[3]
    batch_size = inputs.shape[0]
    
    # Generate random centers for each image in the batch
    y_centers = torch.randint(0, h, (batch_size,), device=inputs.device)
    x_centers = torch.randint(0, w, (batch_size,), device=inputs.device)
    

    for i in range(batch_size):
        y = y_centers[i].item()
        x = x_centers[i].item()
        
        y1 = max(0, y - length // 2)
        y2 = min(h, y + length // 2)
        x1 = max(0, x - length // 2)
        x2 = min(w, x + length // 2)
      
        inputs[i, :, y1:y2, x1:x2] = 0.0
    

    # lam=1.0 means original label, target_b is a copy
    return inputs, labels, labels, 1.0



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = (1. - lam)** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2