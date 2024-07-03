import torch
import random

seed = 42
torch.manual_seed(seed)
random.seed(seed)


def pixel_level(input, label, mask_ratio):
    mask_rand = torch.rand(input.shape)
    mask = (mask_rand < mask_ratio) * (label > 0)

    return mask > 0


def patch_level(mask_ratio):
    H, W = 30, 60
    patch_h = 5
    patch_w = 5
    num_patches = (H // patch_h) * (W // patch_w)
    mask = torch.zeros((H, W)).view(H // patch_h, patch_h, W // patch_w, patch_w).permute(0, 2, 1, 3).reshape(
        num_patches, -1)

    num_masked = int(mask_ratio * num_patches)
    shuffle_indices = torch.rand(num_patches).argsort()
    mask_ind = shuffle_indices[:num_masked]
    mask[mask_ind] = 1
    mask = mask.reshape(H // patch_h, W // patch_w, patch_h, patch_w).permute(0, 2, 1, 3).reshape(H,
                                                                                                  W)
    return mask[:29, :58] > 0
