import random
from typing import Any

import numpy as np
import skimage.color as sc
import torch


def get_patch(*args: Any, patch_size: int=96, scale: int=2):
    ih, iw = args[0].shape[:2]

    ip = patch_size // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + patch_size, tx:tx + patch_size, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args: Any, n_channels: int=3):
    def _set_channel(img: Any):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2tensor(*args: Any, rgb_range: int=255):
    def _np2tensor(img: Any):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2tensor(a) for a in args]

def augment(*args: Any, flip: bool=True, rot: bool=True):
    flip = flip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img: Any):
        if flip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

