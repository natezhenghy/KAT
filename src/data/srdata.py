import glob
import os
from argparse import Namespace
from typing import Any

import imageio
import torch.utils.data as data

from data import common


class SRData(data.Dataset): # type: ignore
    def __init__(self, args: Namespace, name: str='', train: bool=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale
        
        self._set_filesystem(args.dir_data)

        self.images_hr, self.images_lr = self._scan()

        self.repeat = 1
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            self.repeat = max(n_patches // n_images, 1)

    def _set_filesystem(self, dir_data: str):
        if self.train:
            self.dirs_hr = [os.path.join(dir_data, 'HR', i) for i in self.name.split('+')]
            self.dirs_lr = [os.path.join(dir_data, 'X', i) for i in self.name.split('+')]
        else:
            self.dir_hr = os.path.join(dir_data, 'HR', self.name)
            self.dir_lr = os.path.join(dir_data, 'X', self.name)
        self.ext = ('', '.png')

    def _scan(self):
        if self.train:
            names_hr = []
            for dir_hr in self.dirs_hr:
                names_hr.extend(glob.glob(os.path.join(dir_hr, '*')))
        else:
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*'))
            )
        names_lr: list[str] = []
        for f in names_hr:
            names_lr.append(os.path.splitext(f.replace('HR', f'X{self.scale}'))[0] + '.png')
        return names_hr, names_lr


    def __getitem__(self, idx: int):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=3)
        pair_t = common.np2tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename


    def _load_file(self, idx: int):
        idx = idx % len(self.images_hr)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr: Any = imageio.imread(f_hr)
        lr: Any = imageio.imread(f_lr)

        return lr, hr, filename

    def get_patch(self, lr: Any, hr: Any):
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=self.scale
            )
            lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * self.scale, 0:iw * self.scale]

        return lr, hr

    def __len__(self):
        return len(self.images_hr) * self.repeat
