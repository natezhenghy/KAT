import os
from argparse import Namespace

import torch
import torch.nn as nn

from models.kat import KAT


class Model(nn.Module):
    def __init__(self, args: Namespace):
        super(Model, self).__init__()

        self.n_GPUs = args.n_GPUs

        self.model = KAT(args).to('cuda')

        self.load(
            pre_train=args.pre_train
        )

    def forward(self, x: torch.Tensor): # type: ignore
        if self.training:
            return nn.parallel.data_parallel(self.model, x, range(self.n_GPUs))
        else:
            return self.model.forward(x)

    def save(self, apath: str, is_best: bool=False):
        torch.save(self.model.state_dict(), os.path.join(apath, 'model_latest.pt'))

        if is_best:
            torch.save(self.model.state_dict(), os.path.join(apath, 'model_best.pt'))

    def load(self, pre_train: str=''):
        if pre_train:
            load_from = torch.load(pre_train, map_location='cuda')
            self.model.load_state_dict(load_from, strict=False)
