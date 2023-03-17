from argparse import Namespace
from typing import Any

import torch
from tqdm import tqdm

import utility


class Trainer():
    def __init__(self, args: Namespace, loader: Any, model: Any, loss: Any, checkpoint: Any):
        self.args = args
        self.scale = args.scale

        self.checkpoint = checkpoint
        self.loader_test = loader.loader_test
        self.model = model

        self.error_last = 1e8

    def test(self):
        torch.set_grad_enabled(False)

        self.checkpoint.write_log('\nEvaluation:')
        self.checkpoint.add_log(
            torch.zeros(1, len(self.loader_test), 1)
        )
        self.model.eval()

        if self.args.save_results:
            self.checkpoint.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for lr, hr, filename in tqdm(d, ncols=80):
                lr, hr = lr.to('cuda'), hr.to('cuda')
                sr = self.model(lr)
                sr = utility.quantize(sr, self.args.rgb_range)

                save_list = [sr]
                self.checkpoint.log[-1, idx_data, 0] += utility.calc_psnr(
                    sr, hr, self.scale, self.args.rgb_range, dataset=d
                )

                if self.args.save_results:
                    self.checkpoint.save_results(d, filename[0], save_list, self.scale)

            self.checkpoint.log[-1, idx_data, 0] /= len(d)
            best = self.checkpoint.log.max(0)
            self.checkpoint.write_log(
                '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                    d.dataset.name,
                    self.scale,
                    self.checkpoint.log[-1, idx_data, 0],
                    best[0][idx_data, 0],
                    best[1][idx_data, 0] + 1
                )
            )

        if self.args.save_results:
            self.checkpoint.end_background()

        torch.set_grad_enabled(True)
