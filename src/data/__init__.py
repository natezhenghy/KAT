from argparse import Namespace

from torch.utils.data import dataloader

from data.srdata import SRData


class Loader:
    def __init__(self, args: Namespace):
        self.loader_train = None
        if not args.test_only:
            self.loader_train = dataloader.DataLoader(
                SRData(args, name=args.data_train),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            testset = SRData(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.n_threads,
                )
            )
