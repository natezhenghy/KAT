import data
import models
import utility
from option import args
from trainer import Trainer


def main():
    checkpoint = utility.checkpoint(args)

    loader = data.Loader(args)
    model = models.Model(args)
    trainer = Trainer(args, loader, model, None, checkpoint)
    trainer.test()

if __name__ == '__main__':
    main()
