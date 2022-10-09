import argparse

from model import HackathonModel
from dataset import HackathonDataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')

    args = parser.parse_args()

    # CHANGE DATASET CLASS
    train_dataset = HackathonDataset(type='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  num_workers=4,
                                  shuffle=True)

    val_dataset = HackathonDataset(type='validation')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=32,
                                num_workers=4,
                                shuffle=False)

    # CHANGE MODEL
    model = HackathonModel()

    # CHANGE WHICH METRIC IS IMPORTANT TO SAVE CHECKPOINTS
    logger = TensorBoardLogger('.', version=args.version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}/checkpoints',
                                 save_top_k=1,
                                 monitor='iou_val',
                                 mode='max')
    lr_monitor = LearningRateMonitor()

    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      max_epochs=-1,
                      val_check_interval=30,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
