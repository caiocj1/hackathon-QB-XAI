import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn

class HackathonModel(LightningModule):

    def __init__(self):
        super(HackathonModel, self).__init__()
        self.build_model()

    def build_model(self):
        self.downsample1 = nn.Sequential(
            #input
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            #c1
        )
        self.downsample2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            #c2
        )
        self.downsample3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            #c3
        )
        self.downsample4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
            #c4
        )

        self.downsample5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
            # c5
        )

        # self.downsample6 = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU()
        #     # c6
        # )

        self.flatten = nn.Flatten()

        # self.conv_transp1 = nn.ConvTranspose2d(128, 64, 2, stride=(2, 2))
        # self.upsample1 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU()
        # )

        # self.conv_transp2 = nn.ConvTranspose2d(64, 32, 2, stride=(2, 2))
        # self.upsample2 = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.ReLU()
        # )
        #
        # self.conv_transp3 = nn.ConvTranspose2d(32, 16, 2, stride=(2, 2))
        # self.upsample3 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU()
        # )
        #
        # self.conv_transp4 = nn.ConvTranspose2d(16, 8, 2, stride=(2, 2))
        # self.upsample4 = nn.Sequential(
        #     nn.Conv2d(16, 8, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 8, 3, padding=1),
        #     nn.ReLU()
        # )
        #
        # self.final = nn.Sequential(
        #     nn.Conv2d(8, 1, 1),
        #     nn.Sigmoid(),
        #     nn.Flatten()
        # )

        self.linear = nn.Linear(32768, 128)
        self.relu = nn.ReLU()
        self.head = nn.Linear(128, 1)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def _shared_step(self, batch):
        output = self.forward(batch)

        loss = self.calc_loss(output, batch['class'])

        metrics = self.calc_metrics(output, batch['class'])

        return loss, metrics

    def forward(self, batch):

        c1 = self.downsample1(torch.transpose(batch['img'], -1, 1))
        c2 = self.downsample2(c1)
        c3 = self.downsample3(c2)
        c4 = self.downsample4(c3)
        c5 = self.downsample5(c4)

        # u6 = self.conv_transp1(c5)
        # u6 = torch.cat([u6, c4], dim=1)
        # c6 = self.upsample1(u6)
        #
        # u7 = self.conv_transp2(c6)
        # u7 = torch.cat([u7, c3], dim=1)
        # c7 = self.upsample2(u7)
        #
        # u8 = self.conv_transp3(c7)
        # u8 = torch.cat([u8, c2], dim=1)
        # c8 = self.upsample3(u8)
        #
        # u9 = self.conv_transp4(c8)
        # u9 = torch.cat([u9, c1], dim=1)
        # c9 = self.upsample4(u9)
        #
        encoding = self.flatten(c5)

        output = self.head(self.relu(self.linear(encoding)))
        output = torch.squeeze(output)

        return output

    def calc_loss(self, prediction, target):
        ce_loss = nn.BCELoss(reduction='none')
        m = nn.Sigmoid()

        loss = ce_loss(m(prediction), target.float())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        opt = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

        return opt

    def calc_metrics(self, prediction, target):
        metrics = {}

        prediction = prediction > 0
        batch_size = len(prediction)

        metrics['accuracy'] = (prediction == target).sum() / batch_size

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
