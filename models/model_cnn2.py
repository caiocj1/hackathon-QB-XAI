import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn

class HackathonModel(LightningModule):

    def __init__(self):
        super(HackathonModel, self).__init__()
        self.build_model()

    def build_model(self):
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(5),

            nn.Conv2d(96, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(9216, 128)
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

        encoding = self.layers(torch.transpose(batch['img'], -1, 1))

        encoding = self.flatten(encoding)

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
