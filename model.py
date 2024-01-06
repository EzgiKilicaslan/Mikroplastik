import lightning.pytorch as pl
import torch.nn as nn
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, F1Score, MulticlassAccuracy
import torch.nn.functional as F

class ClassicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassicConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
    def forward(self, x):
        x = self.main(x)
        return x
    
class ResidualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.residual_connection_down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.MaxPool2d(2))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual_connection_down(x)
        output = self.main(x)
        return self.relu(output+residual) 

class MicroplasticCNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        metrics = MetricCollection([MulticlassAccuracy(num_classes=2, average='macro'), Accuracy(task="binary"), AUROC(task="binary"), AveragePrecision(task="binary"), F1Score(task="binary")])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.convolutions = nn.ModuleList([])
        channels_in = [3, 8, 8, 16, 16, 32, 32, 64]
        channels_out = [8, 8, 16, 16, 32, 32, 64, 64]

        for i in range(0, 8):
            self.convolutions.append(ClassicConv2d(channels_in[i], channels_out[i]))

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x): # to implement model

        for convolution in self.convolutions:
            x = convolution(x)

        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x.view(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def process_batch(self, batch, batch_idx):
        return batch[0], batch[1].type(torch.FloatTensor).cuda()
    
    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch, batch_idx)

        y_pred = self(x)

        loss = torch.nn.BCEWithLogitsLoss()

        self.log("train_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        y_pred = F.sigmoid(y_pred)
        self.log_dict(self.train_metrics(y_pred, y.type(torch.LongTensor).cuda()), sync_dist=True, on_epoch=True, batch_size=x.shape[0])
        return loss(y_pred, y)
    

    def validation_step(self, batch, batch_idx):
        x, y = self.process_batch(batch, batch_idx)

        y_pred = self(x)

        loss = torch.nn.BCEWithLogitsLoss()

        self.log("val_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        y_pred = F.sigmoid(y_pred)
        y = y.type(torch.LongTensor).cuda()
        self.log_dict(self.valid_metrics(y_pred, y), sync_dist=True, on_epoch=True, batch_size=x.shape[0])


    def test_step(self, batch, batch_idx):
        x, y = self.process_batch(batch, batch_idx)

        y_pred = self(x)

        loss = torch.nn.BCEWithLogitsLoss()

        self.log("test_loss", loss(y_pred, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        y_pred = F.sigmoid(y_pred)
        y = y.type(torch.LongTensor).cuda()
        
        self.log_dict(self.test_metrics(y_pred, y), sync_dist=True, on_epoch=True, batch_size=x.shape[0])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = self.process_batch(batch, batch_idx)

        y_pred = self(x)

        return y_pred
    
class MicroplasticCNNModel2(MicroplasticCNNModel):
    def __init__(self):
        super().__init__()

        self.convolutions = nn.ModuleList([])
        channels_in = [3, 8, 8, 16, 16, 32, 32, 64]
        channels_out = [8, 8, 16, 16, 32, 32, 64, 64]

        for i in range(0, 8):
            self.convolutions.append(ResidualConv2d(channels_in[i], channels_out[i]))
