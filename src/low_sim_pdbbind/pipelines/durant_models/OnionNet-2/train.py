import torch
from torch.utils.data.dataset import Dataset
from lightning import LightningModule


class IMGDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class OnionNet2(LightningModule):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=1
            )

        self.conv2 = torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1
            )

        self.conv3 = torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1
            )

        self.lin1 = torch.nn.Linear(1104000, 100)
        self.bn1 = torch.nn.BatchNorm1d(num_features=100)

        self.lin2 = torch.nn.Linear(100, 50)
        self.bn2 = torch.nn.BatchNorm1d(num_features=50)

        self.lin3 = torch.nn.Linear(50, 1)

        self.loss = torch.nn.MSELoss()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = x.flatten(1, 3)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.lin2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.lin3(x)
        x = self.relu(x)

        return x

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y, y_pred)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y, y_pred)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch):
        x, _ = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=0.001,
        )
        out = {"optimizer": optimizer}

        return out
