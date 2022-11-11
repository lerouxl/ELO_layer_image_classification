import os
import torch
from torch import nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch.nn import functional as F
from torchmetrics import Accuracy
import torch
import torchvision

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_squeezenet():
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    model =  torchvision.models.squeezenet1_1(pretrained=True)
    set_parameter_requires_grad(model)
    
    model.classifier[1] =  nn.Sequential(
        nn.Dropout(),
        nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1)),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ) 
    model.num_classes = 5
    return model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = get_squeezenet()

    def forward(self, x):
        x = self.net(x)
        output = F.log_softmax(x, dim=1)
        return output

class ImageClassifier(LightningModule):
    def __init__(self, model=None, lr=1.0, gamma=0.7, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model or Net()
        self.val_acc = Accuracy()

        checkpoint_path = os.path.join(os.path.dirname(__file__), "weights.pt")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path).state_dict())

    @property
    def example_input_array(self):
        return torch.zeros((1, 3, 196, 196))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.log("val_acc", self.val_acc(logits, y), on_step=True, on_epoch=True)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)


class DataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        preprocess = T.Compose([T.Resize((196, 196)),
                           T.ToTensor(),
                           T.ConvertImageDtype(torch.float),
                           T.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225))])
        return preprocess



    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder("./data/ELO_jpg_subimages_4_class/train", transform=self.transform)
        
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        val_dataset = torchvision.datasets.ImageFolder("./data/ELO_jpg_subimages_4_class/val", transform=self.transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size)


if __name__ == "__main__":
    cli = LightningCLI(
        ImageClassifier, DataModule, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
