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
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.patheffects as pe
from pathlib import Path

def generate_confusion_matrix(conf):

        if True : # To not normalise the confusion matrix , set me to False
            # Normalise the confusion matrix 
            conf = conf / conf.sum() # Check of many percent have received their correct label
            # Normalise the target axis = 1
            conf = conf * 100 # To have results as 100% instead of 1.0

        fig, ax = plt.subplots()
        cax = ax.matshow(conf, interpolation="nearest")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Target")
        ax.set_title("Confusion matrix")
        fig.colorbar(cax)

        #if len(conf)==5:
        #    # If we have a dataset with 5 class, we suppose it's still the ELO powder images class so we set them as title on the axis.
        #    list_class_label = ["Good", "Porous","Bulging","Edge","powder"],
        #    ax.set_xticklabels(list_class_label)
        #    ax.set_yticklabels(list_class_label)

        for (i, j), z in np.ndenumerate(conf):
            ax.text(j, i, '{:0.1f}%'.format(z), ha='center', va='center', color="black", path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        plt.tight_layout()
        return fig

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def get_squeezenet():
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    model = torchvision.models.squeezenet1_0(pretrained=True)
    #set_parameter_requires_grad(model)

    num_classes = 5 # Number of labels or classes 
    model.classifier[1] =  nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1)))
    
    #model.classifier[1] = nn.Sequential(
    #    Flatten(),
    #    nn.Linear(512, 256),#nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1)),
    #    nn.ReLU(),
    #    nn.Linear(256, 128),
    #    nn.ReLU(),
    #    nn.Linear(128,num_classes),
    #    ) 
    model.num_classes = num_classes
    return model

def get_alexnet():
    num_classes=5
    model = torchvision.models.alexnet(pretrained=True)
    #set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224
    model.num_classes = num_classes
    return model

class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #self.net = get_squeezenet()
        self.net = get_alexnet()
        self.kwarg = kwargs

    def forward(self, x):
        x = self.net(x)
        output = F.log_softmax(x, dim=1)
        return output

class ImageClassifier(LightningModule):
    def __init__(self, model=None, batch_size=32, lr=0.001, num_classes=5, checkpoints_dir="checkpoints",**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = int(num_classes)
        self.save_hyperparameters(ignore="model")
        self.model = model or Net(**kwargs)
        self.val_acc_fc = Accuracy("multiclass", num_classes=num_classes)
        self.train_acc_fc = Accuracy('multiclass', num_classes=num_classes)
        self.test_acc_fc = Accuracy('multiclass', num_classes=num_classes)
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoints_dir = checkpoints_dir
        self.kwarg = kwargs

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
        self.log("train_acc", self.train_acc_fc(torch.argmax(logits, axis=1), y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.train_acc_fc(torch.argmax(logits, axis=1), y), on_step=True, on_epoch=True)
        #print(self.num_classes)
        confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(y.device)
        conf = confmat(torch.argmax(logits, axis=1),y)

        return conf

    def validation_epoch_end(self, outs) -> None:
        conf = torch.zeros_like(outs[0])

        for out in outs:
            conf = conf + out

        conf = conf.cpu().detach().numpy()
        
        fig = generate_confusion_matrix(conf)
        confusion_matrix_path = self.checkpoints_dir + f"/Confusion_matrix_epoch_{self.current_epoch}.jpg"
        fig.savefig(confusion_matrix_path, dpi=500)

        self.logger.experiment.add_image(f"Confusion_matrix", np.moveaxis(plt.imread(confusion_matrix_path), 2, 0), global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())

        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_acc", self.train_acc_fc(torch.argmax(logits, axis=1), y), on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adadelta(self.model.parameters(), lr=self.lr)

    @property
    def transform(self):
        preprocess = T.Compose([ T.Resize((224,224)),#T.Resize((196, 196)),
                           T.ToTensor(),
                           T.ConvertImageDtype(torch.float),
                           #T.Normalize(mean=(0.485, 0.456, 0.406),
                           #            std=(0.229, 0.224, 0.225))
                           ])
        return preprocess

    def set_train_dataloader(self, train_data_path: str) -> None:
        """ Load the train dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        train_data_path: str
            Path to the train data.
        """
        train_dataset = torchvision.datasets.ImageFolder(train_data_path, transform=self.transform)
        

        self._train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self) -> DataLoader:
        dl = self._train_dataloader
        if dl is not None:
            return dl
        else:
            raise "Data loader not configured"
    
    def set_val_dataloader(self, val_data_path: str) -> None:
        """ Load the val dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        val_data_path: str
            Path to the train dataset.
        """
        val_dataset = torchvision.datasets.ImageFolder(val_data_path, transform=self.transform)
        
        self._val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        dl = self._val_dataloader
        if dl is not None:
            return dl
        else:
            raise "Data loader not configured"

    def set_test_dataloader(self, test_data_path: str) -> None:
        """ Load the val dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        val_data_path: str
            Path to the train dataset.
        """
        test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=self.transform)

        self._test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        dl = self._test_dataloader
        if dl is not None:
            return dl
        else:
            raise "Data loader not configured"


class DataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        preprocess = T.Compose([T.Resize((224,224)),#T.Resize((196, 196)),
                           T.ToTensor(),
                           T.ConvertImageDtype(torch.float),
                           #T.Normalize(mean=(0.485, 0.456, 0.406),
                           #            std=(0.229, 0.224, 0.225))
                           ])
        return preprocess



    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder("./data/ELO_jpg_subimages_4_class/train", transform=self.transform)
        
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        val_dataset = torchvision.datasets.ImageFolder("./data/ELO_jpg_subimages_4_class/val", transform=self.transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size)

def TrainNN(accelerator, TrainPath, ValPath, TestPath,max_epochs, nb_classes, suffix="", learning_rate=0.001,**kwarg) :
    """Function to train, test and validate the model.

    The ImageClassifier model will be trained using the data from `TrainPath` with the learning rate defined in `learning_rate` for `max_epochs` epochs.
    At each epoch, the AI model is validated on the on the data from `ValPath` at each epoch.
    At the end of the training, the AI model is tested on the data from `TestPath`.

    The checkpoints and logs are saved in a folder in checkpoint name as YYYY_MM_DD_HH_MM_SS_grid_split_`suffix`.
    
    Parameters
    ----------
    accelerator : str
        "cpu" or "gpu" to indicate if the Ai must be train on the GPU (faster) or CPU.
    TrainPath : str
        Path to the train data folder.
    ValPath : str
        Path to the validation data folder (Used at the end of every epoch).
    TestPath : str
        Path to the test data folder (used at the end).
    max_epochs : int
        Max number of epochs to train the AI.
    nb_classes : int
        Number of class for the classification. 
    suffix : str, optional
        Suffix to add to the checkpoint folder name, by default "".
    learning_rate : float, optional
        Learning rate to use during the training, by default 0.001.
    """
    name = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + suffix   # Name of training run
    checkpoints_dir = f"checkpoints/{name}/"                    # Folder that will contain the training run data
  
    pl.seed_everything(51, workers=True)  # Set pytorch lining random seed at 51 for repetability in the training
    model = ImageClassifier(num_classes=nb_classes, checkpoints_dir=checkpoints_dir, lr=learning_rate,**kwarg)                # object model dealing with the training and containing all parameters
    model.set_train_dataloader(TrainPath)
    model.set_val_dataloader(ValPath)
    model.set_test_dataloader(TestPath)

    if accelerator == "cpu": 
        devices = None
    else:
        devices = -1 # Take all GPUs
        model = model.to("cuda")

    # Define the criteria that will define the best model to save
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor='val_acc', #"val_loss_epoch",  # Monitor the val_loss value for each epoch
        mode="max",  # Smaller is the val_loss_step, the better is the result, bigger is val_acc the more accurate it is
        save_top_k=2,  # Save the best model
        filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        #filename="best",  # The best model is saved as "best"
        save_last=True,  # Save the last model trained
    )

    # Log the training evolution data
    logger = TensorBoardLogger(checkpoints_dir,
                               name=name,
                               log_graph=True,  # Require a good self.example_input_array in ImageClassifier
                               )
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,  # Take all GPU available
                         default_root_dir=checkpoints_dir,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=max_epochs,
                         auto_scale_batch_size="binsearch",
                         check_val_every_n_epoch=1,
                         )
    trainer.fit(model)

    torch.save(model.state_dict(), Path(checkpoints_dir) / "last_model_weight.pt")
    # Load the best model seen by the callback
    best_model = checkpoint_callback.best_model_path
    print(f"The best model should be {checkpoint_callback.best_model_path}")
    model.load_from_checkpoint(best_model)

    # TODO: Update model class to suport test
    print("Perform final Test")
    # Test the model at the end with the best model
    test_loss = trainer.test(ckpt_path=best_model)
    #test_loss = trainer.test(model)
    print(f"The test loss is {test_loss}")

    # Confusion matrix with the test data (only place where test data is used)
    print("Create Confusion Matrix")
    for x in iter(model.test_dataloader()):
        pred = model(x[0])

        confmat = ConfusionMatrix(task="multiclass", num_classes=nb_classes).to(x[0].device)
        conf_temp = confmat(torch.argmax(pred, axis=1), x[1])
        try:
            conf = conf + conf_temp
        except:
            conf = conf_temp
    confusion_matrix_path = checkpoints_dir + r"/Confusion_matrix_final.jpg"

    fig= generate_confusion_matrix(conf)
    fig.savefig(confusion_matrix_path, dpi=500)
    logger.experiment.add_image("Confusion matrix",np.moveaxis(plt.imread(confusion_matrix_path),2,0), global_step=trainer.global_step + 1)

    return best_model # Path to the best model (str)
#if __name__ == "__main__":
#    cli = LightningCLI(
#        ImageClassifier, DataModule, seed_everything_default=42, save_config_overwrite=True, run=False
#    )
#    cli.trainer.fit(cli.model, datamodule=cli.datamodule)