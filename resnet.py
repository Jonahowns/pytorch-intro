from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import pytorch_lightning as pl
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
import MLmodels as m


class ResNetClassifier(pl.LightningModule):
    def __init__(self, config, num_classes, resnet_version,
                test_path=None,
                 optimizer='adam',
                 transfer=True):
        super().__init__()

        self.__dict__.update(locals())

        ## Preconfigured Resnet Models provided from pytorch package
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # hyperparameters
        self.lr = config['lr']  # learning rate
        self.batch_size = config['batch_size'] # Number of data to process in single batch
        # for importing different versions of the data
        self.datatype = config['datatype']

        if 'B' in self.datatype and '20' not in self.datatype:
            self.data_length = 40
        else:
            self.data_length = 20

        self.training_data = None
        self.validation_data = None


        # Using a pretrained ResNet backbone
        # Pretraining is using parameters from another model to learn
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features

        # replace final layer for fine tuning
        fcn = [
            nn.Dropout(config['dr']),
            nn.Linear(linear_size, num_classes)
        ]
        if num_classes > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))

        self.fcn = nn.Sequential(*fcn)
        # Change Convolution Sizing to work with our data, in this one hot encoded vectors
        self.resnet_model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

        modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
        self.resnet_model = nn.Sequential(*modules)

    def forward(self, X):
        # Takes our in
        x = self.resnet_model(X) # calls the forward function of the specified resnet model
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def prepare_data(self):
        # import our data
        train, validate, weights = m.get_rawdata(self.datatype, 10, 5, round=8)
        _train = train.copy()
        _validate = validate.copy()

        # Assigns labels for learning
        _train["binary"] = _train["affinity"].apply(m.bi_labelM)
        _validate["binary"] = _validate["affinity"].apply(m.bi_labelM)

        _weights = torch.FloatTensor(weights)
        # instantiate loss criterion, need weights so put this here
        self.criterion = m.SmoothCrossEntropyLoss(weight=_weights, smoothing=0.01)

        self.training_data = _train
        self.validation_data = _validate


    def train_dataloader(self):
        # Data Loading
        train_reader = m.NAReader(self.training_data, shuffle=True, max_length=self.data_length)

        train_loader = torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            # batch_size=self.batch_size,
            collate_fn=m.my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=True
        )

        return train_loader

    def training_step(self, batch, batch_idx):
        seq, x, y = batch
        softmax = self(x)
        train_loss = self.criterion(softmax, y)

        # Convert to labels
        preds = torch.argmax(softmax, 1).clone().double() # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        train_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def val_dataloader(self):
        # Data Loading
        val_reader = m.NAReader(self.validation_data, shuffle=False)

        val_loader = torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            collate_fn=m.my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        return val_loader

    def validation_step(self, batch, batch_idx):
        seq, x, y = batch
        softmax = self(x)
        val_loss = self.criterion(softmax, y)

        # Convert to labels
        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        val_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_acc": val_acc}


if __name__ == "__main__":
    # To call this script use:
    # example usage: python ./folder/resnet.py HCLT

    ### Debugging
    # Our config holds all of our hyperparameters, lr: learning rate, dr: dropout on nn.Drouput layer, batch_size: number of sequences processed at once, epochs: number of training iterations
    config = {'lr': 1e-3, 'dr': 0.1, 'batch_size': 512, 'datatype': 'HCB20T', 'epochs': 10}

    # Step through the individual steps of the pytorch training loop for debugging/etc.
    ## Single Loop debugging
    #
    # Load data from file and make available to lightning module
    # model.prepare_data() # Fetches data accoring to get_raw_data() function in MLmodels.py, Implicitly called by Pytorch Lightning
    #
    # Load either our training of validation dataset into our dataloader
    # d = model.train_dataloader()
    # # d = model.val_dataloader()
    #
    # print("Number of Batches", len(d))
    # for i, batch in enumerate(d):
    #     if i > 0:  # only do the first batch, i = 0
    #         break
    #     model.training_step(batch, i)
    #     # model.validation_step(batch, i)


    # pytorch lightning loop
    rn = ResNetClassifier(config, 2, 152, optimizer='adam')
    # Creating a ResNetClassifier Pytorch Lightning object. The conf

    logger = TensorBoardLogger('tb_logs', name='resnet_trial')    # logging using Tensorboard
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # switching between cpu and gpu is as easy as changing the number on the gpus argument
    plt.fit(rn)  # Starts Training Process















