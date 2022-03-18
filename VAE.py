
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

import pytorch_lightning as pl

import sklearn
from ray import tune

import argparse
import os
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import MLmodels as m
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback


class ATTENTION_VAE(pl.LightningModule):

    def __init__(self,
                 config,
                 hidden_dims=[8, 16, 32, 64, 128],
                 out_image_channels=1,
                 output_size=4, # Output channels, in our case 2 for our binary classifier
                 fcl_layers=[]):

        super(ATTENTION_VAE, self).__init__()

        # Loss Function, Good for binary classification. Avoids overfitting by randomnly assigning a small percentage of the true values for any batch
        self.criterion = m.SmoothCrossEntropyLoss(smoothing=0.01, reduction='sum')
        self.eps = 1.0
        self.replicas = 4

        self.__dict__.update(locals())

        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers['adam']

        # hyperparameters
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        # for importing different versions of the data
        self.datatype = config['datatype']
        self.dr = config['dr']

        kld = 1./self.batch_size
        self.train_criterion = m.SymmetricMSE(1.0, 0.3, kld) # A different loss function

        if 'B' in self.datatype and '20' not in self.datatype:
            self.data_length = 40
        else:
            self.data_length = 20

        self.training_data = None
        self.validation_data = None

        self.hidden_dims = hidden_dims
        self.z_dim = config['z_dim']
        self.out_image_channels = out_image_channels

        self.encoder = None
        self.decoder = None

        # convolution on data, first step in encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(2, 2)),  # 64@96*96
            nn.ReLU(inplace=True),
        )


        self.encoder_block1 = self.encoder_block(
            64, 48, (4,4), 1, (1,1))
        self.encoder_atten1 = m.Self_Attention(48)
        self.encoder_block2 = self.encoder_block(
            48, 32, (4,4), 1, (1,1))
        self.encoder_atten2 = m.Self_Attention(32)
        self.encoder_block3 = self.encoder_block(
            32, 16, (5, 15), (1,2), (1,1))
        self.encoder_atten3 = m.Self_Attention(16)

        # Add extra output channel if we are using Matt's physical constraint
        if self.out_image_channels > 1:
            self.hidden_dims = [
                self.out_image_channels * x for x in self.hidden_dims
            ]

        self.fc1 = self.make_fcn(128, self.z_dim, [128, 112], self.dr)
        self.fc2 = self.make_fcn(128, self.z_dim, [128, 112], self.dr)
        self.fc3 = self.make_fcn(self.z_dim, 128, [128, 128], self.dr)
        self.fcn = self.make_fcn(self.z_dim, output_size, [128, 64], self.dr)

        self.decoder_block2 = self.decoder_block(
            self.hidden_dims[4], self.hidden_dims[3], (4, 5), (1, 5), 0)
        self.decoder_atten2 = m.Self_Attention(self.hidden_dims[3])

        self.decoder_block3 = self.decoder_block(
            self.hidden_dims[3], self.hidden_dims[2], (1, 2), (1, 2), 0)
        self.decoder_atten3 = m.Self_Attention(self.hidden_dims[2])

        self.decoder_block4 = self.decoder_block(
            self.hidden_dims[2], self.hidden_dims[1], (1, 2), (1, 2), 0)
        self.decoder_atten4 = m.Self_Attention(self.hidden_dims[1])

        self.decoder_block5 = self.decoder_block(
            self.hidden_dims[1], self.out_image_channels, (1, 1), (1, 1), 0)
        self.decoder_atten5 = m.Self_Attention(self.hidden_dims[0])

        self.save_hyperparameters()

    def encoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            m.SpectralNorm(
                nn.Conv2d(dim1, dim2, kernel_size=kernel_size,
                          stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )

    def decoder_block(self, dim1, dim2, kernel_size, stride, padding, sigmoid=False):
        return nn.Sequential(
            m.SpectralNorm(
                nn.ConvTranspose2d(
                    dim1, dim2, kernel_size=kernel_size, stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU() if not sigmoid else nn.Sigmoid()
        )

    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers) - 1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i + 1]),
                        nn.BatchNorm1d(fcl_layers[i + 1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i + 1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.conv(x) # convolution on the batch of one-hot encoded sequences x
        h = self.encoder_block1(h)
        h, att_map1 = self.encoder_atten1(h)
        h = self.encoder_block2(h)
        h, att_map2 = self.encoder_atten2(h)
        h = self.encoder_block3(h)
        h, att_map3 = self.encoder_atten3(h)
        h = h.view(h.size(0), -1)  # flatten
        z, mu, logvar = self.bottleneck(h)

        return h, z, mu, logvar, [att_map1, att_map2, att_map3]

    def decode(self, z):
        z = self.fc3(z) #
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1)  # flatten/reshape

        z = self.decoder_block2(z)
        z, att_map2 = self.decoder_atten2(z)
        z = self.decoder_block3(z)
        z, att_map3 = self.decoder_atten3(z)
        z = self.decoder_block4(z)
        z, att_map4 = self.decoder_atten4(z)
        z = self.decoder_block5(z)
        return z, [att_map2, att_map3, att_map4]

    def forward(self, x):
        h, z, mu, logvar, encoder_att = self.encode(x)
        out = self.fcn(z)
        z, decoder_att = self.decode(z)
        return out, z, mu, logvar

    #Lightning Methods
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def prepare_data(self):
        # import our data
        train, validate, weights = m.get_rawdata(self.datatype, 10, 5, round=8)
        _train = train.copy()
        _validate = validate.copy()

        # Assigns labels for learning
        _train["binary"] = _train["affinity"].apply(m.bi_labelM)
        #print(_train[_train["binary"] == 1].count())
        #print(_train[_train["binary"] == 0].count())
        _validate["binary"] = _validate["affinity"].apply(m.bi_labelM)
        #print(_validate[_validate["binary"] == 1].count())
        #print(_validate[_validate["binary"] == 0].count())


        _weights = torch.FloatTensor(weights)
        # instantiate loss criterion, need weights so put this here
        self.criterion = m.SmoothCrossEntropyLoss(weight=_weights, smoothing=0.01, reduction='sum')

        self.training_data = _train
        self.validation_data = _validate


    def train_dataloader(self):
        # Data Loading
        train_reader = m.NAReader(self.training_data, shuffle=True)

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

        # get output from the model, given the inputs
        predictions, xp, mu, logvar = self(x)
        xpp = torch.where(xp > 0.5, 1.0, 0.0)

        recon_acc = (x == xpp).float().mean()
        seq_acc = recon_acc.item()

        loss = self.criterion(predictions, y)

        vae_loss, bce, kld = self.train_criterion(x, xpp, mu, logvar)

        _epoch = self.current_epoch+1 # lightning module member
        _eps = self.eps / (1 + 0.06 * _epoch)
        train_loss = (1 - _eps) * loss + _eps * vae_loss
        # Convert to labels
        preds = torch.argmax(predictions, 1).clone().double() # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        train_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_seq_accuracy", seq_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def val_dataloader(self):
        # Data Loading
        # train_reader = m.NAContrast(_train, n=n, shuffle=True)
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
        seq_aves = []
        pred_aves = []
        for _ in range(self.replicas):
            predictions, xp, mu, logvar = self(x)
            seq_aves.append(xp)
            pred_aves.append(predictions)
        predictions = torch.mean(torch.stack(pred_aves, dim=0), dim=0)
        xp = torch.mean(torch.stack(seq_aves, dim=0), dim=0)

        xpp = torch.where(xp > 0.5, 1.0, 0.0)

        recon_acc = (x == xpp).float().mean()
        seq_acc = recon_acc.item()

        # get loss for the predicted output
        val_loss = torch.nn.CrossEntropyLoss(reduction='sum')(predictions, y)
        vae_loss, bce, kld = self.train_criterion(x, xp, mu, logvar)

        # Convert to labels
        preds = torch.argmax(predictions, 1).clone().double()  # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        val_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_seq_accuracy", seq_acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_acc": val_acc}


# Training Function for Raytune Hyperparameter Optimization
def train_vae(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = pl.Trainer(
        # default_root_dir="~/ray_results/",
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "acc": "ptl/val_accuracy",
                    "train_seq_acc": "ptl/train_seq_accuracy",
                    "val_seq_acc": "ptl/val_seq_accuracy"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    )

    if checkpoint_dir:
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = ATTENTION_VAE._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = ATTENTION_VAE(config, True, 152, image_channels=1,
                              hidden_dims=[128, 128, 128, 128, 128], out_image_channels=1, output_size=2, fcl_layers=[])

    trainer.fit(model)


# Hyperparameter Optimization with the Raytune Libarary
def tune_asha(datatype, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128]),
        "dr": tune.loguniform(0.005, 0.5),
        "z_dim": tune.choice([10, 100, 200]),
        "datatype": datatype
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "acc", "training_iteration", "train_seq_accuracy", "val_seq_accuracy"])

    analysis = tune.run(
        tune.with_parameters(
            train_vae,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="acc",
        mode="max",
        config=config,
        num_samples=num_samples,
        local_dir="./ray_results/",
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_vae_asha")

    print("Best hyperparameters found were: ", analysis.best_config)
    # analysis.to_csv('~/ray_results/' + config['datatype'])


if __name__ == "__main__":

    ### Single Step debugging
    # z_dim is the embedding dimension
    # Intialize model with hyperparameters
    config = {'lr': 1e-3, 'dr': 0.1, 'batch_size': 512, 'z_dim': 10, 'datatype': 'HCLT', "epochs":10}

    # model = ATTENTION_VAE(config, hidden_dims=[128, 128, 128, 128, 128], out_image_channels=1, output_size=2, fcl_layers=[])
    #
    # model.prepare_data()
    # d = model.train_dataloader()
    # for i, batch in enumerate(d):
    #     if i > 0:
    #         break
    #     else:
    #         model.training_step(batch, i)
    #         model.validation_step(batch, i)

    # pytorch lightning loop
    rn = ATTENTION_VAE(config, hidden_dims=[128, 128, 128, 128, 128], out_image_channels=1, output_size=2, fcl_layers=[])

    logger = TensorBoardLogger('tb_logs', name='VAE_trial')  # logging using Tensorboard
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # switching between cpu and gpu is as easy as changing the number on the gpus argument
    plt.fit(rn)  # Starts Training Process












