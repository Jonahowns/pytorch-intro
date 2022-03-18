import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD, AdamW, Adagrad  # Supported Optimizers
from torch.utils.data import DataLoader
from torchsummary import summary

from multiprocessing import cpu_count

import MLmodels as m

from math import floor

# Adapted From https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py

# when beta =1, it is a normal VAE


class BetaVAE(LightningModule):

    def __init__(self, config, debug=False):
        super().__init__()

       # Assign Hyperparameters
        self.datatype = config["datatype"]
        self.latent_dim = config["latent_dim"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.loss_type = config["loss_type"]
        self.C_max = torch.Tensor([config["max_capacity"]])
        self.C_stop_iter = config["capacity_max_iter"]
        self.lr = config['lr']
        optimizer = config["optimizer"]
        self.seed =config['seed']
        self.batch_size = config["batch_size"]
        self.attention_layers = config["attention_layers"]  # Something we might add in the future

        if config["hidden_dims"] is None:
            hidden_dims = [32, 64, 128, 256, 512]
        else:
            hidden_dims = config["hidden_dims"]

        # option for loss function
        self.kld_weight = 1. / self.batch_size

        # Sets worker number for both dataloaders
        if debug:
            self.worker_num = 0
        else:
            try:
                self.worker_num = config["data_worker_num"]
            except KeyError:
                self.worker_num = cpu_count()

        # If GPU is being used, sets pin_mem to True for better performance
        if hasattr(self, "trainer"):  # Sets Pim Memory when GPU is being used
            if hasattr(self.trainer, "on_gpu"):
                self.pin_mem = self.trainer.on_gpu
            else:
                self.pin_mem = False
        else:
            self.pin_mem = False

        if self.loss_type not in ['B', 'H']:
            print(f"Loss Type {self.loss_type} not supported")
            exit(1)

        # optimizer options
        if optimizer == "SGD":
            self.optimizer = SGD
        elif optimizer == "AdamW":
            self.optimizer = AdamW
        elif optimizer == "Adagrad":
            self.optimizer = Adagrad
        else:
            print(f"Optimizer {optimizer} is not supported")
            exit(1)

        # Pytorch Basic Options
        torch.manual_seed(self.seed)  # For reproducibility
        torch.set_default_dtype(torch.float64)  # Double Precision


        # build encoder
        self.block_num = len(hidden_dims) - 1
        self.encoder_input = nn.Conv2d(1, out_channels=hidden_dims[0], kernel_size=3, padding=1)
        self.encoder = []
        for i in range(self.block_num):
            self.encoder.append(self.build_encoder_block(hidden_dims[i], hidden_dims[i+1], kernel=3, padding=1, stride=2))
            if self.attention_layers:
                self.encoder.append(m.Self_Attention(hidden_dims[i+1]))

        reversed_hidden_dims = hidden_dims[::-1]
        # build decoder
        self.decoder_input = nn.Linear(self.latent_dim, reversed_hidden_dims[0] * 4)
        self.decoder = []
        for i in range(self.block_num):
            self.encoder.append(self.build_encoder_block(reversed_hidden_dims[i], reversed_hidden_dims[i + 1], kernel=3, padding=1, stride=2))
            if self.attention_layers:
                self.encoder.append(m.Self_Attention(hidden_dims[i + 1]))

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(reversed_hidden_dims[0],
                               reversed_hidden_dims[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(reversed_hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(reversed_hidden_dims[0], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())


        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)

        self.save_hyperparameters()

    #### General B-VAE Methods
    def build_encoder_block(self, in_dims, out_dims, kernel=3, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_dims, out_channels=out_dims,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU())

    def build_decoder_block(self, in_dims, out_dims, kernel=3, stride=2, padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dims,
                               out_dims,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding,
                               output_padding=1),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.encoder_input(input)
        if self.attention_layers:
            attn_maps = []
            for i in range(self.block_num):
                indx = 2*i
                x = self.encoder[indx](x)
                x, attn_map = self.encoder[indx+1](x)
                attn_maps.append(attn_map)
        else:
            attn_maps = []
            for i in range(self.block_num):
                x = self.encoder[i](x)

        result = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var, attn_maps]

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 512, 2, 2)

        if self.attention_layers:
            attn_maps = []
            for i in range(self.block_num):
                indx = 2 * i
                z = self.decoder[indx](z)
                z, attn_map = self.decoder[indx + 1](z)
                attn_maps.append(attn_map)
        else:
            attn_maps = []
            for i in range(self.block_num):
                z = self.decoder[i](z)

        result = self.final_layer(z)
        return result, attn_maps

    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var, encode_attn_maps = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recon_x, decode_attn_maps = self.decode(z)
        return [recon_x, input, mu, log_var, encode_attn_maps, decode_attn_maps]

    def loss_function(self, recons, input, mu, log_var):

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            C = torch.clamp(self.C_max/self.C_stop_iter * self.current_epoch, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * self.kld_weight * (kld_loss - C).abs()

        return {'loss': loss, 'recon_loss':recons_loss.detach(), 'kld_loss':kld_loss.detach()}

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=self.current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    #### Pytorch Lightning Methods
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def prepare_data(self):
        # import our data
        train, validate, weights = m.get_rawdata(self.datatype, 10, 5, round=8)
        _train = train.copy()
        _validate = validate.copy()

        self.training_data = _train
        self.validation_data = _validate

    def train_dataloader(self):
        # Data Loading
        train_reader = m.NAReader(self.training_data, shuffle=True)

        return DataLoader(
            train_reader,
            batch_size=self.batch_size,
            collate_fn=m.my_collate_unsupervised,
            num_workers=self.worker_num,
            pin_memory=self.pin_mem,
            shuffle=True
        )

    def val_dataloader(self):
        val_reader = m.NAReader(self.validation_data, shuffle=False)

        return DataLoader(
            val_reader,
            batch_size=self.batch_size,
            collate_fn=m.my_collate_unsupervised,
            num_workers=self.worker_num,
            pin_memory=self.pin_mem,
            shuffle=False
        )

    def training_step(self, batch, batch_idx):
        seq, x = batch

        # get output from the model, given the inputs
        [recon_x, input, mu, log_var, encode_attn_maps, decode_attn_maps] = self(x)
        xpp = torch.where(recon_x > 0.5, 1.0, 0.0)

        recon_acc = (x == xpp).float().mean().item()

        loss_dict = self.loss_function(recon_x, x, mu, log_var)
        loss_dict["recon_acc"] = recon_acc.detach()

        self.log("ptl/train_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_recon_acc", recon_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_dict


    def training_epoch_end(self, outputs):
        # These are detached
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_kld_loss = torch.stack([x["kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["recon_loss"] for x in outputs]).mean()
        avg_recon_acc = torch.stack([x["recon_acc"] for x in outputs]).mean()

        # For Tensorboard Logger
        self.logger.experiment.add_scalars("All Scalars", {"Train Loss": avg_loss,
                                                           "Train KLD_Loss": avg_kld_loss,
                                                           "Train Reconstruction Loss": avg_recon_loss,
                                                           "Train Reconstruction Accuracy": avg_recon_acc,
                                                           }, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        seq, x = batch
        # horrible spelling of averages
        seq_aves = []
        pred_aves = []
        for _ in range(self.replicas):
            [recon_x, input, mu, log_var, encode_attn_maps, decode_attn_maps] = self(x)
            seq_aves.append(recon_x)

        xp = torch.mean(torch.stack(seq_aves, dim=0), dim=0)

        xpp = torch.where(xp > 0.5, 1.0, 0.0)

        recon_acc = (x == xpp).float().mean().item()

        loss_dict = self.loss_function(recon_x, x, mu, log_var)
        loss_dict["recon_acc"] = recon_acc.detach()

        self.log("ptl/val_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_recon_acc", recon_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_dict

    def validaction_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_kld_loss = torch.stack([x["kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["recon_loss"] for x in outputs]).mean()
        avg_recon_acc = torch.stack([x["recon_acc"] for x in outputs]).mean()

        # For Tensorboard Logger
        self.logger.experiment.add_scalars("All Scalars", {"Validation Loss": avg_loss,
                                                           "Validation KLD_Loss": avg_kld_loss,
                                                           "Validation Reconstruction Loss": avg_recon_loss,
                                                           "Validation Reconstruction Accuracy": avg_recon_acc,
                                                           }, self.current_epoch)


if __name__ == '__main__':
    config = {"datatype": "HCL",
              "in_channels": 1,
              "latent_dim": 50,
              "beta": 1,  # # only used if loss_type == "B", B = 1 is standard VAE
              "gamma": 1,  # only used if loss_type == "H"
              "hidden_dims": [32, 64, 128, 256, 512],
              "max_capacity": 25,  # only used if loss_type == "H"
              "capacity_max_iter": 1e5,  # only used if loss_type == "H"
              "loss_type": "B",  # B or H
              "data_worker_num": 6,
              "optimizer": "AdamW",
              "lr": 1e-3,
              "seed": 38,
              "batch_size": 5000,
              "attention_layers": False}

    beta_vae = BetaVAE(config, debug=True)
    # summary(beta_vae, (1, 4, 20), 5000)

    beta_vae.prepare_data()
    td = beta_vae.train_dataloader()
    for i, b in enumerate(td):
        if i > 0:
            break
        else:
            seq, x = b
            beta_vae.encode(x)
