import torch
from torch.utils.data import DataLoader
from losses import (
    NowcastingLoss,
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
    grid_cell_regularizer,
)
import pytorch_lightning as pl
import torchvision
from dataset import MyDataset
from common import LatentConditioningStack, ContextConditioningStack
from generators import Sampler, Generator
from discriminators import Discriminator

class DGMR(pl.LightningModule):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = True,
        pretrained: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768, 
        context_channels: int = 384,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels
        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super(DGMR, self).__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.discriminator = Discriminator(input_channels)
        self.visual = Visualization()
        self.save_hyperparameters()
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch
        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            # Cat along time dimension [B, C, T, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = concatenated_outputs.chunk(2, dim=0)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        predictions = [self(images) for _ in range(6)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)
        # Concat along time dimension
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)
        # Cat long batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = concatenated_outputs.chunk(2, dim=0)
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()
       
        self.log_dict(
            {
                "train/d_loss": discriminator_loss,
                "train/g_loss": generator_loss,
                "train/grid_loss": grid_cell_reg,
            },
            prog_bar=True,
            )
         
    def validation_step(self, batch, batch_idx, thr=0.1):
        # pytorch lightning validation step
        # x : b s c h w
        # y = y_hat : b s 1 h w       
        x, y = batch
        y_hat = self(x)
        p, t = y_hat.cpu().numpy(), y.cpu().numpy()
        p = np.where(p < thr, False, True)
        t = np.where(t < thr, False, True)
        TP = (np.logical_and(t, p)).sum() # True positive
        FNP = (np.logical_xor(t, p)).sum() # False postive + False negative
        valid_score = np.where(TP > 0, TP / (TP + FNP), 0) # CSI
        return valid_score
      
    def validation_epoch_end(self, outputs):
        # pytorch lightning - process after validation step
        valid_score = np.average(outputs)
        # (optional) tensorboard
        self.log_dict({
            "valid_score": valid_score
        })
              
    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def train_dataloader(self):
       # Define training dataset
        year = range(2020, 2023)
        dataset = []
        for y in year:
            dataset.append(MyDataset(year=y))
        datasets = torch.utils.data.ConcatDataset(dataset)
        train_loader = DataLoader(datasets, batch_size=16, num_workers=4, shuffle=True)
        return train_loader

    def val_dataloader(self):
        # (optional) Define validation dataset
        dataset = MyDataset(year='2022', split='valid')
        valid_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
        return valid_loader
