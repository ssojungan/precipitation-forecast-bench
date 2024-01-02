import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import Encoder, Decoder, ED
from net_params import Params
from dataset import MyDataset
import numpy as np

### pytorch lightning process ###
# Initialize -> (optional) validation_step (check) -> training_step -> (optional) training_epoch_end
# -> (optional) validation step -> (optional) validation_epoch_end
class trajGRU(pl.LightningModule):
    """
    Parameters
    ----------
    forecast_step : number of forecast frames
    in_channels : number of variables
    height : images height
    width : images width
    ----------
    out : weights-path/{}.ckpt, lightning_logs
    """
    def __init__(
        self,
        forecast_steps: int = 12,
        in_channels: int = 1,
        out_channels: int = 64,
        height: int = 256,
        width: int = 256,
    ):
        super(trajGRU, self).__init__()
        self.forecast_steps = forecast_steps
        self.save_hyperparameters()
        param = Params(in_channels, out_channels, height, width)
        encoder = Encoder(param.trajgru_encoder_params[0], param.trajgru_encoder_params[1])
        decoder = Decoder(param.trajgru_decoder_params[0], param.trajgru_decoder_params[1], forecast_steps)
        self.NORMAL_LOSS_GLOBAL_SCALE = 0.00005
        self.loss_w = torch.arange(1.0, 1.0 + 11 * 0.01, 0.01)      
        self.model = ED(encoder, decoder)
        self.mse_loss = torch.nn.MSELoss()
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # pytorch lightning train step
        # batch: train_dataset return vector
        # batch_idx : train_dataset batch order - 0, 1, ..., n
        # x : b s c h w
        # y = y_hat : b s 1 h w
        # self : forward()
        # TrajGRU
        optimizer = self.optimizers()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.7)
        
        x, y = batch 
        y_hat = self(x)
        
        # COMPUTE WEIGHT LOSS (TRAJGRU)

        weight = torch.where(y < 0.0, 0, torch.where(y < 2, 1,
                    torch.where(y < 5, 2, torch.where(y < 10, 5, torch.where(y < 30, 10, 30)))))
        mse = torch.sum(weight.type_as(images) * ((y_hat - y) ** 2), (2, 3, 4))
        mae = torch.sum(weight.type_as(images) * (torch.abs((y_hat - y))), (2, 3, 4))
        mse = (self.loss_w.type_as(images) * mse).permute(1, 0)
        mae = (self.loss_w.type_as(images) * mae).permute(1, 0)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()
        self.configure_gradient_clipping(optimizer)

        # (optinal) tensorboard 
        # -> (auto save) lightning_logs/{version_0, version_1,...}
        self.log_dict(
            {
                "train_loss": loss,
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
