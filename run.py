import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from utils.data_loading import PhcDataset
import numpy as np
from unet.unet_model import UNet
import argparse

class PhcDataModule(LightningDataModule):
    def __init__(self,  images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', batch_size: int = 32, num_workers: int = 8, **kwargs):
        super().__init__()
        self.phc = PhcDataset(images_dir, masks_dir, scale, mask_suffix)
        self.size = len(self.phc)
        self.batch_size = batch_size
        self.num_workers = num_workers
        



    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every GPU
        self.train_dataset,self.val_dataset,self.test_dataset = random_split(self.phc,[int(self.size*0.8),int(self.size*0.1),int(self.size*0.1)],generator=torch.Generator().manual_seed(0))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class UNetLightning(LightningModule):
    def __init__(self, 
        use_rf: bool = False, # Use Receiver-Field Regularization
        rf_reg_weight: float = 0.1,
        num_classes: int = 2,
        n_channels: int = 1,
        reg_layers: int = 4,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        train_epochs: int = 100,
        **kwargs
        ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(n_channels=n_channels,n_classes=num_classes, use_rf=use_rf)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.use_rf = use_rf
        self.rf_reg_weight = rf_reg_weight
        self.reg_layers = reg_layers
        self.tr_loss = []
        self.tr_acc = []
        self.va_loss = [] 
        self.va_acc = []
        self.tr_temp_loss = []
        self.tr_temp_acc = []
        self.va_temp_loss = []
        self.va_temp_acc = []
    
    def forward(self,x):
        # if self.use_rf:
        #     x = torch.tensor(x,requires_grad=True).to(self.device)
        self.model.device = self.device
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.model.set_use_rf(self.use_rf)
        # if self.use_rf:
        #     pred_mask, rf_loss = self.forward(x)
        # else: pred_mask = self.forward(x)
        if self.use_rf:
            pred_mask, used_areas = self.forward(x)
            used_areas = np.array(used_areas)
        else:
            pred_mask = self.forward(x)
        loss = self.loss_fn(pred_mask, y)
        self.log("train_loss", loss,prog_bar=True)
        if self.use_rf:
            rf_loss = np.mean(used_areas[:self.reg_layers])
            freq_reg_loss = self.rf_reg_weight * rf_loss
            loss += freq_reg_loss
            self.log("train_freq_reg", freq_reg_loss,prog_bar=True)
        self.tr_temp_loss.append(loss)
        self.log("train_total_loss", loss,prog_bar=True)
        self.tr_temp_acc.append(self.accuracy(pred_mask,y))
        self.log("train_accurcay", self.tr_temp_acc[-1],prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.train_epochs)

        return [opt],[cosine_scheduler]

    def training_epoch_end(self, training_step_outputs):
        self.tr_loss.append(sum(self.tr_temp_loss)/len(self.tr_temp_loss))
        self.tr_acc.append(sum(self.tr_temp_acc)/len(self.tr_temp_acc))
        self.tr_temp_loss = []
        self.tr_temp_acc = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.model.set_use_rf(False)
        pred_mask = self.forward(x)
        loss = self.loss_fn(pred_mask, y)
        self.log("train_loss", loss,prog_bar=True)
        self.va_temp_loss.append(loss)
        self.log("valid_loss", loss,prog_bar=True)
        self.va_temp_acc.append(self.accuracy(pred_mask,y))
        self.log("valid_accurcay", self.va_temp_acc[-1],prog_bar=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.va_loss.append(sum(self.va_temp_loss)/len(self.va_temp_loss))
        self.va_acc.append(sum(self.va_temp_acc)/len(self.va_temp_acc))
        self.va_temp_loss = []
        self.va_temp_acc = []
        print("------validation_epoch_end------")
    

    @staticmethod
    def accuracy(y_hat, y):
        return torch.mean((torch.argmax(y_hat,dim=1) == y)*1.0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--channel', type=int, default=1, help='Number of channels')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--rfr', type=bool, default=False, help='Use Receptive Field Regularization')
    parser.add_argument('--rf_reg_weight', type=float, default=0.1, help='Receptive Field Regularization weight')
    parser.add_argument('--rf_reg_layers', type=int, default=4, help='Receptive Field Regularization layer numbers')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    data_mod = PhcDataModule(
            images_dir='/home/MSAI/yzhang253/datasets/PhC_C2DH_U373/dev/img/', 
            masks_dir="/home/MSAI/yzhang253/datasets/PhC_C2DH_U373/dev/mask/",
            scale=0.5, mask_suffix='', 
            use_rf=args.rfr,
            batch_size=args.batch_size, num_workers=48
    )
    model = UNetLightning(
        use_rf=args.rfr,
        rf_reg_weight=args.rf_reg_weight,
        n_channels=args.channel,
        num_classes=args.classes,
        lr=args.lr,
        train_epochs=args.epochs,
        reg_layers=args.rf_reg_layers
    )
    trainer = Trainer(
        default_root_dir="lightning_logs/",
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=100,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, data_mod)
    torch.save([model.tr_loss,model.va_loss,model.tr_acc,model.va_acc],'stats_use_rf={}_rf_reg_weight={}_lr={}_epoch={}.pkl'.format(args.rfr, args.rf_reg_weight,args.lr,args.epochs))