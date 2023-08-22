import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder

from . import config    # Custom config file
from .visualize import plot_incorrect_preds


class Net(pl.LightningModule):
    def __init__(
            self,
            num_classes=10,
            dropout_percentage=0,
            norm='bn',
            num_groups=2,
            criterion=F.cross_entropy,
            learning_rate=0.001,
            weight_decay=0.0
        ):
        super(Net, self).__init__()

        # Define norm
        if norm == 'bn':
            self.norm = nn.BatchNorm2d
        elif norm == 'gn':
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=num_groups, num_channels=in_dim
            )
        elif norm == 'ln':
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=in_dim, num_channels=in_dim
            )
        
        #define loss
        self.criterion = criterion

        #define metrics
        self.accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=num_classes
        )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=num_classes
        )
        
        #define the optimizer hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        #prediction storage
        self.pred_store = {
            "test_preds": torch.tensor([]),
            "test_labels": torch.tensor([]),
            "test_incorrect": [] #?
        }
        self.log_store = {          # not used at all
            "train_loss_epoch": [],
            "train_acc_epoch": [],
            "val_loss_epoch": [],
            "val_acc_epoch": [],
            "test_loss_epoch": [],  # not used
            "test_acc_epoch": [],   # not used
        }

        # Define the network architecture
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 | 1 -> 32x32x64 | 3
            self.norm(64),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 32x32x128 | 5
            nn.MaxPool2d(2, 2),  # 16x16x128 | 6
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l1res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 10
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 14
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 16x16x256 | 18
            nn.MaxPool2d(2, 2),  # 8x8x256 | 19
            self.norm(256),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 8x8x512 | 27
            nn.MaxPool2d(2, 2),  # 4x4x512 | 28
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 36
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 44
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.maxpool = nn.MaxPool2d(4, 4)

        # Classifier
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.l1(x)
        x = x + self.l1res(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x + self.l3res(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        data, target = batch

        #forward pass
        pred = self.forward(data)

        #calculate loss
        loss = self.criterion(pred, target)

        #calculate accuracy
        accuracy = self.accuracy(pred, target)

        #log metrics
        self.log_dict(
            {"train_loss": loss, "train_acc": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
    

    def validation_step(self, batch, batch_idx):
        data, target = batch

        #forward pass
        pred = self.forward(data)

        #calculate loss
        loss = self.criterion(pred, target)

        #calculate accuracy
        accuracy = self.accuracy(pred, target)

        #log metrics
        self.log_dict(
            {"val_loss": loss, "val_acc": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch

        #forward pass
        pred = self.forward(data)
        argmax_pred = pred.argmax(dim=1).cpu() # why cpu here when down

        #calculate loss
        loss = self.criterion(pred, target)

        #calculate accuracy
        accuracy = self.accuracy(pred, target)

        #update confusion matrix
        self.confusion_matrix.update(pred, target)

        #log metrics
        self.log_dict(
            {"test_loss": loss, "test_acc": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        #store the predictions. labels and incorrect predictions

        #converting to cpu
        data, target, pred, argmax_pred = data.cpu(), target.cpu(), pred.cpu(), argmax_pred.cpu()

        #storing the predictions
        self.pred_store["test_preds"] = torch.cat((self.pred_store["test_preds"], argmax_pred), dim=0)
        self.pred_store["test_labels"] = torch.cat((self.pred_store["test_labels"], target), dim=0)

        for d, t, p, o in zip(data, target, argmax_pred, pred):
            if p.eq(t.view_as(p)).item() == False:
                self.pred_store["test_incorrect"].append(
                    (d.cpu(), t, p, o[p.item()].cpu())
                )

        return loss
    
    def find_bestLR_LRFinder(self, optimizer):

        lr_finder = LRFinder(self, optimizer, criterian = self.criterion)
        lr_finder.range_test(
            self.trainer.datamodule.train_dataloader(),
            end_lr=config.LRFINDER_END_LR,
            num_iter=config.LRFINDER_NUM_ITERATIONS,
            step_mode=config.LRFINDER_STEP_MODE 
        )
        # best_lr = None
        # Extract the loss and learning rate from history
        loss = np.array(lr_finder.history['loss'])
        lr = np.array(lr_finder.history['lr'])
    
        # Find the learning rate with steepest negative gradient
        gradient = np.gradient(loss)
        idx = np.argmin(gradient)
        best_lr = lr[idx]
        
        try:
            _, y = lr_finder.plot() 
        except Exception as e:
            pass

        print("BEST_LR: ", best_lr)
        lr_finder.reset()
        
        return best_lr
    
    def configure_optimizers(self):
        optimizer = self.get_only_optimizer()
        best_lr = self.find_bestLR_LRFinder(optimizer)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=best_lr, #used best_lr insted of hard coded values
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=config.NUM_EPOCHS,
            pct_start=5 / config.NUM_EPOCHS,
            div_factor=config.OCLR_DIV_FACTOR,
            three_phase=config.OCLR_THREE_PHASE,
            final_div_factor=config.OCLR_FINAL_DIV_FACTOR,
            anneal_strategy=config.OCLR_ANNEAL_STRATEGY
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def get_only_optimizer(self):
        optimizer = optim.Adam(
            self.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
    
    def on_test_end(self) -> None:
        super().on_test_end()

        #Confusion Matrix
        confmat = self.confusion_matrix.cpu().compute().numpy()
        if config.NORM_CONF_MAT:
            df_confmat = pd.DataFrame(
                confmat / np.sum(confmat, axis=1)[:, None],
                index=[i for i in config.CLASSES],
                columns=[i for i in config.CLASSES],
            )
        else:
            df_confmat = pd.DataFrame(
                confmat,
                index=[i for i in config.CLASSES],
                columns=[i for i in config.CLASSES],
            )
        plt.figure(figsize=(7, 5))
        sns.heatmap(df_confmat, annot=True, cmap="Blues", fmt=".3f", linewidths=0.5)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def plot_incorrect_predictions_helper(self, num_imgs=10):
        plot_incorrect_preds(
            self.pred_store["test_incorrect"], config.CLASSES, num_imgs
        )