import json
import time
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from models.vae import get_noise

class Logger():
    """Plots images during training for visualization.
    """    
    def __init__(self, config, model, train_loader=None):
        self.config = config["train"]
        self.results_dir = self.determine_dir(model)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {self.results_dir}")
        self.vis = next(iter(train_loader))

    def determine_dir(self, model):
        if model.__class__.__name__=="UNet":
            RESULTS_DIR = "segmentation_results"
            self.modelname = "UNet"
        elif model.__class__.__name__=="VAE":
            RESULTS_DIR = "vae_results"
            self.modelname = "VAE"
            self.noise = get_noise(32, self.config["z_dim"], device=self.config["device"])
        else:
            raise Exception("What model is this bro?")
        
        timestr = time.strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / RESULTS_DIR / timestr      

    def visualize_train(self, model, epoch):
        if self.modelname=="UNet":
            predict_logits, _, _ = model(self.vis[0].to(self.config["device"]))
            heatmap = torch.sigmoid(predict_logits)
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(self.vis[0][0,:,:,:].squeeze().detach().cpu(), cmap="gray")
            axs[1].imshow(self.vis[1][0,:,:,:].squeeze().detach().cpu(), cmap="gray")
            axs[2].imshow(heatmap[0,:,:,:].squeeze().detach().cpu(), cmap="hot")
            matplotlib.use('Agg')
            vis_dir = self.results_dir / "train_imgs"
            vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{vis_dir}/{epoch}.png")
            plt.close()

        if self.modelname=="VAE":
            decoder = model.generator
            img_generated = decoder(self.noise) # (32, 1, 64, 64)
            matplotlib.use('Agg')
            plt.imshow(img_generated[0,0,:,:].detach().cpu(), cmap="gray")
            vis_dir = self.results_dir / "train_imgs"
            vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{vis_dir}/{epoch}.png")
            plt.close()

class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss

def load_config(filename):
    """Loads config from .json file

    Arguments:
        filename (string): path to .json file

    Returns:
        config dictionary
    """    
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_config(content, filename):
    """Writes dictionary to .json file

    Arguments:
        content (dictionary)
        filename (string to path)
    """    
    filename = Path(filename)
    with filename.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)