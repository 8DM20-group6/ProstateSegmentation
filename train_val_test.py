import torch
import matplotlib.pyplot as plt
import numpy as np
import seg_metrics.seg_metrics as sg
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from utils import Logger, DiceBCELoss
from tqdm import tqdm
from models.vae import vae_loss, get_noise

class Trainer():
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config["train"]        
        self.device = self.config["device"]
        self.epochs = self.config["epochs"]
        self.decay_lr_after = self.config["decay_lr_after"]
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = Logger(config, model, train_loader)
        self.lr = self.get_lr()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=self.lr_lambda)
        self.loss_fn = self.determine_lossfn(model)
        self.writer = SummaryWriter(log_dir=self.logger.results_dir / "tensorboard")
        self.noise = get_noise(32, self.config["z_dim"], device=self.device)
        print(self.loss_fn, self.lr)

    def train(self):
        best_val_loss = 999        
        for epoch in range(self.epochs):
            current_train_loss = 0.0
            current_valid_loss = 0.0
            current_recon_train_loss = 0.0
            current_recon_valid_loss = 0.0
            current_kld_train_loss = 0.0
            current_kld_valid_loss = 0.0

            self.model.train()    
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch: {epoch+1}/{self.epochs}")        
                for img, mask in tepoch:
                    self.optimizer.zero_grad() # 1   
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar = self.model(img) # 2
                    # print(mu.shape, logvar.shape)
                    if self.logger.modelname=="UNet":
                        loss = self.loss_fn(img_recon, mask)
                    if self.logger.modelname=="VAE":
                        # loss = self.loss_fn(img, img_recon, mu, logvar)
                        loss, recon_loss, kld_loss = self.loss_fn(img, img_recon, mu, logvar)
                    current_train_loss+=loss
                    current_recon_train_loss+=recon_loss
                    current_kld_train_loss+=kld_loss
                    loss.backward()
                    self.optimizer.step()

            # evaluate validation loss
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                for img, mask in self.val_loader:
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar = self.model(img) # 2
                    if self.logger.modelname=="UNet":
                        loss = self.loss_fn(img_recon, mask)
                    if self.logger.modelname=="VAE":
                        # loss = self.loss_fn(img, img_recon, mu, logvar)
                        loss, recon_loss, kld_loss = self.loss_fn(img, img_recon, mu, logvar)
                    current_valid_loss+=loss
                    current_recon_valid_loss+=recon_loss
                    current_kld_valid_loss+=kld_loss
                # vae_model.train() # turns training setting back on

            #print(f"Train: {current_train_loss / len(self.train_loader):.4f} | Validation: {current_valid_loss / len(self.val_loader):.4f}")
            # write to tensorboard log
            self.writer.add_scalars("Loss/train", {
                                    "Total loss": current_train_loss / len(self.train_loader),
                                    "Recon_loss": current_recon_train_loss / len(self.train_loader),
                                    "KLD_loss": current_kld_train_loss / len(self.train_loader),
                                    }, epoch)
            self.writer.add_scalars("Loss/train", {
                                    "Total loss": current_valid_loss / len(self.val_loader),
                                    "Recon_loss": current_recon_valid_loss / len(self.val_loader),
                                    "KLD_loss": current_kld_valid_loss / len(self.val_loader),
                                    }, epoch)
  
        

            if self.logger.modelname=="VAE":
                self.scheduler.step() # step the learning step scheduler

            # ===TO BE IMPLEMENTED===
            # save examples of real/fake images
            # if (epoch + 1) % DISPLAY_FREQ == 0:
            #     img_grid = make_grid(
            #         torch.cat((img_recon[:5], img[:5])), nrow=5, padding=12, pad_value=-1
            #     )
            #     writer.add_image(
            #         "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
            #     )
                
            # TODO: sample noise 
            self.logger.visualize_train(self.model, epoch)
            # TODO: generate images and display
            if current_valid_loss<best_val_loss:
                torch.save(self.model.state_dict(), self.logger.results_dir / "weights.pth")

    def lr_lambda(self, the_epoch):
        """Function for scheduling learning rate"""
        return (
            1.0
            if the_epoch < self.decay_lr_after
            else 1 - float(the_epoch - self.decay_lr_after) / (self.epochs - self.decay_lr_after)
        )    

    def determine_lossfn(self, model):
        if model.__class__.__name__=="UNet":
            loss_fn = DiceBCELoss()
        elif model.__class__.__name__=="VAE":
            loss_fn = vae_loss
        else:
            raise Exception("What model is this bro?")
        
        return loss_fn
    
    def get_lr(self):
        if self.logger.modelname=="UNet":  
            self.lr = self.config["lr_unet"]
        else:
            self.lr = self.config["lr_vae"]
        return self.lr

class Tester():
    def __init__(self, model, weights_path, config, test_loader=None):
        self.config = config["train"]
        self.z_dim = self.config["z_dim"]
        self.device = self.config["device"]
        self.modelname = get_modelname(model)
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.test_loader = test_loader

    def evaluate(self):
        if self.modelname=="UNet":
            images = list()
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                for img, mask in self.test_loader:
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    prediction_mask, _, _ = self.model(img) # 2
                    images.append([img, mask, prediction_mask])

        if self.modelname=="VAE":
            noise = get_noise(32, self.z_dim, self.device)
            decoder = self.model.generator
            images = list()
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                for img in noise:
                    img = img.to(self.device)
                    img_generated = decoder(img)
                    images.append(img_generated)

        self.images = images
        return self.images
    
    def plot_images(self):
        if self.modelname=="UNet":
            rows = len(self.images)
            _, axs = plt.subplots(rows, 4, figsize=(10, (8/3)*rows))
            for i, ax in enumerate(axs):
                ax[0].imshow(self.images[i][0][0,:,:,:].squeeze().detach().cpu(), cmap="gray")
                ax[1].imshow(self.images[i][1][0,:,:,:].squeeze().detach().cpu(), cmap="gray")
                heatmap = torch.sigmoid(self.images[i][2])[0,:,:,:].squeeze().detach().cpu()
                ax[2].imshow(heatmap, cmap="hot")
                pred_mask = np.round(heatmap)
                sub_mask = self.images[i][1][0,:,:,:].squeeze().detach().cpu()-2*pred_mask
                import matplotlib.patches as mpatches
                sub_mask = sub_mask.numpy()
                cmap = {-2:[1.0,0.0,0.0,1],
                        -1:[0.0,1.0,0.0,1],
                         1:[0.0,0.0,1.0,1],
                         0:[0.0,0.0,0.0,1],}
                labels = {-2:'FP', -1:'TP', 1:'FN', 0:'bg',}
                patches =[mpatches.Patch(color=cmap[j], label=labels[j]) for j in cmap]
                overlay = np.array([[cmap[k] for k in j] for j in sub_mask])    
                ax[3].imshow(overlay, interpolation="none")
                ax[3].legend(handles=patches, loc="upper right", labelspacing=0.1)

            plt.subplots_adjust(wspace=0.025, hspace=0.05)

        if self.modelname=="VAE":
            rows = len(self.images) // 4
            _, axs = plt.subplots(rows, 4, figsize=(8, (8/4)*rows))
            for row, ax in enumerate(axs):
                ax[0].imshow(self.images[row*4+0].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[1].imshow(self.images[row*4+1].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[2].imshow(self.images[row*4+2].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[3].imshow(self.images[row*4+3].squeeze().squeeze().detach().cpu(), cmap="gray")

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.025, hspace=0.05)
        plt.show()

    def calc_scores(self):
        out = list()
        for _, (_, mask, pred) in enumerate(self.images):
            file_metric = list()
            for i in range(mask.shape[0]):
                if len(np.unique(mask[i,:,:,:].detach().cpu().numpy()))==2:
                    gt_mask = mask[i,:,:,:].detach().cpu().numpy()
                    pred_mask = np.round(torch.sigmoid(pred[i,:,:,:]).detach().cpu().numpy())
                    metrics = sg.write_metrics(labels=np.unique(gt_mask).tolist(),
                                            gdth_img=gt_mask,
                                            pred_img=pred_mask,
                                            metrics=['hd', 'hd95','dice','fpr','fnr'])
                    file_metric.append(metrics)
            out.append(file_metric)
        return out
              
def get_modelname(model):
    if model.__class__.__name__=="UNet":
        modelname = "UNet"
    elif model.__class__.__name__=="VAE":
        modelname ="VAE"
    return modelname