import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seg_metrics.seg_metrics as sg
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from utils import Logger, DiceBCELoss
from tqdm import tqdm
from models.vae import vae_loss, get_noise
import pandas as pd

class Trainer():
    def __init__(self, model, train_loader, val_loader, config, vae_model=None):
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
        self.loss_fn_mask = nn.BCELoss()
        self.writer = SummaryWriter(log_dir=self.logger.results_dir / "tensorboard")
        self.noise = get_noise(32, self.config["z_dim"], device=self.device)
        self.beta = self.frange_cycle_linear(n_epoch=self.epochs, ratio=1)
        self.vae_model=vae_model
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

            current_mask_train_loss = 0.0

            self.model.train()    
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch: {epoch+1}/{self.epochs}")        
                for img, mask in tepoch:
                    self.optimizer.zero_grad() # 1   
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar, mask_recon = self.model(img) # 2
                    # print(mu.shape, logvar.shape)
                    if self.logger.modelname=="UNet":
                        loss = self.loss_fn(img_recon, mask)
                    if self.logger.modelname=="VAE":
                        # loss = self.loss_fn(img, img_recon, mu, logvar)
                        loss, recon_loss, kld_loss = self.loss_fn(inputs=img, 
                                                                  recons=img_recon, 
                                                                  mu=mu, 
                                                                  logvar=logvar, 
                                                                  beta=self.beta[epoch])
                        mask_loss = self.loss_fn_mask(torch.sigmoid(mask_recon), mask)
                        current_recon_train_loss+=recon_loss
                        current_kld_train_loss+=kld_loss
                        current_mask_train_loss+=mask_loss
                        mask_loss.backward()

                    current_train_loss+=loss
                    loss.backward()
                    self.optimizer.step()
            #fig, axs= plt.subplots(1, 2)
           # axs[0].imshow(mask[0,0,:,:].detach().cpu())
           # axs[1].imshow(mask_recon[0,0,:,:].detach().cpu())
          #  plt.show()
          #  plt.close()
            # evaluate validation loss
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                for img, mask in self.val_loader:
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar, _ = self.model(img) # 2
                    if self.logger.modelname=="UNet":
                        loss = self.loss_fn(img_recon, mask)
                    if self.logger.modelname=="VAE":
                        # loss = self.loss_fn(img, img_recon, mu, logvar)
                        loss, recon_loss, kld_loss = self.loss_fn(inputs=img, 
                                                                  recons=img_recon, 
                                                                  mu=mu, 
                                                                  logvar=logvar, 
                                                                  beta=self.beta[epoch])
                        current_recon_valid_loss+=recon_loss
                        current_kld_valid_loss+=kld_loss
                    current_valid_loss+=loss

            #print(f"Train: {current_train_loss / len(self.train_loader):.4f} | Validation: {current_valid_loss / len(self.val_loader):.4f}")
            # write to tensorboard log
            self.writer.add_scalars("Loss/train", {
                                    "Total loss": current_train_loss / len(self.train_loader),
                                    # "Recon_loss": current_recon_train_loss / len(self.train_loader),
                                    # "KLD_loss": current_kld_train_loss / len(self.train_loader),
                                    # "Mask loss": current_mask_train_loss / len(self.train_loader)
                                    }, epoch)
            self.writer.add_scalars("Loss/val", {
                                    "Total loss": current_valid_loss / len(self.val_loader),
                                    # "Recon_loss": current_recon_valid_loss / len(self.val_loader),
                                    # "KLD_loss": current_kld_valid_loss / len(self.val_loader),
                                    }, epoch)
            # self.writer.add_scalars("Parameters", {
            #                         "Beta": self.beta[epoch],
            #                         }, epoch)
  
            # if self.logger.modelname=="VAE":
                # self.scheduler.step() # step the learning step scheduler

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
    
    def frange_cycle_linear(self, start=0.0, stop=1.0, n_epoch=0, n_cycle=10, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L    

class Tester():
    def __init__(self, model, weights_path, config, test_loader=None):
        self.config = config["train"]
        self.z_dim = self.config["z_dim"]
        self.device = self.config["device"]
        self.modelname = get_modelname(model)
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.weights_path = weights_path
        self.test_loader = test_loader
        self.model.eval()

    def evaluate(self, linear=None):
        if self.modelname=="UNet":
            images = list()
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation
                torch.manual_seed(9)
                for img, mask in self.test_loader:
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    prediction_mask, _, _, _ = self.model(img) # 2
                    images.append([img, mask, prediction_mask])

            self.images = images
            
        if self.modelname=="VAE":
            if linear:
                noise1 = get_noise(1, self.config["z_dim"])
                noise2 = get_noise(1, self.config["z_dim"])
                noise = torch.tensor(np.linspace(noise1, noise2, 32), device=self.device)
            else:
                noise = get_noise(32, self.z_dim, self.device)
                # print(noise[0][:])

            decoder = self.model.generator
            decoder_mask = self.model.generator_mask
            images = list()
            mask = list()
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                self.img_generated = decoder(noise)
                self.mask_generated = np.round(torch.sigmoid(decoder_mask(noise).detach().cpu()))
        # self.images = images[0]
        # return self.images
    
    # def generate_mask(self):

    def plot_images(self):
        if self.modelname=="UNet":
            rows = len(self.images)
            _, axs = plt.subplots(rows, 4, figsize=(10, (8/3)*rows))
            for i, ax in enumerate(axs):
                ax[0].imshow(self.images[i][0][15,:,:,:].squeeze().detach().cpu(), cmap="gray")
                ax[1].imshow(self.images[i][1][15,:,:,:].squeeze().detach().cpu(), cmap="gray")
                heatmap = torch.sigmoid(self.images[i][2])[15,:,:,:].squeeze().detach().cpu()
                ax[2].imshow(heatmap, cmap="hot")
                pred_mask = np.round(heatmap)
                sub_mask = self.images[i][1][15,:,:,:].squeeze().detach().cpu()-2*pred_mask
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
            rows = len(self.img_generated) // 4
            _, axs = plt.subplots(rows, 5, figsize=(8, (8/4)*rows),
                                  gridspec_kw={'width_ratios': [1, 1, 0.05, 1, 1]}
)
            for row, ax in enumerate(axs):
                ax[0].imshow(self.img_generated[row*2+0].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[1].imshow(self.mask_generated[row*2+0].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[3].imshow(self.img_generated[row*2+1].squeeze().squeeze().detach().cpu(), cmap="gray")
                ax[4].imshow(self.mask_generated[row*2+1].squeeze().squeeze().detach().cpu(), cmap="gray")

        for ax in axs[:, 2]: 
            ax.axis("off")  
            
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.025, hspace=0.05)
        plt.show()

    def reconstruct(self, image, mask):
        img_recon, mu, logvar, mask_recon = self.model(image.to(self.device)) # 2
        mask_recon = torch.sigmoid(mask_recon.detach().cpu())
        rows = len(img_recon) // 4
        _, axs = plt.subplots(rows, 4, figsize=(8, (8/4)*rows))
        for row, ax in enumerate(axs):
            ax[0].imshow(image[row].squeeze().squeeze().detach().cpu(), cmap="gray")
            ax[1].imshow(mask[row].squeeze().squeeze().detach().cpu(), cmap="gray")
            ax[2].imshow(img_recon[row].squeeze().squeeze().detach().cpu(), cmap="gray")
            ax[3].imshow(np.round(mask_recon[row].squeeze().squeeze().detach().cpu()), cmap="gray")

        titles = ["Image", "Mask", "Image Recon", "Mask recon"]
        for title, ax in zip(titles, axs[0]):
            ax.set_title(title)

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
                pred_mask = np.round(torch.sigmoid(pred[i,:,:,:]).detach().cpu().numpy())
                gt_mask = mask[i,:,:,:].detach().cpu().numpy()
                if len(np.unique(pred_mask))==2 and len(np.unique(gt_mask))==2:
                    metrics = sg.write_metrics(labels=np.unique(pred_mask).tolist(),
                                            gdth_img=gt_mask,
                                            pred_img=pred_mask,
                                            TPTNFPFN=False,
                                            spacing=[0.488281, 0.488281, 1],
                                            metrics=['hd', 'hd95','dice','recall','fpr','fnr'])

                    file_metric.append(metrics)
            print(len(file_metric))
            out.append(file_metric)
        return out
    
    def show_scores(self, scores):
        pd_data = list()
        for i, batch in enumerate(scores):
            metric_batch_averages = list()
            metric_batch_averages.append(f"Batch {i}")
            for metric in ['hd', 'hd95','dice','recall','fpr','fnr']:
                metric_total = list()

                for slice in batch:
                    metric_total.append(slice[0][metric][1])
                metric_avg = np.mean(metric_total)
                metric_batch_averages.append(round(metric_avg, 3))
            
            pd_data.append(metric_batch_averages)

        df = pd.DataFrame(pd_data, columns=["Batch nr.",
                                            "HD",
                                            "HD95",
                                            "DSC",
                                            "Recall",
                                            "FPR",
                                            "FNR"])
        df.style.set_caption(self.weights_path)

        display(df)
        return df



              
def get_modelname(model):
    if model.__class__.__name__=="UNet":
        modelname = "UNet"
    elif model.__class__.__name__=="VAE":
        modelname ="VAE"
    return modelname