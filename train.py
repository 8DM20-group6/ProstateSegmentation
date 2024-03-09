import torch
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
        if self.logger.modelname=="UNet":  
            self.lr = self.config["lr_unet"]
        else:
            self.lr = self.config["lr_vae"]
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
            
            self.model.train()    
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch: {epoch+1}/{self.epochs}")        
                for img, mask in tepoch:
                    self.optimizer.zero_grad() # 1   
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar = self.model(img) # 2
                    if mu is None:
                        loss = self.loss_fn(img_recon, mask)
                    else:
                        loss = self.loss_fn(img, img_recon, mu, logvar)
                    current_train_loss+=loss
                    loss.backward()
                    self.optimizer.step()

            # evaluate validation loss
            with torch.no_grad():
                self.model.eval() # turns off the training setting to allow evaluation 
                for img, mask in self.val_loader:
                    img, mask = img.to(self.device), mask.float().to(self.device)
                    img_recon, mu, logvar = self.model(img) # 2
                    if mu is None:
                        loss = self.loss_fn(img_recon, mask)
                    else:
                        loss = self.loss_fn(img, img_recon, mu, logvar) # 3
                    current_valid_loss+=loss
                # vae_model.train() # turns training setting back on

            print(f"Train: {current_train_loss / len(self.train_loader):.4f} | Validation: {current_valid_loss / len(self.val_loader):.4f}")
            # write to tensorboard log
            self.writer.add_scalar("Loss/train", current_train_loss / len(self.train_loader), epoch)
            self.writer.add_scalar(
                "Loss/validation", current_valid_loss / len(self.val_loader), epoch
            )
            if self.logger.modelname=="VAE":
                self.scheduler.step() # step the learning step scheduler

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

