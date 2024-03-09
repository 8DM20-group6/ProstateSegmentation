"""
Deep-learning & prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
#%% Imports
from dataset import prostateMRDataset
from utils import load_config
from models import u_net, vae
from train import Trainer

config = load_config("config.json")
train_loader, val_loader = prostateMRDataset(config)
model = u_net.UNet()
# model = vae.VAE()
#%%
trainer = Trainer(model=model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  config=config)
trainer.train()

