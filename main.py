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
from train_val_test import Trainer, Tester
import torch

# vanilla
# beta = 1
# cyclical
# beta = 5
# beta = 10
config = load_config("config.json")
vae_model = vae.VAE()
vae_model.load_state_dict(torch.load(R"beta_10.pth"))
train_loader, val_loader = prostateMRDataset(config=config, vae_model=vae_model, seed=True)
# train_loader, val_loader = prostateMRDataset(config=config)

print(f"Train: {len(train_loader)} | Val: {len(val_loader)}")

#%% ========== Train ==========
model = u_net.UNet()
# model = vae.VAE()
trainer = Trainer(model=model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  config=config)

# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(8,4.0))
# stride = max( int(trainer.epochs / 8), 1)
# plt.plot(range(trainer.epochs), trainer.beta, '-', label='Cyclical', marker= 's', color='k', markevery=stride,lw=2,  mec='k', mew=1 , markersize=10)

#%%
trainer.train()

# %% ========== Generate images ==========
model = vae.VAE()
vae_test = Tester(model=model,
                  weights_path=R"beta_10.pth",
                #   weights_path=R"beta_weights.pth",
                  config=config)
#%%
images = vae_test.evaluate(linear=False)
vae_test.plot_images()
# %% ========== Reconstruction ==========
img, mask = next(iter(train_loader))
vae_test.reconstruct(img, mask)

# %% ========== Segmentation ==========
model = u_net.UNet()
unet_test = Tester(model=model,
                   weights_path=R"segmentation_results\20240319_225054_epochs150_lr0.0001_decay5000\weights.pth",
                   config=config,
                   test_loader=val_loader)

images = unet_test.evaluate()
unet_test.plot_images()
#%% Calculate
scores = unet_test.calc_scores()
df = unet_test.show_scores(scores)

# 1) Beta = 1
# 2) Beta linear cyclical annealign between 0 and 1 (emphasis on reconstruction)
# 3) Beta = 5
# 4) Beta = 10
# 
# Train segmentatino model and use best model.

# synthesis image --> segmentation model --> HD95, DSC, FPR, FNR, TPR 
# %%
