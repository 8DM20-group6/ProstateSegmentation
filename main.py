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

config = load_config("config.json")
train_loader, val_loader = prostateMRDataset(config)
print(f"Train: {len(train_loader)} | Val: {len(val_loader)}")

#%% ========== Train ==========
# model = u_net.UNet()
# model = vae.VAE()
# trainer = Trainer(model=model,
#                   train_loader=train_loader,
#                   val_loader=val_loader,
#                   config=config)

# trainer.train()

# %% ========== Generate images ==========
model = vae.VAE()
vae_test = Tester(model=model,
                  weights_path=R"vae_results\20240310_160304_epochs800_lr0.001_decay500\weights.pth",
                  config=config)
images = vae_test.evaluate()
vae_test.plot_images()

# %% ========== Segmentation ==========
model = u_net.UNet()
unet_test = Tester(model=model,
                   weights_path=R"segmentation_results\20240309_232510\weights.pth",
                   config=config,
                   test_loader=val_loader)

images = unet_test.evaluate()
unet_test.plot_images()
#%%
scores = unet_test.calc_scores()
# %%
# TODO:
# Images blurry -> disentangle latent vectors -> Beta-VAE -> add beta factor to KL loss term?
# Generate corresponding masks -> concatenate dim=1, channel dim?