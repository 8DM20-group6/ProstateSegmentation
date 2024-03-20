# 8DM20 - CSMIA - Deep-learning

This repository contains a PyTorch implementation used for the Capita Selecta in Medical Imaging Analysis project, hosted by University of Technology Eindhoven. The objective is to segment the prostate in MR images. To this end, we train a Variational Auto-Encoder to generate synthetic prostate images along with its corresponding binary mask. The generated segmentations are then used to improve the U-Net segmentation model. The workflow and usage of our method is described below. 
## Group 6

* Olga Capmany Dalmas
* Paula Del Popolo
* Zahra Farshidrokh
* Daniel Le
* Jelle van der Pas
* Marcus Vroemen

## Quick usage
```
See main.py
```
## Dependencies

* Python==3.10.9 (may work with other versions)
* matplotlib==3.8.2
* numpy==1.25.2
* SimpleITK==2.3.1
* SimpleITK==2.3.1
* torch==2.0.1+cu117
* torchvision==0.15.2+cu117
* tqdm==4.66.2

```bash
pip install -r requirements.txt
```
## Folder Structure
Because the `TrainingData` folder is used for this part as well as the registration part it should be located in the parent directory (directory above working directory).
```
ProstateSegmentation
├───main.py - main script 
├───train.py - holds training class
├───utils.py - contains utility methods, logger
├───config.json - holds configuration settings
│
├───models
│   ├───u_net.py 
│   └───vae.py 
│
├───segmentation_results
│   └───%Y%m%d_%H%M%S - contains segmentation weights and plots
│   
└───vae_results
    └───%Y%m%d_%H%M%S - contains data generation weights and plots

```

## Config file 
The config file is in `.json` file format and contains parameters used for data loading and training.

```JSON
{
    "dataloader": {
        "data_dir": "TrainingData",       // Foldername data (IN PARENT DIR!)
        "validation_patients": 2,         // Validation subset size
        "image_size": [64, 64],           // Training image dimensions
        "batch_size": 32                  // Batch size
    },
        
    "train": {
        "device": "cuda",                 // Computation device
        "epochs": 5000,                    // Number of epochs
        "lr_vae": 0.0005,                  // Learning rate VAE model
        "lr_unet": 0.0001,                // Learning rate UNet model
        "decay_lr_after": 5000,             // After this epoch decay LR
        "z_dim": 256                      // Latent vector dimension
    }
}
```