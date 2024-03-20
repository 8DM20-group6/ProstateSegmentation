import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
from pathlib import Path
from torch.utils.data import DataLoader
from models.vae import get_noise

class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size):
        random.seed(42)
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(
                    np.int32
                )
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                    np.int32
                )
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(mode="I"),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return (
            self.norm_transform(
                self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
            ),
            self.img_transform(
                (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
            ),
        )
    
class ExtendDataset(torch.utils.data.Dataset):
    def __init__(self, config, base_dataset, vae_model, seed=False):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config["dataloader"]
        self.nr_synthetic_imgs = self.config["nr_synthetic_imgs"]
        self.batch_size = self.config["batch_size"]
        self.length = len(self.base_dataset) + self.nr_synthetic_imgs*self.batch_size
        self.seed = seed
        self.vae_model = vae_model.to(self.config["device"])
        print(self.length)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index >= self.length:
            f"index should be smaller than {self.length}"
            raise IndexError(f"index should be smaller than {self.length}")
        if index >= len(self.base_dataset):
            if self.seed:
                seed = index
            else:
                seed = False

            noise = get_noise(n_samples=1, 
                              z_dim=self.config["z_dim"],
                              device=self.config["device"],
                              seed=seed)
            self.vae_model.eval()
            decoder = self.vae_model.generator
            decoder_mask = self.vae_model.generator_mask
            with torch.no_grad():
                img = decoder(noise)
                mask = decoder_mask(noise)
            img, mask = img.squeeze(), mask.squeeze()
            img, mask = img.unsqueeze(0), mask.unsqueeze(0)
            mask = np.round(torch.sigmoid(mask.detach().cpu())) #sigmoid 0..1
            mean = torch.mean(img)
            std = torch.std(img)
            transform = transforms.Normalize(mean, std, True)
            transform(img)

        else:
            img, mask = self.base_dataset[index]

        return img.to(self.config["device"]), mask.to(self.config["device"])
            

def prostateMRDataset(config, vae_model=None, seed=False):
    DATA_DIR = Path.cwd().parent / config["dataloader"]["data_dir"]
    NO_VALIDATION_PATIENTS = 2
    IMAGE_SIZE = config["dataloader"]["image_size"]
    BATCH_SIZE = config["dataloader"]["batch_size"]

    patients = [
        path
        for path in DATA_DIR.glob("*")
        if not any(part.startswith(".") for part in path.parts)
    ]
    
    random.seed(42)
    random.shuffle(patients)

    partition = {
        "train": patients[:-NO_VALIDATION_PATIENTS],
        "validation": patients[-NO_VALIDATION_PATIENTS:],
    }
    dataset = ProstateMRDataset(partition["train"], IMAGE_SIZE)
    if vae_model:
        dataset = ExtendDataset(config, dataset, vae_model, seed)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=False)

    valid_dataset = ProstateMRDataset(partition["validation"], IMAGE_SIZE)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=False)
        
    return dataloader, valid_dataloader