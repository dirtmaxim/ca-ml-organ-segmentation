import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # get original image
        ribs_img_name = os.path.join(self.root_dir,
                                     self.csv_file.iloc[idx]['jsrt_png_imgs'])
        original_image = cv2.imread(ribs_img_name)[..., 0]
        original_image = np.expand_dims(original_image, axis=2)
        
        # get masks
        # heart
        heart_mask_name = os.path.join(self.root_dir,
                                      self.csv_file.iloc[idx]['heart_png_masks'])
        heart_mask = cv2.imread(heart_mask_name)[..., 0]
        # lungs
        lungs_mask_name = os.path.join(self.root_dir,
                                       self.csv_file.iloc[idx]['lungs_png_masks'])
        lungs_mask = cv2.imread(lungs_mask_name)[..., 0]
        # clavicles
        clavicles_mask_name = os.path.join(self.root_dir,
                                           self.csv_file.iloc[idx]['clavicles_png_masks'])
        clavicles_mask = cv2.imread(clavicles_mask_name)[..., 0]
        
        target_image = np.zeros((heart_mask.shape[0], heart_mask.shape[1], 3))
        # create 3-channel target image of masks
        target_image[..., 0] = heart_mask
        target_image[..., 1] = lungs_mask
        target_image[..., 2] = clavicles_mask
        
        target_image /= 255.

        if self.transform:
            augmented = self.transform(image=original_image, mask=target_image)
            original_image = augmented['image']
            target_image = augmented['mask']
        
        original_image = (np.float32(original_image) - 127.) / 128.
        
        original_image = torch.Tensor(original_image.swapaxes(0, 2).swapaxes(1, 2))
        target_image = torch.Tensor(target_image.swapaxes(0, 2).swapaxes(1, 2))

        sample = {'image': original_image, 
                  'masks': target_image}
        return sample
    
class AlbumentationsDataset6channels(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # get original image
        ribs_img_name = os.path.join(self.root_dir,
                                     self.csv_file.iloc[idx]['jsrt_png_imgs'])
        original_image = cv2.imread(ribs_img_name)[..., 0]
        original_image = np.expand_dims(original_image, axis=2)
        
        # get masks
        # heart
        heart_mask_name = os.path.join(self.root_dir,
                                      self.csv_file.iloc[idx]['heart_png_masks'])
        heart_mask = cv2.imread(heart_mask_name)[..., 0]
        heart_contour_name = os.path.join(self.root_dir,
                                          self.csv_file.iloc[idx]['heart_png_contours'])
        heart_contour = cv2.imread(heart_contour_name)[..., 0]
        # lungs
        lungs_mask_name = os.path.join(self.root_dir,
                                       self.csv_file.iloc[idx]['lungs_png_masks'])
        lungs_mask = cv2.imread(lungs_mask_name)[..., 0]
        lungs_contour_name = os.path.join(self.root_dir,
                                          self.csv_file.iloc[idx]['lungs_png_contours'])
        lungs_contour = cv2.imread(lungs_contour_name)[..., 0]
        # clavicles
        clavicles_mask_name = os.path.join(self.root_dir,
                                           self.csv_file.iloc[idx]['clavicles_png_masks'])
        clavicles_mask = cv2.imread(clavicles_mask_name)[..., 0]
        clavicles_contour_name = os.path.join(self.root_dir,
                                          self.csv_file.iloc[idx]['clavicles_png_contours'])
        clavicles_contour = cv2.imread(clavicles_contour_name)[..., 0]
        
        target_image = np.zeros((heart_mask.shape[0], heart_mask.shape[1], 6))
        # create 3-channel target image of masks
        target_image[..., 0] = heart_mask
        target_image[..., 1] = lungs_mask
        target_image[..., 2] = clavicles_mask
        target_image[..., 3] = heart_contour
        target_image[..., 4] = lungs_contour
        target_image[..., 5] = clavicles_contour
        
        target_image /= 255.

        if self.transform:
            augmented = self.transform(image=original_image, mask=target_image)
            original_image = augmented['image']
            target_image = augmented['mask']
        
        original_image = (np.float32(original_image) - 127.) / 128.
        
        original_image = torch.Tensor(original_image.swapaxes(0, 2).swapaxes(1, 2))
        target_image = torch.Tensor(target_image.swapaxes(0, 2).swapaxes(1, 2))

        sample = {'image': original_image, 
                  'masks': target_image}
        return sample

