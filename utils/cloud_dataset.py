import torch
from torch.utils.data import Dataset
import cv2
# import pandas as pd
import numpy as np

from .dataset_helper import rle_decode

class CloudDataset(Dataset):
    def __init__(self, data_df, data_folder, transforms=None, preprocessing=None):
        """
        data_df: dataframe. 
          The dataframe for train / valid dataset, read from the csv file. 
        data_folder: string. 
          The full path to the dataset. 
        """
        self.data_csv = data_df
        self.data_folder = data_folder
        self.transforms = transforms
        self.preprocessing = preprocessing

    def _make_mask(self, image_name, shape=(1400, 2100)):
        """
        Create mask for a given image name and shape.
    
        [OUTPUTS]:
        masks: an array with shape (shape[0], shape[1], 4).
          Mask for each class labels.
        """
        encoded_masks = self.data_csv.loc[self.data_csv['image_name'] == image_name, 'EncodedPixels']
        masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

        # Here we assume the encoded masks always follow the same order of labels:
        # Fish, Flower, Gravel, Sugar.
        for idx, label in enumerate(encoded_masks.values):
            if label is not np.nan:
                mask = rle_decode(label, shape)
                masks[:, :, idx] = mask

        return masks

    def _get_original_item(self, image_name):
        img = cv2.imread(f'{self.data_folder}/{image_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self._make_mask(image_name, img.shape[0:2])

        return img, masks

    def get_data_by_index(self, idx):
        """
        Get an original image / mask pair for a given index. 
        """
        image_name = self.data_csv['image_name'].unique()[idx]
        img, masks = self._get_original_item(image_name)

        img = img.transpose(2, 0, 1)
        masks = masks.transpose(2, 0, 1)
        
        return img, masks


    def __getitem__(self, idx):
        """
        Get a data sample (in the format of X/y) by index. 
    
        NOTE: Here we aim at producing X/y pair for each row of the records, so y is
        the mask for a certain class (one of Flower/Sugar/Gravel/Salt). 
        """
        image_name = self.data_csv['image_name'].unique()[idx]

        # img = cv2.imread(f'{self.data_folder}/{image_name}')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # masks = self._make_mask(image_name, img.shape[0:2])
        img, masks = self._get_original_item(image_name)

        if self.transforms:
            augmented = self.transforms(image=img, mask=masks)
            img = augmented['image']
            masks = augmented['mask']

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=masks)
            img = preprocessed['image']
            masks = preprocessed['mask']
        return img, masks

    def __len__(self):
        return self.data_csv['image_name'].nunique()