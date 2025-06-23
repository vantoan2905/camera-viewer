# load_dataset

import os
import json
import torch
from PIL import Image


class LoadDataset():
    def __init__(self, path_image, path_label, transform=None):
        """
        Args:
            path_image (str): Path to the folder containing the input images.
            path_label (str): Path to the folder containing the corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.path_image = path_image
        self.path_label = path_label
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.path_image))
    
    def __getitem__(self, idx):
        """
        Gets a sample from the dataset.
        
        Parameters:
            idx (int): Index of the sample.
        
        Returns:
            image (PIL.Image): The input image.
            labels (dict): The labels corresponding to the input image.
        """
        image_files = os.listdir(self.path_image)
        label_files = os.listdir(self.path_label)
        
        image_path = os.path.join(self.path_image, image_files[idx])
        label_path = os.path.join(self.path_label, label_files[idx])
        
        image = Image.open(image_path).convert("RGB")
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
    