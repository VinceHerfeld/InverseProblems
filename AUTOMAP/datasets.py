import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop
from torch.fft import fft2
from datetime import datetime
import glob
  

def rgb2gray(u):
    return torch.Tensor(0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2])

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        if not isinstance(root_dir, str):
            raise Exception("root_dir shoud be a string")

        self.images = []
        for Id in np.arange(204) :
            self.images += glob.glob(root_dir+"/P"+str(Id)+"/*")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_name = self.images[idx]
        img = rgb2gray(plt.imread(file_name))
        m = min(img.shape[0], img.shape[1])
        img = crop(img, top = 0, left = 0, height = m, width = m)
        return img[:], fft2(img)