import numpy as np
import pickle
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils.image_folder import make_dataset
import h5py
# from utils.utils import get_all_data_loaders,hsi_data_loader, hsi_normalize

class CustomDataset(Dataset):
    def __init__(self,data_root,conf,data_type,transforms):
        self.data_type = data_type
        self.data_root = data_root  
        self.transform = transforms
        self.paths = sorted(make_dataset(self.data_root,conf['max_dataset_size'])) 
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        path = self.paths[index]
        
        if self.data_type == 'HSI':
            img = hsi_data_loader(path)
            hsi = hsi_normalize(img)
            return hsi
        
        elif self.data_type == 'RGB':
            img = (Image.open(path).convert('RGB'))
            rgb = self.transform(img)
            return rgb
                
def hsi_normalize(data, max_=4096, min_ = 0, denormalize=False):
    """
    Using this custom normalizer for RGB and HSI images.  
    Normalizing to -1to1. It also denormalizes, with denormalize = True)
    """
    HSI_MAX = max_
    HSI_MIN = min_

    NEW_MAX = 1
    NEW_MIN = -1
    if(denormalize):
        scaled = (data - NEW_MIN) * (HSI_MAX - HSI_MIN)/(NEW_MAX - NEW_MIN) + HSI_MIN 
        return scaled.astype(np.float32)
    scaled  = (data - HSI_MIN) * (NEW_MAX - NEW_MIN)/(HSI_MAX - HSI_MIN)  + NEW_MIN
    return scaled.astype(np.float32)


def hsi_data_loader(path):
    """
    This loader is created for HSI images, which are present in HDF5 format and contain dataset with key as 'hsi'.
    In case you are using a different HSI dataset with a differt format, you'll have to modify this function. 
    """ 
    
    with h5py.File(path, 'r') as f:
        d = np.array(f['hsi'])
        hs_data = np.einsum('abc -> cab',d)
    return hs_data        
        
                            
