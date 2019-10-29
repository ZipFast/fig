from __future__ import print_function, division 
import os 
import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils


class LocationDataset(Dataset):
    def __init__(self, splotre, lable, transform = None):
        self.splotre = splotre 
        self.lable = lable 
        self.transform = transform 
    
    def __len__(self):
        return len(self.splotre)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        splot = self.splotre[idx] 
        lable = self.lable[idx]
        
        sample = {'dataset':splot, 'lable':lable}
        if self.transform:
            sample = self.transform(sample)
        return sample

class   ToTensor(object):
    def __call__(self, sample):
        dataset, lable = sample['dataset'], sample['lable'] 
        dataset = dataset.transpose((1, 0)) 
        return {
            'dataset': torch.from_numpy(dataset),
            'lable': torch.FloatTensor(lable)
        }

