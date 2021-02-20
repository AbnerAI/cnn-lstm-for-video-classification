import os
import sys
import time
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from scipy import ndimage, misc
from torch.utils.data import Dataset, DataLoader


def default_loader(path):
    return [Image.open(path[i][0]).convert('RGB') for i in range(len(path))]


class Ultra_Dataset_Train(Dataset): 
    def __init__(self, data_dir, transform, seq_len, loader=default_loader): 
        fh = open(data_dir, 'r')
        data_lis = [] 
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data_lis.append((words[0],words[1]))
        
        self.data_lis = data_lis
        self.transform = transform
        self.loader = loader
        self.seq_len = seq_len
    
    def __getitem__(self, index): 
        fn = self.data_lis[index*self.seq_len:index*self.seq_len+self.seq_len]
        img = self.loader(fn) 
        if self.transform is not None:
            imgs = [self.transform(img[i]).unsqueeze(0) for i in range(self.seq_len)]
            for i in range(self.seq_len-1):
                imgs[0] = torch.cat([imgs[0], imgs[i+1]], dim=0)

        if fn[0][1] == 'N':
            label = 1
        elif fn[0][1] == 'C':
            label = 0

        sample = {'imgs': imgs[0], 'label': label, 'path': fn}               
        
        return sample
                                                                     
    def __len__(self):
        return len(self.data_lis)//self.seq_len


class Ultra_Dataset_Eval(Dataset): 
    def __init__(self, data_dir, transform, seq_len, loader=default_loader): 
        fh = open(data_dir, 'r')
        data_lis = [] 
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data_lis.append((words[0],words[1]))
        
        self.data_lis = data_lis
        self.transform = transform
        self.loader = loader
        self.seq_len = seq_len
    
    def __getitem__(self, index): 
        fn = self.data_lis[index*self.seq_len:index*self.seq_len+self.seq_len]
        img = self.loader(fn) 
        if self.transform is not None:
            imgs = [self.transform(img[i]).unsqueeze(0) for i in range(self.seq_len)]
            for i in range(self.seq_len-1):
                imgs[0] = torch.cat([imgs[0], imgs[i+1]], dim=0)

        if fn[0][1] == 'N':
            label = 1
        elif fn[0][1] == 'C':
            label = 0

        sample = {'imgs': imgs[0], 'label': label, 'path': fn}               
        
        return sample
                                                                     
    def __len__(self):
        return len(self.data_lis)//self.seq_len