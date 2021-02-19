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
        return len(self.data_lis)


class Ultra_Dataset_Eval(Dataset):
    def __init__(
            self,
            data_dir,
            label_dir):

        self.data_dir = data_dir
        self.label_dir = label_dir
        '''
        os.walk() 
        By walking through the tree, the file name output in the directory is up or down.
        '''
        self.label_lis = [os.path.join(d, x)
                            for d, dirs, files in os.walk(label_dir)
                            for x in files if x.endswith("_refine_aorta_and_arteries.nii.gz")]
        print('Number of files {}'.format(len(self.label_lis)))

    def __getitem__(self, index):
        
        label_dir = self.label_lis[index]
        # Return to the last file name of the path.
        label_name = os.path.basename(label_dir)
        caseID = label_name.replace('_aorta_and_arteries.nii.gz', '')
        caseID = caseID.replace('_refine', '')
        img_path = os.path.join(self.data_dir, caseID)
        filename = os.path.join(img_path, caseID+'_iso.nii.gz')

        itk_img = sitk.ReadImage(filename)
        # He said: GetArrayFromImage will lose the location and other 3D image information
        # I reserve my opinion and need an experimental test. 2020-06-29
        img_arr = sitk.GetArrayFromImage(itk_img)
        og_shape = img_arr.shape
        '''
        randomly reduce intensity 
        print('Before reducing intensity {}'.format(np.mean(img_arr)))
        '''
        #img_arr = img_arr - random.randint(0, 200)
        # print('After reducing intensity {}'.format(np.mean(img_arr)))
        MIN_BOUND = -100
        MAX_BOUND = 500
        # Set up and down boundaries.
        img_arr[img_arr > MAX_BOUND] = MAX_BOUND
        img_arr[img_arr < MIN_BOUND] = MIN_BOUND
        #start = time.clock()
        ds_img_arr = zoom(
            img_arr, (128/og_shape[0], 128/og_shape[1], 128/og_shape[2]), order=1)
        #print('lai: ',time.clock()-start)
        
        # print('Downsampled shape {}'.format(ds_img_arr.shape))
        itk_label = sitk.ReadImage(label_dir)
        label_arr = sitk.GetArrayFromImage(itk_label)

        label_arr[label_arr == 2] = 0
        label_arr[label_arr == 3] = 0

        label_arr = label_arr.astype(np.float32)
        #print('label_arr: ',label_arr.dtype)
        sys.stdout.flush()
        #sample = {'label': label_arr}
        sample = {'img': img_arr, 'label': label_arr,
                   'img_shape': og_shape, 'ds_img': ds_img_arr, 'filename': filename}
        return sample

    def __len__(self):
        return len(self.label_lis)