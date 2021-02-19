'''
@Author  : Xiangxiang Cui, cuixiangchn@163.com
@Function: For crop image by mask.
@Date    : 2021-01-21
'''
import os
import cv2
import json
import glob
import imageio
import numpy as np
from PIL import Image
from labelme import utils
import matplotlib.pyplot as plt
from labelme.utils import image

def RoiExtraction(msk_arr):
    return min(np.where(msk_arr==1)[1]), max(np.where(msk_arr ==1)[1]), min(np.where(msk_arr==1)[0]), max(np.where(msk_arr == 1)[0])

def main():
    path = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H'
    for parent,dirnames,filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in ['someenv','__pycache__']]
        filenames[:] = [f for f in filenames if f.endswith(".json")]
        for fullfilename in filenames:
            fullpath = os.path.join(parent,fullfilename)
            filename, _ = os.path.splitext(fullpath)
            print(filename)
            # read img
            im = Image.open(fullpath[:-5]+'.jpg')
            im_array = np.array(im)
                        
            # read mask
            msk = Image.open(fullpath[:-5] + '_0_mask.png')
            msk_array = np.array(msk)

            # get roi:  x:width  y:height
            x1, x2, y1, y2 = RoiExtraction(msk_array/255)

            # crop image
            crop_image = im_array[y1:y2, x1:x2]
            # save crop image
            cv2.imencode(".png", crop_image)[1].tofile(filename + '_crop_by_mask.png')

if __name__ == "__main__":
    main()