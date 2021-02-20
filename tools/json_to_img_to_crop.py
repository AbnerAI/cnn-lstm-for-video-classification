import os
import cv2
import glob
import json
import yaml
import base64
import argparse
import warnings
import PIL.Image
import numpy as np
import os.path as osp
from PIL import Image
from labelme import utils
import matplotlib.pyplot as plt

#jsonfile是标记的json文件保存路径，savepath是mask保存路径，savepath1是原图保存路径， savepath2是按mask裁剪后图保存路径
def decompression(jsonfile,savepath,savepath1,savepath2):
    path = jsonfile
    data = json.load(open(path))
    out_dir = osp.basename(path).replace('.', '_')
    out_dir = osp.join(osp.dirname(path), out_dir)
    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')

    img = utils.img_b64_to_arr(imageData)
    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    label_values, label_names = [], []
    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
        label_values.append(lv)
        label_names.append(ln)
    assert label_values == list(range(len(label_values)))
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
    cv2.imencode(".png",lbl[0]*255)[1].tofile(savepath)
    cv2.imencode(".jpg",img)[1].tofile(savepath1)
    [sizey,sizex] = lbl[0].shape
    newimg = np.zeros(img.shape)
    # print(newimg)
    for i in range(sizey):
        for j in range(sizex):
            if(lbl[0][i][j]==1):
                newimg[i][j] = img[i][j]
    cv2.imencode(".png", newimg)[1].tofile(savepath2)
    ppimg = cv2.imdecode(np.fromfile(savepath2, dtype=np.uint8), cv2.IMREAD_COLOR)
    kernel5 = np.ones((5, 5), np.uint8)
    
    thresh = cv2.Canny(ppimg, 0, 60)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel5)
    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    i = np.argmax(areas)
    x, y, w, h = cv2.boundingRect(contours[i])
    finalimg = newimg[y:y+h,x:x+w,:]
    print(finalimg.shape)
    cv2.imencode(".png", finalimg)[1].tofile(savepath2)

if __name__ == "__main__":
    '''
    decompression(jsonfile,savepath,savepath1,savepath2):
    #jsonfile是标记的json文件保存路径，savepath是mask保存路径，savepath1是原图保存路径， savepath2是按mask裁剪后图保存路径
    '''
    path = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H/New'
    # path = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H'
    for parent,dirnames,filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in ['someenv','__pycache__']]
        filenames[:] = [f for f in filenames if f.endswith(".json")]
        for fullfilename in filenames:
            fullpath = os.path.join(parent,fullfilename)
            filename, _ = os.path.splitext(fullpath)
            
            jsonfile = filename + '.json'
            savepath = filename + '_0_mask_new.png'
            savepath1 = filename + '_new.jpg'
            savepath2 = filename + '_crop_by_mask_new.png'
            print(filename)
            decompression(jsonfile,savepath,savepath1,savepath2)