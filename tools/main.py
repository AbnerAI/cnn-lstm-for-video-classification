import os
import cv2
import json
import glob
import numpy as np
from labelme import utils
import matplotlib.pyplot as plt
from labelme.utils import image

def main():
    # path = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H'
    path = '/home/qianxianserver/data/cxx/ultrasound/ultrasound_data/数据整理/数据来源/体检科/体检科所有带标注图片(未取关键帧之前)'
    for parent,dirnames,filenames in os.walk(path):
        # 很多时候需要忽略一些特定目录
        # 忽略 "someenv" and "__pycache__" 目录中
        dirnames[:] = [d for d in dirnames if d not in ['someenv','__pycache__']]
        # 这里完成了对dirnames的筛选，也就是说在接下来的for循环中，
        # someenv和__pycache__将不会被walk
        # 然后，选中所有以".md"结尾的文件
        filenames[:] = [f for f in filenames if f.endswith(".json")]
        for fullfilename in filenames:
            #输出找到的文件目录
            filename, _ = os.path.splitext(os.path.join(parent,fullfilename))
            data = json.load(open(os.path.join(parent,fullfilename)))
            img = utils.img_b64_to_arr(data['imageData'])

            label_name_to_value = {'_background_': 0}

            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            mask=[]
            class_id=[]
            for i in range(1,len(label_name_to_value)):
                mask.append((lbl==i).astype(np.uint8)) 
                class_id.append(i) 
            mask=np.asarray(mask,np.uint8)
            mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0])

            all_mask=0

            for i in range(0,len(class_id)):
                retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY) 
                # cv2.imencode('.jpg', im_at_fixed)[1].tofile(file_path + '/' + filename  + "_{}_mask.jpg".format(i)) 
                cv2.imwrite(filename  + "_{}_mask.png".format(i), im_at_fixed)
                all_mask = all_mask + im_at_fixed
            cv2.imwrite(filename  + "_mask_together.png", all_mask)
            # cv2.imencode('.jpg', all_mask)[1].tofile(file_path + '/' + filename  + "_mask_together.jpg") 
        
if __name__ == "__main__":
    main()