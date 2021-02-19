import os
import glob
import shutil
import numpy as np
import glob

# 目录递归拷贝函数
def dir_copyTree(src, dst):
  names = os.listdir(src)
  # 目标文件夹不存在，则新建
  if not os.path.exists(dst):
    os.mkdir(dst)
  # 遍历源文件夹中的文件与文件夹
  for name in names:
    srcname = os.path.join(src, name)
    dstname = os.path.join(dst, name)
    # 是文件夹则递归调用本拷贝函数，否则直接拷贝文件
    if os.path.isdir(srcname):
        dir_copyTree(srcname, dstname)
    else:
        if (not os.path.exists(dstname)
            or ((os.path.exists(dstname))
            and (os.path.getsize(dstname) != os.path.getsize(srcname)))):
            print(dstname)
            shutil.copy2(srcname, dst)

def main():
    output = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H/New/3/File'
    path = '/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H/New/3'


    if not os.path.exists(output):
        os.mkdir(output)
    
    txtName = output + '/lists.txt'
    f=open(txtName, "a+")
    
    counter = dict()
    for parent,_ ,filenames in os.walk(path):
        filenames[:] = [f for f in filenames if f.endswith("_crop_by_mask.png")]

        parent_path, patient_name = os.path.split(parent)
        parent_path, class_name = os.path.split(parent_path) # class_name: C N 

        if len(filenames) != 0:
            print('parent: ', parent)
            print('filenames: ', filenames)
            if '/N/' in parent:
                for ite in filenames:
                    print(parent + '/' + ite)
                    f.write(parent + '/' + ite + ' N' + '\n')
            elif '/C/' in parent:
                for ite in filenames:
                    print(parent + '/' + ite)
                    f.write(parent + '/' + ite + ' C' + '\n')

    f.close()
    
if __name__ == "__main__":
    main()




# import os
# count = 0

# # 遍历文件夹
# def walkFile(file):
#     for root, dirs, files in os.walk(file):
#         # root 表示当前正在访问的文件夹路径
#         # dirs 表示该文件夹下的子目录名list
#         # files 表示该文件夹下的文件list
        
#         # 遍历文件
#         for f in files:
#             global count
#             count += 1
#             print(os.path.join(root, f))

#         # 遍历所有的文件夹
#         for d in dirs:
#             print(os.path.join(root, d))
#     print("文件数量一共为:", count)

# if __name__ == '__main__':
#     walkFile(r"/home/qianxianserver/data/cxx/ultrasound/TJDataClassification/TJ_H")