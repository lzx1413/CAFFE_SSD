import cv2
import os
img_root = '/mnt/lvmhdd1/dataset/VOCBolt/VOCBolt1/JPEGImages'
for file_path in os.listdir(img_root):
    if 'jpg' not in file_path:
        continue
    img = cv2.imread(os.path.join(img_root,file_path))
    print img.shape
