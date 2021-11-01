import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
# 2012代表年份， 2012train.txt 就是ImageSets\Main 下对应的txt名称，其依次类推， 换成自己数据集需修改sets
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes 是所有类别名称， 换成自己数据集需要修改classes
classes = ["speed limit 20",
    "speed limit 30",
    "speed limit 50",
    "speed limit 60",
    "speed limit 70",
    "speed limit 80",
    "restriction ends 80",
    "speed limit 100",
    "speed limit 120",
    "no overtaking",
    "no overtaking (trucks)",
     "priority at next intersection",
    "priority road",
    "give way",
    "stop",
    "no traffic both ways",
    "no trucks",
    "no entry",
    "danger",
   "bend left",
   "bend right",
    "bend",
    "uneven road",
    "slippery road",
    "road narrows",
    "construction",
    "traffic signal",
    "pedestrian crossing",
    "school crossing",
    "cycles crossing",
    "snow",
    "animals",
    "restriction ends",
    "go right",
     "go left",
    "go straight",
    "go right or straight",
    "go left or straight",
    "keep right",
    "keep left",
    "roundabout",
    "restriction ends (overtaking)",
    "restriction ends (overtaking (trucks))"]

#@params  size  图片宽高
#@params  box  groud truth 框的x y w h
def convert(size, box):
    dw = 1./size[0]              # 用于下面框的坐标和高宽归一化
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0     # 求中心坐标
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('/home/dell/桌面/GTSDB/VOC%s/Annotations/%s.xml'%(year, image_id.split('\n')[0]))    # 读取 image_id 的xml文件
    out_file = open('./labels/%s.txt'%image_id, 'w')      # 保存txt文件地址
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')         # 读取图片 w h
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if cls not in classes or int(difficult) == 1:
            continue
        # cls_id = classes.index(cls)
        # prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
        # danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        # mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
        # other = [6, 12, 13, 14, 17, 32, 41, 42]
        # if cls_id in prohibitory:
        #     cls_id=0
        # elif cls_id in danger:
        #     cls_id=1
        # elif cls_id in mandatory:
        #     cls_id=2
        # elif cls_id in other:
        #     cls_id=3
        cls_id=0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b) # w,h,x,y归一化操作
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')   # 一个框写入

wd = getcwd() # 获取当前地址

image_ids = open(r"/home/dell/桌面/GTSDB/VOC2007/ImageSets/Main/train.txt")  # 读取图片名称
# list_file = open('%s_%s.txt'%("GTSDB", "imageset"), 'w')                                                                                      # 保存所有图片的绝对路径
for image_id in image_ids:
    # list_file.write('./my_yolo_dataset/train/images/{}.jpg\n'.format(image_id))
    convert_annotation("2007", image_id)
# list_file.close()

#########将PPM转为jpg
# from PIL import Image
# for p,x,y in os.walk("/home/dell/桌面/GTSDB/VOC2007/"):
#     for yi in y:
#         pp = os.path.join("/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/my_yolo_dataset/val/images/", yi)
#         img = Image.open(pp)
#         path = os.path.join("my_yolo_dataset/val/images/", yi.split(".ppm")[0] + ".jpg", )
#         img.save(path)
