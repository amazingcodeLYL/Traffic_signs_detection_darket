# coding: utf-8
import os
import re
import shutil
import pickle
import numpy as np
import scipy.io as scio
from torch import Tensor as Tensor
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
from  torch.utils.data import Dataset
import json
from lxml import etree



color_attrs = ['red','yellow','blue','other']
type_attrs = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
           'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
           'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road',
           'give way', 'stop', 'no traffic both ways',
           'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road',
           'slippery road', 'road narrows', 'construction', 'traffic signal', 'pedestrian crossing',
           'school crossing','cycles crossing', 'snow', 'animals', 'restriction ends',
           'go right', 'go left', 'go straight', 'go right or straight','go left or straight', 'keep right',
           'keep left', 'roundabout', 'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))']
direction_attrs = ['prohibitory','danger','mandatory','other']


class Vehicle(Dataset):
    def __init__(self, voc_root, transform, year='VOC2007', train_set='train.txt'):
        self.root = os.path.join(voc_root, "GTSR", year)
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # read class_indict
        try:
            json_file = open('/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/data_processing/GTSRB_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)


        # 加载数据变换
        # if transform is not None:
        self.transform = transform
        # else:  # default image transformation
        #     self.transform = T.Compose([
        #         T.Resize(448),
        #         T.CenterCrop(448),
        #         T.ToTensor(),
        #         T.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
        #     ])

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "PPM":
            raise ValueError("Image format not PPM")

        # 将多标签分开
        self.color_attrs = color_attrs
        self.sign_attrs=direction_attrs
        self.name_attrs=type_attrs

        boxes = []
        labels = []
        signs=[]
        colors=[]
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            # prohibitory=[0,1,2,3,4,5,7,8,9,10,15,16]
            # danger=[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
            # mandatory=[33,34,35,36,37,38,39,40]
            # other=[6,12,13,14,17,32,41,42]
            # if self.class_dict[obj["name"]] in prohibitory:
            #     labels=0
            # elif self.class_dict[obj["name"]]  in danger:
            #     labels=1
            # elif self.class_dict[obj["name"]] in mandatory:
            #     labels=2
            # elif self.class_dict[obj["name"]]  in other:
            #     labels=3
            # signs.append(int(obj["signs"]))
            # colors.append(obj["colors"])
            # iscrowd.append(int(obj["difficult"]))

            # 添加label
            classes = self.class_dict[obj["name"]]
            color_idx = int(np.where(self.color_attrs == np.array(obj["colors"]))[0])
            signs_idx=int(obj["signs"])
            # signs_idx = int(np.where(self.sign_attrs == np.array(int(obj["signs"])))[0])
            # labels_idx = int(np.where(self.name_attrs == np.array(labels))[0])
            label = np.array([color_idx, signs_idx, classes], dtype=int)

            # label = torch.Tensor(label)  # torch.from_numpy(label)
            labels.append(label)  # Tensor(label)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)


        # f_path = os.path.split(self.img_root[idx])[0].split('/')[-2] + \
        #          '/' + os.path.split(self.img_root[idx])[0].split('/')[-1] + \
        #          '/' + os.path.split(self.img_root[idx])[1]



        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

