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
type_attrs=['p27', 'w13', 'il50', 'p9', 'pm10', 'w47', 'i9', 'w25', 'pne', 'p16', 'w7', 'w29', 'w12', 'w22', 'w57',
            'w53', 'ip', 'i11', 'p7', 'w67', 'w60', 'pc', 'i1', 'p21', 'w9', 'pa10', 'i5', 'i15', 'w66', 'p29',
            'p15', 'pnl', 'w31', 'p17', 'w33', 'w36', 'i14', 'w51', 'w63', 'pr40', 'pg', 'w41', 'w27', 'p12',
            'w55', 'ps', 'w26', 'w18', 'pw3', 'w28', 'p3', 'w6', 'w34', 'p25', 'p14', 'w46', 'w54', 'p13', 'p22',
            'w17', 'pl40', 'w40', 'w11', 'w59', 'i3', 'w42', 'w10', 'w21', 'w20', 'w50', 'p23', 'p11', 'w8', 'w5',
            'p1', 'w45', 'w61', 'w2', 'w39', 'p28', 'w30', 'w44', 'w52', 'w1', 'w58', 'w62', 'w56', 'p19', 'w43', 'p4',
            'pd', 'pe', 'p10', 'p2', 'p24', 'pn', 'i8', 'w35', 'w48', 'pb', 'w24', 'w19', 'w38', 'w32', 'i2', 'i7', 'w3',
            'w15', 'p26', 'w4', 'i4', 'i13', 'w65', 'ph3.5',
            'i12', 'i6', 'i10', 'p18', 'w16', 'w49', 'p20', 'p6', 'p5', 'w23', 'w14', 'w64', 'w37', 'p8']
direction_attrs = ['prohibitory','danger','mandatory','other']



class Vehicle(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, transform, train_set="train.txt"):
        self.root = os.path.join(voc_root, "GTSDB", "VOC2007")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # read class_indict
        try:
            json_file = open('/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/data_processing/TT100Knames.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        self.transform = transform


    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):

        # 将多标签分开
        self.color_attrs = color_attrs
        self.sign_attrs = direction_attrs
        self.name_attrs = type_attrs

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
        boxes = []
        labels = []
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
            #     # labels_id.append(0)
            #     labels_id=0
            # elif self.class_dict[obj["name"]]  in danger:
            #     labels_id=1
            # elif self.class_dict[obj["name"]] in mandatory:
            #     labels_id=2
            # elif self.class_dict[obj["name"]]  in other:
            #     labels_id=3
            labels_id=0

            label=np.array([labels_id, labels_id, self.class_dict[obj["name"]]],dtype=int)
            labels.append(label)
            # iscrowd.append(int(obj["difficult"]))
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image = self.transform(image)

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



