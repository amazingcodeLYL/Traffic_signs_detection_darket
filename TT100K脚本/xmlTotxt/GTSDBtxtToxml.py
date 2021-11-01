#!/usr/bin/env python
# coding:utf-8
# from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import numpy as np
import os
import csv
from itertools import islice

# CLASSES = ('speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
#            'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
#            'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road',
#            'give way', 'stop', 'no traffic both ways',
#            'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road',
#            'slippery road', 'road narrows', 'construction', 'traffic signal', 'pedestrian crossing',
#            'school crossing','cycles crossing', 'snow', 'animals', 'restriction ends',
#            'go right', 'go left', 'go straight', 'go right or straight','go left or straight', 'keep right',
#            'keep left', 'roundabout', 'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))')

CLASSES = ('prohibitory', 'danger', 'mandatory', 'other')

for xx in range(0,43):
    if xx<10:
        path=os.listdir("/home/dell/桌面/GTSRB_re/train/Images/0000"+str(xx))
        filepath="/home/dell/桌面/GTSRB_re/train/Images/0000"+str(xx)
        filename="0000"+str(xx)
    elif xx>=10 :
        path = os.listdir("/home/dell/桌面/GTSRB_re/train/Images/000" + str(xx))
        filepath = "/home/dell/桌面/GTSRB_re/train/Images/000" + str(xx)
        filename = "000" + str(xx)

    for x in path:
        if x.endswith(".csv") is True:
            x_path = os.path.join(filepath, x)
            print(x_path)
            with open(x_path, 'r') as read:
                reader = csv.reader(read)
                for lines in islice(reader, 1, None):
                    for i in range(len(lines)):
                        node_root = Element('annotation')
                        node_folder = SubElement(node_root, 'folder')
                        node_folder.text = 'TrainIJCNN2013'
                        line = lines[i].split(';')
                        print(line[1], line[2], line[3], line[4])
                        line_shape = np.reshape(line[3:], (-1, 5))
                        print(line_shape)
                        node_filename = SubElement(node_root, 'filename')
                        img_name = line[0].split('.')[0]
                        node_filename.text = line[0]

                        node_size = SubElement(node_root, 'size')
                        node_width = SubElement(node_size, 'width')
                        node_width.text = line[1]

                        node_height = SubElement(node_size, 'height')
                        node_height.text = line[2]

                        node_depth = SubElement(node_size, 'depth')
                        node_depth.text = '3'

                        # Roi.X1 Roi.Y1  Roi.X2 Roi.Y2
                        for j in range(len(line_shape[:, 4])):
                            node_object = SubElement(node_root, 'object')

                            class_n = int(line_shape[:, 4][j])

                            # 提示牌类signs 颜色colors
                            prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
                            danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                            mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
                            other = [6, 12, 13, 14, 17, 32, 41, 42]
                            if class_n in prohibitory:
                                signs = 0
                                colors = "red"
                                classes_four="prohibitory"
                            elif class_n in danger:
                                signs = 1
                                colors = "yellow"
                                classes_four = "danger"
                            elif class_n in mandatory:
                                signs = 2
                                colors = "blue"
                                classes_four = "mandatory"
                            elif class_n in other:
                                signs = 3
                                colors = "other"
                                classes_four = "other"

                            node_name = SubElement(node_object, 'name')
                            node_name.text = classes_four


                            node_signs = SubElement(node_object, 'signs')
                            node_signs.text = str(signs)

                            node_colors = SubElement(node_object, 'colors')
                            node_colors.text = colors

                            # node_difficult = SubElement(node_object, 'difficult')
                            # node_difficult.text = '0'

                            node_bndbox = SubElement(node_object, 'bndbox')
                            node_xmin = SubElement(node_bndbox, 'xmin')
                            node_xmin.text = line_shape[:, 0][j]
                            node_ymin = SubElement(node_bndbox, 'ymin')
                            node_ymin.text = line_shape[:, 1][j]
                            node_xmax = SubElement(node_bndbox, 'xmax')
                            node_xmax.text = line_shape[:, 2][j]
                            node_ymax = SubElement(node_bndbox, 'ymax')
                            node_ymax.text = line_shape[:, 3][j]
                        Xml = tostring(node_root, pretty_print=True)  # Formatted display, the newline of the newline
                        # dom = parseString(Xml)
                        save_path="/home/dell/桌面/GTSRB_re/"+filename
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        with open(save_path +"/"+ img_name + ".xml", "wb") as f:
                            f.write(Xml)