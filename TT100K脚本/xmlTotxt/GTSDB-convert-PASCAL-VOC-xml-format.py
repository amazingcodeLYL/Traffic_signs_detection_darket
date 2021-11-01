#!/usr/bin/env python
# coding:utf-8

# from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import numpy as np

# CLASSES = ('speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
#            'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
#            'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road',
#            'give way', 'stop', 'no traffic both ways',
#            'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road',
#            'slippery road ', 'slippery road', 'road narrows', 'construction', 'traffic signal', 'pedestrian crossing',
#            'school crossing',
#            'cycles crossing', 'snow', 'animals', 'restriction ends',
#            'go right', 'go left', 'go straight', 'go right or straight', 'keep right',
#            'keep left ', 'roundabout', 'restriction ends', 'restriction ends')


CLASSES = ('prohibitory', 'danger', 'mandatory', 'other')

with open("../test_gt.txt", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        node_root = Element('annotation')

        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'TestIJCNN2013'
        line = lines[i].split(';')
        line_shape = np.reshape(line[1:], (-1, 5))

        node_filename = SubElement(node_root, 'filename')
        img_name = line[0].split('.')[0]
        node_filename.text = line[0]

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '1360'

        node_height = SubElement(node_size, 'height')
        node_height.text = '800'

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for j in range(len(line_shape[:, 4])):
            node_object = SubElement(node_root, 'object')

            class_n = int(line_shape[:, 4][j])

            prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
            danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
            other = [6, 12, 13, 14, 17, 32, 41, 42]
            if class_n in prohibitory:
                signs = 0
                classes_four = "prohibitory"
            elif class_n in danger:
                signs = 1
                classes_four = "danger"
            elif class_n in mandatory:
                signs = 2
                classes_four = "mandatory"
            elif class_n in other:
                signs = 3
                classes_four = "other"

            node_name = SubElement(node_object, 'name')
            node_name.text = classes_four


            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

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

        with open("test_anno_xml/" + img_name + ".xml", "wb") as f:
            f.write(Xml)


