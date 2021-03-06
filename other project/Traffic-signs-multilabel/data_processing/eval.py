import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
      ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset
if __name__ == '__main__':
    annType = ['segm', 'bbox', 'keypoints']#set iouType to 'segm', 'bbox' or 'keypoints'
    annType = annType[1] # bbox specify type here
    cocoGt_file = '/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/data_processing/TT100K_val.json'
    cocoGt = COCO(cocoGt_file)#取得标注集中coco json对象

    cocoDt_file = '/home/dell/桌面/PycharmProjects/Traffic-signs-multilabel/results.json'
    print("Valuation: {}".format(cocoDt_file.split('/')[-1]))
    imgIds = get_img_id(cocoDt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)#取得结果集中image json对象
    imgIds = sorted(imgIds)#按顺序排列coco标注集image_id
    imgIds = imgIds[0:4000]#标注集中的image数据
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds#参数设置
    cocoEval.params.catIds =[0,1,2,3]
    cocoEval.evaluate()#评价
    cocoEval.accumulate()#积累
    cocoEval.summarize()#总结