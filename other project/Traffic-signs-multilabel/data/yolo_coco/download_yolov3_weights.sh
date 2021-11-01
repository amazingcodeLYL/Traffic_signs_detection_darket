#!/bin/bash

# make '/weights' directory if it does not exist and cd into it
# mkdir -p weights && cd weights

# copy darknet weight files, continue '-c' if partially downloaded
# wget -c https://pjreddie.com/media/files/yolov3.weights
# wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
# wget -c https://pjreddie.com/media/files/yolov3-spp.weights

# Traffic-signs-multilabel pytorch weights
# download from Google Drive: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

# darknet53 weights (first 75 layers only)
# wget -c https://pjreddie.com/media/files/darknet53.conv.74

# Traffic-signs-multilabel-tiny weights from darknet (first 16 layers only)
# ./darknet partial cfg/Traffic-signs-multilabel-tiny.cfg Traffic-signs-multilabel-tiny.weights Traffic-signs-multilabel-tiny.conv.15 15
# mv Traffic-signs-multilabel-tiny.conv.15 ../

# new method
python3 -c "from models import *;
attempt_download('weights/yolov3.pt');
attempt_download('weights/yolov3-spp.pt')"
