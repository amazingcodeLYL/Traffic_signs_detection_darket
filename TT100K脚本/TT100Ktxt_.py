
import  os
path="/home/dell/桌面/darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
with open(path,"r") as read:
    with open("2007_test.txt","w") as f:
        for r in read.readlines():
            f.write("/home/dell/桌面/darknet/VOCdevkit/VOC2007/JPEGImages/"+r+".jpg\n")
    f.close()

read.close()
