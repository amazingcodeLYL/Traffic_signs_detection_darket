import os
import PIL.Image as Image
from torchvision import transforms as transforms
import torchvision

# path="/home/dell/桌面/data_TT100K/marks/pad-all/"
# p=os.listdir(path)
# with open("tt100names.txt","w") as f:
#     for x in p:
#         x=x.split(".png")[0]
#         f.write(x+"\n")
#     f.close()


# clases=[]
# for x in p:
#     x=x.split(".png")[0]
#     clases.append(x)
# print(clases)

# train_transforms = torchvision.transforms.Compose([
#     transforms.Resize((64, 128)),
#     transforms.ColorJitter(0.3, 0.3, 0.2),
#     transforms.RandomRotation(5),
#     transforms.ToTensor()
#         ])
#
#
# import cv2
#
# for im in p:
#     if im =="i1.png":
#         xx = os.path.join(path, im)
#         image = Image.open(xx).convert('RGB')
#         image = train_transforms(image)
#
#         new_img_PIL = transforms.ToPILImage()(image).convert('RGB')
#         new_img_PIL.show()  # 处理后的PIL图片
#
#         print(image)
#
#         # new_im = transforms.Resize((50, 200))(image)
#         # new_im.save(os.path.join("/home/dell/桌面/data_TT100K/marks/images", "traiun.png"))
#
#     # print(xx)

res=[]
for i in range(0,600):
    if i<10:
        res.append("0000"+str(i))
    elif i>=10 and i<100:
        res.append("000"+str(i))
    elif i>=100:
        res.append("00" + str(i))
path="/home/dell/桌面/GTSDB/VOC2007/ImageSets/Main/train.txt"
no_intrain=[]
with open(path)  as f:
    for r in f.readlines():
        if r.split("\n")[0] not in res:
            no_intrain.append(r)
    print(no_intrain)

