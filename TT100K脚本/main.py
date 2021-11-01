import os
import csv
from itertools import islice

# for xx in range(42,43):
#     path = os.listdir("/home/dell/桌面/GTSRB_re/train/Images/000"+str(xx))
#     filepath = "/home/dell/桌面/GTSRB_re/train/Images/000"+str(xx)
#
#     save_path = "/home/dell/桌面/GT-000"+str(xx)+".csv"
#     print(save_path)
#     for x in path:
#         if x.endswith(".csv") is True:
#             x_path = os.path.join(filepath, x)
#             print(x_path)
#             with open(x_path, 'r') as read:
#                 reader = csv.reader(read)
#                 with open(save_path, "w") as f:
#                     for lines in islice(reader, 1, None):
#                         lines = lines[0].split(";")
#                         imgname = "img"+str(xx)+"_" + lines[0]
#
#                         f.write(
#                             imgname + ";" + lines[1] + ";" + lines[2] + ";" + lines[3] + ";" + lines[4] + ";" + lines[
#                                 5] + ";" + lines[6] + ";" + lines[7] + "\n")
#                         # print(row)
#                     f.close()

# for xx in range(0,42):
#     if xx < 10:
#         path = os.listdir("/home/dell/桌面/GTSRB_re/train/Images/0000" + str(xx))
#         filepath = "/home/dell/桌面/GTSRB_re/train/Images/0000" + str(xx)
#         filename = "0000" + str(xx)
#     elif xx >= 10:
#         path = os.listdir("/home/dell/桌面/GTSRB_re/train/Images/000" + str(xx))
#         filepath = "/home/dell/桌面/GTSRB_re/train/Images/000" + str(xx)
#         filename = "000" + str(xx)
#
#     save_path = "/home/dell/桌面/train.txt"
#     print(save_path)
#     for x in path:
#         if x.endswith(".csv") is True:
#             x_path = os.path.join(filepath, x)
#             print(x_path)
#             with open(x_path, 'r') as read:
#                 reader = csv.reader(read)
#                 with open(save_path, "a") as f:
#                     for lines in islice(reader, 1, None):
#                         lines = lines[0].split(";")
#                         print(lines[0].split(".ppm")[0])
#                         f.write(lines[0].split(".ppm")[0]+"\n")
#                         # print(row)
#                     f.close()

# with open("/home/dell/桌面/darknet/VOCdevkit/VOC2007/ImageSets/Main/test.txt") as f:
#     with open("test.txt","w") as r:
#         for ii in f.readlines():
#             r.write("/home/dell/桌面/darknet/VOCdevkit/VOC2007/JPEGImages/"+ii+".jpg\n")
#     r.close()

p1=0.88
r1=0.92
x=1.00
res=(1+x)*p1*r1/(x*p1+r1)
print(res)


# r2=93.14
# p3=90.69
# r3=94.99
# p4=93.40
# r4=94.46
# x=1.00
# res=(1+x)*p1*r1/(x*p1+r1)
# res2=(1+x)*p2*r2/(x*p2+r2)
# res3=(1+x)*p3*r3/(x*p3+r3)
# res4=(1+x)*p4*r4/(x*p4+r4)
# # print(res,res2,res3,res4)
