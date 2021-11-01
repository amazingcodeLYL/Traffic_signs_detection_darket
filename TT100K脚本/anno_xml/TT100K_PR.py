import sys
import json
from anno_xml import anno_func
import numpy as np
import pylab as pl
"""
绘制TT100K-PR图
"""
datadir="/home/dell/PycharmProjects/traffic_test/anno_xml"
# result_anno_file =datadir+ "/fastrcnn_result.json"
result_anno_file1 =datadir+ "/record.json"
result_anno_file2=datadir+"/ours_result_annos.json"   #zhu
result_anno_file3= datadir + "/our_record.json"
result_anno_file4= datadir + "/yolov4_record.json"

# results_annos = json.loads(open(result_anno_file).read())
results_annos1 = json.loads(open(result_anno_file1).read())
results_annos2=json.loads(open(result_anno_file2).read())
results_annos3=json.loads(open(result_anno_file3).read())
results_annos4=json.loads(open(result_anno_file4).read())
filedir = datadir + "/annotations.json"
ids = open(datadir + "/ids.txt").read().splitlines()
# ids = open(datadir + "/2007_test.txt").read().splitlines()

annos = json.loads(open(filedir).read())
test_annos = results_annos4
minscore=0.40
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=0,maxboxsize=400,minscore=minscore)
print (sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=0,maxboxsize=32,minscore=minscore)
print (sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=32,maxboxsize=96,minscore=minscore)
print (sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=96,maxboxsize=400,minscore=minscore)
print (sm['report'])
def get_acc_res(results_annos, **argv):
    scs = [obj['score'] for k, img in results_annos['imgs'].items() for obj in img['objects']]
    scs = sorted(scs)
    accs = [0]
    recs = [1]
    for i, score in enumerate(np.linspace(0, scs[-1], 100)):
        sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45,
                                  minscore=score, **argv)
        print("\r%s %s %s" % (i, score, sm['report']))
        sys.stdout.flush()
        accs.append(sm['accuracy'])
        if len(accs) >= 2 and accs[-1] < accs[-2]:
            accs[-1] = accs[-2]
        recs.append(sm['recall'])
    accs.append(1)
    recs.append(0)
    return accs, recs

#
# sizes = [0, 32, 96, 400]
# ac_rc = []
#
# for i in range(4):
#     if i == 3:
#         l = sizes[0]
#         r = sizes[-1]
#     else:
#         l = sizes[i]
#         r = sizes[i + 1]
#     # acc, recs = get_acc_res(results_annos, minboxsize=l, maxboxsize=r)
#     acc1, recs1 = get_acc_res(results_annos1, minboxsize=l, maxboxsize=r)
#     acc2, recs2 = get_acc_res(results_annos2, minboxsize=l, maxboxsize=r)
#     acc3, recs3 = get_acc_res(results_annos3, minboxsize=l, maxboxsize=r)
#     acc4, recs4 = get_acc_res(results_annos4, minboxsize=l, maxboxsize=r)
#     # ac_rc.append([acc, recs])
#     ac_rc.append([acc1, recs1])
#     ac_rc.append([acc2, recs2])
#     ac_rc.append([acc3, recs3 ])
#     ac_rc.append([acc4, recs4])
#     pl.figure()
#     # pl.plot(acc, recs, label='Fast R-CNN')
#     pl.plot(acc1, recs1, label='YOLOv3')
#     pl.plot(acc2, recs2, label='Zhu et al.')
#     pl.plot(acc3, recs3, label='YOLOv4')
#     pl.plot(acc4, recs4, label='Ours')
#     pl.xlabel("precision")
#     pl.ylabel("recall")
#     pl.title("size: (%s,%s]" % (l, r))
#
#
#     pl.legend(bbox_to_anchor=(0, 0), loc="lower left")
#     pl.grid(linestyle='-.')
#     pl.show()
#     pl.savefig("/home/dell/PycharmProjects/traffic_test/anno_xml/ac-rc%s.pdf"%i)
# _ = pl.hist(scs, bins=100)# _ = pl.hist(scs, bins=100)