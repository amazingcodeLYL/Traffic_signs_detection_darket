#import cPickle
import _pickle as cPickle
import matplotlib.pyplot as plt
fr = open('','rb')
inf = cPickle.load(fr)
fr.close()

x=inf['rec']
y=inf['prec']
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR cruve')
plt.plot(x,y)
plt.show()
print('AP:',inf['ap'])
