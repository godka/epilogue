
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import numpy as np

#from pykalman import KalmanFilter
import sys
if len(sys.argv) > 1:
    DIR = sys.argv[1] + '/'
else:
    DIR = './baselines/'
LW = 2
SCALE = 300

def readlog(filename):
    _file = open(filename, 'r')
    _tmpA = []
    for _line in _file:
        lines = _line.split(' ')
        if len(lines) >= 3:
            _tmpA.append(float(lines[1]) / (1e3))
    _tmpA.append(1.0)
    _file.close()
    _filename = filename.split('/')[-1]
    _filename = _filename.split('.')[0]
    _tmpA = np.array(_tmpA)
    print(np.percentile(_tmpA, 75), np.mean(_tmpA), _filename)
    #_tmpA *= 100.
    _a_x, _a_y = np.histogram(_tmpA,bins=100)#, range=(0., 0.2))
    _a_x = np.array(_a_x,dtype=np.float32)
    _a_x = np.insert(_a_x,0,0.)
    _a_x = np.cumsum(_a_x)
    _a_x /= np.max(_a_x)
    return _a_y, _a_x, _filename

plt.switch_backend('Agg')

#better = mpimg.imread('better.eps')
#plt.rc('text', usetex=True)
try:
    plt.rc('text', usetex=True)
except:
    pass
plt.rc('font', family='times')
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 15}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(5, 4))
plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.98)
listX,listY, listLabel = [],[], []
for p in os.listdir('./baselines/'):
    _bitrate_x, _bitrate_y, _f = readlog('./baselines/' + p)
    listX.append(_bitrate_x)
    listY.append(_bitrate_y)
    listLabel.append(_f)

# axim = fig.add_axes([0.65,0.15,0.25,0.25], anchor='SW')
# axim.imshow(better, aspect='auto')
# axim.axis('off')
#ax1.grid(True)
#ax1.set_title('Tiyuntsong')
ax1.set_ylabel(r'\textbf{CDF}')
ax1.set_xlabel(r'\textbf{Rebuf. Time(s)}')
ax1.set_xlim(0., 1.)
#ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))
index = 0
_mcolor = ['darkred', 'darkblue', 'black', 'gray', 'pink']
_line = [':', '-.', '-', '--', '--']
for (x,y,_f) in zip(listX, listY, listLabel):
    l4 = ax1.plot(x, y, _line[index], lw=LW, color=_mcolor[index], label=_f,MarkerSize=15)
    index += 1
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
ax1.legend(fontsize=13)
savefig('figs/cdfrbf.pdf')
print 'done'
