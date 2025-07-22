import glob
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def mopen(dir=None, channel='A'):
    if dir:
        dir=dir
    else:
        dir='.'
    channel = channel
    ifile = glob.glob(dir+'/*.mat')
    ifile.sort()

    if len(ifile) == 0:
        print("No matlab file found")
        exit()

    w = []
    for f in ifile:
        dat = loadmat(f)
        hres, l, _w = dat['Tinterval'][0][0], dat['Length'][0][0], dat[channel].T[0]
        if np.isinf(_w).any():
            continue
        w = np.append(w, _w)


    w = w.reshape(-1, l)
        
    vres = abs(w).max()/2**15
    w = w/vres

    return w, vres, hres

def phiV(x0=0.232):
    x0 = x0
    fo = glob.glob('./*uA')
    fo.sort()
    print(fo)
    for e, i in enumerate(fo):
        #x=A, y=B
        print(f"Loading {i} ...")
        x, vres, hres = mopen(i, 'A')
        xa = np.average(x*vres, axis=0)
        y, vres, hres = mopen(i, 'B')
        ya = np.average(y*vres, axis=0)
        if e == 0:
            yamin = ya.min()
        plt.scatter(xa/x0, (ya-yamin)/2, label=i[2:],s=1)
        plt.xlabel(r'$\rm \Phi_{in}/ (\Phi_0) $', fontsize=16)
        plt.ylabel(r'$\rm V\ (mV)$', fontsize=16)
        #plt.show()

    plt.legend(loc=1, fontsize=14)
    plt.tight_layout()
    plt.show()

