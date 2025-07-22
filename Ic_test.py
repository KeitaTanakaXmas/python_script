import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math

import matplotlib.cm as cm

plt.rcParams['image.cmap']            = 'viridis'
# plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize']       = 15 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize']       = 15 # 軸だけ変更されます
plt.rcParams['xtick.direction']       = 'in' # x axis in
plt.rcParams['ytick.direction']       = 'in' # y axis in 
plt.rcParams['axes.linewidth']        = 1.0 # axis line width
plt.rcParams['axes.grid']             = True # make grid
plt.rcParams['figure.subplot.bottom'] = 0.15

def Ic_func(T,Ic0,Tc):
     return Ic0*(1-T/Tc)**(3/2)

def outB(filename):
    import re

    file_path = filename
    target_line_number = 13

    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number == target_line_number:
                numbers = re.findall(pattern, line)
                if numbers:
                    print(f"Numbers in line {line_number}: {', '.join(numbers)}")
                break  # 指定された行の数字を見つけたらループを終了する
    return float(numbers[0])


def load_ic(Imin):
    ic = []
    B  = []
    files = sorted(glob.glob('*.txt'))
    print(files)
    for file in files:
        f = np.loadtxt(file)
        I = f[:,0][Imin:]
        V = f[:,1][Imin:]
        B.append(outB(file))
        if I[1] < 0 :
            Ic_index = np.argmax(np.diff(V))+1
        if I[1] > 0 :
            Ic_index = np.argmin(np.diff(V))+1
        #plt.scatter(I[1:],np.diff(V))
        #plt.title(f'{file}')
        #plt.scatter(I,V)
        #plt.scatter(I[Ic_index],V[Ic_index],color='black')
        print(I[Ic_index])
        ic.append(I[Ic_index])
        #plt.show()
    return np.array(ic), np.array(B)

def Ic():
    ic = []
    files = sorted(glob.glob('*.txt'))
    print(files)
    for e,file in enumerate(files):
        f = np.loadtxt(file)
        I = f[:,0]
        V = f[:,1]
        Ic_index = np.argmax(np.diff(V))+1
        #plt.scatter(I[1:],np.diff(V))
        #plt.scatter(I,V,color=cm.viridis([e/len(files)]))
        #plt.scatter(I[Ic_index],V[Ic_index],color='black')
        #print(I[Ic_index])
        ic.append(I[Ic_index])
    plt.show()

    return np.array(ic)

def Ic_fit():
    ic = Ic()
    T = np.array([100,120,130,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159])
    #popt, pcov = curve_fit(Ic_func, T, ic, p0=[100,158])
    plt.scatter(T,ic)
    #plt.plot(T,Ic_func(T,*popt))
    #print(popt, pcov)
    plt.show()


def plot_Ic():
    ic, B = load_ic(0)
    print(ic)
    ic = ic[np.argsort(B)]
    B = np.sort(B)
    plt.scatter(B*2.17e-3*1e+6,ic)
    plt.plot(B*2.17e-3*1e+6,ic)
    plt.xlabel(r'$Magnetic\ Field \ (\mu T)$')
    plt.ylabel(r'$Critical\ Current \ (\mu A)$')
    plt.grid(linestyle='dashed')
    # plt.yscale('log')
    plt.show()

def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size/2)

    # 補正部分
    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
	# size%2は奇数偶数での違いに対応するため

    return xx_mean

def multi_plot_Ic():
    dirs = ['130mK','140mK','150mK']
    for dir in dirs:
        os.chdir(f'./{dir}')
        ic, B = load_ic(0)
        print(ic)
        ic = ic[np.argsort(B)]
        B = np.sort(B)
        #B = valid_convolve(B, 3)
        #ic = valid_convolve(ic, 3)
        plt.scatter(B*2.17e-3*1e+6,ic,label=dir,s=5)
        plt.plot(B*2.17e-3*1e+6,ic,lw=2)
        os.chdir('..')
    plt.legend()
    plt.xlabel(r'$Magnetic\ Field \ (\mu T)$')
    plt.ylabel(r'$Critical\ Current \ (\mu A)$')
    plt.grid(linestyle='dashed')
        # plt.yscale('log')
    plt.show()

def pm_plot():
    os.chdir('/Users/keitatanaka/Downloads/ic05/m120mK')
    ic, B = load_ic(0)
    print(ic)
    ic = ic[np.argsort(B)]
    B = np.sort(B)
    plt.scatter(B*2.17e-3*1e+6,-ic,color='blue',label='-bias',marker=',')

    os.chdir('/Users/keitatanaka/Ic_all')
    ic, B = load_ic(0)
    print(ic)
    ic = ic[np.argsort(B)]
    B = np.sort(B)
    plt.scatter(B*2.17e-3*1e+6,ic,color='blue',label='+bias')
    # plt.plot(B*2.17e-3*1e+6,ic)
    plt.xlabel(r'$\rm Magnetic\ Field \ (\mu T)$')
    plt.ylabel(r'$\rm Critical\ Current \ (\mu A)$')
    plt.grid(linestyle='dashed')
    plt.legend()
    plt.show()



def testP():
    import h5py
    f = h5py.File('/Users/keitatanaka/Downloads/iv01/test.hdf5')
    R_0uT = f['ch1/IV/data/120mK/Rtes'][:]
    T_0uT = f['ch1/IV/data/120mK/Ttes'][:]

    f = h5py.File('/Users/keitatanaka/Downloads/iv02/test.hdf5')
    R_10uT = f['ch1/IV/data/120mK/Rtes'][:]
    T_10uT = f['ch1/IV/data/120mK/Ttes'][:]

    f = h5py.File('/Users/keitatanaka/Downloads/iv03/test.hdf5')
    R_5uT = f['ch1/IV/data/120mK/Rtes'][:]
    T_5uT = f['ch1/IV/data/120mK/Ttes'][:]

    fig = plt.figure(figsize=(10.6,6))
    ax  = plt.subplot(1,1,1)
    ax.grid(linestyle="dashed")
    ax.scatter(T_0uT*1e3, R_0uT*1e3, s=10, color='blue', label='0 uT')
    ax.scatter(T_5uT*1e3, R_5uT*1e3, s=10, color='green', label='5 uT')
    ax.scatter(T_10uT*1e3, R_10uT*1e3, s=10, color='red', label='10 uT')
    ax.set_xlabel(r'$T_{TES} \ (mK)$',fontsize=15)
    ax.set_ylabel(r'$R_{TES} \ (m\Omega)$',fontsize=15)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.legend(fontsize=12)
    plt.show()