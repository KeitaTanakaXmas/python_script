import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D
import glob
import re
import lmfit as lf
from lmfit import Model
from scipy.optimize import curve_fit
import datetime

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2022.03.01

print('===============================================================================')
print(f"Analysis of MWMUX ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class VAnalysis:

    def __init__(self):
        self.path = "/Users/tanakakeita/work/microcalorimeters/experiment/SEED_vapor_equipment/thickness/data/SEED"
        self.filelist  =  sorted(glob.glob(f"*.csv"))
        print(self.filelist)

    def read_log(self,filename):
        d = np.genfromtxt(filename,skip_header=3,delimiter=",",encoding="gbk")
        self.frq = d[:,0]
        self.gb = d[:,1]

    def plot_init(self):
        self.fig = plt.figure(figsize=(15,12))
        plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
        plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize']       = 30 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize']       = 30 # 軸だけ変更されます
        plt.rcParams['xtick.direction']       = 'in' # x axis in
        plt.rcParams['ytick.direction']       = 'in' # y axis in 
        plt.rcParams['axes.linewidth']        = 1.0 # axis line width
        plt.rcParams['axes.grid']             = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['scatter.edgecolors']    = 'black'
        self.fs = 30
        self.ps = 50
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.grid(linestyle="dashed")

    def plot_log(self,filename):
        self.read_log(filename=filename)
        self.idx_out()
        self.plot_init()
        self.ax1.plot(self.time,self.pressure,color="black")
        self.ax1.plot(self.time[self.st_idx:self.ed_idx],self.pressure[self.st_idx:self.ed_idx],color="red")
        self.ax2 = self.ax1.twinx()
        self.ax2.plot(self.time,self.Ti_th,label=r"$\rm Ti \ output$")
        self.ax2.plot(self.time,self.Au_th,label=r"$\rm Au \ output$")
        self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=self.fs)
        self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=self.fs)
        self.ax2.set_ylabel(r"$\rm Output power \ (\%)$",fontsize=self.fs)
        self.ax1.set_yscale("log")
        self.ax2.legend(fontsize=20)
        plt.show()
        sfn = "Vlog.png"
        self.fig.savefig(sfn,dpi=300)

    def plot_log_all(self):
        self.plot_init()
        l = len(self.filelist)
        for e,i in enumerate(self.filelist):
            self.read_log(filename=i)
            self.ax1.plot(self.pressure,label=i)
            #ax2 = ax1.twinx()
            #ax2.plot(self.time,self.Ti_th,label=r"$\rm Ti \ output \ (\%) $")
            #ax2.plot(self.time,self.Au_th,label=r"$\rm Au \ output \ (\%) $")
            self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=self.fs)
            self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=self.fs)
            self.ax1.set_yscale("log")
            self.ax1.legend(fontsize=20)
            print("-----------------------------------------------")
            print(f"{i}")
        plt.show()
        sfn = "Vlog_all.png"
        self.fig.savefig(sfn,dpi=300)

    def plot_log_all_Au(self):
        self.plot_init()
        l = len(self.filelist)
        for e,i in enumerate(self.filelist):
            self.read_log(filename=i)
            self.idx_out()
            #ax1.plot(self.time,self.pressure,color="black")
            self.ax1.plot(self.pressure[self.st_idx:self.ed_idx-6],label=i)
            #ax2 = ax1.twinx()
            #ax2.plot(self.time,self.Ti_th,label=r"$\rm Ti \ output \ (\%) $")
            #ax2.plot(self.time,self.Au_th,label=r"$\rm Au \ output \ (\%) $")
            self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=self.fs)
            self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=self.fs)
            self.ax1.set_yscale("log")
            self.ax1.legend(fontsize=20)
            print("-----------------------------------------------")
            print(f"{i}")
            print(f"Start pressure = {self.pressure[self.st_idx]} [Pa]")
            print(f"ReStars pressure = {self.pressure[self.ed_idx-6]} [Pa]")
            print(f"difference = {self.pressure[self.st_idx] - self.pressure[self.ed_idx-6]} [Pa]")
            print(f"time = {self.ed_idx - self.st_idx} [s]")
        plt.show()
        sfn = "Vlog_all_Au.png"
        self.fig.savefig(sfn,dpi=300)

    def plot_log_all_Ti(self):
        self.plot_init()
        l = len(self.filelist)
        for e,i in enumerate(self.filelist):
            self.read_log(filename=i)
            self.idx_out()
            #ax1.plot(self.time,self.pressure,color="black")
            self.ax1.plot(self.pressure[self.st_Ti-10:self.ed_Ti],label=i)
            #ax2 = ax1.twinx()
            #ax2.plot(self.time,self.Ti_th,label=r"$\rm Ti \ output \ (\%) $")
            #ax2.plot(self.time,self.Au_th,label=r"$\rm Au \ output \ (\%) $")
            self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=self.fs)
            self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=self.fs)
            self.ax1.set_yscale("log")
            self.ax1.legend(fontsize=20)
            print("-----------------------------------------------")
            print(f"{i}")
            print(f"Start pressure = {self.pressure[self.st_Ti-10]} [Pa]")
            print(f"End pressure = {self.pressure[self.ed_Ti]} [Pa]")
            print(f"difference = {self.pressure[self.st_Ti-10] - self.pressure[self.ed_Ti]} [Pa]")
            print(f"time = {self.ed_Ti - self.st_Ti} [s]")
        plt.show()
        sfn = "Vlog_all_Ti.png"
        self.fig.savefig(sfn,dpi=300)