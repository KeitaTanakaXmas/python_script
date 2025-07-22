# thickness_cl.py
# 膜厚測定結果の入っているディレクトリに行く /SEED240529a
# ipython
# from thickness_cl import TAnalysis
# T = TAnalysis('SEED240529a')
# T.Au_Ti_All()　
# これを実行すると、Au, Tiの厚みを計算して、全体の分布を出力してくれる。
# W01のAuの膜厚プロットが出てくるので、使いたい部分を選択する
# 最初に真ん中のbarの左側を選択、次に右側、最後に真ん中
# ウィンドウを消すと、計算に使われた領域がプロットされるので、確認したら消す
# これをW15まで行うと、W01のAu+Tiの膜厚が表示されるので、同様に選択。
# W15まで行うと、全体の膜厚分布がプロットされる。

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import SpanSelector
import datetime
import glob
import pandas as pd
import re
from pathlib import Path


__author__ =  'Keita Tanaka'
#__version__=  '2.0.0' #2022.08.05
#__version__=  '3.0.0' #
__version__=  '3.1.0' #2024.06.13

print('===============================================================================')
print(f"Thickness Analysis ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class TAnalysis:

    def __init__(self,hdf5):
        self.st = 1
        self.count = 0
        self.conv_list = sorted(glob.glob('*conv.txt'))
        p         = Path(hdf5)
        self.hdf5 = p.resolve()
        print(self.hdf5)
        self.title = 'Please Select Left Region'

    def linear(self,x,a):
        return a*x

    def xyout(self,file,Hsf=0.25):
        self.fn = np.genfromtxt(file)
        self.y  = self.fn[:]*0.1                     #[nm]
        self.x  = np.arange(0,Hsf*len(self.y),Hsf)   #[nm]

    def thickness_selector(self,x1,x2):
        print(x1,x2)
        if self.count == 0 :
            print("Left resion selected")
            self.left_mask = (x1 < self.x) & (self.x < x2)
            self.ax.plot(self.x[self.left_mask],self.y[self.left_mask],color='red')
            self.title = 'Please Select Right Region'
        elif self.count == 1 :
            print("Right resion selected")
            self.right_mask = (x1 < self.x) & (self.x < x2)
            self.ax.plot(self.x[self.right_mask],self.y[self.right_mask],color='red')
            self.title = 'Please Select Top Region'
        elif self.count == 2 :
            print("Top resion selected")
            self.top_mask = (x1 < self.x) & (self.x < x2)
            self.ax.plot(self.x[self.top_mask],self.y[self.top_mask],color='red')
            self.title = 'Please Select Left Region'
            self.calculate_thickness()
        self.count += 1

        self.ax.set_title(self.title)
        self.fig.canvas.draw()
        if self.count > 2 :
            self.count = 0
        

    def calculate_thickness(self):
        bottom = self.y[(self.left_mask | self.right_mask)]
        top = self.y[self.top_mask]
        self.thickness = np.average(top) - np.average(bottom)
        print(np.std(top,ddof=1))
        print(np.std(top))
        self.error = np.sqrt(np.std(top,ddof=1)**2+np.std(bottom,ddof=1)**2)
        print(f'Thickness = {self.thickness} um')
        print(f'Error = {self.error} um')

    def save_result(self,name,chip_id):
        print('--------------------------------------------------------')
        print(f'Filename = {self.hdf5}')
        with h5py.File(self.hdf5,"a") as f:
            if name in f.keys():
                if chip_id in f[name].keys():
                    del f[name][chip_id]
                print(f'Group {name}/{chip_id} was deleted')
            f.create_dataset(f"{name}/{chip_id}/thickness",data=self.thickness)
            f.create_dataset(f"{name}/{chip_id}/error",data=self.error)
            print(f'Created {name}/thickness = {self.thickness}')
            print(f'Created {name}/error = {self.error}')
            print('--------------------------------------------------------')

    def load_result_for_AuTi(self):
        print('--------------------------------------------------------')
        print('Loading HDF5 file...')
        print(f'Filename = {self.hdf5}')
        with h5py.File(self.hdf5,"r") as f:
            for e,i in enumerate(f['Au'].keys()):
                if e == 0:
                    self.Au_thickness_list = f['Au'][f'{i}/thickness'][...]
                    self.Au_error_list = f['Au'][f'{i}/error'][...]
                else :
                    self.Au_thickness_list = np.append(self.Au_thickness_list,f['Au'][f'{i}/thickness'][...])
                    self.Au_error_list = np.append(self.Au_error_list,f['Au'][f'{i}/error'][...])

            for e,i in enumerate(f['Au+Ti'].keys()):
                if e == 0:
                    self.AuTi_thickness_list = f['Au+Ti'][f'{i}/thickness'][...]
                    self.AuTi_error_list = f['Au+Ti'][f'{i}/error'][...]
                else :
                    self.AuTi_thickness_list = np.append(self.AuTi_thickness_list,f['Au+Ti'][f'{i}/thickness'][...])
                    self.AuTi_error_list = np.append(self.AuTi_error_list,f['Au+Ti'][f'{i}/error'][...])
            self.id_list       = list(f['Au'].keys())
        self.Ti_thickness_list = self.AuTi_thickness_list - self.Au_thickness_list
        self.Ti_error_list     = np.sqrt(self.AuTi_error_list**2+self.AuTi_error_list**2)

    def load_result(self):
        print('--------------------------------------------------------')
        print('Loading HDF5 file...')
        print(f'Filename = {self.hdf5}')
        with h5py.File(self.hdf5,"r") as f:
            for e,i in enumerate(f.keys()):
                if e == 0:
                    self.thickness_list = f[f'{i}/thickness'][...]
                    self.error_list = f[f'{i}/error'][...]
                    #self.time_list = f[f'{i}/time'][...]
                else :
                    self.thickness_list = np.append(self.thickness_list,f[f'{i}/thickness'][...])
                    self.error_list = np.append(self.error_list,f[f'{i}/error'][...])
                    self.time_list = np.append(self.time_list,f[f'{i}/time'][...])
        self.error_list = self.error_list[np.argsort(self.thickness_list)[::-1]]
        #self.time_list = self.time_list[np.argsort(self.thickness_list)[::-1]]
        self.thickness_list = self.thickness_list[np.argsort(self.thickness_list)[::-1]]

        print('Finished')

    def thickness_out(self,file,name,**kwargs):
        if 'Hsf' in kwargs:
            Hsf = kwargs['Hsf']
        else:
            Hsf = 0.25
        self.xyout(file,Hsf)
        self.result_plot(subject='thickness')
        chip_id = re.sub('_conv.txt','',file)
        self.chip_id = chip_id
        self.result_plot(subject='result')
        self.save_result(name=name,chip_id=chip_id)

    def multi_analyze(self,name,**kwargs):
        self.conv_list = sorted(glob.glob('*conv.txt'))
        if 'Hsf' in kwargs:
            Hsf = kwargs['Hsf']
        else:
            Hsf = 0.25
        for e,i in enumerate(self.conv_list):
            self.thickness_out(file=i,name=name)


    def Au_Ti_all(self):
        os.chdir('./AfterAu')
        self.multi_analyze(name='Au')
        os.chdir('../AfterTi')
        self.multi_analyze(name='Au+Ti')
        
        self.load_result_for_AuTi()
        self.result_plot('all_thickness')
        os.chdir('../')

    def Au_Ti_all(self):
        os.chdir('../AfterTi')
        self.multi_analyze(name='Ti')
        
        self.load_result_for_AuTi()
        self.result_plot('all_thickness')
        os.chdir('../')

    def etching_rate(self):
        y = -np.diff(self.thickness_list)
        x = np.diff(self.time_list)
        popt, pcov = curve_fit(self.linear,x,y)
        y_residual = y - self.linear(x,popt)
        print(popt,pcov)
        #self.P.gridplot(x,y,y_residual,popt)



## PLOT ##
    def plot_init(self):
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
        plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize']       = 15 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize']       = 15 # 軸だけ変更されます
        plt.rcParams['xtick.direction']       = 'in' # x axis in
        plt.rcParams['ytick.direction']       = 'in' # y axis in 
        plt.rcParams['axes.linewidth']        = 1.0 # axis line width
        plt.rcParams['axes.grid']             = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['scatter.edgecolors']    = 'black'
        self.fs = 25
        self.ps = 80

    def plot_window(self,style):
        self.plot_init()
        if style == "single":
            self.fig = plt.figure(figsize=(8,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")

        if style == 'multi':

            self.fig=plt.figure(figsize=(14.2,10.7))
            gs_master=GridSpec(nrows=2,ncols=3,width_ratios=[2,1,1])
            self.ax1=self.fig.add_subplot(gs_master[1,0])
            self.ax2=self.fig.add_subplot(gs_master[1,1])
            self.ax3=self.fig.add_subplot(gs_master[1,2])
            self.ax4=self.fig.add_subplot(gs_master[0,0])
            self.ax5=self.fig.add_subplot(gs_master[0,1])
            self.ax6=self.fig.add_subplot(gs_master[0,2])
            self.ax1.grid(linestyle='dashed')
            self.ax2.grid(linestyle='dashed')
            self.ax3.grid(linestyle='dashed')
            self.ax4.grid(linestyle='dashed')
            self.ax5.grid(linestyle='dashed')
            self.ax6.grid(linestyle='dashed')



    def result_plot(self,subject):
        if subject == "thickness":
            self.plot_window(style="single")
            self.ax.plot(self.x,self.y,color='black')
            self.ax.set_title(self.title)
            span = SpanSelector(
            self.ax,
            self.thickness_selector,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.2, facecolor="tab:red"),
            interactive=True,
            drag_from_anywhere=True
            )
            plt.show()

        if subject == "result":
            self.plot_window('single')
            self.ax.plot(self.x,self.y,color='black',label='data')
            self.ax.plot(self.x[self.left_mask],self.y[self.left_mask],color='red')
            self.ax.plot(self.x[self.right_mask],self.y[self.right_mask],color='red')
            self.ax.plot(self.x[self.top_mask],self.y[self.top_mask],color='red',label='selected region')
            self.ax.set_xlabel(r'$\rm Length\ [\mu m]$',fontsize=20)
            self.ax.set_ylabel(r'$\rm Thickness\ [nm]$',fontsize=20)
            self.ax.set_title(f'{self.chip_id}',fontsize=20)
            self.fig.savefig(f'{self.chip_id}.png',dpi=300)

        if subject == "thickness_avg":
            pass

        if subject == "all_thickness":
            self.plot_window(style="multi")
            Au = self.Au_thickness_list
            Ti = self.Ti_thickness_list
            Au_err = self.Au_error_list
            Ti_err = self.Ti_error_list
            nl = np.array(self.id_list)
            print(nl)
            self.ax1.errorbar(nl,Au,yerr=Au_err,capsize=5, fmt='o', markersize=10,color='red',label="Au")
            Au_avg = np.average(Au,weights=1/Au_err)
            Ti_avg = np.average(Ti,weights=1/Ti_err)
            Au_std = np.std(Au,ddof=1)
            Ti_std = np.std(Ti,ddof=1)
            j1 = (nl == "W01") | (nl == "W04") | (nl == "W08") | (nl == "W12") | (nl == "W15")
            j2 = (nl == "W07") | (nl == "W08") | (nl == "W09")
            Au_vper = (Au[j1][np.argmax(Au[j1])]-Au[j1][np.argmin(Au[j1])])*100/Au[j1][np.argmin(Au[j1])]
            Ti_vper = (Ti[j1][np.argmax(Ti[j1])]-Ti[j1][np.argmin(Ti[j1])])*100/Ti[j1][np.argmin(Ti[j1])]
            Au_hper = (Au[j2][np.argmax(Au[j2])]-Au[j2][np.argmin(Au[j2])])*100/Au[j2][np.argmin(Au[j2])]
            Ti_hper = (Ti[j2][np.argmax(Ti[j2])]-Ti[j2][np.argmin(Ti[j2])])*100/Ti[j2][np.argmin(Ti[j2])]
            self.ax2.errorbar(nl[j1],Au[j1],yerr=Au_err[j1],capsize=5, fmt='o', markersize=10,color='red')
            self.ax2.plot(nl[j1],Au[j1],label=r"$\rm Difference:{:.1f} \%$".format(Au_vper),color='red')
            self.ax3.errorbar(nl[j2],Au[j2],yerr=Au_err[j2],capsize=5, fmt='o', markersize=10,color='red')
            self.ax3.plot(nl[j2],Au[j2],label=r"$\rm Difference:{:.1f} \%$".format(Au_hper),color='red')
            self.ax4.errorbar(nl,Ti,yerr=Ti_err,capsize=5, fmt='o', markersize=10,color='blue',label="Ti")
            self.ax5.errorbar(nl[j1],Ti[j1],yerr=Ti_err[j1],capsize=5, fmt='o', markersize=10,color='blue')
            self.ax5.plot(nl[j1],Ti[j1],label=r"$\rm Difference:{:.1f} \%$".format(Ti_vper),color='blue')
            self.ax6.errorbar(nl[j2],Ti[j2],yerr=Ti_err[j2],capsize=5, fmt='o', markersize=10,color='blue')
            self.ax6.plot(nl[j2],Ti[j2],label=r"$\rm Difference:{:.1f} \%$".format(Ti_hper),color='blue')
            self.ax1.set_xlabel("Chip ID",fontsize=16)
            self.ax1.set_ylabel("Thickness[nm]",fontsize=16)
            self.ax4.set_xlabel("Chip ID",fontsize=16)
            self.ax4.set_ylabel("Thickness[nm]",fontsize=16)
            self.ax1.axhspan(Au_avg*0.95,Au_avg*1.05,color="lightgray")
            self.ax1.hlines(Au_avg,nl[0],nl[-1],linestyle="dashed",color="black",label="Average : %.1fnm" %(Au_avg))
            self.ax4.axhspan(Ti_avg*0.95,Ti_avg*1.05,color="lightgray")
            self.ax4.hlines(Ti_avg,nl[0],nl[-1],linestyle="dashed",color="black",label="Average : %.1fnm" %(Ti_avg))
            self.ax2.hlines(np.average(Au[j1]),nl[j1][0],nl[j1][-1],linestyle="dashed",color="black")
            self.ax3.hlines(np.average(Au[j2]),nl[j2][0],nl[j2][-1],linestyle="dashed",color="black")
            self.ax5.hlines(np.average(Ti[j1]),nl[j1][0],nl[j1][-1],linestyle="dashed",color="black")
            self.ax6.hlines(np.average(Ti[j2]),nl[j2][0],nl[j2][-1],linestyle="dashed",color="black")
            hans, labs = self.ax1.get_legend_handles_labels()
            self.ax1.legend(handles=hans[::-1], labels=labs[::-1], fontsize=13)
            self.ax2.legend(handlelength=0,fontsize=13)
            self.ax3.legend(handlelength=0,fontsize=13)
            hans, labs = self.ax4.get_legend_handles_labels()
            self.ax4.legend(handles=hans[::-1], labels=labs[::-1], fontsize=13)
            self.ax5.legend(handlelength=0,fontsize=13)
            self.ax6.legend(handlelength=0,fontsize=13)
            self.fig.text(0.5,0.95,'SEED221118b', ha='center',fontsize=20)
            self.fig.savefig('thickness.png',dpi=300)
            plt.show()


