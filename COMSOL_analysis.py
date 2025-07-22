import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from lmfit import Model
import scipy
from scipy.optimize import curve_fit
from pytes import Util,Filter,Analysis
from scipy import signal,interpolate,integrate
from scipy.signal import find_peaks
import dask
import dask.array as da
import bottleneck as bn
import time
import os
from Basic import Plotter


__author__ =  'Keita Tanaka'
__version__=  '2.0.0' #2022.06.06

print('===============================================================================')
print(f"COMSOL Data Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class AxionTES:
    def __init__(self):
        self.kb        = 1.381e-23 ##[J/K]
        self.Rshunt    = 3.9 * 1e-3
        self.Min       = 9.82 * 1e-11
        self.Rfb       = 1e+5
        self.Mfb       = 8.5 * 1e-11
        self.JtoeV     = 1/1.60217662e-19
        self.excVtoI   = self.Mfb / (self.Min * self.Rfb)
        self.filelist  =  sorted(glob.glob(f"*.txt"))
        self.P = Plotter()
        self.savehdf5  = 'test.hdf5'


    def load_data(self,file):
        f = np.genfromtxt(f"../data/{file}/{file}.txt")
        self.t_sim    = f[:,0]*1e-3  # Simulation time [s]
        self.Tsor_sim = f[:,1]*1e-3  # Source temperature [K]
        self.Tabs_sim = f[:,2]*1e-3  # Absorber temperature [K]
        self.Tpath_sim = f[:,3]*1e-3  # Au path temperature [K]
        self.Ttes_sim = f[:,4]*1e-3  # TES temperature [K]
        self.Tmem_sim = f[:,5]*1e-3  # Membrane temperature [K]
        self.I_cur    = f[:,6]*1e-6  # TES Current calculated by electric circuit module [A]

    def offset_correction(self):
        self.Tb       = self.Ttes_sim[50]
        self.Ib       = self.I_cur[50]  # TES base current
        self.I_cur   -= self.I_cur[50]
        self.I_sim    = -self.I_cur

    def show_properity(self):
        self.t_sim_min    = np.min(np.diff(self.t_sim))
        if self.t_sim_min == 0:
            self.t_sim_min = np.min(np.diff(self.t_sim)[np.diff(self.t_sim)!=0])
        self.pulse_h  = np.max(self.I_sim)
        self.ph_f     = self.pulse_h/np.exp(1)
        self.ph_ten   = self.pulse_h/10
        self.p_s      = self.t_sim[np.argmax(self.I_sim)]
        self.tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(self.ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        self.tau_r    = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-1.025e-3
        self.tau_r_ten = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_ten-self.I_sim[:np.argmax(self.I_sim)]))]
        print(f"Pulse Height(COMSOL data)     = {self.pulse_h*1e+6} uA")
        print(f"Temperature Height(Ttes data) = {np.max(self.Ttes_sim)*1e+3} mK")
        print(f"Temperature Height(Tabs data) = {np.max(self.Tabs_sim)*1e+3} mK")
        print(f"Temperature Height(Tsor data) = {np.max(self.Tsor_sim)*1e+3} mK")
        print(f"Base Temperature = {self.Tb*1e+3} mK")
        print(f"Base Current = {self.Ib*1e+6} uA")
        print(f"Fall Time = {self.tau_f*1e+6} us")
        print(f"Rise Time = {self.tau_r*1e+6} us")
        print(f"Minimum of time difference = {self.t_sim_min} s")

    def load_data_with_instant_analysis(self,file,**kwargs):
        self.load_data(file)
        self.offset_correction()
        self.show_properity()
        if 'multi' in kwargs:
            color = kwargs['color']
            self.P.plotting(self.t_sim*1e+3,self.I_sim*1e+6,new_window=False,color=color,lw=2,label=kwargs["label"])
        else:
            self.P.plotting(self.t_sim*1e+3,self.I_sim*1e+6,lw=2,label=kwargs["label"])

    def cor_somedata(self,files):
        for e,file in enumerate(files):
            if e == 0:
                self.load_data_with_instant_analysis(file,label=file)
            elif e == 1:
                print('multi')
                self.load_data_with_instant_analysis(file,multi=True,color="Red",label=file)
            elif e == 2:
                print('multi')
                self.load_data_with_instant_analysis(file,multi=True,color="Black",label=file)
        plt.show()

class Base:
    """_summary_
    Basic analysis class.
    """
        
    def __init__(self,savehdf5='./test.hdf5',debug=False,device='Axion'):
        self.kb        = 1.381e-23 ##[J/K]
        self.Rshunt    = 3.9 * 1e-3
        self.Min       = 9.82 * 1e-11
        self.Rfb       = 1e+5
        self.Mfb       = 8.5 * 1e-11
        self.JtoeV     = 1/1.60217662e-19
        self.excVtoI   = self.Mfb / (self.Min * self.Rfb)
        self.P = Plotter()
        self.savehdf5  = savehdf5
        self.debug     = debug
        home = os.environ['HOME']
        self.result_file = f'{home}/Dropbox/share/work/microcalorimeters/experiment/COMSOL/AxionTES/5to5/high_rate/analysis/result.hdf5'
        self.result_file_G = f'{home}/Dropbox/share/work/microcalorimeters/experiment/COMSOL/AxionTES/5to5/high_rate/analysis/result_G.hdf5'
        self.setting_exl = f'{home}/Dropbox/share/work/microcalorimeters/experiment/COMSOL/AxionTES/5to5/high_rate/data/params/analysis/Book1.xlsx'
        self.device = device
        if self.device == 'Axion' : 
            self.file_list   = ['run01.txt','run02.txt','run03.txt','run04.txt','run05.txt','run06.txt','run07.txt','run08.txt','run09.txt','run10.txt','run11.txt','run12.txt','run13.txt','run14.txt','run15.txt']
            self.number_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
            self.single_num  = ['11','12','13','14','15']
            self.dist = [90,70,50,30,10,90,70,50,30,10,90,70,50,30,10]
            self.col_list = ['Blue','Blue','Blue','Blue','Blue','Green','Green','Green','Green','Green','Red','Red','Red','Red','Red']

        if self.device == 'Axion' : 
            self.file_list   = ['run11.txt','run12.txt','run13.txt','run14.txt','run15.txt']
            self.number_list = ['11','12','13','14','15']
            self.single_num  = ['11','12','13','14','15']
            self.dist = [90,70,50,30,10]
            self.col_list = ['Blue','Blue','Blue','Blue','Blue']

        if self.device == 'Axion_sel' : 
            self.file_list   = ['run11.txt','run15.txt']
            self.number_list = ['11','15']
            self.single_num  = ['11','15']
            self.dist = [90,10]
            self.col_list = ['Blue','Blue']

        if self.device == 'Axion_sp' : 
            self.file_list   = ['run01.txt','run02.txt','run03.txt','run04.txt','run06.txt','run07.txt','run08.txt','run09.txt','run11.txt','run12.txt','run13.txt','run14.txt']
            self.number_list = ['01','02','03','04','06','07','08','09','11','12','13','14']
            self.single_num  = ['11','12','13','14']
            self.dist = [90,70,50,30,90,70,50,30,90,70,50,30]
            self.col_list = ['Blue','Blue','Blue','Blue','Green','Green','Green','Green','Red','Red','Red','Red']

        elif self.device == 'Normal' :
            self.file_list   = ['run1.txt','run2.txt','run3.txt','run4.txt','run5.txt','run6.txt','run7.txt','run8.txt','run9.txt']
            self.number_list = ['1','2','3','4','5','6','7','8','9']
            self.single_num  = ['1','2','3','4','5','6','7','8','9']
            self.dist = [-50,0,50,-50,0,50,-50,0,50]
            self.col_list = ['Blue','Blue','Blue','Green','Green','Green','Red','Red','Red']

        elif self.device == 'Multi':
            self.file_list = sorted(glob.glob('*.txt'))
            self.number_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
            self.single_num  = ['11','12','13','14','15']
            self.dist = [90,70,50,30,10,90,70,50,30,10,90,70,50,30,10]
            self.col_list = ['Blue','Blue','Blue','Blue','Blue','Green','Green','Green','Green','Green','Red','Red','Red','Red','Red']

        elif self.device == 'Hydra' :
            self.file_list   = ['run1.txt','run2.txt','run3.txt','run4.txt']
            self.number_list = ['1','2','3','4']
            self.single_num  = ['1','2','3','4']
            self.dist = [1,2,3,4]
            self.col_list = ['Blue','Blue','Blue','Blue']
        # self.file_list   = ['run1.txt','run2.txt','run3.txt','run4.txt','run5.txt','run6.txt']
        # self.number_list = ['1','2','3','4','5','6']
        # self.file_list   = ['run11.txt','run13.txt','run15.txt']
        # self.number_list = ['11','13','15']
        # self.single_num = ['1','3','5','11','13','15']
        # self.file_list   = ['run11.txt','run12.txt','run13.txt','run14.txt','run15.txt']
        # self.number_list = ['11','12','13','14','15']
        # self.file_list   = ['run1.txt','run2.txt','run3.txt','run4.txt','run6.txt','run7.txt','run8.txt','run9.txt','run11.txt','run12.txt','run13.txt','run14.txt']
        # self.number_list = ['1','2','3','4','6','7','8','9','11','12','13','14']


    def set_attribute(self,t:float,wi:float,ov:float,shape:str="normal"):
        self.thickness = t
        self.width     = wi
        self.over      = ov
        self.shape     = shape

    def gaus(self, x, norm, sigma, mu):
        return norm * np.exp(-(x-mu)**2 / (2*sigma**2))

    def linear_x(self, x, a, b):
        return a * x + b

    def linear_y(self, y, a, b):
        return (y - b)/a
    
    def req_lin(self,x1,y1,x2,y2):
        a = (y1-y2)/(x1-x2)
        b = (x2*y1-x1*y2)/(x2-x1)
        return a, b

    def LPF_GC(self,x,times,sigma):
        sigma_k = sigma/(times[1]-times[0]) 
        kernel = np.zeros(int(round(3*sigma_k))*2+1)
        for i in range(kernel.shape[0]):
            kernel[i] =  1.0/np.sqrt(2*np.pi)/sigma_k * np.exp((i - round(3*sigma_k))**2/(- 2*sigma_k**2))
            
        kernel = kernel / kernel.sum()
        x_long = np.zeros(x.shape[0] + kernel.shape[0])
        x_long[kernel.shape[0]//2 :-kernel.shape[0]//2] = x
        x_long[:kernel.shape[0]//2 ] = x[0]
        x_long[-kernel.shape[0]//2 :] = x[-1]
            
        x_GC = np.convolve(x_long,kernel,'same')
        
        return x_GC[kernel.shape[0]//2 :-kernel.shape[0]//2]

    def lowcut_buter(self,lowcut,fs,degree):
        nyq = fs/2
        low = lowcut/nyq
        self.b, self.a = signal.butter(degree,low,'low')

    def load_data(self,file:str):
        """_summary_
        Load COMSOL data and store each variable.

        Args:
            file (str): filename of the data file. You must specify filename with file path.

        Return:
            None. Define below variable.
            self.t_sim     # Simulation time [s]
            self.Tsor_sim  # Source temperature [K]
            self.Tabs_sim  # Absorber temperature [K]
            self.Tpath_sim # Au path temperature [K]
            self.Ttes_sim  # TES temperature [K]
            self.Tmem_sim  # Membrane temperature [K]
            self.I_cur     # TES Current calculated by electric circuit module [A]   

        """
        print(f"Loading {file} ...")
        f = np.genfromtxt(f"{file}")
        if self.device == 'Axion' or self.device == 'Axion_sel':
            self.t_rer = f[:,0]*1e-3
            self.t_sim, dup_id    = np.unique(f[:,0]*1e-3,return_index=True) # Simulation time [s]
            self.Tsor_sim = f[:,1][dup_id]*1e-3  # Source temperature [K]
            self.Tabs_sim = f[:,2][dup_id]*1e-3  # Absorber temperature [K]
            self.Tpath_sim = f[:,3][dup_id]*1e-3  # Au path temperature [K]
            self.Ttes_sim = f[:,4][dup_id]*1e-3  # TES temperature [K]
            self.Tmem_sim = f[:,5][dup_id]*1e-3  # Membrane temperature [K]
            self.I_cur    = f[:,6][dup_id]*1e-6  # TES Current calculated by electric circuit module [A]
            # self.hf_1     = f[:,7][dup_id]  # Heat flux [W/m^2]. Fe to Au.
            # self.hf_2     = f[:,8][dup_id]  # Heat flux [W/m^2]. Au to membrane (under Fe).  
            # self.hf_3     = f[:,9][dup_id]  # Heat flux [W/m^2]. Au to membrane (between Fe and TES).  
            # self.hf_4     = f[:,10][dup_id]  # Heat flux [W/m^2]. Au to TES (on TES) .  
        elif self.device == 'Normal' or self.device == 'Multi':
            self.t_sim, dup_id    = np.unique(f[:,0]*1e-3,return_index=True) 
            self.Tsor_sim = f[:,1][dup_id]*1e-3  # Source temperature [K]
            self.Tabs_sim = f[:,1][dup_id]*1e-3  # Absorber temperature [K]
            self.Tpath_sim = f[:,1][dup_id]*1e-3  # Au path temperature [K]
            self.Ttes_sim = f[:,2][dup_id]*1e-3  # TES temperature [K]
            self.Tmem_sim = f[:,3][dup_id]*1e-3  # Membrane temperature [K]
            self.I_cur    = f[:,4][dup_id]*1e-6  # TES Current calculated by electric circuit module [A]

        elif self.device == 'Hydra' :
            self.t_sim, dup_id    = np.unique(f[:,0]*1e-3,return_index=True) 
            self.Tsor_sim = f[:,1][dup_id]*1e-3  # Source temperature [K]
            self.Tabs_sim = f[:,1][dup_id]*1e-3  # Absorber temperature [K]
            self.Tpath_sim = f[:,1][dup_id]*1e-3  # Au path temperature [K]
            self.Ttes_sim = f[:,2][dup_id]*1e-3  # TES temperature [K]
            self.Tmem_sim = f[:,3][dup_id]*1e-3  # Membrane temperature [K]
            self.I_cur    = f[:,6][dup_id]*1e-6  # TES Current calculated by electric circuit module [A]

        if self.debug:
            print(self.t_sim)

    def cal_integrate(self):
        ind = 10
        Tabs_sim = self.Tabs_sim - self.Tabs_sim[ind]
        Tpath_sim = self.Tpath_sim - self.Tpath_sim[ind]
        Ttes_sim = self.Ttes_sim - self.Ttes_sim[ind]
        Tmem_sim = self.Tmem_sim - self.Tmem_sim[ind]
        self.Ttes_max = np.max(Ttes_sim)
        self.Tpath_max = np.max(Tpath_sim)
        self.Tabs_max = np.max(Tabs_sim)
        self.Tmem_max = np.max(Tmem_sim)
        self.I_int = integrate.simps(self.I_sim,self.t_sim)
        self.Tabs_int = integrate.simps(self.Tabs_sim,self.t_sim)
        self.Tpath_int = integrate.simps(self.Tpath_sim,self.t_sim)
        self.Ttes_int = integrate.simps(self.Ttes_sim,self.t_sim)
        self.Tmem_int = integrate.simps(self.Tmem_sim,self.t_sim)
        delTtes = np.diff(self.Ttes_sim)
        max_ind = np.argmax(self.Ttes_sim)-1
        self.Ttes_in = integrate.simps(delTtes[:max_ind],self.t_sim[:max_ind])
        self.Ttes_out = integrate.simps(delTtes[max_ind:],self.t_sim[1:][max_ind:])

    def show_prop(self):
        self.Ttes_max = np.max(self.Ttes_sim)
        self.Tabs_max = np.max(self.Tabs_sim)
        self.Tpath_max = np.max(self.Tpath_sim)
        self.Tmem_max = np.max(self.Tmem_sim)
        self.Ttes_max_time = self.t_sim[np.argmax(self.Ttes_sim)] - 1.025e-3
        self.Tabs_max_time = self.t_sim[np.argmax(self.Tabs_sim)] - 1.025e-3
        self.Tpath_max_time = self.t_sim[np.argmax(self.Tpath_sim)] - 1.025e-3
        self.Tmem_max_time = self.t_sim[np.argmax(self.Tmem_sim)] - 1.025e-3
        # self.delT     = self.hf_3 - np.median(self.hf_3[:20])
        # self.integ_pm   = integrate.simps(self.delT, self.t_rer[:-1])
        print('---------------------------------------------')
        print('properity')
        print(f'TES max temp = {self.Ttes_max*1e+3} mK')
        print(f'Abs max temp = {self.Tabs_max*1e+3} mK')
        print(f'Path max temp = {self.Tpath_max*1e+3} mK')
        print(f'Mem max temp = {self.Tmem_max*1e+3} mK')
        print(f'Ttes max time = {self.Ttes_max_time*1e+6} usec')
        print(f'Tabs max time = {self.Tabs_max_time*1e+6} usec')
        print(f'Tpath max time = {self.Tpath_max_time*1e+6} usec')
        print(f'Tmem max time = {self.Tmem_max_time*1e+6} usec')

    def plot_window(self,style='four'):
        self.P
        if style == 'four':
            self.fig  = plt.figure(figsize=(12,8))
            self.ax1  = plt.subplot(221)
            self.ax2  = plt.subplot(222,sharex=self.ax1)
            self.ax3  = plt.subplot(223,sharex=self.ax1)
            self.ax4  = plt.subplot(224,sharex=self.ax1)
            self.ax1.grid(linestyle="dashed",color='gray')
            self.ax2.grid(linestyle="dashed",color='gray')
            self.ax3.grid(linestyle="dashed",color='gray')
            self.ax4.grid(linestyle="dashed",color='gray')
        if style == 'four_noshare':
            self.fig  = plt.figure(figsize=(12,8))
            self.ax1  = plt.subplot(221)
            self.ax2  = plt.subplot(222)
            self.ax3  = plt.subplot(223)
            self.ax4  = plt.subplot(224)
            self.ax1.grid(linestyle="dashed",color='gray')
            self.ax2.grid(linestyle="dashed",color='gray')
            self.ax3.grid(linestyle="dashed",color='gray')
            self.ax4.grid(linestyle="dashed",color='gray')
        if style == 'three':
            self.fig  = plt.figure(figsize=(12,8))
            self.ax1  = plt.subplot(221)
            self.ax3  = plt.subplot(223,sharex=self.ax1,sharey=self.ax1)
            self.ax4  = plt.subplot(224,sharex=self.ax1,sharey=self.ax1)
            self.ax1.grid(linestyle="dashed",color='gray')
            self.ax3.grid(linestyle="dashed",color='gray')
            self.ax4.grid(linestyle="dashed",color='gray')
        if style == 'three_noshare':
            self.fig  = plt.figure(figsize=(12,8))
            self.ax1  = plt.subplot(221)
            self.ax3  = plt.subplot(223)
            self.ax4  = plt.subplot(224)
            self.ax1.grid(linestyle="dashed",color='gray')
            self.ax3.grid(linestyle="dashed",color='gray')
            self.ax4.grid(linestyle="dashed",color='gray')
        if style == 'single':
            self.fig  = plt.figure(figsize=(8,6))
            self.ax1  = plt.subplot(111)
            self.ax1.grid(linestyle="dashed",color='gray')

    def plot_result_cor(self,name:str):
        self.plot_window('three_noshare')
        # self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        # self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax1.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm PH \ [\mu A]$",fontsize=20)
        self.ax3.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)

        col = ['Red','Green','Blue']
        with h5py.File(self.result_file,"r") as f: 
            
            for e,i in enumerate(f[f'{name}'].keys()):
                
                ph = f[f'{name}/{i}/ph'][:]
                tau_r = f[f'{name}/{i}/tau_r'][:]
                tau_f = f[f'{name}/{i}/tau_f'][:]
                dist = f[f'{name}/{i}/dist'][:]
                pos = float(i)
                pos = np.array([pos,pos,pos,pos,pos])
                self.ax1.scatter(pos,ph*1e+6,color=col[e],label=f'{i} um')
                self.ax3.scatter(pos,tau_r*1e+6,color=col[e])
                self.ax4.scatter(pos,tau_f*1e+6,color=col[e])
        self.ax1.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_result.png',dpi=300)
        plt.show()

    def plot_result_temp(self,name:str):
        self.plot_window('three_noshare')
        # self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        # self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax1.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Tabs \ [mK]$",fontsize=20)
        self.ax3.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Tpath \ [mK]$",fontsize=20)
        self.ax4.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Ttes \ [mK]$",fontsize=20)

        col = ['Red','Green','Blue']
        with h5py.File(self.result_file,"r") as f: 
            
            for e,i in enumerate(f[f'{name}'].keys()):
                
                Ttes = f[f'{name}/{i}/Ttes_max'][:]
                Tabs = f[f'{name}/{i}/Tabs_max'][:]
                Tpath = f[f'{name}/{i}/Tpath_max'][:]
                Tmem = f[f'{name}/{i}/Tmem_max'][:]
                pos = float(i)
                pos = np.array([pos,pos,pos,pos,pos])
                self.ax1.scatter(pos,Tabs*1e+3,color=col[e],label=f'{i} um')
                self.ax3.scatter(pos,Tpath*1e+3,color=col[e])
                self.ax4.scatter(pos,Ttes*1e+3,color=col[e])
        self.ax1.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_result.png',dpi=300)
        plt.show()

    def plot_result_cor_integ(self,name:str):
        self.plot_window('four_noshare')
        # self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        # self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax1.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm PH \ [\mu A]$",fontsize=20)
        self.ax3.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)

        col = ['Red','Green','Blue']
        pr = np.array([0.766,0.560,0.396])
        with h5py.File(self.result_file,"r") as f: 
            
            for e,i in enumerate(f[f'{name}'].keys()):
                
                Tabs_int = f[f'{name}/{i}/Tabs_int'][:]
                Ttes_int = f[f'{name}/{i}/Ttes_int'][:]
                Tpath_int = f[f'{name}/{i}/Tpath_int'][:]
                Ttes_in = f[f'{name}/{i}/Ttes_in'][:]
                Ttes_out = f[f'{name}/{i}/Ttes_out'][:]
                I_int = f[f'{name}/{i}/I_int'][:]
                pos = float(i)
                pos = np.array([pos,pos,pos,pos,pos])
                self.ax1.scatter(pos,Tabs_int,color=col[e],label=f'{i} um')
                self.ax2.scatter(pos,Ttes_in,color=col[e])
                self.ax3.scatter(pos,Ttes_out,color=col[e])
                self.ax4.scatter(pos,Tpath_int,color=col[e])
        #self.ax1.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_result.png',dpi=300)
        plt.show()

    def plot_result_cor_norm(self,name:str):
        self.plot_window('three_noshare')
        # self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        # self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax1.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm PH \ fluctuation\ [\%]$",fontsize=20)
        self.ax3.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ fluctuation \ [\%]$",fontsize=20)
        self.ax4.set_xlabel(rf"$\rm {name} \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ fluctuation \ [\%]$",fontsize=20)

        col = ['blue','cyan','green','orange','red']
        with h5py.File(self.result_file,"r") as f: 
            
            for e,i in enumerate(f[f'{name}'].keys()):
                
                ph = f[f'{name}/{i}/ph'][:]
                tau_r = f[f'{name}/{i}/tau_r'][:]
                tau_f = f[f'{name}/{i}/tau_f'][:]
                dist = f[f'{name}/{i}/dist'][:]
                pos = float(i)
                pos = np.array([pos,pos,pos,pos,pos])
                for j,c in enumerate(col):
                    self.ax1.scatter(pos[j],(ph[j]-ph[2])/ph[2]*100,color=c,label=f'{i} um')
                    self.ax4.scatter(pos[j],(tau_f[j]-tau_f[2])/tau_f[2]*100,color=c)
                    self.ax3.scatter(pos[j],(tau_r[j]-tau_r[2])/tau_r[2]*100,color=c)
        #self.ax1.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_norm_result.png',dpi=300)
        plt.show()

    def plot_pulse(self,name:str,att:str):
        self.plot_window('four_noshare')
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm PH \ [\mu A]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)

        file = self.file_list
        col = ['blue','cyan','green','orange','red','cyan','green','orange','red']     
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            self.ax1.plot(self.t_sim*1e+3,self.I_sim*1e+6,label=self.number_list[e],color=col[e])
            self.ax2.scatter(self.dist[e],self.pulse_h*1e+6,color=col[e],label=self.number_list[e])
            self.ax3.scatter(self.dist[e],self.tau_r*1e+6,color=col[e])
            self.ax4.scatter(self.dist[e],self.tau_f*1e+6,color=col[e])
        #self.ax1.legend(loc='upper right',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_tmpl.png',dpi=300)
        plt.show()

    def plot_temp(self,name:str):
        self.plot_window('four_noshare')
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Absorber \ temp \ [mK]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Au \ strap \ temp \ [mK]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm TES \ temp \ [mK]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Membrane \ temp \ [mK]$",fontsize=20)

        folders = self.file_list     
        for e,i in enumerate(folders):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            self.ax1.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label=i,color=cm.jet([e/len(folders)]))
            self.ax2.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,color=cm.jet([e/len(folders)]),label=i)
            self.ax3.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,color=cm.jet([e/len(folders)]))
            self.ax4.plot(self.t_sim*1e+3,self.Tmem_sim*1e+3,color=cm.jet([e/len(folders)]))
        self.ax1.legend(loc='upper right',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_tmpl.png',dpi=300)
        plt.show()

    def plot_multi_temp(self,name:str):
        self.plot_window('four_noshare')
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Absorber \ temp \ [mK]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Au \ strap \ temp \ [mK]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm TES \ temp \ [mK]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Membrane \ temp \ [mK]$",fontsize=20)

        folders = glob.glob('*')     
        for e,i in enumerate(folders):
            os.chdir(i)
            self.load_data('run15.txt')
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            self.ax1.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label=i,color=cm.jet([e/len(folders)]))
            self.ax2.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,color=cm.jet([e/len(folders)]),label=i)
            self.ax3.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,color=cm.jet([e/len(folders)]))
            self.ax4.plot(self.t_sim*1e+3,self.Tmem_sim*1e+3,color=cm.jet([e/len(folders)]))
            os.chdir('../')
        self.ax1.legend(loc='upper right',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_tmpl.png',dpi=300)
        plt.show()

    def plot_each_temp(self,name:str):
        self.plot_window('three')
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Length \ 125 \ \mu m$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Au \ strap \ temp \ [mK]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Length \ 50 \ \mu m$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Length \ 10 \ \mu m$",fontsize=20)

        folders = glob.glob('*')     
        for e,i in enumerate(folders):
            os.chdir(i)
            print(i)
            self.load_data('run15.txt')
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            if e == 0:
                self.ax1.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label='Tabs',color='Red')
                self.ax1.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,color='Blue',label='Tpath')
                self.ax1.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,color='Green',label='Ttes')
            if e == 1:
                self.ax3.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label='Tabs',color='Red')
                self.ax3.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,color='Blue',label='Tpath')
                self.ax3.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,color='Green',label='Ttes')
            if e == 2:
                self.ax4.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label='Tabs',color='Red')
                self.ax4.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,color='Blue',label='Tpath')
                self.ax4.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,color='Green',label='Ttes')
            
            os.chdir('../')
        self.ax1.legend(loc='upper right',bbox_to_anchor=(1, 1),fontsize=15)
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_each_tmpl.png',dpi=300)
        plt.show()

    def sim_result_out(self,name,att):
        file = self.file_list 
        ph = []
        tau_r = []
        tau_f = []
        Tabs_int = []
        Ttes_int = []
        Tpath_int = []
        Tmem_int = []    
        I_int = []   
        Ttes_in = []
        Ttes_out = []
        Ttes = []
        Tabs = []
        Tpath = []
        Tmem = []
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            self.cal_integrate()
            ph.append(self.pulse_h)
            tau_r.append(self.tau_r)
            tau_f.append(self.tau_f)
            I_int.append(self.I_int)
            Tabs_int.append(self.Tabs_int)
            Ttes_int.append(self.Ttes_int)
            Tpath_int.append(self.Tpath_int)
            Tmem_int.append(self.Tmem_int)
            Ttes_in.append(self.Ttes_in)
            Ttes_out.append(self.Ttes_out)
            Ttes.append(self.Ttes_max)
            Tpath.append(self.Tpath_max)
            Tabs.append(self.Tabs_max)
            Tmem.append(self.Tmem_max)
        with h5py.File(self.result_file,"a") as f:
            if name in f.keys():
                if att in f[f'{name}'].keys():
                    del f[f'{name}/{att}']
            f.create_dataset(f'{name}/{att}/ph',data=ph)
            f.create_dataset(f'{name}/{att}/tau_r',data=tau_r)
            f.create_dataset(f'{name}/{att}/tau_f',data=tau_f)
            f.create_dataset(f'{name}/{att}/dist',data=self.dist)
            f.create_dataset(f'{name}/{att}/I_int',data=I_int)
            f.create_dataset(f'{name}/{att}/Tabs_int',data=Tabs_int)
            f.create_dataset(f'{name}/{att}/Tpath_int',data=Tpath_int)
            f.create_dataset(f'{name}/{att}/Ttes_int',data=Ttes_int)
            f.create_dataset(f'{name}/{att}/Tmem_int',data=Tmem_int)
            f.create_dataset(f'{name}/{att}/Ttes_in',data=Ttes_in)
            f.create_dataset(f'{name}/{att}/Ttes_out',data=Ttes_out)
            f.create_dataset(f'{name}/{att}/Ttes_max',data=Ttes)
            f.create_dataset(f'{name}/{att}/Tpath_max',data=Tpath)
            f.create_dataset(f'{name}/{att}/Tabs_max',data=Tabs)
            f.create_dataset(f'{name}/{att}/Tmem_max',data=Tmem)

    def sim_result_out_G(self,name,G_Fe,G_Au):
        file = self.file_list 
        ph = []
        tau_r = []
        tau_f = []
        Tabs_int = []
        Ttes_int = []
        Tpath_int = []
        Tmem_int = []    
        I_int = []   
        Ttes_in = []
        Ttes_out = []
        Ttes = []
        Tabs = []
        Tpath = []
        Tmem = []
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime(0.8,0.2)
            self.cal_integrate()
            ph.append(self.pulse_h)
            tau_r.append(self.tau_r)
            tau_f.append(self.tau_f)
            I_int.append(self.I_int)
            Tabs_int.append(self.Tabs_int)
            Ttes_int.append(self.Ttes_int)
            Tpath_int.append(self.Tpath_int)
            Tmem_int.append(self.Tmem_int)
            Ttes_in.append(self.Ttes_in)
            Ttes_out.append(self.Ttes_out)
            Ttes.append(self.Ttes_max)
            Tpath.append(self.Tpath_max)
            Tabs.append(self.Tabs_max)
            Tmem.append(self.Tmem_max)
        with h5py.File(self.result_file_G,"a") as f:
            if name in f.keys():
                del f[f'{name}']
            f.create_dataset(f'{name}/ph',data=ph)
            f.create_dataset(f'{name}/tau_r',data=tau_r)
            f.create_dataset(f'{name}/tau_f',data=tau_f)
            f.create_dataset(f'{name}/dist',data=self.dist)
            f.create_dataset(f'{name}/I_int',data=I_int)
            f.create_dataset(f'{name}/Tabs_int',data=Tabs_int)
            f.create_dataset(f'{name}/Tpath_int',data=Tpath_int)
            f.create_dataset(f'{name}/Ttes_int',data=Ttes_int)
            f.create_dataset(f'{name}/Tmem_int',data=Tmem_int)
            f.create_dataset(f'{name}/Ttes_in',data=Ttes_in)
            f.create_dataset(f'{name}/Ttes_out',data=Ttes_out)
            f.create_dataset(f'{name}/Ttes_max',data=Ttes)
            f.create_dataset(f'{name}/Tpath_max',data=Tpath)
            f.create_dataset(f'{name}/Tabs_max',data=Tabs)
            f.create_dataset(f'{name}/Tmem_max',data=Tmem)
            att = f[f'{name}'].attrs
            att['G_Fe'] = G_Fe
            att['G_Au'] = G_Au

    def read_setting_file(self,name):
        self.df = pd.read_excel(self.setting_exl)
        print(self.df)
        self.setting_inf = self.df[self.df['filename']==f'{name}']
        print(self.setting_inf)

    def auto_read(self):
        folders = glob.glob('*')
        for folder in folders:
            os.chdir(folder)
            self.read_setting_file(name=folder)
            self.sim_result_out_G(name=folder,G_Fe=self.setting_inf['G_Fe'],G_Au=self.setting_inf['G_Au'])
            os.chdir('../')

    def G_cmap(self):
        E_del_list = []
        G_Fe_list = []
        G_Au_list = []
        df = pd.DataFrame()
        with h5py.File(self.result_file_G,"r") as f:
            for name in f.keys():        
                att = f[f'{name}'].attrs
                G_Fe = att['G_Fe']
                G_Au = att['G_Au']
                G_Fe_list.append(G_Fe)
                G_Au_list.append(G_Au)
                ph = f[f'{name}']['ph'][:]
                ph_max = np.max(ph)
                ph_min = np.min(ph)
                ph_del = ph_max - ph_min
                ph_avg = (ph_max + ph_min)/2
                ItoE = 14.4e+3/ph_avg
                E_del = ph_del*ItoE
                E_del_list.append(E_del)
                df.at[f'{str(G_Au)}',f'{str(G_Fe)}'] = E_del
        print(df)
        res = np.flipud(np.fliplr(df.to_numpy()))
        G_Fe_list = np.array(G_Fe_list)
        G_Au_list = np.array(G_Au_list)
        E_del_list = np.array(E_del_list)
        print(E_del_list)
        msize=0.1
        xx = np.arange(G_Fe_list.min(),G_Fe_list.max()+msize,msize)
        yy = np.arange(G_Au_list.min(),G_Au_list.max()+msize,msize)
        print(xx)
        x,y = np.meshgrid(G_Fe_list,G_Au_list)
        print(x)
        #nz = scipy.interpolate.griddata((G_Fe_list,G_Au_list),E_del_list,(x,y),method='nearest')
        #plt.contourf(x,y,nz)
        # plt.scatter(G_Fe_list,G_Au_list)
        print(res)
        fig, ax = plt.subplots()
        im = ax.imshow(res,origin='lower',cmap='plasma',aspect='auto')
        plt.colorbar(im)
        plt.grid(None)
        for num_r,row in enumerate(res):
            for num_c,value in enumerate(res[num_r]):
                ax.text(num_c,num_r,str(round(res[num_r][num_c],2)),ha='center',va='center',color='white')
        # plt.xticks([0,1,2],[0.025,0.125,0.25])
        # plt.yticks([0,1,2],[0.483,2.415,4.83])
        plt.xlabel(r"$\kappa_{\rm Fe}\ \rm [W/K/m] $")
        plt.ylabel(r"$\kappa_{\rm Au}\ \rm [W/K/m] $")
        plt.show()

    def plot_pos_dep(self,name):
        E_del_list = []
        att_list = []
        with h5py.File(self.result_file,"r") as f:
            for att in f[f'{name}'].keys():
                ph = f[f'{name}'][f'{att}']['ph'][:]
                ph_max = np.max(ph)
                ph_min = np.min(ph)
                ph_del = ph_max - ph_min
                ph_avg = (ph_max + ph_min)/2
                ItoE = 14.4e+3/ph_avg
                E_del = ph_del*ItoE
                E_del_list.append(E_del)
                att_list.append(float(att))
        att_list = np.array(att_list)
        E_del_list = np.array(E_del_list)[np.argsort(att_list)]
        att_list = np.sort(np.array(att_list))
        self.fig  = plt.figure(figsize=(6,5))
        self.ax  = plt.subplot(111)
        self.ax.scatter(att_list,E_del_list,color="black",s=50)
        # self.ax.scatter(att_list[2],E_del_list[2],color="black",s=200)
        # self.ax.scatter(att_list[1],E_del_list[1],marker="*",color="red",s=800)
        #self.ax.plot(att_list,E_del_list,color="black")        
        self.ax.grid(linestyle="dashed")
        self.ax.set_yscale('log')
        self.ax.set_xscale('log')
        #self.ax.hlines(3.0,par[0],par[2],colors="Red",linestyles='dashed')
        #self.ax.hlines(10,par[0],par[2],colors="Blue",linestyles='dashed')
        self.ax.set_xlabel(rf"$\rm {name} \ [um] $",fontsize=30)
        #self.ax.set_xlabel(rf"$\rm length \ [um] $",fontsize=30)
        self.ax.set_ylabel(r"$\rm Energy \ resolution \ [eV] $",fontsize=30)
        self.fig.tight_layout()
        self.fig.savefig(f"{name}_res.png",dpi=300)                  
        plt.show()                

    def plot_pulse_sel(self,name:str):
        self.P
        self.fig  = plt.figure(figsize=(8,6))
        self.ax1  = plt.subplot(111)
        self.ax1.grid(linestyle="dashed")
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm \Delta I \ [\mu A]$",fontsize=20)
        col_list = ['red','green','blue']
        file = ['run15.txt','run13.txt','run11.txt']
        pop  = ['position 1','position 2','position 3']        
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime()
            self.ax1.plot(self.t_sim*1e+3,self.I_sim*1e+6,label=pop[e],color=col_list[e],lw=3)
        plt.legend(fontsize=20)
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_sel_p.png',dpi=300)
        plt.show()

    def plot_hf(self,name:str):
        res = [9.15, 12.24, 9.92]
        par = [50,100,10]
        self.plot_window()
        #self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Heat \ flux [W/m^2]$",fontsize=20)
        #self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Heat \ flux [\mu W/m^2]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Heat \ flux [\mu W/m^2]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Heat \ flux [mW/m^2]$",fontsize=20)
        dist = [90,70,50,30,10,90,70,50,30,10,90,70,50,30,10]
        col_list = ['Blue','Blue','Blue','Blue','Blue','Green','Green','Green','Green','Green','Red','Red','Red','Red','Red']
        file = glob.glob('*.txt')
        Ttes_max_list = []
        Tabs_max_list = []
        Tpath_max_list = []
        Tmem_max_list = []
        tar           = []     
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime()
            self.show_prop()
            Ttes_max_list.append(self.Ttes_max)
            Tabs_max_list.append(self.Tabs_max)
            Tpath_max_list.append(self.Tpath_max)
            Tmem_max_list.append(self.Tmem_max)
            tar.append(self.pulse_h)
            # self.hf_1 -= np.median(self.hf_1[:20]) 
            # self.hf_2 -= np.median(self.hf_2[:20]) 
            # self.hf_3 -= np.median(self.hf_3[:20]) 
            # self.hf_4 -= np.median(self.hf_4[:20]) 
            self.ax1.plot(self.t_rer[:-1]*1e+3,self.hf_1,label=i,color=cm.jet([e/len(file)]))
            self.ax2.plot(self.t_rer[:-1]*1e+3,self.hf_2*1e+6,label=i,color=cm.jet([e/len(file)]))
            self.ax3.plot(self.t_rer[:-1]*1e+3,self.hf_3*1e+6,label=i,color=cm.jet([e/len(file)]))
            self.ax4.plot(self.t_rer[:-1]*1e+3,self.hf_4*1e+3,label=i,color=cm.jet([e/len(file)]))
        #plt.legend()
        print(tar)
        self.ax1.plot(tar)
        self.ax1.set_title('Fe to Au',fontsize=20)
        self.ax2.set_title('Au to membrane (under Fe)',fontsize=20) 
        self.ax3.set_title('Au to membrane (between Fe and TES)',fontsize=20) 
        self.ax4.set_title('Au to TES (on TES)',fontsize=20) 
        self.ax1.set_xticks(rotation=90)
        # self.ax1.set_yscale("log")
        self.ax1.legend()
        self.fig.tight_layout()
        self.fig.savefig(f'{name}_hf.png',dpi=300)
        print(np.median(tar))
        plt.show()

    def plot_cor(self,name:str):
        self.plot_window()
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm ( \ PH \  - PH_{center} \ ) \ / \ PH_{center} \ [\%]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)
        dist = [90,70,50,30,10,90,70,50,30,10,90,70,50,30,10]
        col_list = ['Blue','Blue','Blue','Blue','Blue','Green','Green','Green','Green','Green','Red','Red','Red','Red','Red']
        file = self.file_list
     
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime()
            self.Tabs_sim -= np.median(self.Tabs_sim[:20])
            self.Ttes_sim -= np.median(self.Ttes_sim[:20])
            self.Tpath_sim -= np.median(self.Tpath_sim[:20])
            self.ax1.plot(self.t_sim*1e+3,self.I_sim*1e+6,label=i,color=cm.jet([e/len(file)]))
            self.ax2.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label=i,color=cm.jet([e/len(file)]))
            self.ax3.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,label=i,color=cm.jet([e/len(file)]))
            self.ax4.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,label=i,color=cm.jet([e/len(file)]))
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.ax1.legend()
        self.fig.savefig(f'{name}_cor.png',dpi=300)
        plt.show()

    def plot_corTtime(self,name:str):
        self.plot_window()
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Current [\mu A]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm ( \ PH \  - PH_{center} \ ) \ / \ PH_{center} \ [\%]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm Distance \ [\mu m]$",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)
        file = self.file_list
        ph = []     
        for e,i in enumerate(file):
            self.load_data(i)
            self.time_divider(1e-3)
            self.calc_interp_for_risetime()
            ph.append(self.pulse_h)
        ph = np.array(ph)
        self.ax1.plot(ph*1e+6)
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.fig.tight_layout()
        self.ax1.legend()
        self.fig.savefig(f'{name}_cor.png',dpi=300)
        plt.show()

    def plot_prop(self,name:str,file:str):
        self.plot_window('single')
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Temperature [mK]$",fontsize=20)
        dist = [90,70,50,30,10,90,70,50,30,10,90,70,50,30,10]
        col_list = ['Blue','Blue','Blue','Blue','Blue','Green','Green','Green','Green','Green','Red','Red','Red','Red','Red']

        self.load_data(file)
        self.time_divider(1e-3)
        self.calc_interp_for_risetime()
        # self.Tabs_sim -= np.median(self.Tabs_sim[:20])
        # self.Ttes_sim -= np.median(self.Ttes_sim[:20])
        # self.Tpath_sim -= np.median(self.Tpath_sim[:20])
        # self.Tmem_sim -= np.median(self.Tmem_sim[:20])
        # Ttes_max_id = np.argmax(self.Ttes_sim)
        # Tabs_max_id = np.argmax(self.Tabs_sim)
        # Tpath_max_id = np.argmax(self.Tpath_sim)
        # Tmem_max_id = np.argmax(self.Tmem_sim)
        self.ax1.plot(self.t_sim*1e+3,self.Tabs_sim*1e+3,label='$T_{abs}$',color='red')
        self.ax1.plot(self.t_sim*1e+3,self.Ttes_sim*1e+3,label='$T_{tes}$',color='blue')
        self.ax1.plot(self.t_sim*1e+3,self.Tpath_sim*1e+3,label='$T_{path}$',color='green')
        self.ax1.plot(self.t_sim*1e+3,self.Tmem_sim*1e+3,label='$T_{mem}$',color='orange')
        # self.ax1.set_xscale("log")
        # self.ax1.set_yscale("log")
        self.ax1.set_xlim(0.0249,0.0257)
        self.ax1.set_ylim(206.35,207.0)
        self.ax1.legend(fontsize=20,loc='best')
        self.ax1.set_title(f'Position {name}',fontsize=20)
        self.fig.tight_layout()
        plt.show()
        self.fig.savefig(f'{name}_cor.pdf')

    def cor_data(self):
        dirs = np.array(glob.glob('*'))
        fdirs = np.array([float(x) for x in dirs])
        idx = np.argsort(fdirs)
        print(idx)
        print(len(idx))
        dirs = dirs[idx]
        print(fdirs)
        print(dirs)
        file = ['run15.txt']
        col = ['Red','Blue']
        self.plot_window('single')
        for j in dirs:
            os.chdir(j)
            for e,i in enumerate(file):
                self.load_data(i)
                self.show_prop()
                self.time_divider(1e-3)
                self.calc_interp_for_risetime()
                self.ax1.plot(self.t_sim*1e+3,self.I_cur*1e+6,label=rf'$L={j}$')
            os.chdir('../')
        self.ax1.set_xlabel('Time [msec]')
        self.ax1.set_ylabel(r'$\rm Current\ [\mu A]$')
        self.ax1.legend(fontsize=20)
        plt.show()

    def time_divider(self,limit:float,exporate_zero:bool=True):
        """_summary_
        Devide time of COMSOL data.
        
        Args:
            limit (float): time limit
            exporate_zero (bool, optional): if data do not include 0s, expolate 0s by using first time set.(Defaults to True)

        Return:
            None. 
            Define below variable.
            self.Ib        # TES base current[A]         
            
            Redefined below variable.
            self.t_sim     # Simulation time [s]
            self.Tsor_sim  # Source temperature [K]
            self.Tabs_sim  # Absorber temperature [K]
            self.Tpath_sim # Au path temperature [K]
            self.Ttes_sim  # TES temperature [K]
            self.Tmem_sim  # Membrane temperature [K]
            self.I_cur     # TES Current calculated by electric circuit module [A]   
        """
        time_mask = (limit <= self.t_sim)
        self.t_sim     = self.t_sim[time_mask]       # Simulation time [s]
        self.Tsor_sim  = self.Tsor_sim[time_mask]    # Source temperature [K]
        self.Tabs_sim  = self.Tabs_sim[time_mask]    # Absorber temperature [K]
        self.Tpath_sim = self.Tpath_sim[time_mask]  # Au path temperature [K]
        self.Ttes_sim  = self.Ttes_sim[time_mask]    # TES temperature [K]
        self.Tmem_sim  = self.Tmem_sim[time_mask]    # Membrane temperature [K]
        self.I_cur     = self.I_cur[time_mask]       # TES Current calculated by electric circuit module [A]   
        self.Ib        = np.median(self.I_cur[:50])  # TES base current[A]
        self.t_sim    -= limit
        self.I_sim     = np.median(self.I_cur[:50]) - self.I_cur
        self.Ie        = self.I_sim[-1]
        if exporate_zero:
            if self.t_sim[0] != 0:
                self.t_sim[0] = 0
                print('Zero sec was Exporated')
        if self.debug:   
            print(self.t_sim)

    def calc_interp_for_risetime(self,max=0.8,min=0.2):
        Imax = np.max(self.I_sim)
        self.pulse_h = Imax
        Imax_id = np.argmax(self.I_sim)
        Irise = self.I_sim[:Imax_id]
        trise = self.t_sim[:Imax_id]
        frise = interpolate.interp1d(Irise,trise)
        trise_max = frise(Imax*max)
        trise_min = frise(Imax*min)
        self.tau_r = trise_max - trise_min
        Ifall = self.I_sim[Imax_id:]
        tfall = self.t_sim[Imax_id:]
        ffall = interpolate.interp1d(Ifall,tfall)
        tfall_max = ffall(Imax*max)
        tfall_min = ffall(Imax*min)
        self.tau_f = tfall_min - tfall_max
        print('--------------------------------------')
        print(rf'{max*100} $\%$ - {min*100} $\%$')
        print(f'Rise time = {self.tau_r*1e+6}')
        print(f'Fall time = {self.tau_f*1e+6}')
        print(f'PH = {Imax*1e+6}uA')

    def calc_interp_for_risetime_T(self,T,max=0.8,min=0.2):
        Imax = np.max(T)
        self.pulse_h = Imax
        Imax_id = np.argmax(T)
        Irise = T[:Imax_id]
        trise = self.t_sim[:Imax_id]
        frise = interpolate.interp1d(Irise,trise)
        trise_max = frise(Imax*max)
        trise_min = frise(Imax*min)
        self.tau_r = trise_max - trise_min
        Ifall = T[Imax_id:]
        tfall = self.t_sim[Imax_id:]
        ffall = interpolate.interp1d(Ifall,tfall)
        tfall_max = ffall(Imax*max)
        tfall_min = ffall(Imax*min)
        self.tau_f = tfall_min - tfall_max
        print('--------------------------------------')
        print(rf'{max*100} $\%$ - {min*100} $\%$')
        print(f'Rise time = {self.tau_r*1e+6}')
        print(f'Fall time = {self.tau_f*1e+6}')

    def sampling(self,sampling_rate:float,max_time=5e-3):
        sampling_rate = sampling_rate
        print(f'Max data time = {max_time*1e+3} msec')
        self.rec_time = np.arange(0,max_time+sampling_rate/2,sampling_rate)
        print(self.rec_time)        
        self.I_interp_func = interpolate.interp1d(self.t_sim,self.I_cur,kind='linear')
        self.I_interp = self.I_interp_func(self.rec_time)
        if self.debug:
            plt.scatter(self.t_sim,self.I_cur,color='Blue',label='raw data')
            plt.scatter(self.rec_time,self.I_interp,color='Red',label='interp data')
            plt.legend(loc='best')
            plt.show()

    def save_data(self, name:str, att:str, number:str, multi:bool=False):
        with h5py.File(self.savehdf5,"a") as f:
            if multi:
                if name in f:
                    if att in f[f'{name}']:
                        if 'analysis' in f[f'{name}/{att}']:
                            if 'trise' in f[f'{name}/{att}/analysis']:
                                del f[f'{name}'][att]['analysis']['trise']
                                del f[f'{name}'][att]['analysis']['tfall']
                                del f[f'{name}'][att]['analysis']['ph']
                                del f[f'{name}'][att]['analysis']['Ib']
                                del f[f'{name}'][att]['analysis']['Ie']
                f.create_dataset(f'{name}/{att}/analysis/trise',data=self.trise)
                f.create_dataset(f'{name}/{att}/analysis/tfall',data=self.tfall)
                f.create_dataset(f'{name}/{att}/analysis/ph',data=self.ph)
                f.create_dataset(f'{name}/{att}/analysis/Ib',data=self.Ib_list)
                f.create_dataset(f'{name}/{att}/analysis/Ie',data=self.Ie_list)

            else:
                if name in f:
                    if att in f[f'{name}']:
                        if 'data' in f[f'{name}/{att}']:
                            if number in f[f'{name}/{att}/data']:
                                del f[f'{name}'][att]['data'][f'{number}']
                        if 'analysis' in f[f'{name}/{att}']:
                            if 'template_pulse' in f[f'{name}/{att}/analysis']:
                                if number in  f[f'{name}/{att}/analysis/template_pulse']:
                                    del f[f'{name}/{att}/analysis/template_pulse/{number}/time']
                                    del f[f'{name}/{att}/analysis/template_pulse/{number}/pulse']
                f.create_dataset(f'{name}/{att}/analysis/template_pulse/{number}/time',data=self.rec_time)
                f.create_dataset(f'{name}/{att}/analysis/template_pulse/{number}/pulse',data=self.I_interp)
                f.create_dataset(f'{name}/{att}/data/{number}/Tsor',data=self.Tsor_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Tabs',data=self.Tabs_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Tpath',data=self.Tpath_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Ttes',data=self.Ttes_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Tmem',data=self.Tmem_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Ites',data=self.I_cur)
                f.create_dataset(f'{name}/{att}/data/{number}/time',data=self.t_sim)
                f.create_dataset(f'{name}/{att}/data/{number}/Ibase',data=self.Ib)

    def template_processor(self,file,limit,sampling_rate,name,att,number,max_time):
        self.load_data(file)
        self.time_divider(limit)
        self.calc_interp_for_risetime()
        self.sampling(sampling_rate,max_time)
        self.save_data(name, att, number)

    def multi_template_processor(self,limit,sampling_rate,name,att,max_time):
        number_list = self.number_list
        file_list = self.file_list
        print(file_list)
        self.trise     = []
        self.tfall     = []
        self.ph        = []
        self.Ib_list   = []
        self.Ie_list   = []
        for file,number in zip(file_list,number_list):
            self.template_processor(file,limit,sampling_rate,name,att,number,max_time)
            self.trise.append(self.tau_r)
            self.tfall.append(self.tau_f)
            self.ph.append(self.pulse_h)
            self.Ib_list.append(self.Ib)
            self.Ie_list.append(self.Ie)
        self.save_data(name,att,number,multi=True)

    def load_template_pulse(self,number,name,att):
        with h5py.File(self.savehdf5,"r") as f:
            self.temp_time  = f[f'{name}/{att}/analysis/template_pulse/{number}/time'][:]
            self.temp_pulse = f[f'{name}/{att}/analysis/template_pulse/{number}/pulse'][:]

    def plot_multi_pulse(self):
        number_list = self.number_list
        for i in number_list:
            self.load_template_pulse(i)
            plt.plot(self.temp_time,self.temp_pulse)
        plt.show()

    def gen_white(self,dlen,std,mean):
        self.white = np.random.normal(mean,std,dlen)

        if self.debug:
            plt.plot(self.white)
            plt.show()

    def gen_single_pulse(self,number,std,name,att):
        self.load_template_pulse(number,name,att)
        dlen = len(self.temp_time)
        med = np.median(self.temp_pulse[0:50])
        self.gen_white(dlen,std,mean=0.)
        self.p = self.temp_pulse + self.white
        self.gen_white(dlen,std,med)
        self.n = self.white
        self.t = self.temp_time
        
        if self.debug:
            plt.plot(self.t, self.p)

    def gen_pulse(self,number,std,N,name,att):
        lowcut = False
        with h5py.File(self.savehdf5,"a") as f:
            if 'gen' in f[f'{name}/{att}']:
                if number in f[f'{name}/{att}/gen']:
                    del f[f'{name}/{att}/gen/{number}']
        print(f'Number {number} pulse processing')
        if lowcut == True:
            self.lowcut_buter(lowcut=1e+6,degree=3,fs=5e+6)
        for i in range(0,N):
            #print(f'Count {i+1} generating...',end='',flush=True)
            self.gen_single_pulse(number,std,name,att)
            if lowcut == True:
                self.p = signal.filtfilt(self.b, self.a, self.p)
                self.n = signal.filtfilt(self.b, self.a, self.n)
            with h5py.File(self.savehdf5,"a") as f:
                if i == 0:
                    f.create_dataset(f'{name}/{att}/gen/{number}/time',data=self.t,dtype=np.float32)
                    #self.pulse_plot()
                f.create_dataset(f'{name}/{att}/gen/{number}/{i}/pulse',data=self.p,dtype=np.float32)
                f.create_dataset(f'{name}/{att}/gen/{number}/{i}/noise',data=self.n,dtype=np.float32)
            del self.t, self.p, self.n

    def pulse_plot(self):
        self.P
        self.fig  = plt.figure(figsize=(8,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r'$\rm Time \ [ms]$',fontsize=20)
        self.ax.set_ylabel(r'$\rm Current \ [\mu A]$',fontsize=20)
        t = np.hstack((self.t,self.t+5e-3))
        pn = np.hstack((self.n,self.p))
        self.ax.plot(t*1e+3, pn*1e+6, color='blue')
        self.fig.savefig('gen_pulse.png',dpi=300)
        plt.show()
  
    def multi_gen_pulse(self,std,N,name,att):
        number_list = self.number_list
        for number in number_list:
            if number in self.single_num:
                print('single number')
                Num = N
            else:
                Num = N*2
                print('double number')
            self.gen_pulse(number=number,std=std,N=Num,name=name,att=att)

    def store_data(self,N,name,att):
        number_list = self.number_list
        with h5py.File(self.savehdf5,"a") as f:
            sel = number_list[0]
            print(f"store {N} data")
            dlen = len(f[f'{name}/{att}/gen/{sel}/time'][:])
            nlen = len(number_list)
            self.pulse = []
            self.noise = []
            counter = 0
            for number in number_list:
                if N == None:
                    Num = len(f[f'{name}/{att}/gen/{number}'].keys())-1
                else:
                    if number in self.single_num:
                        Num = N
                    else:
                        Num = 2*N
                counter += Num
                for i in range(0,Num):
                    self.t = f[f'{name}/{att}/gen/{number}/time'][:]
                    self.pulse.append(f[f'{name}/{att}/gen/{number}/{i}/pulse'][:])
                    self.noise.append(f[f'{name}/{att}/gen/{number}/{i}/noise'][:])
            self.pulse = np.array(self.pulse)
            self.pulse.reshape((counter,dlen))
            self.noise = np.array(self.noise)
            self.noise.reshape((counter,dlen))
            # for i in range(0,len(self.pulse)-1):
            #     plt.plot(self.pulse)
            # plt.show()

    def gen_ofs_ph(self):
        self.ofs = np.median(self.noise,axis=1)
        self.ph = self.ofs - np.min(self.pulse,axis=1)

    def gen_pha(self,N,name,att):
        self.store_data(N,name,att)
        self.gen_ofs_ph()
        print('calc pulse')
        p = self.pulse - self.ofs.reshape((len(self.pulse),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.noise,max_shift=100)
        print("Template generated!")
        self.store_data(None,name,att)
        self.gen_ofs_ph()
        plt.hist(self.ph,bins=200,histtype='step')
        plt.show()
        self.avg = Filter.average_pulse(p,max_shift=100)   
        self.pha, self.ps = Filter.optimal_filter(self.pulse, self.tmpl,max_shift=100)
        self.power_spectruum = Filter.power(self.avg)
        self.pha = self.pha*1e+6
        with h5py.File(self.savehdf5,"a") as f:
            if 'analysis' in f[f'{name}/{att}']:
                if 'pha' in f[f'{name}/{att}/analysis']:
                    del f[f'{name}/{att}/analysis/pha']
                    del f[f'{name}/{att}/analysis/tmpl']
                    del f[f'{name}/{att}/analysis/sn']
                    del f[f'{name}/{att}/analysis/ps']
            f.create_dataset(f'{name}/{att}/analysis/pha',data=self.pha)
            f.create_dataset(f'{name}/{att}/analysis/tmpl',data=self.tmpl)
            f.create_dataset(f'{name}/{att}/analysis/sn',data=self.sn)
            f.create_dataset(f'{name}/{att}/analysis/ps',data=self.ps)
        plt.hist(self.pha,bins=200,histtype='step')
        plt.show()
        self.plot_tmpl(name,att)

    def gen_baseline(self):
        n = self.noise - self.ofs.reshape((len(self.noise),1))
        
    def plot_tmpl(self,name,att):
        self.plot_window()
        self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Current \ [\mu A]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Template \ [\mu A]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm PS $",fontsize=20)
        self.ax3.set_ylabel(r"$\rm PS $",fontsize=20)
        self.ax4.set_xlabel(r"$\rm SN $",fontsize=20)
        self.ax4.set_ylabel(r"$\rm SN $",fontsize=20)
        # with h5py.File(self.savehdf5,"r") as f:
        #     p = f[f'{name}/{att}/gen/1/10/pulse'][:]
        #     n = f[f'{name}/{att}/gen/1/10/noise'][:]
        #     t = f[f'{name}/{att}/gen/1/time'][:]
        # pt = np.hstack((t,t+5e-3))
        # pn = np.hstack((n,p))
        self.ax1.plot(self.avg*1e+6,color='blue')
        self.ax2.plot(self.tmpl,color='blue')
        self.ax3.plot(self.power_spectruum,color='blue')
        self.ax4.plot(self.sn,color='blue')
        #plt.legend()
        self.ax3.set_xscale('log')
        self.ax3.set_yscale('log')
        self.ax4.set_xscale('log')
        self.ax4.set_yscale('log')
        self.fig.tight_layout()
        self.fig.savefig('tmpl.png',dpi=300)
        plt.show()        

    def pha_median(self,N,name,att):
        num = self.number_list
        counter = 1
        self.pha_med = []
        dlen = len(num)
        for i in range(1,dlen+1):
            if str(i) in self.single_num:
                Num = N
            else:
                Num = N*2
            self.pha_med.append(np.median(self.pha[counter:counter+Num]))
            counter += Num
        with h5py.File(self.savehdf5,"a") as f:
            if 'pha_median' in f[f'{name}/{att}/analysis']:
                del f[f'{name}/{att}/analysis/pha_median']
            f.create_dataset(f'{name}/{att}/analysis/pha_median',data=self.pha_med)

    def plot_pha_rel(self,name,att):
        with h5py.File(self.savehdf5,"r") as f:     
            trise     = f[f'{name}/{att}/analysis/trise'][:]
            tfall     = f[f'{name}/{att}/analysis/tfall'][:]
            ph        = f[f'{name}/{att}/analysis/ph'][:]
            Ib        = f[f'{name}/{att}/analysis/Ib'][:]
            Ie        = f[f'{name}/{att}/analysis/Ie'][:]
            pha_med   = f[f'{name}/{att}/analysis/pha_median'][:]
        self.plot_window()
        self.ax1.set_xlabel(r"$\rm PHA $",fontsize=20)
        self.ax1.set_ylabel(r"$\rm Pulse \ height \ [\mu A]$",fontsize=20)
        self.ax2.set_xlabel(r"$\rm PHA $",fontsize=20)
        self.ax2.set_ylabel(r"$\rm Rise \ time \ [\mu sec]$",fontsize=20)
        self.ax3.set_xlabel(r"$\rm PHA $",fontsize=20)
        self.ax3.set_ylabel(r"$\rm Fall \ time \ [\mu sec]$",fontsize=20)
        self.ax4.set_xlabel(r"$\rm PHA $",fontsize=20)
        self.ax4.set_ylabel(r"$\rm Distance [\mu m]$",fontsize=20)
        for i in range(0,len(pha_med)):
            self.ax1.scatter(pha_med[i],ph[i]*1e+6,color=self.col_list[i])
            self.ax2.scatter(pha_med[i],trise[i]*1e+6,color=self.col_list[i])
            self.ax3.scatter(pha_med[i],tfall[i]*1e+6,color=self.col_list[i])
            self.ax4.scatter(pha_med[i],self.dist[i],color=self.col_list[i])
        #plt.legend()
        self.fig.tight_layout()
        self.fig.savefig(name,dpi=300)
        num_lis = np.array(self.number_list)
        print(num_lis[np.argsort(pha_med)])
        plt.show()

    def load_pha(self,name,att):
        with h5py.File(self.savehdf5,"r") as f:
            self.pha = f[f'{name}/{att}/analysis/pha'][:]

    def resolution_fit(self,name,att):
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        if self.device == 'Axion' or self.device == 'Axion_sp':
            self.pha = 14.4 * self.pha / np.median(self.pha[1:])
        else:
            self.pha = 5.9 * self.pha / np.median(self.pha[1:])
        hist, bins = np.histogram(self.pha[1:],bins=100)
        print(len(self.pha))
        bins_mid = bins[:-1] + np.diff(bins)
        pcov, popt = curve_fit(self.gaus,bins_mid,hist,p0=[1,1e-3,np.median(self.pha[1:])])
        print(pcov,popt)
        print(pcov[1])
        res = np.abs(2*pcov[1]*np.sqrt(2*np.log(2))*1e+3)
        print(f'Energy resolution = {np.abs(res)} eV (FWHM)')
        self.ax.step(bins_mid,hist,where='mid',color='black',label='data')
        self.ax.plot(bins_mid,self.gaus(bins_mid,*pcov),color='Red',label='fit')
        self.ax.set_xlabel(r'$\rm Energy \ [keV]$',fontsize=20)
        self.ax.set_ylabel(r'$\rm Counts $',fontsize=20)
        res_2f = format(res, '.2f')
        txt = rf'$\rm \Delta E \ = \ {res_2f} \ eV$'
        self.ax.text(0.025,0.9,txt, transform=self.ax.transAxes,fontsize=20,backgroundcolor='white')
        plt.legend(loc='best',fontsize=20)
        self.fig.savefig('resolution.png',dpi=300)
        baseE = Analysis.baseline(self.sn,14.4e+3)
        print(f'base line resolution = {baseE}')
        with h5py.File(self.savehdf5,"a") as f:
            if 'delE' in f[f'{name}/{att}/analysis']:
                del f[f'{name}/{att}/analysis/delE'], f[f'{name}/{att}/analysis/baseE']
            f.create_dataset(f'{name}/{att}/analysis/delE', data=res)
            f.create_dataset(f'{name}/{att}/analysis/baseE', data=baseE)
        plt.show()

    def resolution_plot(self):
        res = [3.68,3.61,3.42]
        par = [10,50,100]
        res = [3.33,3.36,3.42]
        par = [10,50,125]
        res = [4.27,3.42,3.46]
        par = [0.2,2,5]
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.scatter(par,res,color="black",s=30)
        self.ax.scatter(par[1],res[1],marker="*",color="black",s=300)
        self.ax.plot(par,res,color="black")        
        self.ax.grid(linestyle="dashed")
        self.ax.hlines(3.0,par[0],par[2],colors="Red",linestyles='dashed')
        #self.ax.hlines(10,par[0],par[2],colors="Blue",linestyles='dashed')
        self.ax.set_xlabel(r"$\rm Thickness \ [um] $",fontsize=20)
        self.ax.set_ylabel(r"$\rm Energy \ resolution \ [eV] $",fontsize=20)
        self.fig.savefig("thickness_res.png",dpi=300)                  
        plt.show()

    def pulse_reset(self,name,att):
        with h5py.File(self.savehdf5,"a") as f:
            del f[f'{name}/{att}/gen']
        print('pulse deleted...')

    def plot_hist_pro(self,N):
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        binnum = 100
        num = self.number_list
        counter = 0
        dlen = len(num)
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        self.pha = 14.4 * self.pha / np.median(self.pha[1:])
        for i in range(1,dlen+1):
            if str(i) in self.single_num:
                Num = N
            else:
                Num = N*2
            
            if str(i) in ['11']:
                #col = 'navy'
                l1 = np.hstack((l1,self.pha[counter:counter+Num]))

            if str(i) in ['2','7','12']:
                #col = 'blue'
                l2 = np.hstack((l2,self.pha[counter:counter+Num]))
            if str(i) in ['13']:
                #col = 'green'
                l3 = np.hstack((l3,self.pha[counter:counter+Num]))
            
            if str(i) in ['4','9','14']:
                #col = 'orange'
                l4= np.hstack((l4,self.pha[counter:counter+Num]))
            
            if str(i) in ['15']:
                #col = 'red'
                l5 = np.hstack((l5,self.pha[counter:counter+Num]))

            counter += Num
        print(l1)
        self.ax.hist(self.pha[4001:6000],bins=200,histtype="step",color='red',label='position 1')
        self.ax.hist(self.pha[2001:4000],bins=200,histtype="step",color='green',label='position 2')
        self.ax.hist(self.pha[1:2000],bins=200,histtype="step",color='blue',label='position 3')
        all = np.hstack((l1,l3))
        all = np.hstack((all,l5))
        #self.ax.hist(l2,bins=50,histtype="step",color='blue')
        #self.ax.hist(l4,bins=50,histtype="step",color='orange')
        num = np.array(num)
        #print(num[np.argsort(self.pha_med)])
        self.ax.hist(self.pha[1:],bins=100,histtype='step',color="black")
        self.ax.set_xlabel(r'$\rm Energy \ [keV]$',fontsize=20)
        self.ax.set_ylabel('Counts',fontsize=20)
        plt.legend(fontsize=15)
        self.fig.savefig('hist.png',dpi=300)
        plt.show()

    def plot_hist(self,N):
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        binnum = 100
        num = self.number_list
        counter = 0
        dlen = len(num)
        self.pha = 14.4 * self.pha / np.median(self.pha[1:])
        for e,i in enumerate(self.number_list):
            if str(i) in self.single_num:
                Num = N
            else:
                Num = N*2
            
            self.ax.hist(self.pha[counter+1:counter+Num],bins=200,histtype="step",color=cm.jet([(e)/dlen]),label=i)
            counter += Num

        num = np.array(num)
        #print(num[np.argsort(self.pha_med)])
        self.ax.hist(self.pha[1:],bins=100,histtype='step',color="black")
        self.ax.set_xlabel(r'$\rm Energy \ [keV]$',fontsize=20)
        self.ax.set_ylabel('Counts',fontsize=20)
        plt.legend(fontsize=15)
        self.fig.savefig('hist.png',dpi=300)
        plt.show()

    def plot_hist2(self,N):
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        binnum = 100
        num = self.number_list
        counter = 0
        dlen = len(num)
        self.pha = 14.4 * self.pha / np.median(self.pha[1:])
        for e,i in enumerate(self.number_list):
            if i in self.single_num:
                Num = N
            else:
                Num = N*2
            
            self.ax.hist(self.pha[counter+1:counter+Num],bins=100,histtype="step",color=cm.jet([(e)/dlen]),label=i)
            counter += Num

        num = np.array(num)
        #print(num[np.argsort(self.pha_med)])
        self.ax.hist(self.pha[1:],bins=100,histtype='step',color="black")
        self.ax.set_xlabel(r'$\rm Energy \ [keV]$',fontsize=20)
        self.ax.set_ylabel('Counts',fontsize=20)
        plt.legend(fontsize=15)
        self.fig.savefig('hist.png',dpi=300)
        plt.show()

    def single_process(self,sampling_rate,std,N,name,att,max_time):
        limit = 1e-3
        self.number_list = ['13']
        #self.plot_pulse(name,att)
        self.multi_template_processor(limit,sampling_rate,name,att,max_time)
        self.multi_gen_pulse(std,N,name,att)
        self.gen_pha(100,name,att)
        # self.plot_hist(N)
        # self.pha_median(N,name,att)
        # self.plot_pha_rel(name,att)
        self.resolution_fit(name,att)

    def all_process(self,sampling_rate,std,N,name,att,max_time):
        limit = 1e-3
        self.plot_pulse(name,att)
        self.multi_template_processor(limit,sampling_rate,name,att,max_time)
        self.multi_gen_pulse(std,N,name,att)
        self.gen_pha(100,name,att)
        self.pha_median(N,name,att)
        self.plot_hist(N)
        self.plot_pha_rel(name,att)
        self.resolution_fit(name,att)
        self.pulse_reset(name,att)


class PulseProcessor(Base):

    def __init__(self):
        pass

class CAnalysis:
    '''
    Analysis of the COMSOL data
    '''
    def __init__(self):
        self.kb        = 1.381e-23 ##[J/K]
        self.Rshunt    = 3.9 * 1e-3
        self.Min       = 9.82 * 1e-11
        self.Rfb       = 1e+5
        self.Mfb       = 8.5 * 1e-11
        self.JtoeV     = 1/1.60217662e-19
        self.excVtoI   = self.Mfb / (self.Min * self.Rfb)
        self.filelist  =  sorted(glob.glob(f"*.txt"))
        self.P = Plotter()
        self.savehdf5  = 'test.hdf5'
        dask.config.set({'array.chunk-size': '256MiB'})

    def pfunc(self,t,tau_r,tau_f,A):
        t0=1.025e-3
        return 0*(t<=t0)+A*(np.exp(-(t-t0)/tau_r)-np.exp(-(t-t0)/tau_f))*(t0<t)

    def RLFilter(self,t,R,L,t0):
        self.cf = 1-np.exp(-R*(t-t0)/L)

    def search_t0(self,t0=1.025e-3):
        abs_value = np.abs(self.t_sim-t0)
        self.idx = np.argmax(self.I_sim)

    def risefilter(self,t, R, L):

        def cutoff(t=t, R=R, L=L):
            return 1-np.exp(-R*t/L)

        self.filter_cutoff = np.hstack((np.zeros(int(self.peaks[0])),cutoff(t=t-t[self.peaks[0]+1])[self.peaks[0]:]))

    def moving_average(self,x,num):
        b = np.ones(num)/num
        conv = np.convolve(x,b,mode="same")
        return conv

    def number_to_running_average_cutoff(self,fs,n):
        print(f"Running average cut off frequency : {0.443*fs/np.sqrt(n**2 - 1)}")

    def running_average_cutoff_to_number(self,fc,fs):
        print(f"Running average cut off frequency : {np.sqrt(0.443*fs/fc)**2-1}")


    @dask.delayed
    def moving_average_dask(self,data,num):
        return bn.move_mean(data,window=num,min_count=1)

    @dask.delayed
    def generate_interpolate_pulse_dask(self,t):
        pulse_tfunc = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        return pulse_tfunc(t)

    def interp(self):
        self.peaks, _ = find_peaks(np.diff(self.I_sim), height=0.5e-6)
        self.risefilter(t=self.t_sim,R=(14.025+3.9)*1e-3,L=12e-9)
        self.I_sim = self.I_sim*self.filter_cutoff
        self.t_sim = self.t_sim

    def f_time(self,per):
        ph_f     = self.pulse_h*per/100
        tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        print(rf"Fall Time ({per}%) = {tau_f*1e+6} us")

    def r_time(self,per):
        ph_f     = self.pulse_h*per/100
        tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        print(rf"Fall Time ({per}%) = {tau_f*1e+6} us")

    def load_data(self,file):
        f = np.genfromtxt(f"../data/{file}/{file}.txt")
        self.t_sim    = f[:,0]*1e-3  # Simulation time [s]
        self.Ttes_sim = f[:,1]*1e-3  # TES temperature [K]
        self.Tabs_sim = f[:,2]*1e-3  # Absorber temperature [K]
        self.Tsor_sim = f[:,3]*1e-3  # Source temperature [K]
        self.I_cur    = f[:,4]*1e-6  # TES Current calculated by current module [A]
        self.V_sim    = f[:,5]*1e-6  # TES voltage [V]
        self.Tmem_sim = f[:,6]*1e-3  # Membrane temperature [K]
        self.I_am     = -f[:,7]*1e-6 # TES current calculated by electric circuit module [A]

    def load_data_Tdep(self,file):
        f = np.genfromtxt(f"../data/{file}/{file}.txt")
        self.t_sim    = f[:,0]*1e-3  # Simulation time [s]
        self.Ttes_sim = f[:,1]*1e-3  # TES temperature [K]
        self.Tabs_sim = f[:,2]*1e-3  # Absorber temperature [K]
        self.Tsor_sim = f[:,3]*1e-3  # Source temperature [K]
        self.I_cur    = f[:,4]*1e-6  # TES Current calculated by electric circuit module [A]

    def load_data_axion(self,file):
        f = np.genfromtxt(f"../data/{file}/{file}.txt")
        self.t_sim    = f[:,0]*1e-3  # Simulation time [s]
        self.Tsor_sim = f[:,1]*1e-3  # Source temperature [K]
        self.Tabs_sim = f[:,2]*1e-3  # Absorber temperature [K]
        self.Tpath_sim = f[:,3]*1e-3  # Au path temperature [K]
        self.Ttes_sim = f[:,4]*1e-3  # TES temperature [K]
        self.Tmem_sim = f[:,5]*1e-3  # Membrane temperature [K]
        self.I_cur    = f[:,6]*1e-6  # TES Current calculated by electric circuit module [A]

    def load_data_from_hdf5(self,file,runnum):
        with h5py.File(file,"r") as f: 
            self.I_sim = f[f"{runnum}/interp_pulse"][1:]
            self.t_sim = f[f"{runnum}/interp_time"][:]

    def offset_correction(self):
        self.Tb       = self.Ttes_sim[100]
        self.I_cur   -= np.average(self.I_cur[:100])
        self.Ib       = self.I_am[100]  # TES base current
        self.I_am    -= self.I_am[100]
        self.I_sim    = self.I_am

    def offset_correction_Tdep(self):
        self.Tb       = self.Ttes_sim[50]
        self.Ib       = self.I_cur[50]  # TES base current
        self.I_cur   -= self.I_cur[50]
        self.I_sim    = -self.I_cur

    def show_properity(self):
        self.t_sim_min    = np.min(np.diff(self.t_sim))
        if self.t_sim_min == 0:
            self.t_sim_min = np.min(np.diff(self.t_sim)[np.diff(self.t_sim)!=0])
        self.pulse_h  = np.max(self.I_sim)
        self.ph_f     = self.pulse_h/np.exp(1)
        self.p_s      = self.t_sim[np.argmax(self.I_sim)]
        self.tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(self.ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        self.tau_r    = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-1.025e-3
        self.Ttes_max_time = self.t_sim[np.argmax(self.Ttes_sim)]-1.025e-3
        print(f"Pulse Height(COMSOL data)     = {self.pulse_h*1e+6} uA")
        print(f"Temperature Height(Ttes data) = {np.max(self.Ttes_sim)*1e+3} mK")
        print(f"Temperature Height(Tabs data) = {np.max(self.Tabs_sim)*1e+3} mK")
        print(f"Temperature Height(Tsor data) = {np.max(self.Tsor_sim)*1e+3} mK")
        print(f"Base Temperature = {self.Tb*1e+3} mK")
        print(f"Base Current = {self.Ib*1e+6} uA")
        print(f"Fall Time = {self.tau_f*1e+6} us")
        print(f"Rise Time = {self.tau_r*1e+6} us")
        print(f'Ttes_max_time = {self.Ttes*1e+6} us')
        print(f"Minimum of time difference = {self.t_sim_min} s")
        self.f_time(per=90)
        self.f_time(per=80)

    def show_properity_hdf5(self):
        self.pulse_h  = np.max(self.I_sim)
        self.ph_f     = self.pulse_h/np.exp(1)
        self.ph_ten   = self.pulse_h/10
        self.p_s      = self.t_sim[np.argmax(self.I_sim)]
        self.tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(self.ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        self.tau_r    = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-1.025e-3
        self.tau_r_ten = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_ten-self.I_sim[:np.argmax(self.I_sim)]))]
        print(f"Pulse Height(COMSOL data)     = {self.pulse_h*1e+6} uA")
        print(f"Fall Time = {self.tau_f*1e+6} us")
        print(f"Rise Time = {self.tau_r*1e+6} us")
        print(f"Rise Time from ten percent = {self.tau_r_ten*1e+6} us")
        self.P.plotting(self.t_sim,self.I_sim,scatter=True)
        self.P.plotting(self.tau_r+1.025e-3,self.ph_f,scatter=True,new_window=False)
        self.P.plotting(self.tau_f+self.p_s,self.ph_f,scatter=True,new_window=False)

    def show_properity_Tdep(self):
        self.t_sim_min    = np.min(np.diff(self.t_sim))
        if self.t_sim_min == 0:
            self.t_sim_min = np.min(np.diff(self.t_sim)[np.diff(self.t_sim)!=0])
        self.pulse_h  = np.max(self.I_sim)
        self.ph_f     = self.pulse_h/np.exp(1)
        self.ph_ten   = self.pulse_h/10
        self.p_s      = self.t_sim[np.argmax(self.I_sim)]
        self.tau_f    = self.t_sim[np.argmax(self.I_sim):][np.argmin(np.abs(self.ph_f-self.I_sim[np.argmax(self.I_sim):]))]-self.p_s
        self.tau_r    = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-1.025e-3
        self.tau_r_ten = self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_f-self.I_sim[:np.argmax(self.I_sim)]))]-self.t_sim[:np.argmax(self.I_sim)][np.argmin(np.abs(self.ph_ten-self.I_sim[:np.argmax(self.I_sim)]))]
        print(f"Pulse Height(COMSOL data)     = {self.pulse_h*1e+6} uA")
        print(f"Temperature Height(Ttes data) = {np.max(self.Ttes_sim)*1e+3} mK")
        print(f"Temperature Height(Tabs data) = {np.max(self.Tabs_sim)*1e+3} mK")
        print(f"Temperature Height(Tsor data) = {np.max(self.Tsor_sim)*1e+3} mK")
        print(f"Base Temperature = {self.Tb*1e+3} mK")
        print(f"Base Current = {self.Ib*1e+6} uA")
        print(f"Fall Time = {self.tau_f*1e+6} us")
        print(f"Rise Time = {self.tau_r*1e+6} us")
        print(f"Minimum of time difference = {self.t_sim_min} s")
        self.f_time(per=90)
        self.f_time(per=80)

    def load_data_with_instant_analysis(self,file):
        self.load_data(file)
        self.offset_correction()
        self.show_properity()

    def interp_and_moving_filter_dask(self,file,t,n,runnum):
        start = time.time()
        self.load_data_with_instant_analysis(file)
        interp_time = da.arange(0,2e-3+t,t)
        interp_pulse = self.generate_interpolate_pulse_dask(interp_time)
        avg_pulse = self.moving_average_dask(interp_pulse,n)
        result = avg_pulse.compute()
        res_array = np.array(result) 
        res_time  = np.arange(0,2e-3,2e-3/len(res_array))[1:]
        print(time.time() - start)
        with h5py.File('test.hdf5',"a") as f:
            if runnum in f.keys():
                del f[f"{runnum}"] 
            f.create_dataset(f"{runnum}/interp_pulse",data=res_array[:])
            f.create_dataset(f"{runnum}/interp_time",data=res_time[:])
            self.res_array = res_array
            self.res_time = res_time

    def pulse_fit(self,subject):
        if subject == "data":
            t = self.t
            p = self.p
        elif subject == "sim":
            t = self.t_sim
            p = self.I_sim
        p0 = np.array([1.6e-8,1e-5,np.min(p)]) 
        self.popt, pcov = curve_fit(self.pulse_func,t,p,p0=p0)
        #self.t_sim[np.argmin(self.I_sim)]
        print(self.popt,pcov)
        self.tau_r = self.popt[0]
        self.tau_f = self.popt[1]
        self.t0 = self.t_sim[np.argmin(self.I_sim)]
        print("-----------------------------------------------")
        print("Pulse fit result")
        print(f"Rise time = {self.tau_r}")
        print(f"Fall time = {self.tau_f}")

        self.t_fit = np.hstack((np.linspace(0,self.t0,100),np.linspace(self.t0,np.max(t),1000)))
        self.result_plot(subject="pulse_fit")

    # def rise_fall_fit_s(self):
    #     self.I_sim = self.I_sim
    #     self.t_sim = self.t_sim
    #     p0 = np.array([self.t_sim[np.argmin(self.I_sim)],1.3e-6,130e-6,-np.max(self.I_sim)])
    #     plt.plot(self.t_sim,self.I_sim,".")
    #     self.popt, pcov = curve_fit(self.pfunc,self.t_sim,self.I_sim,p0=p0)
    #     self.tau_r = self.popt[1]
    #     self.tau_f = self.popt[2]
    #     print(self.popt,pcov)
    #     plt.plot(self.t_sim,self.pfunc(self.t_sim,*self.popt))
    #     print(self.tau_r)
    #     print(self.tau_f)
    #     plt.show()

    def pfit(self):
        self.model = Model(self.pfunc,nan_policy="omit")
        self.params = self.model.make_params() 
        self.model.set_param_hint('tau_r',min=0,max=1e-6)
        self.model.set_param_hint('tau_f',min=0,max=1000e-6)
        self.model.set_param_hint('t0',min=0,max=2e-3)
        self.model.set_param_hint('A',min=0,max=20e-6)
        result = self.model.fit(-self.I_sim[100:],t=self.t_sim[100:],tau_r=0.27e-6,tau_f=90e-6,t0=1.025e-3,A=np.max(self.I_sim))
        print(result.fit_report())
        self.fit_res = np.array([result.best_values["t0"],result.best_values["tau_r"],result.best_values["tau_f"],result.best_values["A"]])
        plt.plot(self.t_sim,-self.I_sim,".")
        plt.plot(self.t_sim,self.pfunc(self.t_sim,*self.fit_res))
        plt.show()

    def rise_fall_fit_s(self):
        self.p = -self.I_sim*1e+6
        self.t = self.t_sim
        p0 = np.array([1.3e-6,90e-6,-np.min(self.p)])
        self.popt, pcov = curve_fit(self.pfunc,self.t[self.t>1e-3],self.p[self.t>1e-3],p0=p0)
        self.tau_r = self.popt[0]
        self.tau_f = self.popt[1]
        print(self.popt,pcov)
        plt.plot(self.t,self.p)
        plt.plot(self.t,self.pfunc(self.t,*self.popt))
        plt.show()

    def plot_for_JSAP83(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.p_Idep = f["run003/interp_pulse"][:]
            self.t_Idep = f["run003/interp_time"][:]
            self.p_Tdep = f["run005/interp_pulse"][:]
            self.t_Tdep = f["run005/interp_time"][:]
            self.p_exp  = (f["pulse"][0] - f["pulse"][0][0])*self.excVtoI
            self.t_exp  = f["time"][:]
        self.P.plotting(self.t_exp*1e+3,self.p_exp*1e+6,color='Black',lw=3,label='Xray pulse')
        self.P.plotting(self.t_Idep[self.t_Idep>1e-3]*1e+3,-self.p_Idep[self.t_Idep>1e-3]*1e+6,color='Red',xname=rf'$\rm time \ (ms)$',yname=rf'$\rm Current \ (\mu A)$',new_window=False,lw=3,label=r'${\rm Simulation } \ \alpha \ {\rm and}\ \beta$')
        self.P.plotting((self.t_Tdep[self.t_Tdep>1e-3]+1e-3)*1e+3,-self.p_Tdep[self.t_Tdep>1e-3]*1e+6,new_window=False,color='Blue',lw=3,label=r'${\rm Simulation } \ \alpha$')

    def plot_Idep_vs_exp(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.p_Idep = f["run003/interp_pulse"][:]
            self.t_Idep = f["run003/interp_time"][:]
            self.p_Tdep = f["run005/interp_pulse"][:]
            self.t_Tdep = f["run005/interp_time"][:]
            self.p_exp  = (f["pulse"][0] - f["pulse"][0][0])*self.excVtoI
            self.t_exp  = f["time"][:]
        self.P.plotting(self.t_exp*1e+3,self.p_exp*1e+6,color='Blue',lw=3,label='Xray pulse',xname=rf'$\rm time \ (ms)$',yname=rf'$\rm Current \ (\mu A)$')
        self.P.plotting()
        #self.P.plotting((self.t_Idep-(self.t_Idep[np.argmax(self.p_Idep)]-self.t_exp[np.argmin(self.p_exp)]))*1e+3,self.p_Idep*1e+6,color='Red',xname=rf'$\rm time \ (ms)$',yname=rf'$\rm |Current| \ (\mu A)$',new_window=False,lw=3,label=r'${\rm Simulation } \ \alpha \ {\rm and}\ \beta$',ylog=True)

class Pulse:

    def __init__(self):
        pass

    def load_fits(self,file):
        self.tp,self.p = Util.fopen(f"{file}p.fits")
        #self.tn,self.n = Util.fopen(f"{file}n.fits")

    def save_fits(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "pulse_time" in f.keys():
                del f["pulse_time"]
            f.create_dataset("pulse_time",data=self.tp)

    def load_time(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.t = f["pulse_time"][:]     

    def pulse_cor(self):
        self.p = self.p[0]*self.excVtoI
        self.p -= self.p[0]
        print(f"Pulse Height(EXP data) = {np.min(self.p)}")

    def sim_exp_cor(self,file):
        self.load_fits(file="/Users/keitatanaka/Dropbox/work/microcalorimeters/COMSOL/before_analysis/run011_row")
        self.load_data(file=file)
        self.pulse_cor()
        self.result_plot(subject="sim_exp_cor")

    def pulse_fit(self,file):
        self.load_data(file=file)
        self.pfit()

    def temp_plot(self):
        self.load_time()
        self.plot_window(style="tT")
        l = len(self.filelist)
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=f"TES temperature")
            #self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")
        plt.show()

    def current_plot(self):
        self.load_time()
        self.plot_window(style="tI")
        l = len(self.filelist)
        for i in l:
            self.load_data(file=i)
            self.ax.plot(self.t_sim,self.I_sim*1e+6,label=i,c="blue",lw=3)
        # self.load_data(file="abs5um_Ctes_10.txt")
        # self.ax.plot(self.t_sim,self.I_sim*1e+6,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-12} \ [J/K]  $",c="Green",lw=3)
        #self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")
        plt.legend(loc="best",fontsize=25)
        plt.show()

class Simulator(Base):

    def __init__(self,debug=False) -> None:
        super().__init__(debug=debug)

    def tau(self,C,G):
        return C/G

    def tau_el(self,L,Rth,R,beta):
        return L/(Rth+R*(1+beta))

    def lp(self,Pb,alpha,G,T):
        return Pb*alpha/(G*T)

    def tau_I(self,C,Pb,alpha,G,T):
        return self.tau(C,G)/(1-self.lp(Pb,alpha,G,T))

    def risetime(self,L,Rth,R,beta,Pb,alpha,G,T,C):
        return 1/(1/(2*self.tau_el(L,Rth,R,beta))+1/(2*self.tau_I(C,Pb,alpha,G,T))+np.sqrt((1/self.tau_el(L,Rth,R,beta)-1/self.tau_I(C,Pb,alpha,G,T))**2-4*R*self.lp(Pb,alpha,G,T)*(2+beta)/(L*self.tau(C,G)))/2)

    def falltime(self,L,Rth,R,beta,Pb,alpha,G,T,C):
        return 1/(1/(2*self.tau_el(L,Rth,R,beta))+1/(2*self.tau_I(C,Pb,alpha,G,T))-np.sqrt((1/self.tau_el(L,Rth,R,beta)-1/self.tau_I(C,Pb,alpha,G,T))**2-4*R*self.lp(Pb,alpha,G,T)*(2+beta)/(L*self.tau(C,G)))/2) 

    def Ipulse(self,t,R,beta,Pb,alpha,G,T,C,dT,I,trise,tfall):
        return (self.tau_I(C,Pb,alpha,G,T)/trise-1)*(self.tau_I(C,Pb,alpha,G,T)/tfall-1)*C*dT*(np.exp(-t/trise)-np.exp(-t/tfall))/((2+beta)*I*R*self.tau_I(C,Pb,alpha,G,T)**2*(1/trise-1/tfall))

    def pfunc(self,t,tau_r,tau_f,A):
        return A*(np.exp(-t/tau_r)-np.exp(-t/tau_f))
    
    def pulse(self,t,tau_r,tau_f,A,t0):
        p = self.pfunc(t,tau_r,tau_f,A)
        p = np.zeros()

    def gen_theoretical_pulse(self, sampling_rate, tau_r, tau_f, A, t0, std, t, t_0, dlen, white_segment):
        p_0 = np.zeros(len(t_0))
        p = np.hstack((p_0, self.pfunc(t, tau_r, tau_f, A))) + white_segment[dlen:]
        n = white_segment[:dlen]
        full_t = np.hstack((t_0, t + t0))
        if self.debug:
            tt = np.hstack((full_t, full_t + 10e-3))
            pn = np.hstack((n, p))
            plt.plot(tt, pn)
            plt.show()
        return full_t, p, n

    def make_fractuate_pulse(self, sampling_rate, tau_r, tau_f, A, t0, std, N, frac_rise):
        t = np.arange(0, 10e-3, sampling_rate)
        t_0 = np.arange(0, t0, sampling_rate)
        dlen = len(t) + len(t_0)
        nlen = dlen * 2
        ph_sample = np.random.normal(A, frac_rise, size=N)
        plt.hist(ph_sample, bins=100, histtype="step")
        plt.show()
        # まとめてホワイトノイズ生成（N個分）
        white_noise_all = np.random.normal(0.0, std, size=(N, nlen))
        
        p_list = np.empty((N, dlen))
        n_list = np.empty((N, dlen))

        for i in range(N):
            white_segment = white_noise_all[i]
            full_t, p, n = self.gen_theoretical_pulse(sampling_rate, tau_r, tau_f, ph_sample[i], t0, std, t, t_0, dlen, white_segment)
            p_list[i] = p
            n_list[i] = n

        return full_t, p_list, n_list
    
    def test_gen(self):
        t, p, n = self.make_fractuate_pulse(1e-7,10e-6,100e-6,10e-6,1e-3,5e-8,1000,0.1e-8)
        self.pulse = p
        self.noise = n
        self.gen_ofs_ph()
        print('calc pulse')
        p = self.pulse - self.ofs.reshape((len(self.pulse),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.noise,max_shift=100)
        print("Template generated!")
        self.gen_ofs_ph()
        self.avg = Filter.average_pulse(p,max_shift=100)   
        self.pha, self.ps = Filter.optimal_filter(self.pulse, self.tmpl,max_shift=100)
        self.pha = self.pha/np.mean(self.pha) * 5900
        from Resonance_Scattering_Simulation import GeneralFunction
        G = GeneralFunction()
        G.gaussian_fitting_with_plot(self.pha)
        plt.show()



    def plot_pulse(self):
        t = np.arange(0,1e-3,1e-7)
        p = self.pfunc(t,1e-6,100e-6,10e-6)
        print(np.min(p))
        plt.plot(t,-p)
        p = self.pfunc(t,1e-6+1e-7,100e-6,10e-6)
        print(np.min(p))
        plt.plot(t,-p)
        plt.semilogy()
        plt.semilogx()
        plt.show()

class HeatSize:
    
    def __init__(self):
        self.filelist  =  sorted(glob.glob(f"*.txt"))

    def load_data(self,file):
        self.file = file
        f = np.loadtxt(file)
        self.t = f[:,0]*1e+3
        self.Tabs = f[:,1]
        self.Ttes = f[:,2]
        self.Tsou = f[:,3]
        print(self.t[np.argmax(self.Ttes)])

    def prop_data(self):
        print(f"Filename = {self.file}")
        print(f"Stationaly Temperatre (TES) = {self.Ttes[-1]} mK")
        print(f"Stationaly Temperatre (Abs) = {self.Tabs[-1]} mK")
        print(f"Stationaly Temperatre (Source) = {self.Tsou[-1]} mK")
        print(f"Maximum Source Temperature = {np.max(self.Tsou)} mK")

    def plot_init(self):
        #plt.subplots_adjust(wspace=15, hspace=12)
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
        self.fs = 40
        self.ps = 80

    def plot_window(self,style):
        self.plot_init()
        if style == "tT":
            self.xname = r"$\rm Time \ (\mu s)$"
            self.yname = r"$\rm Temperature \ (mK)$"
            self.fig = plt.figure(figsize=(9,7))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")

        if style == "rT":
            self.xname = r"$\rm Radius \ (\mu m)$"
            self.yname = r"$\rm Temperature \ (mK)$"
            self.fig = plt.figure(figsize=(9,7))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")

        self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
        self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

    def result_plot(self,subject):
        if subject == "tT":
            self.plot_window(style="tT")
            self.ax.plot(self.t,self.Tabs,lw=5,color="Blue")
            self.ax.plot(self.t,self.Ttes,lw=5,color="Red")
            self.ax.plot(self.t,self.Tsou,lw=5,color="Black")
            sfn = self.file + ".png"

        if subject == "Ttes_cor":
            self.plot_window(style="tT")
            for i in self.filelist:
                self.load_data(file=i)
                self.ax.plot(self.t,self.Ttes,lw=5,label=i)
            plt.legend()
            sfn = "Ttes_cor.png"

        if subject == "T_cor":
            self.plot_window(style="rT")
            Ta = 100 + 1.0503366126032503
            r = np.array([1e-3,1,2.5])
            T = np.array([101.05033614902173,101.05032998638724,101.05033383579259])
            self.ax.scatter(r,T-Ta,s=self.ps,color="Blue",label="Circle")
            self.ax.scatter(2.5,101.05033337303897-Ta,s=self.ps,color="Red",label="Cylinder")
            sfn = "heat_size.png"
            print(Ta-T)
            print(Ta-101.05033337303897)
            plt.legend()

        plt.show()
        self.fig.savefig(sfn,dpi=300)

class MakeHDF5:

    def __init__(self,savehdf5):
        self.savehdf5 = savehdf5
        self.open_hdf5 = h5py.File(self.savehdf5)

    def readdata(self):
        self.file_list = glob.glob('*.txt')
        
