import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import re
from scipy.stats import chi2
import glob
import lmfit as lf
from lmfit import Model
from Hdf5Treat import Hdf5Command
from scipy import integrate,signal
from scipy.optimize import curve_fit
from pytes import Util,Filter
from scipy import interpolate
from scipy.signal import find_peaks
from scipy import integrate
hd = Hdf5Command()

__author__ =  'Keita Tanaka, Tasuku Hayashi'
__version__=  '1.0.0' #2021.12.20

print('===============================================================================')
print(f"COMSOL Data Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class CAnalysis:

    def __init__(self):
        self.kb        = 1.381e-23 ##[J/K]
        self.Rshunt    = 3.9 * 1e-3
        self.Min       = 9.82 * 1e-11
        self.Rfb       = 1e+5
        self.Mfb       = 8.5 * 1e-11
        self.JtoeV     = 1/1.60217662e-19
        self.excVtoI   = self.Mfb / (self.Min * self.Rfb)
        self.savehdf5  = "/Users/tanakakeita/work/microcalorimeters/COMSOL/after_1d/TMU542_COMSOL.hdf5"
        self.filelist  =  sorted(glob.glob(f"*.txt"))

    def initialize(self):
        self.savehdf5 = "/Users/tanakakeita/work/microcalorimeters/COMSOL/after_1d/TMU542_COMSOL.hdf5"

    def pulse_func(self,t,tau_r,tau_f,A):
        t0 = self.t_sim[np.argmin(self.I_sim)]
        return A*(np.exp(-(t-t0)/tau_f)-np.exp(-(t-t0)/tau_r))

    def pfunc(self,t,t0,tau_r,tau_f,A):
        return 0*(t<=t0)+A*(np.exp(-(t-t0)/tau_r)-np.exp(-(t-t0)/tau_f))*(t0<t)

    def rise_func(self,t,tau_r,A):
        return A*np.exp(t/tau_r)

    def fall_func(self,t,tau_f,A):
        return A*np.exp(-t/tau_f)

    def cutoff(self,t,R,L,t0):
        self.cf = 1-np.exp(-R*(t-t0)/L)

    def search_t0(self,t0=1.025e-3):
        abs_value = np.abs(self.t_sim-t0)
        self.idx = np.argmax(self.I_sim)

    def risefilter(self,t, R, L):

        def cutoff(t=t, R=R, L=L):
            return 1-np.exp(-R*t/L)

        self.filter_cutoff = np.hstack((np.zeros(int(self.peaks)),cutoff(t=t-t[self.peaks+1])[self.peaks:]))

    def risefilter2(self,t, R, L, t0):

        def cutoff(t=t, R=R, L=L):
            return 1-np.exp(-R*t/L)

        self.filter_cutoff = np.hstack((np.zeros(int(self.idx)),cutoff(self.t_sim)[:-int(self.idx)]))

    def risefilter3(self,t, R, L):

        def cutoff(t=t, R=R, L=L):
            return 1-np.exp(-R*t/L)

        self.filter_cutoff = np.hstack((np.zeros(int(self.peaks)),cutoff(self.t_sim)[:-int(self.peaks)]))
      

    def moving_average(self,x,num):
        b = np.ones(num)/num
        conv = np.convolve(x,b,mode="same")
        return conv

    def RT_emp(self,T,Tc=164.885e-3,A=18.086e-3,B=2.172e-3,C=27.686e-3):
        return A*np.arctan((T-Tc)/B)+C

    def filt_y(self):
        self.moving_filter()
        self.I_sim = self.func(self.tr)
        self.Ttes_sim = self.func2(self.tr)
        self.t_sim = self.tr
        for e,i in enumerate(range(0,len(self.t_sim))):
            if e == 0:
                self.fi_y = self.I_sim[0]
            else:
                self.fi_y = np.append(self.fi_y,(0.8932e-6*(self.t_sim[i]-self.t_sim[i-1])+12e-9*self.I_sim[i-1])/(self.RT_emp(T=self.Ttes_sim[i])*(self.t_sim[i]-self.t_sim[i-1])+12e-9))
        plt.plot(self.t_sim,self.fi_y)
        plt.plot(self.t_sim,self.I_sim)
        plt.show()

    def interp(self):
        func = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        self.peaks, _ = find_peaks(np.diff(func(self.t)), height=2e-6)
        self.i_resampl = func(self.t) - func(self.t)[int(self.peaks-1)]

    def interp_p(self):
        self.interp()
        self.risefilter(t=self.t,R=(14.025+3.9)*1e-3,L=12e-9)
        self.I_sim = -self.i_resampl*self.filter_cutoff
        self.t_sim = self.t

    def interp_p4(self):
        self.peaks, _ = find_peaks(np.abs(np.diff(self.I_sim[10:])), height=2.61e-11)
        print(self.peaks)
        self.peaks = self.peaks[0]
        self.risefilter(t=self.t_sim,R=(13.14+4.0)*1e-3,L=12e-9)
        #plt.plot(self.t_sim, self.I_sim[10]-self.I_sim,".",label=self.i)
        # plt.show()
        self.I_sim = (self.I_sim[15] - self.I_sim)*self.filter_cutoff
        self.pulse_h = -np.min(self.I_sim)
        self.func = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        self.t = np.arange(0.9e-3,2e-3,1e-8)
        self.i_resampl = self.func(self.t)               
        #self.I_sim = self.i_resampl
        self.t_sim = self.t_sim
        # print(len(self.t))
        #plt.plot(self.t_sim,self.I_sim)
        #plt.show()
        #plt.legend(loc="best")

    def interp_p3(self):
        self.func = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        for e,i in enumerate(range(0,len(self.t_sim)-1)):
            if e == 0:
                self.dt_sim = self.t_sim[0]
            else:
                if self.t_sim[i]-self.t_sim[i-1] != 0:
                    self.dt_sim = np.append(self.dt_sim,self.t_sim[i]-self.t_sim[i-1])
        print(self.dt_sim[1:])
        print(np.min(self.dt_sim[2:]))
        self.dt_intp = np.arange(1e-3,self.t_sim[-1],1e-9)
        #self.dt_intp = np.append(self.dt_sim,np.arange(self.t_sim[np.argmin(self.I_sim)+1],self.t[1]-self.t[0]))
        print(f"Interpolate time length = {len(self.dt_intp)}")
        print(f"Interpolate dt = {np.min(self.dt_intp[1:])}")
        print(f"Check dt = {self.dt_sim[1]-self.dt_sim[0]}")
        self.t_sim = self.dt_intp
        self.I_sim = self.func(self.dt_intp)
        plt.plot(self.t_sim,self.I_sim,label="resump")
        #plt.plot(self.t_sim[1:],np.diff(self.I_sim))
        self.peaks, _ = find_peaks(np.abs(np.diff(self.func(self.t_sim))), height=1e-6)
        self.i_resampl = self.func(self.t_sim) - self.func(self.t_sim)[int(self.peaks-1)]
        self.risefilter3(t=self.t_sim,R=(14.025+3.9)*1e-3,L=12e-9)
        self.I_sim = self.i_resampl*self.filter_cutoff
        #plt.plot(self.t_sim,self.I_sim,label="resump")
        self.I_sim = -self.I_sim            
        #plt.show()   

    def interp_p2(self):
        self.search_t0()
        #self.I_sim -= self.I_sim[self.idx-3]
        self.risefilter2(t=self.t_sim,R=(14.025+3.9)*1e-3,L=12e-9,t0=self.t_sim[self.idx]) 
        #self.I_sim = self.I_sim * self.filter_cutoff
        func = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        self.hres = self.t[1] - self.t[0]
        self.tr = np.arange(0,self.t[-1],self.hres/100)        
        self.I_sim = func(self.tr)
        self.t_sim = self.tr

    def interp_time(self):
        self.func = interpolate.interp1d(self.t_sim, self.I_sim, kind='slinear', fill_value='extrapolate')
        for e,i in enumerate(range(0,len(self.t_sim))):
            if e == 0:
                self.dt_sim = self.t[0]
            else:
                self.dt_sim = np.append(self.t_sim,self.t[i]-self.t[i-1])
        print(np.min(self.dt_sim[1:]))
        self.dt_intp = np.arange(0,self.t_sim[-1],np.min(self.dt_sim[1:])/100)
        #self.dt_intp = np.append(self.dt_sim,np.arange(self.t_sim[np.argmin(self.I_sim)+1],self.t[1]-self.t[0]))
        print(f"Interpolate time length = {len(self.dt_intp)}")
        print(f"Interpolate dt = {np.min(self.dt_intp[1:])}")
        print(f"Check dt = {self.dt_sim[1]-self.dt_sim[0]}")
        self.t_sim = self.dt_intp
        self.I_sim = self.func(self.dt_intp)
        #plt.plot(self.dt_intp,self.func(self.dt_intp),".",label="intp")

    def moving_filter(self,num):
        self.I_sim = self.moving_average(self.func(self.dt_intp),num=num)
        self.t_sim = self.moving_average(self.dt_intp,num=num)
        self.I_sim -= self.I_sim[1000] 
        print(f"Filter time length = {len(self.t_sim)}")
        print(f"time resolution = {self.t_sim[1]-self.t_sim[0]}")
        #plt.plot(self.t_sim,self.func(self.t_sim),".",label="filter")
        #plt.show()

    def load_data(self,file):
        f = np.genfromtxt(file)
        self.t_sim    = f[:,0]*1e-3
        self.Ttes_sim = f[:,1]*1e-3
        self.Tabs_sim = f[:,2]*1e-3
        self.Tsor_sim = f[:,3]*1e-3
        self.I_sim    = f[:,4]*1e-6
        self.V_sim    = f[:,5]*1e-6
        self.Tmem_sim = f[:,6]*1e-3
        self.pulse_h  = np.max(self.I_sim)-self.I_sim[10]

        #self.I_sim    = -self.I_sim + self.I_sim[20]
        print(f"Pulse Height(COMSOL data) = {self.pulse_h}")
        print(f"Temperature Height(Ttes data) = {np.max(self.Ttes_sim) - self.Ttes_sim[10]}")
        with open(file) as f:
            self.param = float(re.sub(r"[^\d.]", "",f.readlines()[0]))

    def stack_data(self):
        if self.e == 0 :
            self.tsim_l   = self.t_sim
            self.Ttes_sim_l = self.Ttes_sim
            self.I_sim_l    = self.I_sim
        else :
            self.tsim_l   = np.vstacK((self.tsim_l,self.t_sim))
            self.Ttes_sim_l = np.vstacK((self.Ttes_sim_l,self.Ttes_sim))
            self.I_sim_l    = np.vstacK((self.I_sim_l,self.I_sim))            
          

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
        if style == "tI":
            self.xname = r"$\rm Time \ (\mu s)$"
            self.yname = r"$\rm Current \ (\mu A)$"
            self.fig = plt.figure(figsize=(9,7))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "tT":
            self.xname = r"$\rm Time \ (\mu s)$"
            self.yname = r"$\rm Temperature \ (mK)$"
            self.fig = plt.figure(figsize=(9,7))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "Ctes":
            self.fig = plt.figure(figsize=(18,12))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)
            #self.ax.set_xscale("log")

        if style == "posi":
            self.fig = plt.figure(figsize=(18,12))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)
            #self.ax.set_xscale("log")

        if style == "tT":
            self.xname = r"$\rm Time \ (s)$"
            self.yname = r"$\rm Temperature \ (mK)$"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "tI_res":
            self.fig = plt.figure(figsize=(10.6,6))
            gs = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1])
            self.ax = self.fig.add_subplot(gs1[:,:])
            self.ax2 = self.fig.add_subplot(gs2[:,:],sharex=self.ax)
            self.ax.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)
            self.xname = r"$\rm Time \ (s)$"
            self.yname = r"$\rm Current \ (\mu A)$"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax2.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "heat":
            self.xname = r"$\rm Time \ (s)$"
            self.yname = r"$\rm Current \ (\mu A)$"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "pshape":
            self.xname = r"$\rm Time \ (s)$"
            self.yname = r"$\rm Current \ (\mu A)$"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)


    def result_plot(self,subject):
        if subject == "I_pulse":
            self.plot_window(style="tI")
            self.ax.scatter(self.t_sim,(self.I_sim)*1e+6,c="Blue",s=self.ps)

        if subject == "I_pulse_cor":
            self.plot_window(style="tI")
            self.load_data(file="pulse_test2.txt")
            self.ax.scatter(self.t_sim,(self.I_sim)*1e+6,label=f"1um",c="Blue",s=self.ps)
            self.load_data(file="pulse_test_s1nm.txt")
            self.ax.scatter(self.t_sim,(self.I_sim)*1e+6,label=f"1nm",c="Red",s=self.ps)

        if subject == "tI_res":
            self.plot_window(style="tI_res")
            self.load_data(file="abs5um_first_Gtes0p33_norm.txt")
            self.interp_p4()
            self.t_sim1 = self.t_sim
            self.I_sim1 = self.I_sim
            #self.ax.plot(self.t_sim,(self.I_sim)*1e+6,label=f"1um",color="Blue")
            self.load_data(file="abs5um_sec_norm.txt")
            self.interp_p4()
            #self.ax.plot(self.t_sim,(self.I_sim)*1e+6,label=f"1nm",color="Red")
            # self.t_sim2 = self.t_sim
            # self.I_sim2 = self.I_sim
            # self.ax2.scatter(self.t_sim,(self.I_sim1-self.I_sim2)*1e+6,color="black")


        if subject == "sim_exp_cor":
            self.plot_window(style="tI")
            self.ax.plot(self.tp*1e+6,(self.p)*1e+6,label=f"Xray pulse",color="Red",lw=5)
            self.ax.plot(self.t_sim*1e+6,(self.I_sim)*1e+6,label=f"COMSOL data",color="Blue",lw=5)
            self.ax.set_xlim(0,1700)
            self.ax.legend(loc="best",fontsize=20)

        if subject == "tT":
            self.plot_window(style="tT")
            self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=f"TES temperature",color="Blue")
            self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")

        if subject == "pulse_fit":
            self.plot_window(style="tI")              
            self.ax.scatter(self.tp,(self.p)*1e+6,label=f"COMSOL",c="Blue",s=self.ps)
            self.ax.scatter(self.t_fit,self.pulse_func(self.t_fit,*self.popt)*1e+6,c="Red")

        if subject == "rise_fall":
            self.plot_window(style="tI")
            self.ax.scatter(self.t_rise,self.p_rise*1e+6,label="data")
            self.ax.plot(self.t_rise,self.rise_func(self.t_rise,),label="data")


        if subject == "Ctes_rise":
            self.xname = r"$\rm Ctes \ (pJ/K)$"
            self.yname = r"$\rm Rise \  time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.rise_list*1e+6,color="Red",marker="o",markersize=20)

        if subject == "Ctes_fall":
            self.xname = r"$\rm Ctes \ (pJ/K)$"
            self.yname = r"$\rm Fall \ time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.fall_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Ctes_ph":
            self.xname = r"$\rm Ctes \ (pJ/K)$"
            self.yname = r"$\rm Pulse \ height \ (\mu A)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.ph_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Gtes_rise":
            self.xname = r"$\rm Gtes \ (W/(m \cdot K))$"
            self.yname = r"$\rm Rise \  time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.rise_list*1e+6,color="Red",marker="o",markersize=20)

        if subject == "Gtes_fall":
            self.xname = r"$\rm Gtes \ (W/(m \cdot K))$"
            self.yname = r"$\rm Fall \ time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.fall_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Gtes_ph":
            self.xname = r"$\rm Gtes \ (W/(m \cdot K))$"
            self.yname = r"$\rm Pulse \ height \ (\mu A)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.ph_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Cabs_rise":
            self.xname = r"$\rm Cabs \ (pJ/K)$"
            self.yname = r"$\rm Rise \  time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.rise_list*1e+6,color="Red",marker="o",markersize=20)

        if subject == "Cabs_fall":
            self.xname = r"$\rm Cabs \ (pJ/K)$"
            self.yname = r"$\rm Fall \ time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.fall_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Cabs_ph":
            self.xname = r"$\rm Cabs \ (pJ/K)$"
            self.yname = r"$\rm Pulse \ height \ (\mu A)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.ph_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Gabs_rise":
            self.xname = r"$\rm Gabs \ (W/(m \cdot K))$"
            self.yname = r"$\rm Rise \  time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.rise_list*1e+6,color="Red",marker="o",markersize=20)

        if subject == "Gabs_fall":
            self.xname = r"$\rm Gabs \ (W/(m \cdot K))$"
            self.yname = r"$\rm Fall \ time \ (\mu s)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.fall_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "Gabs_ph":
            self.xname = r"$\rm Gabs \ (W/(m \cdot K))$"
            self.yname = r"$\rm Pulse \ height \ (\mu A)$"
            self.plot_window(style="Ctes")
            self.ax.plot(self.param_list,self.ph_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "p_rise":
            self.xname = r"$\rm Position Number$"
            self.yname = r"$\rm Rise \  time \ (ns)$"
            self.plot_window(style="posi")
            self.ax.plot(self.param_list,self.rise_list*1e+6,color="Red",marker="o",markersize=20)

        if subject == "p_fall":
            self.xname = r"$\rm Position Number$"
            self.yname = r"$\rm Fall \ time \ (\mu s)$"
            self.plot_window(style="posi")
            self.ax.plot(self.param_list,self.fall_list*1e+6,color="Blue",marker="o",markersize=20)

        if subject == "p_ph":
            self.xname = r"$\rm Position Number$"
            self.yname = r"$\rm Pulse \ height \ (\mu A)$"
            self.plot_window(style="posi")
            self.ax.plot(self.param_list,self.ph_list*1e+6,color="Blue",marker="o",markersize=20)

        sfn = f"./{subject}.png"
        #self.ax.legend(loc="best",fontsize=20)
        plt.show()
        self.fig.savefig(sfn,dpi=300)

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

    def rise_fall_fit(self):
        p0 = np.array([1e-8,np.min(self.p_rise)]) 
        self.popt, pcov = curve_fit(self.rise_func,self.t_rise,self.p_rise,p0=p0)
        print(self.popt,pcov)
        Ir = np.abs(np.min(self.I_sim)*0.9/np.exp(1) - self.I_sim)
        self.tau_r = self.t_sim[np.argmin(Ir)] - self.peak_time

        p0 = np.array([1e-6,np.min(self.p_fall)]) 
        self.popt, pcov = curve_fit(self.fall_func,self.t_fall,self.p_fall,p0=p0)
        print(self.popt,pcov)
        self.tau_f = self.popt[0]
        print("-----------------------------------------------")
        print("Pulse fit result")
        print(f"Rise time = {self.tau_r}")
        print(f"Fall time = {self.tau_f}")

    def rise_fall_fit2(self):
        p0 = np.array([1e-6,np.min(self.p_rise)]) 
        self.popt, pcov = curve_fit(self.rise_func,self.t_rise,self.p_rise,p0=p0)
        print(self.popt,pcov)
        self.tau_r = self.popt[0]

        p0 = np.array([1e-6,np.min(self.p_fall)]) 
        self.popt, pcov = curve_fit(self.fall_func,self.t_fall,self.p_fall,p0=p0)
        print(self.popt,pcov)
        self.tau_f = self.popt[0]
        print("-----------------------------------------------")
        print("Pulse fit result")
        print(f"Rise time = {self.tau_r}")
        print(f"Fall time = {self.tau_f}")
        plt.plot(self.t_rise,self.p_rise,".")
        plt.show()

        # self.t_fit = np.hstack((np.linspace(0,self.t0,100),np.linspace(self.t0,np.max(t),1000)))
        # self.result_plot(subject="pulse_fit")

    def rise_fall_fit_s(self):
        self.I_sim = self.I_sim[self.t_sim>0.8e-3]
        self.t_sim = self.t_sim[self.t_sim>0.8e-3] - 0.8e-3
        p0 = np.array([self.t_sim[np.argmin(self.I_sim)]-3e-6,1.3e-6,50e-6,self.pulse_h])
        plt.plot(self.t_sim,self.I_sim,".")
        self.popt, pcov = curve_fit(self.pfunc,self.t_sim,self.I_sim,p0=p0)
        self.tau_r = self.popt[1]
        self.tau_f = self.popt[2]
        print(self.popt,pcov)
        plt.plot(self.t_sim,self.pfunc(self.t_sim,*self.popt))
        plt.show()


## Pulse Data ##

    def load_fits(self,file):
        self.tp,self.p = Util.fopen(f"{file}p.fits")
        self.tn,self.n = Util.fopen(f"{file}n.fits")

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

    def trig_position(self,subject):
        if subject == "data":
            t = self.tp
            p = self.p
        if subject == "sim":
            t = self.t_sim
            p = self.I_sim
        trig_max = 0.9
        trig_min = 0.1
        tmax = np.where(p<np.min(p)*trig_max)
        tmin = np.where(p<np.min(p)*trig_min)
        self.p_rise = p[tmin[0][0]:tmax[0][0]]
        self.t_rise = t[tmin[0][0]:tmax[0][0]]-t[tmin[0][0]]
        self.p_fall = p[tmax[0][-1]:tmin[0][-1]]
        self.t_fall = t[tmax[0][-1]:tmin[0][-1]]-t[tmax[0][-1]]
        # plt.plot(self.t_rise,self.p_rise)
        # plt.plot(self.t_fall,self.p_fall)
        # plt.plot(self.t_sim,self.I_sim,label=self.i)
        # plt.legend()
        # plt.show()


    def sim_exp_cor(self,file):
        self.load_fits(file="/Users/tanakakeita/work/microcalorimeters/COMSOL/before_analysis/run011_row")
        #self.save_fits()
        self.load_data(file=file)
        #self.result_plot(subject="I_pulse")
        self.pulse_cor()
        self.interp_p4()
        self.trig_position(subject="sim")
        self.rise_fall_fit2()
        print(np.min(self.I_sim))
        self.result_plot(subject="sim_exp_cor")


    def main2(self):
        self.load_time()
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            #self.result_plot(subject="I_pulse")
            self.interp_p()
            #self.result_plot(subject="tT")
            #self.result_plot(subject="I_pulse")
            #self.interp_time()
            self.i = i
            #self.I_sim = self.I_sim[10] - self.I_sim
            # self.moving_filter(num=10)
            # for i in range(0,10000):
            #     self.t_sim = np.delete(self.t_sim,0,axis=0)
            #     self.t_sim = np.delete(self.t_sim,-1,axis=0)
            #     self.I_sim = np.delete(self.I_sim,0,axis=0)
            #     self.I_sim = np.delete(self.I_sim,-1,axis=0)
            #plt.plot(self.t_sim,self.I_sim,".")   
            self.trig_position(subject="sim")
            self.rise_fall_fit2()
            #plt.plot(self.t_sim-1.025e-3,self.I_sim,label=f"{self.param}")
            #self.result_plot(subject="I_pulse")
            if e == 0:
                self.rise_list     = self.tau_r
                self.fall_list     = self.tau_f
                self.param_list    = self.param
                self.ph_list       = self.pulse_h
                # self.t_sim_list    = self.t_sim
                # self.Ttes_sim_list = self.Ttes_sim
                # self.Tabs_sim_list = self.Tabs_sim
                # self.I_sim_list    = self.I_sim

            else :
                self.rise_list = np.append(self.rise_list,self.tau_r)
                self.fall_list = np.append(self.fall_list,self.tau_f)
                self.param_list = np.append(self.param_list,self.param)
                self.ph_list = np.append(self.ph_list,self.pulse_h)
                # self.t_sim_list    = np.vstack((self.t_sim_list,self.t_sim))
                # self.Ttes_sim_list = np.vstack((self.Ttes_sim_list,self.Ttes_sim))
                # self.Tabs_sim_list = np.vstack((self.Tabs_sim_list,self.Tabs_sim))
                # self.I_sim_list    = np.vstack((self.I_sim,_list,self.I_sim))

        self.rise_list = self.rise_list[np.argsort(self.param_list)]
        self.fall_list = self.fall_list[np.argsort(self.param_list)]
        self.ph_list = self.ph_list[np.argsort(self.param_list)]
        self.param_list = self.param_list[np.argsort(self.param_list)]

        print(self.ph_list)
        print(self.rise_list)
        print(self.fall_list)
        # self.result_plot(subject="Ctes_rise")
        # self.result_plot(subject="Ctes_fall")
        # self.result_plot(subject="Ctes_ph")
        # self.result_plot(subject="Gtes_rise")
        # self.result_plot(subject="Gtes_fall")
        # self.result_plot(subject="Gtes_ph")        
        # self.result_plot(subject="Cabs_rise")
        # self.result_plot(subject="Cabs_fall")
        # self.result_plot(subject="Cabs_ph")
        self.result_plot(subject="Gabs_rise")
        self.result_plot(subject="Gabs_fall")
        self.result_plot(subject="Gabs_ph") 
        # plt.legend()
        # plt.show()

    def main3(self):
        self.load_time()
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            #self.result_plot(subject="I_pulse")
            #self.I_sim = self.I_sim[10] -self.I_sim
            plt.plot(self.t_sim,self.I_sim,".",label=f"{self.param}")
            #self.search_t0()
            # plt.plot(self.t_sim[self.idx],self.I_sim[self.idx],".",label=f"{self.param}")
            self.interp_p3()
            #self.interp_time()
            #self.I_sim = self.I_sim[10] - self.I_sim 
            #self.moving_filter(num=100)
            #self.filt_y()
            #self.moving_filter()
            self.trig_position(subject="sim")
            self.rise_fall_fit2()
            #self.pulse_fit(subject="sim")
            if e == 0:
                self.rise_list     = self.tau_r
                self.fall_list     = self.tau_f
                self.param_list    = self.param
                self.ph_list       = self.pulse_h
                # self.t_sim_list    = self.t_sim
                # self.Ttes_sim_list = self.Ttes_sim
                # self.Tabs_sim_list = self.Tabs_sim
                # self.I_sim_list    = self.I_sim

            else :
                self.rise_list = np.append(self.rise_list,self.tau_r)
                self.fall_list = np.append(self.fall_list,self.tau_f)
                self.param_list = np.append(self.param_list,self.param)
                self.ph_list = np.append(self.ph_list,self.pulse_h)
                # self.t_sim_list    = np.vstack((self.t_sim_list,self.t_sim))
                # self.Ttes_sim_list = np.vstack((self.Ttes_sim_list,self.Ttes_sim))
                # self.Tabs_sim_list = np.vstack((self.Tabs_sim_list,self.Tabs_sim))
                # self.I_sim_list    = np.vstack((self.I_sim,_list,self.I_sim))

        self.rise_list = self.rise_list[np.argsort(self.param_list)]
        self.fall_list = self.fall_list[np.argsort(self.param_list)]
        self.ph_list = self.ph_list[np.argsort(self.param_list)]
        self.param_list = self.param_list[np.argsort(self.param_list)]

        print(self.ph_list)
        print(self.rise_list)
        print(self.fall_list)
        plt.show()
        # self.result_plot(subject="Ctes_rise")
        # self.result_plot(subject="Ctes_fall")
        # self.result_plot(subject="Ctes_ph")
        # self.result_plot(subject="Gtes_rise")
        # self.result_plot(subject="Gtes_fall")
        # self.result_plot(subject="Gtes_ph")        
        # self.result_plot(subject="Cabs_rise")
        # self.result_plot(subject="Cabs_fall")
        # self.result_plot(subject="Cabs_ph")
        self.result_plot(subject="Gabs_rise")
        self.result_plot(subject="Gabs_fall")
        self.result_plot(subject="Gabs_ph") 

    def temp_plot(self):
        self.load_time()
        self.plot_window(style="tT")
        l = len(self.filelist)
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=f"TES temperature")
            #self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")
        plt.show()

    def temp_plot2(self):
        self.load_time()
        self.plot_window(style="tT")
        l = len(self.filelist)
        self.load_data(file="abs5um_Ctes_0p1.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-14} \ [J/K]  $",c="Red",lw=3)
        self.load_data(file="abs5um_norm.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-13} \ [J/K]  $",c="Blue",lw=3)
        self.load_data(file="abs5um_Ctes_10.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-12} \ [J/K]  $",c="Green",lw=3)
        #self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")
        plt.legend(loc="best",fontsize=25)
        plt.show()

    def temp_plot3(self):
        self.load_time()
        self.plot_window(style="tT")
        l = len(self.filelist)
        self.load_data(file="abs5um_first_Ctes_10_Gtes_cor.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-14} \ [J/K]  $",c="Red",lw=3)
        self.load_data(file="abs5um_Gtes_10.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-13} \ [J/K]  $",c="Blue",lw=3)
        self.load_data(file="abs5um_first_Ctes_10_Gabs_10_Gtes_10_cor.txt")
        self.ax.plot(self.t_sim,self.Ttes_sim*1e+3,label=r"$ C_{TES}\ :\ 4.805 \times 10^{-12} \ [J/K]  $",c="Green",lw=3)
        #self.ax.plot(self.t_sim,self.Tabs_sim*1e+3,label=f"Absorber temperature",color="Red")
        plt.legend(loc="best",fontsize=25)
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

    def main4(self):
        self.load_time()
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            #self.result_plot(subject="I_pulse")
            #self.interp_p2()
            #self.result_plot(subject="tT")
            #self.result_plot(subject="I_pulse")
            self.peaks,_ = find_peaks(np.diff(self.I_sim),height=0.005e-6)
            print(self.peaks)
            self.peak_time = self.t_sim[self.peaks[1]]
            self.I_sim = self.I_sim[10] - self.I_sim
            plt.plot(self.t_sim,self.I_sim,".") 
            self.interp_time()
            self.i = i
            plt.plot(self.peak_time,self.I_sim[self.peaks[0]],".")
            plt.show()   
            self.trig_position(subject="sim")
            self.rise_fall_fit()
            #plt.plot(self.t_sim-1.025e-3,self.I_sim,label=f"{self.param}")
            #self.result_plot(subject="I_pulse")
            if e == 0:
                self.rise_list     = self.tau_r
                self.fall_list     = self.tau_f
                self.param_list    = self.param
                self.ph_list       = self.pulse_h
                # self.t_sim_list    = self.t_sim
                # self.Ttes_sim_list = self.Ttes_sim
                # self.Tabs_sim_list = self.Tabs_sim
                # self.I_sim_list    = self.I_sim

            else :
                self.rise_list = np.append(self.rise_list,self.tau_r)
                self.fall_list = np.append(self.fall_list,self.tau_f)
                self.param_list = np.append(self.param_list,self.param)
                self.ph_list = np.append(self.ph_list,self.pulse_h)
                # self.t_sim_list    = np.vstack((self.t_sim_list,self.t_sim))
                # self.Ttes_sim_list = np.vstack((self.Ttes_sim_list,self.Ttes_sim))
                # self.Tabs_sim_list = np.vstack((self.Tabs_sim_list,self.Tabs_sim))
                # self.I_sim_list    = np.vstack((self.I_sim,_list,self.I_sim))

        self.rise_list = self.rise_list[np.argsort(self.param_list)]
        self.fall_list = self.fall_list[np.argsort(self.param_list)]
        self.ph_list = self.ph_list[np.argsort(self.param_list)]
        self.param_list = self.param_list[np.argsort(self.param_list)]

        print(self.ph_list)
        print(self.rise_list)
        print(self.fall_list)
        # self.result_plot(subject="Ctes_rise")
        # self.result_plot(subject="Ctes_fall")
        # self.result_plot(subject="Ctes_ph")
        # self.result_plot(subject="Gtes_rise")
        # self.result_plot(subject="Gtes_fall")
        # self.result_plot(subject="Gtes_ph")        
        # self.result_plot(subject="Cabs_rise")
        # self.result_plot(subject="Cabs_fall")
        # self.result_plot(subject="Cabs_ph")
        # self.result_plot(subject="Gabs_rise")
        # self.result_plot(subject="Gabs_fall")
        # self.result_plot(subject="Gabs_ph") 
        # plt.legend()
        # plt.show()

    def main5(self):
        #self.load_time()
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            self.i = i
            #self.result_plot(subject="I_pulse")
            #self.interp_p2()
            #self.result_plot(subject="tT")
            #self.result_plot(subject="I_pulse")
            self.interp_p4()
            self.trig_position(subject="sim")
            self.rise_fall_fit2()
            #plt.plot(self.t_sim-1.025e-3,self.I_sim,label=f"{self.param}")
            #self.result_plot(subject="I_pulse")
            if e == 0:
                self.rise_list     = self.tau_r
                self.fall_list     = self.tau_f
                self.param_list    = self.param
                self.ph_list       = self.pulse_h
                # self.t_sim_list    = self.t_sim
                # self.Ttes_sim_list = self.Ttes_sim
                # self.Tabs_sim_list = self.Tabs_sim
                # self.I_sim_list    = self.I_sim

            else :
                self.rise_list = np.append(self.rise_list,self.tau_r)
                self.fall_list = np.append(self.fall_list,self.tau_f)
                self.param_list = np.append(self.param_list,self.param)
                self.ph_list = np.append(self.ph_list,self.pulse_h)
                # self.t_sim_list    = np.vstack((self.t_sim_list,self.t_sim))
                # self.Ttes_sim_list = np.vstack((self.Ttes_sim_list,self.Ttes_sim))
                # self.Tabs_sim_list = np.vstack((self.Tabs_sim_list,self.Tabs_sim))
                # self.I_sim_list    = np.vstack((self.I_sim,_list,self.I_sim))

        self.rise_list = self.rise_list[np.argsort(self.param_list)]
        self.fall_list = self.fall_list[np.argsort(self.param_list)]
        self.ph_list = self.ph_list[np.argsort(self.param_list)]
        self.param_list = self.param_list[np.argsort(self.param_list)]

        print(self.ph_list)
        print(self.rise_list)
        print(self.fall_list)
        # self.result_plot(subject="Ctes_rise")
        # self.result_plot(subject="Ctes_fall")
        # self.result_plot(subject="Ctes_ph")
        # self.result_plot(subject="Gtes_rise")
        # self.result_plot(subject="Gtes_fall")
        # self.result_plot(subject="Gtes_ph")        
        self.result_plot(subject="Cabs_rise")
        self.result_plot(subject="Cabs_fall")
        self.result_plot(subject="Cabs_ph")
        # self.result_plot(subject="Gabs_rise")
        # self.result_plot(subject="Gabs_fall")
        # self.result_plot(subject="Gabs_ph")
        # self.result_plot(subject="p_rise")
        # self.result_plot(subject="p_fall")
        # self.result_plot(subject="p_ph")  
        # plt.legend()
        # plt.show()

    def main7(self):
        #self.load_time()
        for e,i in enumerate(self.filelist):
            print(i)
            self.load_data(file=i)
            self.i = i
            self.interp_p4()
            self.rise_fall_fit_s()



    def main6(self):
        self.load_time()
        for e,i in enumerate(self.filelist):
            self.load_data(file=i)
            self.Ttes_sim -= self.Ttes_sim[0] 
            print(integrate.cumtrapz(self.Ttes_sim, self.t_sim)[-1])
            print(integrate.simps(self.Ttes_sim, self.t_sim))

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
            plt.legend()

        plt.show()
        self.fig.savefig(sfn,dpi=300)
