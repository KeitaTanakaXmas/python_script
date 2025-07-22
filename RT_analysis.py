import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from cal_TES import *

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2022.03.01

print('===============================================================================')
print(f"RT Analysis of TES ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class RAnalysis:

    def __init__(self):
        pass

    def loadTlog(self,filename):
        f = np.loadtxt(filename)
        self.t = f[:,0]
        self.T_ch1 = f[:,1]
        self.T_ch2 = f[:,2]
        self.T_ch3 = f[:,3]
        self.T_ch9 = f[:,4]

    def loadRlog(self,filename):
        f = np.loadtxt(filename)
        self.t2 = f[:,0]
        self.R = f[:,1]

    def loaddata(self,filename):
        f = np.loadtxt(filename)
        fr = np.reshape(f,(3,int(len(f)/3)))
        self.t = fr[0]
        self.T = fr[1]
        self.H = fr[2]

    def R_Tlog(self):
        self.loadTlog('Tlog_20240405203922_test.txt')
        self.loadRlog('Rlog.txt')
        Tt = interpolate.interp1d(self.t, self.T_ch9)
        it = np.arange(min(self.t),max(self.t),1)
        #plt.scatter(self.t, self.T_ch9)
        #plt.plot(it, Tt(it))
        #plt.show()
        print(self.R[-1])
        self.R -= self.R[-1]
        self.plot_window('RT')
        self.ax.scatter(Tt(self.t2[self.t2<max(self.t)]),self.R[self.t2<max(self.t)],s=8)
        plt.show()
        self.fig.savefig('RT_coil.png', dpi=300)


    def loaddata_with_mixing(self,filename):
        f = np.loadtxt(filename)
        fr = np.reshape(f,(4,int(len(f)/4)))
        self.t = fr[0]
        self.T_ch1 = fr[1]
        self.T_ch9 = fr[2]
        self.H = fr[3]

    def tempplot(self,filename):
        self.loaddata(filename)
        self.plot_window('LS370_withHeater')
        self.t -= self.t[0]
        ref_P = 4.0*(self.T/100e-3)**2 
        self.ax.scatter(self.t,self.T*1e+3,color='Blue',label='temp')
        self.ax2.scatter(self.t,self.H*1e+6,color='Red',label='Heater power')
        for i in range(0,len(self.T)-1):
            Power = cal_coper_c(T=self.T[i],V=3.89e-9)
        plt.legend()
        plt.show()
        print(f'Heater Average = {np.average(self.H)} +- {np.std(self.H)}')

    def tempplot_with_mixing(self,filename):
        self.loaddata_with_mixing(filename)
        self.plot_window('LS370_withHeater')
        self.t -= self.t[0]
        ref_P = 4.0e-6*(self.T_ch9/100e-3)**2 
        self.ax.scatter(self.t,self.T_ch1*1e+3,color='Black',label='temp')
        self.ax.scatter(self.t,self.T_ch9*1e+3,color='Blue',label='temp')
        #self.ax.set_yscale('log')
        self.ax2.scatter(self.t,self.H*1e+6,color='Red',label='Heater power')
        # for i in range(0,len(self.T)-1):
        #     Power = cal_coper_c(T=self.T[i],V=3.89e-9)
        plt.legend()
        plt.show()
        print(f'Heater Average = {np.average(self.H)} +- {np.std(self.H)}')

    def Dilution_temp(self,filename):
        self.loadTlog(filename)
        GM_flag = np.argmin(np.abs(self.t-1696582560))
        self.t -= self.t[GM_flag]
        self.t /= 60*60 
        GM_end  = np.argmin(np.abs(self.t-80))
        self.plot_window('LS370')
        self.ax.scatter(self.t,self.T_ch1,color='blue',s=10,label='ch1 mixing')
        self.ax.scatter(self.t,self.T_ch2,color='orange',s=10,label='ch2 still')
        self.ax.scatter(self.t,self.T_ch3,color='green',s=10,label='ch3 conc')
        self.ax.scatter(self.t,self.T_ch9,color='red',s=10,label='ch9 stage')
        #self.ax.scatter(self.t[GM_flag],self.T_ch1[GM_flag],color='black',s=20)
        self.ax.legend() 
        interp_ch1 = interpolate.interp1d(self.T_ch1[:GM_end],self.t[:GM_end])
        interp_ch2 = interpolate.interp1d(self.T_ch2[:GM_end],self.t[:GM_end])
        interp_ch3 = interpolate.interp1d(self.T_ch3[:GM_end],self.t[:GM_end])
        interp_ch9 = interpolate.interp1d(self.T_ch9[:GM_end],self.t[:GM_end])
        self.ax.plot(interp_ch1(self.T_ch1[:GM_end]),self.T_ch1[:GM_end],color='black')
        self.ax.plot(interp_ch2(self.T_ch2[:GM_end]),self.T_ch2[:GM_end],color='black')
        self.ax.plot(interp_ch3(self.T_ch3[:GM_end]),self.T_ch3[:GM_end],color='black')
        self.ax.plot(interp_ch9(self.T_ch9[:GM_end]),self.T_ch9[:GM_end],color='black')
        plt.show()




    def RT_arctan(self,T,Rn,Rc,Tc):
        a = -Tc/np.tan((np.pi*Rc)/(2*(Rn-Rc)))
        return 2*(Rn-Rc)*np.arctan((T-Tc)/a)/np.pi+Rc 

    def loadfile(self,filename,**kwargs):
        f = np.loadtxt(filename)
        self.T         = f[:,1]
        self.T_sig     = f[:,2]
        if f.shape[1] > 3 :
            self.R_ch1     = f[:,3]
            self.R_ch1_sig = f[:,4]
        if f.shape[1] > 5 :
            self.R_ch2     = f[:,5]
            self.R_ch2_sig = f[:,6]
        if f.shape[1] > 7 :
            self.R_ch3     = f[:,7]
            self.R_ch3_sig = f[:,8]
        if f.shape[1] > 9 :
            self.R_ch4     = f[:,9]
            self.R_ch4_sig = f[:,10]
        if f.shape[1] > 11 :
            self.R_ch5     = f[:,11]
            self.R_ch5_sig = f[:,12]
        if f.shape[1] > 13 :
            self.R_ch6     = f[:,13]
            self.R_ch6_sig = f[:,14]
        print(f'Loaded : {filename}')
        if 'filename2' in kwargs:
            f = np.loadtxt(kwargs['filename2'])
            self.T_2         = f[:,1]
            self.T_sig_2     = f[:,2]
            if f.shape[1] > 3 :
                self.R_ch1_2     = f[:,3]
                self.R_ch1_sig_2 = f[:,4]
            if f.shape[1] > 5 :
                self.R_ch2_2     = f[:,5]
                self.R_ch2_sig_2 = f[:,6]
            if f.shape[1] > 7 :
                self.R_ch3_2     = f[:,7]
                self.R_ch3_sig_2 = f[:,8]
            if f.shape[1] > 9 :
                self.R_ch4_2     = f[:,9]
                self.R_ch4_sig_2 = f[:,10]
            if f.shape[1] > 11 :
                self.R_ch5_2     = f[:,11]
                self.R_ch5_sig_2 = f[:,12]
            if f.shape[1] > 13 :
                self.R_ch6_2    = f[:,13]
                self.R_ch6_sig_2 = f[:,14]
            print(f'Loaded : {filename}')
        print(f'Defined following variable')
        print(self.__dict__.keys())


    def sel_ch(self,ch):
        if ch == 1:
            self.R = self.R_ch1
        if ch == 2:
            self.R = self.R_ch2
        if ch == 3:
            self.R = self.R_ch3
        if ch == 4:
            self.R = self.R_ch4
        if ch == 5:
            self.R = self.R_ch5
        if ch == 6:
            self.R = self.R_ch6

    def plot_init(self):
        #plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
        plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize']       = 13 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize']       = 13 # 軸だけ変更されます
        plt.rcParams['xtick.direction']       = 'in' # x axis in
        plt.rcParams['ytick.direction']       = 'in' # y axis in 
        plt.rcParams['axes.linewidth']        = 1.0 # axis line width
        plt.rcParams['axes.grid']             = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.15
       #plt.rcParams['scatter.edgecolors']    = None
        self.fs = 20
        self.ps = 45

    def plot_window(self,style):
        self.plot_init()
        if style == "single":
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel(rf"$\rm Temperature \ (mK)$",fontsize=self.fs)
            self.ax.set_ylabel(rf"$\rm Resistance \ (m\Omega)$",fontsize=self.fs)
            self.ax.grid(linestyle="dashed")
        if style == "LS370":
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_ylabel(rf"$\rm Temperature \ (K)$",fontsize=self.fs)
            self.ax.set_xlabel(rf"$\rm Time \ (s)$",fontsize=self.fs)
            self.ax.grid(linestyle="dashed")
            self.ax.set_yscale('log')
            self.ax.set_title('Temperature Log LS370')

        if style == "RT":
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel(rf"$\rm Temperature \ (K)$",fontsize=self.fs)
            self.ax.set_ylabel(rf"$\rm Resistance \ (m\Omega)$",fontsize=self.fs)
            self.ax.grid(linestyle="dashed")
            #self.ax.set_xscale('log')
            #self.ax.set_yscale('log')
            self.ax.set_title('Temperature and Resistance Log')

        if style == "LS370_withHeater":
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax2 = self.ax.twinx()
            self.ax.set_ylabel(rf"$\rm Temperature \ (mK)$",fontsize=self.fs)
            self.ax2.set_ylabel(rf"$\rm Heater\ Power \ (\mu W)$",fontsize=self.fs)
            self.ax.set_xlabel(rf"$\rm Time \ (s)$",fontsize=self.fs)
            self.ax.grid(linestyle="dashed")
            #self.ax.set_yscale('log')
            self.ax.set_title('Temperature Log LS370')

    def result_plot(self,subject):
        if subject == "RT":
            self.plot_window(style="single")
            if 'R_ch1' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch1*1e+3,".-",label="ch1",lw=2)
            if 'R_ch2' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch2*1e+3,".-",label="ch2",lw=2)
            if 'R_ch3' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch3*1e+3,".-",label="ch3",lw=2)
            if 'R_ch4' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch4*1e+3,".-",label="ch4",lw=2)
            if 'R_ch5' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch5*1e+3,".-",label="ch5",lw=2)
            if 'R_ch6' in self.__dict__.keys():
                self.ax.plot(self.T*1e+3,self.R_ch6*1e+3,".-",label="ch6",lw=2)
            plt.legend(loc="best",fontsize=20)
            plt.show()

        if subject == "cor_I":
            self.plot_window(style="single")
            self.ax.errorbar(self.T,self.R,xerr=self.T_sig,yerr=self.R_sig,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black",label="30uA")
            self.ax.errorbar(self.T_2,self.R_2,xerr=self.T_sig_2,yerr=self.R_sig_2,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black",label="3uA")
            plt.legend(loc="best",fontsize=20)
            plt.show()

        if subject == "fit":
            self.plot_window(style="single")
            self.ax.scatter(self.T*1e+3,self.R*1e+3,s=self.ps,color="Black",label="Data")
            self.ax.plot(self.T*1e+3,self.RT_arctan(self.T,*self.popt)*1e+3,lw=4,color="Red",label="Fitting")
            plt.legend(loc="best",fontsize=20)
            plt.show()

    def fit_arctan(self,ch):
        self.sel_ch(ch)
        p0 = np.array([np.max(self.R),np.max(self.R)/2,0.378])
        self.popt, pcov = curve_fit(self.RT_arctan,self.T,self.R,p0=p0)
        print("----------------------------------------------------------")
        print(f"Normal Resistance: Rn = {self.popt[0]*1e+3} ± {np.sqrt(pcov[0,0])*1e+3} mOhm")
        print(f"Critical Resistance: Rc = {self.popt[1]*1e+3} ± {np.sqrt(pcov[1,1])*1e+3} mOhm")
        print(f"Critical Temperature: Tc = {self.popt[2]*1e+3} ± {np.sqrt(pcov[2,2])*1e+3} mK")
        self.result_plot(subject="fit")


    def plot_txt_out(self,filename):
        self.read_txt(filename)
        self.plot_txt()

    def RT_plot(self,filename):
        self.loadfile(filename)
        self.result_plot(subject="RT")




