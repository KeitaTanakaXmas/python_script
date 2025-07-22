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


__author__ =  'Keita Tanaka'
__version__=  '4.0.0' #2024.04.18

## Append Magnetic dependance

print('===============================================================================')
print(f"Current and Voltage Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class IVAnalysis:

    def __init__(self, ):
        self.Min    = 104.12e-12  #[H]
        self.Mfb    = 85.21e-12  #[H]
        self.Rshunt = 3.9e-3  #[Ohm]
        self.Rfb    = 30e+3  #[Ohm]
        self.CRange68 = 2.295815160785974337606
        self.CRange90 = 4.605170185988091368036
        self.CRange99 = 9.210340371976182736072		

    def load_setting_file(self,mes_number):
        self.setting_file_name = 'setting.hdf5'
        self.f = h5py.File(self.setting_file_name,'a')
        self.Measurement_ID_list = list(self.f['Measurement'].keys())
        self.Measurement_ID_list.sort()
        print('-----------------------------------------------------------')
        if mes_number in self.Measurement_ID_list:
            self.mes_number = mes_number
            self.squid_Ib = int(self.f[f'Measurement/{self.mes_number}/SQUID_current_bias'][...])
            print(f'{self.mes_number} SQUID settings')
            print(f'SQUID current bias = {self.squid_Ib} uA')
            if f'SB{self.squid_Ib}uA' in self.f['Min'].keys():
                self.Min = self.f[f'Min/SB{self.squid_Ib}uA/average'][...]
                self.Mfb = self.f[f'Mfb/SB{self.squid_Ib}uA/average'][...]
            else:
                print(f'SB{self.squid_Ib}uA is not in the setting file.')
                print('SQUID bias current list is as follows:')
                print(self.f['Min'].keys())
            self.Rfb = self.f[f'Measurement/{self.mes_number}/Rfb'][...]*1e+3
            self.excVtoI = self.Mfb / (self.Min * self.Rfb)
            print(f'Min = {self.Min*1e+12} pH')
            print(f'Mfb = {self.Mfb*1e+12} pH')
            print(f'Rfb = {self.Rfb*1e-3} kOhm')
            print(f'SQUID Voltage to Current Factor = {self.excVtoI}')

        else:
            print(f'mes_number {mes_number} is not in the setting file.')
        print('-----------------------------------------------------------')

    def initialize(self,Iunit,savehdf5,mes_number):
        self.data             = {}
        self.data["data"]     = {}
        self.data["analysis"] = {}
        self.ch 	   =  1
        #self.filepath  =  f"./../data/IV/{mes_number}"
        self.filepath = './'
        self.filelist  =  sorted(glob.glob(f"{self.filepath}/*.txt"))
        self.Iunit     =  Iunit
        self.savehdf5  =  savehdf5
        print("-----------------------------------------------")
        print(f"{__name__}")
        print(self.filelist)

    def cal_Ttes(self,G0,n,Pb,Tbath):
        return ((Tbath*1e-3)**n + n*Pb/G0) ** (1/n)

    def linear_no_ofs(self,x,a):
        return a * x

    def linear(self,x,a,b):
        return a * x + b

    def RT_emp(self,T,Tc,A,B,C):
        return A*np.arctan((T-Tc)/B)+C

    def RT_emp2(self,T,Tc=164.885e-3,A=18.086e-3,B=2.172e-3,C=27.686e-3):
        return A*np.arctan((T-Tc)/B)+C

    def RT_exp(self,T,Tc,T1,T2,R0,Rc):
        return R0/((1+np.exp(-(T-Tc)/T1))*((1+np.exp(-(T-Tc)/T2)))) + Rc

    def Pb_func(self,Tbath,G0,Tc,n):
        return G0*((Tc*10**(-3))**n-(Tbath*10**(-3))**n)/n

    def getNearValueID(self, list, num):
        idx = np.abs(np.asarray(list) - num).argmin()
        return idx

    def data_out(self,filename):
        self.Ibias_e = np.loadtxt(filename)[:,0]
        self.Vout_e = np.loadtxt(filename)[:,1]

        with open(filename) as f:
            self.Tbath_e = int(re.sub(r"\D", "",f.readlines()[4]))
        with open(filename) as f:
            self.B_e = int(float(re.findall(r'-?\d+\.\d+', f.readlines()[12])[0])*1e+6)

    def Tbath_out(self):
        for e,i in enumerate(self.filelist):
            self.data_out(filename=str(i))
            if e == 0:
                self.Tbath = self.Tbath_e
                self.B     = self.B_e
            else:
                self.Tbath = np.append(self.Tbath,self.Tbath_e)
                self.B = np.append(self.B,self.B_e)

    def cal_par(self):
        if self.Iunit == "uA":
            self.Ibias_e = self.Ibias_e * 1e-6
            #print(f"Iunit = uA")
        elif self.Iunit == "A":
            self.Ibias_e = self.Ibias_e
            #print("Iunit = A")
        self.Ibias_e[-1] = 0
        self.Vcor_e  =  self.Vout_e - self.Vout_e[-1]
        self.Vout_e  =  self.Vout_e[self.Ibias_e>0]
        self.Vcor_e  =  self.Vcor_e[self.Ibias_e>0]
        self.Ibias_e =  self.Ibias_e[self.Ibias_e>0]
        self.Ites_e  =  np.array(self.Mfb*self.Vcor_e/(self.Min*self.Rfb)) #[A]
        self.Vtes_e  =  np.array((self.Ibias_e-self.Ites_e)*self.Rshunt) #[V]
        self.Rtes_e  =  np.array(self.Vtes_e/self.Ites_e) #[Ohm]
        self.Pb_e    =  np.array(self.Ites_e * self.Vtes_e) #[W] 

    def Normal_fitting(self):
        popt, pcov = curve_fit(self.linear,self.Ibias_e,self.Vout_e)
        self.Vcor_e  =  self.Vout_e -popt[1]
        self.Ites_e  =  np.array(self.Mfb*self.Vcor_e/(self.Min*self.Rfb)) #[A]
        self.Vtes_e  =  np.array((self.Ibias_e-self.Ites_e)*self.Rshunt) #[V]
        self.Rtes_e  =  np.array(self.Vtes_e/self.Ites_e) #[Ohm]
        self.Pb_e    =  np.array(self.Ites_e * self.Vtes_e) #[W] 

    def data_set(self):
        if f"{self.Tbath_e}mK" not in self.data["data"].keys():
            self.data["data"][f"{self.Tbath_e}mK"]              = {}
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"] = {}
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Ibias"] = self.Ibias_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Vout"]  = self.Vout_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Vcor"]  = self.Vcor_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Ites"]  = self.Ites_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Vtes"]  = self.Vtes_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Rtes"]  = self.Rtes_e
        self.data["data"][f"{self.Tbath_e}mK"][f"{self.B_e}uA"]["Pb"]    = self.Pb_e

    def stack_data(self):
        for e,i in enumerate(list(self.data["data"].keys())):
            print(f"{i}mK")
            print(self.data["data"][i].keys())
            if e == 0:
                self.R_a = self.data["data"][f"{i}"]["Rtes"] 
                self.T_a = self.data["data"][f"{i}"]["Ttes"]
                self.I_a = self.data["data"][f"{i}"]["Ites"]
            elif i != "210.0mK" and e != 0:
                self.R_a = np.hstack((self.R_a,self.data["data"][f"{i}"]["Rtes"]))
                self.T_a = np.hstack((self.T_a,self.data["data"][f"{i}"]["Ttes"]))
                self.I_a = np.hstack((self.I_a,self.data["data"][f"{i}"]["Ites"]))

    def all_process(self):
        for i in self.filelist:
            self.data_out(filename=str(i))
            self.cal_par()
            self.data_set()
        self.Tbath_out()

    def Rn_process(self):
        self.data_out(filename=str(self.Rn_filelist))
        self.cal_par()
        self.data_set()

    def save_RTI_txt(self,Tbath):
            R_data = self.data["data"][f"{Tbath}mK"]["Rtes"] 
            T_data = self.data["data"][f"{Tbath}mK"]["Ttes"]
            I_data = self.data["data"][f"{Tbath}mK"]["Ites"]
            data   = np.vstack((R_data,T_data))
            data   = np.vstack((data,I_data))
            np.savetxt(f"RTI_data_{Tbath}.txt",data.T)

    def save_hdf5(self):
        print("-----------------------------------------------")
        print("Saving result in HDF5 File")
        print(f"savefile = {self.savehdf5}")
        with h5py.File(self.savehdf5,"a") as f:
            if f"ch{self.ch}" in f.keys():
                if "IV" in f[f"ch{self.ch}"].keys():
                    del f[f"ch{self.ch}"]["IV"]				
            for i in list(self.data["data"].keys()):
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Ibias",data=self.data["data"][i]["Ibias"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Vout",data=self.data["data"][i]["Vout"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Vcor",data=self.data["data"][i]["Vcor"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Ites",data=self.data["data"][i]["Ites"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Vtes",data=self.data["data"][i]["Vtes"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Rtes",data=self.data["data"][i]["Rtes"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Pb",data=self.data["data"][i]["Pb"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Ttes",data=self.data["data"][i]["Ttes"])
                    f.create_dataset(f"ch{self.ch}/IV/data/{i}/Gtes",data=self.data["data"][i]["Gtes"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/G0",data=self.data["analysis"]["Gfit"]["G0"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/Tc",data=self.data["analysis"]["Gfit"]["Tc"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/n",data=self.data["analysis"]["Gfit"]["n"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/G0_err",data=self.data["analysis"]["Gfit"]["G0_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/Tc_err",data=self.data["analysis"]["Gfit"]["Tc_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/Gfit/n_err",data=self.data["analysis"]["Gfit"]["n_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/Tc",data=self.data["analysis"]["RTfit"]["Tc"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/A",data=self.data["analysis"]["RTfit"]["A"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/B",data=self.data["analysis"]["RTfit"]["B"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/Offset",data=self.data["analysis"]["RTfit"]["Offset"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/Tc_err",data=self.data["analysis"]["RTfit"]["Tc_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/A_err",data=self.data["analysis"]["RTfit"]["A_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/B_err",data=self.data["analysis"]["RTfit"]["B_err"])
            f.create_dataset(f"ch{self.ch}/IV/analysis/RTfit/Offset_err",data=self.data["analysis"]["RTfit"]["Offset_err"])
        print("Finished")

    def load_hdf5(self,Tbath):
        with h5py.File(self.savehdf5,"r") as f:
            self.Rtes_ch1 = f[f"ch1/IV/data/{Tbath}mK/Rtes"][:]
            self.Rtes_ch2 = f[f"ch2/IV/data/{Tbath}mK/Rtes"][:]
            self.Ttes_ch1 = f[f"ch1/IV/data/{Tbath}mK/Ttes"][:]
            self.Ttes_ch2 = f[f"ch2/IV/data/{Tbath}mK/Ttes"][:]
            ch2_list = np.vstack((self.Ttes_ch2,self.Rtes_ch2))
            print(ch2_list.T)
            np.savetxt("test_ch2.txt",ch2_list.T)


    def alpha_diff(self,R,T):
        return T[:-1]*np.diff(R)/(R[:-1]*np.diff(T))

    def moving_average(self,a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def plot_init(self):
        self.fig = plt.figure(figsize=(10.6,6))
        self.ax  = plt.subplot(1,1,1)
        self.ax.grid(linestyle="dashed")
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)
        self.fs = 20
        self.ps = 50

    def result_plot(self,plot_subject):
        self.plot_init()
        sfn = f"./figure/IV/ch{self.ch}_{plot_subject}.png"
        if plot_subject != "RT_emp_plot" and plot_subject != "RT_cor" and plot_subject != "alpha_diff_cor" and plot_subject != "alpha_plot":
            self.data_length = len(self.data["data"].keys())

        if plot_subject == "IbiasVcor":

            self.ax.set_xlabel(r'$\rm I_{bias} \ (\mu A)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm V_{out} \ (V)$',fontsize=self.fs)
            for e,i in enumerate(self.data["data"].keys()):
                self.ax.scatter(self.data["data"][i]["Ibias"]*1e+6,self.data["data"][i]["Vcor"],s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))

        if plot_subject == "RtesPb":

            self.ax.set_xlabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm P_{b} \ (pW/K)$',fontsize=self.fs)
            for e,i in enumerate(self.data["data"].keys()):
                self.ax.scatter(self.data["data"][i]["Rtes"]*1e+3,self.data["data"][i]["Pb"]*1e+12,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))

        if plot_subject == "RtesPb_with_idx":

            self.ax.set_xlabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm P_{b} \ (pW/K)$',fontsize=self.fs)
            print(self.data["data"].keys())
            for e,(i,idx) in enumerate(zip(self.data["data"].keys(),self.idx_list)):
                print(e,i,idx)
                self.ax.scatter(self.data["data"][i]["Rtes"]*1e+3,self.data["data"][i]["Pb"]*1e+12,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))
                self.ax.scatter(self.data["data"][i]["Rtes"][idx-5:idx+5]*1e+3,self.data["data"][i]["Pb"][idx-5:idx+5]*1e+12,s=self.ps,color='black')

        if plot_subject == "IbiasRtes":

            self.ax.set_xlabel(r'$\rm I_{bias} \ (\mu A)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            for e,i in enumerate(self.data["data"].keys()):
                self.ax.scatter(self.data["data"][i]["Ibias"]*1e+6,self.data["data"][i]["Rtes"]*1e+3,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))

        if plot_subject == "VtesItes":

            self.ax.set_xlabel(r'$\rm V_{TES} \ (\mu V)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm I_{TES} \ (\mu A)$',fontsize=self.fs)
            for e,i in enumerate(self.data["data"].keys()):
                self.ax.scatter(self.data["data"][i]["Vtes"]*1e+6,self.data["data"][i]["Ites"]*1e+6,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))

        if plot_subject == "IbiasRtes_single":

            self.ax.set_xlabel(r'$\rm I_{bias} \ (\mu A)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax.scatter(self.data["data"][f"{self.Tbath_s}mK"]["Ibias"]*1e+6,self.data["data"][f"{self.Tbath_s}mK"]["Rtes"]*1e+3,s=self.ps,label=f"{self.Tbath_s}mK",c="Blue")

        if plot_subject == "ItesVtes_single":

            self.ax.set_xlabel(r'$\rm I_{TES} \ (\mu A)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm V_{TES} \ (\mu V)$',fontsize=self.fs)
            self.ax.scatter(self.data["data"][f"{self.Tbath_s}mK"]["Ites"]*1e+6,self.data["data"][f"{self.Tbath_s}mK"]["Vtes"]*1e+6,s=self.ps,label=f"{self.Tbath_s}mK",c="Blue")
            self.ax.plot(self.data["data"][f"{self.Tbath_s}mK"]["Ites"]*1e+6,self.res,color="black")

        if plot_subject == "IbiasItes":

            self.ax.set_xlabel(r'$\rm I_{bias} \ (\mu A)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm I_{TES} \ (\mu A)$',fontsize=self.fs)
            #self.ax.set_title(r'$\rm I_{bias}-V_{out} \ $',fontsize=14)
            def linear(x,a):
                return a*x
            for e,i in enumerate(self.data["data"].keys()):
                Ites = self.data["data"][i]["Ites"]
                Ibias = self.data["data"][i]["Ibias"][np.argmax(Ites):]
                Ites = Ites[np.argmax(Ites):]
                self.ax.scatter(Ibias*1e+6,Ites*1e+6,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))
                popt, pcov = curve_fit(linear,Ites,Ibias)
                print(popt,pcov)
                self.ax.plot(Ibias,linear(Ibias,popt),color="red")
                plt.show()

        if plot_subject == "RtesTtes":

            self.ax.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            for e,i in enumerate(self.data["data"].keys()):
                self.ax.scatter(self.data["data"][i]["Ttes"]*1e+3,self.data["data"][i]["Rtes"]*1e+3,s=self.ps,label=f"{i}",c=cm.jet([e/self.data_length]))

        if plot_subject == "RT_emp_plot":

            self.ax.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax.scatter(self.Ttes*1e+3,self.Rtes*1e+3,s=self.ps,c="Red")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)

        if plot_subject == "alpha_plot":

            self.ax.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm Alpha $',fontsize=self.fs)
            self.ax.scatter(self.Ttes*1e+3,self.Alpha,s=self.ps,c="Blue")
            #self.ax.scatter(self.T*1e+3,self.a,s=self.ps,c="Red")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)

        if plot_subject == "RT_emp_fit":

            del self.ax
            self.fig.clf() 
            gs = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1])
            self.ax = self.fig.add_subplot(gs1[:,:])
            self.ax2 = self.fig.add_subplot(gs2[:,:],sharex=self.ax)
            self.ax.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax2.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax2.set_ylabel(r'$\rm Residual$',fontsize=self.fs)
            self.ax.scatter(self.Ttes*1e+3,self.Rtes*1e+3,s=self.ps,label="data",c="black")
            self.ax.plot(self.Tfit*1e+3,self.RT_emp(self.Tfit,*self.fres)*1e+3,"-",label=f"fit model",color="red")
            self.ax2.scatter(self.Ttes*1e+3,(self.Rtes-self.RT_emp(self.Ttes,*self.fres))*1e+3,color="black")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)

        if plot_subject == "RT_exp_fit":

            del self.ax
            self.fig.clf() 
            gs = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1])
            self.ax = self.fig.add_subplot(gs1[:,:])
            self.ax2 = self.fig.add_subplot(gs2[:,:],sharex=self.ax)
            self.ax.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax2.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax2.set_ylabel(r'$\rm Residual$',fontsize=self.fs)
            self.ax.scatter(self.Ttes*1e+3,self.Rtes*1e+3,s=self.ps,label="data",c="black")
            self.ax.plot(self.Tfit*1e+3,self.RT_exp(self.Tfit,*self.fres)*1e+3,"-",label=f"fit model",color="red")
            self.ax2.scatter(self.Ttes*1e+3,(self.Rtes-self.RT_exp(self.Ttes,*self.fres))*1e+3,color="black")
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)
        
        # if plot_subject == "Gfit":

        # 	self.ax.set_xlabel(r"$\rm T_{bath} \ (mK)$",fontsize=self.fs)
        # 	self.ax.set_ylabel(r"$\rm P_{b} \ (pW/K) $",fontsize=self.fs)
        # 	self.ax.scatter(self.Tbath,self.Pbc*1e+12,s=100,label=f"data",c="black")
        # 	self.ax.plot(self.Tbath,self.Pb_func(self.Tbath,self.G0, self.Tc, self.n)*1e+12,linestyle="dashed",color="red",label="fit")

        if plot_subject == "Gfit":
            self.ax.set_xlabel(r"$\rm T_{bath} \ (mK)$",fontsize=self.fs)
            self.ax.set_ylabel(r"$\rm P_{b} \ (pW/K) $",fontsize=self.fs)
            self.ax.errorbar(self.Tbath,self.Pbc*1e+12,yerr=self.Pbc_err*1e+12,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black",label="data")
            self.ax.plot(self.Tbath,self.Pb_func(self.Tbath,self.G0, self.Tc, self.n)*1e+12,linestyle="dashed",color="red",label="fit")

        if plot_subject == "Gcont":
            self.ax.set_xlabel(r"$\rm n $",fontsize=self.fs)
            self.ax.set_ylabel(r"$ G_0 \ \rm (nW/K) $",fontsize=self.fs)
            cslevels = [self.CRange68,self.CRange90,self.CRange99]
            cont = self.ax.contour(self.step_x,self.step_y*1e+9,self.delc,levels=cslevels,colors=["red","green","blue"])
            fn=[r"$ \rm 68 \%$",r"$ \rm 90 \%$",r"$ \rm 99 \%$"]
            fmt={}
            for l, s in zip(cslevels, fn):
                fmt[l] = s
            cont.clabel(cslevels, fmt=fmt, fontsize=20)
            self.ax.scatter(self.step_x[self.mindelc_idx[1]],self.step_y[self.mindelc_idx[0]]*1e+9,marker="+",color="black",s=200,label="best fit")

        if plot_subject == "RT_cor":
            self.load_hdf5(Tbath=90.0)
            self.ax.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=self.fs)
            self.ax.scatter(self.Ttes_ch1*1e+3,self.Rtes_ch1*1e+3,s=self.ps,label=rf"$\rm Abs \ 300nm$",c="Red")
            self.ax.scatter(self.Ttes_ch2*1e+3,self.Rtes_ch2*1e+3,s=self.ps,label=rf"$\rm Abs \ 5\mu m$",c="Blue")
            sfn = "RT_cor.png"

        if plot_subject == "alpha_diff_cor":
            self.load_hdf5(Tbath=90.0)
            self.ax.set_xlabel(r'$\rm T_{TES} \ (mK)$',fontsize=self.fs)
            self.ax.set_ylabel(r'$\rm Alpha $',fontsize=self.fs)
            self.ax.scatter(self.moving_average(self.Ttes_ch1,2)*1e+3,self.alpha_diff(T=self.Ttes_ch1,R=self.Rtes_ch1),s=self.ps,label=rf"$\rm Abs \ 300nm$",c="Red")
            self.ax.scatter(self.moving_average(self.Ttes_ch2,2)*1e+3,self.alpha_diff(T=self.Ttes_ch1,R=self.Rtes_ch1),s=self.ps,label=rf"$\rm Abs \ 5\mu m$",c="Blue")
            sfn = "Alpha_diff_cor.png"

        self.ax.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)
        self.fig.tight_layout()
        plt.pause(0.1)
        plt.show()
        self.fig.savefig(sfn,dpi=300)

    # def Rn_define(self,Ibias=800,per=0.5):
    # 	for e,i in enumerate(self.data["data"].keys()):
    # 		if e == 0:
    # 			self.Rn = self.data["data"][i]["Rtes"][self.getNearValueID(self.data["data"][i]["Ibias"]*1e+6,Ibias)]
    # 		else:
    # 			self.Rn = np.append(self.Rn,self.data["data"][i]["Rtes"][self.getNearValueID(self.data["data"][i]["Ibias"]*1e+6,Ibias)])
    # 	for e,i in enumerate(self.data["data"].keys()):
    # 		if e == 0:
    # 			self.Pbc = self.data["data"][i]["Pb"][self.getNearValueID(self.data["data"][i]["Rtes"],np.average(self.Rn)*per)]
    # 		else:
    # 			self.Pbc = np.append(self.Pbc,self.data["data"][i]["Pb"][self.getNearValueID(self.data["data"][i]["Rtes"],np.average(self.Rn)*per)])
    # 	print("-----------------------------------------------")
    # 	print(f"Tbath = {self.data['data'].keys()}")
    # 	print(f"TES Normal Resistance : Rn = {np.average(self.Rn)*1e+3} mOhm")
    # 	print(f"TES Resistance near Transition (definition:Rn*{per}) : Rc = {np.average(self.Rn)*1e+3*per} mOhm")

    def Rn_define(self,Ibias=800,per=0.5):
        for e,i in enumerate(self.data["data"].keys()):
            if e == 0:
                self.Rn = self.data["data"][i]["Rtes"][self.getNearValueID(self.data["data"][i]["Ibias"]*1e+6,Ibias)]
            else:
                self.Rn = np.append(self.Rn,self.data["data"][i]["Rtes"][self.getNearValueID(self.data["data"][i]["Ibias"]*1e+6,Ibias)])

        for e,i in enumerate(self.data["data"].keys()):
            self.idx_list = []
            idx = self.getNearValueID(self.data["data"][i]["Rtes"],np.average(self.Rn)*per)
            self.idx_list.append(idx)
            if e == 0:
                self.Pbc     = np.average(self.data["data"][i]["Pb"][idx-5:idx+5])
                self.Pbc_err = np.std(self.data["data"][i]["Pb"][idx-5:idx+5])
            else:
                self.Pbc     = np.append(self.Pbc,np.average(self.data["data"][i]["Pb"][idx-5:idx+5]))
                self.Pbc_err = np.append(self.Pbc_err,np.std(self.data["data"][i]["Pb"][idx-5:idx+5]))

        print("-----------------------------------------------")
        print(f"Tbath = {self.data['data'].keys()}")
        print(f"TES Normal Resistance : Rn = {np.average(self.Rn)*1e+3} mOhm")
        print(f"TES Resistance near Transition (definition:Rn*{per}) : Rc = {np.average(self.Rn)*1e+3*per} mOhm")	

    def Gfit(self):
        p0 = np.array([1e-9,150,3.5])
        print(self.Tbath,self.Pbc)
        print(len(self.Tbath),len(self.Pbc))
        popt,pcov = curve_fit(self.Pb_func,self.Tbath,self.Pbc,p0=p0)
        self.G0, self.Tc, self.n = popt
        self.G0_err, self.Tc_err, self.n_err = np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1]),np.sqrt(pcov[2,2])
        print("-----------------------------------------------")
        print("G Fitting Result")
        print(f"G0 = {self.G0*1e+9} +- {self.G0_err*1e+9} nW/K")
        print(f"Tc = {self.Tc} +- {self.Tc_err} mK")
        print(f"n = {self.n} +- {self.n_err}")
        print("G at 100mK = ",self.G0*(100*1e-3)**(self.n-1))
        print(f"Gtes at {self.Tc}mK = ",self.G0*(self.Tc*1e-3)**(self.n-1))
        self.data["analysis"]["Gfit"]           = {}
        self.data["analysis"]["Gfit"]["G0"]     = self.G0
        self.data["analysis"]["Gfit"]["n"]      = self.n
        self.data["analysis"]["Gfit"]["Tc"]     = self.Tc
        self.data["analysis"]["Gfit"]["G0_err"] = self.G0_err
        self.data["analysis"]["Gfit"]["n_err"]  = self.n_err
        self.data["analysis"]["Gfit"]["Tc_err"] = self.Tc_err
        self.result_plot(plot_subject="Gfit")

    def Gfit2(self):
        self.model = Model(self.Pb_func)
        self.params = self.model.make_params()
        self.model.set_param_hint('G0',min=0,max=1)
        self.model.set_param_hint('n',min=0,max=10)
        self.model.set_param_hint('Tc',min=0,max=300)
        result = self.model.fit(Tbath=self.Tbath,data=self.Pbc,weights=1/self.Pbc_err,G0=1e-9,n=3.5,Tc=165)
        self.G0 = result.best_values["G0"]
        self.n  = result.best_values["n"]
        self.Tc = result.best_values["Tc"]
        self.G0_err = result.params['G0'].stderr
        self.n_err = result.params['n'].stderr
        self.Tc_err = result.params['Tc'].stderr
        self.data["analysis"]["Gfit"]           = {}
        self.data["analysis"]["Gfit"]["G0"]     = self.G0
        self.data["analysis"]["Gfit"]["n"]      = self.n
        self.data["analysis"]["Gfit"]["Tc"]     = self.Tc
        self.data["analysis"]["Gfit"]["G0_err"] = self.G0_err
        self.data["analysis"]["Gfit"]["n_err"]  = self.n_err
        self.data["analysis"]["Gfit"]["Tc_err"] = self.Tc_err
        print(result.fit_report())
        self.result_plot(plot_subject="Gfit")			

    def G_cont(self,step,nmin,nmax,G0min,G0max):
        self.model.set_param_hint('G0',min=0,max=1,vary=False)
        self.model.set_param_hint('n',min=0,max=10,vary=False)
        self.model.set_param_hint('Tc',min=0,max=300,vary=False)

        self.step_x = np.linspace(nmin,nmax,step)
        self.step_y = np.linspace(G0min*1e-9,G0max*1e-9,step)

        for ee,x_e in enumerate(self.step_x):
            for e,y_e in enumerate(self.step_y):
                result = self.model.fit(Tbath=self.Tbath,data=self.Pbc,weights=1/self.Pbc_err,G0=y_e,n=x_e,Tc=self.Tc)
                if e == 0:
                    res_h = np.array([result.redchi])
                    res_a = np.array([x_e,y_e,result.redchi])
                    print(x_e,y_e,result.redchi)
                else:
                    res_h = np.vstack((res_h,np.array([result.redchi])))
                    res_a = np.vstack((res_a,np.array([x_e,y_e,result.redchi])))
                    print(x_e,y_e,result.redchi)
            if ee == 0:
                res = res_h
                res_all = res_a
            else:
                res = np.hstack((res,res_h))
                res_all = np.vstack((res_all,res_a))
        self.delc = res - np.min(res)
        self.mindelc_idx = np.unravel_index(np.argmin(self.delc),self.delc.shape) 
        x = self.step_x
        y = self.step_y
        self.res_all = res_all
        self.result_plot(plot_subject="Gcont")

    def process_Ttes(self):
        for i in self.Tbath:
            Pb = self.data["data"][f"{i}mK"]["Pb"]
            Ttes = self.cal_Ttes(G0=self.G0,n=self.n,Pb=Pb,Tbath=i)
            self.data["data"][f"{i}mK"]["Ttes"] = Ttes
            self.data["data"][f"{i}mK"]["Gtes"] = self.G0*np.array(Ttes)**(self.n-1)

    def process_Ttes_forB(self):
        print(self.Tbath[0])
        for i in self.B:
            print(i)
            Pb = self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Pb"]
            Ttes = self.cal_Ttes(G0=self.G0,n=self.n,Pb=Pb,Tbath=self.Tbath[0])
            self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Ttes"] = Ttes
            self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Gtes"] = self.G0*np.array(Ttes)**(self.n-1)

    def RT_fit(self,Tbath,model='arct'):
        self.Rtes = self.data["data"][f"{Tbath}mK"]["Rtes"]
        self.Ttes = self.data["data"][f"{Tbath}mK"]["Ttes"]
        self.Tbath = Tbath
        if model == 'arct':
            self.fres, pcov = curve_fit(self.RT_emp,self.Ttes,self.Rtes,p0=[0.168,0.025,0.05,0.10])
            Tc_RT,A,B,C = self.fres
            Tc_RT_err,A_err,B_err,C_err = np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1]),np.sqrt(pcov[2,2]),np.sqrt(pcov[3,3])
            print("-----------------------------------------------")
            print("RT fitting by emperical artan model")
            print(f"Tc = {Tc_RT*1e+3} +- {Tc_RT_err*1e+3} mK")
            print(f"A = {A} +- {A_err}")
            print(f"B = {B} +- {B_err}")
            print(f"Offset = {C} +- {C_err}")
            print(f"Rtes = {A}*arctan((T[mK]-{Tc_RT*1e+3})/{B})+{C}")
            self.data["analysis"]["RTfit"]               = {}
            self.data["analysis"]["RTfit"]["Tc"]         = Tc_RT
            self.data["analysis"]["RTfit"]["A"]          = A
            self.data["analysis"]["RTfit"]["B"]          = B
            self.data["analysis"]["RTfit"]["Offset"]     = C
            self.data["analysis"]["RTfit"]["Tc_err"]     = Tc_RT_err
            self.data["analysis"]["RTfit"]["A_err"]      = A_err
            self.data["analysis"]["RTfit"]["B_err"]      = B_err
            self.data["analysis"]["RTfit"]["Offset_err"] = C_err		
            self.Tfit = np.linspace(np.min(self.Ttes),np.max(self.Ttes),int(1e+4))
            self.result_plot(plot_subject="RT_emp_fit")
        elif model == 'exp':
            self.fres, pcov = curve_fit(self.RT_exp,self.Ttes,self.Rtes,p0=[0.17,0.11,0.1,0.03,0.1])
            print(self.fres)
            print(pcov)
            self.Tfit = np.linspace(np.min(self.Ttes),np.max(self.Ttes),int(1e+4))
            self.result_plot(plot_subject="RT_exp_fit")

    def ItesVtes_fit(self):
        self.Rn_Z, pcov = curve_fit(self.linear_no_ofs,self.data["data"][f"{self.Tbath_s}mK"]["Ites"],self.data["data"][f"{self.Tbath_s}mK"]["Vtes"]) 
        self.res = self.linear_no_ofs(self.data["data"][f"{self.Tbath_s}mK"]["Ites"],self.Rn_Z) * 1e+6
        self.result_plot(plot_subject="ItesVtes_single")

    def Rn_for_Z(self,Tbath,Ibias):
        self.Tbath_s = str(int(Tbath))
        self.Rn_filepath  =  f"./../data/IV/ch{self.ch}/Rnormal"
        self.Rn_filelist  =  sorted(glob.glob(f"{self.Rn_filepath}/*.txt"))[0]
        self.Rn_process()
        self.ItesVtes_fit()
        print("-----------------------------------------------")
        print("Normal Resistance for Analysis of Complex Impedance")
        print(f"Rnormal = {float(self.Rn_Z) * 1e+3} mOhm")
        with h5py.File(self.savehdf5,"a") as f:
            if "Rn_Z" in f[f"ch{self.ch}/IV/analysis"].keys(): 
                del f[f"ch{self.ch}/analysis/Rn_Z"]
            f.create_dataset(f"ch{self.ch}/IV/analysis/Rn_Z",data=self.Rn_Z)
        #self.result_plot(plot_subject="IbiasRtes_single")

## Append for COMSOL simlation(2021.12.21) ##

    def RT_plot(self,ch):
        savehdf5 = self.savehdf5
        self.ch  = ch
        with h5py.File(savehdf5,"r") as f:
            self.A = f[f"ch{ch}"]["IV"]["analysis"]["RTfit"]["A"][...]
            self.B = f[f"ch{ch}"]["IV"]["analysis"]["RTfit"]["B"][...]
            self.C = f[f"ch{ch}"]["IV"]["analysis"]["RTfit"]["Offset"][...]
            self.Tc = f[f"ch{ch}"]["IV"]["analysis"]["RTfit"]["Tc"][...]

        self.Ttes = np.linspace(0,0.3,10000)
        self.Rtes = self.RT_emp(T=self.Ttes,Tc=self.Tc,A=self.A,B=self.A,C=self.C)

        self.result_plot(plot_subject="RT_emp_plot")

    def alpha_plot_bef(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.A = f[f"ch{self.ch}"]["IV"]["analysis"]["RTfit"]["A"][...]
            self.B = f[f"ch{self.ch}"]["IV"]["analysis"]["RTfit"]["B"][...]
            self.C = f[f"ch{self.ch}"]["IV"]["analysis"]["RTfit"]["Offset"][...]
            self.Tc = f[f"ch{self.ch}"]["IV"]["analysis"]["RTfit"]["Tc"][...]
        self.Ttes = np.linspace(0,0.3,1000)
        R = self.RT_emp(self.Ttes,self.Tc,self.A,self.B,self.C)
        self.Alpha = self.Ttes[1:] * np.diff(np.log(R))/(np.diff(np.log(self.Ttes))*self.Rtes[1:])
        self.Ttes = self.Ttes[1:]
        # f = np.genfromtxt("res.txt")
        # self.T  = f[:,0]
        # self.a = f[:,1]
        # self.b = f[:,2]
        print(len(self.Ttes))
        print(len(self.Alpha))

        self.result_plot(plot_subject="alpha_plot")

    def RTI_3D_plot(self):
        x1 = self.T_a[self.R_a>1e-3]*1e+3
        x2 = self.I_a[self.R_a>1e-3]*1e+6
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[np.ravel(X1), np.ravel(X2)]
        X_plot = np.c_[np.ravel(X1), np.ravel(X2)]
        y = self.R_a[self.R_a>1e-3]*1e+3
        print(self.R_a)
        print(x1.shape,x2.shape,y.shape)

        fig = plt.figure(figsize=(18,18))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1,x2,y)
        ax.set_title("T-I-R 3D plot",fontsize=20)
        ax.set_xlabel("Temperature[mK]",fontsize=20)
        ax.set_ylabel("Current[uA]",fontsize=20)
        ax.set_zlabel("Resistance[mOhm]",fontsize=20)

        plt.show()

    def inst_data_out(self,Tbath,Ibias):
        idx = self.getNearValueID(self.data["data"][f"{Tbath}mK"]["Ibias"]*1e+6,Ibias)
        Ttes = self.data["data"][f"{Tbath}mK"]["Ttes"][idx]
        Rtes = self.data["data"][f"{Tbath}mK"]["Rtes"][idx]
        Ites = self.data["data"][f"{Tbath}mK"]["Ites"][idx]
        Vtes = self.data["data"][f"{Tbath}mK"]["Vtes"][idx]
        Pb = self.data["data"][f"{Tbath}mK"]["Pb"][idx]
        Gtes = self.data["data"][f"{Tbath}mK"]["Gtes"][idx]
        print("-----------------------------------------------")
        print(f"Tbath = {Tbath} mK")
        print(f"Ibias = {Ibias} uA")
        print(f"Rtes  = {Rtes*1e+3} mOhm")
        print(f"Ites  = {Ites*1e+6} uA")
        print(f"Vtes  = {Vtes*1e+6} uV")
        print(f"Pb    = {Pb*1e+12} pW")
        print(f"Gtes  = {Gtes*1e+9} nW/K")
        print(f"Ttes  = {Ttes*1e+3} mK")

    def all_analysis(self,Iunit,savehdf5,Ibias,mes_number): 
        self.initialize(Iunit,savehdf5,mes_number)
        self.all_process()
        self.result_plot(plot_subject="IbiasVcor")
        self.result_plot(plot_subject="RtesPb")
        self.result_plot(plot_subject="IbiasRtes")
        self.result_plot(plot_subject="VtesItes")
        self.result_plot(plot_subject="IbiasItes")
        self.Rn_define(per=0.5)
        self.result_plot(plot_subject="RtesPb_with_idx")
        self.Gfit2()
        self.process_Ttes()
        self.result_plot(plot_subject="RtesTtes")
        # self.RT_fit(Tbath=130,model='exp')
        self.save_hdf5()
        # self.alpha_plot()
        # self.Rn_for_Z(Tbath=175,Ibias=1500)
        self.inst_data_out(Tbath=110,Ibias=Ibias)
        # self.stack_data()
        # self.RTI_3D_plot()
        # self.save_RTI_txt(Tbath=90.0)

    def ddump(self,ch,Iunit,savehdf5,Ibias):
        self.initialize(ch,Iunit,savehdf5)
        self.all_process()
        self.Rn_define()
        self.Gfit2()
        self.G_cont(step=200,nmin=3.0,nmax=4.0,G0min=50,G0max=400)
        self.process_Ttes()
        self.inst_data_out(Tbath=90.0,Ibias=Ibias)

    def RTp(self):
        fig = plt.figure(figsize=(10,6))
        ax  = plt.subplot(1,1,1)
        ax.grid(linestyle="dashed")
        B = self.B
        for e,i in enumerate(sorted(B)):
            print(i*2e-3)
            b = round(i*2e-2)
            b = b*0.1

            self.Rtes = self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Rtes"]
            self.Ttes = self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Ttes"]
            if i >= 0: 
                ax.scatter(self.Ttes*1e+3,self.Rtes*1e+3,color=cm.jet(b/10),label=f'{b} uT')
            elif i < 0: 
                ax.scatter(self.Ttes*1e+3,self.Rtes*1e+3,color=cm.jet(-b/10),label=f'{b} uT',marker=',')        
        ax.set_xlabel(r'$T_{TES} \ \rm (mK)$',fontsize=15)
        ax.set_ylabel(r'$R_{TES} \ \rm (m \Omega)$',fontsize=15)

        # spine_width = 2  # スパインの太さ
        # for spine in ax.spines.values():
        #     spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in')
        # ax.set_xlim(150.1,170)
        # ax.set_ylim(0,25)
        plt.legend(fontsize=10)
        plt.show()
        fig.savefig('RT_withB.pdf',dpi=300, transparent=True)


    def alpha_plot(self):
        a_list = []
        a_max = []
        a_min = []
        #Rn = 29.338279e-3
        Rn = 22e-3
        fig = plt.figure(figsize=(10,6))
        ax  = plt.subplot(1,1,1)
        ax.grid(linestyle="dashed")
        B = self.B
        print(sorted(B))


        with h5py.File('Exp240405_A5.hdf5', 'a') as f:
            for e,i in enumerate(sorted(B)):
                Rtes = self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Rtes"]
                Ttes = self.data["data"][f"{self.Tbath[0]}mK"][f"{i}uA"]["Ttes"]
                alpha = self.alpha_diff(Rtes, Ttes)
                b = round(i*2e-2)
                b = b*0.1
                R_ratio = Rtes[1:]/Rn
                mask = (R_ratio > 0.10) & (0.25 > R_ratio) 
                if i >= 0: 
                    ax.scatter(R_ratio,alpha,color=cm.jet(b/10),label=f'{b} uT')
                    ax.plot(R_ratio,alpha,color=cm.jet(b/10))
                elif i < 0: 
                    ax.scatter(R_ratio,alpha,color=cm.jet(-b/10),label=f'{b} uT',marker=',')  
                    ax.plot(R_ratio,alpha,'-.',color=cm.jet(-b/10))  
                a_list.append(np.average(alpha[mask]))
                a_max.append(np.max(alpha[mask]))
                a_min.append(np.min(alpha[mask]))    
                if f'{i}' in f.keys():
                    del f[f'{i}']
                print(f'{i}/Rtes')
                print(f.keys())
                f.create_dataset(f'{i}/Rtes', data=Rtes)   
                f.create_dataset(f'{i}/Ttes', data=Ttes)  
                f.create_dataset(f'{i}/alpha', data=alpha)
        ax.set_xlabel(r'$R_{TES}/R_{Normal}$',fontsize=15)
        ax.set_ylabel(r'$\alpha $',fontsize=15)
        ax.set_ylim(0,175)
        plt.legend(fontsize=10)
        plt.show()
        fig.savefig('RT_withB_alpha.pdf',dpi=300, transparent=True)
        a_list = np.array(a_list)
        a_max = np.array(a_max) - a_list
        a_min = a_list - np.array(a_min) 
        a_err = np.vstack((a_min, a_max))
                


        fig = plt.figure(figsize=(10,6))
        ax  = plt.subplot(1,1,1)
        ax.grid(linestyle="dashed")
        ax.errorbar(np.sort(B)*2.17e-3, a_list, a_err,capsize=5, fmt='.', markersize=12, ecolor="black", elinewidth=1.5, markeredgecolor="black", color="w")
        ax.plot(np.sort(B)*2.17e-3, a_list, color='k')
        ax.set_xlabel(r'$\rm Applied\ Magnetic\ Field \ (\mu T)$',fontsize=15)
        ax.set_ylabel(r'$\alpha$',fontsize=15)
        plt.show()
        fig.savefig('alpha-B.pdf',dpi=300, transparent=True)


    def single_data(self,Iunit,savehdf5,Ibias,mes_number):
        self.initialize(Iunit,savehdf5,mes_number)
        self.all_process()
        self.plot_init()
        # self.G0 = 2.3397e-8
        # self.n = 3.1487
        # self.Tc = 159.3298
        # self.G0 = 1.8026e-10
        # self.Tc = 191.694743
        # self.n = 0.94114197
        self.G0 = 8.0745e-08
        self.n = 3.83905694
        self.Tc = 177.749390
        self.process_Ttes_forB()
        self.RTp()
        self.alpha_plot()

    def dplot(self):
        self.fig = plt.figure(figsize=(10.6,6))
        self.ax  = plt.subplot(1,1,1)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r'$\rm I_{bias} \ (\mu A)$',fontsize=15)
        self.ax.set_ylabel(r'$\rm R_{TES} \ (m \Omega)$',fontsize=15)
        with h5py.File('new.hdf5',"a") as f:
            for e,i in enumerate(f['ch1/IV/data'].keys()):
                Ibias = f[f'ch1/IV/data/{i}/Ibias'][:]
                Rtes = f[f'ch1/IV/data/{i}/Rtes'][:]
                self.ax.scatter(Ibias*1e+6,Rtes*1e+3,label=f"2024/04 {i}",c=cm.jet([e/len(f['ch1/IV/data'].keys())]))

        with h5py.File('old.hdf5',"a") as f:
            for e,i in enumerate(f['ch1/IV/data'].keys()):
                Ibias = f[f'ch1/IV/data/{i}/Ibias'][:]
                Rtes = f[f'ch1/IV/data/{i}/Rtes'][:]
                self.ax.scatter(Ibias*1e+6,Rtes*1e+3,marker=',',label=f"2023/03 {i}",c=cm.jet([e/len(f['ch1/IV/data'].keys())]))

        plt.legend()
        plt.show()