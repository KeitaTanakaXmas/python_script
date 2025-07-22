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
from scipy.interpolate import interp1d
from scipy import integrate,signal

from Basic import Plotter

hd = Hdf5Command()

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.11.21

print('===============================================================================')
print(f"Complex Impedance Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class ZAnalysis:

    def __init__(self):
        self.Vac    =  25 * 1e-3
        self.Rshunt =  3.9 * 1e-3
        self.Min    =  9.82 * 1e-11
        self.Rfb    =  1e+5
        self.Mfb    =  8.5 * 1e-11
        self.Rac    =  1e+4 ## ? Check Magnicon Resistance in the load.
        self.Lac    =  470 * 1e-6
        self.Lcom   =  251 * 1e-9
        self.CRange68 = 2.295815160785974337606
        self.CRange90 = 4.605170185988091368036
        self.CRange99 = 9.210340371976182736072

    def initialize(self,ch,savehdf5,Tbath):
        self.ch = ch
        self.filelist = sorted(glob.glob(f"./../data/Z/raw_data/ch{ch}/*.txt"))
        self.savehdf5 = savehdf5
        self.Tbath = str(float(Tbath))

    def getNearValueID(self,arr, num):
        arr = np.array(arr)
        idx = np.argmin(np.abs(arr - num))
        return idx

## Function List ##

    def Z(self,f,Vout,theta):
        Z = self.Vac*self.Rshunt*self.Min*self.Rfb/(2**(1/2)*Vout*self.Mfb*np.exp(theta*1j*np.pi/180)*(self.Rac))
        return Z

    def Z_err(self,f,Vout,theta,Vout_err,theta_err):
        Z_err_real = self.Z(f,Vout,theta) * np.exp(-1j*theta*np.pi/180) *((Vout_err*np.cos(theta*np.pi/180)/Vout)**2 + ((theta_err*np.pi/180)*np.sin(theta*np.pi/180))**2)**(0.5)
        Z_err_imag = self.Z(f,Vout,theta) * np.exp(-1j*theta*np.pi/180) *((Vout_err*np.sin(theta*np.pi/180)/Vout)**2 + ((theta_err*np.pi/180)*np.cos(theta*np.pi/180))**2)**(0.5)
        return Z_err_real + 1j*Z_err_imag

    def Z_TES(self,f,Vout,theta,S=0,N=800):
        frq_S,vout_S,vouterr_S,theta_S,thetaerr_S = self.Z_select(ibias=S)
        Z_TES = self.Z(f,Vout,theta)/self.Tr(S=S) - (self.Rth+2j*np.pi*f*self.L)
        return Z_TES

    def Z_one_theory(self,f):
        Rtes = 10e-3
        Ttes = 0.15
        Pb = 50e-12
        Gtes = 5e-10
        alpha = 100
        beta = 1
        C = 0.83e-12
        lp = Pb * alpha/(Gtes*Ttes)
        tau = C/(Gtes*(-1+Pb*alpha/(Gtes*Ttes)))
        Z_inf = Rtes*(1+beta)
        Z_0 = -Rtes * (lp+beta+1)/(lp-1)
        Z_fit = Z_inf +(Z_0 - Z_inf)/(1-2j*np.pi*f*tau)
        return Z_fit

    def Z_two_theory(self,f):
        Rtes = 10e-3
        Ttes = 0.15
        Pb = 50e-12
        Gtes = 5e-10
        alpha = 100
        beta = 1
        Cabs = 0.8e-12
        Ctes = 0.03e-12
        Gabs = 0.3e-6
        lp = (Pb*alpha)/(Gtes*Ttes)
        tau = (Ctes+Cabs-Cabs*(2j*np.pi*f*Cabs/Gabs)/(1+2j*np.pi*f*Cabs/Gabs))/(Gtes*(lp-1))
        Z_inf = Rtes*(1+beta)
        Z_0 = -Rtes * (lp+beta+1)/(lp-1)
        Z_fit = Z_inf +(Z_0 - Z_inf)/(1-2j*np.pi*f*tau)
        return Z_fit 

    def Z_TES_err(self,frq,Vout,theta,Vout_err,theta_err,S=0):
        frq_S,vout_S,vouterr_S,theta_S,thetaerr_S = self.Z_select(ibias=S)  
        Rth = self.Rth
        Rth_err = self.Rth_err
        L = self.L
        L_err = self.L_err
        cr = (Rth*self.Z(frq_S,vout_S,theta_S).real+(2*np.pi*frq_S*L*self.Z(frq_S,vout_S,theta_S).imag))
        Tr_z_part_real = (Rth*self.Z_err(frq_S,vout_S,theta_S,vouterr_S,thetaerr_S).real/(Rth**2+(2*np.pi*frq_S*L)**2))**2 + ((2*np.pi*frq_S*L)*self.Z_err(frq_S,vout_S,theta_S,vouterr_S,thetaerr_S).imag/(Rth**2+(2*np.pi*frq_S*L)**2))**2
        Tr_R_part_real = ((self.Z(frq_S,vout_S,theta_S).real/(Rth**2+(2*np.pi*frq_S*L)**2)-2*Rth*(cr)/((Rth**2+(2*np.pi*frq_S*L)**2)**2))*Rth_err)**2
        Tr_L_part_real = (((2*np.pi*frq_S)*self.Z(frq_S,vout_S,theta_S).imag/((Rth**2+(2*np.pi*frq_S*L)**2)) - 2*(2*np.pi*frq_S)**2*L*(cr)/((Rth**2+(2*np.pi*frq_S*L)**2)**2))*L_err)**2
        Tr_err_real = (Tr_z_part_real + Tr_R_part_real + Tr_L_part_real)**(1/2)
        
        cg = (Rth*self.Z(frq_S,vout_S,theta_S).imag-(2*np.pi*frq_S*L*self.Z(frq_S,vout_S,theta_S).real))
        Tr_z_part_imag = (Rth*self.Z_err(frq_S,vout_S,theta_S,vouterr_S,thetaerr_S).imag/(Rth**2+(2*np.pi*frq_S*L)**2))**2 + ((-2*np.pi*frq_S*L)*self.Z_err(frq_S,vout_S,theta_S,vouterr_S,thetaerr_S).real/(Rth**2+(2*np.pi*frq_S*L)**2))**2
        Tr_R_part_imag = ((self.Z(frq_S,vout_S,theta_S).imag/(Rth**2+(2*np.pi*frq_S*L)**2)-2*Rth*(cg)/((Rth**2+(2*np.pi*frq_S*L)**2)**2))*Rth_err)**2
        Tr_L_part_imag = ((-(2*np.pi*frq_S)*self.Z(frq_S,vout_S,theta_S).real/((Rth**2+(2*np.pi*frq_S*L)**2)) - 2*(2*np.pi*frq_S)**2*L*(cg)/((Rth**2+(2*np.pi*frq_S*L)**2)**2))*L_err)**2
        Tr_err_imag = (Tr_z_part_imag + Tr_R_part_imag + Tr_L_part_imag)**(1/2)
        
        Tr_err = Tr_err_real + 1j * Tr_err_imag
        co = self.Z(frq,Vout,theta).real *self.Tr().real + self.Z(frq,Vout,theta).imag *self.Tr().imag
        Z_part_real = (self.Tr().real*self.Z_err(frq,Vout,theta,Vout_err,theta_err).real/(abs(self.Tr())**2))**2 + (self.Tr().imag*self.Z_err(frq,Vout,theta,Vout_err,theta_err).imag/(abs(self.Tr())**2))**2
        Tr_part_real = (((self.Z(frq,Vout,theta).real/(abs(self.Tr()))**2)-2*self.Tr().real*(co)/(abs(self.Tr()))**4)*Tr_err.real)**2 + (((self.Z(frq,Vout,theta).imag/(abs(self.Tr()))**2)-2*self.Tr().imag*(co)/(np.abs(self.Tr()))**4)*Tr_err.imag)**2
        Z_TES_err_real = (Z_part_real + Tr_part_real + Rth_err**2 )**0.5
        cf = self.Z(frq,Vout,theta).imag *self.Tr().real - self.Z(frq,Vout,theta).real *self.Tr().imag
        Z_part_imag = (-self.Tr().imag*self.Z_err(frq,Vout,theta,Vout_err,theta_err).real/(abs(self.Tr())**2))**2 + (self.Tr().real*self.Z_err(frq,Vout,theta,Vout_err,theta_err).imag/(abs(self.Tr())**2))**2
        Tr_part_imag = (((self.Z(frq,Vout,theta).imag/(abs(self.Tr()))**2)-2*self.Tr().real*(cf)/(abs(self.Tr()))**4)*Tr_err.real)**2 + (((-self.Z(frq,Vout,theta).real/(abs(self.Tr()))**2)-2*self.Tr().imag*(cf)/(abs(self.Tr()))**4)*Tr_err.imag)**2
        Z_TES_err_imag = (Z_part_imag + Tr_part_imag + (2j*np.pi*frq*L_err)**2 )**0.5
        
        Z_TES_err = Z_TES_err_real + 1j*Z_TES_err_imag
        
        return Z_TES_err

    def Tr(self,S=0):
        frq_S,vout_S,vouterr_S,theta_S,thetaerr_S = self.Z_select(ibias=S)    
        Tr = self.Z(frq_S,vout_S,theta_S)/(self.Rth+2j*np.pi*frq_S*self.L)
        return Tr

    def Rth_L_func(self,f,L,Rth):
        Tr = (self.Rn+ 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))
        return Tr

    def Rth_L_func_N(self,f,L,Rth):
        Tr = (Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))/(self.Rn+ 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))
        return Tr  

    def Rth_L_func_Lpar(self,f,L,Rth,Lpar):
        Tr = (1/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))+1/(2j*np.pi*f*Lpar))/(1/(self.Rn+ 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))+1/(2j*np.pi*f*Lpar))
        return Tr

    def Rth_L_func_Cpar(self,f,L,Rth,Cpar):
        Tr = (1/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))+(2j*np.pi*f*Cpar))/(1/(self.Rn+ 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))+(2j*np.pi*f*Cpar))
        return Tr

    def Rth_L_func_Rpar(self,f,L,Rth,Rpar):
        Tr = (1/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))+1/Rpar)/(1/(self.Rn+ 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))+1/Rpar)
        return Tr

    def Z_fit_one(self,f,alpha,beta,C):
        lp = self.Pb * alpha/(self.Gtes*self.Ttes)
        tau = C/(self.Gtes*(-1+self.Pb*alpha/(self.Gtes*self.Ttes)))
        Z_inf = self.Rtes*(1+beta)
        Z_0 = -self.Rtes * (lp+beta+1)/(lp-1)
        Z_fit = Z_inf +(Z_0 - Z_inf)/(1-2j*np.pi*f*tau)
        return Z_fit

    def Z_fit_two(self,f,alpha,beta,Cabs,Ctes,Gabs):
        lp = (self.Pb*alpha)/(self.Gtes*self.Ttes)
        tau = (Ctes+Cabs-Cabs*(2j*np.pi*f*Cabs/Gabs)/(1+2j*np.pi*f*Cabs/Gabs))/(self.Gtes*(lp-1))
        Z_inf = self.Rtes*(1+beta)
        Z_0 = -self.Rtes * (lp+beta+1)/(lp-1)
        Z_fit = Z_inf +(Z_0 - Z_inf)/(1-2j*np.pi*f*tau)
        return Z_fit 

    def Z_fit_three(f,alpha,beta,Cabs,Ctes,Gabs,Cpath,Gpath):
        A = Gpath/(-2j*np.pi*f*Cpath-Gabs*(1-Cabs/(Gabs+2j*np.pi*f*Cabs)+Glink+Gpath))
        B = 2j*np.pi*f+(-Gpath*(1-A)-Gtes)/Ctes-alpha*Pb/(Ctes*T)
        Z = R0*(1+beta)+Pb*R0*alpha*(2+beta)/(Ctes*T*B)
        return Z

    def R_ta(self,T):
        R0 = 6.7e-3
        I0 = 95.6e-6 
        T0 = 164e-3
        self.Rta = lambda T: R0 + self.alpha_int(T)*R0*(T-T0)/T0
        # + beta_int*R_0*(I-I0)/I0

    def R_interp(self,T,I,T0,I0,R0):
        return R0*(1+(T-T0)*self.alpha_int(T)/T0+(I-I0)*self.beta_int(T)/I0)

    def ploter(self):
        self.R_ta(self.T)
        def int_Rta(T):
            return integrate.quad(self.Rta,np.min(self.Ttes),T)
        print(self.T)
        for i in self.T:
            plt.plot(i,int_Rta(i)[0],".")
        plt.show()

## Rise Fall Time  ##

    def lp_func(self):
        self.lp      = self.Ites**2*self.Rtes*self.alpha/(self.Gtes*self.Ttes)    

    def tau_func(self):
        self.tau     = self.C/self.Gtes

    def tau_I_func(self):
        self.tau_I   = self.tau/(1-self.lp)

    def tau_el_func(self):
        self.tau_el  = self.L/(self.Rth+self.Rtes*(1+self.beta))

    def tau_rise_func(self):
        self.tau_rise = 1/(1/(2*self.tau_el) +1/(2*self.tau_I) + np.sqrt((1/self.tau_el - 1/self.tau_I)**2 - 4*self.Rtes*self.lp*(2+self.beta)/(self.L*self.tau))/2)

    def tau_fall_func(self):
        self.tau_fall = 1/(1/(2*self.tau_el) +1/(2*self.tau_I) - np.sqrt((1/self.tau_el - 1/self.tau_I)**2 - 4*self.Rtes*self.lp*(2+self.beta)/(self.L*self.tau))/2)  

    def LcritM_func(self):
        self.LcritM = self.Rtes*self.tau*(self.lp*(3+self.beta-self.Rth/self.Rtes)+(1+self.beta+self.Rth/self.Rtes)-2*np.sqrt(self.lp*(2+self.beta)*(self.lp*(1-self.Rth/self.Rtes)+(1+self.beta+self.Rth/self.Rtes))))/(self.lp-2)**2

    def LcritP_func(self):
        self.LcritP = self.Rtes*self.tau*(self.lp*(3+self.beta-self.Rth/self.Rtes)+(1+self.beta+self.Rth/self.Rtes)+2*np.sqrt(self.lp*(2+self.beta)*(self.lp*(1-self.Rth/self.Rtes)+(1+self.beta+self.Rth/self.Rtes))))/(self.lp-2)**2


    def func_call(self):
        self.lp_func()
        self.tau_func()
        self.tau_I_func()
        self.tau_el_func()
        self.tau_rise_func()
        self.tau_fall_func()
        self.LcritM_func()
        self.LcritP_func()

## Output Data ##

    def data_out(self):
        self.data = {}
        for e,i in enumerate(self.filelist):
            f          = np.genfromtxt(i)
            frq        = f[:,0]
            vout       = f[:,1]
            vout_err   = f[:,2]
            theta      = f[:,3]
            theta_err  = f[:,4]
            ibias      = int(float(re.findall(r"[-+]?\d*\.\d+|\d+",open(i).readlines()[2])[0]))
            print(ibias)
            self.data[f"{ibias}"]              = {} 
            self.data[f"{ibias}"]["Frq"]       = frq
            self.data[f"{ibias}"]["Vout"]      = vout
            self.data[f"{ibias}"]["Vout_err"]  = vout_err
            self.data[f"{ibias}"]["Theta"]     = theta
            self.data[f"{ibias}"]["Theta_err"] = theta_err
        self.Ibias_list = np.array(list(self.data.keys()),dtype="int")
        self.fit_bias_list = np.delete(self.Ibias_list,0,axis=0)
        self.fit_bias_list = np.delete(self.fit_bias_list,-1,axis=0)

    def ZTES_data_out(self):
        for e,i in enumerate(self.fit_bias_list):
            self.ibias = i
            self.Set_IV_parameter()
            self.Fit_Z_def()
            self.data[f"{self.ibias}"]["fit"]              = {}
            self.data[f"{self.ibias}"]["fit"]["Frq"]       = self.f
            self.data[f"{self.ibias}"]["fit"]["ZTES"]      = self.z
            self.data[f"{self.ibias}"]["fit"]["ZTES_err"]  = self.z_err

    def save_hdf5(self):
        with h5py.File(self.savehdf5,"a") as f:
            if f"ch{self.ch}" in f.keys():
                if "Z" in f[f"ch{self.ch}"].keys():
                    del f[f"ch{self.ch}"] 
            for i in self.data.keys():
                f.create_dataset(f"ch{self.ch}/Z/data/{i}uA/Frq",data=self.data[i]["Frq"])
                f.create_dataset(f"ch{self.ch}/Z/data/{i}uA/Vout",data=self.data[i]["Vout"])
                f.create_dataset(f"ch{self.ch}/Z/data/{i}uA/Vout_err",data=self.data[i]["Vout_err"])
                f.create_dataset(f"ch{self.ch}/Z/data/{i}uA/Theta",data=self.data[i]["Theta"])
                f.create_dataset(f"ch{self.ch}/Z/data/{i}uA/Theta_err",data=self.data[i]["Theta_err"])

    def save_Rth_L(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "Z" in f[f"ch{self.ch}"].keys():
                if "analysis" in f[f"ch{self.ch}"]["Z"].keys():
                    if "Rth_L_fit" in f[f"ch{self.ch}/Z/analysis"].keys():
                        del f[f"ch{self.ch}/Z/analysis/Rth_L_fit"]
            print("------------------------------------------------------------------------")
            print(f"Thevenin Resistance in the TES current Line = {self.Rth} [Ohm]")
            print(f"Thevenin Resistance  error in the TES current Line = {self.Rth_err} [Ohm]")
            print(f"Inductance of the Input Coil = {self.L} [H]")
            print(f"Inductance error of the Input Coil = {self.L_err} [H]")
            print("Save following value ...")
            print(f"Savefile = {self.savehdf5}")
            f.create_dataset(f"ch{self.ch}/Z/analysis/Rth_L_fit/Rth",data=self.Rth)        
            f.create_dataset(f"ch{self.ch}/Z/analysis/Rth_L_fit/Rth_err",data=self.Rth_err)
            f.create_dataset(f"ch{self.ch}/Z/analysis/Rth_L_fit/L",data=self.L)
            f.create_dataset(f"ch{self.ch}/Z/analysis/Rth_L_fit/L_err",data=self.L_err)

    def load_Rth_L(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.Rth = f[f"ch{self.ch}/Z/analysis/Rth_L_fit/Rth"][...]
            self.Rth_err = f[f"ch{self.ch}/Z/analysis/Rth_L_fit/Rth_err"][...]
            self.L = f[f"ch{self.ch}/Z/analysis/Rth_L_fit/L"][...]          
            self.L_err = f[f"ch{self.ch}/Z/analysis/Rth_L_fit/L_err"][...]
            print("------------------------------------------------------------------------")
            print("Load hdf5file value ...")
            print(f"Loadfile = {self.savehdf5}")
            print(f"Thevenin Resistance in the TES current Line = {self.Rth} [Ohm]")
            print(f"Thevenin Resistance  error in the TES current Line = {self.Rth_err} [Ohm]")
            print(f"Inductance of the Input Coil = {self.L} [H]")
            print(f"Inductance error of the Input Coil = {self.L_err} [H]")

    def Z_select(self,ibias):
        frq       = self.data[f"{ibias}"]["Frq"]
        vout      = self.data[f"{ibias}"]["Vout"]
        vout_err  = self.data[f"{ibias}"]["Vout_err"]
        theta     = self.data[f"{ibias}"]["Theta"]
        theta_err = self.data[f"{ibias}"]["Theta_err"]
        return frq,vout,vout_err,theta,theta_err

    def Z_div_out(self,N=800,S=0):
        frq_N, Vout_N, Vouterr_N, Theta_N, Thetaerr_N = self.Z_select(ibias=N)
        frq_S, Vout_S, Vouterr_S, Theta_S, Thetaerr_S = self.Z_select(ibias=S)  
        N_real = self.Z(frq_N,Vout_N,Theta_N).real
        N_imag = self.Z(frq_N,Vout_N,Theta_N).imag
        S_real = self.Z(frq_S,Vout_S,Theta_S).real
        S_imag = self.Z(frq_S,Vout_S,Theta_S).imag
        sig_N_real = self.Z_err(frq_N,Vout_N,Theta_N,Vouterr_N,Thetaerr_N).real
        sig_N_imag = self.Z_err(frq_N,Vout_N,Theta_N,Vouterr_N,Thetaerr_N).imag
        sig_S_real = self.Z_err(frq_S,Vout_S,Theta_S,Vouterr_S,Thetaerr_S).real
        sig_S_imag = self.Z_err(frq_S,Vout_S,Theta_S,Vouterr_S,Thetaerr_S).imag
        over = S_real**2 + S_imag**2
        ov_real = N_real*S_real - N_imag*S_imag
        ov_imag = N_real*S_imag + N_imag*S_real
        Re_part = ((S_real*sig_N_real/over)**2 + (-S_imag*sig_N_imag/over)**2 + ((N_real/over-2*S_real*ov_real/over**2)*sig_S_real)**2 + ((-N_imag/over+2*S_imag*ov_real/over**2)*sig_S_imag)**2)**(1/2)
        Im_part = ((S_imag*sig_N_real/over)**2 + (-S_real*sig_N_imag/over)**2 + ((N_imag/over-2*S_real*ov_imag/over**2)*sig_S_real)**2 + ((N_real/over+2*S_imag*ov_imag/over**2)*sig_S_imag)**2)**(1/2)
        self.Zdiv = self.Z(frq_N,Vout_N,Theta_N)/self.Z(frq_S,Vout_S,Theta_S)
        self.Zdiv_err = Re_part + 1j*Im_part
        self.frq = frq_N
        self.Zn = self.Z(frq_N,Vout_N,Theta_N)
        self.Zs = self.Z(frq_S,Vout_S,Theta_S)

    def IV_result(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.Ibias = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ibias"][:]            
            self.ites = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ites"][:]
            self.vtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Vtes"][:]
            self.rtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Rtes"][:]
            self.pb   = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Pb"][:]
            self.gtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Gtes"][:]
            self.ttes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ttes"][:]
            self.Rn   = f[f"ch{self.ch}"]["IV"]["analysis"]["Rn_Z"][...]

    def Match_bias(self):
        self.IV_result()
        for e,i in enumerate(self.Ibias_list):
            idx = self.getNearValueID(self.Ibias,i*1e-6)
            if e == 0:
                self.Ites = np.array(self.ites[idx])
                self.Vtes = np.array(self.vtes[idx])
                self.Rtes = np.array(self.rtes[idx])
                self.Pb   = np.array(self.pb[idx])
                self.Gtes = np.array(self.gtes[idx])
                self.Ttes = np.array(self.ttes[idx])
            else:
                self.Ites = np.append(self.Ites,self.ites[idx])
                self.Vtes = np.append(self.Vtes,self.vtes[idx])
                self.Rtes = np.append(self.Rtes,self.rtes[idx])
                self.Pb   = np.append(self.Pb,self.pb[idx])
                self.Gtes = np.append(self.Gtes,self.gtes[idx])
                self.Ttes = np.append(self.Ttes,self.ttes[idx])

    def Set_IV_parameter(self):
        self.Match_bias()
        idx = self.getNearValueID(self.Ibias_list,self.ibias)
        self.Ites = self.Ites[idx]
        self.Rtes = self.Rtes[idx]
        self.Pb = self.Pb[idx]
        self.Gtes = self.Gtes[idx]
        self.Ttes = self.Ttes[idx]
        print("------------------------------------------------------------------------")
        print("Data property")
        print(f"Ibias = {self.Ibias_list[idx]} uA")
        print(f"TES Resistance = {self.Rtes} Ohm")
        print(f"Joule Heat = {self.Pb} W/K")
        print(f"Gtes = {self.Gtes} W/K")
        print(f"TES Temperature = {self.Ttes} K")

    def Fit_Z_def(self,S=0):     
        frq,vout,vout_err,theta,theta_err = self.Z_select(ibias=self.ibias)
        z = self.Z_TES(frq,vout,theta,S=S)
        z_err = self.Z_TES_err(frq,vout,theta,vout_err,theta_err)
        mask = (z_err.real<1) & (frq != 50)
        self.z = z[mask]
        self.f = frq[mask]
        self.z_err = z_err[mask]


## Plot function ##

    def plot_init(self):
        #plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap'] = 'jet'
        plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        plt.rcParams["font.size"] = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize'] = 30 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize'] = 30 # 軸だけ変更されます
        plt.rcParams['xtick.direction'] = 'in' # x axis in
        plt.rcParams['ytick.direction'] = 'in' # y axis in 
        plt.rcParams['axes.linewidth'] = 1.0 # axis line width
        plt.rcParams['axes.grid'] = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['scatter.edgecolors'] = 'black'
        self.fs = 40
        self.ps = 45

    def plot_window(self,style):

        self.plot_init()

        if style == "ZRI":
            self.fig = plt.figure(figsize=(24,8))
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.ax1.grid(linestyle="dashed")
            self.ax1.set_xlabel(r"$ \rm Re(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Im(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax2.grid(linestyle="dashed")
            self.ax2.semilogx()
            self.ax2.set_title("Frequency vs Re(Z) plot",fontsize=self.fs)
            self.ax2.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$ \rm Re(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax3.grid(linestyle="dashed")
            self.ax3.semilogx()
            self.ax3.set_title("Frequency vs Im(Z) plot",fontsize=self.fs)
            self.ax3.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax3.set_ylabel(r"$ \rm Im(Z) \ (m\Omega)$",fontsize=self.fs)

        if style == "Tr_p":
            self.fig = plt.figure(figsize=(24,8))
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.ax1.grid(linestyle="dashed")
            self.ax1.set_xlabel(r"$ \rm Re(Tr(f)) \ $",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Im(Tr(f)) \ $",fontsize=self.fs)
            self.ax2.grid(linestyle="dashed")
            self.ax2.semilogx()
            self.ax2.set_title("Frequency vs Re(Tr(f))",fontsize=self.fs)
            self.ax2.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$ \rm Re(Tr(f)) \ $",fontsize=self.fs)
            self.ax3.grid(linestyle="dashed")
            self.ax3.semilogx()
            self.ax3.set_title("Frequency vs Im(Tr(f))",fontsize=self.fs)
            self.ax3.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax3.set_ylabel(r"$ \rm Im(Tr(f)) \ $",fontsize=self.fs)               

        if style == "ZRrIr":
            self.fig = plt.figure(figsize=(24,8))
            gs  = GridSpec(nrows=2,ncols=3,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=3,subplot_spec=gs[:,0])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=3,subplot_spec=gs[0,1])
            gs3 = GridSpecFromSubplotSpec(nrows=2,ncols=3,subplot_spec=gs[1,1])
            gs4 = GridSpecFromSubplotSpec(nrows=2,ncols=3,subplot_spec=gs[0,2])
            gs5 = GridSpecFromSubplotSpec(nrows=2,ncols=3,subplot_spec=gs[1,2])
            self.ax1 = self.fig.add_subplot(gs1[:,:])
            self.ax2 = self.fig.add_subplot(gs2[:,:])
            self.ax3 = self.fig.add_subplot(gs3[:,:],sharex=self.ax2)
            self.ax4 = self.fig.add_subplot(gs4[:,:])
            self.ax5 = self.fig.add_subplot(gs5[:,:],sharex=self.ax4)
            self.ax1.grid(linestyle="dashed")
            #self.ax1.set_title(r"$\rm Corrected \ TES \ Impedance$",fontsize=self.fs)
            self.ax1.set_xlabel(r"$ \rm Re(Z_{TES}) \ (m\Omega)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Im(Z_{TES}) \ (m\Omega)$",fontsize=self.fs)
            self.ax2.grid(linestyle="dashed")
            self.ax2.semilogx()
            self.ax2.set_ylabel(r"$ \rm Re(Z_{TES}) \ (m\Omega)$",fontsize=self.fs)
            self.ax3.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax3.grid(linestyle="dashed")
            self.ax3.semilogx()
            self.ax3.set_ylabel(r"$ \rm Residual  \ (m\Omega)$",fontsize=self.fs)
            self.ax5.set_ylabel(r"$ \rm Residual  \ (m\Omega)$",fontsize=self.fs)
            self.ax4.grid(linestyle="dashed")
            self.ax4.semilogx()
            self.ax4.set_ylabel(r"$ \rm Im(Z_{TES}) \ (m\Omega)$",fontsize=self.fs)            
            self.ax5.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax5.semilogx()
            self.ax5.grid(linestyle="dashed")


        if style == "twoblock_result":
            self.fig = plt.figure(figsize=(12,12))
            self.ax1 = self.fig.add_subplot(321)
            self.ax2 = self.fig.add_subplot(322)
            self.ax3 = self.fig.add_subplot(323)
            self.ax4 = self.fig.add_subplot(324)
            self.ax5 = self.fig.add_subplot(325)
            self.ax6 = self.fig.add_subplot(326)
            self.ax1.set_ylabel("Alpha",fontsize=self.fs)
            self.ax2.set_ylabel("Beta",fontsize=self.fs)
            self.ax3.set_ylabel("Cabs[pJ/K]",fontsize=self.fs)
            self.ax4.set_ylabel("Ctes[pJ/K]",fontsize=self.fs)
            self.ax5.set_ylabel("Gtes[nW/K]",fontsize=self.fs)
            self.ax6.set_ylabel("Beta",fontsize=self.fs)
            self.ax6.set_xlabel("Alpha",fontsize=self.fs)

        if style == "oneblock_result":
            # self.fig = plt.figure(figsize=(12,12))
            # self.ax1 = self.fig.add_subplot(221)
            # self.ax2 = self.fig.add_subplot(222)
            # self.ax3 = self.fig.add_subplot(223)
            # self.ax4 = self.fig.add_subplot(224)
            self.fig = plt.figure(figsize=(24,8))
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.ax1.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax3.grid(linestyle="dashed")
            self.ax1.set_ylabel(r"$\alpha_{I}$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$\beta_{I}$",fontsize=self.fs)
            self.ax3.set_ylabel("C[pJ/K]",fontsize=self.fs)
            self.ax1.set_xlabel("R/Rn",fontsize=self.fs)
            self.ax2.set_xlabel("R/Rn",fontsize=self.fs)
            self.ax3.set_xlabel("R/Rn",fontsize=self.fs)
            #self.ax4.set_ylabel("Beta",fontsize=self.fs)
            #self.ax4.set_xlabel("Alpha",fontsize=self.fs)

        if style == "rise_fall":
            self.fig = plt.figure(figsize=(16,8))
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            self.ax1.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax1.set_ylabel(r"$\rm rise \ time \ (\mu s)$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$\rm fall \ time \ (\mu s)$",fontsize=self.fs)
            self.ax1.set_xlabel(r"$R_{\rm TES}/R_{\rm Normal}$",fontsize=self.fs)
            self.ax2.set_xlabel(r"$R_{\rm TES}/R_{\rm Normal}$",fontsize=self.fs)

        if style == "one_cont":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "RT":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(rf"Temperature \ (mK)",fontsize=self.fs)
            self.ax.set_ylabel(rf"$\rm Resistance \ (m\Omega)$",fontsize=self.fs) 

        if style == "alpha_interp":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(rf"Temperature (mK)",fontsize=self.fs)
            self.ax.set_ylabel(rf"$ \alpha_I $",fontsize=self.fs)    

        if style == "beta_interp":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(rf"Temperature (mK)",fontsize=self.fs)
            self.ax.set_ylabel(rf"$ \beta_I $",fontsize=self.fs)

        if style == "Lcrit":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(rf"$\rm Resistance \ (m\Omega)$",fontsize=self.fs)
            self.ax.set_ylabel(rf"$\rm Inductance \ (nH) $",fontsize=self.fs)

        if style == "twinx":
            self.fig = plt.figure(figsize=(9,8))
            self.ax1 = plt.subplot(111)
            self.ax2 = self.ax1.twinx()
            self.ax1.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax1.set_ylabel(rf"$\alpha_I$",fontsize=self.fs)
            self.ax1.set_xlabel(rf"$R/R_N$",fontsize=self.fs)
            self.ax2.set_ylabel(rf"$\beta_I$",fontsize=self.fs)

    def result_plot(self,plot_subject):

        self.plot_init()

        if plot_subject == "Z" or plot_subject == "ZTES":
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"
            l = len(self.Ibias_list)
            self.plot_window(style="ZRI")
            for e,Ibias in enumerate(self.Ibias_list) :
                frq,vout,vout_err,theta,theta_err = self.Z_select(ibias=Ibias)
                if plot_subject == "Z":
                    z = self.Z(frq,vout,theta)
                    sfn = f"./graph/Z/ch{self.ch}_rawZ_plot.png"
                    self.ax1.set_title("Raw Impedance",fontsize=self.fs)
                elif plot_subject == "ZTES":
                    z = self.Z_TES(frq,vout,theta)                
                    sfn = f"./graph/Z/ch{self.ch}_corZ_plot.png"
                    self.ax1.set_title("Corrected TES Impedance",fontsize=self.fs)
                frq = frq[z.imag<10]
                z = z[z.imag<10]
                if Ibias == 800:
                    self.ax1.scatter(z.real*1e+3,z.imag*1e+3,label=f"210mK 0uA",c=cm.jet([e/l]),s=self.ps)
                    self.ax2.scatter(frq,z.real*1e+3,label=f"210mK 0uA",c=cm.jet([e/l]),s=self.ps)
                    self.ax3.scatter(frq,z.imag*1e+3,label=f"210mK 0uA",c=cm.jet([e/l]),s=self.ps)
                else:                    
                    self.ax1.scatter(z.real*1e+3,z.imag*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
                    self.ax2.scatter(frq,z.real*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
                    self.ax3.scatter(frq,z.imag*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
                self.ax3.legend(loc='best',fontsize=20)

        if plot_subject == "Z_theory":
            self.plot_window(style="ZRI")
            frq = np.hstack((np.linspace(0,1000,10000),np.linspace(1000,1000000,10000)))
            z = self.Z_one_theory(f=frq)
            self.ax1.plot(z.real*1e+3,z.imag*1e+3,color="Blue",lw=5)
            self.ax2.plot(frq,z.real*1e+3,color="Blue",lw=5)
            self.ax3.plot(frq,z.imag*1e+3,color="Blue",lw=5)
            z = self.Z_two_theory(f=frq)
            self.ax1.plot(z.real*1e+3,z.imag*1e+3,color="Red",lw=5)
            self.ax2.plot(frq,z.real*1e+3,color="Red",lw=5)
            self.ax3.plot(frq,z.imag*1e+3,color="Red",lw=5)
            sfn = f"./graph/Z/{plot_subject}.png"            

        if plot_subject == "sel_bias":
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"
            l = len(self.sel_bias)
            self.plot_window(style="ZRI")
            for e,Ibias in enumerate(self.sel_bias) :
                frq,vout,vout_err,theta,theta_err = self.Z_select(ibias=Ibias)
                z = self.Z(frq,vout,theta)
                self.ax1.set_title("Raw Impedance",fontsize=self.fs)
                frq = frq[z.imag<10]
                z = z[z.imag<10]
                self.ax1.scatter(z.real*1e+3,z.imag*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
                self.ax2.scatter(frq,z.real*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
                self.ax3.scatter(frq,z.imag*1e+3,label=f"{Ibias}uA",c=cm.jet([e/l]),s=self.ps)
            self.ax1.legend(loc='best',fontsize=15)            

        if plot_subject == "Rth_L_fit" or plot_subject == "oneblock" or plot_subject == "twoblock":
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"
            l = len(self.Ibias_list)
            if plot_subject == "oneblock" or plot_subject == "twoblock":
                self.z = self.z * 1e+3
                self.z_err = self.z_err * 1e+3
                self.z_fit = self.z_fit * 1e+3
                self.z_fit_p = self.z_fit_p * 1e+3

                sfn = f"./graph/Z/ch{self.ch}_{plot_subject}_{self.ibias}uA.png"

            self.plot_window(style="ZRrIr")
            self.ax1.errorbar(self.z.real,self.z.imag,xerr=self.z_err.real,yerr=self.z_err.imag,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black")
            self.ax1.plot(self.z_fit.real,self.z_fit.imag,"-",color="red")
            self.ax2.scatter(self.f,self.z.real,c="black",s=self.ps,label="data")
            self.ax2.plot(self.f_fit,self.z_fit.real,"-",label="fit result",color="red")
            self.ax3.errorbar(self.f,self.z_fit_p.real - self.z.real,yerr=self.z_err.real,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black",label="data")
            self.ax4.plot(self.f_fit,self.z_fit.imag,"-",label="fit result",color="red")
            self.ax4.scatter(self.f,self.z.imag,c="black",s=self.ps,label="data")
            self.ax5.errorbar(self.f,self.z_fit_p.imag - self.z.imag,yerr=self.z_err.imag,markeredgecolor = "black", color='black',markersize=6,fmt="o",ecolor="black",label="data")
            self.fig.subplots_adjust(hspace=.0)

        if plot_subject == "oneblock_result":
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"
            l = len(self.Ibias_list)
            self.plot_window(style="oneblock_result")
            # self.ax1.errorbar(self.Rpar,self.Alpha,yerr=self.Alpha_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "Red", color='w')
            # self.ax2.errorbar(self.Rpar,self.Beta,yerr=self.Beta_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "Blue", color='w')
            # self.ax3.errorbar(self.Rpar,self.C*1e+12,yerr=self.C_err*1e+12, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "Green", color='w')
            # self.ax4.errorbar(self.Alpha,self.Beta,xerr=self.Alpha_err,yerr=self.Beta_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax1.scatter(self.Rpar,self.Alpha,c="Red",s=200)
            self.ax2.scatter(self.Rpar,self.Beta,c="Blue",s=200)
            self.ax3.scatter(self.Rpar,self.C*1e+12,c="Green",s=200)
            #self.ax3.plot(self.Rpar,np.array([0.077]*len(self.Rpar)),"-.",color="black")
            self.ax3.plot(self.Rpar,np.array([0.825]*len(self.Rpar)),"-.",color="black")

        if plot_subject == "twoblock_result":
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"
            self.plot_window(style="twoblock_result")
            self.ax1.errorbar(self.Rpar,self.Alpha,yerr=self.Alpha_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax2.errorbar(self.Rpar,self.Beta,yerr=self.Beta_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax3.errorbar(self.Rpar,self.Cabs*1e+12,yerr=self.Cabs_err*1e+12, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax4.errorbar(self.Rpar,self.Ctes*1e+12,yerr=self.Ctes_err*1e+12, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax5.errorbar(self.Rpar,self.Gabs*1e+9,yerr=self.Gabs_err*1e+9, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
            self.ax6.errorbar(self.Alpha,self.Beta,xerr=self.Alpha_err,yerr=self.Beta_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')

        if plot_subject == "one_cont":
            self.plot_window(style="one_cont")
            self.ax.contour(self.step_x,self.step_y,self.delc,levels=[self.CRange68,self.CRange90,self.CRange99],colors=["red","green","blue"])
            self.ax.scatter(self.step_x[self.mindelc_idx[1]],self.step_y[self.mindelc_idx[0]],marker="+",color="black",s=200)
            sfn = f"./graph/Z/ch{self.ch}_{self.ibias}uA_oneblock_cont_{self.xname}_{self.yname}.png"

        if plot_subject == "rise_fall":
            self.plot_window(style="rise_fall")
            self.ax1.scatter(self.Rpar,self.tau_rise*1e+6,c="Red",s=200,label="rise time")
            self.ax1.scatter(self.Rpar,self.tau_el*1e+6,c="Black",s=200,label="electrical time")
            self.ax2.scatter(self.Rpar,self.tau_fall*1e+6,c="Blue",s=200)     
            self.ax1.legend(loc='best',fontsize=15)      
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"

        if plot_subject == "alpha_interp":
            self.plot_window(style="alpha_interp")
            self.ax.scatter(self.Ttes*1000,self.alpha,s=200,c="Red",label="data")
            self.ax.plot(self.T*1000,self.alpha_int(self.T),color="Black",label="interp")
            self.ax.legend(loc="best",fontsize=15)
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"

        if plot_subject == "beta_interp":
            self.plot_window(style="beta_interp")
            self.ax.scatter(self.Ttes*1000,self.beta,s=200,c="Blue",label="data")
            self.ax.plot(self.T*1000,self.beta_int(self.T),color="Black",label="interp")
            self.ax.legend(loc="best",fontsize=15)
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"

        if plot_subject == "Lcrit":
            self.plot_window(style="Lcrit")
            self.ax.plot(self.Rtes*1e+3,self.LcritM*1e+9,color="Blue",label=rf"$\rm Lcrit_-$",lw=2)
            #self.ax.scatter(self.Rtes*1e+3,self.LcritM*1e+9,s=200,c="Blue")
            self.ax.plot(self.Rtes*1e+3,self.LcritP*1e+9,color="Red",label=rf"$\rm Lcrit_+$",lw=2)
            #self.ax.scatter(self.Rtes*1e+3,self.LcritP*1e+9,s=200,c="Red")
            self.ax.semilogy()
            self.ax.fill(np.append(self.Rtes*1e+3,self.Rtes[::-1]*1e+3),np.append(self.LcritP*1e+9,self.LcritM[::-1]*1e+9),fc="w",hatch="/")
            self.ax.legend(fontsize=15)
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"

        if plot_subject == "alpha_beta":
            self.plot_window(style="twinx")
            self.ax1.scatter(self.Rpar,self.alpha,color="Red",label=rf"$\alpha_I$",s=200)
            self.ax1.plot(self.Rpar,self.alpha,color="Red",lw=2)
            self.ax2.scatter(self.Rpar,self.beta,color="Blue",label=rf"$\beta_I$",s=200)
            self.ax2.plot(self.Rpar,self.beta,color="Blue",lw=2)
            self.ax1.legend(loc="upper right",bbox_to_anchor=(1.0,1.0),fontsize=25)
            self.ax2.legend(loc="upper right",bbox_to_anchor=(1.0,0.9),fontsize=25)
            sfn = f"./graph/Z/ch{self.ch}_{plot_subject}.png"

        self.fig.tight_layout()
        print(sfn)
        self.fig.savefig(sfn,dpi=300)
        plt.show()

    def Tr_plot(self,S=0):
        self.plot_window(style="Tr_p")
        l   = len(self.Ibias_list)
        frq,vout,vout_err,theta,theta_err = self.Z_select(ibias=S)
        z   = self.Tr()                
        sfn = f"./graph/ch{self.ch}_Tr_plot.png"
        self.ax1.set_title("Transfer function",fontsize=self.fs)
        frq = frq[z.imag<10]
        z   = z[z.imag<10]
        self.ax1.scatter(z.real,z.imag,s=60,label='TMU542')
        self.ax2.scatter(frq,z.real,s=60)
        self.ax3.scatter(frq,z.imag,s=60)
        with h5py.File('Ztrfunc_JAXA120Ea4-E5.hdf5',"r") as f:
            Zreal = f['data']['tr_real'][:]
            Zimag = f['data']['tr_real'][:]
            tr_frq = f['data']['f'][:]
            Z = Zreal + 1j * Zimag
        print(Z)
        self.ax1.scatter(Z.real,Z.imag,s=60,label='JAXA120')
        self.ax2.scatter(tr_frq,Z.real,s=60)
        self.ax3.scatter(tr_frq,Z.imag,s=60)
        self.ax1.legend(loc='best',fontsize=15)
        self.fig.tight_layout()
        self.fig.savefig(sfn,dpi=300)
        plt.show() 

## Fitting Style ##         

    def lmfit_style(self,fmod):
        print("------------------------------------------------------------------------")
        print("Fitting result")
        if fmod == "oneblock":
            self.model = Model(self.Z_fit_one)            
            self.model.set_param_hint('C',min=1e-16,max=1e-11)
            self.model.set_param_hint('alpha',min=0,max=1000)
            self.model.set_param_hint('beta',min=0,max=30)

            result = self.model.fit(self.z,f=self.f,weights=1/self.z_err,alpha=80,beta=0.1,C=1e-13)
            print(result.fit_report())
            self.fit_res = np.array([result.best_values["alpha"],result.best_values["beta"],result.best_values["C"]])
            self.fit_res_err = np.array([result.params['alpha'].stderr,result.params['beta'].stderr,result.params['C'].stderr])
            self.f_fit = np.linspace(10,1.05e+5,100000)
            self.z_fit = self.Z_fit_one(self.f_fit,*self.fit_res)
            self.z_fit_p = self.Z_fit_one(self.f,*self.fit_res)

        if fmod == "one_cont":
            self.model = Model(self.Z_fit_one)
            self.params = self.model.make_params()
            print(self.params.keys())
            for ee,x_e in enumerate(self.step_x):
                for e,y_e in enumerate(self.step_y):
                    if "C" in self.xname:
                        C_e = x_e
                        C_fix = False
                    elif "C" in self.yname:
                        C_e = y_e
                        C_fix = False
                    else:
                        C_e = self.fit_res[2]
                        C_fix = True

                    if "alpha" in self.xname:
                        alpha_e = x_e
                        alpha_fix = False
                    elif "alpha" in self.yname:
                        alpha_e = y_e
                        alpha_fix = False
                    else:
                        alpha_e = self.fit_res[0]
                        alpha_fix = True

                    if "beta" in self.xname:
                        beta_e = x_e
                        beta_fix = False
                    elif "beta" in self.yname:
                        beta_e = y_e
                        beta_fix = False
                    else:
                        beta_e = self.fit_res[1]
                        beta_fix = True

                    #print(C_e,alpha_e,beta_e)
                    self.model.set_param_hint('C',min=1e-16,max=1e-11,vary=C_fix)
                    self.model.set_param_hint('alpha',min=0,max=1000,vary=alpha_fix)
                    self.model.set_param_hint('beta',min=0,max=30,vary=beta_fix)

                    result = self.model.fit(self.z,f=self.f,weights=1/self.z_err,alpha=alpha_e,beta=beta_e,C=C_e)
                    #print(result.fit_report())
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


        if fmod == "twoblock":
            self.model = Model(self.Z_fit_two) 
            self.model.set_param_hint('Cabs',min=1e-16,max=1e-11,vary=False)
            self.model.set_param_hint('Ctes',min=1e-16,max=1e-11,vary=False)
            self.model.set_param_hint('Gabs',min=1e-12,max=10)
            self.model.set_param_hint('alpha',min=0,max=1000)
            self.model.set_param_hint('beta',min=0,max=30)

            result = self.model.fit(self.z,f=self.f,weights=1/self.z_err,alpha=80,beta=0.2,Cabs=0.84e-12*1.4,Ctes=0.048e-12,Gabs=9e-3)
            print(result.fit_report())
            self.fit_res = np.array([result.best_values["alpha"],result.best_values["beta"],result.best_values["Cabs"],result.best_values["Ctes"],result.best_values["Gabs"]])
            self.fit_res_err = np.array([result.params['alpha'].stderr,result.params['beta'].stderr,result.params['Cabs'].stderr,result.params['Ctes'].stderr,result.params['Gabs'].stderr])
            self.f_fit = np.linspace(10,1.05e+5,100000)
            self.z_fit = self.Z_fit_two(self.f_fit,*self.fit_res)
            self.z_fit_p = self.Z_fit_two(self.f,*self.fit_res)

        if fmod == "twoblock_simlut":
            with h5py.File(self.savehdf5,"r") as f:
                self.fit_params = lf.Parameters()
                for i in self.fit_bias_list:
                    i = int(i)
                    self.fit_params.add(name=f"alpha_{i}",value=f[f"ch{self.ch}/Z/analysis/{i}uA/fitting_result/twoblock/alpha"][...],min=0,max=300)
                    self.fit_params.add(name=f"beta_{i}",value=f[f"ch{self.ch}/Z/analysis/{i}uA/fitting_result/twoblock/beta"][...],min=0.1,max=2)
                    self.fit_params.add(name=f"Ctes_{i}",value=f[f"ch{self.ch}/Z/analysis/{i}uA/fitting_result/twoblock/Ctes"][...],min=1e-15,max=1e-11)
                self.fit_params.add(name=f"Cabs",value=0.048e-12,min=1e-15,max=1e-11)
                self.fit_params.add(name=f"Gabs",value=9e-8,min=1e-10,max=1e-6)

                self.simult_res = lf.minimize(fcn=self.cal_resid,params=self.fit_params,method="leastsq")
                print(lf.fit_report(self.simult_res))

    def save_fitting_result(self,fmod):

        with h5py.File(self.savehdf5,"a") as f:

            if "analysis" in f[f"ch{self.ch}"]["Z"].keys():
                if f"{self.ibias}uA" in f[f"ch{self.ch}"]["Z"]["analysis"].keys():
                    if "fitting_result" in f[f"ch{self.ch}"]["Z"]["analysis"][f"{self.ibias}uA"].keys():
                        if fmod in f[f"ch{self.ch}"]["Z"]["analysis"][f"{self.ibias}uA"]["fitting_result"].keys():
                            del f[f"ch{self.ch}"]["Z"]["analysis"][f"{self.ibias}uA"]["fitting_result"][fmod]

                    if "Ibias" not in f[f"ch{self.ch}"]["Z"]["analysis"][f"{self.ibias}uA"]:
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Ibias",data=self.ibias)
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Rtes",data=self.Rtes)
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Ites",data=self.Ites)
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Vtes",data=self.Vtes)
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Ttes",data=self.Ttes)
                        f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/Gtes",data=self.Gtes)

            if fmod == "oneblock":
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/alpha",data=self.fit_res[0]) 
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/beta",data=self.fit_res[1]) 
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/C",data=self.fit_res[2]) 

            if fmod == "twoblock":
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/alpha",data=self.fit_res[0]) 
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/beta",data=self.fit_res[1]) 
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/Cabs",data=self.fit_res[2])
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/Ctes",data=self.fit_res[3])
                f.create_dataset(f"ch{self.ch}/Z/analysis/{self.ibias}uA/fitting_result/{fmod}/Gabs",data=self.fit_res[4])  

    def Rth_L_fit(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z = self.Zdiv[mask]
        self.z_err = self.Zdiv_err[mask]
        self.f = self.frq[mask]
        model = Model(self.Rth_L_func)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        result = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4)
        print(result.fit_report())
        self.f_fit = np.arange(mvalue,1.05e+5,10)
        self.z_fit = self.Rth_L_func(self.f_fit,result.best_values["L"],result.best_values["Rth"])
        self.z_fit_p = self.Rth_L_func(self.f,result.best_values["L"],result.best_values["Rth"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Rth_L_fit_N(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask         = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z       = 1/self.Zdiv[mask]
        self.z_err   = 1/self.Zdiv_err[mask]
        self.f       = self.frq[mask]
        model        = Model(self.Rth_L_func)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        result       = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4)
        print(result.fit_report())
        self.f_fit   = np.arange(mvalue,1.05e+5,10)
        self.z_fit   = self.Rth_L_func(self.f_fit,result.best_values["L"],result.best_values["Rth"])
        self.z_fit_p = self.Rth_L_func(self.f,result.best_values["L"],result.best_values["Rth"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth     = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L       = result.best_values["L"]*1e-9
        self.L_err   = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Rth_L_fit_Lpar(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z = self.Zdiv[mask]
        self.z_err = self.Zdiv_err[mask]
        self.f = self.frq[mask]
        model = Model(self.Rth_L_func_Lpar)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('Lpar',min=0,max=100)
        result = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4,Lpar=1e-3)
        print(result.fit_report())
        self.f_fit = np.arange(mvalue,1.05e+5,10)
        self.z_fit = self.Rth_L_func_Lpar(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["Lpar"])
        self.z_fit_p = self.Rth_L_func_Lpar(self.f,result.best_values["L"],result.best_values["Rth"],result.best_values["Lpar"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Rth_L_fit_Rpar(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z = self.Zdiv[mask]
        self.z_err = self.Zdiv_err[mask]
        self.f = self.frq[mask]
        model = Model(self.Rth_L_func_Rpar)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('Rpar',min=0,max=100)
        result = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4,Rpar=1e-6)
        print(result.fit_report())
        self.f_fit = np.arange(mvalue,1.05e+5,10)
        self.z_fit = self.Rth_L_func_Rpar(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["Rpar"])
        self.z_fit_p = self.Rth_L_func_Rpar(self.f,result.best_values["L"],result.best_values["Rth"],result.best_values["Rpar"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Rth_L_fit_Cpar(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z = self.Zdiv[mask]
        self.z_err = self.Zdiv_err[mask]
        self.f = self.frq[mask]
        model = Model(self.Rth_L_func_Cpar)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('Cpar',min=0,max=1)
        result = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4,Cpar=1e-6)
        print(result.fit_report())
        self.f_fit = np.arange(mvalue,1.05e+5,10)
        self.z_fit = self.Rth_L_func_Cpar(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["Cpar"])
        self.z_fit_p = self.Rth_L_func_Cpar(self.f,result.best_values["L"],result.best_values["Rth"],result.best_values["Cpar"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Rth_L_fit_CLpar(self,N=800,S=0,mvalue=0):
        self.Z_div_out(N=N,S=S)
        self.IV_result()
        mask = (self.frq > mvalue) & (self.Zdiv_err.real < 0.1)
        self.z = self.Zdiv[mask]
        self.z_err = self.Zdiv_err[mask]
        self.f = self.frq[mask]
        model = Model(self.Rth_L_func_Cpar)
        model.make_params(verbose=True)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('Cpar',min=0,max=1)
        result = model.fit(self.z,f=self.f,weights=1/self.z_err,L=12,Rth=4,Cpar=1e-6)
        print(result.fit_report())
        self.f_fit = np.arange(mvalue,1.05e+5,10)
        self.z_fit = self.Rth_L_func_Cpar(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["Cpar"])
        self.z_fit_p = self.Rth_L_func_Cpar(self.f,result.best_values["L"],result.best_values["Rth"],result.best_values["Cpar"])
        self.result_plot(plot_subject="Rth_L_fit")
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9
        self.save_Rth_L()

    def Z_plot_s(self,filename="Z_data.hdf5",ibias=800.0,ch=None):
        self.plot_init(style="ZRI",scaling=None)
        l = len(self.Ibias_list)
        frq,vout,vout_err,theta,theta_err = self.Z_select(ibias=ibias)
        z = self.Z(frq,vout,theta)/self.Tr() - (self.Rth+2j*np.pi*frq*self.L+self.Rn)                
        sfn = f"./graph/ch{self.ch}_Ibias{ibias}_ZTES.png"
        self.ax1.set_title("Corrected TES Impedance",fontsize=self.fs)
        frq = frq[z.imag<10]
        z = z[z.imag<10]
        self.ax1.scatter(z.real*1e+3,z.imag*1e+3,s=self.ps)
        self.ax2.scatter(frq,z.real*1e+3,s=self.ps)
        self.ax3.scatter(frq,z.imag*1e+3,s=self.ps)
        self.ax1.legend(loc='best',fontsize=15)
        self.fig.tight_layout()
        self.fig.savefig(sfn,dpi=300)
                

    def ZTES_fit(self,ibias=400,PLOT=True,Cabsfix=True,fmod="twoblock",save=True):
        self.ibias = ibias
        self.sfn = f"Ibias{ibias}uA_twoblock"
        self.Set_IV_parameter()
        self.Fit_Z_def()
        self.lmfit_style(fmod=fmod)
        if save == True:
            self.save_fitting_result(fmod=fmod)

        if PLOT == True:
            self.result_plot(plot_subject=fmod)

    def ZTES_fit_all(self,PLOT,fmod,save):
        for e,i in enumerate(self.Ibias_list):
            self.ZTES_fit(ibias=i,PLOT=PLOT,fmod=fmod,save=save)
            fres = np.append(self.Rtes,self.fit_res)
            fres = np.append(fres,self.fit_res_err)
            if e == 0:
                self.fres_all = fres
            else:
                self.fres_all = np.vstack((self.fres_all,fres))

        self.fres_all  = np.delete(self.fres_all,0,axis=0)
        self.fres_all  = np.delete(self.fres_all,-1,axis=0)
        self.Rpar      = self.fres_all[:,0]/self.Rn
        self.Alpha     = self.fres_all[:,1]
        self.Beta      = self.fres_all[:,2]

        if fmod == "oneblock":
            self.C         = self.fres_all[:,3]
            self.Alpha_err = self.fres_all[:,4]
            self.Beta_err  = self.fres_all[:,5]
            self.C_err     = self.fres_all[:,6]
            self.result_plot(plot_subject="oneblock_result")

        if fmod == "twoblock":
            self.Cabs         = self.fres_all[:,3]
            self.Ctes         = self.fres_all[:,4]
            self.Gabs         = self.fres_all[:,5]
            self.Alpha_err    = self.fres_all[:,6]
            self.Beta_err     = self.fres_all[:,7]
            self.Cabs_err     = self.fres_all[:,8]
            self.Ctes_err     = self.fres_all[:,9]
            self.Gabs_err     = self.fres_all[:,10]
            self.result_plot(plot_subject="twoblock_result")


### Simultaneously Fitting by Twoblock Model

    def hZTES_out(self):
        for e,i in enumerate(self.fit_bias_list):
            if e == 0:
                self.hfrq  = self.data[f"{i}"]["fit"]["Frq"]
                self.hztes = self.data[f"{i}"]["fit"]["ZTES"]
                self.hztes_err = self.data[f"{i}"]["fit"]["ZTES_err"]
            else:
                self.hfreq = np.hstack((self.hfrq,self.data[f"{i}"]["fit"]["Frq"]))
                self.hztes = np.hstack((self.hztes,self.data[f"{i}"]["fit"]["ZTES"]))
                self.hztes_err = np.hstack((self.hztes_err,self.data[f"{i}"]["fit"]["ZTES_err"]))                        

    def twoblock_simult_mod(self,params):
        for e,i in enumerate(self.fit_bias_list):
            i = int(i)
            if e == 0:
                hresult = self.Z_fit_two(f=self.data[f"{i}"]["fit"]["Frq"],alpha=params[f"alpha_{i}"],beta=params[f"beta_{i}"],Cabs=params[f"Cabs"],Ctes=params[f"Ctes_{i}"],Gabs=params[f"Gabs"])

            else:
                hresult = np.hstack((hresult,self.Z_fit_two(f=self.data[f"{i}"]["fit"]["Frq"],alpha=params[f"alpha_{i}"],beta=params[f"beta_{i}"],Cabs=params[f"Cabs"],Ctes=params[f"Ctes_{i}"],Gabs=params[f"Gabs"])))

        return hresult

    def cal_resid(self,params):
        return (self.hztes.real - self.twoblock_simult_mod(params=params).real)**2/self.hztes_err.real + (self.hztes.imag - self.twoblock_simult_mod(params=params).imag)**2/self.hztes_err.imag


    def ZTES_fit_simult(self):
        self.ZTES_data_out()
        self.hZTES_out()
        self.lmfit_style(fmod="twoblock_simlut")

## stepper FUNC for calculate error ##

    def ZTES_fit_cont(self,ibias,fmod,xname,xmin,xmax,xstep,yname,ymin,ymax,ystep):
        self.ZTES_fit(ibias=300,fmod="oneblock")
        print("------------------------------------------------------------------------")
        print("Best fitting parameters")
        print(f"Alpha = {self.fit_res[0]}")
        print(f"Beta = {self.fit_res[1]}")
        print(f"C = {self.fit_res[2]}")
        print("Start Stepper")
        print(f"X = {xname}, Xmin = {xmin}, Xmax = {xmax}, step = {xstep}")
        print(f"Y = {yname}, Ymin = {ymin}, Ymax = {ymax}, step = {ystep}")

        self.xname  = xname
        self.yname  = yname
        self.step_x = np.linspace(xmin,xmax,xstep)
        self.step_y = np.linspace(ymin,ymax,ystep)
        self.ibias = ibias
        self.sfn = f"Ibias{ibias}uA_{fmod}"
        self.Set_IV_parameter()
        self.Fit_Z_def()
        self.lmfit_style(fmod=fmod)
        self.result_plot(plot_subject=fmod)
        self.res_all[:,2] = self.res_all[:,2] - np.min(self.res_all[:,2])
        print(self.res_all)
        CR68 = self.res_all[self.res_all[:,2]<self.CRange68]
        CR90 = self.res_all[self.res_all[:,2]<self.CRange90]
        CR99 = self.res_all[self.res_all[:,2]<self.CRange99]
        np.argmin(np.abs(self.delc[self.delc<self.CRange68])) 
        print("------------------------------------------------------------------------")
        print("Steppar Result")
        print(f"{self.xname}(1 sigma) : min = {np.min(CR68[:,0])},max = {np.max(CR68[:,0])}")
        print(f"{self.yname}(1 sigma) : min = {np.min(CR68[:,1])},max = {np.max(CR68[:,1])}")
        print(f"{self.xname}(2 sigma) : min = {np.min(CR90[:,0])},max = {np.max(CR90[:,0])}")
        print(f"{self.yname}(2 sigma) : min = {np.min(CR90[:,1])},max = {np.max(CR90[:,1])}")
        print(f"{self.xname}(3 sigma) : min = {np.min(CR99[:,0])},max = {np.max(CR99[:,0])}")
        print(f"{self.yname}(3 sigma) : min = {np.min(CR99[:,1])},max = {np.max(CR99[:,1])}")

## Alpha and Beta correction ##
    def load_result(self):
        self.fmod = "oneblock"
        with h5py.File(self.savehdf5,"r") as f:
            self.n    = f[f"ch{self.ch}"]["IV"]["analysis"]["Gfit"]["n"][...]
            self.L    = f[f"ch{self.ch}"]["Z"]["analysis"]["Rth_L_fit"]["L"][...]
            self.Rth  = f[f"ch{self.ch}"]["Z"]["analysis"]["Rth_L_fit"]["Rth"][...]
            print("-----------------------------------------------")
            print(f"Loading Result ...")
            print(f"Temperature Dependence : n = {self.n}")
            print(f"Inductance in the TES Circuit : L = {self.L} [H]")
            print(f"Resistance in the TES Circuit : Rth = {self.Rth} [Ohm]")

            for e,i in enumerate(f[f"ch{self.ch}"]["Z"]["analysis"].keys()):

                if i != "Rth_L_fit":

                    Ites = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["Ites"][...]
                    Rtes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["Rtes"][...]
                    Ttes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["Ttes"][...]
                    Gtes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["Gtes"][...]


                    print(f"TES Current : Ites = {Ites} [A]")
                    print(f"TES Resistance : Rtes = {Rtes} [Ohm]")
                    print(f"TES Temperature : Ttes = {Ttes} [K]")
                    print(f"Membrane Heat : Gtes = {Gtes} [W/K]")


                    if self.fmod == "oneblock":
                        print(f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"].keys())
                        alpha = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["alpha"][...]                           
                        beta  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["beta"][...]
                        C     = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["C"][...]

                        print(f"Temperature Sensitivity : Alpha = {alpha}")
                        print(f"Current Sensitivity     : Beta  = {beta}")
                        print(f"Abs + TES Heat Capacity : C     = {C}")  

                    if self.fmod == "twoblock":
                        alpha = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["alpha"][...]                           
                        beta  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["beta"][...]
                        Cabs  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["Cabs"][...]
                        Ctes  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["Ctes"][...]
                        Gabs  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{i}"]["fitting_result"][f"{self.fmod}"]["Gabs"][...]
                        print(f"Temperature Sensitivity  : Alpha = {alpha}")
                        print(f"Current Sensitivity      : Beta  = {beta}")
                        print(f"Absorber Heat Capacity   : Cabs  = {Cabs}") 
                        print(f"TES Heat Capacity        : Ctes  = {Ctes}") 
                        print(f"Abs TES Heat Conductance : Gabs  = {Gabs}")

                    res = np.array([Rtes,Ites,Ttes,Gtes,alpha,beta,C])

                    if e == 0:
                        self.res = res
                    else:
                        self.res = np.vstack((self.res,res))
            self.IV_result()
            self.res      = np.delete(self.res,0,axis=0)
            self.res      = np.delete(self.res,-1,axis=0)
            self.Rtes     = self.res[:,0]
            self.Rpar     = self.Rtes/self.Rn
            self.Ites     = self.res[:,1]
            self.Ttes     = self.res[:,2]
            self.Gtes     = self.res[:,3]
            self.alpha    = self.res[:,4]
            self.beta     = self.res[:,5]
            self.C        = self.res[:,6]
            with h5py.File('TMU542_5um_result.hdf5','a') as f:
                f.create_dataset('alpha',data=self.alpha)
                f.create_dataset('beta',data=self.beta)
                f.create_dataset('Rpar',data=self.Rpar)
            # self.out_data = np.vstack((self.Ites,self.beta))
            # #self.out_data = np.vstack((self.out_data,self.beta))
            # np.savetxt("beta_Ites.txt",self.out_data.T)
            # self.func_call()
            # print("Lcrit-")
            # print(self.LcritM)
            # print("Lcrit+")
            # print(self.LcritP)
            # self.result_plot(plot_subject="Lcrit")

    def fitres_out(self,ch,savehdf5,Tbath):
        self.savehdf5  = savehdf5
        self.ch        = ch
        self.Tbath     = Tbath
        self.load_result()
        #self.result_plot(plot_subject="rise_fall")
        self.T         = np.linspace(np.min(self.Ttes),np.max(self.Ttes),1000)
        self.alpha_int = interp1d(self.Ttes,self.alpha,kind="cubic",bounds_error=False,fill_value=0)
        self.beta_int  = interp1d(self.Ttes,self.beta,kind="cubic",bounds_error=False,fill_value=0)

## Alpha & Beta dependence

    def R_ab_func(self):
        self.load_result()
        self.result_plot(plot_subject="alpha_interp")

    def R_int_plot(self):
        self.IV_result()
        Tl = np.linspace(-0.5e-3,0.5e-3,10)
        Il = np.linspace(-10e-6,10e-6,10)
        mask = (self.ttes > 165e-3) & (self.ttes < 168e-3)
        Rtes = self.rtes[mask]
        Ites = self.ites[mask]
        Ttes = self.ttes[mask]
        fig = plt.figure(figsize=(10.6,6))
        ax  = plt.subplot(111)
        ax.grid(linestyle="dashed")
        ax.set_xlabel(rf"$\rm Temperature \ (mK)$",fontsize=self.fs)
        ax.set_ylabel(rf"$\rm Resistance \ (m\Omega)$",fontsize=self.fs)
        for R0,T0,I0 in zip(Rtes,Ttes,Ites):
            print(R0,T0,I0)
            ax.scatter(T0*1e+3,self.R_interp(T=T0,I=I0,T0=165.65e-3,I0=70.97e-6,R0=12.59e-3)*1e+3,color="Red",s=self.ps,label="Z")
            ax.scatter(T0*1e+3,R0*1e+3,color="black",s=self.ps,label="IV")
        plt.show()

    def R_interpting(self):
        self.IV_result()
        Tl = np.linspace(-1e-3,1e-3,30)
        Il = np.linspace(-10e-6,10e-6,30)
        #mask = (self.ttes > 165e-3) & (self.ttes < 168e-3)
        Rtes = self.rtes
        Ites = self.ites
        Ttes = self.ttes
        print(Rtes,Ttes,Ites) 
        for R0,T0,I0 in zip(Rtes,Ttes,Ites):
            print(R0,T0,I0) 
            for e,T in enumerate(Tl):
                for ee,I in enumerate(Il):
                    R = self.R_interp(T=T+T0,I=I+I0,T0=T0,I0=I0,R0=R0)
                    if "res_h" not in locals():
                        res_h = np.array([T+T0,I+I0,R])
                        #print(T+T0,I+I0,R)
                    else:
                        res_h = np.vstack((res_h,np.array([T+T0,I+I0,R])))
                        #print(T+T0,I+I0,R)

        print(res_h[0])
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111,projection="3d")
        pmask = (res_h[:,2]<100)
        #surf = ax.plot_trisurf(res_h[:,0][pmask]*1e+3,res_h[:,1][pmask]*1e+6,res_h[:,2][pmask]*1e+3,cmap="jet",shade=False)
        ax.scatter(Ttes*1e+3,Ites*1e+6,Rtes*1e+3,color="Red")
        #ax.scatter(res_h[:,0]*1e+3,res_h[:,1]*1e+6,res_h[:,2]*1e+3,color="Blue")
        #fig.colorbar(surf)
        #H = ax.hist2d(Ttes*1e+3,Ites, bins=[40,40], cmap="jet")
        #fig.colorbar(H[3],ax=ax)
        ax.set_xlabel(rf"$\rm Temperature \ (mK)$")
        ax.set_ylabel(rf"$\rm Current \ (\mu A)$")
        ax.set_zlabel(rf"$\rm Resistance \ (m\Omega)$")
        ax.set_title("TES T-I-R plot")
        plt.show()
        np.savetxt("R_interp.txt",res_h)

    def R_pro(self):
        self.fitres_out(ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90.0)
        self.R_interpting()

    def beta_fitting(self):
        beta_RSJ = (self.Rn/self.Rtes)**2 - 1
        plt.scatter(self.Rpar,self.beta)
        plt.plot(self.Rpar,beta_RSJ,".-",color="black")
        plt.xlabel(r"$\rm R_n/R$")
        plt.ylabel(r"$\rm \beta_I$")
        print(self.Rn)
        print(self.Rtes)
        plt.show()

## Out Function ##

    def all_analysis(self,ch,savehdf5,Tbath,ibias,fmod,xname,xmin,xmax,xstep,yname,ymin,ymax,ystep):
        self.initialize(ch,savehdf5,Tbath)
        self.data_out()
        #self.result_plot(plot_subject="Z")
        #self.sel_bias = [0.0,800.0]
        #self.result_plot(plot_subject="sel_bias")
        #self.Rth_L_fit()
        self.load_Rth_L()
        #self.result_plot(plot_subject="ZTES")
        #self.ZTES_fit(ibias=300,fmod="oneblock")
        #self.ZTES_fit_all(fmod="oneblock",PLOT=False,save=True)
        #self.ZTES_fit(ibias=400,fmod="twoblock")
        #self.ZTES_fit_all(fmod="twoblock",PLOT=False,save=True)
        #self.ZTES_fit_simult()
        self.ZTES_fit_cont(ibias,fmod,xname,xmin,xmax,xstep,yname,ymin,ymax,ystep)

    def init_call(self,ch,savehdf5,Tbath):
        self.initialize(ch,savehdf5,Tbath)
        self.data_out()
        #self.result_plot(plot_subject="Z")
        #self.sel_bias = [0.0,800.0]
        #self.result_plot(plot_subject="sel_bias")
        self.Rth_L_fit()
        #self.Rth_L_fit_Rpar()
        #self.Rth_L_fit_Lpar()
        #self.Rth_L_fit_Cpar()
        #self.load_Rth_L()
        #self.result_plot(plot_subject="ZTES")
        #self.ZTES_fit(ibias=300,fmod="oneblock")
        #self.ZTES_fit_all(fmod="oneblock",PLOT=False,save=True)
        #self.ZTES_fit(ibias=400,fmod="twoblock")
        #self.ZTES_fit_all(fmod="twoblock",PLOT=False,save=True)
        #self.ZTES_fit_simult()
        #self.fitres_out()
        #print(self.tau_el)
        #print(self.tau_I)
        #self.ploter()
        self.Tr_plot()

    def Lcrit_plot(self):
        self.initialize(ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90.0)
        self.load_result()
        self.result_plot(plot_subject="alpha_beta")

    def beta_fit_RSJ(self):
        self.initialize(ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90.0)
        self.load_result()
        self.beta_fitting()
        
