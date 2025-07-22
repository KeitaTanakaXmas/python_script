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
from scipy.interpolate import interp1d
from scipy import integrate,signal

#from Basic import Plotter


__author__ =  'Keita Tanaka'
__version__=  '2.0.0' #2023.01.12

print('===============================================================================')
print(f"Complex Impedance Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class ZAnalysis:

    def __init__(self):
        self.Vac    =  25 * 1e-3
        self.Rshunt =  3.9 * 1e-3
        # self.Min    =  9.82 * 1e-11
        # self.Mfb    =  8.5 * 1e-11
        self.Rfb    =  100e+3
        self.Rac    =  1e+4 ## ? Check Magnicon Resistance in the load.
        self.Lac    =  470 * 1e-6
        self.Lcom   =  251 * 1e-9
        self.CRange68 = 2.295815160785974337606
        self.CRange90 = 4.605170185988091368036
        self.CRange99 = 9.210340371976182736072
        # self.Min    = 10.118e-11  #[H]
        # self.Mfb    = 8.6e-11  #[H]

        self.Min    = 10.494e-11  #[H]
        self.Mfb    = 8.520e-11  #[H]
        # self.Min    = 98.96e-12  #[H]
        # self.Mfb    = 85.20e-12  #[H]

    def initialize(self,ch,savehdf5,Tbath):
        self.ch = ch
        #self.filelist = sorted(glob.glob(f"./../data/Z/ch{ch}/*.txt"))
        self.filelist = sorted(glob.glob(f"./*.txt"))
        self.savehdf5 = savehdf5
        self.Tbath = int(Tbath)
        print(self.filelist)

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

    def Z_TES(self,f,Z):
        self.Tr_func()
        Z_TES = Z/self.Tr - (self.Rth+2j*np.pi*f*self.L)
        return Z_TES

    def Z_TES_err(self,frq,Vout,theta,Vout_err,theta_err):
        frq_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Frq"]
        vout_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Vout"]
        vouterr_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Vout_err"]
        theta_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Theta"]
        thetaerr_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Theta_err"]    
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
        co = self.Z(frq,Vout,theta).real *self.Tr.real + self.Z(frq,Vout,theta).imag *self.Tr.imag
        Z_part_real = (self.Tr.real*self.Z_err(frq,Vout,theta,Vout_err,theta_err).real/(abs(self.Tr)**2))**2 + (self.Tr.imag*self.Z_err(frq,Vout,theta,Vout_err,theta_err).imag/(abs(self.Tr)**2))**2
        Tr_part_real = (((self.Z(frq,Vout,theta).real/(abs(self.Tr))**2)-2*self.Tr.real*(co)/(abs(self.Tr))**4)*Tr_err.real)**2 + (((self.Z(frq,Vout,theta).imag/(abs(self.Tr))**2)-2*self.Tr.imag*(co)/(np.abs(self.Tr))**4)*Tr_err.imag)**2
        Z_TES_err_real = (Z_part_real + Tr_part_real + Rth_err**2 )**0.5
        cf = self.Z(frq,Vout,theta).imag *self.Tr.real - self.Z(frq,Vout,theta).real *self.Tr.imag
        Z_part_imag = (-self.Tr.imag*self.Z_err(frq,Vout,theta,Vout_err,theta_err).real/(abs(self.Tr)**2))**2 + (self.Tr.real*self.Z_err(frq,Vout,theta,Vout_err,theta_err).imag/(abs(self.Tr)**2))**2
        Tr_part_imag = (((self.Z(frq,Vout,theta).imag/(abs(self.Tr))**2)-2*self.Tr.real*(cf)/(abs(self.Tr))**4)*Tr_err.real)**2 + (((-self.Z(frq,Vout,theta).real/(abs(self.Tr))**2)-2*self.Tr.imag*(cf)/(abs(self.Tr))**4)*Tr_err.imag)**2
        Z_TES_err_imag = (Z_part_imag + Tr_part_imag + (2j*np.pi*frq*L_err)**2 )**0.5
        
        Z_TES_err = Z_TES_err_real + 1j*Z_TES_err_imag
        
        return Z_TES_err

    def Z_div_out(self):
        frq_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Frq"]
        Vout_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Vout"]
        Vouterr_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Vout_err"]
        Theta_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Theta"]
        Thetaerr_S = self.data[f'S'][f"{self.Tsup}"][f"{self.Isup}"]["Theta_err"]  
        frq_N = self.data[f'N'][f"{self.Tnor}"][f"{self.Inor}"]["Frq"]
        Vout_N = self.data[f'N'][f"{self.Tnor}"][f"{self.Inor}"]["Vout"]
        Vouterr_N = self.data[f'N'][f"{self.Tnor}"][f"{self.Inor}"]["Vout_err"]
        Theta_N = self.data[f'N'][f"{self.Tnor}"][f"{self.Inor}"]["Theta"]
        Thetaerr_N = self.data[f'N'][f"{self.Tnor}"][f"{self.Inor}"]["Theta_err"]  
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

    def Tr_func(self):
        self.Tr = self.Zsup/(self.Rth+2j*np.pi*self.frq_S*self.L)

    def Tr_func(self):
        self.Tr = self.Znor/(self.Rn+2j*np.pi*self.frq_S*self.L)

    def Rth_L_func(self,f,L,Rth):
        Tr = (self.Rn + 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))
        return Tr

    def Rth_L_Rpar_func(self,f,L,Rth,Rpar):
        Tr = (self.Rn*Rpar + 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))
        return Tr

    def Rth_L_R_func(self,f,L,Rth,Rpar):
        Tr = (self.Rn + 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3))/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))
        return Tr

    def Rth_L2_func(self,f,L,Rth,L2):
        Tr = (self.Rn + 1j*2*np.pi*f*L*10**(-9)+Rth*10**(-3)+1j*2*np.pi*f*L2*10**(-9))/(Rth*10**(-3)+1j*2*np.pi*f*L*10**(-9))
        return Tr

    def def_IV(self,ibias):
        self.Ites = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Ites']
        self.Vtes = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Vtes']
        self.Rtes = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Rtes']
        self.Gtes = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Gtes']
        self.Ttes = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Ttes']
        self.Pb = self.data['T'][f"{self.Tbath}"][f"{ibias}"]['Pb']

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


    def plot_init(self):
        #plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap'] = 'jet'
        plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        plt.rcParams["font.size"] = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます
        plt.rcParams['xtick.direction'] = 'in' # x axis in
        plt.rcParams['ytick.direction'] = 'in' # y axis in 
        plt.rcParams['axes.linewidth'] = 1.0 # axis line width
        plt.rcParams['axes.grid'] = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['scatter.edgecolors'] = 'black'
        self.fs = 15
        self.ps = 30

    def plot_window(self,style):
        self.plot_init()
        if style == "ZRI":
            self.fig = plt.figure(figsize=(24,4),constrained_layout=True)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.ax1.grid(linestyle="dashed")
            self.ax1.set_xlabel(r"$ \rm Re(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Im(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax2.grid(linestyle="dashed")
            self.ax2.semilogx()
            #self.ax2.set_title("Frequency vs Re(Z) plot",fontsize=self.fs)
            self.ax2.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$ \rm Re(Z) \ (m\Omega)$",fontsize=self.fs)
            self.ax3.grid(linestyle="dashed")
            self.ax3.semilogx()
            #self.ax3.set_title("Frequency vs Im(Z) plot",fontsize=self.fs)
            self.ax3.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax3.set_ylabel(r"$ \rm Im(Z) \ (m\Omega)$",fontsize=self.fs)

        if style == "ZRrIr":
            fs = 10
            self.fig = plt.figure(figsize=(24,6))
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
            self.ax1.set_xlabel(r"$ \rm Re(Z_{TES}) \ (m\Omega)$",fontsize=fs)
            self.ax1.set_ylabel(r"$ \rm Im(Z_{TES}) \ (m\Omega)$",fontsize=fs)
            self.ax2.grid(linestyle="dashed")
            self.ax2.semilogx()
            self.ax2.set_ylabel(r"$ \rm Re(Z_{TES}) \ (m\Omega)$",fontsize=fs)
            self.ax3.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=fs)
            self.ax3.grid(linestyle="dashed")
            self.ax3.semilogx()
            self.ax3.set_ylabel(r"$ \rm Residual  \ (m\Omega)$",fontsize=fs)
            self.ax5.set_ylabel(r"$ \rm Residual  \ (m\Omega)$",fontsize=fs)
            self.ax4.grid(linestyle="dashed")
            self.ax4.semilogx()
            self.ax4.set_ylabel(r"$ \rm Im(Z_{TES}) \ (m\Omega)$",fontsize=fs)            
            self.ax5.set_xlabel(r"$\rm Frequency \ (Hz)$",fontsize=fs)
            self.ax5.semilogx()
            self.ax5.grid(linestyle="dashed")


    def plotting(self,pobj):
        print('plotting')
        if pobj == 'rawZ' or pobj == 'corZ':
            self.plot_window(style='ZRI')
            for e,i in enumerate(self.Ibias_list):
                if pobj == 'rawZ':
                    Z_real = self.data['T'][f"{self.Tbath}"][f"{i}"]['Z_real'] * 1e+3
                    Z_imag = self.data['T'][f"{self.Tbath}"][f"{i}"]['Z_imag'] * 1e+3
                    frq = self.data['T'][f"{self.Tbath}"][f"{i}"]['Frq']
                    sfn = 'rawZ'
                elif pobj == 'corZ':
                    print('corZ plot')
                    Z_real = self.data['T'][f"{self.Tbath}"][f"{i}"]['Ztes_real'] * 1e+3
                    Z_imag = self.data['T'][f"{self.Tbath}"][f"{i}"]['Ztes_imag'] * 1e+3
                    frq = self.data['T'][f"{self.Tbath}"][f"{i}"]['Frq']
                    sfn = 'corZ'
                self.ax1.plot(Z_real,Z_imag,".",label=f"{i}uA",color=cm.jet([e/self.Ibias_len]))
                self.ax2.plot(frq,Z_real,".",label=f"{i}uA",color=cm.jet([e/self.Ibias_len]))
                self.ax3.plot(frq,Z_imag,".",label=f"{i}uA",color=cm.jet([e/self.Ibias_len]))
            self.ax3.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)

        if pobj == 'SN':
            sfn = 'SN'
            self.plot_window(style='ZRI')
            self.frq_S  = self.data['S'][f"{self.Tsup}"][f"{self.Isup}"]["Frq"]
            self.frq_N  = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Frq"]
            self.Zsup = (self.data['S'][f"{self.Tsup}"][f"{self.Isup}"]["Z_real"] + 1j * self.data['S'][f"{self.Tsup}"][f"{self.Isup}"]["Z_imag"])*1e+3
            self.Znor = (self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_real"] + 1j * self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_imag"])*1e+3
            self.ax1.plot(self.Zsup.real,self.Zsup.imag,".",color='Blue')
            self.ax2.plot(self.frq_S,self.Zsup.real,".",color='Blue')
            self.ax3.plot(self.frq_S,self.Zsup.imag,".",label=f"Super",color='Blue')
            self.ax1.plot(self.Znor.real,self.Znor.imag,".",color='Red')
            self.ax2.plot(self.frq_N,self.Znor.real,".",color='Red')
            self.ax3.plot(self.frq_N,self.Znor.imag,".",label=f"Normal",color='Red')
            self.ax3.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=15)

        if pobj == 'Tr_fit':
            sfn = 'Tr_fit'
            self.plot_window('ZRI')
            self.ax1.plot(self.z_tr.real,self.z_tr.imag,".",color='black')
            self.ax2.plot(self.frq_N,self.z_tr.real,".",color='black')
            self.ax3.plot(self.frq_N,self.z_tr.imag,".",color='black')
            self.ax1.plot(self.z_fit.real,self.z_fit.imag,color='red')
            self.ax2.plot(self.f_fit,self.z_fit.real,color='red')
            self.ax3.plot(self.f_fit,self.z_fit.imag,color='red')

        if pobj == 'Ztes_fit':
            sfn = f'Ztes_fit_{self.fit_bias}'
            self.z = self.z * 1e+3
            self.z_err = self.z_err * 1e+3
            self.z_fit = self.z_fit * 1e+3
            self.z_fit_p = self.z_fit_p * 1e+3
            xerr = np.abs(self.z_err.real)
            yerr = np.abs(self.z_err.imag)
            self.plot_window(style="ZRrIr")
            self.ax1.errorbar(self.z.real,self.z.imag,xerr=xerr,yerr=yerr,markeredgecolor = "black", color='black',markersize=4,fmt="o",ecolor="black")
            self.ax1.plot(self.z_fit.real,self.z_fit.imag,"-",color="red")
            self.ax2.scatter(self.frq,self.z.real,c="black",s=self.ps,label="data")
            self.ax2.plot(self.f_fit,self.z_fit.real,"-",label="fit result",color="red")
            self.ax3.errorbar(self.frq,self.z_fit_p.real - self.z.real,yerr=xerr,markeredgecolor = "black", color='black',markersize=4,fmt="o",ecolor="black",label="data")
            self.ax4.plot(self.f_fit,self.z_fit.imag,"-",label="fit result",color="red")
            self.ax4.scatter(self.frq,self.z.imag,c="black",s=self.ps,label="data")
            self.ax5.errorbar(self.frq,self.z_fit_p.imag - self.z.imag,yerr=yerr,markeredgecolor = "black", color='black',markersize=4,fmt="o",ecolor="black",label="data")
            self.fig.subplots_adjust(hspace=.0)

        #plt.pause(0.1)
        plt.show()
        sfn = f'{sfn}.png'
        self.fig.savefig(sfn,dpi=300)

## Output Data ##

    def data_out(self):
        """
        Loading TES Complex impedance data and calculate Z.

        """
        self.data = {}
        for e,i in enumerate(self.filelist):
            f          = np.genfromtxt(i)
            frq        = f[:,0]
            vout       = f[:,1]
            vout_err   = f[:,2]
            theta      = f[:,3]
            theta_err  = f[:,4]
            file = open(i,'r')
            lines = file.readlines() 
            Tbath = int(re.sub(r"\D", "",lines[4]))
            ibias = int(re.sub(r"\D", "",lines[22]))
            state = str(lines[23][20])
            print(Tbath)
            print(ibias)
            
            if state not in self.data.keys():
                self.data[f'{state}'] = {}
                if Tbath not in self.data.keys():
                    self.data[f'{state}'][f"{Tbath}"] = {}
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]              = {} 
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Frq"]       = frq
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Vout"]      = vout
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Vout_err"]  = vout_err
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Theta"]     = theta
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Theta_err"] = theta_err
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Z_real"]    = self.Z(frq,vout,theta).real
            self.data[f'{state}'][f"{Tbath}"][f"{ibias}"]["Z_imag"]    = self.Z(frq,vout,theta).imag
        if len(self.data.keys()) > 1:
            self.Tbath = np.array(list(self.data['T'].keys()),dtype="int")[0]
            self.Tsup = np.array(list(self.data['S'].keys()),dtype="int")[0]
            self.Tnor = np.array(list(self.data['N'].keys()),dtype="int")[0]
            self.Isup = np.array(list(self.data['S'][f'{self.Tsup}'].keys()),dtype="int")[0]
            self.Inor = np.array(list(self.data['N'][f'{self.Tnor}'].keys()),dtype="int")[0]
            self.Ibias_list = np.array(list(self.data['T'][f"{self.Tbath}"].keys()),dtype="int")
            self.Ibias_len = len(self.Ibias_list)
            print("Data loaded.")
            print(f"Base Temperature = {self.Tbath}mK")
            print(f"Ibias = {self.Ibias_list}")
            print(f"Normal State Temperature = {self.Tnor} mK , Ibias = {self.Inor} uA")
            print(f"Super State Temperature = {self.Tsup}mK , Ibias = {self.Isup} uA")

    def load_Rn(self):
        with h5py.File(self.savehdf5,"r") as f:
            #self.Rn = f[f"ch{self.ch}/IV/analysis/Rn_Z"][...]
            self.Rn = 29.35883e-3

    def load_IV(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.Ibias = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ibias"][:]            
            self.ites = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ites"][:]
            self.vtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Vtes"][:]
            self.rtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Rtes"][:]
            self.pb   = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Pb"][:]
            self.gtes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Gtes"][:]
            self.ttes = f[f"ch{self.ch}"]["IV"]["data"][f"{self.Tbath}mK"]["Ttes"][:]
            #self.Rn   = f[f"ch{self.ch}"]["IV"]["analysis"]["Rn_Z"][...]
            self.Rn   = 29.338279e-3

        for e,i in enumerate(self.Ibias_list):
            idx = self.getNearValueID(self.Ibias,i*1e-6)
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Ites'] = np.array(self.ites[idx]) 
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Vtes'] = np.array(self.vtes[idx])
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Rtes'] = np.array(self.rtes[idx])
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Pb']   = np.array(self.pb[idx])
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Gtes'] = np.array(self.gtes[idx])
            self.data['T'][f"{self.Tbath}"][f"{i}"]['Ttes'] = np.array(self.ttes[idx])


    def Tr_fit(self):
        self.load_Rn()
        self.frq_S  = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Frq"]
        self.frq_N  = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Frq"]
        self.Zsup = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_real"] + 1j * self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_imag"]
        self.Znor = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_real"] + 1j * self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_imag"]
        self.z_tr = self.Znor/self.Zsup
        self.Z_div_out()
        self.z_err = self.Zdiv_err
        model = Model(self.Rth_L_func)
        model.make_params(verbose=True)
        # model.set_param_hint('Rpar',min=0.001,max=100)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        result = model.fit(self.z_tr,f=self.frq_S,weights=1/self.z_err,scale_covar=False,L=12,Rth=4.4)
        #self.Rn = 23.5883e-3
        self.Rn = 29.78e-3
        print(result.fit_report())
        self.f_fit = np.arange(0,1.05e+6,100)
        self.z_fit = self.Rth_L_func(self.f_fit,result.best_values["L"],result.best_values["Rth"])
        self.z_fit_p = self.Rth_L_func(self.frq_N,result.best_values["L"],result.best_values["Rth"])
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9

        print('OK')
        self.plotting('Tr_fit')

    def Tr_fit_L(self):
        self.load_Rn()
        self.frq_S  = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Frq"]
        self.frq_N  = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Frq"]
        self.Zsup = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_real"] + 1j * self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_imag"]
        self.Znor = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_real"] + 1j * self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_imag"]
        self.z_tr = self.Znor/self.Zsup
        self.Z_div_out()
        self.z_err = self.Zdiv_err
        model = Model(self.Rth_L2_func)
        model.make_params(verbose=True)
        # model.set_param_hint('Rpar',min=0.001,max=100)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('L2',min=0,max=100)
        result = model.fit(self.z_tr,f=self.frq_S,weights=1/self.z_err,scale_covar=False,L=12,Rth=4.4,L2=10)
        #self.Rn = 23.5883e-3
        self.Rn = 29.78e-3
        print(result.fit_report())
        self.f_fit = np.arange(0,1.05e+6,100)
        self.z_fit = self.Rth_L2_func(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["L2"])
        self.z_fit_p = self.Rth_L2_func(self.frq_N,result.best_values["L"],result.best_values["Rth"],result.best_values["L2"])
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9

        print('OK')
        self.plotting('Tr_fit')

    def Tr_fit_C(self):
        self.load_Rn()
        self.frq_S  = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Frq"]
        self.frq_N  = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Frq"]
        self.Zsup = self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_real"] + 1j * self.data['S'][f"{self.Tsup}"][f'{self.Isup}']["Z_imag"]
        self.Znor = self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_real"] + 1j * self.data['N'][f"{self.Tnor}"][f'{self.Inor}']["Z_imag"]
        self.z_tr = self.Znor/self.Zsup
        self.Z_div_out()
        self.z_err = self.Zdiv_err
        model = Model(self.Rth_L2_func)
        model.make_params(verbose=True)
        # model.set_param_hint('Rpar',min=0.001,max=100)
        model.set_param_hint('Rth',min=3.9,max=10)
        model.set_param_hint('L',min=0,max=100)
        model.set_param_hint('C',min=0,max=1)
        result = model.fit(self.z_tr,f=self.frq_S,weights=1/self.z_err,scale_covar=False,L=12,Rth=4.4,C=1e-3)
        #self.Rn = 23.5883e-3
        self.Rn = 30e-3
        print(result.fit_report())
        self.f_fit = np.arange(0,1.05e+6,100)
        self.z_fit = self.Rth_L2_func(self.f_fit,result.best_values["L"],result.best_values["Rth"],result.best_values["C"])
        self.z_fit_p = self.Rth_L2_func(self.frq_N,result.best_values["L"],result.best_values["Rth"],result.best_values["C"])
        self.Rth = result.best_values["Rth"]*1e-3
        self.Rth_err = result.params['Rth'].stderr*1e-3
        self.L = result.best_values["L"]*1e-9
        self.L_err = result.params['L'].stderr*1e-9

        print('OK')
        self.plotting('Tr_fit')

    def Z_correction(self):
        for i in self.Ibias_list:
            Z = self.data['T'][f"{self.Tbath}"][f'{i}']['Z_real'] + 1j * self.data['T'][f"{self.Tbath}"][f'{i}']['Z_imag']
            frq = self.data['T'][f"{self.Tbath}"][f'{i}']['Frq']
            Vout = self.data['T'][f"{self.Tbath}"][f'{i}']['Vout']
            Vout_err = self.data['T'][f"{self.Tbath}"][f'{i}']['Vout_err']
            theta = self.data['T'][f"{self.Tbath}"][f'{i}']['Theta']
            theta_err = self.data['T'][f"{self.Tbath}"][f'{i}']['Theta_err']
            Ztes = self.Z_TES(frq,Z)
            Ztes_err = self.Z_TES_err(frq,Vout,theta,Vout_err,theta_err)
            self.data['T'][f"{self.Tbath}"][f'{i}']['Ztes_real'] = Ztes.real
            self.data['T'][f"{self.Tbath}"][f'{i}']['Ztes_imag'] = Ztes.imag
            self.data['T'][f"{self.Tbath}"][f'{i}']['Ztes_err_real'] = Ztes_err.real
            self.data['T'][f"{self.Tbath}"][f'{i}']['Ztes_err_imag'] = Ztes_err.imag

    def Ztes_fit(self,ibias):
        self.fit_bias = ibias
        self.def_IV(ibias)
        self.model = Model(self.Z_fit_two) 
        self.model.set_param_hint('Cabs',min=1e-15,max=1e-12)
        self.model.set_param_hint('Ctes',min=1e-15,max=1e-12)
        self.model.set_param_hint('Gabs',min=1e-7,max=1e-3)
        self.model.set_param_hint('alpha',min=0,max=1000)
        self.model.set_param_hint('beta',min=0,max=100)
        self.z = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_real'] + 1j * self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_imag']
        self.z_err = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_err_real'] + 1j * self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_err_imag']
        self.frq = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Frq']
        result = self.model.fit(self.z,f=self.frq,weights=1/self.z_err,alpha=100,beta=20,Cabs=0.84e-12,Ctes=0.048e-12,Gabs=1e-12)
        print(result.fit_report())
        self.fit_res = np.array([result.best_values["alpha"],result.best_values["beta"],result.best_values["Cabs"],result.best_values["Ctes"],result.best_values["Gabs"]])
        self.fit_res_err = np.array([result.params['alpha'].stderr,result.params['beta'].stderr,result.params['Cabs'].stderr,result.params['Ctes'].stderr,result.params['Gabs'].stderr])
        self.f_fit = np.linspace(10,1.05e+6,100000)
        self.z_fit = self.Z_fit_two(self.f_fit,*self.fit_res)
        self.z_fit_p = self.Z_fit_two(self.frq,*self.fit_res)
        self.plotting('Ztes_fit')

    def Ztes_fit_one(self,ibias):
        self.frq = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Frq']
        sel_reg = self.frq < 3.0e+6
        self.fit_bias = ibias
        self.def_IV(ibias)
        self.model = Model(self.Z_fit_one) 
        self.model.set_param_hint('C',min=1e-16,max=1e-11)
        self.model.set_param_hint('alpha',min=0,max=1000)
        self.model.set_param_hint('beta',min=0,max=100)
        self.z = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_real'][sel_reg] + 1j * self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_imag'][sel_reg]
        self.z_err = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_err_real'][sel_reg] + 1j * self.data['T'][f"{self.Tbath}"][f'{ibias}']['Ztes_err_imag'][sel_reg]
        self.frq = self.data['T'][f"{self.Tbath}"][f'{ibias}']['Frq'][sel_reg]
        result = self.model.fit(self.z,f=self.frq,weights=1/self.z_err,alpha=100,beta=2,C=0.84e-12)
        print('-----------------------------------------------------')
        print(f'Ibias = {ibias} uA')
        print(result.fit_report())
        self.fit_res = np.array([result.best_values["alpha"],result.best_values["beta"],result.best_values["C"]])
        self.fit_res_err = np.array([result.params['alpha'].stderr,result.params['beta'].stderr,result.params['C'].stderr])
        self.f_fit = np.linspace(10,1.05e+6,100000)
        self.z_fit = self.Z_fit_one(self.f_fit,*self.fit_res)
        self.z_fit_p = self.Z_fit_one(self.frq,*self.fit_res)
        self.plotting('Ztes_fit')
        self.ax2.set_title(ibias)
        print(self.Rtes)

    def Ztes_fit_all(self):
        for ibias in self.Ibias_list:
            self.Ztes_fit_one(ibias)

    def test(self):
        self.initialize(ch=1,savehdf5='Exp240424.hdf5',Tbath=100)
        self.data_out()
        self.plotting("rawZ")
        self.plotting('SN')

        self.Tr_fit()
        self.Z_correction()
        self.plotting('corZ')
        self.load_IV()
        #self.Ztes_fit_one(130)
        #self.Ztes_fit_all()