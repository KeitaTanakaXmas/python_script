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
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)

__author__ =  'Keita Tanaka, Tasuku Hayashi'
__version__=  '1.0.0' #2021.12.20

print('===============================================================================')
print(f"COMSOL Data Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class PulseSim:

    def __init__(self):
        pass           
          
    def plot_init(self):
        #plt.subplots_adjust(wspace=15, hspace=12)
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
        self.fs = 15
        self.ps = 20

    def plot_window(self,style):
        self.plot_init()

        if style == "pshape":
            self.xname = r"$\rm Time \ (ms)$"
            self.yname = r"$\rm Current \ (\mu A)$"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "heat":
            self.xname = r"$\rm Time \ (s)$"
            self.yname = r"$\rm Magnitude $"
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

    def cal_various_ab(self):
        alpha = np.linspace(10,3000,1000)
        beta = np.linspace(1,10,1000)
        A,B = np.meshgrid(alpha,beta)
        p = -self.Ipulse(t=self.risetime(L=12e-9,Rth=4.0e-3,R=6.77e-3,beta=B,Pb=123.8e-12,alpha=A,G=2.77e-9,T=150.e-3,C=5.24e-9),L=12e-9,Rth=4.0e-3,R=6.77e-3,beta=B,Pb=123.8e-12,alpha=A,G=2.77e-9,T=150.e-3,C=5.24e-9,dT=4.409e-7,I=107.68e-6)

        return A,B,p

    def result_plot(self,subject):
        if subject == "heat":
            self.plot_window(style="heat")
            t = np.linspace(0,1)

            self.ax.scatter(self.t_sim,(self.I_sim)*1e+6,c="Blue",s=self.ps)

        if subject == "pshape":
            self.plot_window(style="pshape")
            t = np.linspace(0,10e-3,100000)
            pfab = self.Ipulse(t=t,L=12e-9,Rth=4.0e-3,R=6.77e-3,beta=1,Pb=123.8e-12,alpha=50,G=2.77e-9,T=150.e-3,C=5.24e-11,dT=4.409e-5,I=107.68e-6)
            pfab2 = self.Ipulse(t=t,L=12e-9,Rth=4.0e-3,R=6.77e-3,beta=1,Pb=123.8e-12,alpha=50,G=2.77e-9,T=150.e-3,C=5.24e-9,dT=4.409e-7,I=107.68e-6)
            self.ax.plot(t*1e+3,pfab*1e+6,lw=3,color="blue",label=r"$l = 100um$")
            self.ax.plot(t*1e+3,pfab2*1e+6,lw=3,color="red",label=r"$l = 1mm$")
            # pfab = self.Ipulse(t=t,L=12e-9,Rth=4e-3,R=15e-3,beta=1,Pb=60e-12,alpha=10,G=1.5e-9,T=160e-3,C=1e-12,dT=0.5e-3,I=65e-6)
            # self.ax.plot(t*1e+3,pfab*1e+6,lw=3,color="red",label=r"$\alpha=10,\beta=1$")
            # pfab = self.Ipulse(t=t,L=12e-9,Rth=4e-3,R=15e-3,beta=2,Pb=60e-12,alpha=100,G=1.5e-9,T=160e-3,C=1e-12,dT=0.5e-3,I=65e-6)
            # self.ax.plot(t*1e+3,pfab*1e+6,lw=3,color="black",label=r"$\alpha=100,\beta=2$")

        if subject == "pshape2":
            X,Y,Z = self.cal_various_ab()
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, Z*1e+6, cmap='viridis')  # カラーマップを描画
            plt.colorbar(label='PH[uA]')  # カラーバーを追加
            plt.xlabel('alpha')
            plt.ylabel('beta')
            plt.title('Color Map of PH')
            plt.show()


        sfn = f"./{subject}.png"
        self.ax.legend(loc="best",fontsize=20)
        plt.show()
        self.fig.savefig(sfn,dpi=300)

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

    def Ipulse(self,t,L,Rth,R,beta,Pb,alpha,G,T,C,dT,I):
        return (self.tau_I(C,Pb,alpha,G,T)/self.risetime(L,Rth,R,beta,Pb,alpha,G,T,C)-1)*(self.tau_I(C,Pb,alpha,G,T)/self.falltime(L,Rth,R,beta,Pb,alpha,G,T,C)-1)*C*dT*(np.exp(-t/self.risetime(L,Rth,R,beta,Pb,alpha,G,T,C))-np.exp(-t/self.falltime(L,Rth,R,beta,Pb,alpha,G,T,C)))/((2+beta)*I*R*self.tau_I(C,Pb,alpha,G,T)**2*(1/self.risetime(L,Rth,R,beta,Pb,alpha,G,T,C)-1/self.falltime(L,Rth,R,beta,Pb,alpha,G,T,C)))


    # def whitenoise(self, sigma=1e-7, mean=0.0, t=10e-3, dt=1e-9):
        
    #     size = t/dt
    #     n = (mean + sigma*norm.rvs(size)) / np.sqrt(dt)
    #     return n

    # def pfunc(self,t,tau_r,tau_f,A):
    #     t0=1.025e-3
    #     return 0*(t<=t0)+A*(np.exp(-(t-t0)/tau_r)-np.exp(-(t-t0)/tau_f))*(t0<t) + whitenoise(t)


    # def LcritM(self):
    #     return self.lp(Pb,alpha,G,T)

    def fall_out(self):
        f = self.falltime(L=12.3e-9,Rth=4.261e-3,R=13.14e-3,beta=1.82,Pb=62e-12,alpha=100,G=1.43e-9,T=162.7e-3,C=0.825e-12)
        print(f)

    def axion_mass_limit_1(self,l,t,b):
        D = 0.808
        F = 0.462
        S = 0.5
        z = 0.56
        C = -1.19*(3*F-D+2*S)/3 + (D+F)*(1-z)/(1+z)
        a = np.sqrt(z)*1.3e+7/(1+z)

        t     = t                             # thickness [m]
        M     = l*l*t*7.874e-3*1e+6 # one pixel 57Fe mass [kg]
#       T     = 200                           # measurement time [day]
        # beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)     # background rate [counts/day/eV/pixel]
        nu    = 1.25                          # fudge factor 
        alpha = 0.69                           # conversion rate

        def R(delE,n,T,b,d): # 3sigma upper limit [events/day/kg]
            #beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)
            beta  = b*1e-3*(24*60*60)     # background rate [counts/day/eV/pixel]
            return (np.sqrt(9*beta*nu*T*n*delE+(9/2)**2)+9/2)/(T*M*n*alpha*d)

        def ma(delE,n,T,b,d): # axion mass limit 3sigma [eV]
            return pow(R(delE,n,T,b,d)/(C**4*3.1e+2),1/4)*a*1e-6 

        def f(delE,n,T,d):
            return a/ma(delE,n,T,b,d)

        def Rtoma(R):
            return pow(R/(C**4*3.1e+2),1/4)*a*1e-6 
        
        def matoR(ma):
            return 3.1e+2*(1e+6*ma/a)**4 * C**4
        

        n = np.linspace(1,1e+6,100000)
        fig, ax1 = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman' #フォントを変える
        plt.rcParams['mathtext.fontset'] = 'stix' #フォントを変える2
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 15 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['figure.subplot.bottom'] = 0.15

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("The number of pixels",fontsize=15)
        ax1.set_ylabel(r"$3\sigma \ \rm upper\ limit \ (events/day/kg)$",fontsize=15)
        ax1.set_xlim(10,1e+6)
        ax1.set_ylim(1e+4,2e+7)
        #plt.ylim(60,300)
        ax2 = ax1.secondary_yaxis('right', functions=(Rtoma, matoR))
        ax2.set_ylabel(r"${\rm Axion\ mass} \ m_a \ \rm (eV)$",fontsize=15)
        #ax2 = ax1.twinx()
        #ax2.set_yscale("log")
        #ax2.set_ylim(ax1.get_ylim())
        #ax2.yaxis.set_major_formatter(lambda x, pos: "{:.1f}".format(Rtoma(x)))
        #ax2.set_yticks([300,200,150,100,80,60])
        ax1.xaxis.grid(linestyle="dashed")

        ax1.plot(n,R(delE=40,n=n,T=100,b=0.8e-6,d=0.6),color="blue",lw=2,label=r"$\Delta E = 40\ {\rm eV}, \ b = 50 \ {\rm \%}$")
        ax1.plot(n,R(delE=20,n=n,T=100,b=0.8e-6,d=0.6),lw=2,color="red",label=r"$\Delta E = 30\ {\rm eV}, \ b = 50 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=0.8e-6,d=0.6),color="green",lw=2,label=r"$\Delta E = 15\ {\rm eV}, \ b = 50 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=1.6e-6,d=0.6),color="black",lw=2,label=r"$\Delta E = 15\ {\rm eV}, \ b = 100 \ {\rm \%}, {\rm Current\ design}$")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.25, 1),fontsize=12)

        xlim = ax1.get_xlim()
        for y in ax2.get_yticks():
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        for y in ax2.get_yticks(minor=True):
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        ax1.set_xlim(xlim)
        ax1.plot(xlim, (matoR(216), matoR(216)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(216+8),"Namba 2007")
        ax1.plot(xlim, (matoR(145), matoR(145)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(145+5),"Darbin+2011")
        #fig.tight_layout()
        #fig.subplots_adjust(hspace=.0)
        from matplotlib.transforms import Bbox as BB
        fig.savefig("1_ax.pdf",dpi=300,bbox_inches=BB([[0,-0.5],[13,6]]))
        plt.show()

    def axion_mass_limit_2(self,l,t,b):
        D = 0.808
        F = 0.462
        S = 0.5
        z = 0.56
        C = -1.19*(3*F-D+2*S)/3 + (D+F)*(1-z)/(1+z)
        a = np.sqrt(z)*1.3e+7/(1+z)

        t     = t                             # thickness [m]
        M     = l*l*t*7.874e-3*1e+6 # one pixel 57Fe mass [kg]
#       T     = 200                           # measurement time [day]
        # beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)     # background rate [counts/day/eV/pixel]
        nu    = 1.25                          # fudge factor 
        alpha = 0.69                           # conversion rate

        def R(delE,n,T,b,d): # 3sigma upper limit [events/day/kg]
            #beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)
            beta  = b*1e-3*(24*60*60)     # background rate [counts/day/eV/pixel]
            return (np.sqrt(9*beta*nu*T*n*delE+(9/2)**2)+9/2)/(T*M*n*alpha*d)

        def ma(delE,n,T,b,d): # axion mass limit 3sigma [eV]
            return pow(R(delE,n,T,b,d)/(C**4*3.1e+2),1/4)*a*1e-6 

        def f(delE,n,T,d):
            return a/ma(delE,n,T,b,d)

        def Rtoma(R):
            return pow(R/(C**4*3.1e+2),1/4)*a*1e-6 
        
        def matoR(ma):
            return 3.1e+2*(1e+6*ma/a)**4 * C**4
        

        n = np.linspace(1,1e+6,100000)
        fig, ax1 = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman' #フォントを変える
        plt.rcParams['mathtext.fontset'] = 'stix' #フォントを変える2
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 15 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['figure.subplot.bottom'] = 0.15

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("The number of pixels",fontsize=15)
        ax1.set_ylabel(r"$3\sigma \ \rm upper\ limit \ (events/day/kg)$",fontsize=15)
        ax1.set_xlim(10,1e+6)
        ax1.set_ylim(1e+4,2e+7)
        #plt.ylim(60,300)
        ax2 = ax1.secondary_yaxis('right', functions=(Rtoma, matoR))
        ax2.set_ylabel(r"${\rm Axion\ mass} \ m_a \ \rm (eV)$",fontsize=15)
        #ax2 = ax1.twinx()
        #ax2.set_yscale("log")
        #ax2.set_ylim(ax1.get_ylim())
        #ax2.yaxis.set_major_formatter(lambda x, pos: "{:.1f}".format(Rtoma(x)))
        #ax2.set_yticks([300,200,150,100,80,60])
        ax1.xaxis.grid(linestyle="dashed")

        ax1.plot(n,R(delE=30,n=n,T=100,b=0.48e-6,d=0.6),lw=2,color="blue",label=r"$\Delta E = 30\ {\rm eV}, \ b = 30 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=0.48e-6,d=0.6),color="green",lw=2,label=r"$\Delta E = 15\ {\rm eV}, \ b = 30 \ {\rm \%}$")
        ax1.plot(n,R(delE=12,n=n,T=100,b=0.48e-6,d=0.6),lw=2,color="red",label=r"$\Delta E = 12\ {\rm eV}, \ b = 30 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=1.6e-6,d=0.6),color="black",lw=2,label=r"$\Delta E = 15\ {\rm eV}, \ b = 100 \ {\rm \%}, {\rm Current\ design}$")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.25, 1),fontsize=12)

        xlim = ax1.get_xlim()
        for y in ax2.get_yticks():
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        for y in ax2.get_yticks(minor=True):
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        ax1.set_xlim(xlim)
        ax1.plot(xlim, (matoR(216), matoR(216)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(216+8),"Namba 2007")
        ax1.plot(xlim, (matoR(145), matoR(145)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(145+5),"Darbin+2011")
        #fig.tight_layout()
        #fig.subplots_adjust(hspace=.0)
        from matplotlib.transforms import Bbox as BB
        fig.savefig("2_ax.pdf",dpi=300,bbox_inches=BB([[0,-0.5],[13,6]]))
        plt.show()

    def axion_mass_limit_3(self,l,t,b):
        D = 0.808
        F = 0.462
        S = 0.5
        z = 0.56
        C = -1.19*(3*F-D+2*S)/3 + (D+F)*(1-z)/(1+z)
        a = np.sqrt(z)*1.3e+7/(1+z)

        t     = t                             # thickness [m]
        M     = l*l*t*7.874e-3*1e+6 # one pixel 57Fe mass [kg]
#       T     = 200                           # measurement time [day]
        # beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)     # background rate [counts/day/eV/pixel]
        nu    = 1.25                          # fudge factor 
        alpha = 0.69                           # conversion rate

        def R(delE,n,T,b,d): # 3sigma upper limit [events/day/kg]
            #beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)
            beta  = b*1e-3*(24*60*60)     # background rate [counts/day/eV/pixel]
            return (np.sqrt(9*beta*nu*T*n*delE+(9/2)**2)+9/2)/(T*M*n*alpha*d)

        def ma(delE,n,T,b,d): # axion mass limit 3sigma [eV]
            return pow(R(delE,n,T,b,d)/(C**4*3.1e+2),1/4)*a*1e-6 

        def f(delE,n,T,d):
            return a/ma(delE,n,T,b,d)

        def Rtoma(R):
            return pow(R/(C**4*3.1e+2),1/4)*a*1e-6 
        
        def matoR(ma):
            return 3.1e+2*(1e+6*ma/a)**4 * C**4
        

        n = np.linspace(1,1e+6,100000)
        fig, ax1 = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman' #フォントを変える
        plt.rcParams['mathtext.fontset'] = 'stix' #フォントを変える2
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 15 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['figure.subplot.bottom'] = 0.15

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("The number of pixels",fontsize=15)
        ax1.set_ylabel(r"$3\sigma \ \rm upper\ limit \ (events/day/kg)$",fontsize=15)
        ax1.set_xlim(10,1e+6)
        ax1.set_ylim(1e+4,2e+7)
        #plt.ylim(60,300)
        ax2 = ax1.secondary_yaxis('right', functions=(Rtoma, matoR))
        ax2.set_ylabel(r"${\rm Axion\ mass} \ m_a \ \rm (eV)$",fontsize=15)
        #ax2 = ax1.twinx()
        #ax2.set_yscale("log")
        #ax2.set_ylim(ax1.get_ylim())
        #ax2.yaxis.set_major_formatter(lambda x, pos: "{:.1f}".format(Rtoma(x)))
        #ax2.set_yticks([300,200,150,100,80,60])
        ax1.xaxis.grid(linestyle="dashed")

        ax1.plot(n,R(delE=100,n=n,T=100,b=1.6e-6,d=1),lw=2,color="blue",label=r"$\Delta E = 100\ {\rm eV}, \ d = 100 \ {\rm \%}$")
        ax1.plot(n,R(delE=30,n=n,T=100,b=1.6e-6,d=1),lw=2,color="green",label=r"$\Delta E = 30\ {\rm eV}, \ d = 100 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=1.6e-6,d=1),lw=2,color="red",label=r"$\Delta E = 15\ {\rm eV}, \ d = 100 \ {\rm \%}$")
        ax1.plot(n,R(delE=15,n=n,T=100,b=1.6e-6,d=0.6),color="black",lw=2,label=r"$\Delta E = 15\ {\rm eV},  \ d = 60 \ {\rm \%}, {\rm Current\ design}$")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.25, 1),fontsize=12)

        xlim = ax1.get_xlim()
        for y in ax2.get_yticks():
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        for y in ax2.get_yticks(minor=True):
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        ax1.set_xlim(xlim)
        ax1.plot(xlim, (matoR(216), matoR(216)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(216+8),"Namba 2007")
        ax1.plot(xlim, (matoR(145), matoR(145)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(145+5),"Darbin+2011")
        #fig.tight_layout()
        #fig.subplots_adjust(hspace=.0)
        from matplotlib.transforms import Bbox as BB
        fig.savefig("3_ax.pdf",dpi=300,bbox_inches=BB([[0,-0.5],[13,6]]))
        plt.show()

    def axion_mass_limit_4(self,l,t,b):
        D = 0.808
        F = 0.462
        S = 0.5
        z = 0.56
        C = -1.19*(3*F-D+2*S)/3 + (D+F)*(1-z)/(1+z)
        print(C)
        a = np.sqrt(z)*1.3e+7/(1+z)

        t     = t                             # thickness [m]
        M     = l*l*t*7.874e-3*1e+6 # one pixel 57Fe mass [kg]
#       T     = 200                           # measurement time [day]
        # beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)     # background rate [counts/day/eV/pixel]
        nu    = 1.25                          # fudge factor 
        alpha = 0.69                           # conversion rate

        def R(delE,n,T,b,d): # 3sigma upper limit [events/day/kg]
            #beta  = (b*1e+4*1e-3)*(24*60*60)*(l*l)
            beta  = b*1e-3*(24*60*60)     # background rate [counts/day/eV/pixel]
            return (np.sqrt(9*beta*nu*T*n*delE+(9/2)**2)+9/2)/(T*M*n*alpha*d)

        def ma(delE,n,T,b,d): # axion mass limit 3sigma [eV]
            return pow(R(delE,n,T,b,d)/(C**4*3.1e+2),1/4)*a*1e-6 

        def f(delE,n,T,d):
            return a/ma(delE,n,T,b,d)

        def Rtoma(R):
            return pow(R/(C**4*3.1e+2),1/4)*a*1e-6 
        
        def matoR(ma):
            return 3.1e+2*(1e+6*ma/a)**4 * C**4
        

        n = np.linspace(1,1e+6,100000)
        fig, ax1 = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman' #フォントを変える
        plt.rcParams['mathtext.fontset'] = 'stix' #フォントを変える2
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 15 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['figure.subplot.bottom'] = 0.15

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("The number of pixels",fontsize=15)
        ax1.set_ylabel(r"$3\sigma \ \rm upper\ limit \ (events/day/kg)$",fontsize=15)
        ax1.set_xlim(10,1e+6)
        ax1.set_ylim(1e+4,2e+7)
        #plt.ylim(60,300)
        ax2 = ax1.secondary_yaxis('right', functions=(Rtoma, matoR))
        ax2.set_ylabel(r"${\rm Axion\ mass} \ m_a \ \rm (eV)$",fontsize=15)
        #ax2 = ax1.twinx()
        #ax2.set_yscale("log")
        #ax2.set_ylim(ax1.get_ylim())
        #ax2.yaxis.set_major_formatter(lambda x, pos: "{:.1f}".format(Rtoma(x)))
        #ax2.set_yticks([300,200,150,100,80,60])
        ax1.xaxis.grid(linestyle="dashed")



        ax1.plot(n,R(delE=200,n=n,T=30,b=1.6e-6,d=0.6),"-.",color="black",lw=2,label=r"$\Delta E = 200 \ {\rm eV}, \ b = 1.6\times 10^{-6} \ {\rm count/s/pixel/keV} , \ 30 \ days $")
        ax1.plot(n,R(delE=200,n=n,T=100,b=1.6e-6,d=0.6),color="black",lw=2,label=r"$\Delta E = 200\ {\rm eV}, \ b = 1.6\times 10^{-6} \ {\rm count/s/pixel/keV} , \ 100 \ days $")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.25, 1),fontsize=12)

        xlim = ax1.get_xlim()
        for y in ax2.get_yticks():
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        for y in ax2.get_yticks(minor=True):
            ax1.plot(xlim, (matoR(y), matoR(y)), ls="--", color="gray", lw=0.5)
        ax1.set_xlim(xlim)
        ax1.plot(xlim, (matoR(216), matoR(216)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(216+8),"Namba 2007")
        ax1.plot(xlim, (matoR(145), matoR(145)), ls="-", color="black", lw=1)
        ax1.text(7e+4,matoR(145+5),"Darbin+2011")
        #fig.tight_layout()
        #fig.subplots_adjust(hspace=.0)
        from matplotlib.transforms import Bbox as BB
        fig.savefig("4_ax.pdf",dpi=300,bbox_inches=BB([[0,-0.5],[13,6]]))
        plt.show()