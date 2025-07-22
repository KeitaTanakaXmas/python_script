import numpy as np
from scipy.special import iv, kv
from scipy.constants import *
import matplotlib.pyplot as plt

class IdealPlot:
    def __init__(self) -> None:
        pass


    def RSJ(T,I,Rn):
        L = 100e-6
        lambda_r = 80e-9
        phi_0 = h/(2*e)
        F = 1e+10 #L*d*fr**2/(3*np.pi*mu*lambda_r)
        Tc_i = 90e-3
        xi_i = 738e-9 ## Sadleir
        xi    = xi_i/np.power(np.abs(T/Tc_i-1),1/2)
        Ic    = F*np.exp(-L/xi)/xi
        #print(Ic) 
        x     = I/Ic
        gamma = hbar*Ic/(2*e*T)
        print(iv(1+(1j*gamma*x).imag,gamma))
        print((iv(1+(1j*gamma*x).imag,gamma)/iv((1j*gamma*x).imag,gamma)).imag)
        R = Rn * (1+(iv(1+(1j*gamma*x).imag,gamma)/iv((1j*gamma*x).imag,gamma)).imag/x)
        return R

    
    def RSJ_ab():
        pass


    def modified_B(self,y):
        b = 3
        v = y + (iv(b,1+(1j*b*y))/iv(b,(1j*b*y))).imag
        return v


    def check(self):
        y = np.arange(0,1.5,0.01)
        v = self.modified_B(y)
        plt.plot(v,y,'-.')
        plt.show()


    def RSJ_Ic(Ic0,w,L):
        B = np.arange(-10e-6,10e-6,0.1e-6)
        Phi_0 = 2.07e-15 #Wb
        B0 = Phi_0/(w*L)
        RSJ_Ic = Ic0*np.abs(np.sin(np.pi*B/B0)/(np.pi*B/B0))


    def Ic_conv(self,Ic0,Bs):
        w = 50e-6
        L = 50e-6        
        B = np.arange(-10e-6,10e-6,0.1e-6)
        Phi_0 = 2.07e-15 #Wb
        B0 = Phi_0/(w*L)
        RSJ_Ic = Ic0*np.abs(np.sin(np.pi*B/B0)/(np.pi*B/B0))
        fluid_Ic = Ic0*(1-np.abs(B/Bs))
        # fluid_Ic /= np.max(RSJ_Ic) 
        # RSJ_Ic /= np.max(RSJ_Ic) 
        plt.plot(B*1e+6,RSJ_Ic*1e+6,color='red',label='RSJ model')
        plt.plot(B*1e+6,fluid_Ic*1e+6,'-.',color='blue',label='Two-fluid model')
        plt.xlabel(r'$\rm Applied\ Magnetic\ Field\ (\mu T)$')
        plt.ylabel(r'$\rm Critical\ Current (\mu A)$')
        plt.legend()
        plt.show()
