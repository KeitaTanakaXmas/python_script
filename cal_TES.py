import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

d_Au = 19320 ##[kg/m^3]
d_Ti = 4506 ##[kg/m^3]
d_Fe = 7874 ##[kg/m^3]
kb = 1.38e-23 ##[J/K]
eV = 1.60218e-19 ##[J]

def cal_Au_cap(l,h,T=100):
    V = l**2 * h ## [m^-3]
    T = T*1e-3 ##[K]
    C = (42*T**3 + 68*T)*V
    return C

def cal_Au_cap_Ax(wi,h,ov,T=100):
    V = 110e-6*110e-6*h + wi*ov*h ## [m^-3]
    T = T*1e-3 ##[K]
    C = (42*T**3 + 68*T)*V
    return C

def cal_Ti_cap(l,h,T=100):
    V = l**2 * h ## [m^-3]
    T = T*1e-3 ##[K]
    C = (2.5*T**3 + 97*T)*V
    return C

def cal_Fe_cap(l,h,T=100):
    V = l**2 * h ## [m^-3]
    T = T*1e-3 ##[K]
    C = (0.911e-4*T + 1.97e-7*T**3)*V*d_Fe*1e+3
    return C

def cal_Au_con(T=100):
    T = T*1e-3 ##[K]
    c = (42*T**3 + 68*T)/d_Au
    return c

def cal_Fe_con(T=100):
    T = T*1e-3 ##[K]
    c = (0.911e-4*T + 1.97e-7*T**3)*1e+3
    return c #[J/g/K]

def cal_mem_c(T,th_SiO2,th_SiNx,mem_area):
    # [J/(kg*K)]
    T = T * 1e+3
    c = (26.5*(49.44*T**1.22+0.207)*th_SiO2+(30*1.58-T**1.54)*th_SiNx/37)*mem_area
    return c

def cal_coper_c(T,V):
    C = (0.6943*T+0.04755*T**3)*V/(63.546e-3*8.96e+3)
    return C

def diffuse_time(material,T,k,x):
    if material == 'Au':
        c = cal_Au_con(T=T)
    elif material == 'Fe':
        c = cal_Fe_con(T=T)
    tau_diff = c*x**2/k
    print(f'{tau_diff*1e+6} usec')
    return tau_diff

def TES_dencity(Auh,Tih):
    d_TES = d_Au*Auh/(Auh+Tih) + d_Ti*Tih/(Auh+Tih)
    print(f"TES dencity = {d_TES} [kg/m^3]")
    return d_TES

def TES_heat_scale(x,Auh,Tih,T,G):
    T = T*1e-3 ##[K]
    c = (42*T**3 + 68*T)*(Auh)/(Auh+Tih) + (2.5*T**3 + 97*T)*(Tih)/(Auh+Tih)
    t = c*(x)**2/G
    print(f"TES vertical time scale = {t}")


def cal_TES_resolution(T,Tb,n,alpha,l,t):
    C = 0.0133e-3*7.874e+6*l*l*t 
    lp = alpha*(1-(Tb/T)**n)/n
    Sig = n*(1-(Tb/T)**(2*n+1))/((2*n+1)*(1-(Tb/T)**n))
    xi = 2*np.sqrt(np.sqrt(1+alpha*lp*Sig)/(alpha*lp))
    print(C)
    return 2.355*xi*np.sqrt(kb*T**2*C)/eV

def saturation_energy(T,C,alpha):
    return C*T/alpha/eV

def Au_Ti_con_cap(l,Auh,Tih,T=100):
    c_Au = cal_Au_cap(l=1,h=1,T=T)
    c_Ti = cal_Ti_cap(l=1,h=1,T=T)
    V_Au = l**2 * Auh
    V_Ti = l**2 * Tih
    c_con = ((c_Au/d_Au * V_Au) + (c_Ti/d_Ti * V_Ti))/(V_Au + V_Ti)
    print(f"{c_con} [J/(kg * K)]") 

def cal_TES_C(l,b,Auh,Tih,T=100):
    Au_C = cal_Au_cap(l=l,h=Auh,T=T) * b/l
    Ti_C = cal_Au_cap(l=l,h=Tih,T=T) * b/l
    print(f"TES Volume = {l*b * (Auh + Tih)} [m^3]")
    print(f"Au heat Capacity = {Au_C} [J/K]")
    print(f"Ti heat Capacity = {Ti_C} [J/K]")
    print(f"TES heat Capacity = {Au_C + Ti_C} [J/K]")

def source_temp(E,r,Tbath):
    T = Tbath + 3*E*1e+3*1.6e-19/(5.72e-4*19320*4*np.pi*r**3)
    print(f"Source Temperature : T = {T*1e+3} [mK] ")

def RT_emp(T,Tc=164.885e-3,A=18.086e-3,B=2.172e-3,C=27.686e-3):
    return A*np.arctan((T-Tc)/B)+C

def alpha_plot():
    x = np.linspace(0,170e-3,1000)
    R = RT_emp(T=x)
    return np.diff(np.log(R))/np.diff(np.log(x))

def tau_eff_inst(alpha,n,C,G):
    return C/(G*(1+alpha/n))

def del_T(C,G,alpha,n,tau_p,tau_m):
    tau_I = C/(G*(1-alpha/n))
    return ((1/tau_I - 1/tau_p)*np.exp(-tau_p/tau_m) + (1/tau_I - 1/tau_m)/np.exp(1))/(1/tau_p-1/tau_m)

def delI(tau_rise,tau_fall,C,G,Pb,alpha,beta,Ttes,Rtes,Ites,delT):
    t     = np.arange(0,1e-3,1e-9)
    tau   = C/G
    LI    = Pb*alpha/(G*Ttes)
    tau_I = tau/(1-LI)
    dI    = (tau_I/tau_rise - 1)*(tau_I/tau_fall - 1)*C*delT*(np.exp(-t/tau_rise)-np.exp(-t/tau_fall))/((2+beta)*Ites*Rtes*tau_I**2*(1/tau_rise-1/tau_fall))
    return t,dI

def plot_delI(tau_rise,tau_fall):
    rise_list = np.logspace(-10,-6,num=10)
    #rise_list = np.array([1e-6])
    for tau_rise in rise_list: 
        t,I = delI(tau_rise=tau_rise,tau_fall=tau_fall,C=1e-12,G=2.77e-9,Pb=124e-12,alpha=40,beta=0,Ttes=206e-3,Rtes=6.77e-3,Ites=135e-6,delT=1)
        plt.plot(t,I)
    plt.show()


def F_test(n1,n2,v1,v2):
    F = v2/v1
    print(F)
    p = stats.f.cdf(F,dfn=n2,dfd=n1)
    print(p)

def Au_strap_area(l):
    strap1 = 110e-6*(110e-6+l+125e-6)
    strap2 = 110e-6*110e-6+(110e-6+20e-6)*20e-6/2+20e-6*l+20e-6*100e-6
    Fe = 100e-6*100e-6
    print(f'Area of the strap1 = {strap1} m3')
    print(f'Area of the strap2 = {strap2} m3')
    print(f'Au strap Rate = {strap2/strap1*100} %')
    print(f'Fe Rate = {Fe/(strap1)*100} %')