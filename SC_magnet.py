import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sqrt, simplify, trigsimp, expand, latex, pi, lambdify
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from scipy import integrate

from Basic import Plotter


class MagneticField:

    def __init__(self):
        self.mu_0 = 1.25663753e-6 #[H/m] 

    def circle_Bz(self):
        self.a, self.r, self.z = symbols("a, r, z", real=True)

        self.Delta2 = (self.r - self.a)**2 + self.z**2
        self.m = -4*self.a*self.r/self.Delta2

        self.A = self.a/sqrt(self.Delta2) * (
            (1 - 2/self.m) * elliptic_k(self.m) + 2/self.m * elliptic_e(self.m)
        )

        #self.Br = simplify(-self.A.diff(self.z) * self.mu_0 / pi) # /I
        Bz = simplify((self.r * self.A).diff(self.r) / self.r * self.mu_0 / pi) # /I
        #self.Bz = lambdify(args,Bz,"numpy")
        Bz = simplify(expand(Bz))
        self.Bz = Bz
        #self.Binst = self.mu_0*((self.a**2-self.r**2)*elliptic_e(self.m)-(self.r+self.a)**2*elliptic_k(self.m))/(2*pi*np.abs(self.r-self.a)*(self.r+self.a)**2)


    def cal_circle_Bz(self,rc,d,x):
        return float(self.Bz.subs([(self.a,rc),(self.r,x),(self.z,d)]))

    def solenoid_Bz(self,rc,rm,w,x,n,t):
        Bz_sum = 0
        self.circle_Bz()
        for i in range(1,n+1):
            for j in range(1,t+1):
                rc_sum  = rc+(i-1)*rm*2
                d       = w+(j-1)*rm*2
                x       = x
                Bz_sum += self.cal_circle_Bz(rc=rc_sum,x=x,d=d)
        self.Bz_I = Bz_sum
        print(self.Bz_I)

    def solenoid_Bz_current(self,rc,rm,w,x,n,t):
        self.solenoid_Bz(rc,rm,w,x,n,t)
        for e,i in enumerate(np.linspace(1e-6,1e-3,10)):
            if e == 0 :
                I = np.array([i])
                B = np.array([self.Bz_I * i])
            else :
                I = np.append(I,i)
                B = np.append(B,self.Bz_I * i)
        return B,I

    def ring(self,I,z,a,n,offset,deplicate,r):

        def single(r):
            mu = 1.25663706212*10**-6 # N A**-2
            phi = 0 # rad
            
            inte = integrate.quad(lambda x, r: (mu*I*n)/(4*np.pi) * (a-(r-offset)*np.cos(x-phi))*a / ((r-offset)**2+(z-deplicate)**2+a**2-2*(r-offset)*a*np.cos(x-phi))**(3/2)
            , 0, 2*np.pi, args=r)[0]
            
            return inte
        
        Bz_sum = single(r)
        return Bz_sum

    # def ring(self,I,z,a,n,offset,deplicate,r):

    #     def single(I):
    #         mu = 1.25663753e-6 # N A**-2
    #         phi = 0 # rad
            
    #         inte = integrate.quad(lambda x, r: (mu*I*n)/(4*np.pi) * (a-(r-offset)*np.cos(x-phi))*a / ((r-offset)**2+(z-deplicate)**2+a**2-2*(r-offset)*a*np.cos(x-phi))**(3/2)
    #         , 0, 2*np.pi, args=I)[0]
            
    #         return inte
        
    #     Bz_sum = single(I)
    #     return Bz_sum


    # def ring_prop(self,layer,wire_dia):
    #     I = np.arange(1*10**-7, 2*10**-3, 1*10**-7)
    #     Bz_sum = np.vectorize(self.ring)
    #     r = 0.0
    #     Bz_list = []
    #     #a_list = np.arange(a, a + wire_dia * (layer+1), wire_dia)
    #     deplicate_list = np.arange(0, wire_dia * (n+1), wire_dia)
    #     for i in range(len(deplicate_list)):
    #         Bz = Bz_sum(I, z, a, n, offset, deplicate_list[i], r)
    #         Bz_list.append(Bz)

    #     Bz_list = np.array(Bz_list)

    def solenoid(self,z,a,r,turn,layer,wire_dia):
        n = 1
        offset = 0
        r = 0.0
        Bz_list = []
        I = np.linspace(1e-6, 1e-3, 50)
        Bz_sum = np.vectorize(self.ring)
        a_list = np.arange(0, wire_dia * (layer+1), wire_dia)
        self.deplicate_list = np.arange(0, wire_dia * (turn+1), wire_dia)
        print(a_list)
        print(self.deplicate_list)
        for i in range(len(a_list)):
            print(f'layer num = {i}')
            for j in range(len(self.deplicate_list)):
                print(f'turn num = {j}')
                Bz = Bz_sum(I, z, a+a_list[i], n, offset, self.deplicate_list[j], r)
                print(f'{Bz}')
                Bz_list.append(Bz)

        Bz_list = np.array(Bz_list)

        return Bz_list,I

    def plot_SC(self,rc,rm,w,x,n,t):
        P = Plotter()
        x = 0
        B0,I = self.solenoid_Bz_current(rc,rm,w,x,n,t)
        x = 1.8e-3
        Bedge,I = self.solenoid_Bz_current(rc,rm,w,x,n,t)
        x = 30e-3
        Bsq,I = self.solenoid_Bz_current(rc,rm,w,x,n,t)
        xname = r"$\rm Current \ (A)$"
        yname = r"$\rm Magnetic \ field \ (\mu T)$"
        P.plotting(I,B0*1e+6,xname=xname,yname=yname,label=r'$\rm 0\ mm$')
        P.plotting(I,Bedge*1e+6,new_window=False,color='Red',label=r'$\rm 1.8\ mm$')
        P.plotting(I,Bsq*1e+6,new_window=False,color='Black',label=r'$\rm 20\ mm$')
        plt.show()

    def plot_IC(self,z,a,turn,layer,wire_dia,rb):
        P = Plotter()
        xname = r"$\rm Current \ (mA)$"
        yname = r"$\rm Magnetic \ field \ (\mu T)$"
        r = 0
        Bz_cen, I = self.solenoid(z,a,r,turn,layer,wire_dia)
        P.plotting(I*1e+3,np.sum(Bz_cen*1e+6, axis=0),label='0 mm (Chip center)',xname=xname,yname=yname)
        print(np.sum(Bz_cen*1e+6, axis=0))
        #r = rb
        #Bz_edge, I = self.solenoid(z,a,r,turn,layer,wire_dia)
        #P.plotting(I*1e+3,np.sum(Bz_edge*1e+6, axis=0),new_window=False,color='Red',label='1.8 mm (Chip edge)')
        #r = 20e-3
        #Bz_list, I = self.solenoid(z,a,r,turn,layer,wire_dia)
        #P.plotting(I*1e+6,np.sum(Bz_list*1e+6, axis=0),new_window=False,color='black',label='20 mm (SQUID position)')
        #P.plotting(np.sum(Bz_edge*1e+6, axis=0),np.sum(Bz_cen*1e+6, axis=0)-np.sum(Bz_edge*1e+6, axis=0),color='black',label='residual')

    def circural_Bz_instant(self,I,r,n,wire_dia):
        n = int(n/2)
        Bz_sum = 0
        for i in range(1,n+1):
            d = wire_dia/2 + (i-1) * wire_dia
            r = r
            Bz_sum += 2 * self.mu_0*I*d**2/(2*(r**2+d**2)**(3/2))
        return Bz_sum

    def solenoid_Bz_instant(self,I,n,wire_dia):
        return self.mu_0*I/wire_dia

    def solenoid_vs_circural(self,I,r,n,wire_dia):
        circ_Bz = self.circural_Bz_instant(I,r,n,wire_dia)
        solenoid_Bz = self.solenoid_Bz_instant(I,n,wire_dia)
        print(f'circ Bz = {circ_Bz*1e+6} uT')
        print(f'solenoid Bz = {solenoid_Bz*1e+6} uT')




