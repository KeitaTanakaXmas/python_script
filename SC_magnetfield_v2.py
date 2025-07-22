import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import integrate
from Basic import Plotter
from scipy.optimize import curve_fit
from lmfit import Model
from matplotlib.patches import Rectangle

class MagneticField:
    """
    Calculating Magnetfield for design of Superconducting Coil.

    """
    def __init__(self):
        self.mu_0 =  1.25663753e-6 #[H/m] 

    def single_circural_Bz(self,coil_radius,z,r):
        """
        Calculate Bz made by circural current.
        coil_radius : Radius of circule [m]
        z           : z distance from coil [m]
        r           : Distance from origin [m]
        """
        Bz = integrate.quad(lambda theta: (self.mu_0 / (4 * np.pi)) * (coil_radius * (coil_radius - r * np.cos(theta))) / (r**2 + z**2 + coil_radius**2 - 2 * r * coil_radius * np.cos(theta))**(3/2),0,2*np.pi)[0]
        return Bz

    def symmetric_circural_Bz(self,coil_radius,z_pulse,z_minus,r):
        """
        Calculate Bz made by symmetric circural current.
        Returning Bz [T/A].
        coil_radius : Radius of circule [m]
        z_pulse     : z distance from coil [m]
        z_minus     : z distance from coil [m]
        r           : Distance from origin [m]
        """
        Bz = integrate.quad(lambda theta: (self.mu_0 / (4 * np.pi)) * (coil_radius * (coil_radius - r * np.cos(theta))) / (r**2 + z_pulse**2 + coil_radius**2 - 2 * r * coil_radius * np.cos(theta))**(3/2) + (self.mu_0 / (4 * np.pi)) * (coil_radius * (coil_radius - r * np.cos(theta))) / (r**2 + z_minus**2 + coil_radius**2 - 2 * r * coil_radius * np.cos(theta))**(3/2),0,2*np.pi)[0]
        return Bz

    def solenoid_Bz(self,coil_radius,material_radius,r,layer,turn,z,**kwargs):
        """
        Calculate Bz made by solenoid coil.
        coil_radius     : Radius of circule [m]
        material_radius : Radius of the coil material [m]
        z               : z distance from coil [m]
        r               : Distance from origin [m]
        layer           : Number of layer (z direction) [int]
        turn            : Number of turn  (r direction) [int]
        """
        single_Bz_function = np.vectorize(self.single_circural_Bz)
        turn_length = material_radius + (turn - 1) * material_radius * 2
        layer_length = material_radius + (layer - 1) * material_radius * 2
        total_radius = turn_length + coil_radius
        z_helmholtz = (total_radius - turn_length/2)/2
        self.z = z
        total_z_length =  layer_length + self.z
        if 'z_fraction' in kwargs:
            frac = kwargs['z_fraction']
            self.z = self.z + frac
            print(f'Seted z fraction = {frac*1e+3} mm')
            print(f'z = {self.z*1e+3} mm')
        print('--------------------------------')
        print('Solenoid coil properities ...')
        print(f'Total number of turn = {turn}')
        print(f'Total number of layer = {layer}')
        print(f'Turn length = {turn_length*1e+3} mm')
        print(f'Layer length = {layer_length*1e+3} mm')
        print(f'z from first turn coil = {self.z*1e+3} mm')
        print(f'Total z length = {total_z_length*1e+3} mm')
        print(f'Total coil radius = {total_radius*1e+3}')
        for i in range(1,layer+1):
            #print('--------------------------------')
            #print(f'layer num = {i}')
            for j in range(1,turn+1):
                    #print(f'turn num = {j}')
                    coil_radius_multi = coil_radius+(i-1)*material_radius*2
                    z_multi = self.z+(j-1)*material_radius*2
                    if i == 1 and j == 1 :
                        Bz  = single_Bz_function(coil_radius_multi,z_multi,r)
                    else :
                        Bz += single_Bz_function(coil_radius_multi,z_multi,r)
        return Bz

    def helmholtz_Bz(self,coil_radius,material_radius,r,layer,turn,**kwargs):
        """
        Calculate Bz made by helmholtz coil.
        coil_radius     : Radius of circule [m]
        material_radius : Radius of the coil material [m]
        z               : z distance from coil [m]
        r               : Distance from origin [m]
        layer           : Number of layer (z direction) [int]
        turn            : Number of turn  (r direction) [int]
        """
        if 'auto_tune' in kwargs:
            auto_tune = kwargs['auto_tune']
        else:
            auto_tune = True 

        symmetric_Bz_function = np.vectorize(self.symmetric_circural_Bz)

        turn_length = turn * material_radius * 2
        layer_length = layer * material_radius * 2
        if auto_tune == True :
            total_radius = turn_length + coil_radius
            z_helmholtz = (total_radius - turn_length/2)/2
            self.z = z_helmholtz
            print('This is helmholtz coil')
        else :
            self.z = kwargs['z']
            print('z is selected by user.')
            print('WARNING!! : This is NOT helmholtz coil')

        if 'z_fraction' in kwargs:
            frac = kwargs['z_fraction']
            self.z = self.z - frac
            print(f'Seted z fraction = {frac*1e+3} mm')
            print(f'z = {self.z*1e+3} mm')

        total_z_length =  layer_length + self.z + 1e-3
        print('--------------------------------')
        print('Helmholtz type coil properities ...')
        print(f'Total number of turn = {turn}')
        print(f'Total number of layer = {layer}')
        print(f'Turn length = {turn_length*1e+3} mm')
        print(f'Layer length = {layer_length*1e+3} mm')
        print(f'z from first turn coil = {self.z*1e+3} mm')
        print(f'Total z length = {total_z_length*1e+3} mm')
        print(f'Total coil radius = {total_radius*1e+3}')
        for i in range(1,layer+1):
            #print('--------------------------------')
            #print(f'layer num = {i}')
            for j in range(1,turn+1):
                    #print(f'turn num = {j}')
                    coil_radius_multi = coil_radius+material_radius+(i-1)*material_radius*2
                    z_multi_pulse = self.z+material_radius+(j-1)*material_radius*2
                    if 'z_fraction' in kwargs:
                        z_multi_minus = -(self.z+frac*2+material_radius+(j-1)*material_radius*2)
                    else:
                        z_multi_minus = -(self.z+material_radius+(j-1)*material_radius*2)
                    Material_Length = 2 * np.pi * coil_radius_multi 
                    if i == 1 and j == 1 :
                        Bz  = symmetric_Bz_function(coil_radius_multi,z_multi_pulse,z_multi_minus,r)
                        self.Material_Lengths = Material_Length
                    else :
                        Bz += symmetric_Bz_function(coil_radius_multi,z_multi_pulse,z_multi_minus,r)
                        self.Material_Lengths += Material_Length
        print(f'Material Length = {self.Material_Lengths} m')
        self.cfc = np.sqrt((total_z_length*1e+3)**2+(total_radius*1e+3)**2)
        return Bz

    def residual_50uT(self,coil_radius,layer,turn,coil_type,material_radius):
        if coil_type == 'helmholtz':
            r = 0
            Bz_cen = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            r = 1.8e-3
            Bz_edge = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
        elif coil_type == 'solenoid':
            r = 0
            Bz_cen = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn)
            r = 1.8e-3
            Bz_edge = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn)
        I_50uT = 50e-6/Bz_cen
        resid = (Bz_cen - Bz_edge) * I_50uT
        print('---------------------------------')
        print(f'I(50uT) = {I_50uT*1e+3}')
        print(f'Bz center = {Bz_cen*1e+6}')
        print(f'Bz edge = {Bz_edge*1e+6}')
        print(f'Residual = {resid*1e+6} uT')
        return np.abs(resid),I_50uT

    def plot_coil_Bz_with_residual(self,coil_radius,coil_type,material_radius,r,layer,turn,**kwargs):
        if 'z' in kwargs:
            z = kwargs['z']
        else:
            z = None
        P = Plotter()
        if coil_type == 'helmholtz' :
            r = 0.0
            Bz_cen   = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            r = 1.8e-3
            Bz_edge  = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            r = 22e-3
            Bz_squid = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            z = self.z
        elif coil_type == 'solenoid' :
            r = 0.0
            z = 7e-3
            Bz_cen   = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z)
            r = 1.0e-3
            z = 7e-3 
            Bz_edge  = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z)
            r = 0.0e-3
            z = 6.4e-3 
            Bz_edge1  = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z)
            r = 1.0e-3
            z = 7.6e-3 
            Bz_edge2  = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z)
            r = 0
            z = 37e-3
            Bz_squid = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z)
            z = self.z
        I = np.linspace(1e-3,40e-3,50)
        xname = r"$\rm Current \ (mA)$"
        yname = r"$\rm Magnetic \ field \ (\mu T)$"
        z = 7.0e-3
        title = f'{coil_type} coil, z = {z*1e+3} mm, layer = {layer}, turn = {turn}'
        resid = (Bz_cen - Bz_edge)*1e+6*I
        P.plotting(I*1e+3,Bz_cen*1e+6*I,y_residual=resid,label='0 mm (Chip center)',xname=xname,yname=yname,title=title,scatter=True,style='residual')
        P.plotting(I*1e+3,Bz_edge*1e+6*I,label='1.8 mm (Chip edge)',new_window=False,color='red',scatter=True,style='residual')
        P.plotting(I*1e+3,Bz_squid*1e+6*I,label='30 mm (SQUID position)',new_window=False,color='black',scatter=True,style='residual')
        self.resid_plot()
        # self.ax.scatter(I*1e+3,Bz_cen*1e+6*I,color='Blue',label='r = 0 mm, z = 7 mm (position1)')
        # self.ax2.scatter(I*1e+3,resid,color='Black')
        # self.ax.scatter(I*1e+3,Bz_edge*1e+6*I,color='Red',label='r = 1.0 mm, z = 7 mm (position2)')
        # self.ax.scatter(I*1e+3,Bz_edge1*1e+6*I,color='Green',label='r = 0 mm, z = 6.4 mm (position3)')
        # self.ax.scatter(I*1e+3,Bz_edge2*1e+6*I,color='Orange',label='r = 1.0 mm, z = 7.6 mm (position4)')
        # self.ax.scatter(I*1e+3,Bz_squid*1e+6*I,color='Black',label='r = 0 mm, z = 37 mm (SQUID position)')
        self.ax.legend(loc='best',fontsize=15)
        self.ax2.set_xlabel(xname,fontsize=20)
        self.ax.set_ylabel(yname,fontsize=20)
        self.ax2.set_ylabel('Residual',fontsize=20)
        self.ax.set_title(title,fontsize=20)
        #self.ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        self.ax2.yaxis.set_major_formatter(ptick.FormatStrFormatter('%.0e'))
        self.ax2.tick_params(axis='y',labelsize=10)
        self.fig.savefig('coil_Bz.png',dpi=300)
        plt.show()

    def plot_coil_Bz_with_residual_z(self,coil_radius,coil_type,material_radius,r,layer,turn,**kwargs):
        P = Plotter()
        if coil_type == 'helmholtz' :
            r = 0.0
            res, I_50uT = self.residual_50uT(coil_radius,layer,turn,coil_type='helmholtz',material_radius=material_radius)
            Bz_cen   = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            z_fraction = np.linspace(-3e-3,3e-3,100)
            B_z_residual = []
            B_z_frac = []
            for i in z_fraction:
                B = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn,z_fraction=i)
                B_z_residual.append((B-Bz_cen)*I_50uT)
                B_z_frac.append(B*I_50uT)
            B_z_residual = np.array(B_z_residual)
        elif coil_type == 'solenoid' :
            r = 0.0
            res, I_50uT = self.residual_50uT(coil_radius,layer,turn,coil_type='solenoid',material_radius=material_radius)
            Bz_cen   = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn)
            z_fraction = np.linspace(-1e-3,1e-3,41)
            B_z_residual = []
            for i in z_fraction:
                B = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z_fraction=i)
                B_z_residual.append((B-Bz_cen)*I_50uT)
            B_z_residual = np.array(B_z_residual)
        xname = r"$\rm z (mm)$"
        yname = r"$\rm Bz(z=z) \ (\mu T)$"
        title = f'{coil_type} coil, layer = {layer}, turn = {turn}'
        print(z_fraction)
        print(B_z_residual)
        B_z_frac = np.array(B_z_frac)
        self.single_plot()
        self.ax.scatter(z_fraction*1e+3,B_z_frac*1e+6,color='red')
        self.ax.set_xlabel(xname,fontsize=20)
        self.ax.set_ylabel(yname,fontsize=20)
        self.ax.set_title(title,fontsize=20)
        self.fig.savefig('coil_Bz_zfraction.png',dpi=300)
        plt.show()
        plt.show()

    def plot_coil_Bz_with_residual_r(self,coil_radius,coil_type,material_radius,layer,turn,**kwargs):
        P = Plotter()
        if coil_type == 'helmholtz' :
            r = 0.0
            res, I_50uT = self.residual_50uT(coil_radius,layer,turn,coil_type='helmholtz',material_radius=material_radius)
            Bz_cen   = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
            z_fraction = np.linspace(-3e-3,3e-3,100)
            B_z_residual = []
            for i in z_fraction:
                B = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn,z_fraction=i)
                B_z_residual.append((B-Bz_cen)*I_50uT)
            B_z_residual = np.array(B_z_residual)
        elif coil_type == 'solenoid' :
            r = 0.0
            res, I_50uT = self.residual_50uT(coil_radius,layer,turn,coil_type='solenoid',material_radius=material_radius)
            Bz_cen   = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn)
            z_fraction = np.linspace(-1e-3,1e-3,41)
            B_z_residual = []
            for i in z_fraction:
                B = self.solenoid_Bz(coil_radius,material_radius,r,layer,turn,z_fraction=i)
                B_z_residual.append((B-Bz_cen)*I_50uT)
            B_z_residual = np.array(B_z_residual)
        xname = r"$\rm z (mm)$"
        yname = r"$\rm Bz(z=0)-Bz(z=z) \ (\mu T)$"
        title = f'{coil_type} coil, layer = {layer}, turn = {turn}'
        print(z_fraction)
        print(B_z_residual)
        P.plotting(z_fraction*1e+3,B_z_residual*1e+6,color='red',scatter=True,xname=xname,yname=yname,title=title)
        plt.show()

    def plot_coil_Bz_r(self,coil_radius,coil_type,material_radius,r,layer,turn,**kwargs):
        if 'z' in kwargs:
            z = kwargs['z']
        else:
            z = None
        P = Plotter()
        r_all = np.arange(0,50e-3,1e-3)
        Bz_r = []
        if coil_type == 'helmholtz' :
            for r in r_all:
                Bz   = self.helmholtz_Bz(coil_radius,material_radius,r,layer,turn)
                Bz_r.append(Bz)
            z = self.z
        elif coil_type == 'solenoid' :
            for r in r_all:
                Bz   = self.solenoid_Bz(coil_radius,material_radius,z,r,layer,turn)
                Bz_r.append(Bz)
            z = self.z
        Bz_r = np.array(Bz_r)
        I = np.linspace(1e-3,40e-3,50)
        xname = r"$\rm position \ (mm)$"
        yname = r"$\rm Magnetic \ field \ (\mu T)$"
        title = f'{coil_type} coil, z = {z*1e+3} mm, layer = {layer}, turn = {turn}'
        I_50uT = 50e-6/Bz_r[0]
        self.single_plot()
        self.ax.scatter(r_all*1e+3,Bz_r*1e+6*I_50uT,color='Blue')
        self.ax.set_xlabel(xname,fontsize=20)
        self.ax.set_ylabel(yname,fontsize=20)
        self.ax.set_title(title,fontsize=20)
        self.fig.savefig('coil_Bz_r.png',dpi=300)
        plt.show()

    def plot_coil_50uT(self,coil_radius,material_radius,layer_max,turn_max):
        P = Plotter()
        resid = []
        lay = []
        tur = []
        l = []
        for e_lay,layer in enumerate(range(1,turn_max)):
            for e_tur,turn in enumerate(range(1,layer_max)):
                residual,I_50uT = self.residual_50uT(coil_radius,layer,turn,coil_type="helmholtz",material_radius=material_radius)
                resid.append(residual)
                print('-----------------------------------------------------')
                lay.append(layer)
                tur.append(turn)
                l.append(self.cfc)
        resid = np.array([resid])*1e+6
        layer_grid = np.arange(1,layer_max,1)
        turn_grid = np.arange(1,turn_max,1)
        X, Y = np.meshgrid(turn_grid,layer_grid)
        extent = (X.min(), X.max(), Y.min(), Y.max())
        resid = np.reshape(resid,[layer_grid.shape[0],turn_grid.shape[0]])
        l     = np.reshape(l,[layer_grid.shape[0],turn_grid.shape[0]])
        fig = plt.figure(figsize=(8,6))
        ax  = plt.subplot(111)
        ax.set_xlabel('Layer number',fontsize=20)
        ax.set_ylabel('Turn number',fontsize=20)
        cont=ax.contour(X,Y,resid,  levels=[0.05,0.1,0.5], colors=['red'],linestyles=['--'])
        cont.clabel(fmt='%1.2fuT', fontsize=14)
        im = ax.imshow(resid,cmap='viridis',origin='lower',extent=extent)
        ax.grid(False)
        fig.colorbar(im,orientation="vertical").set_label(label='Residual [uT]',size=15)
        print(resid)
        l=np.array(l)
        print(l)
        fac = 10
        condition = np.kron(l < 22, np.ones((fac,fac)))
        print(condition)
        ax.contour(condition, levels=[0.5], colors=['black'],extent=extent,linestyles=['-.'])
        fig.savefig('50uT_resid.png',dpi=300)
        plt.show()

    def resid_plot(self):
        self.fig = plt.figure(figsize=(8,6))
        self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
        self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
        self.ax  = self.fig.add_subplot(self.gs1[:,:])
        self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
        self.ax.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")
        self.ax2.set_ylabel("Residual",fontsize=20)
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()

    def single_plot(self):
        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle="dashed")
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()


class Measurement:

    def __init__(self):
        pass

    def linear(self,x,a):
        return a*x

    def loaddata(self,filename):
        f = np.loadtxt(filename)
        self.I = f[:,0]
        self.Vg = -f[:,1] * 1e-2 #[V] -> [T]
        #Vg = Vg - Vg[int((len(Vg)-1)/2)]
        self.Vg_err = f[:,2]/(np.sqrt(100)) * 1e-2 #[V] -> [T]
        #Vg_err = np.sqrt(Vg_err**2+Vg_err[int((len(Vg)-1)/2)]**2)

    def plot_mg(self,filename):
        f = np.loadtxt(filename)
        I = f[:,0]
        V = f[:,1]
        print(len(V))
        V = V - V[int((len(V)-1)/2)]
        Verr = f[:,2]
        Verr = np.sqrt(Verr**2+Verr[int((len(V)-1)/2)]**2)
        Vg = f[:,3] * 1e-2 #[V] -> [T]
        Vg = Vg - Vg[int((len(Vg)-1)/2)]
        Vg_err = f[:,4] * 1e-2 #[V] -> [T]
        Vg_err = np.sqrt(Vg_err**2+Verr[int((len(Vg)-1)/2)]**2)

        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r"$\rm Current \ (mA)$")
        self.ax.set_ylabel(r"$\rm Magnetic \ field \ (\mu T)$")
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        self.ax.errorbar(I*1e+3,Vg*1e+6,Vg_err*1e+6,markeredgecolor = "blue", color='blue',markersize=6,fmt="o",ecolor="blue",label='Data')

        Bz_cen   = MagneticField().helmholtz_Bz(coil_radius=15e-3,material_radius=0.22e-3,r=0,layer=12,turn=3)
        I_coil = np.arange(-80e-3,80e-3+1e-3,1e-3)
        self.ax.plot(I_coil*1e+3,Bz_cen*I_coil*1e+6,color='black',label='Design value')
        self.ax.legend()

        plt.show()

    def plot_gaus(self,filename):
        self.loaddata(filename)
        I = self.I
        Vg = self.Vg
        Vg_err = self.Vg_err
        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r"$\rm Current \ (mA)$")
        self.ax.set_ylabel(r"$\rm Magnetic \ field \ (\mu T)$")
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        self.ax.errorbar(I*1e+3,Vg*1e+6,Vg_err*1e+6,markeredgecolor = "blue", color='blue',markersize=6,fmt="o",ecolor="blue",label='Data')

        Bz_cen   = MagneticField().helmholtz_Bz(coil_radius=15e-3,material_radius=0.22e-3,r=0,layer=12,turn=3)
        I_coil = np.arange(-100e-3,100e-3+1e-3,1e-3)
        self.ax.plot(I_coil*1e+3,Bz_cen*I_coil*1e+6,color='black',label='Design value')
        self.ax.legend()
        self.fig.savefig(f'{filename[:-4]}.png',dpi=300)

        plt.show()

    def fit_cal(self,filename):
        self.loaddata(filename)
        self.model = Model(self.linear) 

        result = self.model.fit(self.Vg,x=self.I,weights=1/self.Vg_err,a=1)
        print(result.fit_report())
        self.I_fit = np.linspace(np.min(self.I),np.max(self.I),1000)
        self.fit_res = self.linear(self.I,result.best_values["a"])
        a = result.best_values["a"]
        a_err = result.params['a'].stderr
        self.resid = (self.Vg - self.fit_res)

        self.fig = plt.figure(figsize=(8,6))
        self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
        self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
        self.ax  = self.fig.add_subplot(self.gs1[:,:])
        self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
        self.ax.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")

        self.ax2.set_xlabel(r"$\rm Current \ (mA)$")
        self.ax.set_ylabel(r"$\rm Magnetic \ field \ (\mu T)$")
        self.ax2.set_ylabel(r'$\rm Residual \ (\mu T)$')

        Bz_cen   = MagneticField().helmholtz_Bz(coil_radius=15e-3,material_radius=0.22e-3,r=0,layer=12,turn=3)
        I_coil = np.arange(-100e-3,100e-3+1e-3,1e-3)
        self.ax.plot(I_coil*1e+3,Bz_cen*I_coil*1e+6,color='red',label='Design value')

        self.ax.errorbar(self.I*1e+3,self.Vg*1e+6,self.Vg_err*1e+6,markeredgecolor = "blue", color='blue',markersize=3,fmt="o",ecolor="blue",label='Data')
        self.ax.plot(self.I*1e+3,self.fit_res*1e+6,color='Black',label='fit')
        self.ax2.errorbar(self.I*1e+3,self.resid*1e+6,self.Vg_err*1e+6,markeredgecolor = "blue", color='blue',markersize=3,fmt="o",ecolor="blue")

        self.ax.plot(self.I*1e+3,self.linear(self.I,a+a_err)*1e+6,'-.',color='Black')
        self.ax.plot(self.I*1e+3,self.linear(self.I,a-a_err)*1e+6,'-.',color='Black')
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        self.ax.legend()
        plt.show()
        self.fig.savefig(f'{filename[:-4]}.png',dpi=300)


        plt.show()

    def plot_mg_double(self,filename,filename2):
        self.loaddata(filename)
        I = self.I
        Vg = self.Vg
        Vg_err = self.Vg_err
        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r"$\rm Current \ (mA)$")
        self.ax.set_ylabel(r"$\rm Magnetic \ field \ (\mu T)$")
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        self.ax.errorbar(I*1e+3,Vg*1e+6,Vg_err*1e+6,markeredgecolor = "blue", color='blue',markersize=6,fmt="o",ecolor="blue",label='normal')

        self.loaddata(filename2)
        I = self.I
        Vg = self.Vg
        Vg_err = self.Vg_err
        self.ax.errorbar(I*1e+3,Vg*1e+6,Vg_err*1e+6,markeredgecolor = "red", color='red',markersize=6,fmt="o",ecolor="red",label='with 55Fe source')

        self.ax.legend()
        self.fig.savefig(f'{filename2[:-4]}.png',dpi=300)

        plt.show()

    def cal_mg(self,filename):
        def linear(x,a):
            return a*x
        f = np.loadtxt(filename)
        I = f[:,0]
        V = f[:,1]
        print(len(V))
        V = V - V[int((len(V)-1)/2)]
        Verr = f[:,2]
        Verr = np.sqrt(Verr**2+Verr[int((len(V)-1)/2)]**2)
        Vg = f[:,3] * 1e-2 #[V] -> [T]
        Vg = Vg - Vg[int((len(Vg)-1)/2)]
        Vg_err = f[:,4] * 1e-2 #[V] -> [T]
        Vg_err = np.sqrt(Vg_err**2+Verr[int((len(Vg)-1)/2)]**2)

        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel(r"$\rm Vlotage \ (\mu V)$")
        self.ax.set_ylabel(r"$\rm Magnetic \ field \ (\mu T)$")
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        self.ax.errorbar(V*1e+6,Vg*1e+6,yerr=Vg_err*1e+6,xerr=Verr*1e+6,markeredgecolor = "blue", color='blue',markersize=6,fmt="o",ecolor="blue",label='Data')
        popt, pcov = curve_fit(linear,V,Vg)
        print(popt,pcov)
        x = np.arange(-200e-6,200e-6+1e-6,1e-6)
        self.ax.plot(x*1e+6,linear(x,popt)*1e+6,color='black')
        plt.show()