import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
import numpy as np

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.11.11

print('===============================================================================')
print(f"Basic Script ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class Plotter:
    """Plotter is Class to plot in matplotlib.
    Textsize, Font, and plot style is optimized in initialization.  
    User can plot beautiful figure easily.
    """
    def __init__(self):
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman'
        plt.rcParams['mathtext.fontset']      = 'stix'
        plt.rcParams["font.size"]             = 20 
        plt.rcParams['xtick.labelsize']       = 15 
        plt.rcParams['ytick.labelsize']       = 15 
        plt.rcParams['xtick.direction']       = 'in'
        plt.rcParams['ytick.direction']       = 'in'
        plt.rcParams['axes.linewidth']        = 1.0
        plt.rcParams['axes.grid']             = True
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['scatter.edgecolors']    = 'black'

    def plot_window(self,style:str):
        if style == "single":
            self.fig = plt.figure(figsize=(9,7))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")

        if style == "twinx":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax1 = plt.subplot(111)
            self.ax1.grid(linestyle="dashed")
            self.ax2 = self.ax1.twinx()

        if style == 'residual':
            self.fig = plt.figure(figsize=(8,6))
            self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
            self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
            self.ax  = self.fig.add_subplot(self.gs1[:,:])
            self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
            self.ax.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")

    def plotting(self,x,y,*args,**kwargs):

        if 'style' in kwargs:
            style = kwargs['style']
        else:
            style = 'single'

        if 'new_window' not in kwargs:
            self.plot_window(style)

        if 'color' in kwargs:
            color = kwargs['color']
        else :
            color = "Blue"

        if 'lw' in kwargs:
            lw = kwargs['lw']
        else :
            lw = 1

        if 'label' in kwargs:
            label = kwargs['label']
        else :
            label = None

        if 'x2' in kwargs:
            x2 = kwargs['x2']
            y2 = kwargs['y2']
            self.ax.plot(x,y,"Blue",label=label)
            self.ax.scatter(x2,y2,color="Red",label=label)
        elif 'yerr' in kwargs:
            yerr = kwargs['yerr']
            self.ax.errorbar(x,y,yerr,color='Blue',markersize=6,fmt="o",ecolor="black")
        else:
            if 'scatter' in kwargs:
                self.ax.scatter(x,y,color=color,label=label)
            else:
                self.ax.plot(x,y,color=color,lw=lw,label=label)

        if 'xname' in kwargs:
            xname = kwargs['xname']
            self.ax.set_xlabel(xname,fontsize=20)
        if 'yname' in kwargs:
            yname = kwargs['yname']
            self.ax.set_ylabel(yname,fontsize=20)
        if 'title' in kwargs:
            title = kwargs['title']
            self.ax.set_title(title,fontsize=20)

        if 'xlog' in kwargs:
            self.ax.set_xscale('log')
        if 'ylog' in kwargs:
            self.ax.set_yscale('log')

        if 'xlim' in kwargs:
            xlim = kwargs['xlim']
            self.ax.set_xlim(xlim[0],xlim[1])
        if 'ylim' in kwargs:
            ylim = kwargs['ylim']
            self.ax.set_ylim(ylim[0],ylim[1])

        if style == 'residual':
            if 'y_residual' in kwargs:
                y_residual = kwargs['y_residual']
                self.ax2.scatter(x,y_residual,color='Black')
                self.ax2.grid('dashed')
                self.ax2.set_ylabel("Residual",fontsize=20)
                self.fig.subplots_adjust(hspace=.0)
                self.fig.align_labels()

        self.ax.legend(loc='best',fontsize=20)
        #plt.show()

    def gridplot(self,x,y,y_residual,popt):
        self.plot_window(style='residual')
        self.ax.plot(x,y,".",color="black",label='Data')
        self.ax.plot(x,F.linear(x,*popt),color='red',label='Fit')
        self.ax.legend(fontsize=20)
        self.ax2.plot(x,y_residual,".",color="black")
        #self.ax2.plot(x,np.zeros(len(x)),color="red")
        self.ax2.set_xlabel(r"$\rm Time \ (min)$",fontsize=20)
        self.ax2.set_ylabel("Residual",fontsize=20)
        self.ax.set_ylabel(r"$\rm Etching thickness\ (nm)$",fontsize=20)
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        plt.show()
        self.fig.savefig("test.png",dpi=300)

    def gridplot_linear_fit(self,x,y,y_residual,popt):
        F = Function()
        self.plot_window(style='residual')
        self.ax.plot(x,y,".",color="black",label='Data')
        self.ax.plot(x,F.linear(x,*popt),color='red',label='Fit')
        self.ax.legend(fontsize=20)
        self.ax2.plot(x,y_residual,".",color="black")
        #self.ax2.plot(x,np.zeros(len(x)),color="red")
        self.ax2.set_xlabel(r"$\rm Time \ (min)$",fontsize=20)
        self.ax2.set_ylabel("Residual",fontsize=20)
        self.ax.set_ylabel(r"$\rm Etching thickness\ (nm)$",fontsize=20)
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        plt.show()
        self.fig.savefig("test.png",dpi=300)

class MemoryMonitor:

    def __init__(self):
        pass

class Function:

    def __init__(self):
        pass

    def linear(self,x,a):
        return a*x

    def linear_offset(self,x,a,b):
        return a*x + b

class Fit:

    def __init__(self):
        pass

    def linear_fitting(self,x,y):
        self.popt, self.pcov = curve_fit(self.linear,x,y)





