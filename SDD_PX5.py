import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit


plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.size"]             = 18 
plt.rcParams['xtick.labelsize']       = 15 
plt.rcParams['ytick.labelsize']       = 15 
plt.rcParams['xtick.direction']       = 'in'
plt.rcParams['ytick.direction']       = 'in'
plt.rcParams['axes.linewidth']        = 1.0

class SDD:
    def __init__(self) -> None:
        pass

    def linear(self,x,a,b):
        return a*x + b

    def rebin(self, x, y, rebin=2, renorm=False):
        if not len(x) % rebin == 0: print("ERROR : len(x)/rebin was not an integer."); return -1, -1 
        ndiv = int(len(x)/rebin)
        xdiv = np.split(x, ndiv)
        ydiv = np.split(y, ndiv)
        xd = np.mean(xdiv, axis=1) # mean of bins 
        yd = np.sum(ydiv, axis=1) # sum of entries 
        if renorm: yd = yd/rebin
        return xd, yd

    def read_mca(self,file):
        line_list = []
        with open(file,encoding='Shift_JIS') as f:
            for ind,line in enumerate(f):
                line_list.append(line.replace('\n',''))
                if 'CALIBRATION' in line:
                    cal_start = ind + 2
                elif 'ROI' in line:
                    cal_end = ind
                elif 'DATA' in line:
                    data_start = ind + 1
                elif '<<END>>' in line:
                    data_end = ind

            cal_list = []
            for line in line_list[cal_start:cal_end]:
                l = line.split(' ')
                cal_list.append(l)
            cal_list.append([0,0])
            cal_list = np.array(cal_list)
            self.cal_channel = np.array(cal_list[:,0],dtype=float)
            self.cal_energy = np.array(cal_list[:,1],dtype=float)
            self.data = np.array(line_list[data_start:data_end],dtype=int)
            self.channel = np.arange(1,len(self.data)+1,1)
            print(self.data)
            print(len(self.data))

    def read_mac_calibration(self,file):
        line_list = []
        with open(file,encoding='Shift_JIS') as f:
            for ind,line in enumerate(f):
                line_list.append(line.replace('\n',''))
                if 'CALIBRATION' in line:
                    cal_start = ind + 2
                elif 'ROI' in line:
                    cal_end = ind

            cal_list = []
            for line in line_list[cal_start:cal_end]:
                l = line.split(' ')
                cal_list.append(l)
            cal_list.append([0,0])
            cal_list = np.array(cal_list)
            self.cal_channel = np.array(cal_list[:,0],dtype=float)
            self.cal_energy = np.array(cal_list[:,1],dtype=float)


    def calibration(self):
        popt, pcov = curve_fit(self.linear,self.cal_channel,self.cal_energy)
        print(pcov,popt)
        print(self.cal_channel)
        self.cal_func = popt
        self.energy = self.linear(self.channel,self.cal_func[0],self.cal_func[1])
        f = np.linspace(np.min(self.cal_channel),np.max(self.cal_channel),len(self.channel+1))
        resid = self.cal_energy - self.linear(self.cal_channel,*self.cal_func)
        self.fig = plt.figure(figsize=(8,6))
        self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
        self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
        self.ax  = self.fig.add_subplot(self.gs1[:,:])
        self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
        self.ax.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")
        self.ax.set_ylabel('Energy [keV]')
        self.ax2.set_ylabel('Residual [eV]')
        self.ax2.set_xlabel('Channel')
        self.ax.scatter(self.cal_channel,self.cal_energy,color='black')
        self.ax.plot(f,self.linear(f,*popt),'--',color='red')
        self.ax2.scatter(self.cal_channel,resid*1e+3,color='black')
        self.ax2.hlines(0,np.min(self.cal_channel),np.max(self.cal_channel),linestyle='dashed',color='red')
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        plt.show()
        self.fig.savefig('calibration.png',dpi=300)

    def energy_plot(self):
        self.fig = plt.figure(figsize=(7,5))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ax.set_xlabel('Energy [keV]')
        self.ax.set_ylabel('Count')
        self.ax.step(self.energy,self.data)
        # self.ax.set_yscale('log')
        plt.show()
        self.fig.savefig('energy_plot.png',dpi=300)

    def out(self,file):
        self.read_mca(file)
        self.calibration()
        self.energy_plot()
        #self.energy,self.data = self.rebin(self.energy,self.data,rebin=16)
        #self.energy_plot()

    def out_cal(self,file):
        self.read_mca(file)
        self.read_mac_calibration('/Users/keitatanaka/Dropbox/share/work/microcalorimeters/experiment/xray_source_spectrum/230907_SDD/241Am/Am_01.mca')
        self.calibration()
        self.energy_plot()
        self.energy,self.data = self.rebin(self.energy,self.data,rebin=4)
        self.energy_plot()


