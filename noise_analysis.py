# 2021.11.11 noise_analysis.py
from pytes import Util,Filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
from lmfit import Model
import h5py
import pathlib
import glob
from Hdf5Treat import Hdf5Command
from scipy import integrate,signal
from Basic import Plotter

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.11.11

print('===============================================================================')
print(f"Noise Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')


class Noise:

    def __init__(self,):
        self.kb      = 1.381e-23 ##[J/K]
        self.Rshunt  = 3.9 * 1e-3
        self.Min     = 9.82 * 1e-11
        self.Rfb     = 1e+5
        self.Mfb     = 8.5 * 1e-11
        self.JtoeV   = 1/1.60217662e-19
        self.excVtoI = self.Mfb / (self.Min * self.Rfb)
        self.P = Plotter()
        print(self.excVtoI)

    def initialize(self,ch):
        self.ch        = ch
        self.filepath  = f"./../data/Noise/raw_data/ch{self.ch}"
        self.filelist  = [p.name for p in pathlib.Path(self.filepath).glob("*n.fits") if p.is_file()]

    def gen_white(self,dlen,std,mean=0.):
        self.white = np.random.normal(mean,std,dlen)


    def load_data(self,fname,smooth):
        """Loading data from fits file.

        Args:
            fname (str) : filename, Not need file path 
            smooth(int) : smoothing factor in time scale

        Define Self func:
            self.fname, self.smooth = fname,smooth
            self.t : Data time.   
            self.n : Voltage data of the Noise. 

        Notes:
            filepath -> Auto setting
            If you set smooth = 5, Data will be average 5 points.

        """
        self.fname, self.smooth = fname, smooth
        nt,n       = Util.fopen(f"{self.filepath}/{self.fname}uAn.fits")
        pt,p       = Util.fopen(f"{self.filepath}/{self.fname}uAp.fits")
        self.t     = np.append(nt,nt[-1]+nt)
        self.t     = self.t[0:int(self.t.shape[0]/self.smooth)]
        self.n     = np.hstack((n,p))
        self.n     = np.reshape(self.n,(-1,int(self.n.shape[1]/self.smooth)))

        self.timeR = self.t[-1]-self.t[0]
        self.Srate = self.t.shape[-1]/(self.timeR)
        self.Sfrq  = 1/((self.t[1]-self.t[0])*self.t.shape[-1])
        print(f'n shape = {self.n.shape}')

    def test(self,std):
        self.initialize(ch=2)
        self.load_data(fname='300',smooth=1)
        s = np.std(self.n,axis=1)
        s = np.reshape(s,len(s))
        s = s*self.excVtoI
        s = sorted(s)
        dlen = 316*3254
        self.gen_white(dlen=dlen,std=std)
        self.white = np.reshape(self.white,(316,3254))
        self.white = np.std(self.white,axis=1)
        self.white = sorted(self.white)
        plt.hist(s[:-1],bins=100,histtype='step')
        plt.hist(self.white[:-2],bins=500,histtype='step')
        plt.show()

    def spec(self,std):
        self.initialize(ch=2)
        self.load_data(fname='300',smooth=1)
        hres = self.t[1] - self.t[0]
        self.noise_spectrum(data=self.n,hres=hres)
        plt.step(self.frq[2:],self.nspec[2:],where='mid')
        npower = np.sum(self.nspec[self.frq<1e+5][2:])
        print(npower)
        print(self.Sfrq)
        dlen = 316*3254
        self.gen_white(dlen=dlen,std=std)
        self.white = np.reshape(self.white,(316,3254))
        self.noise_spectrum(data=self.white,hres=hres,VtoI=False)
        npower = np.sum(self.nspec[self.frq<1e+5][2:])
        print(npower)
        plt.step(self.frq,self.nspec,where='mid')
        plt.semilogx()
        plt.semilogy()
        plt.show()       


    def hdf5_wave(self,hdf5file):
        with h5py.File(hdf5file,"r") as f:
            vres = f['waveform']['vres'][...]
            hres = f['waveform']['hres'][...]
            w    = f['waveform']['wave'][:]
            dlen = int(len(w[0])/2)
            p    = w[:,dlen:]*vres
            n    = w[:,:dlen]*vres
            t    = np.arange(0,dlen*hres+hres/2,hres)
        return t,n

    def data_cor(self):
        self.P
        t1, n1 = self.hdf5_wave("run012_b64.hdf5")
        t2, n2 = Util.fopen('Axion_run007n.fits')
        t3, n3 = Util.fopen('TMU542_run011_rown.fits')
        std1 = np.std(n1*self.excVtoI,axis=1)
        std2 = np.std(n2*self.excVtoI,axis=1)
        std3 = np.std(n3*self.excVtoI,axis=1)
        plt.hist(std1,bins=1000,histtype='step',label='JAXA120Ea4 run012',color='Black')
        plt.hist(std2,bins=100,histtype='step',label='Old Axion run007',color='Blue')
        plt.hist(std3,bins=200,histtype='step',label='TMU542 run011',color='Red')
        plt.xlabel(r'$\rm Standard \ deviation \ [A]$',fontsize=20)
        plt.ylabel('Count',fontsize=20)
        plt.legend()
        plt.show()

    def noise_spectrum(self,data,hres,VtoI=True):
        hres = hres
        if VtoI == True:
            self.nspec = np.sqrt(Filter.average_noise(data)*(hres*data.shape[-1])) * self.excVtoI
        else:
            self.nspec = np.sqrt(Filter.average_noise(data)*(hres*data.shape[-1]))
        self.frq = np.fft.rfftfreq(data.shape[-1],hres)
        self.len_n = len(np.fft.rfft(data[0]))
        self.len_m = len(data[0])

class NAnalysis:

    def __init__(self,):
        self.kb      = 1.381e-23 ##[J/K]
        self.Rshunt  = 3.9 * 1e-3
        self.Min     = 9.82 * 1e-11
        self.Rfb     = 1e+5
        self.Mfb     = 8.5 * 1e-11
        self.JtoeV   = 1/1.60217662e-19
        self.excVtoI = self.Mfb / (self.Min * self.Rfb)
        print(self.excVtoI)

    def initialize(self,ch,savehdf5,Tbath,fmod):
        self.ch        = ch
        self.savehdf5  = savehdf5
        self.Tbath     = float(Tbath*1e-3)
        self.fmod      = fmod
        self.filepath  = f"./../data/Noise/raw_data/ch{self.ch}"
        self.filelist  = [p.name for p in pathlib.Path(self.filepath).glob("*n.fits") if p.is_file()]
        self.bias_list = [s[:-8] for s in self.filelist]

    def lowpass(self, x, fp, fs, gpass, gstop):
        fn = self.Srate / 2                           #ナイキスト周波数
        wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
        ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
        return y                                      #フィルタ後の信号を返す

    def lowpass_filter(self,lpfc):

        m = self.data[f"{self.fname}uA"]["len_m"]
        n = self.data[f"{self.fname}uA"]["len_n"]
        h = np.blackman(m)*np.sinc(np.float(lpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        self.w = np.abs(np.fft.rfft(h))

    def cutoff(self,f,lpfc):
        return 1-np.exp(-f/lpfc)


## TES Eigen Noise Function ##

    def func_init(self):
        if self.fmod == "twoblock":
            self.C = self.Cabs + self.Ctes
        self.F_func()
        self.lp_func()
        self.tau_func()
        self.tau_I_func()
        self.tau_el_func()
        self.s_I_func()
        self.s_abs_func()
        self.s_Ites_func()
        self.s_IL_func()
        self.s_Iptes_func()
        self.s_ITFN_func()
        self.tau_rise_func()
        self.tau_fall_func()
        #print(self.tau_rise)
        #print(self.tau_fall)
        #print(self.tau_el)
        #print(self.tau_I)
        #self.inst_res()

        print(f"Factor F = {self.F}")

    def F_func(self):
        self.F       = self.n * (1-(self.Tbath/self.Ttes)**(2*self.n-1)) / ((2*self.n+1)*(1-(self.Tbath/self.Ttes)**(self.n)))

    def lp_func(self):
        self.lp      = self.Ites**2*self.Rtes*self.alpha/(self.Gtes*self.Ttes)    

    def tau_func(self):
        self.tau     = self.C/self.Gtes

    def tau_I_func(self):
        self.tau_I   = self.tau/(1-self.lp)

    def tau_el_func(self):
        self.tau_el  = self.L/(self.Rth+self.Rtes*(1+self.beta))

    def s_I_func(self):
        if self.fmod == "oneblock": 
            self.s_I = -1/(self.Ites*self.Rtes*(self.L/(self.tau_el*self.Rtes*self.lp) + (1 - self.Rth/self.Rtes) + 2j*np.pi*self.frq_m*self.L*self.tau*(1/self.tau_I + 1/self.tau_el)/(self.Rtes*self.lp) - (2*np.pi*self.frq_m)**2 * self.tau*self.L/(self.lp*self.Rtes)))
        if self.fmod == "twoblock":
            self.s_I = 1/((self.Ites*(self.Rth+2j*np.pi*self.frq_m*self.L+self.Rtes*(1+self.beta))/(self.lp*self.Gtes))*(self.Gtes*(self.lp-1)-self.Gabs-2j*np.pi*self.frq_m*self.Ctes+self.Gabs**2/(self.Gabs+2j*np.pi*self.frq_m*self.Cabs))-self.Rtes*self.Ites*(2+self.beta))

    def s_abs_func(self):
        if self.fmod == "oneblock":
            self.s_abs = 0
        if self.fmod == "twoblock":
            self.s_abs = self.s_I/(1+2j*np.pi*self.frq_m*self.Cabs/self.Gabs)

    def s_Ites_func(self):
        self.s_Ites  = np.sqrt(4*self.kb*self.Ttes*self.Rtes*self.Ites**2*(1+2*self.beta)*(1+(2*np.pi*self.frq_m*self.tau)**2)*np.abs(self.s_I)**2/self.lp**2)

    def s_IL_func(self):
        self.s_IL    = np.sqrt(4*self.kb*self.Tbath*self.Ites**2*self.Rth*(self.lp-1)**2*(1+(self.tau_I*2*np.pi*self.frq_m)**2)*np.abs(self.s_I)**2/self.lp**2)

    def s_Iptes_func(self):
        self.s_Iptes = np.sqrt(4*self.kb*self.Ttes**2*self.Gtes*self.F*np.abs(self.s_I)**2)

    def s_ITFN_func(self):
        if self.fmod == "oneblock":
            self.s_abs = 0
        if self.fmod == "twoblock":
            self.s_ITFN = np.sqrt(4*self.kb*self.Ttes**2*self.Gtes)*np.abs(self.s_I-self.s_abs)

    def Sp_tot(self,frq):
        return self.S_Ites(frq=frq) + self.S_IL(frq=frq) + self.S_Iptes(frq=frq) + self.S_Ites_exs(frq=frq) + self.S_Iptes_exs(frq=frq)

    def S_Ites(self,frq):
        return (4*self.kb*self.Ttes*self.Rtes*self.Ites**2*(1+2*self.beta)*(1+(2*np.pi*frq*self.tau)**2)/self.lp**2)

    def S_IL(self,frq):
        return (4*self.kb*self.Tbath*self.Ites**2*self.Rth*(self.lp-1)**2*(1+(self.tau_I*2*np.pi*frq)**2)/self.lp**2)

    def S_Iptes(self,frq):
        return (4*self.kb*self.Ttes**2*self.Gtes*self.F)

    def S_Ites_exs(self,frq):
        return self.M**2*(4*self.kb*self.Ttes*self.Rtes*self.Ites**2*(1+2*self.beta)*(1+(2*np.pi*frq*self.tau)**2)/self.lp**2)  

    def S_Iptes_exs(self,frq):
        return self.A**2*(4*self.kb*self.Ttes**2*self.Gtes*self.F)      

    def inst_res(self):
        self.inst_res   = 2 * np.sqrt(2*np.log(2)) * np.sqrt(4*self.kb*self.Ttes**2*self.C*np.sqrt(self.n*(1+2*self.beta)*self.F/(1-(self.Tbath/self.Ttes)**self.n))/self.alpha)

    def ExN_func(self,frq,M,A):
        return np.sqrt((M * self.s_Ites * 1e+12)**2 + (A * self.s_Iptes * 1e+12)**2)

    def ExN_Phonon(self,frq,A):
        return (A * self.s_Iptes[0:len(frq)])**2

    def ExN_Johnson(self,frq,M):
        return (M * self.s_Ites[0:len(frq)]*np.sqrt(1+2*self.beta))**2

    def tau_rise_func(self):
        self.tau_rise = 1/(1/(2*self.tau_el) +1/(2*self.tau_I) + np.sqrt((1/self.tau_el - 1/self.tau_I)**2 - 4*self.Rtes*self.lp*(2+self.beta)/(self.L*self.tau))/2)

    def tau_fall_func(self):
        self.tau_fall = 1/(1/(2*self.tau_el) +1/(2*self.tau_I) - np.sqrt((1/self.tau_el - 1/self.tau_I)**2 - 4*self.Rtes*self.lp*(2+self.beta)/(self.L*self.tau))/2)  

## Data Loading ##

    def load_data(self,fname,smooth):
        """Loading data from fits file.

        Args:
            fname (str) : filename, Not need file path 
            smooth(int) : smoothing factor in time scale

        Define Self func:
            self.fname, self.smooth = fname,smooth
            self.t : Data time.   
            self.n : Voltage data of the Noise. 

        Notes:
            filepath -> Auto setting
            If you set smooth = 5, Data will be average 5 points.

        """
        self.fname, self.smooth = fname, smooth
        nt,n       = Util.fopen(f"{self.filepath}/{self.fname}uAn.fits")
        pt,p       = Util.fopen(f"{self.filepath}/{self.fname}uAp.fits")
        self.t     = np.append(nt,nt[-1]+nt)
        self.t     = self.t[0:int(self.t.shape[0]/self.smooth)]
        self.n     = np.hstack((n,p))
        self.n     = np.reshape(self.n,(-1,int(self.n.shape[1]/self.smooth)))

        self.timeR = self.t[-1]-self.t[0]
        self.Srate = self.t.shape[-1]/(self.timeR)
        self.Sfrq  = 1/((self.t[1]-self.t[0])*self.t.shape[-1])

    def nspec_inf(self):
        print("-----------------------------------------------")
        print(f"Time Range         : {self.timeR} [s]")
        print(f"Data Number        : {self.n.shape[0]}")
        print(f"Sampling Rate      : {self.Srate} [Samples/s]")
        print(f"Sampling Frequency : {self.Sfrq} [Hz]")
        print(f"Frequency Range    : - {self.t.shape[-1]/(self.t[-1]-self.t[0])/2} [Hz]")
        print(f"Smoothing Factor   : {self.smooth}")

    def load_hdf5(self,readout):
        with h5py.File(self.savehdf5,"r") as f:
            self.data = {}
            self.hdf5_keys = list(f[f"ch{self.ch}"]["Noise"]["data"].keys())
            print(self.hdf5_keys)  
            for i in self.hdf5_keys:
                self.data[i] = {}
                self.data[i]["nspec"] = f[f"ch{self.ch}"]["Noise"]["data"][i]["nspec"][:]
                self.data[i]["nspec_frq"] = f[f"ch{self.ch}"]["Noise"]["data"][i]["nspec_frq"][:]
                self.data[i]["len_m"] = f[f"ch{self.ch}"]["Noise"]["data"][i]["len_m"][...]
                self.data[i]["len_n"] = f[f"ch{self.ch}"]["Noise"]["data"][i]["len_n"][...]
            self.readout_nspec = self.data[f"{readout}"]["nspec"]
            self.readout_frq = self.data[f"{readout}"]["nspec_frq"] 

    def load_result(self):
        fname = self.fname
        with h5py.File(self.savehdf5,"r") as f:
            self.n    = f[f"ch{self.ch}"]["IV"]["analysis"]["Gfit"]["n"][...]
            self.L    = f[f"ch{self.ch}"]["Z"]["analysis"]["Rth_L_fit"]["L"][...]*3
            self.Rth  = f[f"ch{self.ch}"]["Z"]["analysis"]["Rth_L_fit"]["Rth"][...]
            self.Ites = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["Ites"][...]
            self.Rtes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["Rtes"][...]
            self.Ttes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["Ttes"][...]
            self.Gtes = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["Gtes"][...]
            self.Rn   = f[f"ch{self.ch}"]["IV"]["analysis"]["Rn_Z"][...]

            print("-----------------------------------------------")
            print(f"Loading Result ...")
            print(f"Temperature Dependence : n = {self.n}")
            print(f"Inductance in the TES Circuit : L = {self.L} [H]")
            print(f"Resistance in the TES Circuit : Rth = {self.Rth} [Ohm]")
            print(f"TES Current : Ites = {self.Ites} [A]")
            print(f"TES Resistance : Rtes = {self.Rtes} [Ohm]")
            print(f"TES Temperature : Ttes = {self.Ttes} [K]")
            print(f"Membrane Heat : Gtes = {self.Gtes} [W/K]")

            if self.fmod == "oneblock":
                print(f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"].keys())
                self.alpha = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["alpha"][...]                           
                self.beta  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["beta"][...]
                self.C     = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["C"][...]

                print(f"Temperature Sensitivity : Alpha = {self.alpha}")
                print(f"Current Sensitivity : Beta = {self.beta}")
                print(f"Absorber + TES Heat Capacity : C = {self.C}")  

            if self.fmod == "twoblock":
                self.alpha = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["alpha"][...]                           
                self.beta  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["beta"][...]
                self.Cabs  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["Cabs"][...]
                self.Ctes  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["Ctes"][...]
                self.Gabs  = f[f"ch{self.ch}"]["Z"]["analysis"][f"{fname}uA"]["fitting_result"][f"{self.fmod}"]["Gabs"][...]
                print(f"Temperature Sensitivity : Alpha = {self.alpha}")
                print(f"Current Sensitivity : Beta = {self.beta}")
                print(f"Absorber Heat Capacity : Cabs = {self.Cabs}") 
                print(f"TES Heat Capacity : Ctes = {self.Ctes}") 
                print(f"Absorber to TES Heat Conductance : Gabs = {self.Gabs}") 

    def noise_spectrum(self):
        hres = self.t[1] - self.t[0]
        self.nspec = np.sqrt(Filter.average_noise(self.n)*(hres*self.n.shape[-1])) * self.excVtoI
        self.frq = np.fft.rfftfreq(self.n.shape[-1],hres)
        self.len_n = len(np.fft.rfft(self.n[0]))
        self.len_m = len(self.n[0])

    def save_nspec(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "Noise" in f[f"ch{self.ch}"].keys():
                if f"{self.fname}uA" in f[f"ch{self.ch}/Noise/data"].keys():
                    del f[f"ch{self.ch}/Noise/data/{self.fname}uA"]
            f.create_dataset(f"ch{self.ch}/Noise/data/{self.fname}uA/nspec",data=self.nspec)
            f.create_dataset(f"ch{self.ch}/Noise/data/{self.fname}uA/nspec_frq",data=self.frq)
            f.create_dataset(f"ch{self.ch}/Noise/data/{self.fname}uA/len_m",data=self.len_m)
            f.create_dataset(f"ch{self.ch}/Noise/data/{self.fname}uA/len_n",data=self.len_n)

    def savenspec_txt(self):
        outdata = np.vstack((self.frq,self.nspec))
        np.savetxt("nspec_300uA.txt",outdata.T)
        print(outdata.T)

    def all_nspec(self):
        self.initialize(hdf5name="noise_spec.hdf5")
        self.plot_init(style="nspec")
        flist = glob.glob("*p.fits")
        for e,i in enumerate(flist):
            idx = flist[e].find("p.fits")
            fname = flist[e][:idx]
            self.load_data(fname=fname)
            self.noise_spectrum()
            self.save_nspec()

## Plot function ##

    def plot_init(self,style):
        #plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
        plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize']       = 20 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize']       = 20 # 軸だけ変更されます
        plt.rcParams['xtick.direction']       = 'in' # x axis in
        plt.rcParams['ytick.direction']       = 'in' # y axis in 
        plt.rcParams['axes.linewidth']        = 1.0 # axis line width
        plt.rcParams['axes.grid']             = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['scatter.edgecolors']    = None
        self.fs = 25
        self.ps = 30

        if style == "nspec":
            self.fig = plt.figure(figsize=(12,8))
            self.ax1 = self.fig.add_subplot(111)
            self.ax1.grid(linestyle="dashed")
            self.ax1.set_xlabel(r"$ \rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Current \ Power \ Density \ (pA/\sqrt{Hz})$",fontsize=self.fs)

        if style == "nspec_res":

            self.fig = plt.figure(figsize=(12,8))
            gs = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1])
            self.ax1 = self.fig.add_subplot(gs1[:,:])
            self.ax2 = self.fig.add_subplot(gs2[:,:],sharex=self.ax1)
            self.ax1.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax2.set_xlabel(r"$ \rm Frequency \ (Hz)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Current \ Power \ Density \ (pA/\sqrt{Hz})$",fontsize=self.fs)
            self.ax2.set_ylabel(r"$ \rm Residual$",fontsize=self.fs)    
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=.0)

        if style == "Mfactor":
            self.fig = plt.figure(figsize=(24,8))
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.ax1.grid(linestyle="dashed")
            self.ax2.grid(linestyle="dashed")
            self.ax3.grid(linestyle="dashed")
            self.ax1.set_ylabel("M factor",fontsize=self.fs)
            self.ax1.set_xlabel("Rtes/Rn",fontsize=self.fs)
            self.ax2.set_xlabel("alpha",fontsize=self.fs)
            self.ax3.set_xlabel("beta",fontsize=self.fs)           

        if style == 'hist':
            self.fig = plt.figure(figsize=(12,8))
            self.ax1 = self.fig.add_subplot(111)
            self.ax1.grid(linestyle="dashed")
            self.ax1.set_xlabel(r"$ \rm Current \ (uA)$",fontsize=self.fs)
            self.ax1.set_ylabel(r"$ \rm Counts $",fontsize=self.fs)


    def plot_nspec(self,subject):
        
        if subject == "raw_data":
            self.plot_init(style="nspec")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_ylim(1,1e+3)
            self.ax1.set_xlim(np.min(self.frq[self.frq>0]),np.max(self.frq[self.frq>0]))
            self.ax1.step(self.frq,self.nspec*1e+12,where="mid")
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png"

        if subject == "frq":
            self.plot_init(style="nspec")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            #self.ax1.set_ylim(1,1e+3)
            self.ax1.set_xlim(np.min(self.frq[self.frq>0]),np.max(self.frq[self.frq>0]))
            self.ax1.step(self.frq,self.nspec*np.sqrt(self.frq)*1e+12,where="mid")
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec_frq.png"

        if subject == "noise_hist":
            self.plot_init(style='hist')
            avg_noise = Filter.average_noise(self.n)*self.excVtoI*1e+6
            self.ax1.hist(avg_noise[avg_noise<1],bins=1000,histtype='step')
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec_hist.png"

        if subject == "select_noise":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png" 
            self.plot_init(style="nspec")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_ylim(1,1e+3)
            l = len(self.sel)
            for e,i in enumerate(self.sel):
                self.sel_bias = i
                self.ax1.set_xlim(np.min(self.data[i]["nspec_frq"][self.data[i]["nspec_frq"] > 0]),np.max(self.data[i]["nspec_frq"]))
                self.ax1.step(self.data[self.sel_bias]["nspec_frq"],self.data[self.sel_bias]["nspec"]*1e+12,c=cm.jet(e/l),label=f"{i}",where="mid")
            sfn = f"./graph/Noise/ch{self.ch}_sel_nspec.png"

        if subject == "readout":
            self.plot_init(style="nspec")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_ylim(1,1e+3)
            self.ax1.step(self.readout_frq,self.readout_nspec*1e+12,label="Read Out Noise",c="Blue",where="mid")
            read_const = np.median(self.readout_nspec[self.readout_frq>1e+4])
            cut_value  = read_const/np.sqrt(2)
            cut_frq = self.readout_frq[np.argmin(np.abs(cut_value-self.readout_nspec))]
            print(f"cut_value = {cut_value}")
            print(f"cut frq = {cut_frq}")
            sfn = f"./graph/Noise/ch{self.ch}_readout.png"

        if subject == "eigen_noise":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png" 
            self.plot_init(style="nspec")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_xlim(np.min(self.frq),np.max(self.frq))
            self.ax1.set_ylim(1,1e+3)
            self.ax1.scatter(self.readout_frq,self.readout_nspec*1e+12,label="Read Out Noise",s=self.ps,c="Blue")
            self.ax1.scatter(self.frq,self.nspec*1e+12,label=f"Ibias : {self.fname}uA",s=self.ps,c="Black")
            self.ax1.plot(self.frq_m,self.s_Ites*1e+12,label="TES Johnson Noise",ls="-.",c="c")
            self.ax1.plot(self.frq_m,self.s_IL*1e+12,label="Rth Johnson Noise",ls="-.",c="g")
            self.ax1.plot(self.frq_m,self.s_Iptes*1e+12,label="Phonon Noise",ls="-.",c="y")
            self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2)
            self.ax1.plot(self.frq_m,self.totnoise*1e+12,label="Total Noise",ls="-",c="Red")

        if subject == "eigen_noise_res":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png" 
            self.plot_init(style="nspec_res")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_xlim(np.min(self.data[f"{self.fname}uA"]["nspec_frq"][self.data[f"{self.fname}uA"]["nspec_frq"] > 0]),np.max(self.data[f"{self.fname}uA"]["nspec_frq"]))
            self.ax1.set_ylim(1,1e+3)
            self.ax2.set_ylim(-25,25)
            self.ax1.step(self.readout_frq,self.readout_nspec*1e+12,label="Read Out Noise",c="Blue",where="mid")
            self.ax1.step(self.data[f"{self.fname}uA"]["nspec_frq"],self.data[f"{self.fname}uA"]["nspec"]*1e+12,label=f"Ibias : {self.fname}uA",c="Black",where="mid")
            self.ax1.plot(self.frq_m,self.s_Ites*1e+12,label="TES Johnson Noise",ls="-.",c="c")
            self.ax1.plot(self.frq_m,self.s_IL*1e+12,label="Rth Johnson Noise",ls="-.",c="g")
            self.ax1.plot(self.frq_m,self.s_Iptes*1e+12,label="Phonon Noise",ls="-.",c="y")

            if self.fmod == "oneblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2)

            if self.fmod == "twoblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + self.s_ITFN**2)
                self.ax1.plot(self.frq_m,self.s_ITFN*1e+12,label="ITFN Noise",ls="-.",c="m")

            self.ax1.plot(self.frq_m,self.totnoise*1e+12,label="Total Noise",ls="-",c="Red")
            self.ax2.scatter(self.data[f"{self.fname}uA"]["nspec_frq"],(self.data[f"{self.fname}uA"]["nspec"]-self.totnoise)*1e+12,s=self.ps,c="Black")

        if subject == "ExN_fit":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png" 
            self.plot_init(style="nspec_res")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_xlim(np.min(self.data[f"{self.fname}uA"]["nspec_frq"][self.data[f"{self.fname}uA"]["nspec_frq"] > 0]),np.max(self.data[f"{self.fname}uA"]["nspec_frq"]))
            self.ax1.set_ylim(1,1e+3)
            self.ax2.set_ylim(-25,25)
            self.ax1.step(self.readout_frq,self.readout_nspec*1e+12,label="Read Out Noise",c="Blue",where="mid")
            self.ax1.step(self.data[f"{self.fname}uA"]["nspec_frq"],self.data[f"{self.fname}uA"]["nspec"]*1e+12,label=f"Ibias : {self.fname}uA",c="Black",where="mid")
            self.ax1.plot(self.frq_m,self.s_Ites*1e+12,label="TES Johnson Noise",ls="-.",c="c")
            self.ax1.plot(self.frq_m,self.s_IL*1e+12,label="Rth Johnson Noise",ls="-.",c="g")
            self.ax1.plot(self.frq_m,self.s_Iptes*1e+12,label="Phonon Noise",ls="-.",c="y")

            if self.fmod == "oneblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + (self.M*self.s_Ites)**2 + (self.A*self.s_Iptes)**2)

            if self.fmod == "twoblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + self.s_ITFN**2)
                self.ax1.plot(self.frq_m,self.s_ITFN*1e+12,label="ITFN Noise",ls="-.",c="m")

            self.ax1.plot(self.frq_m,self.totnoise*1e+12,label="Total Noise",ls="-",c="Red")
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],self.M*self.s_Ites*1e+12,label="M Factor")
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],self.A*self.s_Iptes*1e+12,label="A Factor")
            self.ax2.scatter(self.data[f"{self.fname}uA"]["nspec_frq"],(self.data[f"{self.fname}uA"]["nspec"]-self.totnoise)*1e+12,s=self.ps,c="Black")
            self.ax1.set_ylabel(r"$ \rm Current \ Power \ Density \ (pA^2/Hz)$",fontsize=self.fs)

        if subject == "ExN_fit_resid":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec.png" 
            self.plot_init(style="nspec_res")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_xlim(np.min(self.data[f"{self.fname}uA"]["nspec_frq"][self.data[f"{self.fname}uA"]["nspec_frq"] > 0]),np.max(self.data[f"{self.fname}uA"]["nspec_frq"]))
            self.ax1.set_ylim(1e+2,1e+5)
            self.ax2.set_ylim(-1000,1000)

            if self.fmod == "oneblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + (self.M*self.s_Ites)**2 + (self.A*self.s_Iptes)**2)

            if self.fmod == "twoblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + self.s_ITFN**2)
                self.ax1.plot(self.frq_m,self.s_ITFN*1e+12,label="ITFN Noise",ls="-.",c="m")

            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],(self.M*self.s_Ites)**2*1e+24,label="Excess Johnson Noise",ls="-.",c="c")
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],(self.A*self.s_Iptes)**2*1e+24,label="Excess Phonon Noise",ls="-.",c="y")
            self.ax1.step(self.data[f"{self.fname}uA"]["nspec_frq"],self.Nresid*1e+24,label=f"Ibias : {self.fname}uA_residual",c="Black",where="mid")
            self.ax2.scatter(self.data[f"{self.fname}uA"]["nspec_frq"],(self.Nresid - (self.M*self.s_Ites)**2 - (self.A*self.s_Iptes)**2)*1e+24,s=self.ps,c="Black")        
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],((self.M*self.s_Ites)**2+(self.A*self.s_Iptes)**2)*1e+24,c="Red")
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_ExNfit_nspec.png"

        if subject == "eigen_noise_res_ExN":
            sfn = f"./graph/Noise/ch{self.ch}_{self.fname}uA_nspec_fitres.png" 
            self.plot_init(style="nspec_res")
            self.ax1.set_xscale("log")
            self.ax1.set_yscale("log")         
            self.ax1.set_xlim(np.min(self.data[f"{self.fname}uA"]["nspec_frq"][self.data[f"{self.fname}uA"]["nspec_frq"] > 0]),np.max(self.data[f"{self.fname}uA"]["nspec_frq"]))
            self.ax1.set_ylim(1,1e+3)
            self.ax2.set_ylim(-25,25)
            self.ax1.step(self.readout_frq,self.readout_nspec*1e+12,label="Read Out Noise",c="Blue",where="mid")
            self.ax1.step(self.data[f"{self.fname}uA"]["nspec_frq"],self.data[f"{self.fname}uA"]["nspec"]*1e+12,label=f"Ibias : {self.fname}uA",c="Black",where="mid")
            self.ax1.plot(self.frq_m,self.s_Ites*1e+12,label="TES Johnson Noise",ls="-.",c="c")
            self.ax1.plot(self.frq_m,self.s_IL*1e+12,label="Rth Johnson Noise",ls="-.",c="g")
            self.ax1.plot(self.frq_m,self.s_Iptes*1e+12,label="Phonon Noise",ls="-.",c="y")
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],self.M*self.s_Ites*1e+12,label="Excess Johnson Noise",ls="-.",c="m")
            self.ax1.plot(self.data[f"{self.fname}uA"]["nspec_frq"],self.A*self.s_Iptes*1e+12,label="Excess Phonon Noise",ls="-.",c="#984ea3")


            if self.fmod == "oneblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + (self.M*self.s_Ites)**2 + (self.A*self.s_Iptes)**2)

            if self.fmod == "twoblock":
                self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + self.s_ITFN**2)
                self.ax1.plot(self.frq_m,self.s_ITFN*1e+12,label="ITFN Noise",ls="-.",c="m")

            self.ax1.plot(self.frq_m,self.totnoise*1e+12,label="Total Noise",ls="-",c="Red")
            self.ax2.scatter(self.data[f"{self.fname}uA"]["nspec_frq"],(self.data[f"{self.fname}uA"]["nspec"]-self.totnoise)*1e+12,s=self.ps,c="Black") 

        if subject == "Mfactor":
            sfn = f"./graph/Noise/ch{self.ch}_Mfactor.png" 
            self.plot_init(style="Mfactor")
            self.ax1.scatter(self.Rpar_list,self.M_list,s=self.ps,c="Black")
            self.ax2.scatter(self.alpha_list,self.M_list,s=self.ps,c="Black")
            self.ax3.scatter(self.beta_list,self.M_list,s=self.ps,c="Black")

        self.ax1.legend(loc='best',fontsize=15)
        self.fig.savefig(sfn,dpi=300)
        plt.show()

## Excess Noise Fitting ##

    def ExN_fit(self,sel,Johnson_frq,Phonon_frq):

        if self.fmod == "oneblock":
            self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2)

        if self.fmod == "twoblock":
            self.totnoise = np.sqrt(self.s_Ites**2 + self.s_IL**2 + self.s_Iptes**2 + self.readout_nspec**2 + self.s_ITFN**2)

        self.Nresid = self.data[f"{self.fname}uA"]["nspec"]**2 - self.totnoise**2

        if sel == True:

            f = self.data[f"{self.fname}uA"]["nspec_frq"]
            model = Model(self.ExN_Johnson) 
            model.set_param_hint('M',min=0,max=10)
            result = model.fit(self.Nresid[f<Johnson_frq],frq=f[f<Johnson_frq],weights=f[f<Johnson_frq],M=2)
            print(result.fit_report())
            self.M = result.best_values["M"]*np.sqrt(1+2*self.beta)

            model = Model(self.ExN_Phonon)
            model.make_params(verbose=True)
            model.set_param_hint('A',min=0,max=10)
            result = model.fit((self.Nresid-(self.M*self.s_Ites*np.sqrt(1+2*self.beta))**2)[f<Phonon_frq],frq=f[f<Phonon_frq],weights=f[f<Phonon_frq],A=1)
            print(result.fit_report())
            self.A = result.best_values["A"]

        self.plot_nspec(subject="ExN_fit_resid")
        self.plot_nspec(subject="eigen_noise_res_ExN")

    def cal_resolution(self):
        Sp_tot =  lambda frq: 4/self.Sp_tot(frq=frq)
        self.RMS_tot = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_tot, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV
        Sp_phonon =  lambda frq: 4/self.S_Iptes(frq=frq)
        self.RMS_phonon = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_phonon, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV
        Sp_Johnson =  lambda frq: 4/self.S_Ites(frq=frq)
        self.RMS_Johnson = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_Johnson, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV
        Sp_Johnson_ext =  lambda frq: 4/self.S_IL(frq=frq)
        self.RMS_Johnson_ext = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_Johnson_ext, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV
        Sp_Johnson_exs =  lambda frq: 4/self.S_Ites_exs(frq=frq)
        self.RMS_Johnson_exs = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_Johnson_exs, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV
        Sp_phonon_exs =  lambda frq: 4/self.S_Iptes_exs(frq=frq)
        self.RMS_phonon_exs = 2*np.sqrt(2*np.log(2)) * integrate.quad(Sp_phonon_exs, 0, np.max(self.frq_m))[0]**(-1/2) * self.JtoeV        
        print("-----------------------------------------------")
        print(f"Delta E from Integration total : delE = {self.RMS_tot}")
        print(f"Phonon Noise = {self.RMS_phonon}")
        print(f"Johnson Noise = {self.RMS_Johnson}")
        print(f"Johnson Noise Extra = {self.RMS_Johnson_ext}")
        print(f"Excess Phonon Noise = {self.RMS_phonon_exs}")
        print(f"Excess Johnson Noise = {self.RMS_Johnson_exs}")
        #print(np.sqrt(RMS_phonon**2+RMS_Johnson**2+RMS_Johnson_ext**2)*self.JtoeV)
        self.save_resolution()

    def save_resolution(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "analysis" in f[f"ch{self.ch}/Noise"]:
                if f"{self.fname}uA" in f[f"ch{self.ch}/Noise/analysis"]:
                    del f[f"ch{self.ch}/Noise/analysis/{self.fname}uA"]
            f.create_dataset(f"ch{self.ch}/Noise/analysis/{self.fname}uA/M",data=self.M)
            f.create_dataset(f"ch{self.ch}/Noise/analysis/{self.fname}uA/A",data=self.A)
            f.create_dataset(f"ch{self.ch}/Noise/analysis/{self.fname}uA/Resolution",data=self.RMS_tot)

    def load_resolution(self):
        with h5py.File(self.savehdf5,"a") as f:
            for e,i in enumerate(f[f"ch{self.ch}/Noise/analysis"].keys()):
                self.fname = i
                self.load_result()
                if e == 0:
                    self.M_list = f[f"ch{self.ch}/Noise/analysis/{i}/M"][...]
                    self.A_list = f[f"ch{self.ch}/Noise/analysis/{i}/A"][...]
                    self.alpha_list = self.alpha
                    self.beta_list = self.beta
                    self.Rpar_list = self.Rtes/self.Rn
                else :
                    self.M_list = np.append(self.M_list,f[f"ch{self.ch}/Noise/analysis/{i}/M"][...])
                    self.A_list = np.append(self.A_list,f[f"ch{self.ch}/Noise/analysis/{i}/A"][...])
                    self.alpha_list = np.append(self.alpha_list,self.alpha)
                    self.beta_list = np.append(self.beta_list,self.beta)
                    self.Rpar_list = np.append(self.Rpar_list,self.Rtes/self.Rn)
                print(self.M_list)
                print(self.alpha_list)
                print(self.Rpar_list)


    def read_key(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.bias = f[f"ch{self.ch}/Noise/data"].keys()


    def Mfactor(self,ch,savehdf5,Tbath,fmod):
        self.initialize(ch=ch,savehdf5=savehdf5,Tbath=Tbath,fmod=fmod)
        self.load_resolution()
        self.plot_nspec(subject="Mfactor")

## Out Put Function ##            

    def ExN_fitting(self,ch,savehdf5,fname,readout,fmod,Johnson_frq,Phonon_frq,Tbath):
        self.fname = fname
        self.initialize(ch=ch,savehdf5="210908_TMU542.hdf5",Tbath=Tbath,fmod=fmod)
        self.load_hdf5(readout=readout)
        self.frq_m = self.readout_frq
        self.load_result()
        self.func_init()
        #print(self.inst_res*self.JtoeV)
        #self.plot_nspec(subject="eigen_noise_res")
        self.ExN_fit(sel=True,Johnson_frq=Johnson_frq,Phonon_frq=Phonon_frq)
        self.cal_resolution()
        self.plot_nspec(subject="eigen_noise_res")

    def readout_cutoff(self,ch=2,savehdf5="210908_TMU542.hdf5",fname="300uA",Tbath=90.0,fmod="oneblock"):
        self.fname = fname
        self.initialize(ch=ch,savehdf5="210908_TMU542.hdf5",Tbath=Tbath,fmod=fmod)
        self.load_hdf5(readout="210mK_0uA")
        self.load_result()
        #self.func_init()
        self.plot_nspec(subject="readout")



    def all_ExN_fitting(self,ch,savehdf5,readout,fmod,Johnson_frq,Phonon_frq,Tbath):
        self.initialize(ch=ch,savehdf5="210908_TMU542.hdf5",Tbath=Tbath,fmod=fmod)
        for fname in self.bias_list:
            print('filename')
            print(fname)
            if fname == '210mK_0uA':
                pass
            else:
                self.ExN_fitting(ch=ch,savehdf5=savehdf5,fname=fname,readout=readout,fmod=fmod,Johnson_frq=Johnson_frq,Phonon_frq=Phonon_frq,Tbath=Tbath)


    def plot_nspec_all(self):
        for fname in self.bias_list:
            self.initialize(ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90)
            self.load_data(fname=fname)
            self.noise_spectrum()
            self.plot_nspec(subject="select_noise")        
    
    def save_plot_nspec(self,fname):
        self.initialize(ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90,fmod="oneblock")
        self.load_data(fname=fname,smooth=1)
        self.noise_spectrum()
        self.savenspec_txt()
        #self.plot_nspec(subject="raw_data")
        self.plot_nspec(subject='frq')


    def all_sc(self,ch,savehdf5):
        ch2_bias = np.arange(260,430,20)
        for i in ch2_bias:
            self.out_sc(ch=ch,savehdf5=savehdf5,fname=i)

    def save_all_nspec(self,ch=2,savehdf5="210908_TMU542.hdf5",Tbath=90,smooth=5,fmod="oneblock"):
        self.initialize(ch=ch,savehdf5=savehdf5,Tbath=Tbath,fmod=fmod)
        print(self.bias_list)
        for e,fname in enumerate(self.bias_list):
            self.load_data(fname=fname,smooth=smooth)
            if e == 0:
                self.nspec_inf()
            self.noise_spectrum()
            self.save_nspec()
    
    def plot_sel(self,sel,ch,savehdf5,Tbath,fmod,readout):
        self.initialize(ch=ch,savehdf5=savehdf5,Tbath=Tbath,fmod=fmod)
        self.sel = sel
        self.fname = "Dummy"
        self.load_hdf5(readout=readout)
        self.plot_nspec(subject="select_noise") 

    def resolution_out(self,ch,savehdf5,Tbath,fmod,readout):
        self.initialize(ch=ch,savehdf5=savehdf5,Tbath=Tbath,fmod=fmod)
        with h5py.File(self.savehdf5,"r") as f:
            key = f[f"ch{self.ch}/Noise/analysis"].keys()
            for i in key:
                print(i)
                print(f[f"ch{self.ch}/Noise/analysis/{i}/Resolution"][...])



