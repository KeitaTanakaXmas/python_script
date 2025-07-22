# 2021.12.28 pulse_analysis.py
from pytes import Util,Filter,Analysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
import h5py
from scipy.signal import argrelextrema
from scipy import interpolate
import astropy.io.fits as pf
from Basic import Plotter
from function import *
import time
import xspec
import subprocess

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.12.28

print('===============================================================================')
print(f"Pulse Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class PAnalysis:

    def __init__(self):
        self.kb      = 1.381e-23 ##[J/K]
        self.Rshunt  = 3.9 * 1e-3
        self.Min     = 9.82 * 1e-11
        self.Rfb     = 1e+5
        self.Mfb     = 8.5 * 1e-11
        self.JtoeV   = 1/1.60217662e-19
        self.excVtoI = self.Mfb / (self.Min * self.Rfb)
        self.savehdf5 = "test.hdf5"
        self.P = Plotter()

## Function List ##

    def quad(self,x,a,b,c):
        return a*x**2 + b*x+ c

    def rise_func(self,t,tau_r,A):
        return A*np.exp(t/tau_r)

    def fall_func(self,t,tau_f,A):
        return A*np.exp(-t/tau_f)

    def pfunc(self,t,tau_r,tau_f,A):
        t0=0.25e-3
        return 0*(t<=t0)+A*(np.exp(-(t-t0)/tau_r)-np.exp(-(t-t0)/tau_f))*(t0<t)

    def gaus(self, x, norm, sigma, mu):
        return norm * np.exp(-(x-mu)**2 / (2*sigma**2))

## Function List ##

    def moving_average(self,x,num):
        b = np.ones(num)/num
        conv = np.convolve(x,b,mode="same")
        return conv

    def load_fits(self,file):
        self.tp,self.p = Util.fopen(f"{file}p.fits")
        self.tn,self.n = Util.fopen(f"{file}n.fits")
        self.t         = self.tp

    def reshape_hdf5(self,file):
        with h5py.File(file,"a") as f:
            self.vres = f['waveform']['vres'][...]
            self.hres = f['waveform']['hres'][...]
            self.p = f['waveform']['pulse'][:]
            self.n = f['waveform']['noise'][:]
            p = self.p.reshape(len(self.p),len(self.p[0]))
            n = self.p.reshape(len(self.n),len(self.n[0]))
            del f['waveform/pulse'], f['waveform/noise']
            f.create_dataset('waveform/pulse', data=p)
            f.create_dataset('waveform/noise', data=n)

    def load_hdf5(self,file):
        with h5py.File(file,"a") as f:
            self.vres = f['waveform']['vres'][...]
            self.hres = f['waveform']['hres'][...]
            self.p = f['waveform']['pulse'] * self.vres
            self.n = f['waveform']['noise'] * self.vres
            self.t = np.arange(self.p.shape[-1]) * self.hres
            self.tp = self.t
            self.tn = self.t

    def save_pulse(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "pulse" in f.keys():
                del f["pulse"],f["noise"],f["time"]
            f.create_dataset("pulse",data=self.p)
            f.create_dataset("noise",data=self.n)
            f.create_dataset("time",data=self.tp)

    def load_pulse(self):
        with h5py.File(self.savehdf5,"a") as f:
            self.p = f['pulse'][:]
            self.n = f['noise'][:]
            self.t = f['time'][:]

    def save_tmpl(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "tmpl" in f.keys():
                del f["tmpl"],f["sn"],f["pha"],f['ps']
            f.create_dataset("tmpl",data=self.tmpl)
            f.create_dataset("sn",data=self.sn)
            f.create_dataset("ps",data=self.ps)
            f.create_dataset("pha",data=self.pha)

    def load_tmpl(self):
        with h5py.File(self.savehdf5,"r") as f:
            self.tmpl = f['tmpl'][:]
            self.sn = f['sn'][:]
            self.ps = f['ps'][...]
            self.pha = f['pha'][:]

    def save_se(self):
        with h5py.File(self.savehdf5,"a") as f:
            if "pha_e" in f.keys():
                del f["pha_e"]
            f.create_dataset("pha_e",data=self.pha_e)                

    def cal_NEP(self):
        print(np.sqrt((self.sn**2).sum()*2)) 
        print(Analysis.baseline(self.sn))

    def offset(self):
        self.ofs = np.median(self.n,axis=1)
        self.ofs_std = np.std(self.ofs)
        #self.ofs_mv = self.moving_average(self.ofs,num=50)

    def offset_sigma_filter(self,sigma):
        mask = (np.median(self.ofs)-self.ofs_std*sigma<self.ofs) & (self.ofs<np.median(self.ofs)+self.ofs_std*sigma)
        print(f"{np.median(self.ofs)-self.ofs_std*sigma} - {np.median(self.ofs)+self.ofs_std*sigma}")
        self.p = self.p[mask]
        self.n = self.n[mask] 

    def offset_correction(self):
        self.p = self.p - self.ofs.reshape((len(self.p),1))

    def pulse_height(self):
        self.ph = self.ofs - np.min(self.p,axis=1)

    def load_func(self):
        self.offset()
        self.pulse_height()

    def ofs_filter(self,xmin,xmax):
        self.p = self.p[xmin:xmax]        
        self.n = self.n[xmin:xmax]

    def ph_filter(self,xmin,xmax):
        self.ph_mask = (xmin < self.ph) & (self.ph < xmax)

    def filtering(self):
        self.p = self.p[self.ph_mask]
        self.n = self.n[self.ph_mask]
        self.ph = self.ph[self.ph_mask]
        self.ofs = self.ofs[self.ph_mask]

    def gen_pha(self):
        p = self.p[self.ph_mask] - self.ofs[self.ph_mask].reshape((len(self.p[self.ph_mask]),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.n[self.ph_mask],max_shift=100)
        self.pha, self.ps = Filter.optimal_filter(self.p, self.tmpl, max_shift=100)
        self.result_plot(subject="tmpl")

    def gen_baseline(self):
        self.baseline, self.base_ps = Filter.optimal_filter(self.n, self.tmpl, max_shift=100)

    def gen_tmpl(self,max_shift):
        p = self.p[self.ph_mask] - self.ofs[self.ph_mask].reshape((len(self.p[self.ph_mask]),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.n[self.ph_mask],max_shift=max_shift)
## Plot ##
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
        plt.rcParams['figure.subplot.bottom'] = 0.15
        self.fs = 15
        self.ps = 45

    def plot_window(self,style):
        self.plot_init()

        if style == "one":
            self.fig = plt.figure(figsize=(8,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "one_large":
            self.fig = plt.figure(figsize=(10.6,6))
            self.ax  = plt.subplot(111)
            self.ax.grid(linestyle="dashed")
            self.ax.set_xlabel(f"{self.xname}",fontsize=self.fs)
            self.ax.set_ylabel(f"{self.yname}",fontsize=self.fs)

        if style == "cor":
            self.fig = plt.figure(figsize = (16,8))
            self.gs = GridSpec(1,2,width_ratios=(1,4))
            self.ax1 = plt.subplot(self.gs[0])
            self.ax2 = plt.subplot(self.gs[1],sharey=self.ax1)
            self.ax1.invert_xaxis()
            plt.setp(self.ax2.get_yticklabels(), visible=False)
            plt.subplots_adjust(wspace=0.1)

        if style == 'four':
            self.fig  = plt.figure(figsize=(12,8))
            self.ax1  = plt.subplot(221)
            self.ax2  = plt.subplot(222)
            self.ax3  = plt.subplot(223)
            self.ax4  = plt.subplot(224)
            self.ax1.grid(linestyle="dashed",color='gray')
            self.ax2.grid(linestyle="dashed",color='gray')
            self.ax3.grid(linestyle="dashed",color='gray')
            self.ax4.grid(linestyle="dashed",color='gray')
          
    def result_plot(self,subject):

        if subject == "ofs":
            self.xname = f"Number"
            self.yname = rf"$\rm Offset \ (V) $"
            self.plot_window(style="one")
            self.ax.plot(self.ofs,".b")
            #self.ax.plot(self.ofs_mv[100:-100],"r")
            self.ax.hlines(np.median(self.ofs)+self.ofs_std*3,0,len(self.ofs),color="black", linestyles='dashed')
            self.ax.hlines(np.median(self.ofs)-self.ofs_std*3,0,len(self.ofs),color="black", linestyles='dashed')

        if subject == "junk":
            self.xname = rf"$\rm Time \ (s)$"
            self.yname = rf"$\rm Voltage \ (V) $"
            self.plot_window(style="one")
            for i in range(0,self.p_j.shape[0]):
                self.ax.plot(self.tp,self.p_j[i],".")

        if subject == "ph":
            self.xname = rf"$\rm Pulse Height \ (V)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one_large")
            self.ax.hist(self.ph,bins=5000,histtype="step",color='Blue')

        if subject == "ph_current":
            self.xname = rf"$\rm Pulse Height \ (\mu A)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one")
            self.ax.hist(self.ph*self.excVtoI*1e+6,bins=5000,histtype="step",color='Blue')
            print(np.median(self.ph*self.excVtoI*1e+6))

        if subject == "tmpl":
            self.xname = f"Time (s)"
            self.yname = f"Template (arb. unit)"
            self.plot_window(style="one")
            self.ax.plot(self.tp,self.tmpl,"b")

        if subject == "pha":
            self.xname = rf"$\rm Pulse Height \ (V)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one")
            self.ax.hist(self.pha,bins=1024*5,histtype="step")

        if subject == "pha_e":
            self.xname = rf"$\rm Pulse Height \ (V)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one")
            self.ax.hist(self.pha_e,bins=1024*5,histtype="step")

        if subject == "pha_ofs":
            self.plot_window(style="cor")
            self.ax1.hist(self.pha,orientation="horizontal", bins=1024*7, histtype="step")
            self.ax2.plot(self.ofs, self.pha, ".")
            self.ax1.set_ylabel("Pulse height amplitude",fontsize=self.fs)
            self.ax2.set_title("Offset vs PHA",fontsize=self.fs)
            self.ax2.set_xlabel("Offset[V]",fontsize=self.fs) 

        if subject == "se":
            self.xname = r"$\alpha$"
            self.yname = r"$H(\alpha)$"
            self.plot_window(style="one")
            self.ax.plot(self.alpha_list,self.H,".")
            self.ax.set_title("Spectral Entropy",fontsize=self.fs)
            boxdic = {"facecolor" : "white","edgecolor" : "black","boxstyle" : "Round","linewidth" : 1}
            self.fig.text(0.72, 0.82, 'phareg=%.4f to %.4f' %(self.pa[0],self.pa[1]), ha='left',fontsize=20,bbox=boxdic)
            self.fig.text(0.72, 0.74, 'Hmin=%.4f' %(np.min(self.H)), ha='left',fontsize=20,bbox=boxdic)
            self.fig.text(0.72, 0.66, 'αmin=%.4f' %(self.ma), ha='left',fontsize=20,bbox=boxdic)

        if subject == "pha_ofs_cor":
            self.plot_window(style="cor")
            self.ax1.hist(self.pha,orientation="horizontal", bins=1024*5, histtype="step")
            self.ax2.plot(self.ofs, self.pha, ".",label="pha")
            self.ax1.hist(self.pha_e,orientation="horizontal", bins=1024*5, histtype="step")
            self.ax2.plot(self.ofs, self.pha_e, ".",label="pha after correction")
            self.ax1.set_ylabel("Pulse height amplitude",fontsize=self.fs)
            self.ax2.set_title("Offset vs PHA",fontsize=self.fs)
            self.ax2.set_xlabel("Offset[V]",fontsize=self.fs)
            self.ax2.legend(loc="best",fontsize=18)

        if subject == "all_pulse":
            self.xname = rf"$\rm Time \ (s)$"
            self.yname = rf"$\rm Voltage \ (V) $"
            self.plot_window(style="one")
            for i in range(0,len(self.p)):
                self.ax.plot(self.tp,self.p[i])

        if subject == "single_pulse":
            self.xname = rf"$\rm Time \ (s)$"
            self.yname = rf"$\rm Current \ (\mu A) $"
            self.plot_window(style="one")
            self.ax.plot(self.tp,(self.p[0]-self.p[0,0])*self.excVtoI*1e+6)
            self.ax.hlines((np.min(self.p[0])-self.p[0,0])*self.excVtoI*1e+6*0.9,0,self.tp[-1],color="black", linestyles='dashed')
            self.ax.hlines((np.min(self.p[0])-self.p[0,0])*self.excVtoI*1e+6*0.1,0,self.tp[-1],color="black", linestyles='dashed')


        if subject == "rise":
            self.xname = rf"$\rm Time \ (\mu s)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one")
            self.ax.hist(self.tau_r*1e+6,bins=10000,histtype="step",color='Blue',rwidth=4)
            print(np.median(self.tau_r*1e+6))

        if subject == "fall":
            self.xname = rf"$\rm Time \ (\mu s)$"
            self.yname = rf"$\rm Counts $"
            self.plot_window(style="one")
            self.ax.hist(self.tau_f*1e+6,bins=10000,histtype="step",color='Blue',rwidth=4)
            print(np.median(self.tau_f*1e+6))

        if subject == 'tmpl_inf':
            self.plot_window('four')
            self.ax1.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
            self.ax1.set_ylabel(r"$\rm Current \ [\mu A]$",fontsize=20)
            self.ax2.set_xlabel(r"$\rm Time \ [msec]$",fontsize=20)
            self.ax2.set_ylabel(r"$\rm Template \ [\mu A]$",fontsize=20)
            self.ax3.set_xlabel(r"$\rm PS $",fontsize=20)
            self.ax3.set_ylabel(r"$\rm PS $",fontsize=20)
            self.ax4.set_xlabel(r"$\rm SN $",fontsize=20)
            self.ax4.set_ylabel(r"$\rm SN $",fontsize=20)
            #self.ax1.plot(self.avg*1e+6,color='blue')
            self.ax2.plot(self.tmpl,color='blue')
            self.ax3.plot(self.ps,color='blue')
            self.ax4.plot(self.sn,color='blue')
            #plt.legend()
            # self.ax3.set_xscale('log')
            # self.ax3.set_yscale('log')
            self.ax4.set_xscale('log')
            self.ax4.set_yscale('log')
            self.fig.tight_layout()
            print(self.ps)

        plt.show()
        sfn = f"{subject}.png"            
        self.fig.savefig(sfn,dpi=300)

## Analysis ##
    def p_cl(self,psig=20,nsig=5,tsig=5,order=50,trig=0.1):
            length = self.p.shape[1]
            pulse_cl = []
            p_j      = []
            count_t = self.p.shape[0]
            p  = self.p.reshape((1, len(self.p)*length))[0]
            p -= np.median(p)
            p  = p.reshape((int(len(p)/length), length))
            p_mask   = []
            print('checking double pulse...')
            for e, i in enumerate(p):
                p_b  = np.correlate(i,[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1], mode='same')
                flag = argrelextrema(-p_b, np.greater_equal, order=order)[0]
                threshold = -(np.median(p_b[0:90])+(psig*np.std(p_b[0:90])))
                check = p_b[flag]<threshold
                if len(p_b[flag][check==True]) > 1:
                    p_mask.append(False)
                    p_j.append(i)
                else:
                    if length==1000:
                        tmin = 80
                        tmax = 120
                    else:
                        tmin = trig*length*0.80
                        tmax = trig*length*1.20
                    if (tmin<flag[check])&(flag[check]<tmax):
                        pulse_cl.append(i)
                        p_mask.append(True)
                    else:
                        p_mask.append(False)
                        p_j.append(i)
                print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)
            count_j = int(len(p_j))
            count_cl = int(len(pulse_cl))
            #print('\n')
            print(f'Number of Junck pulse  {count_j}')
            print(f'Number of clean events {count_cl}')
            if count_t == 0:
                pass
            else:
                print(f'Pulse removal ratio {count_cl/count_t*100:.2f}%\n')
            #print('\n')
            pulse_cl = np.asarray(pulse_cl)
            self.p_j = np.asarray(p_j)
            self.p_mask = p_mask

    def cal_se(self,pa=[1.5,1.7],sr=[-0.4,-0.1]):

        self.alpha_list = np.arange(sr[0],sr[1],0.001)
        bs = int(round(1e+5/np.max(self.pha),0))
        fs_pha,b_pha = np.histogram(self.pha,bins=bs)
        H = []
        for alpha in self.alpha_list:
            Ha = []
            pha_e = self.pha*(1+alpha*(self.ofs - np.median(self.ofs)))
            fs,b = np.histogram(pha_e,bins=b_pha)
            bin_mask = (pa[0] <= b) & (b <= pa[1])
            fs_r = fs[bin_mask[:-1]]
            ad = np.sum(fs_r)
            bn = b[bin_mask]
            for i in range(0,len(bn)-1):
                if fs_r[i]>0:
                    Ha.append(-fs_r[i]/ad*np.log2(fs_r[i]/ad))
            Hs = np.sum(Ha)
            H.append(Hs)
            print(Hs)
        self.ma = self.alpha_list[np.argmin(H)]
        self.H = np.array(H)
        print(self.ma,np.min(H))
        self.pha_e = self.pha*(1+float(self.ma)*(self.ofs - np.median(self.ofs)))
        print("ma = ",self.ma)
        self.pa = pa
        self.result_plot(subject="se")

    def calculate_rise_fall_time(self,t,p,max=0.8,min=0.2):
        self.tau_r = []
        self.tau_f = []
        p = p - np.median(p[:,:100],axis=1).reshape((len(p),1))
        for i in range(0,len(p)):
            try:
                Imax = np.min(p[i])
                Imax_id = np.argmin(p[i])
                Irise = p[i][:Imax_id]
                trise = t[:Imax_id]
                frise = interpolate.interp1d(Irise,trise)
                trise_max = frise(Imax*max)
                trise_min = frise(Imax*min)
                trise = trise_max - trise_min
                self.tau_r.append(trise)
                Ifall = p[i][Imax_id:]
                tfall = t[Imax_id:]
                ffall = interpolate.interp1d(Ifall,tfall)
                tfall_max = ffall(Imax*max)
                tfall_min = ffall(Imax*min)
                tfall = tfall_min - tfall_max
                self.tau_f.append(tfall)
                if i == 0:
                    print(trise_max,trise_min)
                    print(tfall_max,tfall_min)
                    plt.plot(t,p[i])
                    plt.scatter(trise_max,Imax*max,color='red')
                    plt.scatter(trise_min,Imax*min,color='red')
                    plt.scatter(tfall_max,Imax*max,color='green')
                    plt.scatter(tfall_min,Imax*min,color='green')
                    plt.show()
            except ValueError:
                pass
        self.tau_r = np.array(self.tau_r)
        self.tau_f = np.array(self.tau_f)
        self.result_plot('rise')
        self.result_plot('fall')

    def make_resp(self,rmfname='test.rmf', bmin=1000., bmax=30000.):
        bin_min = bmin/1e3
        bin_max = bmax/1e3
        resp_param = 'genrsp',\
         'inrfil=none',\
         'rmffil='+rmfname,\
         'resol_reln=constant',\
         'resol_file=no',\
         'fwhm=0.0001',\
         'disperse=no',\
         'tlscpe=DUMMY',\
         'instrm=DUMMY',\
         'resp_reln=linear',\
         'resp_low='+str(bin_min),\
         'resp_high='+str(bin_max),\
         'resp_number='+str(int(bin_max-bin_min)),\
         'chan_reln=linear',\
         'chan_low='+str(bin_min),\
         'chan_high='+str(bin_max),\
         'chan_number='+str(int(bin_max-bin_min)),\
         'efffil=none',\
         'detfil=none',\
         'filfil=none',\
         'max_elements=1000000'
        
        resp_param = np.asarray(resp_param)
        subprocess.call(resp_param)

    def histogram(self, pha, binsize=1.0):
        """
        Create histogram
        
        Parameter:
            pha:        pha data (array-like)
            binsize:    size of bin in eV (Default: 1.0 eV)
        
        Return (n, bins)
            n:      photon count
            bins:   bin edge array
        
        Note:
            - bin size is 1eV/bin.
        """
        
        # Create histogram
        bins = np.arange(np.floor(pha.min()), np.ceil(pha.max())+binsize, binsize)
        n, bins = np.histogram(pha, bins=bins)
        
        return n, bins

    def group_bin(self, n, bins, min=1):
        """
        Group PHA bins to have at least given number of minimum counts
        
        Parameters (and their default values):
            n:      counts
            bins:   bin edges
            min:    minimum counts to group (Default: 100)
        
        Return (grouped_n, grouped_bins)
            grouped_n:      grouped counts
            grouped_bins:   grouped bin edges
        """
        
        grp_n = []
        grp_bins = [bins[0]]

        n_sum = 0

        for p in zip(n, bins[1:]):
            n_sum += p[0]
            
            if n_sum >= min:
                grp_n.append(n_sum)
                grp_bins.append(p[1])
                n_sum = 0
        
        return np.asarray(grp_n), np.asarray(grp_bins)

    def fits2xspec(self,binsize=1, exptime=1, fwhm=0.0001, gresp=False, garf=False, name='test', arfname='test.arf', TEStype='TMU542', Datatype='PHA', chan='ch65',pha=None):
        pha = pha
        # separate bins
        if Datatype=='PHA':
            n, bins = self.histogram(pha[pha>0], binsize=binsize)
        else:
            n, bins = self.histogram(pha[pha>0.05], binsize=binsize)
        # py.figure()
        # py.hist(pha, bins=bins, histtype='stepfilled', color='k')
        # py.show()

        # par of fits
        filename = name + ".fits"
        rmfname = name + ".rmf"
        Exposuretime = exptime
        tc = int(n.sum())
        chn = len(bins)-1
        #x = (bins[:-1]+bins[1:])/2
        x = np.arange(0, (len(bins)-1), 1)
        y = n

        # print('#fits file name : %s' %(filename))
        # print('#info to make responce')
        # print('genrsp')
        # print('none')
        # print('%s' %rmfname)
        # print('constant')
        # print('no')
        # print('%f' %fwhm)
        # print('linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( 'linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( '5000000')
        # print( 'DUMMY')
        # print( 'DUMMY')
        # print( 'none')
        # print( 'none')
        # print( 'none')
        # print( '\n')
        
        resp_param = 'genrsp',\
         'inrfil=none',\
         'rmffil='+rmfname,\
         'resol_reln=constant',\
         'resol_file=no',\
         'fwhm='+str(fwhm),\
         'disperse=no',\
         'tlscpe=DUMMY',\
         'instrm=DUMMY',\
         'resp_reln=linear',\
         'resp_low='+str(bins.min()/1.e3),\
         'resp_high='+str(bins.max()/1.e3),\
         'resp_number='+str(int((bins.max()-bins.min())/binsize)),\
         'chan_reln=linear',\
         'chan_low='+str(bins.min()/1.e3),\
         'chan_high='+str(bins.max()/1.e3),\
         'chan_number='+str(int((bins.max()-bins.min())/binsize)),\
         'efffil=none',\
         'detfil=none',\
         'filfil=none',\
         'max_elements=5000000'
        
        resp_param = np.asarray(resp_param)
        while True:
            try:
                subprocess.call(resp_param)
                break
            except OSError:
                print('Please install HEASOFT')
        
        if gresp==True:
            pass
            # if TEStype == 'Th229':
            #     mrTh._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
            # else:
            #     mr._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
        
        if garf==True:
            xspec.arf._make_arf_(pha, binsize=1, aname=arfname, chan=chan)

        # make fits
        col_x = pf.Column(name='CHANNEL', format='J', array=np.asarray(x))
        col_y = pf.Column(name='COUNTS', format='J', unit='count', array=np.asarray(y))
        cols  = pf.ColDefs([col_x, col_y])
        tbhdu = pf.BinTableHDU.from_columns(cols)

        exthdr = tbhdu.header
        exthdr['XTENSION'] = ('BINTABLE', 'binary table extension')
        exthdr['EXTNAME']  = ('SPECTRUM', 'name of this binary table extension')
        exthdr['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
        exthdr['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
        exthdr['HDUVERS1'] = ('1.2.0', 'Obsolete - included for backwards compatibility')
        exthdr['HDUVERS']  = ('1.2.0', 'Version of format (OGIP memo OGIP-92-007)')
        exthdr['HDUCLAS2'] = ('TOTAL', 'Gross PHA Spectrum (source + bkgd)')
        exthdr['HDUCLAS3'] = ('COUNT', 'PHA data stored as Counts (not count/s)')
        exthdr['TLMIN1']   = (0, 'Lowest legal channel number')
        exthdr['TLMAX1']   = (chn-1, 'Highest legal channel number')
        exthdr['TELESCOP'] = ('TES', 'Telescope (mission) name')
        exthdr['INSTRUME'] = (TEStype, 'Instrument name')
        exthdr['FILTER']   = ('NONE', 'no filter in use')
        exthdr['EXPOSURE'] = (exptime, 'Exposure time')
        exthdr['AREASCAL'] = (1.000000E+00, 'area scaling factor') #??
        exthdr['BACKFILE'] = ('none', 'associated background filename')
        exthdr['BACKSCAL'] = (1, 'background file scaling factor')
        exthdr['CORRFILE'] = ('none', 'associated correction filename')
        exthdr['CORRSCAL'] = (1.000000E+00, 'correction file scaling factor')
        exthdr['RESPFILE'] = ('none', 'associated redistrib matrix filename')
        exthdr['ANCRFILE'] = ('none', 'associated ancillary response filename')
        exthdr['PHAVERSN'] = ('1992a', 'obsolete')
        exthdr['DETCHANS'] = (chn, 'total number possible channels')
        exthdr['CHANTYPE'] = ('PI', 'channel type (PHA, PI etc)')
        exthdr['POISSERR'] = (bool(True), 'Poissonian errors to be assumed')
        exthdr['STAT_ERR'] = (0, 'no statistical error specified')
        exthdr['SYS_ERR']  = (0, 'no systematic error specified')
        exthdr['GROUPING'] = (0, 'no grouping of the data has been defined')
        exthdr['QUALITY']  = (0, 'no data quality information specified')
        #HISTORY  FITS SPECTRUM extension written by WTPHA2 1.0.1
        exthdr['DATAMODE'] = ('STANDARD', 'Datamode')
        exthdr['OBJECT']   = ('PERSEUS CLUSTER', 'Name of observed object')
        exthdr['ONTIME']   = (exptime, 'On-source time')
        exthdr['LIVETIME'] = (exptime, 'On-source time')
        exthdr['DATE-OBS'] = ('2006-08-29T18:55:07', 'Start date of observations')
        exthdr['DATE-END'] = ('2006-09-02T01:54:19', 'End date of observations')
        exthdr['TSTART']   = (0, 'start time of experiment in total second')
        exthdr['TSTOP']    = (0, 'end time of experiment in total second')
        exthdr['TELAPSE']  = (exptime, 'elapsed time')
        exthdr['MJD-OBS']  = (exptime, 'MJD of data start time')
        exthdr['MJDREFI']  = (51544, 'MJD reference day')
        exthdr['MJDREFF']  = (7.428703703703700E-04, 'MJD reference (fraction of day)')
        exthdr['TIMEREF']  = ('LOCAL', 'reference time')
        exthdr['TIMESYS']  = ('TT', 'time measured from')
        exthdr['TIMEUNIT'] = ('s', 'unit for time keywords')
        exthdr['EQUINOX']  = (2.000E+03, 'Equinox of celestial coord system')
        exthdr['RADECSYS'] = ('FK5', 'celestial coord system')
        # exthdr['USER']     = ('tasuku', 'User name of creator')
        exthdr['FILIN001'] = ('PerCluster_work1001.xsl', 'Input file name')
        exthdr['FILIN002'] = ('PerCluster_work1002.xsl', 'Input file name')
        exthdr['CREATOR']  = ('extractor v5.23', 'Extractor')
        exthdr['DATE']     = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')
        exthdr['ORIGIN']   = ('NASA/GSFC', 'origin of fits file')
        exthdr['TOTCTS']   = (tc, 'Total counts in spectrum')
        exthdr['RA_PNT']   = (4.993100000000000E+01, 'File average of RA(degrees)')
        exthdr['DEC_PNT']  = (4.152820000000000E+01, 'File average of DEC(degrees)')
        exthdr['SPECDELT'] = (1, 'Binning factor for spectrum')
        exthdr['SPECPIX']  = (0, 'The rebinned channel corresponding to SPECVAL')
        exthdr['SPECVAL']  = (0.000000000000000E+00, 'Original channel value at center of SPECPIX')
        exthdr['DSTYP1']   = ('GRADE', 'Data subspace descriptor: name')
        exthdr['DSVAL1']   = ('0:11', 'Data subspace descriptor: value')
        exthdr['DSTYP2']   = (Datatype, 'Data subspace descriptor: name')
        exthdr['DSVAL2']   = ('0:4095', 'Data subspace descriptor: value')
        exthdr['DSTYP3']   = ('POS(X,Y)', 'Data subspace descriptor: name')
        exthdr['DSREF3']   = (':REG00101', 'Data subspace descriptor: reference')
        exthdr['DSVAL3']   = ('TABLE', 'Data subspace descriptor: value')
        #HISTORY extractor v5.23
        exthdr['CHECKSUM'] = ('54fW64ZV54dV54ZV', 'HDU checksum updated 2014-04-07T08:16:02')
        exthdr['DATASUM']  = ('9584155', 'data unit checksum updated 2014-04-07T08:16:02')

        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu, tbhdu])
        thdulist.writeto(filename)

    def cross_correlate(self, data1, data2, max_shift=None, method='interp'):
        """
        Calculate a cross correlation for a given set of data.
        
        Parameters (and their default values):
            data1:      pulse/noise data (array-like)
            data2:      pulse/noise data (array-like)
            max_shift:  maximum allowed shifts to calculate maximum cross correlation
                        (Default: None = length / 2)
            method:     interp - perform interpolation for obtained pha and find a maximum
                                (only works if max_shift is given)
                        integ  - integrate for obtained pha
                        none   - take the maximum from obtained pha
                        (Default: interp)
        
        Return (max_cor, shift)
            max_cor:    calculated max cross correlation
            shift:      required shift to maximize cross correlation
            phase:      calculated phase
        """

        # Sanity check
        if len(data1) != len(data2):
            raise ValueError("data length does not match")

        # if given data set is not numpy array, convert them
        data1 = np.asarray(data1).astype(dtype='float64')
        data2 = np.asarray(data2).astype(dtype='float64')
        
        # Calculate cross correlation
        if max_shift == 0:
            return np.correlate(data1, data2, 'valid')[0] / len(data1), 0, 0
        
        # Needs shift
        if max_shift is None:
            max_shift = len(data1) / 2
        else:
            # max_shift should be less than half data length
            max_shift = min(max_shift, len(data1) / 2)
        
        # Calculate cross correlation

        cor = np.correlate(data1, np.concatenate((data2[-max_shift:], data2, data2[:max_shift])), 'valid')
        ind = cor.argmax()

        self.concate_data = np.concatenate((data2[-max_shift:], data2, data2[:max_shift]))
        self.cor = cor
        self.ind = ind

        if method == 'interp' and 0 < ind < len(cor) - 1:
            return (cor[ind] - (cor[ind-1] - cor[ind+1])**2 / (8 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))) / len(data1), ind - max_shift, (cor[ind-1] - cor[ind+1]) / (2 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))
        elif method == 'integ':
            return sum(cor), 0, 0
        elif method in ('none', 'interp'):
            # Unable to interpolate, and just return the maximum
            return cor[ind] / len(data1), ind - max_shift, 0
        else:
            raise ValueError("Unsupported method")

    def save_pcl(self,file):
        with h5py.File(file,"a") as f:
            if 'pmask' in f:
                del f['pmask']
            f.create_dataset('pmask',data=self.p_mask)

    def load_pmask(self,file):
        with h5py.File(file,"a") as f:        
            self.p_mask = f['pmask'][:]

## out func ##
    def resolution_fit(self):
        self.fig  = plt.figure(figsize=(7,6))
        self.ax  = plt.subplot(111)
        self.ax.grid(linestyle="dashed")
        self.ph = 5.9 * self.ph / np.median(self.ph[1:])
        hist, bins = np.histogram(self.ph[1:],bins=100)
        print(len(self.ph))
        bins_mid = bins[:-1] + np.diff(bins)
        pcov, popt = curve_fit(self.gaus,bins_mid,hist,p0=[1,1e-3,np.median(self.ph[1:])])
        print(pcov,popt)
        print(pcov[1])
        res = np.abs(2*pcov[1]*np.sqrt(2*np.log(2))*1e+3)
        print(f'Energy resolution = {np.abs(res)} eV (FWHM)')
        self.ax.step(bins_mid,hist,where='mid',color='black',label='data')
        self.ax.plot(bins_mid,self.gaus(bins_mid,*pcov),color='Red',label='fit')
        self.ax.set_xlabel(r'$\rm Energy \ [keV]$',fontsize=20)
        self.ax.set_ylabel(r'$\rm Counts $',fontsize=20)
        res_2f = format(res, '.2f')
        txt = rf'$\rm \Delta E \ = \ {res_2f} \ eV$'
        self.ax.text(0.025,0.9,txt, transform=self.ax.transAxes,fontsize=20,backgroundcolor='white')
        plt.legend(loc='best',fontsize=20)
        self.fig.savefig('resolution.png',dpi=300)
        #baseE = Analysis.baseline(self.sn,14.4e+3)
        #print(f'base line resolution = {baseE}')
        plt.show()

    def out(self):
        self.load_fits(file="run011_row")
        self.ofs_filter(2000,10000)
        self.load_func()
        self.offset_sigma_filter(3)
        self.result_plot(subject="ofs")
        self.result_plot(subject="ph")
        #self.save_pulse()
        #self.load_pulse()
        self.t = self.tp
        #self.rise_fall_fit_s()
        self.load_func()
        self.result_plot(subject="ph")
        self.ph_filter(xmin=0.906,xmax=0.912)
        #self.filtering()
        #self.save_pulse()
        #self.all_fit()
        #self.result_plot(subject="single_pulse")
        #self.trig_position(subject="data",num=0,trig_max=0.9,trig_min=0.1)
        #self.rise_fall_fit()
        #self.all_fit()
        #self.result_plot(subject="all_pulse")
        self.gen_pha()
        self.save_tmpl()
        #self.load_pha()
        #self.cal_NEP()
        #self.load_pha()
        self.result_plot(subject="pha")
        self.result_plot(subject="pha_ofs")
        self.cal_se(pa=[0.90,0.96])
        self.save_se()
        #self.load_se()
        self.result_plot(subject="pha_ofs_cor")
        self.result_plot(subject="pha_e")

    def out3(self):
        self.load_hdf5('run045_b64.hdf5')
        # self.p_cl()
        # self.p = self.p[self.p_mask]
        # self.n = self.n[self.p_mask]
        # self.save_pcl('run045_analysis.hdf5')
        self.load_func()
        # self.offset_sigma_filter(3)
        # self.result_plot('ofs')
        # self.result_plot('ph')
        # self.load_func()
        self.ph_filter(xmin=0.175,xmax=0.185)

        p1 = self.p[self.ph_mask]
        ph1 = self.ph[self.ph_mask]

        self.ph_filter(xmin=0.280,xmax=0.30)
        p2 = self.p[self.ph_mask]
        ph2 = self.ph[self.ph_mask]
        self.ph_filter(xmin=0.51,xmax=0.53)
        p3 = self.p[self.ph_mask]
        ph3 = self.ph[self.ph_mask]
        self.ph_filter(xmin=1.41,xmax=1.43)
        p4 = self.p[self.ph_mask]
        ph4 = self.ph[self.ph_mask]
        all_rise,all_fall,all_ph = [], [], []
        self.calculate_rise_fall_time(self.t,p1)
        all_rise.append(self.tau_r)
        all_fall.append(self.tau_f)
        self.calculate_rise_fall_time(self.t,p2)
        all_rise.append(self.tau_r)
        all_fall.append(self.tau_f)
        self.calculate_rise_fall_time(self.t,p3)
        all_rise.append(self.tau_r)
        all_fall.append(self.tau_f)
        self.calculate_rise_fall_time(self.t,p4)
        all_rise.append(self.tau_r)
        all_fall.append(self.tau_f)
        plt.hist(all_fall,bins=10000,histtype='step')
        plt.show()
        # self.gen_tmpl(max_shift=max_shift)
        # self.pha, self.ps, test = self.cross_correlate(self.tmpl, self.p[0], max_shift=max_shift)
        #plt.plot(self.concate_data)
        # plt.plot(self.cor)
        # print(self.cor.shape,self.tmpl.shape,self.p[0].shape)
        #plt.scatter(self.cor[self.ind])
        # plt.show()

    def rise_fall(self):
        self.load_fits('run011_row')
        self.load_func()
        #self.result_plot('ofs')
        self.offset_sigma_filter(3)
        self.load_func()
        #self.result_plot('ofs')
        self.offset_sigma_filter(3)
        self.load_func()
        #self.result_plot('ph')
        self.ph_filter(xmin=1.34,xmax=1.38)
        self.filtering()
        self.gen_pha()
        self.save_tmpl()

    def out2(self):
        self.load_fits('run011_row')
        self.load_func()
        self.result_plot('ofs')
        self.offset_sigma_filter(3)
        self.load_func()
        self.result_plot('ofs')
        self.offset_sigma_filter(3)
        self.load_func()
        self.result_plot('ph')
        self.ph_filter(xmin=1.34,xmax=1.38)
        self.gen_pha()
        self.save_tmpl()
        self.save_pulse()

    def out2_2(self):
        self.load_tmpl()

        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        self.ax.grid(linestyle='dashed') 
        self.ax.set_xlabel('PHA')
        self.ax.set_ylabel('Counts')
        self.ax.hist(self.pha,bins=10000,histtype='step')
        self.fig.savefig('pha.png',dpi=300)
        plt.show()

        MnKa_mask = (1.3 < self.pha) & (self.pha < 1.4)
        Ka1_mask  = (1.360 < self.pha) & (self.pha < 1.365)
        Ka2_mask  = (1.355 < self.pha) & (self.pha < 1.360)
        Kb_mask   = (1.476 < self.pha) & (self.pha < 1.484)
        pha_Ka1_med = np.median(self.pha[Ka1_mask])
        pha_Ka2_med = np.median(self.pha[Ka2_mask])
        pha_Kb_med  = np.median(self.pha[Kb_mask])
        E_ka2 = 5887.338044247787
        E_ka1 = 5897.91000621118
        E_kb  = 6490.0
        E     = np.array([0,E_ka2,E_ka1,E_kb])
        pha_med = np.array([0,pha_Ka2_med,pha_Ka1_med,pha_Kb_med])    
        popt, pcov = curve_fit(self.quad,pha_med,E)
        print(popt,pcov)
        pha_fake = np.linspace(0,1.5,100)
        resid = E - self.quad(pha_med,*popt)
        # plt.scatter(pha_med,E)
        # plt.plot(pha_fake,self.quad(pha_fake,*popt),'-.')
        # plt.show()

        self.fig = plt.figure(figsize=(8,6))
        self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
        self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
        self.ax  = self.fig.add_subplot(self.gs1[:,:])
        self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
        self.ax.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")
        self.ax.set_ylabel('Energy [eV]')
        self.ax2.set_ylabel('Residual [eV]')
        self.ax2.set_xlabel('PHA')
        self.ax.scatter(pha_med,E,color='black')
        self.ax.plot(pha_fake,self.quad(pha_fake,*popt),'--',color='red')
        self.ax2.scatter(pha_med,resid,color='black')
        self.ax2.hlines(0,np.min(pha_fake),np.max(pha_fake),linestyle='dashed',color='red')
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        plt.show()
        self.fig.savefig('calibration.png',dpi=300)
        plt.hist(self.quad(self.pha,*popt),bins=10000,histtype='step')
        plt.show()
        cal_res = np.array([*popt])
        with h5py.File(self.savehdf5,"a") as f:
            if 'cal' in f.keys():
                del f['cal']
            f.create_dataset('cal',data=cal_res)
        data = self.quad(self.pha,*popt)
        E_mask = (5870 < data) & (data < 5920)
        n,bins = self.histogram(data[E_mask])
        n,bins = self.group_bin(n,bins,min=1)
        bins = bins[:-1] + np.diff(bins)
        print(bins)
        plt.scatter(bins,n,color='black')
        plt.show()   
        self.MnKafit(x=bins,y=n) 
        #self.fits2xspec(pha=data[E_mask])

    def base_ana(self):
        self.load_tmpl()
        self.load_pulse()
        self.gen_baseline()
        with h5py.File(self.savehdf5,"r") as f:
            cal = f['cal'][:]
        #plt.hist(self.baseline,bins=100,histtype='step')
        self.base_E = self.quad(self.baseline,*cal)
        # plt.hist(self.base_E,bins=100,histtype='step')
        n,bins = self.histogram(self.base_E)
        n,bins = self.group_bin(n,bins,min=1)
        bins_mid = bins[:-1] + np.diff(bins)
        popt,pcov = curve_fit(self.gaus,bins_mid,n)
        print(popt,pcov)
        pha_fake = np.linspace(np.min(bins_mid),np.max(bins_mid),100)
        resid = n - self.gaus(bins_mid,*popt)
        print(Analysis.sigma2fwhm(popt[1]))

        self.fig = plt.figure(figsize=(8,6))
        self.gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        self.gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[0,:])
        self.gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=self.gs[1,:])
        self.ax  = self.fig.add_subplot(self.gs1[:,:])
        self.ax2 = self.fig.add_subplot(self.gs2[:,:],sharex=self.ax)
        self.ax.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")
        self.ax.set_ylabel('Counts')
        self.ax2.set_ylabel('Residual')
        self.ax2.set_xlabel('Energy [eV]')
        self.ax.scatter(bins_mid,n,color='black')
        self.ax.plot(pha_fake,self.gaus(pha_fake,*popt),'--',color='red')
        self.ax2.scatter(bins_mid,resid,color='black')
        self.ax2.hlines(0,np.min(pha_fake),np.max(pha_fake),linestyle='dashed',color='red')
        self.fig.subplots_adjust(hspace=.0)
        self.fig.align_labels()
        plt.show()
        self.fig.savefig('base_fit.png',dpi=300)

    def MnKafit(self,x,y):
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'serif'
        import numpy as np
        import scipy as sp
        import scipy.optimize as so
        import scipy.special

        def calcchi(params,consts,model_func,xvalues,yvalues,yerrors):
            model = model_func(xvalues,params,consts)
            chi = (yvalues - model) / yerrors
            return(chi)

        # optimizer
        def solve_leastsq(xvalues,yvalues,yerrors,param_init,consts,model_func):
            param_output = so.leastsq(
                calcchi,
                param_init,
                args=(consts,model_func, xvalues, yvalues, yerrors),
                full_output=True)
            param_result, covar_output, info, mesg, ier = param_output
            error_result = np.sqrt(covar_output.diagonal())
            dof = len(xvalues) - 1 - len(param_init)
            chi2 = np.sum(np.power(calcchi(param_result,consts,model_func,xvalues,yvalues,yerrors),2.0))
            return([param_result, error_result, chi2, dof])

        def mymodel(x,params, consts, tailonly = False):
            norm,gw,gain,P_tailfrac,P_tailtau,bkg1,bkg2 = params    
            # norm : nomarlizaion
            # gw : sigma of gaussian
            # gain : gain of the spectrum 
            # P_tailfrac : fraction of tail 
            # P_tailtau : width of the low energy tail
            # bkg1 : constant of background
            # bkg2 : linearity of background    
            initparams = [norm,gw,gain,bkg1,bkg2]
            def rawfunc(x): # local function, updated when mymodel is called 
                return MnKalpha(x,initparams,consts=consts)               
            model_y = smear(rawfunc, x, P_tailfrac, P_tailtau, tailonly=tailonly)
            return model_y

       

        def MnKalpha(xval,params,consts=[]):
            norm,gw,gain,bkg1,bkg2 = params
            # norm : normalization 
            # gw : sigma of the gaussian 
            # gain : if gain changes
            # consttant facter if needed 
            # Mn K alpha lines, Holzer, et al., 1997, Phys. Rev. A, 56, 4554, + an emperical addition
            energy = np.array([ 5898.853, 5897.867, 5894.829, 5896.532, 5899.417, 5902.712, 5887.743, 5886.495])
            lgamma =  np.array([    1.715,    2.043,    4.499,    2.663,    0.969,   1.5528,    2.361,    4.216]) # full width at half maximum
            amp =    np.array([    0.790,    0.264,    0.068,    0.096,   0.0714,   0.0106,    0.372,      0.1])

            prob = (amp * lgamma) / np.sum(amp * lgamma) # probabilites for each lines. 

            model_y = 0 
            if len(consts) == 0:
                consts = np.ones(len(energy))
            else:
                consts = consts

            for i, (ene,lg,pr,con) in enumerate(zip(energy,lgamma,prob,consts)):
                voi = voigt(xval,[ene*gain,lg*0.5,gw])
                model_y += norm * con * pr * voi

            background = bkg1 * np.ones(len(xval)) + (xval - np.mean(xval)) * bkg2
            model_y = model_y + background
            # print "bkg1,bkg2 = ", bkg1,bkg2, background
            return model_y

        def voigt(xval,params):
            center,lw,gw = params
            # center : center of Lorentzian line
            # lw : HWFM of Lorentzian (half-width at half-maximum (HWHM))
            # gw : sigma of the gaussian 
            z = (xval - center + 1j*lw)/(gw * np.sqrt(2.0))
            w = scipy.special.wofz(z)
            model_y = (w.real)/(gw * np.sqrt(2.0*np.pi))
            return model_y

        def smear(rawfunc, x, P_tailfrac, P_tailtau, tailonly = False):
            if P_tailfrac <= 1e-5:
                return rawfunc(x)

            dx = x[1] - x[0]
            freq = np.fft.rfftfreq(len(x), d=dx)
            rawspectrum = rawfunc(x)
            ft = np.fft.rfft(rawspectrum)
            if tailonly:
                ft *= P_tailfrac * (1.0 / (1 - 2j * np.pi * freq * P_tailtau) - 0)
            else:
                ft += ft * P_tailfrac * (1.0 / (1 - 2j * np.pi * freq * P_tailtau) - 1)

            smoothspectrum = np.fft.irfft(ft, n=len(x))
            if tailonly:
                pass
            else:
                smoothspectrum[smoothspectrum < 0] = 0
            return smoothspectrum    

        yerr = np.sqrt(y)

        gfwhm = 6
        gw = gfwhm / 2.35
        norm = 800.0
        gain = 1.0001
        bkg1 = 1.0
        bkg2 = 0.0
        P_tailfrac = 0.25
        P_tailtau = 10
        init_params=[norm,gw,gain,P_tailfrac,P_tailtau,bkg1,bkg2]
        consts = [1,1,1,1,1,1,1,1]

        model_y = mymodel(x,init_params,consts)

        plt.figure(figsize=(12,8))
        plt.title("Mn Kalpha fit (initial values)")
        plt.xlabel("Energy (eV)")
        plt.errorbar(x, y, yerr=yerr, fmt='ko', label = "data")
        plt.plot(x, model_y, 'r-', label = "model")
        plt.legend(numpoints=1, frameon=False, loc="upper left")
        plt.grid(linestyle='dotted',alpha=0.5)
        plt.savefig("fit_MnKalpha_init.png")
        plt.show()

        # do fit 
        result, error, chi2, dof = solve_leastsq(x, y, yerr, init_params, consts, mymodel)

        # get results 
        norm = np.abs(result[0])
        norme = np.abs(error[0])

        gw = np.abs(result[1])
        gwe = np.abs(error[1])

        gain = np.abs(result[2])
        gaine = np.abs(error[2])

        tailfrac = np.abs(result[3])
        tailfrac_e = np.abs(error[3])
        tailtau = np.abs(result[4])
        tailtau_e = np.abs(error[4])

        bkg1 = np.abs(result[5])
        bkg1e = np.abs(error[5])
        bkg2 = np.abs(result[6])
        bkg2e = np.abs(error[6])

        fwhm = 2.35 * gw
        fwhme = 2.35 * gwe

        label1 = "N = " + str("%4.2f(+/-%4.2f)" % (norm,norme)) + " g = " + str("%4.2f(+/-%4.2f)" % (gain,gaine)) + " dE = " + str("%4.2f(+/-%4.2f)" % (fwhm,fwhme) + " (FWHM)")
        label2 = "tailfrac = " + str("%4.2f(+/-%4.2f)" % (tailfrac,tailfrac_e)) + ", tailtau = " + str("%4.2f(+/-%4.2f)" % (tailtau,tailtau_e)) 
        label3 = "chi/dof = " + str("%4.2f"%chi2) + "/" + str(dof) + " = " + str("%4.2f"%  (chi2/dof))
        label4 = "bkg1 = " + str("%4.2f(+/-%4.2f)" % (bkg1,bkg1e)) + " bkg2 = " + str("%4.2f(+/-%4.2f)" % (bkg2,bkg2e))

        fitmodel = mymodel(x,result,consts)
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
        plt.title("Mn Kalpha fit : " + label1 + "\n" + label2 + ", " + label3 + "\n" + label4)
        #plt.xlabel("Energy (eV)")
        plt.errorbar(x, y, yerr=yerr, fmt='ko', label = "data")
        plt.plot(x, fitmodel, 'r-', label = "model")
        background = bkg1 * np.ones(len(x)) + (x - np.mean(x)) * bkg2
        plt.plot(x, background, 'b-', label = "background", alpha = 0.9, lw = 1)
        eye = np.eye(len(consts))
        for i, oneeye in enumerate(eye):
            plt.plot(x, mymodel(x,result,consts=oneeye), alpha = 0.7, lw = 1, linestyle="--", label = str(i+1))
        plt.grid(linestyle='dotted',alpha=0.5)
        plt.legend(numpoints=1, frameon=False, loc="upper left")


        ax2 = plt.subplot2grid((3,1), (2,0))    
        plt.xscale('linear')
        plt.yscale('linear')
        plt.xlabel(r'Energy (eV)')
        plt.ylabel(r'Resisual')
        resi = fitmodel - y 
        plt.errorbar(x, resi, yerr = yerr, fmt='ko')
        plt.legend(numpoints=1, frameon=False, loc="best")
        plt.grid(linestyle='dotted',alpha=0.5)    
        plt.savefig("fit_MnKalpha_result.png")
        plt.show()

    def MnKa1(self):
        energy = np.array([ 5898.853, 5897.867, 5894.829, 5896.532, 5899.417, 5902.712])
        lgamma =  np.array([    1.715,    2.043,    4.499,    2.663,    0.969,   1.5528]) # full width at half maximum
        amp =    np.array([    0.790,    0.264,    0.068,    0.096,   0.0714,   0.0106])
        E_med = np.sum(energy*amp)
        print(E_med)     

    def MnKa_ration(self):
        """_summary_
        HOLZER + 1997
        E -> LINE ENERGY
        Iint -> Integrated intensity
        F -> Normalization factor
        """
        E_Ka1 = np.array([5898.853, 5897.867, 5894.829, 5896.532, 5899.417])
        Iint_Ka1 = np.array([0.353, 0.141, 0.079, 0.066, 0.005])
        E_Ka2 = np.array([5887.743, 5886.495])
        Iint_Ka2 = np.array([0.229,0.110])
        F_Ka1 = 1/np.sum(Iint_Ka1)
        F_Ka2 = 1/np.sum(Iint_Ka2)

        CenterEnergy_Ka1 = np.sum(E_Ka1 * (Iint_Ka1 * F_Ka1))
        CenterEnergy_Ka2 = np.sum(E_Ka2 * (Iint_Ka2 * F_Ka2))

        print(CenterEnergy_Ka1)
        print(CenterEnergy_Ka2)


    def tmpl_ana(self):
        self.load_tmpl()
        self.result_plot('tmpl_inf')


    def offset_analysis(self,file):
        self.load_fits(file=file)
        self.load_func()
        self.result_plot(subject="ofs")

    def ph_analysis(self,file,ofs_min,ofs_max):
        self.load_fits(file=file)
        self.ofs_filter(ofs_min,ofs_max)
        self.load_func()
        self.offset_sigma_filter(3)
        self.result_plot(subject="ofs")
        self.result_plot(subject="ph")
        self.result_plot(subject="ph_current")
        self.all_fit()
        self.result_plot(subject="all_pulse")

    def pha_analysis(self):
        self.load_hdf5()
        pha = self.pha_e
        self.fits2xspec(name='test',pha=pha)

    def noise_spectrum(self,data,time):
        spec = Filter.power(data)
        self.hres = time[1] - time[0]
        self.frq = np.fft.rfftfreq(data.shape[-1],self.hres)
        self.spec = np.sqrt(spec*(self.hres*data.shape[-1]))*self.excVtoI

    def noise_ana(self):
        self.load_fits('run011_row')
        self.all_spec = []
        for i in range(0,1000):
            self.noise_spectrum(data=self.n[i],time=self.tn)
            self.all_spec.append(self.spec)
        avg = np.average(self.all_spec,axis=0)
        sig = np.std(self.all_spec,axis=0)
        spec_ar = np.array(self.all_spec)
        print(spec_ar.shape)
        plt.errorbar(self.frq,avg,yerr=sig)
        #plt.step(self.frq,avg)
        #a = np.sqrt(Filter.average_noise(self.n)*(self.hres*self.n.shape[-1])) * self.excVtoI
        plt.step(self.frq,avg)
        plt.semilogx()
        plt.semilogy()
        plt.show()