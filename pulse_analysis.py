import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from lmfit import Model
from scipy import integrate
from pytes import Util,Filter,Analysis

class PulseAnalysis:
    def __init__(self) -> None:
        self.JtoeV   = 1/1.60217662e-19
        self.Min    = 10.118e-11  #[H]
        self.Mfb    = 8.6e-11  #[H]
        self.Rfb     = 100e+3
        self.excVtoI = self.Mfb / (self.Min * self.Rfb)

    def gaus(self, x, norm, sigma, mu):
        return norm * np.exp(-(x-mu)**2 / (2*sigma**2))

    def normalized_gaus(self, x, sigma, mu):
        return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2 / (2*sigma**2))

    def load_setting_file(self,run_number):
        self.setting_file_name = 'setting.hdf5'
        self.f = h5py.File(self.setting_file_name,'a')
        self.Pulse_ID_list = list(self.f['Pulse'].keys())
        self.Pulse_ID_list.sort()
        print('-----------------------------------------------------------')
        if run_number in self.Pulse_ID_list:
            self.run_number = run_number
            self.squid_Ib = int(self.f[f'Pulse/{self.run_number}/SQUID_current_bias'][...])
            print(f'{self.run_number} SQUID settings')
            print(f'SQUID current bias = {self.squid_Ib} uA')
            if f'SB{self.squid_Ib}uA' in self.f['Min'].keys():
                self.Min = self.f[f'Min/SB{self.squid_Ib}uA/average'][...]
                self.Mfb = self.f[f'Mfb/SB{self.squid_Ib}uA/average'][...]
            elif f'Sb{self.squid_Ib}uA' in self.f['Min'].keys():
                self.Min = self.f[f'Min/Sb{self.squid_Ib}uA/average'][...]
                self.Mfb = self.f[f'Mfb/Sb{self.squid_Ib}uA/average'][...]
            else:
                print(f'SB{self.squid_Ib}uA is not in the setting file.')
                print('SQUID bias current list is as follows:')
                print(self.f['Min'].keys())
            self.Rfb = self.f[f'Pulse/{self.run_number}/Rfb'][...]*1e+3
            self.excVtoI = self.Mfb / (self.Min * self.Rfb)
            print(f'Min = {self.Min*1e+12} pH')
            print(f'Mfb = {self.Mfb*1e+12} pH')
            print(f'Rfb = {self.Rfb*1e-3} kOhm')
            print(f'SQUID Voltage to Current Factor = {self.excVtoI}')

        else:
            print(f'run_number {run_number} is not in the setting file.')
        print('-----------------------------------------------------------')

    def old_file_converter(self,filename):
        filename = filename
        with h5py.File(filename,'a') as f:
            w = f['waveform/wave'][:]
            if 'pulse' in f['waveform'].keys():
                del f['waveform/pulse'], f['waveform/noise']
            half_m = int(w.shape[1]/2)
            n, p = np.hsplit(w,[half_m])
            f.create_dataset('waveform/pulse',data=p)
            f.create_dataset('waveform/noise',data=n)

    def load_pulse_file(self,pulse_file_name):
        self.pulse_file_name = pulse_file_name
        self.pulse_file = h5py.File(self.pulse_file_name,'a')
        self.vres = self.pulse_file['waveform']['vres'][...]
        self.hres = self.pulse_file['waveform']['hres'][...]
        self.p = self.pulse_file['waveform']['pulse'] * self.vres
        self.n = self.pulse_file['waveform']['noise'] * self.vres
        self.t = np.arange(self.p.shape[-1]) * self.hres
        self.tp = self.t
        self.tn = self.t

    def convert_VtoI(self):
        self.p *= self.excVtoI
        self.n *= self.excVtoI 

    def offset(self):
        self.ofs = np.average(self.n,axis=1)
        self.ofs_std = np.std(self.ofs)
        #self.ofs_mv = self.moving_average(self.ofs,num=50)

    def offset_sigma_filter(self,sigma):
        mask = (np.median(self.ofs)-self.ofs_std*sigma<self.ofs) & (self.ofs<np.median(self.ofs)+self.ofs_std*sigma)
        print(f"{np.median(self.ofs)-self.ofs_std*sigma} - {np.median(self.ofs)+self.ofs_std*sigma}")
        self.p = self.p[mask]
        self.n = self.n[mask] 

    def offset_correction(self):
        self.p = self.p - self.ofs.reshape((len(self.p),1))
        self.n = self.n - self.ofs.reshape((len(self.p),1))

    def pulse_height(self):
        self.ph = self.ofs - np.min(self.p,axis=1)

    def pulse_area(self):
        self.parea = integrate.trapezoid(self.p,self.t)
        print(self.parea)

    def ph_filter(self,xmin,xmax):
        self.ph_mask = (xmin < self.ph) & (self.ph < xmax)

    def filtering(self):
        self.p = self.p[self.ph_mask]
        self.n = self.n[self.ph_mask]
        self.ph = self.ph[self.ph_mask]
        self.ofs = self.ofs[self.ph_mask]

    def gen_pha(self, max_shift):
        p = self.p[self.ph_mask] - self.ofs[self.ph_mask].reshape((len(self.p[self.ph_mask]),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.n[self.ph_mask],max_shift=max_shift)
        self.pha, self.ps = Filter.optimal_filter(self.p, self.tmpl, max_shift=max_shift)
        self.result_plot(subject="tmpl")

    def gen_baseline(self):
        self.baseline, self.base_ps = Filter.optimal_filter(self.n, self.tmpl, max_shift=100)

    def gen_tmpl(self,max_shift):
        p = self.p[self.ph_mask] - self.ofs[self.ph_mask].reshape((len(self.p[self.ph_mask]),1))
        self.tmpl,self.sn = Filter.generate_template(p,self.n[self.ph_mask],max_shift=max_shift)

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
            self.ax1.hist(self.pha,orientation="horizontal", bins=1024, histtype="step")
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

    def gaus_noise_generator(self,sigma):
        pass

    def out(self,run_number,pulse_file_name,min=-0.1):
        #self.load_setting_file(run_number)
        self.load_pulse_file(f'../data/pulse/{pulse_file_name}')
        self.convert_VtoI()
        self.offset()
        plt.plot(self.ofs)
        plt.show()
        plt.hist(self.ofs,bins=100,histtype='step')
        print(len(self.ofs))
        plt.show()
        self.pulse_height()
        plt.hist(self.ph,bins=1000,histtype='step')
        plt.show()
        plt.hist(self.n.flatten(),bins=1000,histtype='step')
        plt.show()
        self.offset_correction()
        self.noise_distribution(min=min)

    def out2(self,run_number,pulse_file_name,min=-0.1):
        self.load_setting_file(run_number)
        self.load_pulse_file(f'../data/pulse/{pulse_file_name}')
        self.convert_VtoI()
        self.offset()
        plt.plot(self.ofs)
        plt.show()
        plt.hist(self.ofs,bins=100,histtype='step')
        print(len(self.ofs))
        plt.show()
        self.pulse_height()
        plt.hist(self.ph,bins=1000,histtype='step')
        plt.show()
        plt.hist(self.n.flatten(),bins=1000,histtype='step')
        plt.show()
        #self.offset_correction()
        self.noise_distribution(min=min)

    def pana(self,filename):
        self.load_pulse_file(filename)
        #self.convert_VtoI()
        self.offset()
        self.offset_sigma_filter(3)
        self.offset()
        # plt.plot(self.ofs,'.')
        # plt.show()
        self.offset_correction()
        self.pulse_area()
        # plt.title('pulse area')
        # plt.hist(-self.parea,bins=10000,histtype='step')
        # plt.show()
        self.pulse_height()
        # plt.title('pulse height')
        # plt.hist(self.ph,bins=1000,histtype='step')
        # plt.show()
        # plt.scatter(-self.parea, self.ph)
        # plt.show()
        self.ph_filter(0,2)
        for i in range(len(self.p[self.ph_mask])):
            plt.plot(self.t,self.p[self.ph_mask][i])
        plt.show()
        #self.ph_filter(1.6e-7,2.1e-7)
        #self.filtering()
        #self.gen_pha()
        #self.result_plot('pha')

        

    def noise_distribution(self,min=-0.1):
        n, bins = self.histogram(self.n.flatten()*1e+6,binsize=1e-3)
        n, bins = self.group_bin(n,bins,min=20)
        bins_mid = bins[:-1] + np.diff(bins)
        print(f"bins_mid = {bins_mid}")
        n, bins_mid = n[bins_mid>min], bins_mid[bins_mid>min]
        total_count = np.sum(n)
        print(f"total count = {total_count}")
        poisson_err = np.sqrt(n)
        self.model = Model(self.gaus) 
        result = self.model.fit(n,x=bins_mid,weights=1/poisson_err,norm=1e+3,sigma=1e-3,mu=0.5868)
        print(result.fit_report())

        self.fig = plt.figure(figsize=(12,8))
        gs = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1])
        self.ax1 = self.fig.add_subplot(gs1[:,:])
        self.ax2 = self.fig.add_subplot(gs2[:,:],sharex=self.ax1)
        self.ax1.grid(linestyle="dashed")
        self.ax2.grid(linestyle="dashed")
        self.ax2.set_xlabel(r"$ \rm Current \ (\mu A)$",fontsize=15)
        self.ax1.set_ylabel(r"$ \rm Count$",fontsize=15)
        self.ax2.set_ylabel(r"$ \rm Residual$",fontsize=15)    
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=.0)

        self.ax1.errorbar(bins_mid,n,yerr=poisson_err,fmt='o',color='black',label='data')
        self.ax1.plot(bins_mid,result.best_fit,color='red',label='fit')
        self.ax2.scatter(bins_mid,(n-result.best_fit)/poisson_err,color='black')
        plt.show()
        self.fig.savefig('./figure/noise_distribution.png',dpi=300)

    def gaus_probability(self,sigma,trig,num):
        from scipy.stats import norm

        # 正規分布のパラメータ
        sigma = sigma
        mu = 0

        # 1回の試行で0.1以上の値を取る確率
        probability_above_01_once = 1 - norm.cdf(trig, mu, sigma)

        # 1000回の試行で0.1以上の値が少なくとも1回出る確率
        probability_at_least_once = 1 - (1 - probability_above_01_once) ** num

        print(f"{num}回の試行で{trig}が一回以上出る確率:", probability_at_least_once)
        return probability_at_least_once


class NoiseAnalysis:

    def __init__(self) -> None:
        self.Min    = 98.96e-12  #[H]
        self.Mfb    = 85.20e-12  #[H]
        self.Rfb    = 100e+3
        self.excVtoI = self.Mfb / (self.Min * self.Rfb)

    def loadfile(self,filename):
        filename = filename
        with h5py.File(filename,'a') as f:
            self.vres = f['waveform']['vres'][...]
            self.hres = f['waveform']['hres'][...]
            self.n = f['waveform']['wave'] * self.vres
            self.t = np.arange(self.n.shape[-1]) * self.hres

    def noise_spectrum(self):
        hres = self.t[1] - self.t[0]
        self.nspec = np.sqrt(Filter.average_noise(self.n)*(hres*self.n.shape[-1])) * self.excVtoI
        self.frq = np.fft.rfftfreq(self.n.shape[-1],hres)
        self.len_n = len(np.fft.rfft(self.n[0]))
        self.len_m = len(self.n[0])

    def plot_init(self):
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
        #plt.rcParams['scatter.edgecolors']    = None
        self.fs = 25
        self.ps = 30
        self.fig = plt.figure(figsize=(12,8))
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.grid(linestyle="dashed")
        self.ax1.set_xlabel(r"$ \rm Frequency \ (Hz)$",fontsize=self.fs)
        self.ax1.set_ylabel(r"$ \rm Current \ Power \ Density \ (pA/\sqrt{Hz})$",fontsize=self.fs)

    def plotting(self,label):
        self.ax1.set_xscale("log")
        self.ax1.set_yscale("log")         
        self.ax1.set_ylim(1,1e+3)
        self.ax1.set_xlim(np.min(self.frq[self.frq>0]),np.max(self.frq[self.frq>0]))
        self.ax1.step(self.frq,self.nspec*1e+12,where="mid",label=label)

    def out(self):
        self.plot_init()
        files = ['/Volumes/SUNDISK_SSD/noise/Normal_13A_ON_5mA/Normal_13A_ON_5mA.hdf5','/Volumes/SUNDISK_SSD/noise/Normal_13A_ON/Normal_13A_ON.hdf5', '/Volumes/SUNDISK_SSD/noise/Normal_13A_OFF/Normal_13A_OFF.hdf5']
        labels = ['1-3A ON, 5mA', '1-3A ON', '1-3A OFF']
        for e,filename in enumerate(files):
            self.loadfile(filename)
            self.noise_spectrum()
            self.plotting(label=labels[e])
        plt.legend()
        plt.show()