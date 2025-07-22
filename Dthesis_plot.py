

import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np

class DPlot:
    def __init__(self):
        params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'text.usetex': False,
            'font.family': 'serif'
            }

        plt.rcParams.update(params)

    def plot_style(self,style='double'):
        self.fig = plt.figure(figsize=(8,6))
        if style == 'single':
            self.ax  = self.fig.add_subplot(111)
        spine_width = 2  # スパインの太さ
        if style == 'double':
            self.ax  = self.fig.add_subplot(211)
            self.ax2  = self.fig.add_subplot(212,sharex=self.ax)
            for spine in self.ax2.spines.values():
                spine.set_linewidth(spine_width)
            self.ax2.tick_params(axis='both',direction='in',width=1.5)
        if style == 'triple':
            self.ax  = self.fig.add_subplot(311)
            self.ax2  = self.fig.add_subplot(312,sharex=self.ax)
            self.ax3  = self.fig.add_subplot(313,sharex=self.ax)
            for spine in self.ax2.spines.values():
                spine.set_linewidth(spine_width)
            self.ax2.tick_params(axis='both',direction='in',width=1.5)
            for spine in self.ax3.spines.values():
                spine.set_linewidth(spine_width)
            self.ax3.tick_params(axis='both',direction='in',width=1.5)
        for spine in self.ax.spines.values():
            spine.set_linewidth(spine_width)
        self.ax.tick_params(axis='both',direction='in',width=1.5)
        self.fig.align_labels()


    def led_offset(self):
        # FITSファイルの読み込み
        file = '/opt/CALDB/data/xrism/resolve/bcf/time/xa_rsl_mxsparam_20190101v003.fits'
        f = fits.open(file)
        data = f[1].data
        ILED = data['ILED']
        LED1_start_delt = data['LED1STARTDELT']
        LED1_stop_delt = data['LED1STOPDELT']
        MAX_Current = 1.0327
        BRIGHT_Current = 0.4245
        # プロットの作成
        self.plot_style('single')
        self.ax.plot(ILED[1:], LED1_start_delt[1:], color='black')
        self.ax.set_xlabel('LED Current (mA)')
        self.ax.set_ylabel('Pulse start offset (ms)')
        f = interp1d(ILED[1:], LED1_start_delt[1:])
        self.ax.scatter(MAX_Current, f(MAX_Current), color='red', label=f'MAX MODE {MAX_Current:.2f} mA')
        self.ax.scatter(BRIGHT_Current, f(BRIGHT_Current), color='blue', label=f'BRIGHT MODE {BRIGHT_Current:.2f} mA')
        self.ax.text(MAX_Current+0.5, f(MAX_Current), f'{f(MAX_Current):.2f} ms', fontsize=15, color='red', 
                verticalalignment='bottom', horizontalalignment='center')
        self.ax.text(BRIGHT_Current+0.5, f(BRIGHT_Current), f'{f(BRIGHT_Current):.2f} ms', fontsize=15, color='blue', 
                verticalalignment='bottom', horizontalalignment='center')
        self.ax.set_xlim(-0.01,5.1)
        self.ax.set_ylim(0, 1.1)
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.savefig('LED_Current_Start_Offset.pdf', dpi=300, transparent=True)
        plt.show()

        self.fig.clear()
        self.plot_style('single')
        self.ax.plot(ILED[1:], LED1_stop_delt[1:], color='black')
        #plt.scatter(ILED[1:], LED1_start_delt[1:], color='black')
        self.ax.set_xlabel('LED Current (mA)')
        self.ax.set_ylabel('Pulse stop offset (ms)')
        f = interp1d(ILED[1:], LED1_stop_delt[1:])
        self.ax.scatter(MAX_Current, f(MAX_Current), color='red', label=f'MAX MODE {MAX_Current:.2f} mA')
        self.ax.scatter(BRIGHT_Current, f(BRIGHT_Current), color='blue', label=f'BRIGHT MODE {BRIGHT_Current:.2f} mA')
        self.ax.text(MAX_Current+0.5, f(MAX_Current), f'{f(MAX_Current):.3f} ms', fontsize=15, color='red', 
                verticalalignment='bottom', horizontalalignment='center')
        self.ax.text(BRIGHT_Current+0.5, f(BRIGHT_Current)+0.003, f'{f(BRIGHT_Current):.3f} ms', fontsize=15, color='blue', 
                verticalalignment='bottom', horizontalalignment='center')
        self.ax.set_xlim(-0.01,5.1)
        self.ax.set_ylim(0.023, 0.059)
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.savefig('LED_Current_Stop_Offset.pdf', dpi=300, transparent=True)
        offset = 15.625e-3 #MXSSCDLT
        SCDLTNOM = 0.004e-3
        SCDLTRED = 0.004e-3
        MXSMARGN = 0.060e-3

    def led_afterglow(self):
        file = '/opt/CALDB/data/xrism/resolve/bcf/time/xa_rsl_mxsparam_20190101v003.fits'
        f = fits.open(file)
        data = f[2].data
        PLS_LEN = data['EFFPLSLEN']
        LED1_AGDELT = data['LED1AGDELT']

        self.plot_style('single')
        self.ax.plot(PLS_LEN, LED1_AGDELT, color='black')
        self.ax.set_xlabel('Effective pulse length (ms)')
        self.ax.set_ylabel('LED1 Afterglow duration (ms)')

        # # スプライン補間の作成
        f = interp1d(PLS_LEN, LED1_AGDELT)
        max_teff_len = 0.332 #[ms]
        bright_teff_len = 0.207 #[ms]
        self.ax.scatter(max_teff_len, f(max_teff_len), color='red', label=f'MAX MODE {max_teff_len:.2f} ms')
        self.ax.scatter(bright_teff_len, f(bright_teff_len), color='blue', label=f'BRIGHT MODE {bright_teff_len:.2f} ms')

        # テキストをプロットに追加（線から少し離す）
        self.ax.text(max_teff_len, f(max_teff_len) + 15, f'{f(max_teff_len):.2f} ms', fontsize=15, color='red', 
                verticalalignment='bottom', horizontalalignment='center')
        self.ax.text(bright_teff_len, f(bright_teff_len) + 5, f'{f(bright_teff_len):.2f} ms', fontsize=15, color='blue', 
                verticalalignment='bottom', horizontalalignment='center')
        
        # 軸を対数スケールに設定
        self.ax.set_title('Afterglow threshold = 1e-4')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.legend()
        self.fig.savefig('LED_Afterglow.pdf', dpi=300, transparent=True)
        plt.show()

    def mxs_phase_threshold_max(self):
        import numpy as np
        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        def XRISMtime(t):
            return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

        directory = '/Volumes/SUNDISK_SSD/PKS_XRISM/from_sawadasan/pks0745'
        evt_cl = f'{directory}/xa000112000rsl_p0px5000_cl_brt_mxsphase.evt.gz'
        time_threshold = 1.5330e8
        time_threshold_min = 1.53275e8

        # mxs_phase_threshold = 12.17e-3
        exposure = 4.270113990098238E+04
        plen = 250e-3
        cnt = []
        pt = []
        col_list = ['green','black','blue','red','orange']
        with fits.open(evt_cl) as f:
            for mxs_phase_threshold in np.logspace(-8,np.log10(150e-3),100):
                ITYPE = f['EVENTS'].data['ITYPE']
                time = f['EVENTS'].data['TIME']
                MXS_phase = f['EVENTS'].data['MXS_PHASE']
                mask = (MXS_phase > mxs_phase_threshold)
                time = time[mask]
                all_counts = len(time)
                real_exposure = exposure * (plen-mxs_phase_threshold)/plen
                cnt.append(all_counts/real_exposure)
                pt.append(mxs_phase_threshold)



        params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'text.usetex': False,
            'font.family': 'serif'
            }

        plt.rcParams.update(params)

        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)

        print(pt)
        pt = np.array(pt)
        ax.step(pt*1e3,cnt, color='black')
        ax.set_xscale('log')
        ax.set_xlabel('MXS phase threshold (ms)')
        ax.set_ylabel('Count rate (count/s)')
        fig.savefig('/Users/keitatanaka/Dropbox/figure_for_Dthesis/mxs_phase_threshold_brt.pdf',dpi=300)
        #plt.xlim(1e-1,1e1)

        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)

        print(pt)
        pt = np.array(pt)
        ax.scatter(pt*1e3,cnt, color='black')
        ax.plot(pt*1e3,cnt, color='black')
        ax.set_xscale('log')
        ax.set_xlabel('MXS phase threshold (ms)')
        ax.set_ylabel('Count rate (count/s)')
        ax.set_ylim(14.0, 14.1)
        ax.axvline(20, color='red', linestyle='dashed', linewidth=1)
        fig.savefig('/Users/keitatanaka/Dropbox/figure_for_Dthesis/mxs_phase_threshold_brt_zoom.pdf',dpi=300)

        # 対数ビンを作成
        bins = np.logspace(np.log10(0.3*1e-3), np.log10(max(MXS_phase)), 300)

        # ヒストグラムを計算（確率密度ではなくカウント）
        counts, bin_edges = np.histogram(MXS_phase, bins=bins)

        # 各ビンの幅を計算
        bin_widths = np.diff(bin_edges)

        # 縦軸の値をビン幅で割る
        normalized_counts = counts / bin_widths

        # グラフを描画
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.step(bin_edges[:-1]*1e+3, normalized_counts, color='black', label=f'MXS phase < {mxs_phase_threshold*1e3:.2f} ms')

        # 対数スケール
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('MXS phase (ms)')
        ax.set_ylabel('Count rate (count/s)')
        ax.axvspan(0.3, 0.59, color='gray', alpha=0.5)
        ax.axvspan(0.59, 0.59+0.207, color='blue', alpha=0.5)
        ax.axvspan(0.59+0.207, 0.59+0.207+0.047, color='gray', alpha=0.5)
        ax.set_title('BRIGHT MODE')
        plt.show()

    def Fe_filter_info(self):
        import numpy as np
        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from xrism_tools import Local
        local = Local()

        def XRISMtime(t):
            return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

        adr2_time = XRISMtime(Time(local.dates['ADR2']))
        t_start, t_end = '2023-11-08 10:21:04',  '2023-11-11 12:01:04'
        t0, t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
        tlist = np.linspace(t0, t1)
        print('--------------------------')
        print(t0)
        fig, ax = plt.subplots(3,  figsize=(12, 8))
        ax[-1].set_xlim(t0, t1)
        #ax[1].set_xlabel('Time (ks) from ' + t_start)
        ax[1].set_xticks(np.arange(t0, t1, 50000))
        ax[1].set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))
        ax[2].set_xlabel('Time (ks) from ' + t_start)
        ax[2].set_xticks(np.arange(t0, t1, 50000))
        ax[2].set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))
        directory = '/Volumes/SUNDISK_SSD/PKS_XRISM/from_sawadasan/pks0745'
        evt_cl = f'{directory}/xa000112000rsl_p0px5000_cl_max_mxsphase.evt.gz'
        evt_cl2 = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/MAX_MODE/xa000112000rsl_p0px5000_cl_trial_th_0.0001_dtag_0.evt'
        evt_cl3 = '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/xa000112000rsl_p0px5000_cl.evt.gz'
        time_threshold = 1.5330e8 + 30e+3 -12.5e3
        time_threshold_min = 1.53275e8 - 1.9e3
        
        with fits.open(evt_cl3) as f:

            ITYPE = f['EVENTS'].data['ITYPE']
            time = f['EVENTS'].data['TIME'][ITYPE==0]
            # for only Fe data
            time_mask_fe = (time < time_threshold_min)
            time_mask_max = (time < time_threshold) & (time > time_threshold_min)
            time_mask_bright = (time > adr2_time)
            PHA_max = f['EVENTS'].data['PI'][ITYPE==0][time_mask_max]
            PHA_max = PHA_max/2000 
            PHA_fe = f['EVENTS'].data['PI'][ITYPE==0][time_mask_fe]
            PHA_fe = PHA_fe/2000
            PHA_bright = f['EVENTS'].data['PI'][ITYPE==0][time_mask_bright]
            PHA_bright = PHA_bright/2000
            bin = np.arange(0, 3e4/2000, 10/2000)
            events, BIN_max = np.histogram(PHA_max, bin)
            events, BIN_bright = np.histogram(PHA_bright, bin)
            events, BIN_fe = np.histogram(PHA_fe, bin)
            ax[0].hist(PHA_max,bins=BIN_max,histtype='step',color='red',label='Max mode',density=True)
            ax[0].hist(PHA_bright,bins=BIN_bright,histtype='step',color='blue',label='Bright mode',density=True)
            ax[0].hist(PHA_fe,bins=BIN_fe,histtype='step',color='black',label='Only Fe Source',density=True)
            ax[0].set_yscale('log')


            bin = np.arange(np.min(time),np.max(time))[::100]
            events, BIN = np.histogram(time, bin)
            events = np.where(events == 0, np.nan, events)/100
            ax[1].scatter(bin[:-1], events, s=1, c='black')
            ax[1].set_xlim(np.min(time),np.max(time))
            #ax[1].axvline(time_threshold, color='red', linestyle='dashed', linewidth=1, alpha=0.5)
            #ax[1].axvline(time_threshold_min, color='red', linestyle='dashed', linewidth=1, alpha=0.5)
            #ax[1].set_ylim(8, 15)
        #ax[1].axvline(time_threshold, color='red', linestyle='dashed', linewidth=1, alpha=0.5)
        #ax[1].axvline(time_threshold_min, color='red', linestyle='dashed', linewidth=1, alpha=0.5)
        datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs'
        datadir = '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk'
        ehk =  f'{datadir}/xa000112000.ehk'
        hk1 =  f'{datadir}/xa000112000rsl_a0.hk1'

        with fits.open(hk1) as f:

            HK_SXS_FWE = f['HK_SXS_FWE'].data
            MXS_TIME    = HK_SXS_FWE['TIME']
            LED_ON    = HK_SXS_FWE['TIME_LED1_ON']
            LED_ON_TI = HK_SXS_FWE['FWE_TI_LED1_ON']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_SET_CAL']
            MXS_1_V     = HK_SXS_FWE['FWE_V_LED1_CAL']

            ax[2].plot(MXS_TIME,MXS_1_I,c='k',label='MXS1')
            ax[2].set_xlim(np.min(time),np.max(time))

        ax[0].set_xlim(2,10)
        ax[1].set_ylabel('Count rate\n (count/s)')
        ax[2].set_ylabel('LED Current\n (mA)')
        ax[0].set_ylabel('Normalized Counts')
        ax[0].set_xlabel('Energy (keV)')
        ax[0].legend()
        off_x = 0.3
        off_y = 100
        ax[0].text(5.5-off_x, off_y, 'CrKa', fontsize=10, fontweight='bold')
        ax[0].text(5.98-off_x, off_y, 'MnKa', fontsize=10, fontweight='bold')
        ax[0].text(6.65-off_x, off_y, 'MnKb', fontsize=10, fontweight='bold')
        ax[0].text(8.15-off_x, off_y, 'CuKa', fontsize=10, fontweight='bold')
        ax[0].text(8.98-off_x, off_y, 'CuKb', fontsize=10, fontweight='bold')
        ax[0].set_ylim(0, 500)

        ax[1].axvline(adr2_time, color='red', linestyle='--', label='ADR2')
        ax[2].axvline(adr2_time, color='red', linestyle='--', label='ADR2')
        ax[1].axvspan(0, time_threshold_min, color='gray', alpha=0.3)
        ax[1].axvspan(time_threshold_min, time_threshold, color='red', alpha=0.3)
        ax[1].axvspan(adr2_time, 10e8, color='blue', alpha=0.3)
        fig.tight_layout()
        plt.show()
        fig.savefig('55Fe_info.pdf',dpi=300)


    def ghf_diff(self):
        import matplotlib.pyplot as plt
        from astropy.io import fits
        import numpy as np
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/mxs_shift_analysis/repro_mxs_10ms/xa000112000rsl_000_fe55.ghf')[1].data
        time_nom = f['TIME']
        shift_nom = f['SHIFT']
        pixel_nom = f['PIXEL']
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/mxs_shift_analysis/xa000112000rsl_000_fe55.ghf.gz')[1].data
        start = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/mxs_shift_analysis/xa000112000rsl_000_fe55.ghf.gz')[1].header['TSTART']
        time_10m = f['TIME']
        shift_10m = f['SHIFT']
        pixel_10m = f['PIXEL']
        from scipy.interpolate import interp1d
        method = 'linear'
        max_shifts = []
        bright_shifts = []
        pix_list = [23]
        for pix in pix_list:
            nom_pix_mask = pixel_nom == pix
            nom_interp_shift = interp1d(time_nom[nom_pix_mask], shift_nom[nom_pix_mask], fill_value='extrapolate', kind=method)
            th_pix_mask = pixel_10m == pix
            th_interp_shift = interp1d(time_10m[th_pix_mask], shift_10m[th_pix_mask], fill_value='extrapolate', kind=method)
            dif_x = np.linspace(np.min(time_nom[nom_pix_mask]), np.max(time_nom[nom_pix_mask]), 100)
            dif_y = nom_interp_shift(dif_x) - th_interp_shift(dif_x)
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot((time_nom[nom_pix_mask]-start)*1e-3, shift_nom[nom_pix_mask], color='black')
            ax[0].scatter((time_nom[nom_pix_mask]-start)*1e-3, shift_nom[nom_pix_mask], s=10, color='black', label='Nominal')
            ax[0].plot((time_10m[th_pix_mask]-start)*1e-3, shift_10m[th_pix_mask], color='red')
            ax[0].scatter((time_10m[th_pix_mask]-start)*1e-3, shift_10m[th_pix_mask], s=10, color='red', label='MXS phase < 10ms')
            ax[1].plot((dif_x-start)*1e-3, dif_y, color='black')
            max_shift    = np.nanmean(dif_y[dif_x<1.5332e8])
            bright_shift = np.nanmean(dif_y[dif_x>1.5334e8])
            max_shifts.append(max_shift)
            bright_shifts.append(bright_shift)
            ax[0].legend()
            ax[0].set_title(f'Pixel {pix}')
            ax[0].set_ylabel('Shift (eV)')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Shift difference (eV)')

        # hk = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1')['HK_SXS_FWE'].data
        # time = hk['TIME']
        # set_i = hk['FWE_I_LED1_SET_CAL']
        # ax[0].plot((time-start)*1e-3, set_i, color='blue')
        pix_list = pix_list[0]
        fig.tight_layout()
        fig.savefig(f'ghf_diff_{str(pix_list)}.pdf', dpi=300)

        plt.show()

    def ghf_diff2(self):
        from scipy.interpolate import interp1d
        from astropy.io import fits
        import numpy as np
        import matplotlib.pyplot as plt
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/mxs_shift_analysis/repro_mxs_10ms/xa000112000rsl_000_fe55.ghf')[1].data
        time_nom = f['TIME']
        shift_nom = f['SHIFT']
        pixel_nom = f['PIXEL']
        plt.scatter(time_nom, shift_nom, s=0.5)
        f = fits.open('xa000112000rsl_000_fe55.ghf.gz')[1].data
        time_10m = f['TIME']
        shift_10m = f['SHIFT']
        pixel_10m = f['PIXEL']
        method = 'linear'
        max_shifts = []
        bright_shifts = []
        pix_list = range(0,36)
        for pix in pix_list:
            nom_pix_mask = pixel_nom == pix
            nom_interp_shift = interp1d(time_nom[nom_pix_mask], shift_nom[nom_pix_mask], fill_value='extrapolate', kind=method)
            th_pix_mask = pixel_10m == pix
            th_interp_shift = interp1d(time_10m[th_pix_mask], shift_10m[th_pix_mask], fill_value='extrapolate', kind=method)
            dif_x = np.linspace(np.min(time_nom[nom_pix_mask]), np.max(time_nom[nom_pix_mask]), 100)
            dif_y = nom_interp_shift(dif_x) - th_interp_shift(dif_x)
            # ax[0].plot(time_nom[nom_pix_mask], shift_nom[nom_pix_mask], color='black')
            # ax[0].scatter(time_nom[nom_pix_mask], shift_nom[nom_pix_mask], s=10, color='black', label='Nominal')
            # ax[0].plot(time_10m[th_pix_mask], shift_10m[th_pix_mask], color='red')
            # ax[0].scatter(time_10m[th_pix_mask], shift_10m[th_pix_mask], s=10, color='red', label='thresh=10ms')
            # ax[1].plot(dif_x, dif_y, color='black')
            max_shift = np.nanmean(dif_y[dif_x<1.5332e8])
            bright_shift = np.nanmean(dif_y[dif_x>1.5334e8])
            max_shifts.append(max_shift)
            bright_shifts.append(bright_shift)

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        max_shifts = np.array(max_shifts)
        bright_shifts = np.array(bright_shifts)
        ax[0].scatter(pix_list, max_shifts, label='Max', color='black')
        ax[0].axhline(np.mean(max_shift), color='black', linestyle='--')
        ax[0].scatter(pix_list, bright_shifts, label='Bright', color='red')
        ax[0].axhline(np.mean(bright_shift), color='red', linestyle='--')
        ax[0].set_xlabel('Pixel number')
        ax[0].set_ylabel('Shift (eV)')
        ax[0].legend()
        y1 = np.array([23,21,19,11])
        y2 = np.array([24, 22, 20, 10, 13, 14])
        y3 = np.array([25, 26, 18, 17, 15, 16])
        y4 = np.array([0, 7, 8, 33, 34, 35])
        y5 = np.array([2, 4, 6, 32, 31, 28])
        y6 = np.array([30, 29, 1, 3, 5])

        off = 0.2
        for i in y1:
            ax[1].scatter(1, max_shifts[i], color='red')
            ax[1].scatter(1+off, bright_shifts[i], color='red')
        for i in y2:
            ax[1].scatter(2, max_shifts[i], color='blue')
            ax[1].scatter(2+off, bright_shifts[i], color='blue')
        for i in y3:
            ax[1].scatter(3, max_shifts[i], color='green')
            ax[1].scatter(3+off, bright_shifts[i], color='green')
        for i in y4:
            ax[1].scatter(4, max_shifts[i], color='orange')
            ax[1].scatter(4+off, bright_shifts[i], color='orange')
        for i in y5:
            ax[1].scatter(5, max_shifts[i], color='purple')
            ax[1].scatter(5+off, bright_shifts[i], color='purple')
        for i in y6:
            ax[1].scatter(6, max_shifts[i], color='brown')
            ax[1].scatter(6+off, bright_shifts[i], color='brown')
        ax[1].set_ylim(-0.2, 0.2)
        plt.show()

    def ghf_diff_scott_fe_mxs(self):
        import matplotlib.pyplot as plt
        from astropy.io import fits
        import numpy as np
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/ghf_cor/xa000112000rsl_000_cra_scott.ghf')[1].data
        time_scott = f['TIME']
        #shift_scott = f['SHIFT']
        temp_scott = f['TEMP_FIT']
        pixel_scott = f['PIXEL']
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/ghf_cor/xa000112000rsl_000_cra.ghf')[1].data
        fe_start = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/ghf_cor/xa000112000rsl_000_fe55.ghf.gz')[1].header['TSTART']
        start = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/ghf_cor/xa000112000rsl_000_cra.ghf')[1].header['TSTART']
        time_mxs = f['TIME']
        #shift_mxs = f['SHIFT']
        temp_mxs = f['TEMP_FIT']
        pixel_mxs = f['PIXEL']
        f = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/ghf_cor/xa000112000rsl_000_fe55.ghf.gz')[1].data
        time_fe = f['TIME']
        #shift_fe = f['SHIFT']
        temp_fe = f['TEMP_FIT']
        pixel_fe = f['PIXEL']
        time_mask_mxs = time_mxs > fe_start 
        time_mask_scott = time_scott < fe_start
        time_merge = np.append(time_scott[time_mask_scott], time_mxs[time_mask_mxs])
        pixel_merge = np.append(pixel_scott[time_mask_scott], pixel_mxs[time_mask_mxs])
        temp_merge  = np.append(temp_scott[time_mask_scott], temp_mxs[time_mask_mxs])

        pix_list = [23]
        for pix in pix_list:
            merge_pix_mask = pixel_merge == pix
            fe_pix_mask = pixel_fe == pix
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot((time_merge[merge_pix_mask]-start)*1e-3, temp_merge[merge_pix_mask]*1e3, color='black')
            ax.scatter((time_merge[merge_pix_mask]-start)*1e-3, temp_merge[merge_pix_mask]*1e3, s=10, color='black', label='MXS')
            ax.plot((time_fe[fe_pix_mask]-start)*1e-3, temp_fe[fe_pix_mask]*1e3, color='blue')
            ax.scatter((time_fe[fe_pix_mask]-start)*1e-3, temp_fe[fe_pix_mask]*1e3, s=10, color='blue', label='55Fe')

            ax.legend()
            ax.set_title(f'Pixel {pix}')
            ax.set_ylabel('Effective Temperature (mK)')
            ax.set_xlabel("Time (ks)")


        # hk = fits.open('/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1')['HK_SXS_FWE'].data
        # time = hk['TIME']
        # set_i = hk['FWE_I_LED1_SET_CAL']
        # ax[0].plot((time-start)*1e-3, set_i, color='blue')
        pix_list = pix_list[0]
        fig.tight_layout()
        fig.savefig(f'ghf_diff_{str(pix_list)}.pdf', dpi=300)

        plt.show()

  
    def mxs_pulse_plot(self, file="xa000112000rsl_p0px5000_cl_brt_mxsphase.evt.gz"):
        file="xa000112000rsl_p0px5000_cl_max_mxsphase.evt.gz"
        f = fits.open(file)[1]
        mxs_phase = f.data["MXS_PHASE2"][:]
        bins_edge = np.logspace(-4,np.log10(100e-3),1000)
        bins_width = bins_edge[1:] - bins_edge[0:-1]
        n, bins = np.histogram(mxs_phase, bins=bins_edge)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.step(bins_edge[1:]*1e3,n/bins_width, color="darkblue", label="Max")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlabel("MXS phase (ms)")
        # ax.set_ylabel("Count rate (cnt/s)")
        # fig.savefig("max_pulse.png",dpi=300)
        # plt.show()

        file="xa000112000rsl_p0px5000_cl_brt_mxsphase.evt.gz"
        f = fits.open(file)[1]
        mxs_phase = f.data["MXS_PHASE2"][:]
        bins_edge = np.logspace(-4,np.log10(300e-3),1000)
        bins_width = bins_edge[1:] - bins_edge[0:-1]
        n, bins = np.histogram(mxs_phase, bins=bins_edge)
        #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.step(bins_edge[1:]*1e3,n/bins_width, color="orange", label="Bright")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MXS phase (ms)")
        ax.set_ylabel("Count rate (cnt/s)")
        ax.set_xlim(0.1,90)
        ax.legend()
        fig.savefig("max_brt_pulse.png",dpi=300)

        plt.show()