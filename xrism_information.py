from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
import matplotlib.patches as patches
from scipy.stats import poisson
import glob
import os
from matplotlib.ticker import FuncFormatter
from xspec import *
import h5py
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.special import gamma
from astropy import units as u
from astropy import constants as c
from astropy.time import Time
import csv
from matplotlib.colors import LinearSegmentedColormap

# viridis の前半分を取り出す
viridis_half = LinearSegmentedColormap.from_list("viridis_half", cm.viridis(np.linspace(0.5, 1.0, 256)))

class XTools:

    def __init__(self) -> None:
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in'
            }

        plt.rcParams.update(self.plot_params)

    def loadfits(self, filename):
        filename = filename
        hdul = fits.open(filename)
        hdul.info()
        self.hdul = hdul
        self.data = np.array(hdul[0].data)

    def plot_style(self,style='single'):
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
            'text.usetex': False,
            'font.family': 'serif'
            }

        plt.rcParams.update(params)
        if style == 'single':
            self.fig = plt.figure(figsize=(8,6))
            self.ax  = self.fig.add_subplot(111)
            spine_width = 2  # スパインの太さ
            for spine in self.ax.spines.values():
                spine.set_linewidth(spine_width)
            self.ax.tick_params(axis='both',direction='in',width=1.5)
            self.ax.grid(linestyle='dashed')
        elif style == 'double':
            self.fig = plt.figure(figsize=(12,6))
            self.ax = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            spine_width = 2
            for spine in self.ax.spines.values():
                spine.set_linewidth(spine_width)
            self.ax.tick_params(axis='both',direction='in',width=1.5)
            for spine in self.ax2.spines.values():
                spine.set_linewidth(spine_width)
            self.ax2.tick_params(axis='both',direction='in',width=1.5)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.fig.align_labels()

    def mjd_add_sec(self,mjd,sec):
        mjd_start = Time("2019-01-01", scale='utc').mjd

        # MJD58484 と追加する秒数
        mjd_target = mjd
        seconds_to_add = sec

        # MJD から Time オブジェクトを作成
        time_target = Time(mjd_target, format='mjd') + seconds_to_add

        # 2019年1月1日からの経過時間（秒）を計算
        elapsed_time_seconds = (time_target.mjd - mjd_start) * 86400  # 1日 = 86400秒

        # 結果を表示
        print(f"2019年1月1日から {time_target.iso} までの経過時間は {elapsed_time_seconds:.0f} 秒です。")

    def process_data(self, binsize):
        print('----------------------')
        print('processing data...')
        self.bin_size = binsize
        self.pixel_size = 18.9/640 * self.bin_size
        self.axis_center = round(477/self.bin_size)
        self.reduction_size = round(253/self.bin_size)
        self.reduction_size2 = round(800/self.bin_size)
        self.data = self.data[self.reduction_size:, self.reduction_size:]
        self.data = self.data[:-self.reduction_size, :-self.reduction_size]
        self.data_img = self.data
        self.data = self.data[:self.reduction_size2,:self.reduction_size2]

    def pixel_distance(self):
        print('----------------------')
        print('calculate pixel distance...')
        x = np.arange(1, self.data.shape[0] + 1)
        i_diff = x - self.axis_center
        i_diff_squared = i_diff ** 2
        i_diff_squared_scaled = i_diff_squared * self.pixel_size ** 2

        # Calculate the distances without using loops
        dist_matrix = np.sqrt(i_diff_squared_scaled[:, np.newaxis] + i_diff_squared_scaled)

        # Replace NaN values with 0
        dist_matrix[np.isnan(dist_matrix)] = 0

        # Flatten the distance matrix to get a 1D array
        self.dist_list = dist_matrix.flatten()
        print(np.sort(self.dist_list)[:10])

    def log_binning(self,x,y,min=-1.5,max=1.0,num=100):
        print('----------------------')
        print('calculate log binning...')
        bin_div = np.logspace(min,max,num=num)
        for i in range(0,len(bin_div)-1):
            if i == 0:
                mask = x < bin_div[i]
                if len(x[mask]) > 1:
                    x_bin = np.nanmean(x[mask])
                    y_bin = np.nanmean(y[mask])
                elif len(x[mask]) == 0:
                    pass
                else:
                    x_bin = x[mask]
                    y_bin = y[mask]

            else:
                mask = (bin_div[i-1] < x) & (x < bin_div[i])

                if len(x[mask]) > 1:
                    x_bin = np.append(x_bin,np.nanmean(x[mask]))
                    y_bin = np.append(y_bin, np.nanmean(y[mask]))
                elif len(x[mask]) == 0:
                    pass
                else:
                    x_bin = np.append(x_bin,x[mask])
                    y_bin = np.append(y_bin, y[mask])
        print("--------------")
        print(x_bin, y_bin)
        return x_bin, y_bin

    def data_reduction(self):
        self.data_1d = self.data.reshape(-1)
        self.dist_list = np.array(self.dist_list).reshape(-1)
        self.data_1d = self.data_1d[~np.isnan(self.data_1d)]
        self.dist_list = self.dist_list[~np.isnan(self.dist_list)]
        self.data_1d   = self.data_1d[np.argsort(self.dist_list)]
        self.dist_list = np.sort(self.dist_list)    

    def img_plot(self, filename):
        self.loadfits(filename)
        exposure = self.hdul[0].header['EXPOSURE']
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(self.data / exposure, origin='lower')
        
        # カラーバーを作成し、キャプションを追加
        cbar = self.fig.colorbar(im)
        #cbar.set_label('counts / sec', fontsize=12)  # キャプションを設定
        self.ax.set_title('BRIGHT MODE count/s', fontsize=15)
        # x 軸と y 軸の ticks を消す
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # ピクセルごとの値を中央に表示
        rows, cols = self.data.shape
        for i in range(rows):
            for j in range(cols):
                # 各ピクセルの位置に指数形式のテキストを追加
                self.ax.text(j, i, f'{self.data[i, j] / exposure:.2e}', ha='center', va='center', color='white', fontsize=8)
        
        # 軸の範囲を設定
        self.ax.set_xlim(-0.5, 5.5)
        self.ax.set_ylim(-0.5, 5.5)
        self.fig.savefig(f'./figure/{filename}.png',dpi=300,transparent=True)
        plt.show()

    def plot_count(self):
        # サンプルデータを作成
        data = self.data_img

        # 100以上の値を持つデータの行列番号を格納するリスト
        data_reduction = np.argwhere(data >= 100)

        # 二次元マップを作成
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='viridis',origin='lower', norm=LogNorm())

        # 100以上の値を持つ場合は赤色で表示
        for i, j in data_reduction:
            plt.plot(j, i, marker='s', color='red')

        plt.colorbar(label='Value')
        plt.title('2D Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

        print("100以上の値を持つデータの行列番号:", data_reduction)

    def count_hist(self):
        cnt = self.data_img.reshape(-1)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.hist(cnt, bins=1000, histtype='step', color='blue')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        plt.show()

    def init_plot(self):
        self.fig = plt.figure(figsize=(10.6, 8))
        self.ax2 = self.fig.add_subplot(111)
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        self.ax2.grid(linestyle='dashed')
        self.ax2.set_title(r'Radial profile',fontsize=15)
        self.ax2.set_ylabel(r"$\rm counts \ arcmin^2\ Normalized\ to\ center\ bin$",fontsize=15)
        self.ax2.set_xlabel(r"$\rm Radius\ from\ (xc, yc)\ [arcmin]$",fontsize=15)        

    def plotting(self):
        self.ax2.scatter(self.dist_bin, self.data_bin/self.data_bin[0]/self.dist_bin*self.dist_bin[0])
        self.ax2.plot(self.dist_bin, self.data_bin/self.data_bin[0]/self.dist_bin*self.dist_bin[0])

    def out(self):
        self.loadfits('SFP_trial_1_-10.img')
        self.process_data(binsize=1)
        self.pixel_distance()
        self.data_reduction()
        self.dist_bin, self.data_bin = self.log_binning(x=self.dist_list,y=self.data_1d)
        self.img_plot()
        self.init_plot()
        self.plotting()

    def out_multi(self):
        file = sorted(list(glob.glob("SFP*.img")))
        self.init_plot()
        for f in file:
            self.loadfits(f)
            self.process_data(binsize=1)
            self.pixel_distance()
            self.data_reduction()
            self.dist_bin, self.data_bin = self.log_binning(x=self.dist_list,y=self.data_1d)

            self.plotting()
        plt.show()

    def count_map(self,filename,binsize):
        self.loadfits(filename)
        self.process_data(binsize=binsize)
        self.pixel_distance()
        self.data_reduction()
        self.dist_bin, self.data_bin = self.log_binning(x=self.dist_list,y=self.data_1d)
        self.img_plot()
        self.count_hist()
        self.plot_count()

    def poisson_plot(self):
        x = np.arange(1,20,1)
        rv = poisson.pmf(x,2.375)
        print(rv)
        plt.plot(x,rv,color='blue')
        plt.scatter(x,rv,color='blue')
        plt.vlines(10,-0.01,0.27,color='black',linestyle='dashed')
        plt.hlines(0.01,0,19,color='red',linestyle='dashed')
        plt.xlabel('counts')
        plt.ylabel('probability')
        plt.grid(linestyle='dashed')
        plt.show()

    def pixel_by_pixel(self,pixel='all'):
        file = glob.glob('*px5000_cl.evt.gz')
        if pixel=='all':
            pixel = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]

    def gain_check(self):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveをプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue', 'black', 'red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        pixel = 0
        for pixel in range(36):
            self.plot_style()
            for idx, temp in enumerate(temperatures):
                H_values = [data[factor][idx, pixel] for factor in factor_name]
                self.ax.plot(x, gain_curve(x, *H_values)*1e-3, label=f"{temp}", color=col[idx])
                
            self.ax.set_title(f"Gain Curve for Pixel {pixel}")
            self.ax.set_xlabel('ADU')
            self.ax.set_ylabel('Energy (keV)')

            self.ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
            self.ax.legend()
            self.ax.grid(linestyle='dashed')
            #plt.show()
            self.fig.savefig(f'./figure/pixel{pixel}_gain_curve.pdf',dpi=300,transparent=True)

    def gain_check_temp_avg(self):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        print(data)
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveをプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue', 'black', 'red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        
        # ADU=2e4に対応するインデックスを取得
        adu_value = 2e4
        adu_index = np.abs(x - adu_value).argmin()

        # ADU=2e4におけるgain_49/gain_50とピクセル番号を保存するリスト
        pixel_gain_ratios = []
        div = 3

        self.plot_style('double')

        for pixel in range(36):
            H_values_49 = [data[factor][0, pixel] for factor in factor_name]
            H_values_50 = [data[factor][1, pixel] for factor in factor_name]
            H_values_51 = [data[factor][2, pixel] for factor in factor_name]
            gain_49     = gain_curve(x, *H_values_49)
            gain_50     = gain_curve(x, *H_values_50)
            gain_51     = gain_curve(x, *H_values_51)
            gain_avg    = (gain_49 + gain_51) / 2
            gain_cor    = np.sqrt(gain_49 * gain_51)
    
            if pixel % div == 0:   
                self.ax.plot(x, gain_avg/gain_50, label=f"pixel {pixel}", color=cm.jet(pixel//div /12))
                self.ax2.plot(x, gain_cor/gain_50, label=f"pixel {pixel}", color=cm.jet(pixel//div /12))

            elif pixel % div == 1:   
                self.ax.plot(x, gain_avg/gain_50,'--' ,label=f"pixel {pixel}", color=cm.jet(pixel//div /12))
                self.ax2.plot(x, gain_cor/gain_50,'--' ,label=f"pixel {pixel}", color=cm.jet(pixel//div /12))

            elif pixel % div == 2:   
                self.ax.plot(x, gain_avg/gain_50,'-.' ,label=f"pixel {pixel}", color=cm.jet(pixel//div /12))
                self.ax2.plot(x, gain_cor/gain_50,'-.' ,label=f"pixel {pixel}", color=cm.jet(pixel//div /12))
        
        self.ax.set_xlabel('ADU')
        self.ax.set_ylabel('((49 mK + 51 mK) / 2) / 50 mK')
        self.ax2.set_ylabel('sqrt(49 mK * 51 mK) / 50 mK')
        self.ax2.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        self.ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        self.ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        self.fig.tight_layout()
        plt.show()

        self.fig.savefig(f'./figure/gain_curve_cor.pdf', dpi=300, transparent=True)
        
    def gain_check_pixel_cor(self,check_temp):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        print(data)
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveをプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue', 'black', 'red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        self.plot_style()
        for pixel in range(36):
            for idx, temp in enumerate(temperatures):
                if temp == f"{check_temp}mK":
                    H_values = [data[factor][idx, pixel] for factor in factor_name]
                    self.ax.plot(x, gain_curve(x, *H_values)*1e-3, label=f"pixel {pixel}", color=cm.jet(pixel/35))
                
        self.ax.set_title(f"Gain Curve ({check_temp} mK)")
        self.ax.set_xlabel('ADU')
        self.ax.set_ylabel('Energy (keV)')
        self.ax.legend(fontsize=10,loc="center left", bbox_to_anchor=(1, 0.5),ncol=3)
        self.ax.grid(linestyle='dashed')
        self.ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
        plt.show()
        self.fig.savefig(f'./figure/{check_temp}mK_gain_curve.pdf',dpi=300,transparent=True)

    def gain_check_temp_cor(self, check_temp):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        print(data)
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveをプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue', 'black', 'red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        
        # ADU=2e4に対応するインデックスを取得
        adu_value = 2e4
        adu_index = np.abs(x - adu_value).argmin()

        # ADU=2e4におけるgain_49/gain_50とピクセル番号を保存するリスト
        pixel_gain_ratios = []

        self.plot_style()

        for pixel in range(36):
            H_values_49 = [data[factor][0, pixel] for factor in factor_name]
            H_values_50 = [data[factor][1, pixel] for factor in factor_name]
            H_values_51 = [data[factor][2, pixel] for factor in factor_name]
            gain_49     = gain_curve(x, *H_values_49)
            gain_50     = gain_curve(x, *H_values_50)
            gain_51     = gain_curve(x, *H_values_51)

            if check_temp == 49:
                self.ax.plot(x, gain_49/gain_50, label=f"pixel {pixel}", color=cm.jet(pixel/35))
            if check_temp == 51:
                self.ax.plot(x, gain_51/gain_50, label=f"pixel {pixel}", color=cm.jet(pixel/35))

            # ADU=2e4におけるgain_49/gain_50の値をリストに保存
            gain_ratio = gain_49[adu_index] / gain_50[adu_index]
            pixel_gain_ratios.append((pixel, gain_ratio))

            self.ax.text(x[1], gain_49[1]/gain_50[1], f"{pixel}", fontsize=5, color="black")
        
        self.ax.set_xlabel('ADU')
        if check_temp == 49:
            self.ax.set_ylabel('Curve (49 mK)/Curve (50 mK)')
        if check_temp == 51:
            self.ax.set_ylabel('Curve (51 mK)/Curve (50 mK)')
        self.ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
        self.ax.grid(linestyle='dashed')
        self.ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        self.fig.tight_layout()
        plt.show()

        if check_temp == 49:
            self.fig.savefig(f'./figure/{check_temp}mK_gain_curve_49_50.pdf', dpi=300, transparent=True)
        if check_temp == 51:
            self.fig.savefig(f'./figure/{check_temp}mK_gain_curve_49_51.pdf', dpi=300, transparent=True)

        # gain_49/gain_50が大きい順にソートして返す
        pixel_gain_ratios.sort(key=lambda x: x[1], reverse=True)
        print(pixel_gain_ratios)
        return pixel_gain_ratios

    def gain_check_inflection_point(self,check_temp=None,pixel=None):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        # gain_curveの定義
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        # gain_curveの2回微分
        def gain_curve_second_derivative(x, H2, H3, H4, H5, H6, H7):
            return 2 * H2 + 6 * H3 * x + 12 * H4 * x**2 + 20 * H5 * x**3 + 30 * H6 * x**4 + 42 * H7 * x**5

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveの2回微分をプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue','black','red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        
        self.plot_style()
        if pixel != None:
            pixel=pixel
            for idx, temp in enumerate(temperatures):      
                    H_values = [data[factor][idx, pixel] for factor in factor_name]
                    
                    # 2回微分のプロット
                    d2y = gain_curve_second_derivative(x, *H_values[2:])
                    self.ax.plot(x, d2y, label=f"Pixel {pixel} ({temp})",color=col[idx])

        else:   
            for pixel in range(36):
            
                for idx, temp in enumerate(temperatures):
                    if temp == f"{check_temp}mK":                
                        H_values = [data[factor][idx, pixel] for factor in factor_name]
                        
                        # 2回微分のプロット
                        d2y = gain_curve_second_derivative(x, *H_values[2:])
                        self.ax.plot(x, d2y, label=f"Pixel {pixel}",color=cm.jet(pixel/35))
            
        self.ax.set_xlabel('ADU')
        self.ax.set_ylabel('Second Derivative of Gain')
        self.ax.legend(ncol=3,fontsize=10)  # 凡例を二段に設定
        self.ax.grid(linestyle='dashed')
        self.ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
        if check_temp == None:
            self.ax.set_title(f"Second Derivative of Gain Curve (pixel {pixel})")
            self.fig.savefig(f'./figure/inflection_point_pixel{pixel}.pdf',dpi=300,transparent=True)
        else:
            self.ax.set_title(f"Second Derivative of Gain Curve ({check_temp}mK)")
            self.fig.savefig(f'./figure/inflection_point_{check_temp}mK.pdf',dpi=300,transparent=True)
        plt.show()

    def gain_check_data(self):
        file = '/opt/CALDB/data/xrism/resolve/bcf/gain/xa_rsl_gainpix_20190101v005.fits'
        self.loadfits(file)
        factor_name = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']  # H1からH6までを確認する
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        
        for pixel in range(36):
            for factor in factor_name:
                for idx in range(3):  # 温度ごとにチェック
                    if data[factor][idx, pixel] == 0:
                        print(f"Pixel {pixel} - {factor} at temperature index {idx} is 0")

    def gain_check_outp(self, pixel_num):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']  # H1からH6までを確認
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        
        # 指定されたピクセルのH1-H6の値をCSVファイルに出力
        csv_filename = f"pixel_{pixel_num}_H1_H6.csv"
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Temperature', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'])  # ヘッダー
            
            temperatures = ['49mK', '50mK', '51mK']
            
            for idx, temp in enumerate(temperatures):
                row = [temp] + [data[factor][idx, pixel_num] for factor in factor_name]
                writer.writerow(row)
        
        print(f"Pixel {pixel_num} のH1-H6を {csv_filename} に出力しました。")

    def plot_shift_fit(self,num):
        from scipy.optimize import fsolve
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz'
        # file = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf'
        self.loadfits(file)
        binmesh = self.hdul[1].data['BINMESH']
        spec = self.hdul[1].data['SPECTRUM']
        fit = self.hdul[1].data['FITPROF']
        shift = self.hdul[1].data['SHIFT']
        cor_fit = self.hdul[1].data['COR_FIT']
        avg_fit = self.hdul[1].data['AVGFIT']
        avg_bin = self.hdul[1].data['AVGBIN']
        Te = self.hdul[1].data['TEMP_AVE'][num]
        num = num
        pixnum = self.hdul[1].data['PIXEL'][num]
        print('pixel num')
        print(pixnum)
        print(cor_fit[num])
        print(avg_fit[num])
        #plt.scatter(avg_fit,self.hdul[1].data['SHIFT'])
        #plt.scatter(avg_fit,self.hdul[1].data['TEMP_AVE'])
        #plt.scatter(self.hdul[1].data['SHIFT'],avg_fit)
        #plt.show()
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        self.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得と確認
        data = {factor: self.hdul[1].data[factor] for factor in factor_name}
        for factor in factor_name:
            print(f"{factor}: {data[factor].shape}")
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        def solve_gain_curve(y_target, H0, H1, H2, H3, H4, H5, H6, H7):
            # 方程式を定義（gain_curve(x) - y_target = 0）
            func = lambda x: gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7) - y_target
            
            # fsolve で初期値を与えて方程式を解く
            x_initial_guess = 1.0  # 初期値は適宜変更
            x_solution = fsolve(func, x_initial_guess)
            return x_solution

        # 各ピクセルでの49mK, 50mK, 51mKのgain curveをプロット
        temperatures = ['49mK', '50mK', '51mK']
        col = ['blue', 'black', 'red']
        x = np.linspace(0, 40000, 10000)  # x軸: エネルギーや他のパラメータの範囲を指定
        H_values = [data[factor][1, pixnum] for factor in factor_name]
        H_values_49 = [data[factor][0, pixnum] for factor in factor_name]
        H_values_51 = [data[factor][2, pixnum] for factor in factor_name]
        MnKa_center = 5894.1673
        PHA_MnKa_50    = solve_gain_curve(MnKa_center,*H_values)
        PHA_MnKa_49    = solve_gain_curve(MnKa_center,*H_values_49)
        PHA_MnKa_51    = solve_gain_curve(MnKa_center,*H_values_51)
        E_bin = gain_curve(binmesh[num],*H_values)
                

        plt.step(E_bin,spec[num],color='black')
        plt.plot(E_bin,fit[num],color='red')
        # plt.step(binmesh[num],spec[num],color='black')
        # plt.plot(binmesh[num],fit[num],color='red')
        print(f'shift(ADU) = {shift[num]}')
        print(f'Te = {Te*1e+3:.2f} mK')
        shift_eV = gain_curve(shift[num],*H_values)
        plt.title(f'Pixel {pixnum}, Te {Te*1e+3:.2f} mK')
        plt.vlines(gain_curve(avg_fit[num],*H_values),0,50,linestyle='dashed',color='red')
        plt.vlines(MnKa_center,0,50,linestyle='dashed',color='black')
        print(f'shift = {gain_curve(avg_fit[num],*H_values) - MnKa_center} eV')
        # plt.vlines(avg_bin[num],0,50,linestyle='dashed',color='red')
        # plt.vlines(avg_fit[num],0,50,linestyle='dashed',color='black')
        plt.xlim(5860,5930)
        #plt.xlim(18250,18500)
        plt.show()
        #plt.scatter(49,gain_curve(PHA_MnKa,*H_values_49),color='blue')
        #plt.scatter(50,gain_curve(PHA_MnKa,*H_values),color='blue')
        #plt.scatter(51,gain_curve(PHA_MnKa,*H_values_51),color='blue')
        plt.scatter([49,50,51],[PHA_MnKa_49,PHA_MnKa_50,PHA_MnKa_51],color='blue')
        plt.plot([50,51],[PHA_MnKa_50,PHA_MnKa_51],color='blue')
        plt.scatter(Te*1e+3,avg_fit[num],color='red')
        plt.show()

    def ehk_check(self,calmethod="55Fe"):
        """
        plot hk data

        """
        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)

        obsid = '000112000'
        t_start, t_end = '2023-11-08 10:21:04',  '2023-11-11 12:01:04'
        if calmethod == "55Fe":
            datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve'
            ehk =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/auxil/xa000112000.ehk.gz'
            hk1 =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1'
            evt_uf = glob.glob(f'{datadir}/event_uf/*_uf.evt.gz')
            evt_cl = glob.glob(f'{datadir}/event_cl/*_cl.evt.gz')
        elif calmethod == "mxs":
            datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs'
            ehk =  f'{datadir}/xa000112000.ehk'
            hk1 =  f'{datadir}/xa000112000rsl_a0.hk1'
            evt_uf = glob.glob(f'{datadir}/*.evt')
            evt_cl = glob.glob(f'{datadir}/*_cl.evt.gz')

        dates = {
            'ADR1': '2023-11-08 20:10:01',
            'ADR2': '2023-11-10 15:10:00'
        }

        def XRISMtime(t):
            return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

        t0, t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
        tlist = np.linspace(t0, t1)
        print('--------------------------')
        print(t0)
        print(evt_cl)

        colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})
        fig, ax = plt.subplots(4, sharex=True, figsize=(6, 8))
        ax[-1].set_xlim(t0, t1)
        ax[-1].set_xlabel('Time (ks) from ' + t_start)
        ax[-1].set_xticks(np.arange(t0, t1, 50000))
        ax[-1].set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))
        FW_mean = []
        FW_deg = [210,150,270,90,330]
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            ax[0].plot(HK_SXS_TEMP['TIME'], HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot(HK_SXS_FWE['TIME'], HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW position \n (deg)')
            for i in range(5):
                FW_mean.append(np.median(HK_SXS_FWE['TIME'][np.round(HK_SXS_FWE['FWE_FW_POSITION1_CAL'],-1)==FW_deg[i]]))
            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_SET_CAL']

            ax[2].plot(MXS_TIME,MXS_1_I,c='k')

            ax[2].set_ylabel('LED1 Current\n (mA)')
        for file in evt_cl:
            with fits.open(file) as f:
                TIME = f['EVENTS'].data['TIME']
                PI = f['EVENTS'].data['PI']/2000 # keV
                PI_filter = (PI > 2) & (PI < 8)
                TIME = TIME[PI_filter]
                # ITYPE = f['EVENTS'].data['ITYPE']
                # for i in range(len(ITYPE)):
                    # if ITYPE[i] == 6:
                    #    ax[3].axvline(TIME[i], lw=0.5, color='b', alpha=0.25)
                bin = np.arange(int(t0), int(t1))[::100]
                events, BIN = np.histogram(TIME, bin)
                events = np.where(events == 0, np.nan, events)/100
                ax[3].scatter(bin[:-1], events, s=1, c='black')
                ax[3].set_ylabel('Count rate\n (count/s)')
                ax[3].set_ylim(0.01, 50)
                ax[3].set_yscale('log')


        self.t0, self.t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
        self.times = {key: XRISMtime(Time(val)) for key, val in dates.items()}

        for date_label, date_value in self.times.items():
            color = 'black'
            if date_label in ['ADR1', 'ADR2']:
                color = 'red'
            
            for axis in ax.ravel():
                axis.axvline(date_value, color=color, linestyle='--', label=date_label)


        FWs = ['Open', 'ND', 'Be', 'OBF', '55Fe']
        fw_y = 70
        y_up = 35
        FW_y = [fw_y-y_up,-fw_y,fw_y-y_up,-fw_y,-fw_y]
        print(FW_mean)
        for i in range(5):
            ax[1].text(FW_mean[i]-1.4e4, FW_deg[i]+FW_y[i], FWs[i], fontsize=15, fontweight='bold', color='black')

        fig.tight_layout()
        plt.show()
        fig.savefig('ehk_check.png',dpi=300)

    def ehk_check_num(self,obsid='000111000'):

        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)

        obsid = obsid
        hk1 =  f'./xa{obsid}rsl_a0.hk1.gz'


        colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})
        fig, ax = plt.subplots(3, sharex=True, figsize=(6, 8))
        FW_mean = []
        FW_deg = [210,150,270,90,330]
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            start_time = HK_SXS_TEMP['TIME'][0]
            ax[0].plot((HK_SXS_TEMP['TIME']-start_time)*1e-3, HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot((HK_SXS_FWE['TIME']-start_time)*1e-3, HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW position')
            for i in range(5):
                FW_mean.append(np.median(HK_SXS_FWE['TIME'][np.round(HK_SXS_FWE['FWE_FW_POSITION1_CAL'],-1)==FW_deg[i]]))
            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_SET_CAL']

            ax[2].plot((MXS_TIME-start_time)*1e-3,MXS_1_I,c='k')

            ax[2].set_ylabel('LED1 Current\n (mA)')
            ax[2].set_xlabel('Time (ks)')


        # FWs = ['Open', 'ND', 'Be', 'Poly', '55Fe']
        # fw_y = 70
        # y_up = 35
        # FW_y = [fw_y-y_up,-fw_y,fw_y-y_up,-fw_y,-fw_y]
        # print(FW_mean)
        # for i in range(5):
        #     ax[1].text(FW_mean[i]-1.4e4, FW_deg[i]+FW_y[i], FWs[i], fontsize=15, fontweight='bold', color='black')
        ax[1].set_yticks([210,150,270,90,330], ['Open', 'ND', 'Be', 'Poly', '55Fe'])
        ax[0].set_title(f'ObsID: {obsid}')
        fig.tight_layout()
        #plt.show()
        fig.savefig(f'./figure/{obsid}_ehk_check.png',dpi=300,transparent=True)

    def pi_plot(self,obsid='000112000',evt_cl='xa000112000_rlsgain.gti'):
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)

        from xrism_tools import Local
        local = Local()
        # ax[0].axvline(local.data_diffs['ADR2']*1e-3, color='red', linestyle='--', label='ADR2')


        file_path1 = './xa000112000rsl_p0px5000_cl_brt_mxsphase_1_risetime_screening.evt'

        col = ['black', 'red']
        lab = ['ghf liner cor', 'ghf no cor']
        tstart = 153138136
        adr2 = local.data_diffs['ADR2']
        thresh = tstart + adr2
            # メインプロットと残差プロットの設定
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        for e,file_path in enumerate([file_path1]):
            # FITSファイルを開く
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                itype = data['ITYPE']
                time = data['TIME']
                phase = data['MXS_PHASE']
                pi = np.array(data['PI'])/2000
                time_filter = (time > thresh) & (itype == 0) & (phase > 20e-3) & (pi > 2) & (pi < 8)
                time_filter2 = (time < thresh) & (itype == 0) & (phase > 20e-3) & (pi > 2) & (pi < 8)
                time_filter3 = (time > thresh) & (itype == 0) & (phase < 20e-3) & (pi > 2) & (pi < 8)

            axes.hist(pi[time_filter], bins=30000, histtype='step', color='black', label=lab[e], density=True)
            # axes.hist(pi[time_filter2], bins=30000, histtype='step', color='red', label=lab[e], density=True)
            axes.hist(pi[time_filter3], bins=30000, histtype='step', color='blue', label=lab[e], density=True)
            axes.set_xlim(2,8)
            axes.set_yscale('log')
            axes.set_xlabel('Energy (keV)')
            axes.set_ylabel('Normalized counts')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./figure/{obsid}_ehk_check.png',dpi=300,transparent=True)

    def ehk_check_num_gti(self,obsid='000112000',evt_cl='xa000112000_rlsgain.gti'):

        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)

        obsid = obsid
        hk1 =  f'/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa{obsid}rsl_a0.hk1'


        colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})
        fig, ax = plt.subplots(3, sharex=True, figsize=(6, 8))
        FW_mean = []
        FW_deg = [210,150,270,90,330]
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            start_time = HK_SXS_TEMP['TIME'][0]
            ax[0].plot((HK_SXS_TEMP['TIME']-start_time)*1e-3, HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot((HK_SXS_FWE['TIME']-start_time)*1e-3, HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW position')
            for i in range(5):
                FW_mean.append(np.median(HK_SXS_FWE['TIME'][np.round(HK_SXS_FWE['FWE_FW_POSITION1_CAL'],-1)==FW_deg[i]]))
            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_SET_CAL']

            ax[2].plot((MXS_TIME-start_time)*1e-3,MXS_1_I,c='k')

            ax[2].set_ylabel('LED1 Current\n (mA)')
            ax[2].set_xlabel('Time (ks)')


        # FWs = ['Open', 'ND', 'Be', 'Poly', '55Fe']
        # fw_y = 70
        # y_up = 35
        # FW_y = [fw_y-y_up,-fw_y,fw_y-y_up,-fw_y,-fw_y]
        # print(FW_mean)
        # for i in range(5):
        #     ax[1].text(FW_mean[i]-1.4e4, FW_deg[i]+FW_y[i], FWs[i], fontsize=15, fontweight='bold', color='black')
        ax[1].set_yticks([210,150,270,90,330], ['Open', 'ND', 'Be', 'Poly', '55Fe'])
        ax[0].set_title(f'ObsID: {obsid}')

        with fits.open(evt_cl) as f:
                GTI = f['GTI'].data
                for a in ax:
                    for i in range(len(GTI) - 1):
                        a.axvspan((GTI[i][1]-start_time)*1e-3, (GTI[i+1][0]-start_time)*1e-3, color='gray', alpha=0.3)

        from xrism_tools import Local
        local = Local()
        ax[0].axvline(local.data_diffs['ADR2']*1e-3, color='red', linestyle='--', label='ADR2')
        ax[1].axvline(local.data_diffs['ADR2']*1e-3, color='red', linestyle='--', label='ADR2')
        ax[2].axvline(local.data_diffs['ADR2']*1e-3, color='red', linestyle='--', label='ADR2')

        fig.tight_layout()
        plt.show()
        fig.savefig(f'./figure/{obsid}_ehk_check.png',dpi=300,transparent=True)

    def ehk_check_num_all(self,obsid='000103000'):
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)

        obsid = obsid
        hk1 =  f'./xa{obsid}rsl_a0.hk1'
        ehk = f'./xa{obsid}.ehk.gz'
        evt_cl = f'occulation_nxb_px5000_cl.evt'
        evt_cl = f'xa000103000_rlsgain.gti'

        colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})
        fig, ax = plt.subplots(5, sharex=True, figsize=(6, 8))
        FW_mean = []
        FW_deg = [210,150,270,90,330]
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            start_time = HK_SXS_TEMP['TIME'][0]
            ax[0].plot((HK_SXS_TEMP['TIME']-start_time)*1e-3, HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot((HK_SXS_FWE['TIME']-start_time)*1e-3, HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW position')
            for i in range(5):
                FW_mean.append(np.median(HK_SXS_FWE['TIME'][np.round(HK_SXS_FWE['FWE_FW_POSITION1_CAL'],-1)==FW_deg[i]]))
            # MXS_TIME    = HK_SXS_FWE['TIME']
            # MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_SET_CAL']

            # ax[2].plot((MXS_TIME-start_time)*1e-3,MXS_1_I,c='k')

            # ax[2].set_ylabel('LED1 Current\n (mA)')
            # ax[2].set_xlabel('Time (ks)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[4].plot((HK_SXS_FWE['TIME']-start_time)*1e-3, HK_SXS_FWE['FWE_HV1_LEVEL'], c='k')
            ax[4].set_ylabel('HV1 Level')

        with fits.open(ehk) as f:
            EHK = f['EHK'].data
            ax[2].plot((EHK['TIME']-start_time)*1e-3, EHK['ELV'], c='k')
            ax[2].axhline(-5, color='r', lw=1)
            ax[2].set_ylim(-60, 105)
            ax[2].set_ylabel('Elevation (deg)')
            ax[3].plot((EHK['TIME']-start_time)*1e-3, EHK['SAA'], c='k')
            ax[3].axhline(0, color='r', lw=1)
            ax[3].set_ylabel('SAA')


        # with fits.open(evt_cl) as f:
        #         GTI = f['STDGTI'].data
        #         for a in ax:
        #             for i in range(len(GTI) - 1):
        #                 a.axvspan((GTI[i][1]-start_time)*1e-3, (GTI[i+1][0]-start_time)*1e-3, color='gray', alpha=0.3)
        # with fits.open('occulation_nxb_px5000_cl.evt') as f:
        #         GTI = f['GTI'].data
        #         for a in ax:
        #             for i in range(len(GTI) - 1):
        #                 a.axvspan((GTI[i][1]-start_time)*1e-3, (GTI[i+1][0]-start_time)*1e-3, color='gray', alpha=0.3)
                        # if GTI[0][0] > t0:
                        #     a.axvspan(t0, GTI[0][0], color='r', alpha=0.25)
                        # if GTI[-1][1] < t1:
                        #     a.axvspan(GTI[-1][1], t1, color='r', alpha=0.25)

        ax[1].set_yticks([210,150,270,90,330], ['Open', 'ND', 'Be', 'Poly', '55Fe'])
        ax[0].set_title(f'ObsID: {obsid}')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./figure/{obsid}_ehk_check_nont.png',dpi=300,transparent=False)

    def multi_ehk_check(self):
        import re
        filenames = glob.glob('./*hk1.gz')
        for filename in filenames:
            obsid = re.search(r'xa(\d{9})', filename).group(1)
            self.ehk_check_num(obsid)

    def xtend_EtoPHA(self,E:float):
        """
        Convert energy to Xtend PHA
        E: energy in keV
        return: PHA
        """
        a = 1584/9.5
        b = 83-0.5*a
        print('-----------------')
        print(f'Energy = {E} keV')
        print(f'PHA = {a*E+b}')
        return E*a+b

    def branching_ratio(self,file='xa000112000rsl_p0px1000_cl_2_exclude_153171242_153178242_PIX_17.evt'):
        f = fits.open(file)
        evt = f[1]
        evt_all = len(evt.data['ITYPE'])
        Hp, Mp, Ms, Lp, Ls = self.ideal_branching_ratio(count_rate=0.0456)
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yscale('log')
        # self.ax.grid(linestyle='dashed')
        self.ax.set_title(r'Branching Ratio')
        self.ax.set_ylabel("Ratio") 
        n, bins, patches = self.ax.hist(evt.data['ITYPE'],bins=[-0.5,0.5,1.5,2.5,3.5,4.5], density=True, align='mid', edgecolor='black', label='data')
        #print(evt.data['ITYPE'])
        # 各ビンの上にテキストを表示
        for i in range(len(bins)-1):
            bin_center = (bins[i] + bins[i+1]) / 2  # ビンの中央のx位置
            height = n[i]  # ビンの高さ
            self.ax.text(bin_center+0.5, height, f'{height*100:.2f}%', ha='right', va='bottom')
        self.ax.set_xticks(ticks=[0,1,2,3,4],labels=['Hp', 'Mp', 'Ms', 'Lp', 'Ls'])
        self.ax.plot([0,1,2,3,4],[Hp, Mp, Ms, Lp, Ls],color='black', label='theory')
        self.ax.scatter([0,1,2,3,4],[Hp, Mp, Ms, Lp, Ls],color='black')
        print(Hp, Mp, Ms, Lp, Ls)
        self.ax.set_xlabel('Grade')
        self.ax.set_ylim(1e-4,1.9)
        self.ax.legend()
        plt.show()
        self.fig.savefig('branching_ratio.png',dpi=300)

    def ideal_branching_ratio(self,count_rate):
        nu   = count_rate
        s    = 80e-6
        t_HR = (1024 - 150)*s
        t_MR = (256 - 37)*s
        Hp = np.exp(-2*nu*t_HR)
        Mp = np.exp(-nu*t_HR)*(np.exp(-nu*t_MR)-np.exp(-nu*t_HR))
        Ms = np.exp(-nu*t_MR)*(np.exp(-nu*t_MR)-np.exp(-nu*t_HR))
        Lp = np.exp(-nu*t_HR)*(1-np.exp(-nu*t_MR))
        Ls = (1-np.exp(-nu*t_MR))*(1+np.exp(-nu*t_MR)-np.exp(-nu*t_HR))
        return Hp, Mp, Ms, Lp, Ls

    def plot_lcurve(self,file='xa000112000rsl_p0px1000_cl_2_exclude_153171242_153178242_PIX_17_lcurve_b128.fits'):
        f = fits.open(file)
        lcurve = f[1].data
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        # self.ax.set_yscale('log')
        # self.ax.grid(linestyle='dashed')
        self.ax.set_title(r'Light Curve')
        self.ax.set_ylabel("Rate (count/s)") 
        self.ax.scatter(lcurve["TIME"],lcurve["RATE"])
        self.ax.plot(lcurve["TIME"],lcurve["RATE"])
        self.ax.set_xlabel('TIME')
        print(f'median = {np.median(lcurve["RATE"])}')
        print(f'average = {np.average(lcurve["RATE"])}')
        #self.ax.set_ylim(1e-3,1.9)
        # self.ax.axhline(np.median(lcurve["RATE"]),0,np.max(lcurve['TIME']),color='black',linestyle='dashed')
        plt.show()
        self.fig.savefig('lcurve.png',dpi=300)      

    def show_info(self, evtfile):
        f = fits.open(evtfile)
        print(f.info())
        evt = f['EVENTS'].data
        time = evt['TIME']
        pi = evt['PI']
        itype = evt['ITYPE']
        type_mask = itype == 0
        
        ## fig
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        bin_edge = np.arange(np.min(time),np.max(time),50)
        hist, bins = np.histogram(time, bins=bin_edge)
        ax[0].scatter((bins[:-1]-bins[:-1][0])*1e-3,hist/50,s=1)
        ax[0].set_xlabel('Time (ksec)')
        ax[0].set_ylabel('Count/s (bin=50s)')
        ax[0].set_title('Light Curve')

        e_bin = np.arange(2,10,1e-3)
        ehist,ebin = np.histogram(pi[type_mask]/2000,bins=e_bin)
        ax[1].step(ebin[:-1],ehist)
        ax[1].set_xlabel('Energy (keV)')
        ax[1].set_ylabel('Count')
        ax[1].set_yscale('log')
        ax[1].set_title('Energy Spectrum')
        fig.tight_layout()
        plt.show()

    def show_info_cor(self, evtfile, evtfile2):
        f = fits.open(evtfile)
        print(f.info())
        evt = f['EVENTS'].data
        time = evt['TIME']
        pi = evt['PI']
        itype = evt['ITYPE']
        type_mask = itype == 0
        bin_edge = np.arange(np.min(time),np.max(time),50)
        hist, bins = np.histogram(time, bins=bin_edge)
        e_bin = np.arange(2,10,1e-3)
        ehist,ebin = np.histogram(pi[type_mask]/2000,bins=e_bin)

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].scatter((bins[:-1]-bins[:-1][0])*1e-3,hist/50,s=1)
        ax[1].step(ebin[:-1],ehist)

        f = fits.open(evtfile2)
        print(f.info())
        evt = f['EVENTS'].data
        time = evt['TIME']
        pi = evt['PI']
        itype = evt['ITYPE']
        type_mask = itype == 0
        bin_edge = np.arange(np.min(time),np.max(time),50)
        hist, bins = np.histogram(time, bins=bin_edge)
        e_bin = np.arange(2,10,1e-3)
        ehist,ebin = np.histogram(pi[type_mask]/2000,bins=e_bin)


        ax[0].scatter((bins[:-1]-bins[:-1][0])*1e-3,hist/50,s=1)
        ax[1].step(ebin[:-1],ehist)

        ax[0].set_xlabel('Time (ksec)')
        ax[0].set_ylabel('Count/s (bin=50s)')
        ax[0].set_title('Light Curve')
        ax[1].set_xlabel('Energy (keV)')
        ax[1].set_ylabel('Count')
        ax[1].set_yscale('log')
        ax[1].set_title('Energy Spectrum')
        fig.tight_layout()
        plt.show()

    def cor_pi(self, evtfile, evtfile2):
        f = fits.open(evtfile)
        f2 = fits.open(evtfile2)
        print(f.info())
        evt = f['EVENTS'].data
        time = evt['TIME']
        pi = evt['PI']
        itype = evt['ITYPE']
        type_mask = itype == 0
        evt2 = f2['EVENTS'].data
        time2 = evt2['TIME']
        pi2 = evt2['PI']
        itype2 = evt2['ITYPE']
        type_mask2 = itype2 == 0

        
        ## fig
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        bin_edge = np.arange(np.min(time),np.max(time),50)
        hist, bins = np.histogram(time, bins=bin_edge)
        bin_edge2 = np.arange(np.min(time2),np.max(time2),50)
        hist2, bins2 = np.histogram(time2, bins=bin_edge2)
        ax[0].scatter((bins[:-1]-bins[:-1][0])*1e-3,hist/50,s=1,color='black')
        ax[0].scatter((bins2[:-1]-bins2[:-1][0])*1e-3,hist2/50,s=1,color='red')
        ax[0].set_xlabel('Time (ksec)')
        ax[0].set_ylabel('Count/s (bin=50s)')
        ax[0].set_title('Light Curve')

        e_bin = np.arange(2,10,1e-3)
        ehist,ebin = np.histogram(pi[type_mask]/2000,bins=e_bin,color='black')
        ehist2,ebin2 = np.histogram(pi2[type_mask2]/2000,bins=e_bin,color='red')
        ax[1].step(ebin[:-1],ehist)
        ax[1].step(ebin2[:-1],ehist2)
        ax[1].set_xlabel('Energy (keV)')
        ax[1].set_ylabel('Count')
        ax[1].set_yscale('log')
        ax[1].set_title('Energy Spectrum')
        fig.tight_layout()
        plt.show()

    def plot_cal_fwhm(self):
        file = '/Users/keitatanaka/xraydata/caldb/data/xrism/resolve/bcf/response/xa_rsl_rmfparam_20190101v006.fits.gz'
        f = fits.open(file)
        energy = f[1].data['ENERGY']
        fwhm_MnKa = []
        pixel = []
        for i in range(36):
            fwhm = f[1].data[f'PIXEL{i}'][:]
            #plt.plot(energy,fwhm,label=f'pixel {i}')
            fwhm_MnKa.append(fwhm[np.abs(energy-6000).argmin()])
            pixel.append(i)
        plt.scatter(pixel,fwhm_MnKa)
        #plt.xlim(2000,8000)
        plt.show()

    def resolve_det_image_plot(self,file):
        # FITSファイルを読み込む
        hdu_list = fits.open(file)
        image_data = hdu_list[0].data  
        hdu_list.close()

        # 画像のデータ範囲を取得（ゼロ除去のため）
        image_data = np.where(image_data > 0, image_data, np.nan)
        height, width = image_data.shape
        print(width, height)
        offset = 0.5
        exposure = 14.204e+3
        # 描画
        plt.figure(figsize=(8, 6))
        plt.imshow(image_data/exposure, origin='lower', cmap=viridis_half, norm=LogNorm(), extent=[offset, width+offset, offset, height+offset])

        # 軸の設定
        plt.xlim(offset, image_data.shape[1]-2+offset)  # x軸の範囲（最小値1）
        plt.ylim(offset, image_data.shape[0]-2+offset)  # y軸の範囲（最小値1）

        # カラーバーをつける
        cbar = plt.colorbar(label='Count rate (cts/s)')
        
        # 軸ラベルを設定
        plt.xlabel("Detector X")
        plt.ylabel("Detector Y")

        from matplotlib.ticker import MultipleLocator
    # グリッドを0.5ずらす
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))  # X軸の副目盛り間隔
        ax.yaxis.set_major_locator(MultipleLocator(1))  # Y軸の副目盛り間隔
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # X軸の副グリッド
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))  # Y軸の副グリッド

        self.ALL_PIXEL = [" ", "14", "16", "8", "6", "5", "11", "13", "15", "7", "4", "3", "9", "10", "17", "0", "2", "1", "19", "20", "18", "35", "28", "27", "21", "22", "25", "33", "31", "29", "23", "24", "26", "34", "32", "30"]
        plt.grid(True, which="minor", linestyle="--", linewidth=0.5)  # 副目盛りのグリッドを適用
        for num, pix in enumerate(self.ALL_PIXEL):
            plt.text(num%6+1,int(num/6)+1, pix, ha="center", va="center")

        #plt.title("FITS Image (Log-Scale)")
        plt.savefig("mxs_max.png", dpi=300)
        plt.show()

    def image_plot(self,file):
        # FITSファイルを読み込む
        hdu_list = fits.open(file)
        image_data = hdu_list[0].data  
        hdu_list.close()

        # 画像のデータ範囲を取得（ゼロ除去のため）
        image_data = np.where(image_data > 0, image_data, np.nan)
        height, width = image_data.shape
        print(width, height)
        offset = 0
        exposure = 1
        # 描画
        plt.figure(figsize=(8, 6))
        plt.imshow(image_data/exposure, origin='lower', cmap=viridis_half, norm=LogNorm(), extent=[offset, width+offset, offset, height+offset])

        # 軸の設定
        # plt.xlim(offset, image_data.shape[1]-2+offset)  # x軸の範囲（最小値1）
        # plt.ylim(offset, image_data.shape[0]-2+offset)  # y軸の範囲（最小値1）

        # カラーバーをつける
        cbar = plt.colorbar(label='Count rate (cts/s)')
        
        # 軸ラベルを設定
        plt.xlabel("Detector X")
        plt.ylabel("Detector Y")

        from matplotlib.ticker import MultipleLocator
    # グリッドを0.5ずらす
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MultipleLocator(1))  # X軸の副目盛り間隔
        # ax.yaxis.set_major_locator(MultipleLocator(1))  # Y軸の副目盛り間隔
        # ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # X軸の副グリッド
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))  # Y軸の副グリッド

        # self.ALL_PIXEL = [" ", "14", "16", "8", "6", "5", "11", "13", "15", "7", "4", "3", "9", "10", "17", "0", "2", "1", "19", "20", "18", "35", "28", "27", "21", "22", "25", "33", "31", "29", "23", "24", "26", "34", "32", "30"]
        # plt.grid(True, which="minor", linestyle="--", linewidth=0.5)  # 副目盛りのグリッドを適用
        # for num, pix in enumerate(self.ALL_PIXEL):
        #     plt.text(num%6+1,int(num/6)+1, pix, ha="center", va="center")

        #plt.title("FITS Image (Log-Scale)")
        plt.savefig("mxs_max.png", dpi=300)
        plt.show()

from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class InterpolatorPlotter:
    def __init__(self, obsid, t_start, t_end, dates):
        self.obsid = obsid
        self.t_start = t_start
        self.t_end = t_end
        self.dates = dates
        self.t0, self.t1 = self.XRISMtime(Time(t_start)), self.XRISMtime(Time(t_end))
        self.times = {key: self.XRISMtime(Time(val)) for key, val in dates.items()}
        self.plot_params = {
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight': 500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in'
        }

        plt.rcParams.update(self.plot_params)
        
    def XRISMtime(self, t):
        return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

    def create_interpolator(self, data, pixel_mask, column_name):
        return interp1d(
            data['TIME'][pixel_mask],
            data[column_name][pixel_mask],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

    def load_data(self, file_path, column_name):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            return data

    def plot_correlation_calpix(self, calpix_data, fe55_data, column_name):
        pixel_12_mask = calpix_data['PIXEL'] == 12
        pixel_12_interp = self.create_interpolator(calpix_data, pixel_12_mask, column_name)

        fig, axs = plt.subplots(6, 6, figsize=(16, 9))
        axs = axs.ravel()

        for i in range(36):
            pixel_mask = fe55_data['PIXEL'] == i
            if len(fe55_data[column_name][pixel_mask]) > 0:
                pixel_interp = self.create_interpolator(fe55_data, pixel_mask, column_name)

                time_range = np.linspace(self.t0, self.t1, num=1000)
                pixel_12_values = pixel_12_interp(time_range)
                pixel_values = pixel_interp(time_range)

                axs[i].scatter(pixel_values*1e+3, pixel_12_values*1e+3, s=1)
                axs[i].set_title(f'PIXEL {i}')
                #axs[i].set_xlabel('calpix PIXEL 12')
                #axs[i].set_ylabel(f'55Fe PIXEL {i}')

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.obsid}_correlation.png', dpi=300)

    def plot_correlation(self, fe55_data, column_name, pixel_num):
        pixel_12_mask = fe55_data['PIXEL'] == pixel_num
        pixel_12_interp = self.create_interpolator(fe55_data, pixel_12_mask, column_name)

        fig, axs = plt.subplots(6, 6, figsize=(16, 9))
        axs = axs.ravel()

        for i in range(36):
            pixel_mask = fe55_data['PIXEL'] == i
            if len(fe55_data[column_name][pixel_mask]) > 0:
                pixel_interp = self.create_interpolator(fe55_data, pixel_mask, column_name)

                time_range = np.linspace(self.t0, self.t1, num=1000)
                pixel_12_values = pixel_12_interp(time_range)
                pixel_values = pixel_interp(time_range)

                axs[i].scatter(pixel_values*1e+3, pixel_12_values*1e+3, s=1)
                axs[i].set_title(f'PIXEL {i}')
                #axs[i].set_xlabel('calpix PIXEL 12')
                #axs[i].set_ylabel(f'55Fe PIXEL {i}')
        plt.title(f'PIXEL {pixel_num}')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{self.obsid}_correlation_{pixel_num}.png', dpi=300)

    def run(self, file_paths, column_name):
        calpix_data = self.load_data(file_paths['calpix'], column_name)
        fe55_data = self.load_data(file_paths['55Fe'], column_name)
        self.plot_correlation(calpix_data, fe55_data, column_name)

    def create_interpolators(self, data, pixel_masks, column_name, interpname):
        interpolators = {interpname: {}}
        for i in range(0, 36):
            pixel_mask = data['PIXEL'] == i
            if len(data[column_name][pixel_mask]) > 0:
                interpolators[interpname][i] = interp1d(
                    data['TIME'][pixel_mask], 
                    data[column_name][pixel_mask] , 
                    kind='linear',
                    bounds_error=False, 
                    fill_value=(data[column_name][pixel_mask][0], data[column_name][pixel_mask][-1])
                )
        return interpolators

    def load_and_plot_fits(self, file_path, ax, color_map, linestyle, scatter, column_name, interpname, marker='.'):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            pixel_masks = {i: data['PIXEL'] == i for i in range(36)}
            interp = self.create_interpolators(data, pixel_masks, column_name, interpname)
            self.plot_pixel_data(data, ax, pixel_masks, color_map, linestyle, scatter, column_name, interpname, marker)
            return data, interp

    def plot_pixel_data(self, data, ax, pixel_masks, color_map, linestyle, scatter, column_name, interpname, marker='.'):
        for i in range(0, 36):
            row, col = divmod(i, 6)
            if row < 6:
                pixel_mask = pixel_masks[i]
                if scatter:
                    if column_name == 'TEMP_FIT':
                        ax[0, row].scatter(data['TIME'][pixel_mask], data[column_name][pixel_mask]*1e3 , color=color_map((i % 6) / 6), label=f'{i}', marker=marker)
                    else:
                        ax[0, row].scatter(data['TIME'][pixel_mask], data[column_name][pixel_mask] , color=color_map((i % 6) / 6), label=f'{i}', marker=marker)

            if column_name == 'TEMP_FIT':
                ax[0, row].plot(data['TIME'][pixel_mask], data[column_name][pixel_mask]*1e3 ,linestyle, color=color_map((i % 6) / 6))
            else:
                ax[0, row].plot(data['TIME'][pixel_mask], data[column_name][pixel_mask] , linestyle, color=color_map((i % 6) / 6))

        # for a in ax[0]:
        #     a.legend(fontsize=6)
        if column_name == 'TEMP_FIT':
            ax[0, 0].set_ylabel('TEMP_FIT (mK)')
            ax[1, 0].set_ylabel('TEMP_DIFF (mK)')
        elif column_name == 'SHIFT':
            ax[0, 0].set_ylabel(f'SHIFT (eV)')
            ax[1, 0].set_ylabel('SHIFT_DIFF (eV)')

    def plot_cal_pix(self, ax, interp, color_map):
        file = 'xa000112000rsl_000_pxcal.ghf.gz'

    def plot_diff(self, ax, interp_55Fe, interp_mxs, color_map, colmun_name):
        common_pixels = set(interp_55Fe["55Fe"].keys()) & set(interp_mxs["mxs"].keys())
        for i in range(0, 36):
            if i in common_pixels:
                time_values = np.linspace(self.t0, self.t1, num=1000)
                temp_55Fe = interp_55Fe["55Fe"][i](time_values)
                temp_mxs = interp_mxs["mxs"][i](time_values)
                temp_diff = temp_55Fe - temp_mxs

                row, col = divmod(i, 6)
                if colmun_name == 'TEMP_FIT':
                    ax[1, row].plot(time_values, temp_diff*1e+3, label=f"Diff {i}", color=color_map((i % 6) / 6))
                else:
                    ax[1, row].plot(time_values, temp_diff, label=f"Diff {i}", color=color_map((i % 6) / 6))
        
        ax[1, 0].set_ylim(-0.3, 0.3)

    def plot_MXS_current(self, ax):
        datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs'
        ehk =  f'{datadir}/xa000112000.ehk'
        hk1 =  f'{datadir}/xa000112000rsl_a0.hk1'
        with fits.open(hk1) as f:
            HK_SXS_FWE = f['HK_SXS_FWE'].data

            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_CAL']
            MXS_2_I     = HK_SXS_FWE['FWE_I_LED2_CAL']
            for i in range(0,6):
                ax[2,i].plot(MXS_TIME,MXS_1_I,c='k',label='MXS1')

            ax[2,0].set_ylabel('LED Current (mA)')

            #ax[-1,-1].fill_betweenx(MXS_TIME,49.98,50.02,where=MXS_1_I > 0.5,color='gray',alpha=0.3)

    def plot_MXS_GTI(self, ax):
        datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs'
        ehk =  f'{datadir}/xa000112000.ehk'
        hk1 =  f'{datadir}/xa000112000rsl_a0.hk1'
        mxs_gti = f'{datadir}/xa000112000rsl_mxcs.gti'
        with fits.open(mxs_gti) as f:
            MXS1_ON = f['GTIMXSCSON1'].data

            MXS_start    = MXS1_ON['START'].reshape(-1)
            MXS_stop     = MXS1_ON['STOP'].reshape(-1)
            print(MXS_start)
            print(MXS_stop)
            #ax[-1,-1].fill_betweenx(10,MXS_start,MXS_stop,color='gray',alpha=0.3)
            for i in range(len(MXS_start)):
                for col in range(2):
                    for row in range(6):
                        ax[col,row].axvspan(MXS_start[i],MXS_stop[i],color='gray',alpha=0.3)
            #ax[2,0].set_ylabel('LED Current (mA)')

    def run_interp(self, file_paths, column_name):
        fig, ax = plt.subplots(2, 6, figsize=(18, 6), sharex=True, sharey='row')
        ax[-1, -1].set_xlim(self.t0, self.t1)
        ax[-1, -1].set_xticks(np.arange(self.t0, self.t1, 100e+3))
        ax[-1, -1].set_xticklabels(np.arange(0, (self.t1 - self.t0) // 1000, 100))

        data_55Fe, interp_55Fe = self.load_and_plot_fits(file_paths['55Fe'], ax, cm.jet, '-', True, column_name, '55Fe', 'o')
        # data_55Fe, interp_calpix = self.load_and_plot_fits(file_paths['calpix'], ax, cm.jet, '.', False, column_name, 'calpix')
        data_mxs, interp_mxs = self.load_and_plot_fits(file_paths['mxs'], ax, cm.jet, '-.', True, column_name, 'mxs', '*')
        
        self.plot_diff(ax, interp_55Fe, interp_mxs, cm.jet, column_name)

        # Add vertical dashed lines at the specified dates
        for date_label, date_value in self.times.items():
            color = 'black'
            if date_label in ['ADR1', 'ADR2']:
                color = 'red'
            elif date_label == 'LED_BRIGHT':
                color = 'blue'
            
            for axis in ax.ravel():
                axis.axvline(date_value, color=color, linestyle='--', label=date_label)

        # self.plot_MXS_current(ax)
        #self.plot_MXS_GTI(ax)
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{column_name}_temperature_tracking.png', dpi=300, transparent=True)

    def plot_selected_pixel_data(self, data, ax, selected_pixels, color_map, linestyle, scatter, column_name, marker='.', name='55Fe'):
        """
        特定のピクセルデータをプロットするための関数
        """
        selected_pixels = [int(pixel) for pixel in selected_pixels]  # 選択されたピクセルを整数に変換
        for i in selected_pixels:
            pixel_mask = data['PIXEL'] == i
            if not np.any(pixel_mask):  # ピクセルマスクが存在しない場合はスキップ
                print(f"Warning: No data for pixel {i}.")
                continue

            if scatter:
                if column_name == 'TEMP_FIT':
                    ax.scatter(data['TIME'][pixel_mask], data[column_name][pixel_mask]*1e3,
                            color=color_map((i % 36) / 36), label=f"Pixel {i} {name}", marker=marker)
                else:
                    ax.scatter(data['TIME'][pixel_mask], data[column_name][pixel_mask],
                            color=color_map((i % 36) / 36), label=f"Pixel {i} {name}", marker=marker)

            if column_name == 'TEMP_FIT':
                ax.plot(data['TIME'][pixel_mask], data[column_name][pixel_mask]*1e3,
                        linestyle, color=color_map((i % 36) / 36))
            else:
                ax.plot(data['TIME'][pixel_mask], data[column_name][pixel_mask],
                        linestyle, color=color_map((i % 36) / 36))

        ax.set_xlabel("Time (ks)")
        if column_name == 'TEMP_FIT':
            ax.set_ylabel('Temperature (mK)')
        elif column_name == 'SHIFT':
            ax.set_ylabel('Energy Shift (eV)')
        ax.legend(fontsize=10)

    def plot_selected_pixels_interp(self, file_paths, column_name, selected_pixels, interp):
        """
        特定のピクセルのみを一つの図にプロットし、差分を下部にプロットする関数
        """
        if interp == True:
            fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # ファイルデータをロード
        with fits.open(file_paths['55Fe']) as hdul_55fe:
            data_55Fe = hdul_55fe[1].data
            self.plot_selected_pixel_data(data_55Fe, ax, selected_pixels, cm.tab20b, '-.', True, column_name, '.', '55Fe')

        with fits.open(file_paths['mxs']) as hdul_mxs:
            data_mxs = hdul_mxs[1].data
            self.plot_selected_pixel_data(data_mxs, ax, selected_pixels, cm.tab20b, '-', True, column_name, 'v', 'MXS')

        with fits.open(file_paths['calpix']) as hdul_mxs:
            data_mxs = hdul_mxs[1].data
            self.plot_selected_pixel_data(data_mxs, ax, selected_pixels, cm.tab20b, '-', True, column_name, 'v', 'MXS')

        # 差分を計算して下部にプロット
        selected_pixels = [int(pixel) for pixel in selected_pixels]  # ピクセルを整数に変換
        if interp == True:
            for i in selected_pixels:
                mask_55Fe = data_55Fe['PIXEL'] == i
                mask_mxs = data_mxs['PIXEL'] == i
                if not (np.any(mask_55Fe) and np.any(mask_mxs)):
                    print(f"Warning: No data for pixel {i} in one of the datasets.")
                    continue

                # TIMEとTEMP_FITの補間関数を作成
                interp_55Fe = interp1d(data_55Fe['TIME'][mask_55Fe], data_55Fe[column_name][mask_55Fe],
                                    kind='linear', bounds_error=False, fill_value="extrapolate")
                interp_mxs = interp1d(data_mxs['TIME'][mask_mxs], data_mxs[column_name][mask_mxs],
                                    kind='linear', bounds_error=False, fill_value="extrapolate")

                # 時間範囲で差分を計算
                time_values = np.linspace(self.t0, self.t1, 1000)
                diff = interp_55Fe(time_values) - interp_mxs(time_values)
                if interp == True:
                    if column_name == 'TEMP_FIT':
                        ax_diff.plot(time_values, diff * 1e3, label=f"Pixel {i} Diff", color=cm.tab20b((i % 36) / 36))
                    else:
                        ax_diff.plot(time_values, diff, label=f"Pixel {i} Diff", color=cm.tab20b((i % 36) / 36))

        #ax_diff.set_xlabel("Time")
        #ax_diff.legend(fontsize=8)

        # 指定した日時に垂直線を追加
        for date_label, date_value in self.times.items():
            color = 'black'
            if date_label in ['ADR1', 'ADR2']:
                color = 'red'
            elif date_label == 'LED_BRIGHT':
                color = 'blue'
            ax.axvline(date_value, color=color, linestyle='--')
            if interp == True:
                ax_diff.axvline(date_value, color=color, linestyle='--')

        # x軸上部に日時ラベルを追加
        ax_top = ax.twiny()
        custom_ticks = list(self.times.values())  # 日時をticksとして利用
        custom_labels = list(self.times.keys())   # ラベルとして利用
        ax_top.set_xticks(custom_ticks)           # ticksをセット
        ax_top.set_xticklabels(custom_labels, rotation=45, ha='left')  # ラベルを設定し、回転
        ax_top.set_xlim(self.t0, self.t1)         # 上部軸の範囲を下部と合わせる
        ax_top.xaxis.set_tick_params(pad=10)
        if interp == True:
            ax_diff.set_xticks(np.arange(self.t0, self.t1, 100e+3),[0,100,200])
            ax_diff.set_ylabel("Temp Diff")
        else:
            ax.set_xticks(np.arange(self.t0, self.t1, 100e+3),[0,100,200])
        # 軸の設定とレイアウト調整
        ax.set_xlim(self.t0, self.t1)
        if column_name == 'TEMP_FIT':
            ax.set_ylabel('Temperature (mK)')
        elif column_name == 'SHIFT':
            ax.set_ylabel('Energy Shift (eV)')

        # 上下の間隔を縮小してx軸をくっつける
        #fig.subplots_adjust(hspace=0.)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{column_name}_selected_pixels_with_difference.png', dpi=300, transparent=True)
# InterpolatorPlotter.plot_selected_pixels_interp = plot_selected_pixels_interp

# 新しい関数を実行するためのコードを追加
def plot_gain_tracking():
    obsid = '000112000'
    t_start, t_end = '2023-11-08 10:21:00', '2023-11-11 12:01:04'
    dates = {
        'Open': '2023-11-08 10:21:00',
        'ND': '2023-11-08 23:51:31',
        'Be': '2023-11-09 05:26:31',
        'OBF': '2023-11-09 13:02:51',
        '55Fe': '2023-11-09 16:47:51',
        'ADR1': '2023-11-08 20:10:01',
        'ADR2': '2023-11-10 15:10:00'
        #'LED_BRIGHT': '2023-11-10 12:01:01'
    }
# 80.592 ksec (55Fe-ADR2)
    file_paths = {
        'mxs': '/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf',
        '55Fe': '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz',
        'calpix': '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/xa000112000rsl_000_pxcal.ghf.gz'
    }

    column_name = 'TEMP_FIT'  # or any other column you want to use

    plotter = InterpolatorPlotter(obsid, t_start, t_end, dates)

    # 特定のピクセルを選択してプロット
    selected_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
    selected_pixels = ['19', '35']
    selected_pixels = ['00', '17', '18', '35']
    # plotter.run_interp(file_paths, column_name)
    #plotter.plot_selected_pixels_interp(file_paths, column_name, selected_pixels, False)
    Fe_data = plotter.load_data(file_paths['55Fe'], column_name)
    calpix_data = plotter.load_data(file_paths['calpix'], column_name)
    plotter.plot_correlation(Fe_data, column_name, 2)
    
