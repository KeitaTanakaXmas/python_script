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
import csv

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

    def img_plot(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(self.data_img,  origin='lower', norm=LogNorm())
        self.fig.colorbar(im)

        # 四角形を描画する領域の座標
        x = self.axis_center
        y = self.axis_center
        width = 1
        height = 1

        # 四角形のパッチを作成
        rect = patches.Rectangle((x - 0.5, y - 0.5), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
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

    def plot_shift_fit(self,num):
        from scipy.optimize import fsolve
        file1 = '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz'
        file2 = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf'
        hdul = fits.open(file1)
        Fe_data = np.array(hdul[0].data)
        hdul = fits.open(file2)
        MXS_data = np.array(hdul[0].data)


    def ehk_check(self,calmethod="55Fe" ):

        from astropy.io import fits
        from astropy.time import Time
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        params = {#'backend': 'pdf',
            'axes.labelsize': 14,
            'axes.linewidth': 0.7,
            'font.size': 14,
            'legend.fontsize': 10,
            'legend.borderpad': 0.5,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'text.usetex': True,
            'font.family': 'serif'
            }

        obsid = '000112000'
        t_start, t_end = '2023-11-08 10:21:04',  '2023-11-11 12:01:04'
        if calmethod == "55Fe":
            datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve'
            ehk =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/auxil/xa000112000.ehk.gz'
            hk1 =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1'
            evt_uf = glob.glob(f'{datadir}/event_uf/*.evt.gz')
            evt_cl = glob.glob(f'{datadir}/event_cl/*1000*.evt.gz')
        elif calmethod == "mxs":
            datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs'
            ehk =  f'{datadir}/xa000112000.ehk'
            hk1 =  f'{datadir}/xa000112000rsl_a0.hk1'
            evt_uf = glob.glob(f'{datadir}/*.evt')
            evt_cl = glob.glob(f'{datadir}/*5000*.evt')


        # evt_cl = glob('./resolve/event_uf/screen_test.evt')


        def XRISMtime(t):
            return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

        t0, t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
        tlist = np.linspace(t0, t1)
        print('--------------------------')
        print(t0)
        print(evt_cl)

        colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})

        plt.rcParams.update(params)
        fig, ax = plt.subplots(4, sharex=True, figsize=(6, 8))
        ax[-1].set_xlim(t0, t1)
        ax[-1].set_xlabel('Time (ks) from ' + t_start)
        ax[-1].set_xticks(np.arange(t0, t1, 50000))
        ax[-1].set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))

        with fits.open(ehk) as f:
            EHK = f['EHK'].data
            # ax[2].plot(EHK['TIME'], EHK['ELV'], c='k')
            # ax[2].axhline(-5, color='r', lw=1)
            # ax[2].set_ylim(-60, 105)
            # ax[2].set_ylabel('Elevation (deg)')
            # ax[3].plot(EHK['TIME'], EHK['SAA'], c='k')
            # ax[3].axhline(0, color='r', lw=1)
            # ax[3].set_ylabel('SAA')
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            ax[0].plot(HK_SXS_TEMP['TIME'], HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot(HK_SXS_FWE['TIME'], HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW pos (deg)')

            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_CAL']
            MXS_2_I     = HK_SXS_FWE['FWE_I_LED2_CAL']

            ax[2].plot(MXS_TIME,MXS_1_I,c='k',label='MXS1')
            ax[2].plot(MXS_TIME,MXS_2_I,c='blue',label='MXS2')

            ax[2].set_ylabel('LED Current (mA)')
            ax[2].legend()
        for file in evt_uf:
            with fits.open(file) as f:
                TIME = f['EVENTS'].data['TIME']
                # ITYPE = f['EVENTS'].data['ITYPE']
                # for i in range(len(ITYPE)):
                    # if ITYPE[i] == 6:
                    #    ax[3].axvline(TIME[i], lw=0.5, color='b', alpha=0.25)
                bin = np.arange(int(t0), int(t1))[::100]
                events, BIN = np.histogram(TIME, bin)
                events = np.where(events == 0, np.nan, events)/100
                ax[3].axhline(200, color='r', lw=1)
                ax[3].scatter(bin[:-1], events, s=1, c='black')
                ax[3].set_ylabel('Count rate (count/s)')
                ax[3].set_ylim(0, 250)
        for file in evt_cl:
            with fits.open(file) as f:
                    GTI = f['GTI'].data
                    for a in ax:
                        for i in range(len(GTI) - 1):
                            a.axvspan(GTI[i][1], GTI[i+1][0], color='r', alpha=0.25)
                        if GTI[0][0] > t0:
                            a.axvspan(t0, GTI[0][0], color='r', alpha=0.25)
                        if GTI[-1][1] < t1:
                            a.axvspan(GTI[-1][1], t1, color='r', alpha=0.25)
        fig.tight_layout()
        plt.show()
        fig.savefig('ehk_check.png',dpi=300)

    def ehk_check_single(self, datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve'):

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
            ehk =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/auxil/xa000112000.ehk.gz'
            hk1 =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1'
            evt_uf = glob.glob(f'{datadir}/event_uf/*.evt.gz')
            evt_cl = glob.glob(f'{datadir}/event_cl/*.evt.gz')
            # evt_cl = glob('./resolve/event_uf/screen_test.evt')


            def XRISMtime(t):
                return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

            t0, t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
            tlist = np.linspace(t0, t1)
            print('--------------------------')
            print(t0)

            colormap_FW = pd.DataFrame({'FW': ['0', '1', '5'], 'c': ['b', 'k', 'r']})

            fig, ax = plt.subplots(1, sharex=True, figsize=(6, 3))
            ax.set_xlim(t0, t1)
            ax.set_xlabel('Time (ks)')
            ax.set_xticks(np.arange(t0, t1, 50000))
            ax.set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))

            # with fits.open(ehk) as f:
            #     EHK = f['EHK'].data
                # ax[2].plot(EHK['TIME'], EHK['ELV'], c='k')
                # ax[2].axhline(-5, color='r', lw=1)
                # ax[2].set_ylim(-60, 105)
                # ax[2].set_ylabel('Elevation (deg)')
                # ax[3].plot(EHK['TIME'], EHK['SAA'], c='k')
                # ax[3].axhline(0, color='r', lw=1)
                # ax[3].set_ylabel('SAA')
            with fits.open(hk1) as f:
                # HK_SXS_TEMP = f['HK_SXS_TEMP'].data
                # ax[0].plot(HK_SXS_TEMP['TIME'], HK_SXS_TEMP['ST1_CTL'], c='k')
                # ax[0].set_ylim(50.25, 51)
                # ax[0].set_ylabel('$T$ (mK)')
                HK_SXS_FWE = f['HK_SXS_FWE'].data
                # ax.plot(HK_SXS_FWE['TIME'], HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k', lw=2)
                # ax.set_ylim(0.1, 360)
                # ax.set_ylabel('FW pos (deg)')

                MXS_TIME    = HK_SXS_FWE['TIME']
                MXS_1_I     = HK_SXS_FWE['FWE_I_LED1']
                MXS_2_I     = HK_SXS_FWE['FWE_I_LED2']

                ax.plot(MXS_TIME,MXS_1_I,c='k',label='MXS1')
                ax.plot(MXS_TIME,MXS_2_I,c='blue',label='MXS2')
                ax.set_ylabel('LED Current (mA)')
                ax.legend()
            # for file in evt_uf:
            #     with fits.open(file) as f:
            #         TIME = f['EVENTS'].data['TIME']
            #         # ITYPE = f['EVENTS'].data['ITYPE']
            #         # for i in range(len(ITYPE)):
            #             # if ITYPE[i] == 6:
            #             #    ax[3].axvline(TIME[i], lw=0.5, color='b', alpha=0.25)
            #         bin = np.arange(int(t0), int(t1))[::100]
            #         events, BIN = np.histogram(TIME, bin)
            #         events = np.where(events == 0, np.nan, events)/100
            #         ax[3].axhline(200, color='r', lw=1)
            #         ax[3].scatter(bin[:-1], events, s=1, c='black')
            #         ax[3].set_ylabel('Count rate (count/s)')
            #         ax[3].set_ylim(0, 250)
            # for file in evt_cl:
            #     with fits.open(file) as f:
            #             GTI = f['GTI'].data
            #             for a in ax:
            #                 for i in range(len(GTI) - 1):
            #                     a.axvspan(GTI[i][1], GTI[i+1][0], color='r', alpha=0.25)
            #                 if GTI[0][0] > t0:
            #                     a.axvspan(t0, GTI[0][0], color='r', alpha=0.25)
            #                 if GTI[-1][1] < t1:
            #                     a.axvspan(GTI[-1][1], t1, color='r', alpha=0.25)
            spine_width = 2  # スパインの太さ
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis='both',direction='in',width=1.5)
            fig.tight_layout()
            plt.show()
            fig.savefig('ehk_check_led.png',dpi=300,transparent=True)

    def ehk_check_double(self,calmethod="55Fe"):

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

        datadir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve'
        ehk =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/auxil/xa000112000.ehk.gz'
        hk1 =  '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/hk/xa000112000rsl_a0.hk1'
        evt_uf = glob.glob(f'{datadir}/event_uf/*.evt.gz')
        evt_cl = glob.glob(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/all_pixel/xa000112000rsl_p0px5000_cl_3_Hp_without_calpix.evt')

        evt_cl2 = glob.glob(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_cl_data/all_pixel/xa000112000rsl_p0px5000_cl_3_Hp_without_calpix.evt')
        mxs = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf'
        Fe = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz'
        # evt_cl = glob('./resolve/event_uf/screen_test.evt')

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
        ax[-1].set_xlabel('Time (ks)')
        ax[-1].set_xticks(np.arange(t0, t1, 50000))
        ax[-1].set_xticklabels(np.arange(0, (t1 - t0)//1000, 50))

        # with fits.open(ehk) as f:
        #     EHK = f['EHK'].data
            # ax[2].plot(EHK['TIME'], EHK['ELV'], c='k')
            # ax[2].axhline(-5, color='r', lw=1)
            # ax[2].set_ylim(-60, 105)
            # ax[2].set_ylabel('Elevation (deg)')
            # ax[3].plot(EHK['TIME'], EHK['SAA'], c='k')
            # ax[3].axhline(0, color='r', lw=1)
            # ax[3].set_ylabel('SAA')
        with fits.open(hk1) as f:
            HK_SXS_TEMP = f['HK_SXS_TEMP'].data
            ax[0].plot(HK_SXS_TEMP['TIME'], HK_SXS_TEMP['ST1_CTL'], c='k')
            ax[0].set_ylim(50.25, 51)
            ax[0].set_ylabel('$T$ (mK)')
            HK_SXS_FWE = f['HK_SXS_FWE'].data
            ax[1].plot(HK_SXS_FWE['TIME'], HK_SXS_FWE['FWE_FW_POSITION1_CAL'], c='k')
            ax[1].set_ylim(0, 360)
            ax[1].set_ylabel('FW pos\n (deg)')

            MXS_TIME    = HK_SXS_FWE['TIME']
            MXS_1_I     = HK_SXS_FWE['FWE_I_LED1_CAL']
            MXS_2_I     = HK_SXS_FWE['FWE_I_LED2_CAL']

            ax[2].plot(MXS_TIME,MXS_1_I,c='k',label='MXS1')
            ax[2].plot(MXS_TIME,MXS_2_I,c='blue',label='MXS2')

            ax[2].set_ylabel('LED Current\n (mA)')
            ax[2].legend()


        with fits.open(mxs) as hdul:
            data = hdul[1].data
            pixel_mask = data['PIXEL'] == 21
            temp = data["TEMP_FIT"][pixel_mask]
            time = data["TIME"][pixel_mask]
            ax[3].plot(time,temp*1e+3,color="blue")
            ax[3].scatter(time,temp*1e+3,label="mxs",color="blue")


        with fits.open(Fe) as hdul:
            data = hdul[1].data
            pixel_mask = data['PIXEL'] == 21
            temp = data["TEMP_FIT"][pixel_mask]
            time = data["TIME"][pixel_mask]
            ax[3].plot(time,temp*1e+3,color="black") 
            ax[3].scatter(time,temp*1e+3,label="55Fe",color="black")               

        ax[3].legend()
        ax[3].set_ylabel("TEMP FIT\n (mK)")
        ax[3].set_ylim(49.975,50.025)


        self.t0, self.t1 = XRISMtime(Time(t_start)), XRISMtime(Time(t_end))
        self.times = {key: XRISMtime(Time(val)) for key, val in dates.items()}

        for date_label, date_value in self.times.items():
            color = 'black'
            if date_label in ['ADR1', 'ADR2']:
                color = 'red'
            elif date_label == 'LED_BRIGHT':
                color = 'blue'
            
            for axis in ax.ravel():
                axis.axvline(date_value, color=color, linestyle='--', label=date_label)

        te = t0 + 150e+3

        for file in evt_cl2:
            with fits.open(file) as f:
                    GTI = f['GTI'].data
                    for a in ax:
                        for i in range(len(GTI) - 1):
                            if  GTI[i][0] < te:
                                if GTI[i+1][0] > te:
                                    a.axvspan(GTI[i][1], t1, color='gray', alpha=0.25)
                                else:
                                    a.axvspan(GTI[i][1], GTI[i+1][0], color='gray', alpha=0.25)
                        if GTI[0][0] > t0:
                            a.axvspan(t0, GTI[0][0], color='gray', alpha=0.25)
                        if GTI[-1][1] < te:
                            a.axvspan(GTI[-1][1], t1, color='gray', alpha=0.25)


        fig.tight_layout()
        plt.show()
        fig.savefig('ehk_check.png',dpi=300)

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
#2.4e+4 - 1.6e+5
# 153247679
class XspecFit:
    def __init__(self,savefile):
        print('initialize')
        self.savefile=savefile
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
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

    def initialize(self,fileName,apecroot="/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9",rebin=1):
        self.fileName=fileName
        AllData.clear()
        Xset.restore(fileName=self.fileName)
        Xset.addModelString("APECROOT",apecroot)
        Xset.abund='lpgs'
        AllData.notice("2.0-20.0")
        AllData.ignore("**-2.0 20.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        Plot.setRebin(rebin,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)

    def set_init(self,apecroot="/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9",rebin=1,spec=None,rmf=None,arf=None,rng=(2.0,20.0),subject='data',multiresp=True):
        AllData.clear()
        Xset.addModelString("APECROOT",apecroot)
        Xset.abund='lpgs'
        self.load_spectrum(spec,rmf,arf,multiresp)
        self.set_data_range(rng)
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        Plot.setRebin(rebin,1000)
        self.set_xydata(subject)

    def set_data_range(self,rng=(2.0,20.0)):
        AllData.notice(f"{rng[0]}-{rng[1]}")
        AllData.ignore(f"**-{rng[0]} {rng[1]}-**")

    def load_and_setdata(self,spec,rmf,arf,rebin=1,rng=(2.0,20.0),multiresp=False):
        AllData.clear()
        self.load_spectrum(spec,rmf,arf,multiresp)
        AllData.notice(f"{rng[0]}-{rng[1]}")
        AllData.ignore(f"**-{rng[0]} {rng[1]}-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        Plot.setRebin(rebin,1000)
        self.set_xydata()

    def set_xydata(self,subject='data',plotgroup=1):
        Plot(subject)
        self.xs=Plot.x(plotgroup,1)
        self.ys=Plot.y(plotgroup,1)
        if subject == 'data':
            self.xe=Plot.xErr(plotgroup,1)
            self.ye=Plot.yErr(plotgroup,1)        

    def load_spectrum(self,spec,rmf,arf,multiresp=True):
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        if multiresp == True:
            s.multiresponse[1] = rmf

    def save_data_pixel(self,pixel):
        with h5py.File(self.savefile, 'a') as f:
            if f"pixel{pixel}" in f.keys():
                del f[f'pixel{pixel}']
            f.create_group(f'pixel{pixel}')
            f[f'pixel{pixel}'].create_dataset('xs',data=self.xs)
            f[f'pixel{pixel}'].create_dataset('ys',data=self.ys)
            f[f'pixel{pixel}'].create_dataset('xe',data=self.xe)
            f[f'pixel{pixel}'].create_dataset('ye',data=self.ye)

    def load_data_pixel(self,pixel):
        with h5py.File(self.savefile, 'r') as f:
            self.xs = f[f'pixel{pixel}']['xs'][:]
            self.ys = f[f'pixel{pixel}']['ys'][:]
            self.xe = f[f'pixel{pixel}']['xe'][:]
            self.ye = f[f'pixel{pixel}']['ye'][:]

    def load_pixel_by_pixel(self):
        #self.all_pixels  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ["00", "17", "18", "35"]
        for num,pixel in enumerate(self.all_pixels):
            spec = f"1000_PIXEL_{pixel}_merged_b1.pi"
            rmf = f"1000_pix{pixel}_L_without_Lp.rmf"
            arf = f"1000_pix{pixel}_image_1p8_8keV_1e7.arf"
            self.load_and_setdata(spec=spec,rmf=rmf,arf=arf,rebin=3)
            self.save_data_pixel(pixel)

    def cor_eff_area(self):
        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        #self.ax.set_title(r'Pixel by Pixel')
        ax.set_ylabel("Effective area (cm$^2$)") 
        ax2.set_ylabel("Transmission (1/OPEN)") 
        ax2.set_xlabel('Energy (keV)')

        name = ['OPEN','OBF', 'ND', 'Be']
        col  = ['black','blue','red','green']
        for number,identifier in enumerate(['1000','2000','3000','4000']):
            spec = f'{identifier}_center_merged_b1.pi'
            rmf = f'{identifier}_center_L_without_Lp.rmf'
            arf = f'{identifier}_center_image_1p8_8keV_1e7.arf'
            self.set_init(spec=spec,rmf=rmf,arf=arf,rebin=1,subject='eff',multiresp=False)
            ax.plot(self.xs,self.ys,label=name[number],color=col[number])
            if number == 0:
                open_energy = np.array(self.xs)
                open_eff    = np.array(self.ys)
            else:
                ax2.plot(np.array(self.xs), np.array(self.ys)/open_eff, color=col[number])
        ax.legend(fontsize=12)
        fig.tight_layout()
        name = ['OBF', 'ND', 'Be']
        col  = ['blue','red','green']
        file_dir = '/opt/CALDB/data/xrism/resolve/bcf/quanteff/'
        files = ['xa_rsl_fwpoly_20190101v002.fits','xa_rsl_fwnd_20190101v001.fits','xa_rsl_fwbe_20190101v002.fits']
        for number,file in enumerate(files):
            print(f'{file_dir}{file}')
            hdul = fits.open(f'{file_dir}{file}')[1]
            print(hdul.header)
            energy = hdul.data['ENERGY']*1e-3
            transmission = hdul.data['TRANSMISSION']
            ax2.plot(energy,transmission,'-.', color=col[number])
        plt.show()
        fig.savefig(f'./figure/cor_eff_area.png',dpi=300,transparent=True)
        
    def cor_transmission(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(linestyle='dashed')
        self.ax.set_ylabel("Transmission") 
        self.ax.set_xlabel('Energy (keV)')        
        name = ['OBF', 'ND', 'Be']
        col  = ['blue','red','green']
        file_dir = '/opt/CALDB/data/xrism/resolve/bcf/quanteff/'
        files = ['xa_rsl_fwpoly_20190101v002.fits','xa_rsl_fwnd_20190101v001.fits','xa_rsl_fwbe_20190101v002.fits']
        for number,file in enumerate(files):
            print(f'{file_dir}{file}')
            hdul = fits.open(f'{file_dir}{file}')[1]
            print(hdul.header)
            energy = hdul.data['ENERGY']
            transmission = hdul.data['TRANSMISSION']
            self.ax.plot(energy,transmission, color=col[number],label=name[number])
        self.ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #self.ax.set_xscale('log')
        #self.ax.set_yscale('log')
        self.fig.tight_layout()
        #self.fig.savefig(f'./figure/cor_trans.png',dpi=300,transparent=True)
        plt.show()
        
    def steppar(self,com):
        Plot.device = '/xs'
        Fit.steppar(com)
        Plot('contour')
        step_result = Fit.stepparResults('delstat')
        print(step_result)
        print(step_result.shape)
        plt.contour(step_result[0],step_result[1],step_result[2],levels=[2.3,4.61,9.21])
        plt.show()

    def plot_pixel_by_pixel(self,error=True):
        self.all_pixels = ["00", "17", "18", "35"]
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(linestyle='dashed')
        self.ax.set_title(r'Pixel by Pixel')
        self.ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$") 
        self.ax.set_xlabel('Energy (keV)')
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)
        self.ax.tick_params(axis='both',direction='in',width=1.5)
        for num,pixel in enumerate(self.all_pixels):
            self.load_data_pixel(pixel)
            if error == True:
                self.ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",color=cm.jet(num/len(self.all_pixels)),label=f'pixel {pixel}')
            else:
                self.ax.plot(self.xs,self.ys,color=cm.jet(num/len(self.all_pixels)),label=f'pixel {pixel}')
        self.ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #self.ax.set_xlim(5.99,6.11)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.fig.align_labels()
        self.fig.tight_layout()
        er = 'error' if error == True else 'no_error'
        self.fig.savefig(f'./figure/pixel_by_pixel_{er}.png',dpi=300,transparent=True)
        plt.show()

    def fitting(self,file='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm',spec='1000_center_merged_b1.pi',rmf='1000_center_L_without_Lp.rmf',arf='1000_center_image_1p8_8keV_1e7.arf',rebin=1,modname='test1',rng=(2.0,20.0)):
        self.set_init(apecroot='/Users/keitatanaka/apec_modify/del_wz/apec_v3.0.9_51',spec=spec,rmf=rmf,arf=arf,rebin=rebin,rng=rng)
        Xset.restore(file)
        Fit.statMethod = "cstat"
        self.fit_error(error='1.0 2,3,4,5,6,8,10,12,14,det:1',error_calc=True)
        self.result_pack()
        self.savemod(modname=modname)
        self.plotting(modname)         

    def fitting_some_fwdata(self):
        identifiers = ['1000','2000','3000','4000']
        # identifiers = []
        for identifier in identifiers:
            spec = f'{identifier}_center_merged_b1.pi'
            rmf  = f'{identifier}_center_L_without_Lp.rmf'
            arf  = f'{identifier}_center_image_1p8_8keV_1e7.arf'
            if identifier == "2000":
                self.fitting(spec=spec,rmf=rmf,arf=arf,modname=identifier,rng=(2.0,15.0))
            else:
                self.fitting(spec=spec,rmf=rmf,arf=arf,modname=identifier)

    def fitting_some_fwdata(self):
        identifiers = ['1000','2000','3000','4000']
        # identifiers = []
        for identifier in identifiers:
            spec = f'{identifier}_center_merged_b1.pi'
            rmf  = f'{identifier}_center_L_without_Lp.rmf'
            arf  = f'{identifier}_center_image_1p8_8keV_1e7.arf'
            if identifier == "2000":
                self.fitting(spec=spec,rmf=rmf,arf=arf,modname=identifier,rng=(2.0,15.0))
            else:
                self.fitting(spec=spec,rmf=rmf,arf=arf,modname=identifier)

        # self.simultaneous_some_fwdata()

    def model_plot(self,file='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/MnKa_model_plot.xcm'):
        Xset.restore(file)
        Plot('model')
        x = np.array(Plot.x())
        y = np.array(Plot.model(1,1))
        plt.plot(x,y)
        weighted_sum = np.sum(x * y)
        total_counts = np.sum(y)
        center_energy = weighted_sum / total_counts
        plt.vlines(center_energy,0,5,linestyle='-.',color='green')
        print(center_energy)
        plt.show()


    def fitting_bapec(self,modname='test',rebin=1,region='all',identifiers=['1000'],line='w',rng=(2.0,20.0)):
        '''
        fitting bapec model
        region : str : 'all' or 'center' or 'outer'
        '''
        for identifier in identifiers:
            spec = f'{identifier}_{region}_merged_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
        if line == 'w':
            model_file = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_w.xcm'
            err        = '1.0 2,14,15,16,17,18,20,22,det:1'
        if line == 'wz':
            model_file = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm'
            err        = '1.0 2,14,15,16,17,18,20,22,24,26,det:1'
        self.set_init(apecroot="/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9",spec=spec,rmf=rmf,arf=arf,rebin=rebin,rng=rng)
        Xset.restore(model_file)
        Fit.statMethod = "cstat"
        self.fit_error(error=err,error_calc=True)
        self.result_pack()
        self.savemod(modname=modname)
        self.plotting(modname,x_rng=[5.9,6.2],logging=False)       

    def simultaneous_some_fwdata(self,region='center',line='wz'):
        AllData.clear()
        identifiers = ['1000', '2000', '3000', '4000']
        col_list    = ['black', 'red', 'green', 'blue'] 
        for identifier in identifiers:
            spec = f'{identifier}_{region}_merged_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'

            spectrum = Spectrum(spec)
            spectrum.response = rmf
            spectrum.response.arf = arf
            spectrum.multiresponse[1] = rmf

        if line == 'w':
            model_file = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_w.xcm'
            err        = '1.0 2,14,15,16,17,18,20,22,det:1'
        if line == 'wz':
            model_file = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm'
            err        = '1.0 2,14,15,16,17,18,20,22,24,26,det:1'
        Xset.addModelString("APECROOT",'/Users/keitatanaka/apec_modify/del_wz/apec_v3.0.9_51')
        Xset.abund='lpgs'
        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        AllData.ignore("2:**-2.0 15.0-**")
        AllData.ignore("3:**-2.0 17.0-**")
        AllData.ignore("4:**-2.0 17.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        Plot.setRebin(1,1000)
        Xset.restore(model_file)
        Fit.statMethod = "cstat"
        Plot.device = '/xs'
        Plot('data')

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
           spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)


        self.fit_error(err,True)
        Plot.add = True
        for group_num, identifier in enumerate(identifiers):
            Plot('data')
            self.set_xydata(plotgroup=group_num+1)
            mod_y = Plot.model(group_num+1)
            self.get_model_comps(plotgroup=group_num+1)
        #     ax.plot(self.x,self.y,label="data",lw=ps,color=col_list[group_num])
        #     ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=ps,elinewidth=ps,color=col_list[group_num],label=identifiers[group_num])
        #     #ax.step(self.xs,self.ys,color=col_list[group_num],label=identifiers[group_num])
        #     ax.plot(self.xs,mod_y,'-.',lw=ps,color=col_list[group_num])
        #     ax2.errorbar(self.xres,self.yres,yerr=self.yres_e,xerr=self.xres_e,fmt="o",markersize=ps,elinewidth=ps,color=col_list[group_num])
        # ax.set_xlim(2,15)
        # x_rng = (5.9 < self.xs) & (self.xs < 6.2)
        # #yrng = ys[x_rng]
        # #ax.set_ylim(1e-3,np.max(yrng)+0.06)
        # #ax.set_yscale('log')
        # ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        # #ax.set_title(f"{modname}")
        # fig.align_labels()
        # fig.tight_layout()
        # plt.show()
        self.result_pack()
        self.savemod('simultaneous')
        fig.savefig(f"./figure/simultaneous_fitting.pdf",dpi=300,transparent=True)

    def fitting_some_fwdata(self,region='center',line='wz'):
        identifiers = ['1000','2000','3000','4000']
        # identifiers = []
        for identifier in identifiers:
            if identifier == "2000":
                self.fitting_bapec(modname=identifier,identifiers=[identifier],line=line,region=region,rng=(2.0,15.0))
            else:
                self.fitting_bapec(modname=identifier,identifiers=[identifier],line=line,region=region,rng=(2.0,17.0))

        self.simultaneous_some_fwdata(region=region,line=line)

    def fitting_pixel_by_pixel(self,file='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm'):

        self.all_pixels = ["00", "17", "18", "35"]
        for num,pixel in enumerate(self.all_pixels):
            spec = f"1000_PIXEL_{pixel}_merged_b1.pi"
            rmf = f"1000_pix{pixel}_L_without_Lp.rmf"
            arf = f"1000_pix{pixel}_image_1p8_8keV_1e7.arf"
            self.load_and_setdata(spec=spec,rmf=rmf,arf=arf,multiresp=True)
            # self.save_data_pixel(pixel)
            self.set_init(apecroot='/Users/keitatanaka/apec_modify/del_wz/apec_v3.0.9_51',spec=spec,rmf=rmf,arf=arf,rebin=1)
            Xset.restore(file)
            Fit.statMethod = "cstat"
            # Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/resolve_nxb_pheno_continuum_lines.mo')
            # model = AllModels(1)
            # model(1).frozen = False
            self.fit_error(error='1.0 2,3,4,5,6,8,10,12,14,det:1',error_calc=False,detector=True)
            self.result_pack()
            self.savemod(modname=f'pixel{pixel}')
            self.plotting(f'pixel{pixel}')

    def fitting_pixel_by_pixel_55Fe_line(self,line='MnKa'):
        # self.all_pixels = ["00", "17", "18", "35"]
        # self.all_pixels  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        # self.all_pixels = ['07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        # self.all_pixels = ['21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.all_pixels  = ['00']
        if line == 'MnKa':
            rng = (5.8,6.0)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKa_gsmooth_f1.xcm'
        if line == 'MnKb':
            rng = (6.4,6.6)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKb.xcm'
        for num,pixel in enumerate(self.all_pixels):
            spec = f"5000_PIXEL_{pixel}_merged_b1.pi"
            rmf = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag.rmf'
            arf = None
            self.set_init(apecroot='/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9"',spec=spec,rmf=rmf,arf=arf,rebin=3,rng=rng,multiresp=True)
            Xset.restore(model_xcm)
            Fit.statMethod = "cstat"
            Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/resolve_nxb_pheno_continuum_lines.mo')
            self.set_data_range(rng=rng)
            self.fit_error(error='1.0 1,2,4,det:1',error_calc=False,detector=False)
            model = AllModels(1)
            model.zashift.Redshift.frozen = False
            self.fit_error(error='1.0 1,2,4',error_calc=True,detector=True)
            self.result_pack()
            self.savemod(modname=f'pixel{pixel}')
            self.plotting_55Fe_line(f'pixel{pixel}',x_rng=rng,logging=False,bgplot=True,line=line)

    def fitting_all_pixel_55Fe_line(self,line='MnKa',identifier='5000'):
        if line == 'MnKa':
            rng = (5.8,6.0)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKa_gsmooth_f1.xcm'
        if line == 'MnKb':
            rng = (6.4,6.6)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKb.xcm'
        spec = f"{identifier}_all_merged_b1.pi"
        rmf = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag.rmf'
        arf = None
        self.load_and_setdata(spec=spec,rmf=rmf,arf=arf,rng=rng,multiresp=True)
        self.set_init(apecroot='/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9"',spec=spec,rmf=rmf,arf=arf,rebin=3)
        Xset.restore(model_xcm)
        Fit.statMethod = "cstat"
        Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/resolve_nxb_pheno_continuum_lines.mo')
        model = AllModels(1)
        model(1).frozen = False
        self.fit_error(error='1.0 1,2,4,det:1',error_calc=True,detector=True)
        self.result_pack()
        self.savemod(modname=f'{line}')
        self.plotting_55Fe_line(f'{line}',x_rng=rng,logging=True,line=line)

    def result_plot_pixel_by_pixel(self):
        self.all_pixels = ["00", "17", "18", "35"]
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(421) # redshift
        self.ax2 = self.fig.add_subplot(422) # temperature
        self.ax3 = self.fig.add_subplot(423) # abundance
        self.ax4 = self.fig.add_subplot(424) # velocity
        self.ax5 = self.fig.add_subplot(425) # w norm
        self.ax6 = self.fig.add_subplot(426) # z norm
        self.ax7 = self.fig.add_subplot(427) # w/z ratio
        self.ax8 = self.fig.add_subplot(428)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax3.grid(linestyle='dashed')
        self.ax4.grid(linestyle='dashed')
        self.ax5.grid(linestyle='dashed')
        self.ax6.grid(linestyle='dashed')
        self.ax7.grid(linestyle='dashed')
        self.ax8.grid(linestyle='dashed')
        self.ax.set_title(r'Redshift')
        self.ax2.set_title(r'Temperature')
        self.ax3.set_title(r'Abundance')
        self.ax4.set_title(r'Velocity')
        self.ax5.set_title(r'W norm')
        self.ax6.set_title(r'Z norm')
        self.ax7.set_title(r'W FWHM')
        self.ax8.set_title(r'Z FWHM')
        self.ax.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax2.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax3.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax4.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax5.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax6.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax7.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax8.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])

            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        with h5py.File("test.hdf5", 'r') as f:
            for num,pixel in enumerate(self.all_pixels):
                # self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                # self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                # self.ax3.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                # self.ax4.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                # self.ax5.scatter(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                # self.ax6.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                # self.ax7.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...]/f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                print((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3))
                self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5,label='55Fe')
                self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                self.ax3.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                self.ax4.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                self.ax5.errorbar(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                self.ax6.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                self.ax7.errorbar(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        with h5py.File("mxs.hdf5", 'r') as f:
            for num,pixel in enumerate(self.all_pixels):
                # self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                # self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                # self.ax3.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                # self.ax4.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                # self.ax5.scatter(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                # self.ax6.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                # self.ax7.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...]/f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5,label='MXS')
                self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                self.ax3.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                self.ax4.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                self.ax5.errorbar(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                self.ax6.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                self.ax7.errorbar(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)            
            # self.ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))

            self.fig.align_labels()
            self.fig.tight_layout()
            self.fig.savefig(f'./figure/pixel_by_pixel.png',dpi=300,transparent=True)
            plt.show()

    def gain_check_temp_cor(self, check_temp):
        file = '/Volumes/SUNDISK_SSD/PKS_XRISM/gain/xa_rsl_gainpix_20190101v006.fits'
        X = XTools()
        X.loadfits(file)
        factor_name = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        # データ取得
        data = {factor: X.hdul[1].data[factor] for factor in factor_name}
        
        def gain_curve(x, H0, H1, H2, H3, H4, H5, H6, H7):
            return H0 + x*H1 + x**2 * H2 + x**3 * H3 + x**4 * H4 + x**5 * H5 + x**6 * H6 + x**7 * H7

        adu_index = np.argmin(np.abs(np.linspace(0, 40000, 10000) - 2e4))  # ADU=2e4に対応するindex
        gain_ratios = []  # 結果を格納するリスト
        
        for pixel in range(36):
            H_values_49 = [data[factor][0, pixel] for factor in factor_name]
            H_values_50 = [data[factor][1, pixel] for factor in factor_name]
            gain_49 = gain_curve(2e4, *H_values_49)
            gain_50 = gain_curve(2e4, *H_values_50)
            gain_ratio = gain_49 / gain_50
            
            gain_ratios.append((pixel, gain_ratio))
        
        return gain_ratios  # ADU=2e4時のgain_49/gain_50とピクセル番号を返す

    def result_plot_mxs(self, line='MnKa'):
        # ピクセルと対応するgain_49/gain_50の比率を取得
        gain_ratios = self.gain_check_temp_cor(check_temp=49)
        gain_ratios_dict = dict(gain_ratios)  # ピクセル番号をキーにした辞書に変換
        
        self.all_pixels = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.cnt_rate = np.array([3515, 3507, 3209, 19032, 14191, 18636,14525,15297,8378,8805,8371,9280,18888,13686,19192,13417,18715,12774,8626,8140,3194,2826,3653])/155.48e+3
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')

        fwhm = np.array([])
        shift = np.array([])
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        E = 5.9e+3

        with h5py.File(self.savefile, 'r') as f:
            for num, pixel in enumerate(self.all_pixels):
                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    redshift = f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...] * E
                    gain_ratio = gain_ratios_dict.get(num, None)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)

                    if gain_ratio is not None:
                        # gain_ratioとredshiftの相関をプロット
                        self.ax.errorbar(gain_ratio,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                        #self.ax2.errorbar(gain_ratio,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                        #self.ax.scatter(gain_ratio, redshift, label=f"pixel {pixel}", color="black")
                        self.ax.text(gain_ratio, redshift, f"{pixel}", color="red")
                

        self.ax.set_xlabel('Gain Ratio (49mK/50mK, @AUD 2e4)')
        self.ax.set_ylabel('Shift (eV)')
        self.fig.savefig("max_gain_ratio.pdf",dpi=300,transparent=True)
        #self.ax.legend(loc="upper right", fontsize=8)
        plt.show()

    def result_plot_pixel_by_pixel_cal(self,line='MnKa'):
        #self.all_pixels = ["00", "17", "18", "35"]
        #self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.cnt_rate = np.array([3515, 3507, 3209, 19032, 14191, 18636,14525,15297,8378,8805,8371,9280,18888,13686,19192,13417,18715,12774,8626,8140,3194,2826,3653])/155.48e+3
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        # self.ax.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        # self.ax2.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        if line == 'MnKa':
            self.ax.set_ylabel('Shift (eV)\n @5.9 keV')
            E = 5.9e+3
        elif line == 'MnKb':
            self.ax.set_ylabel('Shift (eV)\n @6.4 keV')
            E = 6.4e+3
        self.ax2.set_ylabel('FWHM (eV)\n @6.0 keV')
        self.ax2.set_xlabel('Pixel ID')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File(self.savefile, 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                    self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='red')

        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel_{line}.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_pixel_by_pixel_cal_MnKa_MnKb(self):
        self.all_pixels = ["00", "17", "18", "35"]
        #self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        # self.ax.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        # self.ax2.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax.set_ylabel('Shift (eV)\n @5.9 keV,6.4 keV')
        self.ax2.set_ylabel('FWHM (eV)\n @6.0 keV')
        self.ax2.set_xlabel('Pixel ID')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        E = 5.9e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File('pixel_by_pixel_MnKa.hdf5', 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    if num == 0:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                    else:    
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='black')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='black')

        fwhm = np.array([])
        shift = np.array([])
        E = 6.4e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File('pixel_by_pixel_MnKb.hdf5', 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    if num == 0:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),fmt="o",markersize=5, capsize=5,color='blue',label='MnKb')
                    else:    
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),fmt="o",markersize=5, capsize=5,color='blue')
                    self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),fmt="o",markersize=5, capsize=5,color='blue')
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='blue')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='blue')
        self.ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel_MnKa_MnKb.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_pixel_by_pixel_cal_line(self):
        self.lines = ["MnKa","MnKb"]
        
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax.set_xticks([0,1],labels=self.lines)
        self.ax2.set_xticks([0,1],labels=self.lines)
        self.ax.set_ylabel('Shift (eV) (@5.9keV)')
        self.ax2.set_ylabel('FWHM (eV) (@6.0keV)')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        E = 5.9e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File(self.savefile, 'r') as f:
            for num,line in enumerate(self.lines):

                if f'{line}' in f.keys():
                    self.ax.errorbar(num,f[f'{line}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'{line}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'{line}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax2.errorbar(num,f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                print(line)
                print(f'shift = {f[f'{line}/fitting_result']['1/zashift']['Redshift']['value'][...]*E} + {f[f'{line}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E} - {f[f'{line}/fitting_result']['1/zashift']['Redshift']['em'][...]*E} eV')
                print(f'fwhm = {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm} + {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm} - {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm} eV')
        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_each_fw(self,error_calc=False):
        self.identifiers = ['1000','2000','3000','4000','simultaneous']
        self.fwname      = ['OPEN', 'OBF', 'ND', 'Be', 'All']
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(421) # redshift
        self.ax2 = self.fig.add_subplot(422) # temperature
        self.ax3 = self.fig.add_subplot(423) # abundance
        self.ax4 = self.fig.add_subplot(424) # velocity
        self.ax5 = self.fig.add_subplot(425) # w norm
        self.ax6 = self.fig.add_subplot(426) # z norm
        self.ax7 = self.fig.add_subplot(427) # w/z ratio
        self.ax8 = self.fig.add_subplot(428)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax3.grid(linestyle='dashed')
        self.ax4.grid(linestyle='dashed')
        self.ax5.grid(linestyle='dashed')
        self.ax6.grid(linestyle='dashed')
        self.ax7.grid(linestyle='dashed')
        self.ax8.grid(linestyle='dashed')
        self.ax.set_title(r'Redshift')
        self.ax2.set_title(r'Temperature')
        self.ax3.set_title(r'Abundance')
        self.ax4.set_title(r'Velocity')
        self.ax5.set_title(r'W norm')
        self.ax6.set_title(r'Z norm')
        self.ax7.set_title(r'W FWHM')
        self.ax8.set_title(r'Z FWHM')
        self.ax.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax2.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax3.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax4.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax5.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax6.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax7.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax8.set_xticks([0,1,2,3,4],labels=self.fwname)

            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        with h5py.File("55Fe.hdf5", 'r') as f:
            for num,identifier in enumerate(self.identifiers):
                # self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                # self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                # self.ax3.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                # self.ax4.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                # self.ax5.scatter(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                # self.ax6.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                # self.ax7.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...]/f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                if error_calc==True:
                    self.ax.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5,label='55Fe')
                    self.ax2.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['kT']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['kT']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax3.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax4.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax5.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax6.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'{identifier}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    if f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...] < 0:
                        print(identifier)
                        print(f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...])
                    if -f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...] < 0:
                        print(identifier)
                        print(-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...])
                    else:
                        self.ax7.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                    print(np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)))
                    if f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...] < 0:
                        pass
                    if -f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...] < 0:
                        pass
                    else:
                        self.ax8.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                else:
                    self.ax.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                    self.ax2.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                    self.ax3.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                    self.ax4.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                    self.ax5.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                    self.ax6.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                    self.ax7.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='black')
                    self.ax8.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='black')
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        with h5py.File("mxs.hdf5", 'r') as f:
            for num,identifier in enumerate(self.identifiers):
                # self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                # self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                # self.ax3.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                # self.ax4.scatter(num,f[f'pixel{pixel}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                # self.ax5.scatter(num,f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                # self.ax6.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                # self.ax7.scatter(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['norm']['value'][...]/f[f'pixel{pixel}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                if error_calc==True:
                    self.ax.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5,label='mxs')
                    self.ax2.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['kT']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['kT']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax3.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax4.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax5.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax6.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'{identifier}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax7.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)
                    if f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...] < 0:
                        pass
                    if -f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...] < 0:
                        pass
                    else:
                        self.ax8.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)
                else:
                    self.ax.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],color='blue')
                    self.ax2.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],color='blue')
                    self.ax3.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='blue')
                    self.ax4.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],color='blue')
                    self.ax5.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='blue')
                    self.ax6.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],color='blue')
                    self.ax7.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='blue')
                    self.ax8.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='blue')

            self.fig.align_labels()
            self.fig.tight_layout()
            self.fig.savefig(f'./figure/pixel_by_pixel.png',dpi=300,transparent=True)
            plt.show()

    def fitting2(self,file='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm',spec='1000_center_merged_b1.pi',rmf='1000_center_L_without_Lp.rmf',arf='1000_center_image_1p8_8keV_1e7.arf',apecroot='/Users/keitatanaka/apec_modify/del_w_z/apec_v3.0.9_51',rebin=1,modname='test'):
        self.set_init(apecroot=apecroot,spec=spec,rmf=rmf,arf=arf,rebin=rebin)
        Xset.restore(file)
        Fit.statMethod = "cstat"
        self.fit_error(error='1.0 2,14,15,16,17,18,20,22,24,26,det:1',error_calc=False)
        self.model = AllModels(1)
        self.model.bvapec.Redshift.frozen = False
        self.model.bvapec.Velocity.frozen = False
        self.fit_error(error='1.0 2,14,15,16,17,18,20,22,24,26,det:1',error_calc=True)
        self.result_pack()
        self.savemod(modname=modname)
        self.plotting(modname) 

    def get_parameters_name(self,model):
        parameters = {}
        m = model
        model_components = m.componentNames
        for counter,name in enumerate(model_components):
            print('-----------------')
            print(f'model name: {name}')
            parameters[counter+1]       = {}
            parameters[counter+1][name] = {} 
            model_parameters            = m.__getattribute__(name).parameterNames
            for p in model_parameters:
                print(f'parameter name: {p}')
                parameters[counter+1][name][p] = {}
        return parameters

    def result_pack(self):
        model=self.model
        fit_result = self.get_parameters_name(model)
        counter = 0
        for e in range(0,len(fit_result.keys())):
            mod = str(list(fit_result[e+1].keys())[0])
            print(mod)
            for param in list(fit_result[e+1][mod].keys()):
                print(param)
                fit_result[e+1][f'{mod}'][param]['value'] = AllModels(1)(counter+1).values[0]
                print(model.__getattribute__(mod).__getattribute__(param).link)
                if model.__getattribute__(mod).__getattribute__(param).frozen == False and model.__getattribute__(mod).__getattribute__(param).link == '':
                    fit_result[e+1][mod][param]['em']    = AllModels(1)(counter+1).error[0] - AllModels(1)(counter+1).values[0]
                    fit_result[e+1][mod][param]['ep']    = AllModels(1)(counter+1).error[1] - AllModels(1)(counter+1).values[0]
                    print(fit_result[e+1][mod][param]['value'],fit_result[e+1][mod][param]['em'],fit_result[e+1][mod][param]['ep'])
                else:
                    print(fit_result[e+1][mod][param]['value'])
                counter += 1
        if hasattr(self, 'model_background'):
            fit_result['bg'] = {}
            fit_result['bg']['pow'] = {}
            fit_result['bg']['pow']['norm'] = {}
            fit_result['bg']['pow']['norm']['value'] = AllModels(1,'det')(1).values[0]
            fit_result['bg']['pow']['norm']['em'] =AllModels(1,'det')(1).error[0] - AllModels(1,'det')(1).values[0]
            fit_result['bg']['pow']['norm']['ep'] = AllModels(1,'det')(1).error[1] - AllModels(1,'det')(1).values[0]
        self.fit_result = fit_result
        self.statistic = Fit.statistic
        self.dof = Fit.dof

    def fit_error(self,error='1.0 2,3,4,5,6,8,10,det:1',error_calc=True,detector=True):
        Fit.perform()
        self.model = AllModels(1)
        if detector == True:
            self.model_background = AllModels(1,'det')
        if error_calc == True:
            Fit.error(error)
        Plot('data delc')
        Plot.add = True
        self.y=Plot.model()
        self.ys_comps=[]
        comp_N=1
        while(True):
            try:
                ys_tmp = Plot.addComp(comp_N,1,1)
                comp_N += 1
                # execlude components with only 0
                if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                    continue
                self.ys_comps.append(ys_tmp)
            except:
                break  
        Plot("delc")
        self.xres=Plot.x(1,1)
        self.yres=Plot.y(1,1)
        self.xres_e=Plot.xErr(1,1)
        self.yres_e=Plot.yErr(1,1)      

    def get_model_comps(self,plotgroup=1):
        self.ys_comps=[]
        comp_N=1
        while(True):
            try:
                ys_tmp = Plot.addComp(comp_N,plotgroup,1)
                comp_N += 1
                # execlude components with only 0
                if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                    continue
                self.ys_comps.append(ys_tmp)
            except:
                break  
        Plot("delc")
        self.xres=Plot.x(plotgroup,1)
        self.yres=Plot.y(plotgroup,1)
        self.xres_e=Plot.xErr(plotgroup,1)
        self.yres_e=Plot.yErr(plotgroup,1)
        return self.ys_comps, self.xres, self.yres, self.xres_e, self.yres_e  

    def bapec(self):
        self.initialize('outer_bapec.xcm')
        self.fit_error('1.0 2,3,4,5,6,det:1')
        self.result_pack()
        self.savemod("bapec")

    def bapec_del_line(self,lines,xcm,modname):
        ## re: error
        if 'w' in lines:
            apecroot = '/Users/keitatanaka/apec_modify/del_w/apec_v3.0.9_51'
            error    = '1.0 2,3,4,5,6,8,10,det:1'
            modpar   = ['nH','kT','Z','z','v','norm','He_w_E','He_w_sig','He_w_z','He_w_norm']
        if 'w_z' in lines:
            apecroot = '/Users/keitatanaka/apec_modify/del_w_z/apec_v3.0.9_51'
            error    = '1.0 2,3,4,5,6,8,10,12,14,det:1'
            modpar   = ['nH','kT','Z','z','v','norm','He_w_E','He_w_sig','He_w_z','He_w_norm','He_z_E','He_z_sig','He_z_z','He_z_norm']
        if 'w_z_Lya' in lines:
            apecroot = '/Users/keitatanaka/apec_modify/del_z_w_Lya/apec_v3.0.9_51'
            error    = '1.0 2,3,4,5,6,8,10,12,14,16,18,22,det:1'
            modpar   = ['nH','kT','Z','z','v','norm','He_z_E','He_z_sig','He_z_z','He_z_norm','He_w_E','He_w_sig','He_w_z','He_w_norm','H_Lya2_E','H_Lya2_sig','H_Lya2_z','H_Lya2_norm','H_Lya1_E','H_Lya1_sig','H_Lya1_z','H_Lya1_norm']
        if 'all' in lines:
            apecroot = '/Users/keitatanaka/apec_modify/del_Heab_Lya/apec_v3.0.9_51'
            error    = '1.0 2,3,4,5,6,8,10,14,16,18,20,22,24,26,30,32,38,det:1'
            modpar   = ['nH','kT','Z','z','v','norm','He_z_E','He_z_sig','He_z_z','He_z_norm','He_y_E','He_y_sig','He_y_z','He_y_norm','He_x_E','He_x_sig','He_x_z','He_x_norm','He_w_E','He_w_sig','He_w_z','He_w_norm','H_Lya2_E','H_Lya2_sig','H_Lya2_z','H_Lya2_norm','H_Lya1_E','H_Lya1_sig','H_Lya1_z','H_Lya1_norm','He_b2_E','He_b2_sig','He_b2_z','He_b2_norm','He_b1_E','He_b1_sig','He_b1_z','He_b1_norm']
        # else :
        #     apecroot = "/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9"
        #     error    = '1.0 2,3,5,6,8,10,det:1'
        #     modpar=['nH','kT','Z','z','v','norm']
        print(apecroot)
        self.initialize(fileName=xcm,apecroot=apecroot)
        self.fit_error(error,True)
        self.result_pack()
        self.savemod(modname)        

    def savemod(self,modname):
        print(modname)
        with h5py.File(self.savefile,'a') as f:
            if modname in f.keys():
                del f[modname]
                print('model is deleted')
            f.create_group(modname)
            f.create_group(f'{modname}/fitting_result')
            f[modname].create_dataset("xs",data=self.xs)
            f[modname].create_dataset("ys",data=self.ys)
            f[modname].create_dataset("xe",data=self.xe)
            f[modname].create_dataset("ye",data=self.ye)
            f[modname].create_dataset("y",data=self.y)
            f[modname].create_dataset("yscomps",data=self.ys_comps)
            f[modname].create_dataset("xres",data=self.xres)
            f[modname].create_dataset("yres",data=self.yres)
            f[modname].create_dataset("xres_e",data=self.xres_e)
            f[modname].create_dataset("yres_e",data=self.yres_e)
            f[modname].create_dataset("statistic",data=self.statistic)
            f[modname].create_dataset("dof",data=self.dof)

            for model_number in self.fit_result.keys():
                model_components = list(self.fit_result[model_number].keys())
                for model_component in model_components:
                    for k,v in self.fit_result[model_number][model_component].items():
                        print(model_number,model_component,k,v)
                        if str(model_number) not in f[f'{modname}/fitting_result'].keys():
                            f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}')
                        f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}/{k}')
                        f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset('value',data=v['value'])
                        if f'em' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("em",data=v['em'])
                            print('em =', v['em'])
                        if f'ep' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("ep",data=v['ep'])   
                        print(f[modname]['fitting_result'][str(model_number)][model_component].keys())    

    def plotting(self,modname,error=True,x_rng=[5.9,6.2],logging=False):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.step(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        #ax.plot(xs,ys_comps[1],'-.',label="bapec(Hot)",lw=ps,color='orange')
        ax.plot(xs,ys_comps[0],'-.',label=r"bapec",lw=ps,color='orange')
        #ax.plot(xs,ys_comps[0],'-.',label=r"bapec(Hot)",lw=ps,color='orange')
        ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color='darkblue')
        #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ y",lw=ps,color='mediumblue')
        #ax.plot(xs,ys_comps[3],'-.',label=r"Fe XXV He $\alpha$ x",lw=ps,color='blue')
        ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        #ax.plot(xs,ys_comps[2],label=r"$\rm Fe\ He\alpha \ z$",lw=1,color='green')
        #ax.plot(xs,ys_comps[5],'-.',label=r"$\rm Fe\ H \ Ly\alpha 2$",lw=ps,color='brown')
        #ax.plot(xs,ys_comps[6],'-.',label=r"$\rm Fe\ H \ Ly\alpha 1$",lw=ps,color='salmon')
        #ax.plot(xs,ys_comps[7],'-.',label=r"Fe XXV He $\beta$ 2",lw=ps,color='darkgreen')
        #ax.plot(xs,ys_comps[8],'-.',label=r"Fe XXV He $\beta$ 1",lw=ps,color='green')
        
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-2,np.max(yrng)+1)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        #ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_55Fe_line(self,modname,error=True,x_rng=[5.9,6.2],logging=False,line='MnKa',bgplot=True):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.plot(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        ax.plot(xs,ys_comps[0],'-.',lw=ps,color='blue')
        ax.plot(xs,ys_comps[1],'-.',lw=ps,color='blue')
        ax.plot(xs,ys_comps[2],'-.',lw=ps,color='blue')
        ax.plot(xs,ys_comps[3],'-.',lw=ps,color='blue')
        ax.plot(xs,ys_comps[4],'-.',lw=ps,color='blue')
        bgs = 5
        if line == 'MnKa':
            ax.plot(xs,ys_comps[5],'-.',lw=ps,color='blue')
            ax.plot(xs,ys_comps[6],'-.',lw=ps,color='blue')
            ax.plot(xs,ys_comps[7],'-.',lw=ps,color='blue')
            bgs = 8
    
        if bgplot == True:
            bg = sum([ys_comps[i] for i in range(bgs,len(ys_comps))])

            ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='orange')

        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        if logging==True:
            ax.set_ylim(1e-3,np.max(yrng)+100.0)
        else:
            ax.set_ylim(1e-3,np.max(yrng)+2.0)
        ax2.set_ylim(-5,5)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        ax.legend(fontsize=12,loc='upper right')
        ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
        fig.align_labels()
        fig.tight_layout()
        #plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_cor_data(self,modname,error=True,x_rng=[5.9,6.2],logging=False):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]

        fig = plt.figure(figsize=(9,6))
        ax  = fig.add_subplot(111)
        ls=15
        ps=2


        with h5py.File('55Fe.hdf5','a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="55Fe")

        with h5py.File('mxs.hdf5','a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="blue",label="mxs")
        
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-5,np.max(yrng)+0.06)
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #ax.set_title(f"{modname}")
        ax.set_xlabel("Energy[keV]",fontsize=ls)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        if logging==True:
            ax.set_yscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_raw(self,modname):

        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)
        ls=15
        ps=2
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$")
        ax.set_xlabel("Energy[keV]")
        #ax.set_ylim(1e-5,1.0)
        #ax.set_xlim(2,8)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',which='both',direction='in',width=1.5)
        data_dir = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/analysis/resolve/created_data'
        dirs = ['1000','2000','3000','4000']
        dirname = ['Open','Poly','ND','Be']
        rmf = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/analysis/resolve/created_data/rmf_arf/outer_S.rmf'
        arf = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/analysis/resolve/created_data/rmf_arf/outer_image_1p8_8keV_1e7.arf'
        for e,d in enumerate(dirs):
            AllData.clear()
            Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
            Fit.statMethod = "cstat"
            s = Spectrum(f"{data_dir}/{d}/outer_b1.pi")
            s.response = rmf
            s.response.arf = arf
            AllData.notice("1.8-15.0")
            AllData.ignore("**-1.8 15.0-**")
            Plot.xAxis="keV"
            Fit.query = 'yes'
            Plot.add = True
            Plot.setRebin(2,1000)
            Plot('data')
            self.xs=Plot.x(1,1)
            self.ys=Plot.y(1,1)
            self.xe=Plot.xErr(1,1)
            self.ye=Plot.yErr(1,1)
            ax.plot(self.xs,self.ys,color=cm.jet(e/(len(dirs)-1)),label=f"{dirname[e]}",lw=2)
        ax.legend(fontsize=12)
        ax.set_xlim(5.9,6.3)
        fig.align_labels()
        plt.show()
        fig.savefig(f"{modname}.png",dpi=300)

    def plotting_raw_com(self,modname):
        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)
        ls=15
        ps=2
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$")
        ax.set_xlabel("Energy[keV]")
        #ax.set_ylim(1e-5,1.0)
        #ax.set_xlim(2,8)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',which='both',direction='in',width=1.5)
        data_dir = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/analysis/resolve/Hp_ana/reana_with_log'
        spec = ['center_merged_b1_wLs.pi','outer_merged_b1_wLs.pi']
        rmf = ['center_L_without_Ls.rmf','outer_L_without_Ls.rmf']
        arf = ['center_image_1p8_8keV_1e7.arf','outer_image_1p8_8keV_1e7.arf']
        col = ['black','red']
        lab = ['center','outer']
        for e,i in enumerate(spec):
            AllData.clear()
            Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
            Fit.statMethod = "cstat"
            print(f"{data_dir}/{spec[e]}")
            s = Spectrum(f"{data_dir}/{spec[e]}")
            s.response = f"{data_dir}/{rmf[e]}"
            s.response.arf = f"{data_dir}/{arf[e]}"
            AllData.notice("1.8-15.0")
            AllData.ignore("**-1.8 15.0-**")
            Plot.xAxis="keV"
            Fit.query = 'yes'
            Plot.add = True
            Plot.setRebin(2,1000)
            Plot('data')
            self.xs=Plot.x(1,1)
            self.ys=Plot.y(1,1)
            self.xe=Plot.xErr(1,1)
            self.ye=Plot.yErr(1,1)
            ax.plot(self.xs,self.ys,color=col[e],label=f"{lab[e]}",lw=ps)
        ax.legend(fontsize=12)
        ax.set_xlim(5.9,8)
        # ax.set_yscale('log')
        fig.align_labels()
        plt.show()
        fig.savefig(f"{modname}.png",dpi=300,transparent=True)

    def plotting_xtend(self,modname,cool=True):
        base_name = os.path.basename(self.savefile)
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        #ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax.plot(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        ax.plot(xs,ys_comps[4],'-.',label=r"Source Hot",lw=ps,color='brown')
        if cool == True:
            ax.plot(xs,ys_comps[5],'-.',label=r"Source Cool",lw=ps,color='cyan')
        ax.plot(xs,ys_comps[0],'-.',label="CXB",lw=ps,color='blue')
        ax.plot(xs,ys_comps[1],'-.',label=r"MWH",lw=ps,color='orange')
        ax.plot(xs,ys_comps[2],'-.',label=r"Hot",lw=ps,color='red')
        ax.plot(xs,ys_comps[3],'-.',label=r"LHB",lw=ps,color='green')
        # ax.plot(xs,ys_comps[6],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[7],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[8],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[9],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[10],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[11],'-.',label=r"NXB",lw=ps,color='black')
        # ax.plot(xs,ys_comps[12],'-.',label=r"NXB",lw=ps,color='black')
        print('-----------------')
        print(len(ys_comps))
        if cool == True:
            bg = sum([ys_comps[i] for i in range(6,len(ys_comps)-1)])
        else :
            bg = sum([ys_comps[i] for i in range(5,len(ys_comps))])

        ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='black')
        ax.set_xlim(0.4,15.0)
        ax.set_ylim(1e-5,10)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(modname)
        ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax2.hlines(0,0,15,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        tick_label_fontweight = 'bold'  # 数字の太さ
        tick_line_width = 1.5           # メモリ線の太さ
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=12)
        ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        fig.align_labels()
        def log_formatter(val, pos):
            return f'{val:.1f}'

        ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
        plt.rcParams.update(self.plot_params)
        fig.subplots_adjust(right=0.75)  # 右側にスペースを確保
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.png",dpi=300,transparent=True)

    def com_plotting(self,xcm='center_gaus_without_w.xcm',flux_min=6.0688,flux_max=6.078,flux=False):
        fig=plt.figure(figsize=(8,6))
        gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax=fig.add_subplot(gs1[:,:])
        ax2=fig.add_subplot(gs2[:,:],sharex=ax)
        #fig.subplots_adjust(hspace=.0)
        fig.align_labels()

        #modelname = 'bapec_gaus_without_w'
        modelname = 'del_w'
        with h5py.File(self.savefile,'r') as f:
            print(f[modelname].keys())
            Z = float(f[modelname]['fitting_result']['2']['bapec']['Abundanc']['value'][...])
            kT = float(f[modelname]['fitting_result']['2']['bapec']['kT']['value'][...])
            nH = float(f[modelname]['fitting_result']['1']['TBabs']['nH']['value'][...])
            norm = float(f[modelname]['fitting_result']['2']['bapec']['norm']['value'][...])
            v = float(f[modelname]['fitting_result']['2']['bapec']['Velocity']['value'][...])
            z = float(f[modelname]['fitting_result']['2']['bapec']['Redshift']['value'][...])

            # H_Lya1_E = float(f[modelname]['H_Lya1_E'][...])
            # H_Lya1_sig = float(f[modelname]['H_Lya1_sig'][...])
            # H_Lya1_norm = float(f[modelname]['H_Lya1_norm'][...])
            # H_Lya1_z = float(f[modelname]['H_Lya1_z'][...])
            # H_Lya2_E = float(f[modelname]['H_Lya2_E'][...])
            # H_Lya2_sig = float(f[modelname]['H_Lya2_sig'][...])
            # H_Lya2_norm = float(f[modelname]['H_Lya2_norm'][...])
            # H_Lya2_z = float(f[modelname]['H_Lya2_z'][...])
            He_w_E = float(f[modelname]['fitting_result']['3']['zgauss']['LineE']['value'][...])
            He_w_sig = float(f[modelname]['fitting_result']['3']['zgauss']['Sigma']['value'][...])
            He_w_norm = float(f[modelname]['fitting_result']['3']['zgauss']['norm']['value'][...])
            He_w_z = float(f[modelname]['fitting_result']['3']['zgauss']['Redshift']['value'][...])
            # He_z_E = float(f[modelname]['He_z_E'][...])
            # He_z_sig = float(f[modelname]['He_z_sig'][...])
            # He_z_norm = float(f[modelname]['He_z_norm'][...])
            # He_z_z = float(f[modelname]['He_z_z'][...])

        Xset.restore(fileName=xcm)
        Plot.setRebin(1,1000)
        Plot.xAxis="keV"
        Plot('data')
        xs=Plot.x(1,1)
        ys=Plot.y(1,1)
        xe=Plot.xErr(1,1)
        ye=Plot.yErr(1,1)
        ps = 2
        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        #Xset.addModelString("APECROOT","/Users/keitatanaka/modify/apec_v3.0.9_51")
        m1 = Model("tbabs*bapec")
        m1.setPars(nH,kT,Z,z,v,norm)
        Plot('data')
        x1 = Plot.x(1,1)
        y1 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_apec = AllData(1).flux

        Xset.addModelString("APECROOT","/Users/keitatanaka/apec_modify/del_w/apec_v3.0.9_51")
        Plot('data')
        x3 = Plot.x(1,1)
        y3 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_cont = AllData(1).flux

        #Xset.addModelString("APECNOLINES","no")
        Xset.addModelString("APECROOT","/Users/keitatanaka/apec_modify/del_w/apec_v3.0.9_51")
        m1 = Model("tbabs*bapec+zgauss")
        #m1.setPars(nH,kT,Z,z,v,norm,H_Lya1_E,H_Lya1_sig,H_Lya1_z,H_Lya1_norm,H_Lya2_E,H_Lya2_sig,H_Lya2_z,H_Lya2_norm,He_w_E,He_w_sig,He_w_z,He_w_norm,He_z_E,He_z_sig,He_z_z,He_z_norm)
        m1.setPars(nH,kT,Z,z,v,norm,He_w_E,He_w_sig,He_w_z,He_w_norm)
        Plot('data')
        x2 = Plot.x(1,1)
        y2 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_gaus = AllData(1).flux
        ax.plot(x2,y2,color='black',lw=ps,label='bapec + gaus(w)')
        ax.plot(x1,y1,'-.',color='red',lw=ps, label='bapec expected w')
        ax.plot(x3,y3,'-.',color='orange',lw=ps, label='bapec continuum')
        ax.set_xlim(5.9,6.15)
        ax.set_ylim(1e-8,0.8)
        ax2.set_xlim(5.9,6.15)
        y1 = np.array(y1)
        ax2.plot(x1,y1/y2,lw=ps,color='black')
        ax2.set_ylim(0.5,1.5)
        ax.legend(fontsize=12)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        ax2.set_xlabel("Energy[keV]")
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$")
        ax2.set_ylabel('Ratio')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',which='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',which='both',direction='in',width=1.5)
        print('---------------------')
        print(f'flux calculated {flux_min} - {flux_max} keV')
        print(f'bapec = {flux_apec[0]}')
        print(f'bapec + gaus(w) = {flux_gaus[0]}')
        print(f'continuum = {flux_cont[0]}')
        print(f'ratio = {(flux_gaus[0]-flux_cont[0])/(flux_apec[0]-flux_cont[0])}')
        print(f'effective tau = {-np.log((flux_gaus[0]-flux_cont[0])/(flux_apec[0]-flux_cont[0]))}')
        if flux == True :
            ax.axvspan(flux_min,flux_max,color='gray',alpha=0.5)
            ax2.axvspan(flux_min,flux_max,color='gray',alpha=0.5)
        plt.show()
        if 'center' in xcm:
            print('center')
            sfn = 'center_col.png'
        if 'outer' in xcm:
            print('outer')
            sfn = 'outer_col.png'
        fig.savefig(sfn,dpi=300,transparent=True)

    def com_all_plotting(self,xcm='center_diag.xcm',flux_min=6.0688,flux_max=6.078,flux=False):
        fig=plt.figure(figsize=(8,6))
        gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax=fig.add_subplot(gs1[:,:])
        ax2=fig.add_subplot(gs2[:,:],sharex=ax)
        #fig.subplots_adjust(hspace=.0)
        fig.align_labels()

        #modelname = 'bapec_gaus_without_w'
        modelname = 'all'
        with h5py.File(self.savefile,'r'
        
        
        
        ) as f:
            print(f[modelname].keys())
            Z = float(f[modelname]['fitting_result']['2']['bapec']['Abundanc']['value'][...])
            kT = float(f[modelname]['fitting_result']['2']['bapec']['kT']['value'][...])
            nH = float(f[modelname]['fitting_result']['1']['TBabs']['nH']['value'][...])
            norm = float(f[modelname]['fitting_result']['2']['bapec']['norm']['value'][...])
            v = float(f[modelname]['fitting_result']['2']['bapec']['Velocity']['value'][...])
            z = float(f[modelname]['fitting_result']['2']['bapec']['Redshift']['value'][...])
            He_z_E = float(f[modelname]['fitting_result']['3']['zgauss']['LineE']['value'][...])
            He_z_sig = float(f[modelname]['fitting_result']['3']['zgauss']['Sigma']['value'][...])
            He_z_norm = float(f[modelname]['fitting_result']['3']['zgauss']['norm']['value'][...])
            He_z_z = float(f[modelname]['fitting_result']['3']['zgauss']['Redshift']['value'][...])
            He_y_E = float(f[modelname]['fitting_result']['4']['zgauss_4']['LineE']['value'][...])
            He_y_sig = float(f[modelname]['fitting_result']['4']['zgauss_4']['Sigma']['value'][...])
            He_y_norm = float(f[modelname]['fitting_result']['4']['zgauss_4']['norm']['value'][...])
            He_y_z = float(f[modelname]['fitting_result']['4']['zgauss_4']['Redshift']['value'][...])
            He_x_E = float(f[modelname]['fitting_result']['5']['zgauss_5']['LineE']['value'][...])
            He_x_sig = float(f[modelname]['fitting_result']['5']['zgauss_5']['Sigma']['value'][...])
            He_x_norm = float(f[modelname]['fitting_result']['5']['zgauss_5']['norm']['value'][...])
            He_x_z = float(f[modelname]['fitting_result']['5']['zgauss_5']['Redshift']['value'][...])
            He_w_E = float(f[modelname]['fitting_result']['6']['zgauss_6']['LineE']['value'][...])
            He_w_sig = float(f[modelname]['fitting_result']['6']['zgauss_6']['Sigma']['value'][...])
            He_w_norm = float(f[modelname]['fitting_result']['6']['zgauss_6']['norm']['value'][...])
            He_w_z = float(f[modelname]['fitting_result']['6']['zgauss_6']['Redshift']['value'][...])
            H_Lya2_E = float(f[modelname]['fitting_result']['7']['zgauss_7']['LineE']['value'][...])
            H_Lya2_sig = float(f[modelname]['fitting_result']['7']['zgauss_7']['Sigma']['value'][...])
            H_Lya2_norm = float(f[modelname]['fitting_result']['7']['zgauss_7']['norm']['value'][...])
            H_Lya2_z = float(f[modelname]['fitting_result']['7']['zgauss_7']['Redshift']['value'][...])
            H_Lya1_E = float(f[modelname]['fitting_result']['8']['zgauss_8']['LineE']['value'][...])
            H_Lya1_sig = float(f[modelname]['fitting_result']['8']['zgauss_8']['Sigma']['value'][...])
            H_Lya1_norm = float(f[modelname]['fitting_result']['8']['zgauss_8']['norm']['value'][...])
            H_Lya1_z = float(f[modelname]['fitting_result']['8']['zgauss_8']['Redshift']['value'][...])
            H_Heb2_E = float(f[modelname]['fitting_result']['9']['zgauss_9']['LineE']['value'][...])
            H_Heb2_sig = float(f[modelname]['fitting_result']['9']['zgauss_9']['Sigma']['value'][...])
            H_Heb2_norm = float(f[modelname]['fitting_result']['9']['zgauss_9']['norm']['value'][...])
            H_Heb2_z = float(f[modelname]['fitting_result']['9']['zgauss_9']['Redshift']['value'][...])
            H_Heb1_E = float(f[modelname]['fitting_result']['10']['zgauss_10']['LineE']['value'][...])
            H_Heb1_sig = float(f[modelname]['fitting_result']['10']['zgauss_10']['Sigma']['value'][...])
            H_Heb1_norm = float(f[modelname]['fitting_result']['10']['zgauss_10']['norm']['value'][...])
            H_Heb1_z = float(f[modelname]['fitting_result']['10']['zgauss_10']['Redshift']['value'][...])
            


        Xset.restore(fileName=xcm)
        Plot.setRebin(1,1000)
        Plot.xAxis="keV"
        Plot('data')
        xs=Plot.x(1,1)
        ys=Plot.y(1,1)
        xe=Plot.xErr(1,1)
        ye=Plot.yErr(1,1)
        ps = 2
        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        #Xset.addModelString("APECROOT","/Users/keitatanaka/modify/apec_v3.0.9_51")
        m1 = Model("tbabs*bapec")
        m1.setPars(nH,kT,Z,z,v,norm)
        Plot('data')
        x1 = Plot.x(1,1)
        y1 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_apec = AllData(1).flux

        Xset.addModelString("APECROOT","/Users/keitatanaka/apec_modify/del_Heab_Lya/apec_v3.0.9_51")
        Plot('data')
        x3 = Plot.x(1,1)
        y3 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_cont = AllData(1).flux

        #Xset.addModelString("APECNOLINES","no")
        Xset.addModelString("APECROOT","/Users/keitatanaka/apec_modify/del_Heab_Lya/apec_v3.0.9_51")
        m1 = Model("tbabs*bapec+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss")
        #m1.setPars(nH,kT,Z,z,v,norm,H_Lya1_E,H_Lya1_sig,H_Lya1_z,H_Lya1_norm,H_Lya2_E,H_Lya2_sig,H_Lya2_z,H_Lya2_norm,He_w_E,He_w_sig,He_w_z,He_w_norm,He_z_E,He_z_sig,He_z_z,He_z_norm)
        m1.setPars(nH,kT,Z,z,v,norm,He_z_E,He_z_sig,He_z_z,He_z_norm,He_y_E,He_y_sig,He_y_z,He_y_norm,He_x_E,He_x_sig,He_x_z,He_x_norm,He_w_E,He_w_sig,He_w_z,He_w_norm,H_Lya2_E,H_Lya2_sig,H_Lya2_z,H_Lya2_norm,H_Lya1_E,H_Lya1_sig,H_Lya1_z,H_Lya1_norm,H_Heb2_E,H_Heb2_sig,H_Heb2_z,H_Heb2_norm,H_Heb1_E,H_Heb1_sig,H_Heb1_z,H_Heb1_norm)
        Plot('data')
        x2 = Plot.x(1,1)
        y2 = Plot.model(1,1)
        AllModels.calcFlux(f"{flux_min} {flux_max}")
        flux_gaus = AllData(1).flux
        ax.plot(x2,y2,color='black',lw=ps,label='bapec + gaus')
        ax.plot(x1,y1,'-.',color='red',lw=ps, label='bapec expected')
        #ax.plot(x3,y3,'-.',color='orange',lw=ps, label='bapec continuum')
        ax.set_xlim(5.9,8.0)
        ax.set_ylim(1e-8,0.8)
        ax2.set_xlim(5.9,8.0)
        y1 = np.array(y1)
        ax2.plot(x1,y1/y2,lw=ps,color='black')
        ax2.set_ylim(0.5,2.0)
        ax.legend(fontsize=12)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        ax2.set_xlabel("Energy[keV]")
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$")
        ax2.set_ylabel('Ratio')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',which='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',which='both',direction='in',width=1.5)
        print('---------------------')
        print(f'flux calculated {flux_min} - {flux_max} keV')
        print(f'bapec = {flux_apec[0]}')
        print(f'bapec + gaus(w) = {flux_gaus[0]}')
        print(f'continuum = {flux_cont[0]}')
        print(f'ratio = {(flux_gaus[0]-flux_cont[0])/(flux_apec[0]-flux_cont[0])}')
        print(f'effective tau = {-np.log((flux_gaus[0]-flux_cont[0])/(flux_apec[0]-flux_cont[0]))}')
        if flux == True :
            ax.axvspan(flux_min,flux_max,color='gray',alpha=0.5)
            ax2.axvspan(flux_min,flux_max,color='gray',alpha=0.5)
        plt.show()
        if 'center' in xcm:
            print('center')
            sfn = 'center_col.png'
        if 'outer' in xcm:
            print('outer')
            sfn = 'outer_col.png'
        fig.savefig(sfn,dpi=300,transparent=True)

    def formatter(self, values, significant_digits):
        def format_to_significant_figures(value, significant_digits):
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        def format_to_significant_digit(value, significant_digits):
            if value == 0:
                return 0
            return significant_digits - int(np.floor(np.log10(abs(value)))) - 1

        decimals1 = format_to_significant_digit(values[1], significant_digits)
        decimals2 = format_to_significant_digit(values[2], significant_digits)
        min_decimals = min(decimals1, decimals2)
        # ここでは、有効数字に基づいて数値をフォーマットした後の値をそのまま使用します。
        # 小数点以下の桁数を調整する必要がある場合は、別のアプローチを検討する必要があります。
        formatted_num1 = round(values[0], min_decimals)
        formatted_num2 = round(values[1], min_decimals)
        formatted_num3 = round(values[2], min_decimals)
        return np.array([formatted_num1, formatted_num2, formatted_num3])

    def make_latex_table(self, keyname):
        import h5py
        from jinja2 import Template
        import subprocess
        import numpy as np

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマットする"""
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        # HDF5ファイルの読み込み
        file_path = self.savefile
        keyname = keyname

        with h5py.File(file_path, 'r') as f:
            dataset = f[keyname]['fitting_result']
            number_keys = list(dataset.keys())

            # データの抽出と整理
            data = {}
            for number in number_keys:
                comp = list(dataset[number].keys())[0]
                keys = list(dataset[number][comp].keys())
                for key in keys:
                    v_e = list(dataset[number][comp][key].keys())
                    if 'em' in v_e:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        value = dataset[number][comp][base_key]['value'][()]
                        ep = dataset[number][comp][base_key]['ep'][()]
                        em = dataset[number][comp][base_key]['em'][()]
                        values = self.formatter([value, ep, em], 2)
                        data[modname][base_key] = {
                            'value': values[0],
                            'ep': values[1],
                            'em': values[2]
                        }
                    else:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        data[modname][base_key] = {
                            'value': format_significant_figures(dataset[number][comp][base_key]['value'][()], 5)
                        }

            # statisticとdofの読み込み
            statistic = format_significant_figures(f[keyname]['statistic'][()],6)
            dof = f[keyname]['dof'][()]

        print('-------------')
        print(statistic, dof)
        
        # 各key_numに対するパラメータ数の計算
        param_counts = {key_num: len(values) for key_num, values in data.items()}

        latex_template = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{multirow}
        \renewcommand{\arraystretch}{r}
        \pagestyle{empty}
        \begin{document}
        \renewcommand{\arraystretch}{1.5}\begin{table}[htbp]
            \centering
            \scalebox{1.5}{\begin{tabular}{|c|c|c|}
            \hline
            Model Number & Parameter & Value \\
            \hline
            {% for key_num, values in data.items() %}
                {% set row_count = param_counts[key_num] %}
                {% for key, value in values.items() %}
                    {% if loop.first %}
                        \multirow {{row_count}}{}{}{{ key_num }} & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% else %}
                        & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% endif %}
                {% endfor %}
            \hline
            {% endfor %}
            Statistic & \multicolumn{2}{c|}{${{ statistic }}$} \\
            \hline
            dof & \multicolumn{2}{c|}{${{ dof }}$} \\
            \hline
            \end{tabular}
            }
        \end{table}
        \end{document}
        """



        # Jinja2テンプレートのレンダリング
        template = Template(latex_template)
        latex_content = template.render(data=data, param_counts=param_counts, statistic=statistic, dof=dof)

        # LaTeXソースファイルの書き込み
        with open('table.tex', 'w') as f:
            f.write(latex_content)

        # LaTeXコンパイルを実行してPDFを生成（非対話モードで）
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'])

        print("PDF table has been generated as table.pdf")

    def make_latex_table_xtend(self, keyname):
        import h5py
        from jinja2 import Template
        import subprocess
        import numpy as np

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマットする"""
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        # HDF5ファイルの読み込み
        file_path = self.savefile

        with h5py.File(file_path, 'r') as f:
            dataset = f[keyname]['fitting_result']
            number_keys = list(dataset.keys())

            # データの抽出と整理
            data = {}
            print(number_keys)
            for number in number_keys:
                if number != '1' and number != '2' and number != '3' and number != '4' and number!= '5':
                    comp = list(dataset[number].keys())[0]
                    keys = list(dataset[number][comp].keys())
                    for key in keys:
                        v_e = list(dataset[number][comp][key].keys())
                        if 'em' in v_e:
                            base_key = key
                            modname = f'{number}:{comp}'
                            if modname not in data.keys():
                                data[modname] = {}
                            value = dataset[number][comp][base_key]['value'][()]
                            ep = dataset[number][comp][base_key]['ep'][()]
                            em = dataset[number][comp][base_key]['em'][()]
                            values = self.formatter([value, ep, em], 2)
                            data[modname][base_key] = {
                                'value': values[0],
                                'ep': values[1],
                                'em': values[2]
                            }
                        else:
                            base_key = key
                            modname = f'{number}:{comp}'
                            if modname not in data.keys():
                                data[modname] = {}
                            data[modname][base_key] = {
                                'value': format_significant_figures(dataset[number][comp][base_key]['value'][()], 5)
                            }

            # statisticとdofの値を取得
            statistic = f[keyname]['statistic'][()]
            dof = f[keyname]['dof'][()]

        # 各key_numに対するパラメータ数の計算
        param_counts = {key_num: len(values) for key_num, values in data.items()}

        latex_template = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{multirow}
        \renewcommand{\arraystretch}{r}
        \pagestyle{empty}
        \begin{document}
        \renewcommand{\arraystretch}{1.5}\begin{table}[htbp]
            \centering
            \scalebox{1.0}{\begin{tabular}{|c|c|c|}
            \hline
            Model Number & Parameter & Value \\
            \hline
            {% for key_num, values in data.items() %}
                {% set row_count = param_counts[key_num] %}
                {% for key, value in values.items() %}
                    {% if loop.first %}
                        \multirow {{row_count}}{}{}{{ key_num }} & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% else %}
                        & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% endif %}
                {% endfor %}
            \hline
            {% endfor %}
            Statistic & \multicolumn{2}{c|}{${{ statistic }}$} \\
            \hline
            DOF & \multicolumn{2}{c|}{${{ dof }}$} \\
            \hline
            \end{tabular}
            }
        \end{table}
        \end{document}
        """

        # Jinja2テンプレートのレンダリング
        template = Template(latex_template)
        latex_content = template.render(data=data, param_counts=param_counts, statistic=statistic, dof=dof)

        # LaTeXソースファイルの書き込み
        with open(f'./latex_table/{keyname}_table.tex', 'w') as f:
            f.write(latex_content)

        # LaTeXコンパイルを実行してPDFを生成（非対話モードで）
        subprocess.run(['pdflatex', '-interaction=nonstopmode', f'./latex_table/{keyname}_table.tex'])

        print(f"PDF table has been generated as {keyname}_table.pdf")

    def make_latex_table_xtend_multi(self):
        with h5py.File(self.savefile, 'r') as f:
            keynames = f.keys()
            for keyname in keynames:
                self.make_latex_table_xtend(keyname)

    def make_apec_model(self,del_line='all'):
        # re: cp coco_file from opt/heasoft
        from astropy.io import fits
        from astropy import constants
        from astropy import units
        import numpy as np

        # FITファイルの読み込み
        file_path = '/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9_51_line.fits'
        f = fits.open(file_path)

        # 各HDUを処理
        for i in range(2, 53):
            # データを取得
            data = f[i].data
            element = np.array(data['Element'])
            ion = np.array(data['Ion'])
            up = np.array(data['UpperLev'])
            down = np.array(data['LowerLev'])
            Lambda = np.array(data['Lambda'])

            # エネルギー計算
            E = constants.c * constants.h / (Lambda * units.angstrom)

            # 条件に合致するマスクを作成
            if del_line == 'all':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))| ((ion == 25) & (element == 26) & (up == 2) & (down == 1)) | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  | ((ion == 26) & (element == 26) & (up == 4) & (down == 1))  | ((ion == 25) & (element == 26) & (up == 6) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 5) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 11) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))
                new_file_path = '/Users/keitatanaka/apec_modify/del_Heab_Lya/apec_v3.0.9_51_line.fits'

            elif del_line == 'RS_line':
                mask = ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  | ((ion == 26) & (element == 26) & (up == 4) & (down == 1))  | ((ion == 25) & (element == 26) & (up == 6) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 5) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 11) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))
                new_file_path = '/Users/keitatanaka/apec_modify/del_RS_line/apec_v3.0.9_51_line.fits'                

            elif del_line == 'w_z_Lya':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))| ((ion == 25) & (element == 26) & (up == 2) & (down == 1)) | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  | ((ion == 26) & (element == 26) & (up == 4) & (down == 1)) 
                new_file_path = '/Users/keitatanaka/apec_modify/del_z_w_Lya/apec_v3.0.9_51_line.fits'
            elif del_line == 'w_z':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 2) & (down == 1))
                new_file_path = '/Users/keitatanaka/apec_modify/del_w_z/apec_v3.0.9_51_line.fits'
            elif del_line == 'w':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))
                new_file_path = '/Users/keitatanaka/apec_modify/del_w/apec_v3.0.9_51_line.fits'
            # |   (ion == 25) & (element == 26) & (up == 7) & (down == 1))  w
            # | ((ion == 25) & (element == 26) & (up == 2) & (down == 1))  z
            # | ((ion == 25) & (element == 26) & (up == 6) & (down == 1))  x
            # | ((ion == 25) & (element == 26) & (up == 5) & (down == 1))  y
            # | ((ion == 25) & (element == 26) & (up == 11) & (down == 1))  He b2
            # | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))  He b1
            # | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  Lya2 
            # | ((ion == 26) & (element == 26) & (up == 4) & (down == 1)   Lya1
            # 条件を満たす行を削除
            # [6.63658   6.667548  6.6822977 6.700402  7.8720117 7.88152   6.9518557 6.9730673] keV
            new_data = data[~mask]

            # 新しいHDUを作成（既存のHDUヘッダーを再利用）
            f[i] = fits.BinTableHDU(data=new_data, header=f[i].header)
            print(E[mask==True].to('keV'))
        # 新しいファイルに保存
        
        f.writeto(new_file_path, overwrite=True)

        # ファイルを閉じる
        f.close()

    def out2(self):
        #self.com_plotting()
        self.savefile = "center_mod.hdf5"
        self.bapec_del_line(lines='w_z',xcm='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm',modname='del_w_z')
        self.plotting('del_w_z')
        # self.savefile = "outer_mod.hdf5"
        # self.bapec_del_line(lines='w_z',xcm='/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/bapec_without_wz.xcm',modname='del_w_z')
        # self.plotting('del_w_z')
        # self.savefile = "center.hdf5"
        # self.bapec_del_line(lines='w',xcm='center_gaus_without_w.xcm',modname='del_w')
        # self.plotting('del_w')

    def load_hdf5_file(self,keyname):
        import h5py
        import numpy as np

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマットする"""
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        # HDF5ファイルの読み込み
        file_path = self.savefile
        keyname = keyname

        with h5py.File(file_path, 'r') as f:
            dataset = f[keyname]['fitting_result']
            number_keys = list(dataset.keys())

            # データの抽出と整理
            data = {}
            for number in number_keys:
                comp = list(dataset[number].keys())[0]
                keys = list(dataset[number][comp].keys())
                for key in keys:
                    v_e = list(dataset[number][comp][key].keys())
                    if 'em' in v_e:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        value = dataset[number][comp][base_key]['value'][()]
                        ep = dataset[number][comp][base_key]['ep'][()]
                        em = dataset[number][comp][base_key]['em'][()]
                        values = self.formatter([value, ep, em], 2)
                        data[modname][base_key] = {
                            'value': values[0],
                            'ep': values[1],
                            'em': values[2]
                        }
                    else:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        data[modname][base_key] = {
                            'value': format_significant_figures(dataset[number][comp][base_key]['value'][()], 5)
                        }

            # statisticとdofの読み込み
            statistic = format_significant_figures(f[keyname]['statistic'][()],6)
            dof = f[keyname]['dof'][()]
            self.data = data

    def ratio_plot(self,keyname):
        self.load_hdf5_file(keyname)
        print(self.data.keys())
        z = self.data['3:zgauss']['norm']['value'][...]
        y = self.data['4:zgauss_4']['norm']['value'][...]
        x = self.data['5:zgauss_5']['norm']['value'][...]
        w = self.data['6:zgauss_6']['norm']['value'][...]
        y_z, x_z, w_z = y/z, x/z, w/z
        return z, y, x, w
        # fig, ax = plt.subplots()
        # ax.plot(1,y_z,'o',label='y/z')
        # ax.plot(2,x_z,'o',label='x/z')
        # ax.plot(3,w_z,'o',label='w/z')
        # ax.plot(1,)
        # #ax.set_xlabel('z')
        # ax.set_ylabel('ratio')
        # ax.set_xticks([1,2,3],['y/z','x/z','w/z'])
        # ax.legend()
        # print(f'w/z = {w/z}')
        # plt.show()
        
#for xtend

    def absorbed_vapec_fit(self,spec,rmf,arf):
        AllData.clear()
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("0.4-15.0")
        AllData.ignore("**-0.4 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        # m = Model("Tbabs*vapec")
        # m.setPars(0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1028,3e-2)
        # m.TBabs.nH.frozen=True
        # m.vapec.Fe.frozen=False
        # m.vapec.Ni.frozen=False
        # s.multiresponse[1] = rmf
        # Xset.restore(fileName='xtend_nxb_average.mo')
        # Fit.perform()
        Plot.device="/xs"
        Plot('ld')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)

    def vapec_fit(self,spec,rmf,arf,name):
        AllData.clear()
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("1.0-15.0")
        AllData.ignore("**-1.0 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        m = Model("Tbabs*vapec")
        m.setPars(0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1028,3e-2)
        m.TBabs.nH.frozen=False
        m.vapec.Fe.frozen=False
        m.vapec.Ni.frozen=False
        s.multiresponse[1] = rmf
        Xset.restore(fileName='xtend_nxb_average.mo')
        Fit.perform()
        Plot.device="/xs"
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)
        self.fit_error('1.0 1,2,14,15,17,det:1')
        parameters = self.get_parameters_name(m)
        self.result_pack(modpar=parameters)
        self.savemod(name)

    def xtend_initialize(self,apecroot="/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9",rebin=1):
        AllData.clear()
        Xset.addModelString("APECROOT",apecroot)
        Xset.abund='lpgs'
        AllData.notice("1.0-20.0")
        AllData.ignore("**-1.0 20.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        Plot.setRebin(rebin,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)

    def vapec_sxdb_fit_hot(self,spec,rmf,arf,name,regionfile):
        self.calculate_sxdb_norm(regionfile=regionfile)
        AllData.clear()
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        Fit.statMethod = "cstat"
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("1.0-15.0")
        AllData.ignore("**-1.0 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        # Plot.setRebin(5,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)
        m = Model("Tbabs*(pow+apec+apec)+apec+Tbabs*(vapec)")
        m.setPars(self.halo_nh,1.4,self.CXB_norm,self.MWH_kT,1,0,self.MWH_norm,self.Hot_kT,1,0,self.Hot_norm,self.LHB_kT,1,0,self.LHB_norm,0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1031,3e-2)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=True
        m.apec.kT.frozen = True
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = True
        m.apec_5.kT.frozen = True
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = True
        m.apec_4.kT.frozen = True
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = True
        m.TBabs_6.nH.frozen=False
        m.vapec.Fe.frozen=False
        m.vapec.Ni.frozen=False
        s.multiresponse[1] = rmf
        Xset.restore(fileName='xtend_nxb_average.mo')
        self.fit_error('1.0 16,17,29,30,32,det:1',True)
        self.result_pack()
        self.savemod(name)
        self.plotting_xtend(name,False)

    def vapec_sxdb_fit(self,spec,rmf,arf,name,regionfile):
        self.calculate_sxdb_norm(regionfile=regionfile)
        AllData.clear()
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        Fit.statMethod = "cstat"
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("1.0-15.0")
        AllData.ignore("**-1.0 15.0-**")
        Plot.xAxis = "keV"
        Fit.query  = 'yes'
        Plot.add   = True
        # Plot.setRebin(5,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)
        m = Model("Tbabs*(pow+apec+apec)+apec+Tbabs*(vapec+vapec)")
        m.setPars(self.halo_nh,1.4,self.CXB_norm,self.MWH_kT,1,0,self.MWH_norm,self.Hot_kT,1,0,self.Hot_norm,self.LHB_kT,1,0,self.LHB_norm,0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,1,1,0.1031,3e-2,2.0,1,1,1,1,1,1,1,1,1,1,1,1,1.0,0.1031,3e-2)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=True
        m.apec.kT.frozen = True
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = True
        m.apec_5.kT.frozen = True
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = True
        m.apec_4.kT.frozen = True
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = True
        m.TBabs_6.nH.frozen=False
        m.vapec.Fe.frozen=False
        m.vapec.Ni.frozen=False
        m.vapec_8.Fe.frozen=True
        m.vapec_8.kT.frozen=True
        s.multiresponse[1] = rmf
        Xset.restore(fileName='xtend_nxb_average.mo')
        self.fit_error('1.0 16,17,29,30,32,33,45,48,det:1',False)
        m.vapec_8.kT.frozen=False
        self.fit_error('1.0 16,17,29,30,32,33,45,48,det:1',True)
        self.result_pack()
        self.savemod(name)
        self.plotting_xtend(name)

    def vapec_sxdb_fit_free(self,spec,rmf,arf,name,regionfile):
        self.calculate_sxdb_norm(regionfile=regionfile)
        AllData.clear()
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        Fit.statMethod = "chi"
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("0.4-15.0")
        AllData.ignore("**-0.4 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        # Plot.setRebin(5,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)
        m = Model("Tbabs*(pow+apec+apec)+apec+Tbabs*(vapec+vapec)")
        m.setPars(self.halo_nh,1.4,self.CXB_norm,self.MWH_kT,1,0,self.MWH_norm,self.Hot_kT,1,0,self.Hot_norm,self.LHB_kT,1,0,self.LHB_norm,0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1028,3e-2,2.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1028,3e-2)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=True
        m.apec.kT.frozen = True
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = True
        m.apec_5.kT.frozen = True
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = True
        m.apec_4.kT.frozen = True
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = True
        s.multiresponse[1] = rmf
        Xset.restore(fileName='xtend_nxb_average.mo')
        self.fit_error('1.0 16,17,29,30,32,det:1',False)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=False
        m.apec.kT.frozen = False
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = False
        m.apec_5.kT.frozen = False
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = False
        m.apec_4.kT.frozen = False
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = False
        self.fit_error('1.0 16,17,29,30,32,det:1',False)
        self.result_pack()
        self.savemod(name)
        self.plotting_xtend(name)

    def vapec_sxdb_fit_free_hot(self,spec,rmf,arf,name,regionfile):
        self.calculate_sxdb_norm(regionfile=regionfile)
        AllData.clear()
        Xset.addModelString("APECROOT","/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9")
        Fit.statMethod = "chi"
        s = Spectrum(spec)
        s.response = rmf
        s.response.arf = arf
        AllData.notice("0.4-15.0")
        AllData.ignore("**-0.4 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        # Plot.setRebin(5,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)
        m = Model("Tbabs*(pow+apec+apec)+apec+Tbabs*(vapec)")
        m.setPars(self.halo_nh,1.4,self.CXB_norm,self.MWH_kT,1,0,self.MWH_norm,self.Hot_kT,1,0,self.Hot_norm,self.LHB_kT,1,0,self.LHB_norm,0.35,8.0,1,1,1,1,1,1,1,1,1,1,1,0.3,0.3,0.1028,3e-2)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=True
        m.apec.kT.frozen = True
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = True
        m.apec_5.kT.frozen = True
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = True
        m.apec_4.kT.frozen = True
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = True
        s.multiresponse[1] = rmf
        Xset.restore(fileName='xtend_nxb_average.mo')
        self.fit_error('1.0 16,17,29,30,32,det:1',False)
        m.TBabs.nH.frozen=True
        m.powerlaw.PhoIndex.frozen=True
        m.powerlaw.norm.frozen=True
        m.apec.kT.frozen = False
        m.apec.Abundanc.frozen = True
        m.apec.Redshift.frozen = True
        m.apec.norm.frozen = False
        m.apec_5.kT.frozen = True
        m.apec_5.Abundanc.frozen = True
        m.apec_5.Redshift.frozen = True
        m.apec_5.norm.frozen = False
        m.apec_4.kT.frozen = False
        m.apec_4.Abundanc.frozen = True
        m.apec_4.Redshift.frozen = True
        m.apec_4.norm.frozen = False
        m.vapec.Fe.frozen=False
        m.vapec.Ni.frozen=False
        m.setPars({4:'0.23,0.01,0.1,0.1,0.3,0.3'})
        m.setPars({8:'0.91,0.01,0.5,0.5,1.5,1.5'})
        #m.setPars({4:0.23,0.01,0.1,0.1,0.3,0.3})
        self.fit_error('1.0 16,17,29,30,32,det:1',False)
        self.result_pack()
        self.savemod(name)
        self.plotting_xtend(name,False)

    def calculate_sxdb_norm(self,regionfile,halo_nh=0.435,MWH_kT=0.23,MWH_EM=8.94e-3,Hot_kT=0.91,Hot_EM=8.27e-4,LHB_kT=0.1,LHB_EM=8.17e-3,CXB_SB=9.51):
        from xrism_tools import XRISM_Region
        from xspec_tools import UnitConverter
        R = XRISM_Region()
        FOV = R.area_calculation(regionfile1='PKSXtendRegSky.reg',regionfile2=regionfile,regionfile3='PKS_xtend_region_exclude_Fe_source_pre.reg')
        U = UnitConverter(FOV_manual=FOV)
        self.MWH_norm = U.EM2ApecNorm(MWH_EM,'manual')
        self.Hot_norm = U.EM2ApecNorm(Hot_EM,'manual')
        self.LHB_norm = U.EM2ApecNorm(LHB_EM,'manual')
        self.CXB_norm = U.SB2PowNorm(CXB_SB,'manual')
        self.MWH_kT = MWH_kT
        self.Hot_kT = Hot_kT
        self.LHB_kT = LHB_kT
        self.halo_nh = halo_nh
        
    def multi_plot(self):
        self.vapec_fit('reg1_det_merged.pi','reg1_det.rmf','reg1_det.arf','reg1')
        self.vapec_fit('reg2_det_merged.pi','reg2_det.rmf','reg2_det.arf','reg2')
        self.vapec_fit('reg3_det_merged.pi','reg3_det.rmf','reg3_det.arf','reg3')
        self.vapec_fit('reg4_det_merged.pi','reg4_det.rmf','reg4_det.arf','reg4')
        self.vapec_fit('reg5_det_merged.pi','reg5_det.rmf','reg5_det.arf','reg5')

    def multi_plot_sxdb(self):
        self.vapec_sxdb_fit('reg1_det_merged.pi','reg1_det.rmf','reg1_det.arf','reg1','./make_region/region_1.reg')
        self.vapec_sxdb_fit('reg2_det_merged.pi','reg2_det.rmf','reg2_det.arf','reg2','./make_region/region_2.reg')
        self.vapec_sxdb_fit('reg3_det_merged.pi','reg3_det.rmf','reg3_det.arf','reg3','./make_region/region_3.reg')
        self.vapec_sxdb_fit('reg4_det_merged.pi','reg4_det.rmf','reg4_det.arf','reg4','./make_region/region_4.reg')
        self.vapec_sxdb_fit('reg5_det_merged.pi','reg5_det.rmf','reg5_det.arf','reg5','./make_region/region_5.reg')
        self.make_latex_table_xtend_multi()

    def result_plot(self):
        rad = np.array([0,2.5,6,9.5,13.5,18.5])
        rad_med = np.array([(rad[e]+rad[e+1])/2 for e in range(len(rad)-1)])
        rad_max = np.array([rad[e+1] for e in range(len(rad)-1)]) - rad_med
        rad_min = rad_med - np.array([rad[e] for e in range(len(rad)-1)])
        rad_err = np.vstack((rad_min,rad_max))
        fig,ax = plt.subplots(5,1, sharex='all', figsize=(12,8))

        nh = []
        nh_ep = []
        nh_em = []        
        temp = []
        temp_ep = []
        temp_em = []
        Fe = []
        Fe_ep = []
        Fe_em = []
        Ni = []
        Ni_ep = []
        Ni_em = []
        bg = []
        bg_ep = []
        bg_em = []
        with h5py.File(self.savefile,'r') as f:
            for reg in f.keys():
                nh.append(float(f[reg]['fitting_result/6/TBabs']['nH']['value'][...]))
                nh_ep.append(float(f[reg]['fitting_result/6/TBabs']['nH']['ep'][...]))
                nh_em.append(-float(f[reg]['fitting_result/6/TBabs']['nH']['em'][...]))
                temp.append(float(f[reg]['fitting_result/7/vapec']['kT']['value'][...]))
                temp_ep.append(float(f[reg]['fitting_result/7/vapec']['kT']['ep'][...]))
                temp_em.append(-float(f[reg]['fitting_result/7/vapec']['kT']['em'][...]))
                Fe.append(float(f[reg]['fitting_result/7/vapec']['Fe']['value'][...]))
                Fe_ep.append(float(f[reg]['fitting_result/7/vapec']['Fe']['ep'][...]))
                Fe_em.append(-float(f[reg]['fitting_result/7/vapec']['Fe']['em'][...]))
                Ni.append(float(f[reg]['fitting_result/7/vapec']['Ni']['value'][...]))
                Ni_ep.append(float(f[reg]['fitting_result/7/vapec']['Ni']['ep'][...]))
                Ni_em.append(-float(f[reg]['fitting_result/7/vapec']['Ni']['em'][...]))
        nh_err = np.vstack((nh_ep,nh_em))
        temp_err = np.vstack((temp_ep,temp_em))
        Fe_err = np.vstack((Fe_ep,Fe_em))
        Ni_err = np.vstack((Ni_ep,Ni_em))
        Fe = np.array(Fe)
        Ni = np.array(Ni)
        print(temp)
        print(temp_err)
        temp_err[0][-1] = 0
        ax[0].errorbar(rad_med,nh,xerr=rad_err,yerr=nh_err, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='black')
        ax[0].grid(linestyle='dashed')
        ax[1].errorbar(rad_med,temp,xerr=rad_err,yerr=temp_err, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='black')
        ax[1].grid(linestyle='dashed')
        ax[2].errorbar(rad_med,Fe,xerr=rad_err,yerr=Fe_err, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='black')
        ax[2].grid(linestyle='dashed')
        ax[3].errorbar(rad_med,Ni,xerr=rad_err,yerr=Ni_err, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='black')
        ax[3].grid(linestyle='dashed')
        ax[4].errorbar(rad_med,Fe/Ni,xerr=rad_err, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='black')
        ax[4].grid(linestyle='dashed')
        print(temp)
        print(temp_ep)
        print(temp_em)
        print(Fe)
        print(Ni)
        print(Fe/Ni)
        george = np.loadtxt('george_plot_digit.txt')
        ax[1].scatter(george[:,0],george[:,1],color='red')
        ax[0].set_ylabel(r'$N_H \ (\rm 10^{-21} \ cm^2)$')
        ax[1].set_ylabel(r'$kT \ (\rm keV)$')
        ax[2].set_ylabel(r'${\rm Fe} \ (Z/Z_{sun})$')
        ax[3].set_ylabel(r'${\rm Ni} \ (Z/Z_{sun})$')
        ax[4].set_ylabel(r'$\rm Fe/Ni $')
        ax[4].set_xlabel(r'Radius (arcmin)')
        plt.show()
        fig.savefig('result.png',dpi=300)

    def out(self):
        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(111)
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy[keV]",fontsize=16)
        ax.set_ylabel("Normalized counts s$^{-1}$ keV$^{-1}$",fontsize=16)
        self.absorbed_vapec_fit('reg1_det_merged.pi','reg1_det.rmf','reg1_det_beta_1e7.arf')
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=1,elinewidth=1,color='black',label='0-2.5 arcmin')
        self.absorbed_vapec_fit('reg2_det_merged.pi','reg2_det.rmf','reg2_det_beta_1e7.arf')
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=1,elinewidth=1,color='red',label='2.5-6 arcmin')
        self.absorbed_vapec_fit('reg3_det_merged.pi','reg3_det.rmf','reg3_det_beta_1e7.arf')
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=1,elinewidth=1,color='green',label='6-9.5 arcmin')
        self.absorbed_vapec_fit('reg4_det_merged.pi','reg4_det.rmf','reg4_det_beta_1e7.arf')
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=1,elinewidth=1,color='blue',label='9.5-13.5 arcmin')
        self.absorbed_vapec_fit('reg5_det_merged.pi','reg5_det.rmf','reg5_det_beta_1e7.arf')
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=1,elinewidth=1,color='magenta',label='13.5-18.5 arcmin')
        plt.legend(fontsize=12)
        ax.grid(linestyle='dashed',linewidth=1.5)
        ax.tick_params(axis='both',which='both',direction='in',labelsize=14)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # ここで枠線の太さを設定
        fig.savefig('multi.png',dpi=300)
        plt.show()

class Xtend:

    def __init__(self,savefile):
        self.savefile=savefile

    def initialize(self,fileName,apecroot="/opt/heasoft/heasoft-6.30.1/spectral/modelData/apec_v3.0.9"):
        self.fileName=fileName
        AllData.clear()
        Xset.restore(fileName=self.fileName)
        Xset.addModelString("APECROOT",apecroot)
        AllData.notice("1.0-15.0")
        AllData.ignore("**-1.0 15.0-**")
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Plot.add = True
        # Plot.setRebin(3,1000)
        Plot('data')
        self.xs=Plot.x(1,1)
        self.ys=Plot.y(1,1)
        self.xe=Plot.xErr(1,1)
        self.ye=Plot.yErr(1,1)

class XRISM_Region:

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from shapely.geometry import Point, Polygon, box, MultiPolygon
    from shapely.geometry.polygon import LinearRing
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import Wedge as MplWedge
    import pyregion
    from shapely.affinity import rotate
    from shapely.ops import unary_union
    from shapely.plotting import plot_polygon, plot_points, plot_line
    from astropy.io import fits
    from astropy.wcs import WCS
    from regions import Regions, PixCoord
    import os


    def __init__(self):
        self.str2deg2 = 3282.80635
        self.deg22str = 1 / self.str2deg2


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

    def make_multi_radius_region(self,center_ra=116.881311953645, center_dec=-19.2948862315813):
        # ra = 116.881311953645
        # dec = -19.2948862315813
        # 半径のリスト（内側から順に）
        inner_radii = [2.5, 6, 9.5, 13.5, 18.5] # arcmin


        # リージョンファイルに書き込むヘッダ情報
        header = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        """

        # リージョンファイルに円を書き込む関数
        def write_annulus_to_region_file(file, ra, dec, inner_radius, outer_radius):
            with open(file, 'w') as f:
                f.write(header)
                f.write(f"fk5\nannulus({ra},{dec},{inner_radius*60}\",{outer_radius*60}\")\n")

        def write_circle_to_region_file(file, ra, dec, radius):
            with open(file, 'w') as f:
                f.write(header)
                f.write(f"fk5\ncircle({ra},{dec},{radius*60}\")\n")
            
        # 中抜き円を作成してファイルに書き込む
        for number,inner_radius in enumerate(inner_radii):
            print(number,len(inner_radii))
            if number == 0:
                filename = f'region{number+1}_sky_fk5.reg'
                print('circle')
                write_circle_to_region_file(filename, center_ra, center_dec, inner_radius)
                print(f"Making Region file : {filename}")
            if number < len(inner_radii)-1:
                filename = f'region_{number+2}.reg'
                print('annulus')
                outer_radius = inner_radii[number+1]  
                write_annulus_to_region_file(filename, center_ra, center_dec, inner_radius, outer_radius)
                print(f"Making Region file : {filename}")
            else :
                pass


    def pyregion_to_shapely(self,region_file):
        import pyregion
        from shapely.geometry import Point, Polygon, box, MultiPolygon
        from shapely.affinity import rotate
        from shapely.ops import unary_union
        with open(region_file, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
        
        region = pyregion.parse("\n".join(lines))
        shapes = []
        for shape in region:
            if shape.name == 'circle':
                center, radius = shape.coord_list[:2], shape.coord_list[2]
                circle = Point(center).buffer(radius)
                shapes.append(circle)
            elif shape.name == 'polygon':
                vertices = [(shape.coord_list[i], shape.coord_list[i+1]) for i in range(0, len(shape.coord_list), 2)]
                polygon = Polygon(vertices)
                shapes.append(polygon)
            elif shape.name == 'box':
                x_center, y_center, width, height, angle = shape.coord_list[:5]
                minx = x_center - width / 2
                maxx = x_center + width / 2
                miny = y_center - height / 2
                maxy = y_center + height / 2
                angle = -angle 
                box_shape = box(minx, miny, maxx, maxy)
                rotated_box = rotate(box_shape, angle, origin=(x_center, y_center))
                shapes.append(rotated_box)
            elif shape.name == 'annulus':
                center, inner_radius, outer_radius = shape.coord_list[:2], shape.coord_list[2], shape.coord_list[3]
                outer_circle = Point(center).buffer(outer_radius)
                inner_circle = Point(center).buffer(inner_radius)
                annulus = outer_circle.difference(inner_circle)
                shapes.append(annulus)
        return unary_union(shapes)

    def pyregion_to_shapely_multipolygon(self,region_file):
        import pyregion
        from shapely.geometry import Point, Polygon, box, MultiPolygon
        from shapely.affinity import rotate
        from shapely.ops import unary_union
        with open(region_file, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
        
        region = pyregion.parse("\n".join(lines))
        shapes = []
        for shape in region:
            if shape.name == 'polygon':
                vertices = [(shape.coord_list[i], shape.coord_list[i+1]) for i in range(0, len(shape.coord_list), 2)]
                polygon = Polygon(vertices)
                shapes.append(polygon)
            elif shape.name == 'box':
                x_center, y_center, width, height, angle = shape.coord_list[:5]
                minx = x_center - width / 2
                maxx = x_center + width / 2
                miny = y_center - height / 2
                maxy = y_center + height / 2
                angle = -angle 
                box_shape = box(minx, miny, maxx, maxy)
                rotated_box = rotate(box_shape, angle, origin=(x_center, y_center))
                shapes.append(rotated_box)
        return MultiPolygon(shapes)


    def area_calculation(self,regionfile1='PKSXtendRegSky.reg',regionfile2='region_2.reg',regionfile3='PKS_xtend_region_exclude_Fe_source_pre.reg'):
        from shapely.plotting import plot_polygon, plot_points, plot_line
        region1 = self.pyregion_to_shapely_multipolygon(regionfile1)
        region2 = self.pyregion_to_shapely(regionfile2)
        region3 = self.pyregion_to_shapely(regionfile3)
        region4 = self.pyregion_to_shapely('OoT_reg_exclude_sky.reg')
        # プロット設定
        fig, ax = plt.subplots()
        plot_polygon(region1,label='Xtend',color='blue',add_points=False)
        plot_polygon(region2,label='Source',color='red',add_points=False)
        plot_polygon(region3,label='55Fe',color='green',add_points=False)
        plot_polygon(region4,label='OoT reg',color='green',add_points=False)

        intersection1 = region1.intersection(region2)
        intersection_minus = intersection1.difference(region3)
        intersection_area = intersection1.area
        intersection_minus_area = intersection_minus.area

        print(f"Region 1 area: {region1.area} deg^2")
        print(f"Region 2 area: {region2.area} deg^2")
        print(f"Region 3 area: {region3.area} deg^2")
        print(f"The area of the overlapping region (1 & 2) is: {intersection_area*self.deg22str} str")
        print(f"The area of the overlapping region (1 & 2 minus 3) is: {intersection_minus_area*self.deg22str} str")

        ax.set_title('Region Overlap')
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.grid(linestyle='dashed')
        ax.legend()
        ax.invert_xaxis()

        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        basename = regionfile2.split('/')[-1]
        fig.savefig(f'./figure/{basename}.png',dpi=300)
        return intersection_minus_area*self.deg22str # str


    def convert_physical(self,fits_file='test_sky.fits',region_file_sky='./make_region/region_2.reg',region_file_physical='region_2_physical.reg'):

        from regions import Regions, PixCoord
        with fits.open(fits_file) as hdul:
            wcs = WCS(hdul[0].header)
            print(wcs)
            
            # リージョンファイルをsky座標で読み込む
            reg = Regions.read(region_file_sky)
            print(reg)
            
            # sky座標からphysical座標に変換
            pixel_regions = []
            for region in reg:
                pixel_region = region.to_pixel(wcs)
                pixel_regions.append(pixel_region)
            
            # 物理座標でリージョンファイルを保存
            Regions(pixel_regions).write(region_file_physical, format='ds9')

        print(f"Region file saved in physical coordinates: {region_file_physical}")

    def make_region_and_convert(self):
        self.make_multi_radius_region()
        files = glob.glob('region_*.reg')
        for file in files:
            base_name = file.split('_')[0]
            print(base_name)  # "region2"
            self.convert_physical(region_file_sky=file,region_file_physical=f'{base_name}__sky_physical.reg')

class Cluster:
    def __init__(self):
        from scipy.special import gamma
        # import scipy.constants as const
        from astropy import units as u
        from astropy import constants as c
        # プランク定数（Planck constant）
        h_SI = c.h  # SI単位系のプランク定数（J·s）
        self.h_CGS = h_SI * 1e7  # ジュールをエルグに変換（1 J = 10^7 erg）

        # 古典電子半径（Classical electron radius）
        self.r_e_SI = (c.e.gauss**2 / c.m_e / c.c**2).to('m')  # SI単位系の古典電子半径（m）
        self.r_e_CGS = self.r_e_SI * 1e2  # メートルをセンチメートルに変換（1 m = 10^2 cm）

        # 光速（Speed of light）
        c_SI = c.c  # SI単位系の光速（m/s）
        self.c_CGS = c_SI * 1e2  # メートルをセンチメートルに変換（1 m = 10^2 cm）

        # ボルツマン定数（Boltzmann constant）
        k_B_SI = 1  # SI単位系のボルツマン定数（J/K）
        self.k_B_CGS = k_B_SI * 1e7  # ジュールをエルグに変換（1 J = 10^7 erg）

        self.K2keV = 1/c.e.value/1e+3
        self.keV2K = 1/self.K2keV

        print(f'h = {c.h}')
        print(f're = {self.r_e_SI}')
        print(f'c = {c.c}')
        self.Fe_H = 3.27e-5 # Feの太陽金属量 Lodders+2009 Table 4 Fe/H 
        self.eV2J = c.e.gauss
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

    def kpc_to_arcmin_with_redshift(self,physical_size_kpc, redshift, H0=67.4, Om0=0.315):
        """
        キロパーセク (kpc) を赤方偏移 (redshift) を使用してアークミニッツ (arcmin) に変換する

        :param physical_size_kpc: 物体の物理的な大きさ (kpc)
        :param redshift: 赤方偏移
        :param H0: ハッブル定数 (km/s/Mpc)
        :param Om0: 物質密度パラメータ
        :return: アークミニッツ (arcmin) での角度
        """
        # 宇宙論モデルの設定
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        
        # 光学距離の計算
        angular_diameter_distance = cosmo.angular_diameter_distance(redshift)
        
        # kpcをMpcに変換
        physical_size_Mpc = physical_size_kpc / 1000.0
        
        # 物理的な大きさを角度に変換
        arcmin = (physical_size_Mpc / angular_diameter_distance.value) * (180 / np.pi) * 60
        print(f"{physical_size_Mpc*1e3} kpc の物体が赤方偏移 {redshift} の距離にある場合、角度は {arcmin:.2f} arcmin です")
        return arcmin

    def line_manager(self,Z=26,state='w'):
        import pyatomdb
        # declare the Collisional Ionization Equilibrium session
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(1,10,50)
        if state == 'w':
            z1 = Z-1
            up = 7
        elif state == 'x':
            z1 = Z-1
            up = 6
        elif state == 'y':
            z1 = Z-1
            up = 5
        elif state == 'z':
            z1 = Z-1
            up = 2
        elif state == 'Lya2':
            z1 = Z
            up = 3
        elif state == 'Lya1':
            z1 = Z
            up = 4
        elif state == 'Heb1':
            z1 = Z-1
            up = 13
        elif state == 'Heb2':
            z1 = Z-1
            up = 11
        else:
            print('state is not defined')
        self.z1 = z1
        ldata = sess.return_line_emissivity(kTlist, Z, z1, up, 1)
        self.line_energy = ldata['energy'] * 1e+3
        print('----------------------')
        print('Line Energy')
        print(f'state = {state}')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {up}')
        print(f'line energy = {self.line_energy} eV')
        self.oscillator_strength(Z=Z,z1=z1,upperlev=up,lowerlev=1)
        
    def DeltaE_color_map(self):
        # 温度と速度の範囲を定義
        T_range = np.linspace(0.1, 10, 100)  # keV
        v_range = np.linspace(0, 1000e3, 100)  # m/s

        # 結果を保存するための配列を初期化
        DeltaE_values = np.zeros((len(T_range), len(v_range)))

        # 各温度と速度について DeltaE を計算
        for i, T in enumerate(T_range):
            for j, v in enumerate(v_range):
                DeltaE_values[i, j] = self.DeltaE(T=T, v=v)

        # カラーマップを作成
        T_mesh, v_mesh = np.meshgrid(T_range, v_range)
        plt.figure(figsize=(10, 8))
        plt.contourf(T_mesh, v_mesh*1e-3, DeltaE_values.T, 20, cmap='viridis')
        plt.colorbar(label='Delta E (Energy Change)')
        plt.xlabel('Temperature (keV)')
        plt.ylabel('Velocity (km/s)')
        plt.title('Energy Change as a Function of Temperature and Velocity')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.show()        

    def Sigma_0(self,T=6,E0=6700,v=200e+3,A=55.845):
        self.DeltaE(T=T,E0=E0,v=v,A=A)
        self.sigma_0 = np.sqrt(np.pi)*c.h*self.r_e_SI*c.c*self.f/(self.deltaE*self.eV2J)
        print('----------------------')
        print('Sigma 0')
        print(f'sigma0 = {self.sigma_0}')
        print(f'f = {self.f}')
        return self.sigma_0

    def cal_RS_tau(self,beta=0.572,N0=0.055,rc=62,T=5.5,Z=0.5,v=200,A=55.845,state='w'):
        # PKS beta = 0.572, rc = 62 kpc, N0=0.055, z=0.5, T=6
        # perseus beta = 0.53, rc = 1.26 arcmin, N0-0.05, z=0.01755, 1arcmin = 20.81 kpc
        T0 = 1e+7*self.K2keV
        self.line_manager(Z=26,state=state)
        self.DeltaE(T=T,E0=self.line_energy,v=v,A=A)
        self.Sigma_0(T=T0,E0=self.line_energy,v=0,A=A)
        if state == 'w' or state == 'z':
            self.ion_fraction(25)
        elif state == 'Lya2' or state == 'Lya1':
            self.ion_fraction(26)
        print('----------------------')
        print('ion fraction')
        print(self.iz(T))
        self.vlist = np.arange(0,1000e+3,100)
        self.RS_tau_c = self.RS_tau(beta=beta,N0=N0,rc=rc,T=T,Z=Z,v=self.vlist,A=A)
        # print('----------------------')
        # print('RS tau')
        # plt.plot(self.vlist*1e-3,self.RS_tau_c)
        # plt.semilogy()
        #plt.show()

    def cal_RS_tau_multi(self,beta=0.572,N0=0.055,rc=62,T=5.5,Z=0.5,v=200,A=55.845):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        self.cal_RS_tau(beta=beta,N0=N0,rc=rc,T=T,Z=Z,v=v,A=A,state='w')
        ax.plot(self.vlist*1e-3,self.RS_tau_c,label='Fe w')
        self.cal_RS_tau(beta=beta,N0=N0,rc=rc,T=T,Z=Z,v=v,A=A,state='z')
        ax.plot(self.vlist*1e-3,self.RS_tau_c,label='Fe z')
        self.cal_RS_tau(beta=beta,N0=N0,rc=rc,T=T,Z=Z,v=v,A=A,state='Lya2')
        ax.plot(self.vlist*1e-3,self.RS_tau_c,label='Fe Lya2')
        self.cal_RS_tau(beta=beta,N0=N0,rc=rc,T=T,Z=Z,v=v,A=A,state='Lya1')
        ax.plot(self.vlist*1e-3,self.RS_tau_c,label='Fe Lya1')
        ax.legend(fontsize=15)
        ax.set_xlabel('v [km/s]',fontsize=15)
        ax.set_ylabel(r'$\tau_{RS}$',fontsize=15)
        ax.grid(linestyle='dashed')
        # ax.set_yscale('log')
        ax.hlines(1,0,1000,linestyle='dashed',color='black')
        fig.savefig('RS_tau.png',dpi=300)
        plt.show()

    def RS_tau(self,beta=0.572,N0=0.055,rc=62,T=5.5,Z=0.5,v=200,A=55.845):
        g = 5/3
        mu = 0.62
        c = np.sqrt(g*c.k*T*self.keV2K/(mu*c.m_p))
        M = v/c
        return 2.7*gamma(3*beta/2-1/2)/gamma(3*beta/2)*N0/1e-3*Z*self.iz(T)*rc/250*self.sigma_0*1e+4/1e-16*(T*self.keV2K/1e+7*(1+1.4*A*M**2)**(-1/2))

    def per_ne(self,r):
        ne = ((4.6e-2/(1+(r/55)**2)**1.8) + 4.8e-3/(1+(r/200)**2)**0.87)*(1-0.06*np.exp(-((r-30)/9)**2)*(1+0.04*np.exp(-((r-15)/8)**2)))
        return ne
    
    def per_Te(self,r):
        Te = 7.5*(1+(r/58)**3.5)/(2.45+(r/58)**3.6) * (1.55+(r/20)**2.04)/(1+(r/20)**2)
        return Te
    
    def per_Z(self,r):
        Z = 0.35*(2+1.1*(r/70)**2.7)*(1+(r/1.5))/(1.1+(r/70)**2.7)/(0.8+(r/1.5))
        return Z

    def pks_ne(self,r):
        beta = 0.572
        ne = 0.055*(1+(r/62)**2)**(-3*beta/2)
        return ne
    
    def pks_Te(self,r):
        Te = 135 * ((r/230)**2.4) * (1+r/230)**(-4.1)
        return Te

    def DeltaE(self,T=6,E0=6700,v=200e+3,A=55.845):
        T = T*self.keV2K # keV to K
        M = v/c.c
        self.deltaE = E0*(2*c.k*T/(A*c.m_p*c.c**2)+2*M**2)**(1/2)
        print('----------------------')
        print('Delta E')
        print('E = ' + str(E0) + ' eV')
        print('v = ' + str(v) + ' m/s')
        print('A = ' + str(A))
        print('T = ' + str(T/self.keV2K) + ' keV')
        print(f'delta E = {self.deltaE} eV')
        return self.deltaE

    def delE_r(self,r):
        v = self.v * u.m * u.s**-1
        g = 5/3
        mu = 0.62
        T = self.per_Te(r)*u.keV
        ci = np.sqrt(g*T/(mu*c.m_p)).to('m/s')
        M = v/c.c
        print(f'M = {M}')
        E0 = self.line_energy * u.eV
        A = 55.845
        delE = E0*(2*T/(A*c.m_p*c.c**2)+2*M**2)**(1/2)
        return delE.to('eV')

    def delE_r_pks(self,r):
        v = self.v * u.m * u.s**-1
        g = 5/3
        mu = 0.62
        T = self.pks_Te(r)*u.keV
        ci = np.sqrt(g*T/(mu*c.m_p)).to('m/s')
        M = v/c.c
        print(f'M = {M}')
        E0 = self.line_energy * u.eV
        A = 55.845
        delE = E0*(2*T/(A*c.m_p*c.c**2)+2*M**2)**(1/2)
        return delE.to('eV')

    def integrand_per(self,r):
        # per_Teに依存するDeltaEを計算
        #deltaE = self.DeltaE(T=self.per_Te(r),E0=6700,v=0,A=55.845)
        # per_Z, per_ne, per_Teを使用してtauを計算
        tau = np.sqrt(np.pi) * c.h * self.r_e_SI * c.c * self.f * self.iz(self.per_Te(r)) * self.per_Z(r) * self.Fe_H * self.per_ne(r)*u.cm**-3 /1.16  / (self.delE_r(r))
        print('----------------------')
        print('r = ' + str(r) + ' kpc')
        print(f'Te = {self.per_Te(r)} keV')
        print(f'Z = {self.per_Z(r)}')
        print(f'ne = {self.per_ne(r)} cm^-3')
        print(f'ion fraction = {self.iz(self.per_Te(r))}')
        print(f'delta E = {self.delE_r(r)}')
        print(f'tau = {tau.to('/kpc')}')
        return tau.to('/kpc').value

    def tau_L(self):
        ne=1e-3*u.cm**-3
        Z = 0.5*3.16e-5
        kTe = 3 * u.keV
        delE = 3 * u.eV
        L = 1 * u.Mpc
        f = 0.72
        tau = np.sqrt(np.pi) * c.h * self.r_e_SI * c.c * f *  Z * ne/1.16 * L / (delE)
        return tau.to('')

    def plot_style(self):
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
        self.fig = plt.figure(figsize=(8,6))
        self.ax  = self.fig.add_subplot(111)
        spine_width = 2  # スパインの太さ
        for spine in self.ax.spines.values():
            spine.set_linewidth(spine_width)
        self.ax.tick_params(axis='both',direction='in',width=1.5)
        self.fig.align_labels()

    def RS_tau_per(self, r_range=(0, 830),target='tau'):
        from scipy.integrate import quad
        self.plot_style()
        # インテグレーション
        v_list = np.linspace(0,1000e+3,5)
        # for stage in ['w','z','Lya1','Lya2']:
        for stage in ['w']:
            self.line_manager(Z=26,state=stage)
            self.ion_fraction(self.z1-1)
            tau_list = np.array([])
            for v in v_list:
                self.v = v
                tau_integral, err = quad(self.integrand_per, r_range[0], r_range[1], limit=500000, epsabs=1e-25, epsrel=1e-25)
                
                print("Integral of tau from {} to {} kpc: {}".format(r_range[0], r_range[1], tau_integral))
                print("Error: {}".format(err))
                tau_list = np.append(tau_list,float(tau_integral))
        
            if target == 'tau':
                self.ax.plot(v_list*1e-3,tau_list,label=stage,lw=2)
                self.ax.scatter(v_list*1e-3,tau_list)
        if target == 'tau':
            self.ax.hlines(1,0,1000,colors='black',linestyle='dashed',lw=2)
            self.ax.legend()
            self.ax.set_xlabel('Velocity (km/s)')
            self.ax.set_ylabel('Optical Depth')
            self.fig.savefig('Perseus_optical_depth.png',dpi=300)
            plt.show()
        if target == 'velocity':
            p = 1/(1+0.43*tau_list)
            w_z = p*3.550/1.197


            self.ax.plot(v_list*1e-3,w_z)
            self.ax.scatter(v_list*1e-3,w_z)
            self.ax.hlines(2.45, 0, 1000, colors='black',linestyle='dashed',label='observed ratio')
            self.ax.hlines(3.55/1.197, 0, 1000, colors='orange',linestyle='dashed',label='opticalliy thin ratio')
            self.ax.set_ylabel('w/z ratio')
            self.ax.set_xlabel('Velocity (km/s)')
            self.ax.legend()
            self.fig.savefig('Perseus_wz_ratio_velocity.png',dpi=300)
            
            plt.show()

    def test_met(self,filename,modname):
        with h5py.File(filename) as f:
            z = f[f"{modname}/fitting_result/3/zgauss/norm/value"][...]
            z_ep = f[f"{modname}/fitting_result/3/zgauss/norm/ep"][...]
            z_em = f[f"{modname}/fitting_result/3/zgauss/norm/em"][...]
            w = f[f"{modname}/fitting_result/4/zgauss_4/norm/value"][...]
            w_ep = f[f"{modname}/fitting_result/4/zgauss_4/norm/ep"][...]
            w_em = f[f"{modname}/fitting_result/4/zgauss_4/norm/em"][...]
            T = f[f"{modname}/fitting_result/2/bapec/kT/value"][...]
            w_thin, z_thin = self.emissivity(T)
            w_z_ratio = w/z
            w_z_err_ep   = np.sqrt((w_ep/z)**2+(w*z_ep/z**2)**2) + w_z_ratio
            w_z_err_em   = w_z_ratio + np.sqrt((w_em/z)**2+(w*z_em/z**2)**2)
            w_z_thin     = w_thin/z_thin
            print(w,w_em,w_ep)
            print(z,z_em,z_ep)
            print(w_z_ratio,w_z_err_em,w_z_err_ep)
            print(4.497/1.421)
            print(w_z_thin)
        self.plot_style()
        print(T)
        self.ax.hlines(w_z_ratio, 0, 1000, colors='black',linestyle='dashed',label='observed ratio')
        self.ax.hlines(w_z_err_ep, 0, 1000, colors='black',linestyle='dashed',label='observed ratio')
        self.ax.hlines(w_z_err_em, 0, 1000, colors='black',linestyle='dashed',label='observed ratio')
        self.ax.hlines(w_z_thin, 0, 1000, colors='orange',linestyle='dashed',label='opticalliy thin ratio')
        self.ax.set_ylabel('w/z')
        self.ax.set_xlabel('Velocity (km/s)')
        plt.show()


    def integrand_pks(self,r):
        print('----------------------')
        print('r = ' + str(r) + ' kpc')
        print(f'Te = {self.pks_Te(r)} keV')
        print(f'ne = {self.pks_ne(r)} cm^-3')
        print(f'ion fraction = {self.iz(self.pks_Te(r))}')
        print(f'delta E = {self.delE_r(r)}')
        tau = np.sqrt(np.pi) * c.h * self.r_e_SI * c.c * self.f * self.iz(self.pks_Te(r)) * 0.5 * self.Fe_H  * self.pks_ne(r)*u.cm**-3 /1.16  / (self.delE_r_pks(r))
        print(f'tau = {tau.to('/kpc')}')
        return tau.to('/kpc').value

    def RS_tau_pks(self, r_range=(0, 570),target='tau'):
        from scipy.integrate import quad
        # インテグレーション
        self.plot_style()
        v_list = np.linspace(0,1000e+3,5)
        for stage in ['w']:
        #for stage in ['w','z','x','y','Lya1','Lya2','Heb1','Heb2']:
            self.line_manager(Z=26,state=stage)
            self.ion_fraction(self.z1-1)
            tau_list = np.array([])
            for v in v_list:
                self.v = v
                print(self.integrand_pks(100))
                tau_integral, err = quad(self.integrand_pks, r_range[0], r_range[1], limit=500000, epsabs=1e-25, epsrel=1e-25)
                
                print("Integral of tau from {} to {} kpc: {}".format(r_range[0], r_range[1], tau_integral))
                print("Error: {}".format(err))
                tau_list = np.append(tau_list,float(tau_integral))
            if target == 'tau':
                self.ax.plot(v_list*1e-3,tau_list,label=stage,lw=2)
                self.ax.scatter(v_list*1e-3,tau_list)
        if target == 'tau':
            self.ax.hlines(1,0,1000,colors='black',linestyle='dashed',lw=2)
            self.ax.legend()
            self.ax.set_xlabel('Velocity (km/s)')
            self.ax.set_ylabel('Optical Depth')
            self.fig.savefig('PKS_optical_depth.png',dpi=300,transparent=True)
            plt.show()
        if target == 'velocity':
            p = 1/(1+0.43*tau_list)
            w_z = p*4.497/1.421
            with h5py.File("center.hdf5") as f:
                z = f["simultaneous/fitting_result/3/zgauss/norm/value"][...]
                z_ep = f["simultaneous/fitting_result/3/zgauss/norm/ep"][...]
                z_em = f["simultaneous/fitting_result/3/zgauss/norm/em"][...]
                w = f["simultaneous/fitting_result/4/zgauss_4/norm/value"][...]
                w_ep = f["simultaneous/fitting_result/4/zgauss_4/norm/ep"][...]
                w_em = f["simultaneous/fitting_result/4/zgauss_4/norm/em"][...]
                vobs = f["simultaneous/fitting_result/2/bvapec/Velocity/value"][...]
                vobs_ep = f["simultaneous/fitting_result/2/bvapec/Velocity/ep"][...]
                vobs_em = f["simultaneous/fitting_result/2/bvapec/Velocity/em"][...]
                kT = f["simultaneous/fitting_result/2/bvapec/kT/value"][...]
                w_thin, z_thin = self.emissivity(kT)
                w_z_ratio = w/z
                w_z_err_ep   = np.sqrt((w_ep/z)**2+(w*z_ep/z**2)**2) + w_z_ratio
                w_z_err_em   = w_z_ratio - np.sqrt((w_em/z)**2+(w*z_em/z**2)**2)
                print(w,w_em,w_ep)
                print(z,z_em,z_ep)
                print(w_z_ratio,w_z_err_em,w_z_err_ep)
                print(kT)
                print(w_thin/z_thin)
            
            self.ax.plot(v_list*1e-3,w_z)
            self.ax.scatter(v_list*1e-3,w_z)
            self.ax.hlines(w_z_ratio, 0, 1000, colors='black',linestyle='dashed')
            self.ax.axhspan(w_z_err_ep, w_z_err_em, color='gray',alpha=0.3,label='observed ratio')
            self.ax.axvspan(vobs+vobs_em, vobs+vobs_ep, color='red',alpha=0.3,label='observed velocity')
            self.ax.hlines(w_thin/z_thin, 0, 1000, colors='orange',linestyle='dashed',label='opticalliy thin ratio')
            self.ax.set_ylabel('w/z')
            self.ax.set_xlabel('Velocity (km/s)')
            self.ax.legend()
            self.fig.savefig('PKS_wz_ratio_velocity.png',dpi=300)
            plt.show()

    def ion_fraction(self,stage):

        import pyatomdb
        from scipy import interpolate
        # Boltzmann定数を使って温度の単位変換をします (1 eV = 11604.525 K)
        k_Boltzmann_keV = 8.617333262145e-5  # keV/K

        # 温度範囲を設定します（単位：keV）
        temperatures_keV = np.linspace(0, 20, 100)

        # 温度をケルビンに変換
        temperatures_K = temperatures_keV / k_Boltzmann_keV

        # 元素の原子番号を指定します（Fe の場合は 26）
        Z = 26

        # イオン分布を計算します
        ion_fractions = np.zeros((len(temperatures_K), Z+1))

        # pyatomdb.apec を使用してイオン分布を計算
        for i, T in enumerate(temperatures_keV):
            ion_fractions[i, :] = pyatomdb.apec.return_ionbal(Z, T, teunit='keV', datacache=False)

        # プロットします
        # plt.figure(figsize=(10, 6))
        # for ion_stage in range(20,Z+1):
        #     plt.plot(temperatures_keV, ion_fractions[:, ion_stage], label=f'Fe {ion_stage+1}')
        self.iz  = interpolate.interp1d(temperatures_keV,ion_fractions[:,stage])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Temperature (keV)')
        # plt.ylabel('Ion Fraction')
        # plt.title('Ion Fraction of Fe as a Function of Temperature (keV)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    def oscillator_strength(self,Z=26,z1=25,upperlev=7,lowerlev=1):
        import pyatomdb
        self.f = pyatomdb.atomdb.get_oscillator_strength(Z, z1, upperlev, lowerlev, datacache=False)
        print('----------------------')
        print('Oscillator Strength')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {upperlev}, lowerlev = {lowerlev}')
        print(f'f = {self.f}')
        
    def emissivity(self,T):
        import pyatomdb
        # declare the Collisional Ionization Equilibrium session
        sess = pyatomdb.spectrum.CIESession()

        # set response
        # sess.set_response('aciss_meg1_cy22.grmf', arf = 'aciss_meg1_cy22.garf')

        # get the emissivity of the resonance line of O VII at 5 different temperatures

        #kTlist = numpy.array([1.0,2.0,3.0,4.0])
        kTlist = np.linspace(5.0,6.0,500)

        fig = plt.figure()

        ax= fig.add_subplot(111)

        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(T, 26, 25, up, 1)
            #ax.semilogy(ldata['Te'], ldata['epsilon'],label=up)
            # print(up)
            # print(ldata['energy'])
            if up == 2:
                z = ldata['epsilon']
                print('z')
                print(ldata['epsilon'])
                #ax.semilogy(ldata['Te'], ldata['epsilon'],label="z")
            if up == 7:
                w = ldata['epsilon']
                print('w')
                print(ldata['epsilon'])

        return w, z
                #ax.semilogy(ldata['Te'], ldata['epsilon'],label="w")
        #     if up == 6:
        #         x = ldata['epsilon']
        #         ax.semilogy(ldata['Te'], ldata['epsilon'],label="x")
        #     if up == 5:
        #         y = ldata['epsilon']
        #         ax.semilogy(ldata['Te'], ldata['epsilon'],label="y")
        # Gratio = (x+y+z)/w
        # 7: Resonance w
        # 6: Intercombination x
        # 5: Intercombination y
        # 2: Forbidden z
        # for up in [2,3,4,5,6]:
        #     ldata = sess.return_line_emissivity(kTlist, 26, 26, up, 1)
        #     print(up)
        #     print(ldata['energy'])
        #     if up == 3:
        #         Lya2 = ldata['epsilon']
        #         ax.semilogy(ldata['Te'], ldata['epsilon'],label="Lya1")
        #     if up == 4:
        #         Lya1 = ldata['epsilon']
        #         ax.semilogy(ldata['Te'], ldata['epsilon'],label="Lya2")
        #ax.plot(ldata['Te'], Gratio)
        # print(ldata.keys())
        # ax.set_xlabel("Temperature (keV)")
        # ax.grid(linestyle='dashed')
        #ax.set_ylabel("G ratio")
        #ax.set_xscale("log")
        # ax.set_ylabel("Emissivity (ph cm$^3$ s$^{-1}$)")
        # ax.set_title("Fe XXV, w, x, y, z")
        # plt.legend()
        # plt.show()
        # save image files
        # fig.savefig('Fe_XXVI_emissivity.pdf',dpi=300)

class Information:

    def __init__(self) -> None:
        pass

    def loadfits(self,file):
        hdul = fits.open(file)
        hdul.info()
        return hdul


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
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
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

    def plot_correlation(self, calpix_data, fe55_data, column_name):
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
        self.plot_MXS_GTI(ax)
        fig.tight_layout()
        plt.show()
        fig.savefig(f'{column_name}_temperature_tracking.png', dpi=300, transparent=True)

    def plot_selected_pixels_interp(self, file_paths, column_name, selected_pixels):
        """
        特定のピクセルのみを一つの図にプロットする関数（run_interpと同様のプロットスタイルで）
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # ファイルデータをロードしてプロットする
        data_55Fe, interp_55Fe = self.load_and_plot_fits(file_paths['55Fe'], ax, cm.jet, '-.', False, column_name, '55Fe')
        data_calpix, interp_calpix = self.load_and_plot_fits(file_paths['calpix'], ax, cm.jet, '.', False, column_name, 'calpix')
        data_mxs, interp_mxs = self.load_and_plot_fits(file_paths['mxs'], ax, cm.jet, '-', True, column_name, 'mxs')

        # 選択したピクセルのみをプロット
        common_pixels = set(interp_55Fe["55Fe"].keys()) & set(interp_mxs["mxs"].keys()) & set(selected_pixels)
        for i in selected_pixels:
            if i in common_pixels:
                time_values = np.linspace(self.t0, self.t1, num=1000)
                temp_55Fe = interp_55Fe["55Fe"][i](time_values)
                temp_mxs = interp_mxs["mxs"][i](time_values)
                temp_diff = temp_55Fe - temp_mxs

                if column_name == 'TEMP_FIT':
                    ax.plot(time_values, temp_diff*1e+3, label=f"Pixel {i} Diff", color=cm.jet((int(i) % 6) / 6))
                else:
                    ax.plot(time_values, temp_diff, label=f"Pixel {i} Diff", color=cm.jet((int(i) % 6) / 6))

        ax.legend(fontsize=8)
        ax.set_xlabel("Time")
        if column_name == 'TEMP_FIT':
            ax.set_ylabel('Temperature Difference (mK)')
        elif column_name == 'SHIFT':
            ax.set_ylabel('Energy Shift Difference (eV)')

        # 指定した日時に垂直線を追加
        for date_label, date_value in self.times.items():
            color = 'black'
            if date_label in ['ADR1', 'ADR2']:
                color = 'red'
            elif date_label == 'LED_BRIGHT':
                color = 'blue'
            
            ax.axvline(date_value, color=color, linestyle='--', label=date_label)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'{column_name}_selected_pixels_combined_plot.png', dpi=300, transparent=True)

# InterpolatorPlotterクラスに新しいメソッドを追加
# InterpolatorPlotter.plot_selected_pixels_interp = plot_selected_pixels_interp

# 新しい関数を実行するためのコードを追加
def plot_gain_tracking():
    obsid = '000112000'
    t_start, t_end = '2023-11-08 10:21:00', '2023-11-11 12:01:04'
    dates = {
        'ND': '2023-11-08 23:51:31',
        'Be': '2023-11-09 05:26:31',
        'OBF': '2023-11-09 13:02:51',
        'Fe': '2023-11-09 16:47:51',
        'ADR1': '2023-11-08 20:10:01',
        'ADR2': '2023-11-10 15:10:00',
        'LED_BRIGHT': '2023-11-10 12:01:01'
    }

    file_paths = {
        'mxs': '/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf',
        '55Fe': '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz',
        'calpix': '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/resolve/event_uf/xa000112000rsl_000_pxcal.ghf.gz'
    }

    column_name = 'TEMP_FIT'  # or any other column you want to use

    plotter = InterpolatorPlotter(obsid, t_start, t_end, dates)

    # 特定のピクセルを選択してプロット
    selected_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
    plotter.run_interp(file_paths, column_name)
