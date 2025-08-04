import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy import constants as const
import astropy.io.fits as fits
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.integrate import quad, simpson, romberg, cumtrapz 
# from scipy.special import erfinv
from scipy.optimize import curve_fit
from lmfit import Model
import time
import h5py
from datetime import datetime
import sys
import pyatomdb
from line_profiler import LineProfiler, profile
from dataclasses import dataclass, asdict
# import concurrent.futures
# import xspec
import subprocess
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os
import glob
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
import re
from pathlib import Path

atomic_data_file = 'atomic_data.hdf5'
# Better Comments Extension
# ! : Debug
# ? : For thesis plot code
# TODO: For future work with detailed explanation
# FIXME: For fixing bugs
# // : For dividing sections

'''
Simulation code of the resonance scattering of photons in a cluster.

AtomicDataManager() : make the atomic data from atomdb.
InitialPhotonGenerator() : generate the initial photons position.
RadiationField() : generate the spectrum.
'''


def profile_func(func):
    '''
    Profile function decorator
    This decorator is used to profile the function.
    '''
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.runcall(func, *args, **kwargs)
        profiler.print_stats()
    return wrapper

def profile_func_for_simulation(func):
    '''
    Profile function decorator
    This decorator is used to profile the function.
    '''
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        
        # scatter_generated_photons_divで使用する関数を追加
        profiler.add_function(Simulation.process_single_photon)
        profiler.add_function(Simulation.initialize)
        profiler.add_function(Simulation.generate_photon)
        profiler.add_function(Simulation.handle_thomson_scattering)
        profiler.add_function(Simulation.handle_resonance_scattering)
        profiler.add_function(Simulation._resonance_scatter)
        
        profiler.runcall(func, *args, **kwargs)
        profiler.print_stats()
    return wrapper

class GeneralFunction:
    def __init__(self):
        pass

    def histogram(self, pha, binsize=None):
        """
        Create histogram with automatic binsize if not given.
        
        Parameters:
            pha:        pha data (array-like)
            binsize:    size of bin in eV (optional)
            
        Returns:
            n:      photon count
            bins:   bin edge array
        """
        pha = np.asarray(pha)
        
        if binsize is None:
            # Freedman-Diaconis rule
            q75, q25 = np.percentile(pha, [75 ,25])
            iqr = q75 - q25
            binsize = 2 * iqr / np.cbrt(len(pha))  # optimal bin width
            print(f'binsize: {binsize}') # debug print(binsize)
            if binsize == 0:
                binsize = 1.0  # fallback

        bins = np.arange(np.floor(pha.min()), np.ceil(pha.max()) + binsize, binsize)
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

    def gaussian(self, x, amp, cen, sigma):
        if sigma <= 0:
            return np.full_like(x, np.nan)
        return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

    def gaussian_fitting(self, x, y, p0=None):
        model = Model(self.gaussian)

        if p0 is None:
            p0 = {
                'amp': np.max(y),
                'cen': np.sum(x * y) / np.sum(y),  # 加重平均
                'sigma': np.sqrt(np.sum(y * (x - np.mean(x))**2) / np.sum(y))  # 加重標準偏差
            }

        params = model.make_params(amp=p0['amp'], cen=p0['cen'], sigma=p0['sigma'])
        params['amp'].min = 0
        params['sigma'].min = 1e-6

        result = model.fit(y, params, x=x)

        return result

    def gaussian_fitting_normalized(self, x, y, p0=None):
        from lmfit.models import ExpressionModel
        # 定義：amp = 1 / (sqrt(2π) * sigma)
        expr = "1/(sqrt(2*pi)*sigma) * exp(-0.5*((x - cen)/sigma)**2)"
        model = ExpressionModel(expr)

        if p0 is None:
            p0 = {
                'cen': np.sum(x * y) / np.sum(y),
                'sigma': np.sqrt(np.sum(y * (x - np.mean(x))**2) / np.sum(y))
            }

        params = model.make_params(cen=p0['cen'], sigma=p0['sigma'])
        params['sigma'].min = 1e-6

        result = model.fit(y, params, x=x)
        return result

    def gaussian_fitting_with_plot(self, data, ax=None, title=None, binsize=None):
        n, bins = self.histogram(data, binsize=binsize)
        n, bins = self.group_bin(n, bins, min=1)
        bins_center = 0.5 * (bins[1:] + bins[:-1])

        if ax is None:
            P = PlotManager(subplot_shape=(1, 1))
            ax = P.ax

        ax.step(bins_center, n, where='mid', color="darkblue", label="data")

        try:
            result = self.gaussian_fitting(bins_center, n)

            # フィット曲線描画
            x_fit = np.linspace(bins[0], bins[-1], 500)
            y_fit = result.eval(x=x_fit)
            ax.plot(x_fit, y_fit, color="crimson", label="Gaussian fit")

            # フィット結果と誤差の取得
            amp = result.params['amp'].value
            amp_err = result.params['amp'].stderr or 0
            cen = result.params['cen'].value
            cen_err = result.params['cen'].stderr or 0
            sigma = result.params['sigma'].value
            sigma_err = result.params['sigma'].stderr or 0

            # テキストにして図中に描く
            fit_text = (
                f"$\\mu$ = {cen:.2f} ± {cen_err:.2f}\n"
                f"$\\sigma$ = {sigma:.2f} ± {sigma_err:.2f}\n"
                f"amp = {amp:.1f} ± {amp_err:.1f}"
            )
        except Exception as e:
            # フィットに失敗した場合
            fit_text = "Fit failed"
            print(f"[WARNING] Gaussian fitting failed: {e}")

        # テキスト表示（成功しても失敗しても）
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

        if title is not None:
            ax.set_title(title)

    def gaussian_fitting_with_plot_xy(self, x, y, ax=None, title=None, show_residuals=False):
        if ax is None:
            if show_residuals:
                fig = plt.figure(figsize=(8, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax = fig.add_subplot(gs[0])
                ax_resid = fig.add_subplot(gs[1], sharex=ax)
            else:
                P = PlotManager(subplot_shape=(1, 1))
                ax = P.ax
        else:
            ax_resid = None

        bins_center = x
        n = y
        ax.step(bins_center, n, where='mid', color="black", label="data")
        result = self.gaussian_fitting(bins_center, n)

        # フィット曲線描画
        x_fit = np.linspace(bins_center[0], bins_center[-1], 500)
        y_fit = result.eval(x=x_fit)
        ax.plot(x_fit, y_fit, color="crimson", label="Gaussian fit")

        # フィット結果と誤差
        amp = result.params['amp'].value
        amp_err = result.params['amp'].stderr or 0
        cen = result.params['cen'].value
        cen_err = result.params['cen'].stderr or 0
        sigma = result.params['sigma'].value
        sigma_err = result.params['sigma'].stderr or 0

        fit_text = (
            f"$\\mu$ = {cen:.2f} ± {cen_err:.2f}\n"
            f"$\\sigma$ = {sigma:.2f} ± {sigma_err:.2f}\n"
            f"amp = {amp:.3f} ± {amp_err:.3f}"
        )
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

        if title is not None:
            ax.set_title(title)

        ax.legend(loc="upper right")

        # 残差プロット
        if show_residuals:
            y_fit_data = result.eval(x=bins_center)
            residuals = n - y_fit_data
            ax_resid.step(bins_center, residuals, where='mid', color="gray")
            ax_resid.axhline(0, color="black", linestyle="--", linewidth=1)
            ax_resid.set_ylabel("Residuals")
            ax_resid.set_xlabel("x")
            ax_resid.tick_params(axis='x', direction='in', width=1.5)
            ax_resid.tick_params(axis='y', direction='in', width=1.5)

        # 軸ラベル（共通）
        ax.set_ylabel("Counts")
        if not show_residuals:
            ax.set_xlabel("x")

    def ensure_unit(self, value, default_unit):
        """
        Ensure that the value has the specified unit.
        Parameters
        ----------
        value : float or Quantity
            The value to check.
        default_unit : Quantity
            The default unit to apply if the value does not have a unit.
        Returns
        -------
        value : Quantity
            The value with the specified unit.
        """
        if hasattr(value, 'unit'):
            return value
        else:
            return value * default_unit


    def multi(self):
        P = PlotManager((2,4),(12.8,8))
        print(P.axes)
        for i in range(0,8):
            data = np.random.normal(0, 1, 1000)
            self.gaussian_fitting_with_plot(data=data, ax=P.axes[i])
        P.fig.tight_layout()
        plt.show()

@dataclass
class DataAnalysisInfo:
    '''
    Data Analysis Information dataclass
    This class is used to store the data and analysis information.
    You should use this class with the ClusterManager class.
    '''
    ne_reference: str                  = 'None'
    T_reference: str                   = 'None'
    Z_reference: str                   = 'None'
    ne_interpolate_method: str         = 'None'
    ne_square_interpolate_method: str  = 'None'
    T_interpolate_method: str          = 'None'
    Z_interpolate_method: str          = 'None'
    cluster_name: str                  = 'None'
    r: np.ndarray                      = None
    ne: np.ndarray                     = None
    T: np.ndarray                      = None
    Z: np.ndarray                      = None
    f_ne: interp1d                     = None
    f_ne_square: interp1d              = None
    f_T: interp1d                      = None
    f_Z: interp1d                      = None

    def print_info(self):
        '''
        Print the data analysis information
        '''
        info_dict = asdict(self)
        print('\n' + '='*40)
        print('       Data Analysis Information')
        print('='*40)
        for key, value in info_dict.items():
            print(f"{key:30}: {value}")
        print('='*40 + '\n')

class ClusterManager:
    '''
    Cluster Manager class
    This class is used to manage the cluster data.
    Cluster data is stored in the DataAnalysisInfo class.

    Parameters
    ----------
    cluster_name : str
        The name of the cluster to manage.

    Attributes
    ----------
    Cluster : object
        The cluster object to manage.

    '''
    def __init__(self,cluster_name='None',rmin=0.1,rmax=1000,division=50,load_data=True,method='nearest',setting='xrism', ne=0.01, kT=5.0, Z=0.4) -> None:
        self.cluster_name = cluster_name
        self.rmin         = rmin
        self.rmax         = rmax
        self.division     = division
        self.load_data    = load_data
        self.method       = method
        self.setting      = setting
        self.dummy_ne     = ne
        self.dummy_kT     = kT
        self.dummy_Z      = Z
        self.load_cluster()

    def load_cluster(self):
        if self.cluster_name == 'perseus':
            self.Cluster     = Perseus()
            if self.load_data == True:
                self.Cluster.make_perseus_setting(self.method, self.setting)
            
        elif self.cluster_name == 'pks':
            self.Cluster = PKS()
            if self.load_data == True:
                self.Cluster.load_pks_setting()

        elif self.cluster_name == 'abell478':
            self.Cluster = Abell478()
            if self.load_data == True:
                self.Cluster.load_setting()

        elif self.cluster_name == 'dummy':
            self.Cluster = DummyCluster()
            if self.load_data == True:
                self.Cluster.load_setting(ne=self.dummy_ne, kT=self.dummy_kT, Z=self.dummy_Z)
        
        else:
            raise ValueError('Cluster name is not defined.')

    def divide_cluster_sphere_logscale(self):
        r = np.logspace(np.log10(self.rmin), np.log10(self.rmax), int(self.division))
        return r
    
    def rebin_data_logscale(self, data):
        data_counts, bins_edge = np.histogram(data, bins=self.divide_cluster_sphere_logscale())
        # np.clipでインデックスをbins_edgeの範囲に収める
        rebin_data = np.array([bins_edge[np.clip(np.digitize(value, bins_edge), 0, len(bins_edge) - 1)] for value in data])
        return rebin_data

    def divide_cluster_sphere_linear(self):
        if self.cluster_name == 'perseus':
            r_mask = (self.Cluster.data.r >= self.rmin) & (self.Cluster.data.r <= self.rmax)
            return self.Cluster.data.r[r_mask]
        elif self.cluster_name == 'pks' or self.cluster_name == 'abell478':
            print('PKS')
            print(np.linspace(self.rmin, self.rmax, self.division))
            print('rmax = ', self.rmax)
            print('rmin = ', self.rmin)
            print('division = ', self.division)
            return np.linspace(self.rmin, self.rmax, self.division)
        elif self.cluster_name == 'dummy':
            print('Dummy')
            print(np.linspace(self.rmin, self.rmax, self.division))
            print('rmax = ', self.rmax)
            print('rmin = ', self.rmin)
            print('division = ', self.division)
            return np.linspace(self.rmin, self.rmax, self.division)
    
    def assign_nearest_value(self, data, ref_values):
        """
        各要素 data[i] を、ref_values の中で最も近い値に置き換える（高速・低メモリ）

        Parameters
        ----------
        data : ndarray of shape (N,)
            任意のデータ（1次元）
        ref_values : ndarray of shape (M,)
            置き換え候補の配列（昇順ソートされていること）

        Returns
        -------
        nearest_data : ndarray of shape (N,)
            各 data[i] に最も近い ref_values の値
        """
        ref_values = np.asarray(ref_values).ravel()
        data = np.asarray(data).ravel()

        indices = np.searchsorted(ref_values, data, side='left')
        indices = np.clip(indices, 1, len(ref_values) - 1)

        left = ref_values[indices - 1]
        right = ref_values[indices]

        nearest = np.where(np.abs(data - left) < np.abs(data - right), left, right)
        return nearest

class Perseus:
    '''
    Perseus Cluster class
    This class is used to manage the Perseus cluster data.

    Parameters
    ----------
    data : DataAnalysisInfo
        The data analysis information to store.

    '''
    def __init__(self, data: DataAnalysisInfo = None) -> None:
        if data is None:
            data = DataAnalysisInfo()
        self.data = data
        self.data.cluster_name = 'perseus'
        self.redshift = 0.017284
        self.arcmin2kpc = 20.818 # from furukawa san s setting
        self.local_directory = os.path.expandvars(
    "$DROPBOX/share/work/astronomy/PKS/rs_simulation/cluster_setting_files/perseus"
)

    def load_setting(self, filename='setting.yaml'):
        try:
            f = np.loadtxt(filename)
            print(f.ndim)
            if f.ndim == 1:  # 一次元の場合
                return f, None  # 一列のみで返す
            elif f.shape[1] == 1:  # 一列のみ
                return f[:, 0], None  # 一列のみで返す
            elif f.shape[1] == 2:  # 二列以上
                return f[:, 0], f[:, 1]  # 最初の二列を返す
            elif f.shape[1] == 3:  # 三列以上
                return f[:, 0], f[:, 1], f[:, 2]
            else:
                return None, None  # データが空の場合
        except Exception as e:
            print(f"Error loading settings from {filename}: {e}")
            return None, None
    
    def load_perseus_setting(self, setting='xrism'):
        '''
        Load the setting of the Perseus cluster from the files.
        '''
        if setting == 'chandra':
            r,_,Z  = self.load_setting(f'{self.local_directory}/Chandra_sph_sym_abund_arcmin_hokan.dat')
            r,_,ne = self.load_setting(f'{self.local_directory}/Chandra_sph_sym_dens_arcmin_hokan.dat')
            r,_,T  = self.load_setting(f'{self.local_directory}/Chandra_sph_sym_temp_arcmin_hokan.dat')
            r *= self.arcmin2kpc
            #Z *= 4.68/3.27     # for abundance conversion from Anders & Grevesse (1989) to Lodders et al. (2009)
            self.data.ne_reference = 'Chandra_sph_sym_abund_arcmin_hokan.dat'
            self.data.T_reference  = 'temperature_profile.dat'
            self.data.Z_reference  = 'abundance_profile.dat'
        else:
            r, Z  = self.load_setting(f'{self.local_directory}/abundance_profile.dat')
            r, ne = self.load_setting(f'{self.local_directory}/electron_density_profile.dat')
            r, T  = self.load_setting(f'{self.local_directory}/temperature_profile.dat')
            Z *= 4.68/3.27     # for abundance conversion from Anders & Grevesse (1989) to Lodders et al. (2009)
            self.data.ne_reference = 'electron_density_profile.dat'
            self.data.T_reference  = 'temperature_profile.dat'
            self.data.Z_reference  = 'abundance_profile.dat'
        self.data.r            = r
        self.data.ne           = ne 
        self.data.T            = T
        self.data.Z            = Z
    
    def interpolate_nearest(self, x, y):
        f = interp1d(x, y, kind='nearest', fill_value='extrapolate')
        return f

    def interpolate_linear(self, x, y):
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        return f

    def make_perseus_setting(self, method='nearest', setting='xrism'):
        '''
        Make the setting of the Perseus cluster.
        '''
        self.load_perseus_setting(setting=setting)
        if method == 'nearest':
            self.data.f_ne                    = self.interpolate_nearest(self.data.r, self.data.ne)
            self.data.ne_interpolate_method   = '1d_nearest'
            self.data.f_ne_square             = self.interpolate_nearest(self.data.r, self.data.ne**2) 
            self.data.ne_square_interpolate_method = '1d_nearest'
            self.data.f_T                     = self.interpolate_nearest(self.data.r, self.data.T)
            self.data.T_interpolate_method    = '1d_nearest'
            self.data.f_Z                     = self.interpolate_nearest(self.data.r, self.data.Z)
            self.data.Z_interpolate_method    = '1d_nearest'
        if method == 'linear':
            self.data.f_ne                    = self.interpolate_linear(self.data.r, self.data.ne)
            self.data.ne_interpolate_method   = '1d_linear'
            self.data.f_ne_square             = self.interpolate_linear(self.data.r, self.data.ne**2) 
            self.data.ne_square_interpolate_method = '1d_linear'
            self.data.f_T                     = self.interpolate_linear(self.data.r, self.data.T)
            self.data.T_interpolate_method    = '1d_linear'
            self.data.f_Z                     = self.interpolate_linear(self.data.r, self.data.Z)
            self.data.Z_interpolate_method    = '1d_linear'
        self.data.print_info()
        return self.data.r, self.data.ne, self.data.T, self.data.Z, self.data.ne**2

    def _plot_radial_ion_fraction(self):
        atomc_data = AtomicDataManager(atomic_data_file,'3.0.8')
        FeXXVI = atomc_data.load_ion_fraction(26, 26,True)
        FeXXV = atomc_data.load_ion_fraction(26, 25,True)
        FeXXIV = atomc_data.load_ion_fraction(26, 24,True)

        plt.plot(self.data.r, FeXXVI(self.data.T), label='Fe XXVI')
        plt.plot(self.data.r, FeXXV(self.data.T), label='Fe XXV')
        plt.plot(self.data.r, FeXXIV(self.data.T), label='Fe XXIV')
        f = np.loadtxt('fraction_H_file.dat')
        plt.plot(f[:,0], f[:,1], label='Hydrogen Fraction')
        f = np.loadtxt('fraction_He_file.dat')
        plt.plot(f[:,0], f[:,1], label='Helium Fraction')
        f = np.loadtxt('fraction_Li_file.dat')
        plt.plot(f[:,0], f[:,1], label='Lithium Fraction')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Ion Fraction')
        plt.legend()
        plt.show()

    def _plot_perseus_setting(self):
        R = RadiationField(0,0,0)
        R.plot_style('triple')
        self.load_perseus_setting(setting='xrism')
        R.ax3.plot(self.data.r, self.data.Z, label='Abundance',color='black')
        R.ax.plot(self.data.r, self.data.ne, label='Electron Density',color='black')
        R.ax2.plot(self.data.r, self.data.T, label='Temperature',color='black')
        R.ax3.scatter(self.data.r, self.data.Z, label='Abundance',color='black')
        R.ax.scatter(self.data.r, self.data.ne, label='Electron Density',color='black')
        R.ax2.scatter(self.data.r, self.data.T, label='Temperature',color='black')
        # R.ax.set_xlabel('Radius (kpc)')
        # R.ax2.set_xlabel('Radius (kpc)')
        R.ax3.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel(r'$n_e \ (\rm cm^{-3})$')
        R.ax2.set_ylabel(r'$kT \ (\rm keV)$')
        R.ax3.set_ylabel(r'$Z$')
        #R.ax.set_title('Perseus Cluster Setting')
        R.ax.set_xscale('log')
        R.ax.set_yscale('log')
        R.ax.set_ylim(1e-4,0.1)
        R.ax.set_xlim(1,1e+3)
        # R.ax.legend()
        plt.show()
        R.fig.savefig('perseus_setting.pdf', dpi=300, transparent=True)

class PKS:
    def __init__(self, data: DataAnalysisInfo = None) -> None:
        if data is None:
            data = DataAnalysisInfo()
        self.data = data
        self.data.cluster_name = 'pks'
        self.redshift = 0.1028
        self.local_directory = os.path.expandvars(
    "$DROPBOX/share/work/astronomy/PKS/rs_simulation/cluster_setting_files/pks0745"
)

    def load_setting(self, filename='setting.yaml'):
        try:
            f = np.loadtxt(filename)
            print(f.ndim)
            if f.ndim == 1:  # 一次元の場合
                return f, None  # 一列のみで返す
            elif f.shape[1] == 1:  # 一列のみ
                return f[:, 0], None  # 一列のみで返す
            elif f.shape[1] == 2:  # 二列以上
                return f[:, 0], f[:, 1]  # 最初の二列を返す
            elif f.shape[1] == 3:  # 三列以上
                return f[:, 0], f[:, 1], f[:, 2]
            else:
                return None, None  # データが空の場合
        except Exception as e:
            print(f"Error loading settings from {filename}: {e}")
            return None, None
    
    def load_pks_setting(self, method='sanders_george_digit'):
        if method == 'beta_model_chen':
            r = np.linspace(0, 1000, 1000)
            ne = self.beta_model_chen(r)
            T = self.george_T(r)
            Z = self.const_Z(r)
            self.data.ne_reference = 'beta_model_chen'
            self.data.T_reference = 'george'
            self.data.Z_reference = 'constant_0.4'
            self.data.r = r
            self.data.ne = ne
            self.data.T = T
            self.data.Z = Z
            f_ne = interp1d(r, ne, kind='nearest', fill_value='extrapolate')
            f_T = interp1d(r, T, kind='nearest', fill_value='extrapolate')
            self.data.f_ne = f_ne
            self.data.ne_interpolate_method = '1d_nearest'
            self.data.f_ne_square = interp1d(r, ne**2, kind='nearest', fill_value='extrapolate')
            self.data.ne_square_interpolate_method = '1d_nearest'
            self.data.f_T = f_T
            self.data.T_interpolate_method = '1d_nearest'
            self.data.f_Z = interp1d(r, Z, kind='nearest', fill_value='extrapolate')
            self.data.Z_interpolate_method = '1d_nearest'
            self.data.print_info()

        elif method == 'sanders_digit':
            r = np.linspace(0, 500, 400)
            r_ne, ne = self.load_setting('sanders_ne_digit.txt')
            r_T, T = self.load_setting('sanders_Te_digit.txt')
            #r_Z, Z = self.load_setting('sanders_Z_digit.txt')
            print(r_ne, ne)
            r_Z = r_ne
            Z = np.full(len(r_Z), 0.4)
            Z *= 4.68/3.27     # for abundance conversion from Anders & Grevesse (1989) to Lodders et al. (2009)
            self.data.ne_reference = 'sanders_digit'
            self.data.T_reference = 'sanders_digit'
            self.data.Z_reference = 'const'
            self.data.r = r
            self.data.ne = ne
            self.data.T = T
            self.data.Z = Z
            self.r_ne = r_ne
            self.r_T = r_T
            self.r_Z = r_Z
            f_ne = interp1d(r_ne, ne, kind='linear', fill_value='extrapolate')
            f_T = interp1d(r_T, T, kind='linear', fill_value='extrapolate')
            self.data.f_ne = f_ne
            self.data.ne_interpolate_method = '1d_linear'
            self.data.f_ne_square = interp1d(r_ne, ne**2, kind='linear', fill_value='extrapolate')
            self.data.ne_square_interpolate_method = '1d_linear'
            self.data.f_T = f_T
            self.data.T_interpolate_method = '1d_linear'
            self.data.f_Z = interp1d(r_Z, Z, kind='linear', fill_value='extrapolate')
            self.data.Z_interpolate_method = '1d_linear'
            self.data.print_info()

        elif method == 'george_digit':
            r = np.linspace(500, 2500, 100)
            r_ne, ne = self.load_setting('george_nH_digit.txt')
            ne = ne / 1.16
            for i in range(0, len(ne)):
                print(r_ne[i], ne[i])
            r_T, T = self.load_setting('george_Te_digit.txt')
            r_Z = r_ne
            Z = np.full(len(r_Z), 0.4)
            Z *= 4.68/3.27     # for abundance conversion from Anders & Grevesse (1989) to Lodders et al. (2009)
            self.data.ne_reference = 'george_digit'
            self.data.T_reference  = 'george_digit'
            self.data.r  = r
            self.data.ne = ne
            self.data.T  = T
            self.data.Z  = Z
            self.r_ne    = r_ne
            self.r_T     = r_T
            self.r_Z     = r_Z
            f_ne = interp1d(r_ne, ne, kind='linear', fill_value='extrapolate')
            f_T  = interp1d(r_T, T, kind='linear', fill_value='extrapolate')
            self.data.f_ne = f_ne
            self.data.ne_interpolate_method = '1d_linear'
            self.data.f_ne_square = interp1d(r_ne, ne**2, kind='linear', fill_value='extrapolate')
            self.data.ne_square_interpolate_method = '1d_linear'
            self.data.f_T = f_T
            self.data.T_interpolate_method = '1d_linear'
            self.data.print_info()

        elif method == 'sanders_george_digit':
            r = np.linspace(0, 1000, 400)
            r_ne, ne = self.load_setting(f'{self.local_directory}/sanders_george_ne_digit.txt')
            r_T, T = self.load_setting(f'{self.local_directory}/sanders_george_Te_digit.txt')
            r_Z = r_ne
            Z = np.full(len(r_Z), 0.4)
            Z *= 4.68/3.27     # for abundance conversion from Anders & Grevesse (1989) to Lodders et al. (2009)
            self.data.ne_reference = 'sanders_george_digit'
            self.data.T_reference = 'sanders_george_digit'
            self.data.Z_reference = 'const'
            self.data.r = r
            self.data.ne = ne
            self.data.T = T
            self.data.Z = Z
            self.r_ne   = r_ne
            self.r_T = r_T
            self.r_Z = r_Z
            f_ne = interp1d(r_ne, ne, kind='linear', fill_value='extrapolate')
            f_T  = interp1d(r_T, T, kind='linear', fill_value='extrapolate')
            self.data.f_ne = f_ne
            self.data.ne_interpolate_method = '1d_linear'
            self.data.f_ne_square = interp1d(r_ne, ne**2, kind='linear', fill_value='extrapolate')
            self.data.ne_square_interpolate_method = '1d_linear'
            self.data.f_T = f_T
            self.data.T_interpolate_method = '1d_linear'
            self.data.f_Z = interp1d(r_Z, Z, kind='linear', fill_value=(Z[0], 0.4), bounds_error=False)
            self.data.Z_interpolate_method = '1d_linear'
            self.data.print_info()

    def interpolate_nearest(self, x, y):
        f = interp1d(x, y, kind='nearest', fill_value='extrapolate')
        return f

    def nfw_density(self,r,r_200,c,H0=70):
        r_200 = r_200 * u.Mpc
        def delta_c(c):
            return (200/3) * (c**3) / (np.log(1+c) - c/(1+c))
        rho_c = 3 * (H0*u.km/u.s/u.Mpc)**2 / (8 * np.pi * const.G)
        rs = r_200 / c
        return (delta_c(c) * rho_c / ((r/rs) * (1 + r/rs)**2)).to('g/cm^3')

    def baryon_density(self,r, r_200, c, H0=70, f_b=0.16):
        return f_b * self.nfw_density(r, r_200, c, H0)

    def electron_density(self,r, r_200, c, f_b, mu=0.6):
        rho_b = self.baryon_density(r, r_200, c, f_b)
        return rho_b / (mu * const.m_p.value)

    def nfw_mass(self, r, r_200, c, H0=70):
        r_200 = r_200 * u.Mpc
        rs = r_200 / c
        rho_c = 3 * (H0*u.km/u.s/u.Mpc)**2 / (8 * np.pi * const.G)
        def delta_c(c):
            return (200/3) * (c**3) / (np.log(1+c) - c/(1+c))
        M = 4 * np.pi * (rs**3) * delta_c(c) * rho_c * (np.log(1+r/rs) - r/rs*1/(1+r/rs))
        
        # 太陽質量で規格化
        M_normalized_solar = (M / const.M_sun.to('g')).to('')
        return M_normalized_solar

    def _plot_NFW(self):
        r = np.linspace(5,500,10) * u.kpc
        mass = self.nfw_mass(r, r_200=2.2, c=3.88)
        # プロット
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Mass Profile (/Solar Mass)', color=color)
        ax1.plot(r, mass, color=color)
        # ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.tick_params(axis='y', labelcolor=color)


        plt.show()

    def const_Z(self,r):
        return np.full_like(r, 0.4)

    def beta_model_chen(self,r):
        r = r * u.kpc
        n0=5.5e-2 * u.cm**-3
        rc=62 * u.kpc
        beta=0.572
        return (n0 * (1 + (r/rc)**2) ** (-3*beta / 2)).to(u.cm ** (-3)).value

    def george_T(self,r):
        Te = 135 * ((r/230)**2.4) * (1+r/230)**(-4.1)
        return Te

    def _plot_pks_setting(self):
        R = RadiationField(2,8,0.5)
        R.plot_style('triple')
        self.load_pks_setting(method='sanders_digit')
        R.ax.scatter(self.r_ne, self.data.ne, label='Sanders+2014, Figure 9', color='black')
        R.ax2.scatter(self.r_T, self.data.T, label='Temperature', color='black')
        R.ax3.scatter(self.r_Z, self.data.Z, label='Abundance', color='black')
        R.ax.plot(self.r_ne, self.data.ne, color='black')
        R.ax2.plot(self.r_T, self.data.T, label='Temperature', color='black')
        R.ax3.plot(self.r_Z, self.data.Z, label='Abundance', color='black')

        self.load_pks_setting(method='george_digit')
        col2 = 'red'
        R.ax.scatter(self.r_ne, self.data.ne, label='George+2009, Figure 4', color=col2)
        R.ax2.scatter(self.r_T, self.data.T, label='Temperature', color=col2)
        R.ax3.scatter(self.r_Z, self.data.Z, label='Abundance', color=col2)
        R.ax.plot(self.r_ne, self.data.ne, color=col2)
        R.ax2.plot(self.r_T, self.data.T, label='Temperature', color=col2)
        R.ax3.plot(self.r_Z, self.data.Z, label='Abundance', color=col2)
        #R.ax.set_xlabel('Radius (kpc)')
        R.ax3.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel(r'$n_e \ (\rm cm^{-3})$')
        R.ax2.set_ylabel(r'$kT \ (\rm keV)$')
        R.ax3.set_ylabel(r'$Z$')
        #R.ax.set_title('PKS Setting')
        R.ax.set_xscale('log')
        R.ax.set_yscale('log')
        R.ax.legend()
        plt.show()
        R.fig.savefig('pks_setting.pdf',dpi=300,transparent=True)

    def _plot_pks_setting2(self):
        R = RadiationField(2,8,0.5)
        R.plot_style('triple')
        self.load_pks_setting(method='sanders_george_digit')
        R.ax.scatter(self.r_ne, self.data.ne, color='black')
        R.ax2.scatter(self.r_T, self.data.T, label='Temperature', color='black')
        R.ax3.scatter(self.r_Z, self.data.Z, label='Abundance', color='black')
        R.ax.plot(self.r_ne, self.data.ne, color='black')
        R.ax2.plot(self.r_T, self.data.T,  color='black')
        R.ax3.plot(self.r_Z, self.data.Z, label='Abundance', color='black')
        R.ax3.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel(r'$n_e \ (\rm cm^{-3})$')
        R.ax2.set_ylabel(r'$kT \ (\rm keV)$')
        R.ax3.set_ylabel(r'$Z$')
        #R.ax.set_title('PKS Setting')
        R.ax.set_xscale('log')
        R.ax.set_yscale('log')
        R.ax.axvline(81.9, color='black', linestyle='--', label="Resolve center")
        R.ax.axvline(245.8, color='red', linestyle='-.', label="Resolve outer")
        R.ax2.axvline(81.9, color='black', linestyle='--')
        R.ax2.axvline(245.8, color='red', linestyle='-.')
        R.ax3.axvline(81.9, color='black', linestyle='--')
        R.ax3.axvline(245.8, color='red', linestyle='-.')
        R.ax.legend()
        R.fig.savefig("pks_setting.pdf")
        plt.show()

class ACCEPT:
    '''
    This class manage the accept data which chandra analysis results of the cluster.
    ACCEPT include radial distribution of the ne, kT.
    ne(r) is deprojected analysis result, but kT is NOT deproject result.
    So, this cluster data is not completely, you should only use to estimate RS probability.
    '''
    def __init__(self):
        pass

class Abell478:

    def __init__(self, data: DataAnalysisInfo = None) -> None:
        if data is None:
            data = DataAnalysisInfo()
        self.data = data
        self.data.cluster_name = 'abell478'
        self.redshift = 0.088
        self.local_directory = os.path.expandvars(
    "$DROPBOX/share/work/astronomy/PKS/rs_simulation/cluster_setting_files/abell478"
)
    
    def load_setting(self, method='potter2023_chandra'):
        if method == 'potter2023_chandra':
            # parameters from Table 3 free Tcool Chandra
            T_pars = dict(T0=8.0, rt=260.0, a=-0.11, b=4.5, c_over_b=0.05,
                        Tmin=5.03, rc=73.7, ac=10.0)
            ne_pars = dict(n1=1.41e-4, r1=59, beta1=0.67,
                            n2=3.5e-5, r2=160, beta2=0.68)
            r = np.linspace(0.1, 1000, 400)
            ne = self.double_beta_model(r, **ne_pars) * 282.49 # conversion to norm from apec
            T = self.T_3D(r, **T_pars)
            r_Z = np.loadtxt(f'{self.local_directory}/potter2023_chandra_abundance.txt')[:,0]
            Z = np.loadtxt(f'{self.local_directory}/potter2023_chandra_abundance.txt')[:,1]
            self.data.ne_reference = 'potter2023_chandra_deproject'
            self.data.T_reference  = 'potter2023_chandra_deproject'
            self.data.Z_reference  = 'potter2023_chandra_project'
            self.r_ne = r
            self.r_T = r
            self.r_Z = r_Z
            self.data.ne = ne
            self.data.T = T
            self.data.Z = Z
            self.data.f_ne = interp1d(r, ne, kind='linear', fill_value='extrapolate')
            self.data.f_ne_square = interp1d(r, ne**2, kind='linear', fill_value='extrapolate')
            self.data.f_T = interp1d(r, T, kind='linear', fill_value='extrapolate')
            self.data.f_Z = interp1d(
                r_Z, Z,
                kind='linear',
                bounds_error=False,
                fill_value=(Z[0], Z[-1])   # 左端は Z[0]、右端は Z[-1] で固定
            )
            self.data.print_info()

    def _plot_setting(self):
        self.load_setting(method='potter2023_chandra')
        P = PlotManager((3,1),(8,8),sharex=True)
        P.axes[0].plot(self.r_ne, self.data.ne, label='ne')
        P.axes[1].plot(self.r_T, self.data.T, label='T')
        P.axes[2].plot(self.r_ne, self.data.f_Z(self.r_ne), label='Z')
        P.axes[0].set_xscale('log')
        P.axes[1].set_xscale('log')
        P.axes[2].set_xscale('log')
        P.axes[0].set_yscale('log')
        P.axes[0].set_ylabel(r'$n_e \ (\rm cm^{-3})$')
        P.axes[1].set_ylabel(r'$kT \ (\rm keV)$')
        P.axes[2].set_ylabel(r'$Z$')
        P.axes[2].set_xlabel('Radius (kpc)')
        P.axes[0].set_xlim(10, 1000)
        from xrism_proposal_calc import read_profile_unique
        prof = read_profile_unique('/Users/keitatanaka/Dropbox/AO2_proposal/analysis/cluster_catalog/chandra_accept/all_profiles.dat', target_names=['ABELL_0478'])
        r = np.linspace(0.1, 1000, 1000)
        #P.axes[0].scatter(prof['ABELL_0478']['r'], prof['ABELL_0478']['ne'])
        f = np.loadtxt('sanderson2005_ne_digit.txt')
        #P.axes[0].scatter(f[:,0], f[:,1])
        P.fig.savefig('abell478_potter_profile.png', dpi=300)
        plt.show()


    def T_3D(self, r, T0, rt, a, b, c_over_b, Tmin, rc, ac):
        """
        3D temperature profile (Potter+2023, eqs. 2 & 3).
        r : kpc
        returns T(r) in keV
        """
        c = c_over_b * b
        outer = (r/rt)**(-a) / (1 + (r/rt)**b)**(c/b)
        Tcool = ((r/rc)**ac + Tmin/T0) / (1 + (r/rc)**ac)
        return T0 * outer * Tcool

    def double_beta_model(self, r, n1, r1, beta1, n2, r2, beta2):
        """
        Double-beta model for the electron density profile.

        Parameters
        ----------
        r : 1D array of floats
            Radii at which to evaluate the electron density.
        n1 : float
            Central electron density of the first component.
        r1 : float
            Core radius of the first component.
        beta1 : float
            Power-law index of the first component.
        n2 : float
            Central electron density of the second component.
        r2 : float
            Core radius of the second component.
        beta2 : float
            Power-law index of the second component.

        Returns
        -------
        ne : 1D array of floats
            Electron density at the given radii.
        """
        term1 = n1**2 * (1 + (r/r1)**2)**(-3*beta1)
        term2 = n2**2 * (1 + (r/r2)**2)**(-3*beta2)
        return (term1 + term2)**0.5
    
class DummyCluster:
    def __init__(self, data: DataAnalysisInfo = None) -> None:
        if data is None:
            data = DataAnalysisInfo()
        self.data = data
        self.data.cluster_name = 'dummy'
        self.redshift = 0.0

    def load_setting(self, method='linear', ne=0.01, kT=5.0, Z=0.4):
        if method == 'linear':
            r    = np.array([0])
            r_ne = r
            ne   = np.full(len(r), ne)
            r_T  = r
            T    = np.full(len(r), kT)
            r_Z  = r 
            Z    = np.full(len(r), Z)
            self.data.ne_reference =  'nearest' 
            self.data.T_reference  =  'nearest'
            self.data.Z_reference  =  'nearest'
            self.data.r  = r
            self.data.ne = ne
            self.data.T  = T
            self.data.Z  = Z
            self.r_ne    = r_ne
            self.r_T     = r_T
            self.r_Z     = r_Z
            f_ne = interp1d(r_ne, ne, kind='nearest', fill_value='extrapolate')
            f_T  = interp1d(r_T, T, kind='nearest', fill_value='extrapolate')
            self.data.f_ne = f_ne
            self.data.ne_interpolate_method = '1d_linear'
            self.data.f_ne_square = interp1d(r_ne, ne**2, kind='nearest', fill_value='extrapolate')
            self.data.ne_square_interpolate_method = '1d_linear'
            self.data.f_T = f_T
            self.data.T_interpolate_method = '1d_linear'
            self.data.f_Z = interp1d(r_Z, Z, kind='nearest', fill_value=(Z[0], 0.4), bounds_error=False)
            self.data.Z_interpolate_method = '1d_linear'
            self.data.print_info()

class HDF5Manager:
    '''
    HDF5 Manager class to manage the HDF5 file.
    '''

    def __init__(self, filename='data.hdf5') -> None:
        self.filename = filename
        self.lazy_array = None

    def get_matching_datasets(self, attributes=None, precision=1):
        """
        高速化された属性一致データセットの取得メソッド。
        """
        matching_links = []
        rounded_attributes = {key: round(value, precision) if isinstance(value, (int, float)) else value
                            for key, value in (attributes or {}).items()}

        try:
            with h5py.File(self.filename, 'r') as f:
                for name, obj in f.items():
                    # データセットでない場合をスキップ
                    if not isinstance(obj, h5py.Dataset):
                        continue

                    # 属性が一致するかどうかのフラグ
                    match = True
                    for key, value in rounded_attributes.items():
                        attr_value = obj.attrs.get(key)
                        if attr_value is None:
                            match = False
                            break
                        if isinstance(value, (int, float)) and isinstance(attr_value, (int, float)):
                            if round(attr_value, precision) != value:
                                match = False
                                break
                        elif attr_value != value:
                            match = False
                            break

                    # すべての属性が一致する場合、リンクを追加
                    if match:
                        matching_links.append(name)

        except Exception as e:
            print(f"Error reading HDF5 file: {e}")

        return matching_links

    def load_dataset_by_link(self, dataset_links=None, length=None, length_mode=False):
        """
        高速化されたデータセット読み込みメソッド
        """
        datas = []
        total_length = 0

        try:
            with h5py.File(self.filename, 'r') as f:
                for link in dataset_links:
                    # 必要なデータ量だけ取得
                    dataset = f[link]
                    if length and total_length + dataset.shape[0] > length:
                        remaining_length = length - total_length
                        datas.append(dataset[:remaining_length])
                        total_length += remaining_length
                        break
                    else:
                        datas.append(dataset[:])
                        total_length += dataset.shape[0]

                    # 指定したlengthに達したら終了
                    if length and total_length >= length:
                        break

        except Exception as e:
            print(f"Error loading HDF5 dataset: {e}")

        datas = np.concatenate(datas) if datas else np.array([])
        return len(datas) if length_mode else datas

    def print_hdf5_contents(self):
        """
        Print the contents of the specified HDF5 file.
        """
        try:
            with h5py.File(self.filename, 'r') as f:
                print(f"Contents of HDF5 file: {self.filename}\n")
                
                # Print all groups and datasets
                def printname(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

                f.visititems(printname)

                # Print global attributes
                print("\nGlobal Attributes:")
                for key, value in f.attrs.items():
                    print(f"{key}: {value}")

                # Print groups and their attributes
                for group_name in f.keys():
                    group = f[group_name]
                    print(f"\nAttributes of group '{group_name}':")
                    for key, value in group.attrs.items():
                        print(f"{key}: {value}")

                print("\nDone.")
        
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")

    def load_simulation_data(self, filename='data.hdf5', keyname='data'):
        """
        Load the simulation data from the specified HDF5 file.
        """
        with h5py.File(filename, 'r') as f:
            energy            = f[keyname]['energy'][:]
            position          = f[keyname]['position'][:]
            previous_position = f[keyname]['previous_position'][:]
            initial_position  = f[keyname]['initial_position'][:]
            initial_direction = f[keyname]['initial_direction'][:]
            initial_energy    = f[keyname]['initial_energy'][:]
            scatter           = f[keyname]['scatter'][:]
            direction         = f[keyname]['direction'][:]
            velocity          = f[keyname].attrs['velocity']
            for num, photon_id in enumerate(f[keyname]['RS_position'].keys()):
                if num == 0:
                    RS_position = f[keyname]['RS_position'][f'{photon_id}'][:]
                else:
                    RS_position = np.vstack((RS_position, f[keyname]['RS_position'][f'{photon_id}'][:]))

        return energy, position, previous_position, scatter, velocity, initial_position, initial_direction, initial_energy, RS_position, direction

class Photon:
    '''
    Single Photon class
    This class is used to manage the photon data.
    '''
    def __init__(self,x=0,y=0,z=0,energy=0,photon_id=None,rs_line_list=['w','x','y','z','u','r','t','q']) -> None:
        self.x      = x
        self.y      = y
        self.z      = z
        self.energy = energy # eV
        self.reserved_energy   = energy # for save temporary energy for frame conversion
        self.ion_rest_energy   = 0.0
        self.initial_energy    = energy
        self.photon_id         = str(photon_id)
        self.current_position  = np.array([0,0,0])
        self.previous_position = np.array([0,0,0])
        self.initial_position  = np.array([0,0,0])
        self.direction         = np.array([0,0,0])
        self.initial_direction = np.array([0,0,0])
        self.rs_line_list      = rs_line_list
        self.data  = {}
        self.flags = self.create_flags()

    def show_status(self):
        print(f"Photon ID: {self.photon_id}")
        print(f"Current Position: {self.current_position}")
        print(f"Previous Position: {self.previous_position}")
        print(f"Energy: {self.energy}")
        print(f"Flags: {self.flags}")

    def create_flags(self):
        flags = {}
        flags['scatter']    = False
        return flags

    def get_flag(self, line):
        return self.flags.get(line, None)

    def random_on_sphere(self,r):
        cos_theta = -2.0 * np.random.rand() +1.0
        phi       = 2.0 * np.pi * np.random.rand()
        theta     = np.arccos(cos_theta)
        return np.array([r, theta, phi])
    
    def random_in_sphere(self,r):
        cos_theta = -2.0 * np.random.rand() +1.0
        phi       = 2.0 * np.pi * np.random.rand()
        theta     = np.arccos(cos_theta)
        r         = r * np.power(np.random.rand(),1/3)
        return np.array([r, theta, phi])

    def polartocart(self,data):
        r, theta, phi = data[0], data[1], data[2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def add_polar(self, data1, data2):
        x1, y1, z1 = self.polartocart(data1)
        x2, y2, z2 = self.polartocart(data2)
        x = x1 + x2
        y = y1 + y2
        z = z1 + z2
        return self.carttopolar(np.array([x, y, z]))

    def add_polar_array(self, data1, data2):
        x1, y1, z1 = self.polartocart_array(data1)
        x2, y2, z2 = self.polartocart_array(data2)
        x = x1 + x2
        y = y1 + y2
        z = z1 + z2
        return np.array(self.carttopolar_array(np.array([x, y, z])))

    def add_vector(self, data, direction, radius):
        r_direction, theta_direction, phi_direction = direction[:, 0], direction[:, 1], direction[:, 2]
        r, theta, phi = data[:, 0], data[:, 1], data[:, 2]
        added_r = radius * r_direction
        return np.vstack((added_r, theta_direction, phi_direction)).T

    def unit_vector(self, data1, data2):
        x1, y1, z1 = self.polartocart(data1)
        x2, y2, z2 = self.polartocart(data2)
        v1 = np.array([x1, y1, z1])
        v2 = np.array([x2, y2, z2])
        vec = v2 - v1
        return vec/np.linalg.norm(vec)

    def polartocart_array(self, data):
        r, theta, phi = data[:,0], data[:,1], data[:,2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack((x, y, z), axis=1)

    def carttopolar(self, data):
        x, y, z = data[0], data[1], data[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        # Convert phi to range [0, 2*pi]
        if phi < 0:
            phi += 2 * np.pi
        return np.array([r, theta, phi])
    
    def carttopolar_array(self,data):
        x, y, z = data[:,0], data[:,1], data[:,2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        phi[phi<0] += 2 * np.pi
        return np.stack((r, theta, phi), axis=1)

    def make_rotation_matrix(self, m):
        """
        make rotation matrix from vector m
        m: rotation axis
        use ex) R.T @ V
        """
        m = m / np.linalg.norm(m)  # z' 軸
        ref = np.array([0, 0, 1]) if abs(m[2]) < 0.99 else np.array([1, 0, 0])
        
        x = np.cross(ref, m)
        x /= np.linalg.norm(x)      # x' 軸
        y = np.cross(m, x)          # y' 軸

        R = np.column_stack([x, y, m])  # 回転行列（列が x', y', z'）
        # print(f"Rotation Matrix:\n{R}")
        return R
    
    def visualize_rotation(self, m):
        # 回転行列を作る
        R = self.make_rotation_matrix(m)

        # 元の座標軸（ワールド座標）: X, Y, Z
        origin = np.zeros(3)
        X = np.array([1, 0, 0])
        Y = np.array([0, 1, 0])
        Z = np.array([0, 0, 1])

        # 新しい座標軸: X', Y', Z'
        Xp = R[:, 0]
        Yp = R[:, 1]
        Zp = R[:, 2]
        
        # プロット設定
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # 軸の範囲を少し余裕を持って設定
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])

        # 元の軸をプロット
        ax.quiver(*origin, *X, length=1, arrow_length_ratio=0.1, label='X', color="black")
        ax.quiver(*origin, *Y, length=1, arrow_length_ratio=0.1, label='Y', color="black")
        ax.quiver(*origin, *Z, length=1, arrow_length_ratio=0.1, label='Z', color="black")

        # 新しい軸をプロット
        ax.quiver(*origin, *Xp, length=1, arrow_length_ratio=0.1, label="X'")
        ax.quiver(*origin, *Yp, length=1, arrow_length_ratio=0.1, label="Y'")
        ax.quiver(*origin, *Zp, length=1, arrow_length_ratio=0.1, label="Z'")
        x = np.array([[0, 0, 0], [1, 1, 1]])
        ax.plot3D(x[:,0],x[:,1],x[:,2],'-.', color="black", alpha=0.5)
        x = np.vstack((np.array([0,0,0]), R @ np.array([1, 1, 1])))
        ax.plot3D(x[:,0],x[:,1],x[:,2],'-.', color="blue", alpha=0.5)
        #ax.plot3D(np.array([0,0,0]),R.T @ np.array([1,1,1]),'-.',color="blue", alpha=0.5)
        print("-.-.-.-.-.-.-.-.-.-")
        print(R @ np.array([1,1,1]))
        af_conv = R @ np.array([1,1,1])
        x = np.vstack((np.array([0,0,0]), R.T @ af_conv))
        ax.plot3D(x[:,0],x[:,1],x[:,2],'-.', color="red", alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        print(f"Rotation Matrix:\n{R}")
        plt.title('Original Axes vs. Rotated Axes')
        plt.show()

    def gaussian_energy(self,center_energy,sigma,size=1):
        return np.random.normal(loc=center_energy, scale=sigma, size=size)
    
    def generate_random_photon(self,r,center_energy):
        self.current_position  = self.random_on_sphere(r)
        self.direction         = self.random_on_sphere(1)
        self.initial_position  = self.current_position
        self.initial_direction = self.direction
        self.previous_position = np.array([0,0,0])
        self.energy            = center_energy
        self.initial_energy    = center_energy

    def calculate_photon_angle(self):
        """
        Calculate the angle vector of the photon.
        angle_vector = current_position - previous_position
        magnitude = np.linalg.norm(angle_vector)
        """
        current_position_cart = self.polartocart(self.current_position)
        previous_position_cart = self.polartocart(self.previous_position)
        angle_vector = current_position_cart - previous_position_cart
        magnitude = np.linalg.norm(angle_vector)
        return angle_vector/magnitude

    def propagate_to_rmax(self, position, direction, rmax, coordinate='polar'):
        """
        光子の現在位置と方向から、r = rmax に達する位置を計算する。
        
        Parameters
        ----------
        position : ndarray (3,)
            現在位置ベクトル
        direction : ndarray (3,)
            単位方向ベクトル
        rmax : float
            球の最大半径
            
        Returns
        -------
        new_position : ndarray (3,)
            r = rmax に到達したときの位置ベクトル（デカルト）
        """

        if coordinate == 'polar':
            position = self.polartocart(position)
            direction = self.polartocart(direction)

        r0_dot_v = np.dot(position, direction)
        r0_norm2 = np.dot(position, position)
        discriminant = r0_dot_v**2 + rmax**2 - r0_norm2

        if discriminant < 0:
            raise ValueError("交差なし（外向き or 範囲外）")

        s = -r0_dot_v + np.sqrt(discriminant)
        new_position = position + s * direction
        if coordinate == 'polar':
            new_position = self.carttopolar(new_position)
        return new_position


    def propagate_to_rmax_array(self, position, direction, rmax, coordinate='polar'):
        """
        複数の光子の現在位置と方向から、r = rmax に達する位置を計算する（ベクトル化対応）。
        
        Parameters
        ----------
        position : ndarray (N, 3) or (3,)
            現在位置ベクトル（極座標またはデカルト）
        direction : ndarray (N, 3) or (3,)
            単位方向ベクトル（極座標またはデカルト）
        rmax : float
            球の最大半径
        coordinate : str
            'polar' または 'cartesian'。入力が極座標なら 'polar'。

        Returns
        -------
        new_position : ndarray (N, 3)
            r = rmax に到達したときの位置ベクトル（元と同じ座標系）
        """



        if coordinate == 'polar':
            position = self.polartocart_array(position)
            direction = self.polartocart_array(direction)

        # 内積とノルムの計算（バッチ処理）
        r0_dot_v = np.einsum('ij,ij->i', position, direction)
        r0_norm2 = np.einsum('ij,ij->i', position, position)
        discriminant = r0_dot_v**2 + rmax**2 - r0_norm2

        if np.any(discriminant < 0):
            raise ValueError("交差なし（外向き or 範囲外）の光子が存在します")

        s = -r0_dot_v + np.sqrt(discriminant)
        new_position = position + (s[:, np.newaxis] * direction)

        if coordinate == 'polar':
            new_position = self.carttopolar_array(new_position)

        return new_position

    def _check_pos_3D(self):
        pos = np.array([])
        energy = np.array([])
        for i in range(0,100000):
            pos_e = self.polartocart(self.random_on_sphere(1))
            if i==0:
                pos = pos_e
            else:
                pos = np.vstack((pos,pos_e))
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        data_hist = pos
        print(data_hist)
        ax2.scatter(data_hist[:,0],data_hist[:,1],data_hist[:,2])
        plt.show()

    def _check_pos(self):
        pos = np.array([])
        pos_b = np.array([])
        energy = np.array([])
        for i in range(0,10000):
            pos_e_b = self.random_in_sphere(1)
            pos_e = self.carttopolar(self.polartocart(pos_e_b))
            if i==0:
                pos = pos_e
                pos_b = pos_e_b
            else:
                pos = np.vstack((pos,pos_e))
                pos_b = np.vstack((pos_b,pos_e_b))
        fig = plt.figure()
        ax2 = fig.add_subplot(311)
        ax3 = fig.add_subplot(312)
        ax4 = fig.add_subplot(313)
        data_hist = pos
        print(data_hist)
        ax2.hist(data_hist[:,0],bins=500,histtype='step',label='convert')
        ax3.hist(data_hist[:,1],bins=500,histtype='step')
        ax4.hist(data_hist[:,2],bins=500,histtype='step')
        data_hist = pos_b
        ax2.hist(data_hist[:,0],bins=500,histtype='step',label='raw')
        ax3.hist(data_hist[:,1],bins=500,histtype='step')
        ax4.hist(data_hist[:,2],bins=500,histtype='step')
        ax2.legend()
        plt.show()

    def _check_energy_pos(self):
        pos = np.array([])
        energy = np.array([])
        for i in range(0,10000):
            self.generate_random_photon(1e-5,6700,3)
            pos_e = self.current_position
            if i==0:
                pos = pos_e
                energy = self.energy
            else:
                pos = np.vstack((pos,pos_e))
                energy = np.append(energy, self.energy)
        fig = plt.figure()
        ax2 = fig.add_subplot(411)
        ax3 = fig.add_subplot(412)
        ax4 = fig.add_subplot(413)
        ax5 = fig.add_subplot(414)
        data_hist = pos
        print(data_hist)
        ax2.hist(data_hist[:,0],bins=500,histtype='step')
        ax3.hist(data_hist[:,1],bins=500,histtype='step')
        ax4.hist(data_hist[:,2],bins=500,histtype='step')
        ax5.hist(energy,bins=500,histtype='step')
        plt.show()    

class PhotonManager:
    '''
    Photon Manager class
    This class is used to manage the photon data.
    '''
    def __init__(self) -> None:
        self.photon_dataset = {
            'initial_position': [],
            'initial_direction': [],
            'previous_position': [],
            'position': [],
            'initial_energy': [],
            'energy': [],
            'scatter': [],
            'direction': [],
            'RS_position': {}
        }

    def store_photon_data(self, photon: Photon, num):
        self.photon_dataset['initial_position'].append(photon.initial_position)
        self.photon_dataset['initial_direction'].append(photon.initial_direction)
        self.photon_dataset['previous_position'].append(photon.previous_position)
        self.photon_dataset['position'].append(photon.current_position)
        self.photon_dataset['initial_energy'].append(photon.initial_energy)
        self.photon_dataset['energy'].append(photon.energy)
        self.photon_dataset['scatter'].append(photon.flags['scatter'])
        self.photon_dataset['direction'].append(photon.direction)
        if hasattr(photon, 'RS_position') == True:
            self.photon_dataset[f'RS_position'][f'{num}'] = photon.RS_position
    
class InitialPhotonGenerator:
    '''
    Initial Photon Generator class
    This class is used to generate initial position of photons.
    The photon distribution is generated based on the electron density squared distribution.
    The generated photons are saved in an HDF5 file.
    '''
    def __init__(self, ClusterManager, savefilename) -> None:
        '''
        parameter

        ----------
        cluster_name : str
            The name of the cluster.
        rmin : float
            The minimum radius of the cluster.
        rmax : float
            The maximum radius of the cluster.
        division : int
            The number of divisions of the cluster.
        savefilename : str
            The name of the HDF5 file to save the generated photons.
        '''
        self.CM  = ClusterManager
        self.cluster_name = self.CM.cluster_name
        self.rmin = self.CM.rmin
        self.rmax = self.CM.rmax
        self.division = self.CM.division
        if savefilename == None:
            self.savefilename = f'initial_photon_distribution_{self.cluster_name}.hdf5'
        else:
            self.savefilename = savefilename

    def generate_and_save(self, size=1000, save=True):
        """
        Generate random electron density squared samples and save them to an HDF5 file.
        
        Parameters:
        filename (str): The name of the HDF5 file.
        analysis_info (DataAnalysisInfo): The data analysis information to save.
        """
        samples = self.generate_random_ne_squared(size=size)
        if save == True:
            self.save_photon_distribution_hdf5(samples, self.savefilename, self.CM.Cluster.data)
        return samples

    def generate_random_ne_squared(self, size=1000,  _check=False):
        '''
        Generate random electron density squared samples using the cluster data.
        Parameters
        ----------
        size : int
            The number of samples to generate.
        _check : bool
            If True, return the integral value of the electron density squared.
        Returns
        -------
        samples : array
            The generated samples.
        integral_value : float
            The integral value of the electron density squared.
        '''
        size     = int(size)
        ne_max   = np.max(self.CM.Cluster.data.f_ne(self.rmax))
        r_values = np.linspace(0, self.rmax, self.division)
        max_p    = np.max([self.CM.Cluster.data.f_ne_square(r)*4*np.pi*r**2 for r in r_values])
        samples  = []
        while len(samples) < size:
            r = np.random.uniform(0, self.rmax)
            p = np.random.uniform(0, max_p)
            if p < self.CM.Cluster.data.f_ne_square(r)*4*np.pi*r**2 :
                samples.append(r)

        if _check:
            def integrand(r):
                return self.CM.Cluster.data.f_ne_square(r) * 4 * np.pi * r**2
            integral_value, _ = quad(integrand, 0, self.rmax)
            return np.array(samples), integral_value

        return np.array(samples)

    def save_photon_distribution_hdf5(self, samples, filename='initial_photon_distribution.hdf5', analysis_info: DataAnalysisInfo = None):
        """
        Save generated photon data to an HDF5 file.
        
        Parameters:
        filename (str): The name of the HDF5 file.
        analysis_info (DataAnalysisInfo): The data analysis information to save.
        """

        with h5py.File(filename, 'a') as f:
            dataset_name = f"photons_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset = f.create_dataset(dataset_name, data=samples)
            
            # DataAnalysisInfo の情報を属性として保存
            if analysis_info is not None:
                dataset.attrs['ne_reference']                 = analysis_info.ne_reference
                dataset.attrs['T_reference']                  = analysis_info.T_reference
                dataset.attrs['Z_reference']                  = analysis_info.Z_reference
                dataset.attrs['ne_interpolate_method']        = analysis_info.ne_interpolate_method
                dataset.attrs['ne_square_interpolate_method'] = analysis_info.ne_square_interpolate_method
                dataset.attrs['T_interpolate_method']         = analysis_info.T_interpolate_method
                dataset.attrs['Z_interpolate_method']         = analysis_info.Z_interpolate_method
                dataset.attrs['cluster_name']                 = analysis_info.cluster_name
            
            else:
                print('DataAnalysisInfo is not defined.')

            dataset.attrs['size'] = len(samples)
            dataset.attrs['timestamp'] = datetime.now().isoformat()


            print(f"Photons and metadata saved to {filename} in dataset {dataset_name}") 
    
# ?: for thesis plot code
def _check_ne_dist(size=10000, cluster_name='perseus', rmin=0, rmax=1000, division=400):
    CM = ClusterManager(cluster_name=cluster_name, rmin=rmin, rmax=rmax, division=division)
    IP = InitialPhotonGenerator(CM, None)
    generate_photon, norm = IP.generate_random_ne_squared(size=size, _check=True)

    r = CM.Cluster.data.r
    f_ne2 = CM.Cluster.data.f_ne_square

    r_edges = r    
    rebined_photon = CM.assign_nearest_value(generate_photon, CM.Cluster.data.r)

    R = PlotManager()
    ax = R.axes[0]

    # === 実サンプルのヒストグラム ===
    ax.hist(generate_photon, bins=1000, density=False, histtype='step', color='black', label='Sampled photons')
    ax.hist(rebined_photon, bins=1000, density=False, histtype='step', color='blue', label='Rebinned photons')
    # print(len(generate_photon), len(rebined_photon))
    print(sorted(rebined_photon))
    # === 理論分布 ===
    ax.step(r, f_ne2(r) * 4 * np.pi * r**2 / norm, color='red', label=r'$n_e^2 \times 4\pi r^2$')

    ax.legend()
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Normalized Probability Density of $n_e^2 \\times 4\\pi r^2$\nSample size = {size}')
    R.fig.savefig('ne_dist.pdf', dpi=300, transparent=True)
    plt.show()

def _check_ne_dist_data(size=10000):
        with h5py.File('initial_photon_distribution.hdf5', 'r') as f:
            dataset = f['photons_20241111_235700']
            generate_photon = dataset[...]

        _generate_photon, norm = self.generate_random_ne_squared(size=size, _check=True)
        r = self.CM.Cluster.data.r
        print(generate_photon)
        R = RadiationField()
        R.plot_style('single')
        bins = self.CM.divide_cluster_sphere_linear()
        photon, r_bins = np.histogram(generate_photon, bins=bins, density=True)
        self.rebined_radial_distribution = sorted(self.CM.rebin_data_linear(generate_photon))
        print(self.rebined_radial_distribution)
        bin_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        #R.ax.scatter(bin_centers, photon, color='black', label='sampling data2')
        R.ax.hist(generate_photon, bins=r, density=True, histtype='step', color='black', label='sampling data')
        R.ax.step(r, self.CM.Cluster.data.f_ne_square(r)*4*np.pi*r**2/norm, color='red', where='mid', label='ne squared')
        #R.ax.scatter(r, self.CM.Cluster.data.f_ne_square(r)/norm, color='red')
        #R.ax.set_xscale('log')
        #self.ax.set_yscale('log')
        R.ax.legend()
        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Probability Density')
        R.ax.set_title(f'Normalized Probability Density of Electron Density Squared\n Sample size = {size}')
        R.fig.savefig('ne_dist.pdf',dpi=300,transparent=True)
        plt.show()

class RSLineManager:
    '''
    Resonance Scattering Line Manager class
    This class is used to manage the resonance scattering line data.

    Parameters
    ----------
    line_list : list
        The list of resonance scattering lines to manage.
    dE : float
        The energy range to consider for each line (eV). e.g. 1

    Attributes
    ----------
    energy_ranges : dict
        The energy ranges for each line.
    dE : float
        The energy range to consider for each line.

    '''
    def __init__(self, rs_line_list, dE, atomdb_version, abundance_table, load_furukawa) -> None:
        self.rs_line_list  = rs_line_list
        self.dE            = dE
        self.atomdb_version = atomdb_version
        self.abundance_table = abundance_table

        #! : This is tempolary setting. Please Remove it finnaly.
        self.load_furukawa = load_furukawa
        self.line_energies = {}
        self.natural_width = {}
        atomic_data = AtomicDataManager(atomic_data_file, self.atomdb_version, load_furukawa)
        for line in self.rs_line_list:
            atomic_data.load_line_data(state=line)
            self.line_energies[line] = atomic_data.line_energy
            self.natural_width[line] = atomic_data.natural_width
        self.energy_ranges = np.array([
            (self.line_energies[line] - dE, self.line_energies[line] + dE)
            for line in self.rs_line_list if line in self.line_energies
        ])

    def is_in_range(self, photon_energy):
        """
        単一の光子エネルギーに対して範囲をチェックし、
        どの輝線に属するかを返す。
        """
        line_names_e = []
        for i, (min_energy, max_energy) in enumerate(self.energy_ranges):
            if min_energy <= photon_energy <= max_energy:
                line_names_e.append(self.rs_line_list[i])
        return line_names_e

    def define_RS_probability_list(self,rmax=1, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0, mode="natural"):
        '''
        Define the probability list for the resonance scattering.
        '''
        physics = Physics(line_state='w',atomdb_version=self.atomdb_version, abundance_table=self.abundance_table,load_furukawa=self.load_furukawa)
        self.probability_func_list = {}
        for line in self.rs_line_list:
            center_energy  = self.line_energies[line]
            minimum_energy = center_energy - self.dE
            maximum_energy = center_energy + self.dE
            if mode == "natural":
                prob_func = physics.resonance_scattering_probability_field_natulal(line_state=line, rmax=rmax, Emin=minimum_energy, Emax=maximum_energy, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            elif mode == "effective":
                prob_func = physics.resonance_scattering_probability_field(line_state=line, rmax=rmax, Emin=minimum_energy, Emax=maximum_energy, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            elif mode == "mfp":
                prob_func = physics.resonance_scattering_mfp_field(line_state=line, rmax=rmax, Emin=minimum_energy, Emax=maximum_energy, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            elif mode == "tau":
                prob_func = physics.resonance_scattering_tau_field(line_state=line, rmax=rmax, Emin=minimum_energy, Emax=maximum_energy, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            elif mode == "tau_dL":
                prob_func = physics.resonance_scattering_tau_dL_field(line_state=line, rmax=rmax, Emin=minimum_energy, Emax=maximum_energy, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            self.probability_func_list[line] = prob_func
        return self.probability_func_list

    def define_RS_sigma_list(self,rmax=1,Emin=6500, Emax=8000, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Define the probability list for the resonance scattering.
        '''
        physics = Physics()
        self.probability_func_list = {}
        for line in self.rs_line_list:
            center_energy = self.line_energies[line]
            prob_func = physics.resonance_scattering_sigma_field(line_state=line, rmax=rmax, Emin=Emin, Emax=Emax, dn=dn, dL=dL, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            #prob_func = physics.resonance_scattering_mean_free_path(line_state=line, rmax=rmax, Emin=Emin, Emax=Emax, dn=dn, f_ne=f_ne, f_T=f_T, f_Z=f_Z, velocity=velocity)
            self.probability_func_list[line] = prob_func
        return self.probability_func_list

class Physics:
    '''
    Physics class to manage the physical constants and calculations.
    This class is used to manage the physical constants and calculations.
    '''
    def __init__(self, line_state, atomdb_version, abundance_table, load_furukawa) -> None:
        self.r_e_SI = (const.e.gauss**2 / const.m_e / const.c**2).to('m')  # SI単位系の古典電子半径（m）
        self.K2keV  = 1/const.e.gauss/1e+3
        self.keV2K  = 1/self.K2keV
        if abundance_table == 'lpgs':
            self.Fe_H   = 3.27e-5 # Feの太陽金属量 Lodders+2009 Table 4 Fe/H 
        elif abundance_table == 'AG89':
            self.Fe_H   = 4.68e-5 # Feの太陽金属量 angr 
        elif abundance_table == 'wilm':
            self.Fe_H   = 2.69e-5 # Feの太陽金属量 wilm
        self.eV2J   = const.e.gauss
        self.k_Boltzmann_keV = const.k_B.value * const.e.value *1e-3
        self.load_furukawa = load_furukawa
        self.atom = AtomicDataManager(atomic_data_file, atomdb_version, load_furukawa)
        self.set_atomic_state(line_state)
        self.ne_to_nH_factor = 1/1.17
        self.deltaE  = 0 
        self.sigma_0 = 0

    def ensure_unit(self, value, default_unit):
        """
        Ensure that the value has the specified unit.
        Parameters
        ----------
        value : float or Quantity
            The value to check.
        default_unit : Quantity
            The default unit to apply if the value does not have a unit.
        Returns
        -------
        value : Quantity
            The value with the specified unit.
        """
        if hasattr(value, 'unit'):
            return value
        else:
            return value * default_unit

    def set_atomic_state(self, line_state, Z=26):
        '''
        Set the atomic state and ion fraction for the given line state.
        #! : Now only Iron is supported.
        Parameters
        ----------
        line_state : str
            The line state to set. Please see the AtomicDataManager for available states.
        Z : int
            The atomic number of the element. default is 26 (Iron).
        '''
        #! : This is tempolary setting. Please Remove it finnaly.
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=Z, stage=self.atom.z1)

    def DeltaE(self,T,E0,v,A=55.845):
        '''
        Calculate the delta E due to thermal Doppler effect, turbulence, and natural width.
        The formula is given by:
        delta E = E0 * sqrt(2 * k_Boltzmann * T / A + 2 * v^2 / c^2 + natural_width^2 / E0^2)
        Natural width is given by the atomic data.
        The turbulent velocity is the 1D line-of-sight velocity.
        delta E is characterized by the following equation with sigma:
        delta E = sqrt(2) * sigma
        Parameters
        ----------
        T : float
            The temperature in keV.
        E0 : float
            The initial energy in eV.   
        v : float
            The velocity in km/s.
        A : float
            The atomic mass in amu.
        Returns
        -------
        deltaE : float
            The energy change in eV.
        '''
        T  = self.ensure_unit(T, u.keV)
        E0 = self.ensure_unit(E0, u.eV)
        v  = self.ensure_unit(v, u.km/u.s)
        self.deltaE = (E0*((2*T)/(A*u.u*const.c**2) + 2*v**2/const.c**2 + self.atom.natural_width**2/E0**2)**(1/2)).to(u.eV)
        return self.deltaE

    def Sigma_0(self,T,E0,v,A=55.845,E=None):
        '''
        Calculate the cross section for resonance scattering.
        The formula is given by:
        sigma_0 = sqrt(pi) * h * r_e * c * f / deltaE
        where:
        - h is the Planck constant
        - r_e is the classical electron radius
        - c is the speed of light
        - f is the oscillator strength
        - deltaE is the energy change due to thermal Doppler effect, turbulence, and natural width
        Parameters
        ----------
        T : float
            The temperature in keV.
        E0 : float
            The initial energy in eV.
        v : float
            The velocity in km/s.
        A : float
            The atomic mass in amu.
        E : float
            The energy in eV.
        Returns
        -------
        sigma_0 : float
            The cross section in cm^2.
        '''
        T  = self.ensure_unit(T, u.keV)
        E0 = self.ensure_unit(E0, u.eV)
        v  = self.ensure_unit(v, u.km/u.s)
        self.DeltaE(T=T,E0=E0,v=v,A=A)
        
        #print(f'deltaE = {self.deltaE}')
        if E is None:
            self.sigma_0 = (np.sqrt(np.pi)*const.h*self.r_e_SI*const.c*self.atom.f/(self.deltaE))
        else:
            E = self.ensure_unit(E, u.eV)
            self.sigma_0 = np.nan_to_num((np.sqrt(np.pi)*const.h*self.r_e_SI*const.c*self.atom.f/(self.deltaE)).to(u.cm**2)*np.exp(-((E-E0)/self.deltaE)**2))
        return self.sigma_0

    def Sigma_0_natural_delE(self,E0=6700.4,E=None):
        self.deltaE = self.atom.natural_width
        self.Sigma_0_delE_func(E0=E0, delE=self.deltaE, E=E)
        return self.sigma_0

    def Sigma_0_delE_func(self, E0, delE, E=None):
        '''
        Calculate the cross section for resonance scattering depending delE.
        The precise definition is written in the Sigma_0 function.

        Parameters
        ----------
        E0 : float
            The initial energy in eV.
        delE : float
            The energy change in eV.
        E : float
            The energy in eV.
        Returns
        -------
        sigma_0 : float
            The cross section in cm^2.
        '''
        E0   = self.ensure_unit(E0, u.eV)
        delE = self.ensure_unit(delE, u.eV)
        if E is None:
            self.sigma_0 = (np.sqrt(np.pi)*const.h*self.r_e_SI*const.c*self.atom.f/(delE)).to(u.cm**2)
        else:
            E = self.ensure_unit(E, u.eV)
            self.sigma_0 = np.nan_to_num((np.sqrt(np.pi)*const.h*self.r_e_SI*const.c*self.atom.f/(delE)).to(u.cm**2)*np.exp(-((E-E0)/delE)**2))
        return self.sigma_0


    def _compute_resonance_scattering(self, line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="tau", broadening=False, delE=None):
        '''
        Calculate the resonance scattering probability or tau or sigma.
        Parameters
        ----------
        line_state : str
            The line state to set. Please see the AtomicDataManager for available states.
        rmax : float
            The maximum radius in kpc.
        Emin : float
            The minimum energy in eV.
        Emax : float
            The maximum energy in eV.
        dn : int
            The number of points in the mesh grid.
        dL : float
            The distance in kpc.
        f_ne : function
            The function to calculate the electron density.
        f_T : function
            The function to calculate the temperature.
        f_Z : function
            The function to calculate the metallicity.
        velocity : float
            The velocity in km/s.
        style : str
            The style of the calculation. Can be "tau", "prob", "sigma", and "mfp".
        broadening : bool
            Whether to include broadening in the calculation.
        Returns
        -------
        interp_function : RegularGridInterpolator
            The interpolated function for the resonance scattering probability or tau or sigma.
        '''
        # Ensure units for the quantities
        dL = self.ensure_unit(dL, u.kpc)

        # Load atomic data
        self.set_atomic_state(line_state)

        # Create mesh grid for radius and energy
        r = np.linspace(0, rmax, dn)
        e = np.linspace(Emin, Emax, dn)
        R, E = np.meshgrid(r, e, indexing='ij')

        # Define the common part of the computation
        def compute_prob(R, E):
            tau_factor = (f_ne(R) *u.cm**(-3) * self.ne_to_nH_factor * self.Fe_H * f_Z(R) * self.atom.iz(f_T(R)*u.keV) * dL).to('cm**-2')
            if broadening:
                common_term = self.Sigma_0(T=f_T(R), E0=self.atom.line_energy, v=velocity, E=E) * tau_factor
            else:
                common_term = self.Sigma_0_natural_delE(E0=self.atom.line_energy, E=E) * tau_factor
                if delE != None:
                    common_term = self.Sigma_0_delE_func(E0=self.atom.line_energy, delE=delE, E=E) * tau_factor

            if style == "tau":
                return common_term.to('').value
            elif style == "tau_dL":
                return (common_term / dL).to('/kpc').value
            elif style == "prob":
                return 1 - np.exp(-common_term.to(''))
            elif style == "sigma":
                return (common_term/tau_factor).to('cm**2')
            elif style == "mfp":
                return (dL/common_term).to('kpc').value

        # Compute the scattering probability or tau
        prob_or_tau = compute_prob(R, E)
        
        # Interpolate the result
        interp_function = RegularGridInterpolator((r, e), prob_or_tau, bounds_error=False, fill_value=0)
        
        return interp_function
    
    def resonance_scattering_tau_field(self, line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the probability of resonance scattering (tau).
        '''
        # Call the common helper function with tau=True
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="tau", broadening=True)

    def resonance_scattering_tau_dL_field(self, line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the probability of resonance scattering (tau/dL).
        '''
        # Call the common helper function with tau=True
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="tau_dL", broadening=True)

    def resonance_scattering_probability_field(self, line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the probability of resonance scattering.
        '''
        # Call the common helper function with tau=False
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="prob", broadening=True)
    
    def resonance_scattering_probability_field_natulal(self,line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the probability of resonance scattering.
        '''
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="prob", broadening=False)
    
    def resonance_scattering_probability_field_delE(self,line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0, delE=2):
        '''
        Calculate the probability of resonance scattering.
        '''
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="prob", broadening=True, delE=delE)

    def resonance_scattering_probability_field_natulal_return_sigma(self,line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the probability of resonance scattering.
        '''
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="sigma", broadening=True)

    def resonance_scattering_probability_field_natural_velocity(self, line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        pass

    def resonance_scattering_mfp_field(self, line_state='w', rmax=1000, Emin=6600, Emax=6800, dn=1000, dL=1e-2, f_ne=None, f_T=None, f_Z=None, velocity=0):
        '''
        Calculate the mean free path of resonance scattering.
        '''
        # Call the common helper function with mfp=True
        return self._compute_resonance_scattering(line_state, rmax, Emin, Emax, dn, dL, f_ne, f_T, f_Z, velocity, style="mfp", broadening=True)

    def thomson_scattering(self, energy):
        '''
        Calculate the Thomson scattering cross section.
        '''
        if not isinstance(energy, u.Quantity):
            energy = energy * u.eV
        x = (2 * energy / (const.m_e * const.c**2)).to('')
        sigma = 8 * np.pi * (self.r_e_SI**2) * (1 - x) / 3
        return sigma.to('cm^2')

    def _compute_thomson_scattering(self, Emin, Emax, rmax, dn, dL, f_ne, style):
        '''
        Calculate the Thomson scattering cross section.
        '''
        # Ensure units for the quantities
        dL = self.ensure_unit(dL, u.kpc)
        r = np.linspace(0, rmax, dn)
        e = np.linspace(Emin, Emax, dn)
        R, E = np.meshgrid(r, e, indexing='ij')

        def compute_prob(R, E):
            common_term = self.thomson_scattering(E * u.eV) * dL * f_ne(R) * u.cm**(-3)
            if style == "prob":
                return 1 - np.exp(-(common_term).to(''))
            elif style == "sigma":
                return (common_term / (f_ne(R) * u.cm**(-3))).to('cm^2')
            elif style == "tau":
                return (common_term).to('')
            elif style == "mfp":
                return (dL / common_term).to('kpc')
        
        # Compute the scattering probability or tau
        prob_or_tau = compute_prob(R, E)
        
        # Interpolate the result
        interp_function = RegularGridInterpolator((r, e), prob_or_tau, bounds_error=False, fill_value=None)
        
        return interp_function

    def thomson_scattering_probability_field(self, Emin, Emax, rmax, dn, dL, f_ne):
        '''
        Calculate the probability of Thomson scattering.
        '''
        return self._compute_thomson_scattering(Emin, Emax, rmax, dn, dL, f_ne, style="prob")
    
    def thomson_scattering_mfp_field(self, Emin, Emax, rmax, dn, dL, f_ne):
        '''
        Calculate the mean free path of Thomson scattering.
        '''
        return self._compute_thomson_scattering(Emin, Emax, rmax, dn, dL, f_ne, style="mfp")

    def thomson_scattering_tau_field(self, Emin, Emax, rmax, dn, dL, f_ne):
        '''
        Calculate the tau of Thomson scattering.
        '''
        return self._compute_thomson_scattering(Emin, Emax, rmax, dn, dL, f_ne, style="tau")

    def _plot_Thomson_scattering_sigma(self):
        '''
        Plot the Thomson scattering cross section.
        '''
        energy = np.linspace(2000, 8000, 1000) * u.eV
        sigma = self.thomson_scattering(energy)
        plt.plot(energy, sigma)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Cross Section (cm^2)')
        plt.title('Thomson Scattering Cross Section')
        plt.show()

    def _plot_Thomson_scattering_probability(self):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 1
        interp_function = self.thomson_scattering_probability_field(2000, 8000, 1000, 1000, dL)
        r = np.linspace(0, 1000, 1000)
        e = np.linspace(2000, 8000, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        prob = interp_function((Ri, Ei))
        print(interp_function((0, 2000)), interp_function((0, 8000)))
        print(interp_function((0, 2000)), interp_function((1000, 2000)))
        # プロット
        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        im = R.ax.imshow(prob, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='cividis', aspect='auto', origin='lower', norm=LogNorm())

        # カラーバーの追加
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability')

        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {dL} kpc')

        R.fig.savefig('Thomson_scattering_probability.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_Thomson_scattering_mfp(self):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 1
        CM = ClusterManager(cluster_name='perseus', rmin=0, rmax=1000)
        interp_function = self.thomson_scattering_mfp_field(2000, 8000, 1000, 1000, dL, CM.Cluster.data.f_ne)
        r = np.linspace(0, 1000, 1000)
        e = np.linspace(2000, 8000, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        prob = interp_function((Ri, Ei))
        print(interp_function((0, 2000)), interp_function((0, 8000)))
        print(interp_function((0, 2000)), interp_function((1000, 2000)))
        # プロット
        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        im = R.ax.imshow(prob, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='cividis', aspect='auto', origin='lower', norm=LogNorm())

        # カラーバーの追加
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability')

        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {dL} kpc')

        R.fig.savefig('Thomson_scattering_mfp.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_Thomson_scattering_tau(self):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 1
        CM = ClusterManager(cluster_name='perseus', rmin=0, rmax=1000)
        interp_function = self.thomson_scattering_tau_field(2000, 8000, 1000, 1000, dL, CM.Cluster.data.f_ne)
        r = np.linspace(0, 1000, 1000)
        e = np.linspace(2000, 8000, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        prob = interp_function((Ri, Ei))
        # プロット
        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        im = R.ax.imshow(prob, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='cividis', aspect='auto', origin='lower', norm=LogNorm())

        # カラーバーの追加
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability')

        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {dL} kpc')

        R.fig.savefig('Thomson_scattering_tau.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_Thomson_scattering_tau_integrated(self):
        '''
        Plot the energy dependence of Thomson optical depth integrated over radius.
        '''
        dL = 1  # 積分ステップ（kpc）
        CM = ClusterManager(cluster_name='perseus', rmin=0, rmax=1000)
        interp_function = self.thomson_scattering_tau_field(
            Emin=2000, Emax=8000, dn=1000, rmax=1000, dL=dL,
            f_ne=CM.Cluster.data.f_ne
        )

        # 半径とエネルギーの配列を定義
        r = np.linspace(0, 1000, 1000)  # kpc
        e = np.linspace(2000, 8000, 1000)  # eV

        # メッシュ生成
        Ri, Ei = np.meshgrid(r, e, indexing='xy')

        # τ(r, E) を取得
        tau_re = interp_function((Ri, Ei))  # shape: (1000, 1000)

        # 半径方向に積分 → τ(E)
        tau_E = np.trapz(tau_re, r, axis=1)  # エネルギーごとの光学的厚み

        # プロット
        R = RadiationField()
        R.plot_style('single')

        R.ax.plot(e, tau_E, color='black')
        R.ax.set_xlabel('Energy (eV)')
        R.ax.set_ylabel(r'Integrated Optical Depth $\tau(E)$')
        R.ax.set_title(f'Thomson Optical Depth vs Energy\n Integrated over 0–1000 kpc')

        R.fig.savefig('Thomson_tau_energy_profile.pdf', dpi=300, transparent=True)
        plt.show()

    def tau_L(self,ne,T,Z,L):
        L = L * u.Mpc
        ne = ne * u.cm**(-3)
        tau = self.sigma_0 * Z* self.Fe_H * ne/1.16 * L * self.iz(T)
        print(f'iz = {self.iz(T)}')
        print(f'tau = {tau.to("")}')
        return tau.to('')

    def calc_tau_L(self,ne,T,Z,L,line_state='w',Ef=None):
        L  = L * u.Mpc
        ne = ne * u.cm**(-3)

        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)
        self.f  = self.atom.f
        self.iz = self.atom.iz

        self.Sigma_0_natural_delE(E0=self.atom.line_energy,E=self.atom.line_energy+Ef)
        print(self.atom.line_energy)
        #self.Sigma_0(T=T,E0=self.atom.line_energy,v=0)
        tau = self.sigma_0 * Z* self.Fe_H * ne/1.16 * L * self.iz(T)
        print(f'iz = {self.iz(T)}')
        print(f'tau = {tau.to("")}')
        print(f'prob = {1 - np.exp(-tau)}')
        return tau.to('')

    def integrated_tau_E0(self, f_ne, f_T, f_Z, max_radius, line_state='w', debug_mode=True):
        """
        Calculate the tau of resonance scattering.
        """

        self.atom = AtomicDataManager("atomic_data.hdf5")
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)

        self.f = self.atom.f
        self.iz = self.atom.iz
        line_energy = self.atom.line_energy

        def integrand(r):
            try:
                T = f_T(r)
                ne = f_ne(r)
                Z = f_Z(r)
                ion_frac = self.iz(T)
                sigma = self.Sigma_0(T=T, E0=line_energy, v=0)

                val = (sigma * Z * self.Fe_H * ne * u.cm**(-3) / 1.16 * ion_frac)

                result = val.to('1/kpc').value  # 単位換算後、スカラー値を取得

                if np.isnan(result) or np.isinf(result):
                    print(f"Warning: nan/inf at r = {r}, skipping")
                    return 0.0

                return result

            except Exception as e:
                print(f"Exception at r = {r}: {e}")
                return 0.0

        if debug_mode == True:
            def integrand(r):
                try:
                    print(f"\n---- r = {r:.4f} ----")

                    T = f_T(r)
                    print(f"f_T({r}) = {T}")

                    ne = f_ne(r)
                    print(f"f_ne({r}) = {ne}")

                    Z = f_Z(r)
                    print(f"f_Z({r}) = {Z}")

                    ion_frac = self.iz(T)
                    print(f"iz({T}) = {ion_frac}")

                    sigma = self.Sigma_0(T=T, E0=line_energy, v=0)
                    print(f"Sigma_0(T={T}) = {sigma}")

                    val = sigma * Z * self.Fe_H * ne * u.cm**(-3) / 1.16 * ion_frac
                    print(f"val (before to) = {val}")

                    result = val.to('1/kpc').value
                    print(f"result = {result}")

                    if np.isnan(result):
                        print("→ ⚠️ result is NaN")
                    elif np.isinf(result):
                        print("→ ⚠️ result is Inf")

                    return result

                except Exception as e:
                    print(f"❌ Exception at r = {r}: {e}")
                    return 0.0


        integral, _ = quad(integrand, 0, max_radius)
        print(f'Integral = {integral}')
        return integral

    def calc_tau_L_E(self, ne=0.01, T=5.0, Z=0.4, L=0.1, line_state='w', mode="tau"):
        from scipy.signal import fftconvolve
        L = L * u.Mpc
        ne = ne * u.cm**(-3)
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)

        tau_factor_dL = (
            ne / 1.12
            * self.Fe_H
            * Z
            * self.atom.iz(T)
            * L/10
            * np.sqrt(np.pi)
            * const.h
            * self.r_e_SI
            * const.c
            * self.atom.f
        ).to("eV").value

        tau_factor = (
            ne / 1.12
            * self.Fe_H
            * Z
            * self.atom.iz(T)
            * L
            * np.sqrt(np.pi)
            * const.h
            * self.r_e_SI
            * const.c
            * self.atom.f
        ).to("eV").value

        print(f"tau_factor = {tau_factor}")
        print(f"iz = {self.atom.iz(T)}")
        print(f"f = {self.atom.f}")

        R = RadiationField()
        R.plot_style("single")

        E = np.linspace(6690, 6710, 1000)
        dx = E[1] - E[0]

        C = ClusterManager(cluster_name="dummy", rmin=0, rmax=99.9, division=1000,
                        ne=ne, kT=T, Z=Z)
        print("L")
        print(L.to("kpc").value)

        prob = self.resonance_scattering_probability_field_natulal(
            line_state=line_state,
            rmax=99.9,
            Emin=6690,
            Emax=6710,
            dn=1000,
            dL=(L/10).to("kpc").value,
            f_ne=C.Cluster.data.f_ne,
            f_T=C.Cluster.data.f_T,
            f_Z=C.Cluster.data.f_Z,
            velocity=0
        )

        tau = -np.log(1 - prob((np.full(len(E), 100), E)))
        prob_v = prob((np.full(len(E), 100), E))

        if mode == "tau":
            R.ax.plot(E, tau, label=rf"$\Delta E = {self.deltaE}$")
        else:
            R.ax.plot(E, prob_v, label=rf"$\Delta E = {self.deltaE}$")

        print(f"e func area = {np.sum(tau * dx)}, 0.3eV")

        E0 = 6700.4
        sigma = 2.15
        kernel_radius = 5 * sigma
        kernel_x = np.arange(E0 - kernel_radius, E0 + kernel_radius + dx, dx)
        kernel = 1 / (np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((kernel_x - E0) / sigma) ** 2)
        tau_conv = fftconvolve(tau, kernel, mode="same") * dx
        R.ax.plot(E, tau_conv, label="Numerical Convolution", linestyle="--")

        for delE in [0.3, 2.15*np.sqrt(2)]:
            C_delE = ClusterManager(cluster_name="dummy", rmin=0, rmax=99.9, division=1000,
                                    ne=ne, kT=T, Z=Z)
            prob_delE = self.resonance_scattering_probability_field_delE(
                line_state=line_state,
                rmax=99.9,
                Emin=6690,
                Emax=6710,
                dn=1000,
                dL=L.to("kpc").value,
                f_ne=C_delE.Cluster.data.f_ne,
                f_T=C_delE.Cluster.data.f_T,
                f_Z=C_delE.Cluster.data.f_Z,
                velocity=0,
                delE=delE
            )

            print(f"iz = {self.atom.iz(T)}")
            print(f"f = {self.atom.f}")

            tau_delE = -np.log(1 - prob_delE((np.full(len(E), 100), E)))
            prob_v_delE = prob_delE((np.full(len(E), 100), E))

            if mode == "tau":
                R.ax.plot(E, tau_delE, label=rf"$\Delta E = {delE}$")
                R.ax.plot(E, tau_delE * 0.45, label=rf"$\Delta E = {delE}, 0.45$")
            else:
                R.ax.plot(E, prob_v_delE, label=rf"$\Delta E = {delE}$")

        print(f"e func area = {np.sum(tau * dx)}, 2.9eV")

        S = Simulation(cluster_name="dummy", step_dist=L.to("kpc").value)
        all_photon, scattered_photon = S.return_photon()
        ratio = len(scattered_photon) / len(all_photon)
        bins_edge = np.linspace(6690, 6710, 1000)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n, bins = np.histogram(scattered_photon, bins=bins_edge)
        prob_sim = scattered_photon_n / all_photon_n
        bins_center = 0.5 * (bins[1:] + bins[:-1])
        tau_sim = -np.log(1 - prob_sim)

        if mode == "tau":
            R.ax.step(bins_center, tau_sim, color="black", where="mid", label="Simulation")
        else:
            R.ax.step(bins_center, prob_sim, color="black", where="mid", label="Simulation")

        print(f"Ratio = {ratio}")

        R.ax.legend()
        plt.show()

        GeneralFunction().gaussian_fitting_with_plot_xy(bins_center[tau_sim>1e-5], tau_sim[tau_sim>1e-5])

    def tau_comp_sim(self, ne=0.01, T=5.0, Z=0.4, L=100, v=0, line_state='w', idx=-1):
        
        L = L * u.kpc
        ne = ne * u.cm**(-3)

        P = PlotManager()
        S = Simulation(cluster_name="dummy", step_dist=L.to("kpc").value)
        all_photon, scattered_photon = S.return_photon(filename='simulation_perseus_digit.hdf5',idx=idx)
        ratio = len(scattered_photon) / len(all_photon)
        bins_edge = np.linspace(6690, 6710, 250)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n, bins = np.histogram(scattered_photon, bins=bins_edge)
        prob_sim = scattered_photon_n / all_photon_n
        bins_center = 0.5 * (bins[1:] + bins[:-1])
        tau_sim = -np.log(1 - prob_sim)
        P.axes[0].step(bins_center, tau_sim, color="black", where="mid", label="Simulation")
        
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)

        print(f"iz = {self.atom.iz(T)}")
        print(f"f = {self.atom.f}")

        E = np.linspace(6690, 6710, 1000)

        delE = self.DeltaE(T=T, E0=self.atom.line_energy, v=v)

        for delE in [delE]:
            C_delE = ClusterManager(cluster_name="dummy", rmin=0, rmax=99.9, division=1000,
                                    ne=ne, kT=T, Z=Z)
            prob_delE = self.resonance_scattering_probability_field_delE(
                line_state=line_state,
                rmax=99.9,
                Emin=6690,
                Emax=6710,
                dn=1000,
                dL=L.to("kpc").value,
                f_ne=C_delE.Cluster.data.f_ne,
                f_T=C_delE.Cluster.data.f_T,
                f_Z=C_delE.Cluster.data.f_Z,
                velocity=0,
                delE=delE
            )

            tau_delE = -np.log(1 - prob_delE((np.full(len(E), 100), E)))
            prob_v_delE = prob_delE((np.full(len(E), 100), E))
            lab = rf"$\Delta E = {delE.value:.2f}$"+" eV" + f"\n $ (kT={T}$"+r" keV," + f"$v={v}$" + " km/s)"
            P.axes[0].plot(E, tau_delE, label=lab, color='crimson')
        
        P.axes[0].set_xlabel('Energy (eV)', fontsize=12)
        P.axes[0].set_ylabel(r'$\tau$', fontsize=12)
        P.axes[0].legend(fontsize=10)
        P.fig.savefig(f'tau_comp_sim_{v}.pdf', dpi=300, transparent=True)
        plt.show()

    def tau_comp_sim2(self, ne=0.01, T=5.0, Z=0.4, L=100, v=0, line_state='w', idx=-1):
        L = L * u.kpc
        ne = ne * u.cm**(-3)

        P = PlotManager()
        S = Simulation(cluster_name="dummy", step_dist=L.to("kpc").value)
        all_photon, scattered_photon = S.return_photon(filename='simulation_perseus_digit.hdf5', idx=idx)

        bins_edge = np.linspace(6690, 6710, 40)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n, _ = np.histogram(scattered_photon, bins=bins_edge)
        bins_center = 0.5 * (bins[1:] + bins[:-1])

        # 過剰に0のところは無視 or small number で置換
        mask = all_photon_n > 0
        prob_sim = np.zeros_like(bins_center)
        prob_sim[mask] = scattered_photon_n[mask] / all_photon_n[mask]

        # τsimとその誤差
        tau_sim = np.zeros_like(prob_sim)
        delta_tau_sim = np.zeros_like(prob_sim)
        tau_sim[mask] = -np.log(1 - prob_sim[mask])
        delta_tau_sim[mask] = np.sqrt(prob_sim[mask] / ((1 - prob_sim[mask]) * all_photon_n[mask]))

        P.axes[0].errorbar(bins_center[mask], tau_sim[mask], yerr=delta_tau_sim[mask],
                        fmt='o', color="black", label="Simulation", markersize=5)

        # === 理論分布の計算 ===
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)

        print(f"iz = {self.atom.iz(T)}")
        print(f"f = {self.atom.f}")

        E = bins_center
        delE = self.DeltaE(T=T, E0=self.atom.line_energy, v=v)

        C_delE = ClusterManager(cluster_name="dummy", rmin=0, rmax=100, division=1000,
                                ne=ne, kT=T, Z=Z)
        prob_delE = self.resonance_scattering_probability_field_delE(
            line_state=line_state,
            rmax=1000,
            Emin=6690,
            Emax=6710,
            dn=1000,
            dL=L.to("kpc").value,
            f_ne=C_delE.Cluster.data.f_ne,
            f_T=C_delE.Cluster.data.f_T,
            f_Z=C_delE.Cluster.data.f_Z,
            velocity=0,
            delE=delE
        )
        tau_theory = -np.log(1 - prob_delE((np.full(len(E), 100), E)))

        lab = rf"$\Delta E = {delE.value:.2f}$ eV" + f"\n$kT={T}$ keV, $v={v}$ km/s"
        P.axes[0].plot(E, tau_theory, label=lab, color='crimson')

        # === τsim 誤差と理論との一致度評価 ===
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_sim[mask] = -np.log(1 - prob_sim[mask])
            delta_tau_sim[mask] = np.sqrt(prob_sim[mask] / ((1 - prob_sim[mask]) * all_photon_n[mask]))

        # エラーが0にならないように保護
        min_error = 1e-3
        delta_tau_sim = np.maximum(delta_tau_sim, min_error)

        # 理論が有限かつ散乱確率が1未満で、統計数も十分な点だけ使う
        valid = (
            mask
            & np.isfinite(tau_theory)
            & (prob_sim < 0.999)
            & (all_photon_n > 10)
        )

        # χ² 計算
        chi_squared = np.sum(((tau_sim[valid] - tau_theory[valid]) / delta_tau_sim[valid]) ** 2)
        dof = np.sum(valid)
        reduced_chi2 = chi_squared / dof if dof > 0 else np.nan
        # === p値の計算 ===
        p_value = chi2.sf(chi_squared, df=dof) if dof > 0 else np.nan

        # === 出力 ===
        print(f"Chi² = {chi_squared:.2f}")
        print(f"reduced Chi² = {reduced_chi2:.2f}")
        print(f"p-value = {p_value:.3g}")
        # === ラベルなど整備 ===
        P.axes[0].set_xlabel('Energy (eV)', fontsize=12)
        P.axes[0].set_ylabel(r'$\tau$', fontsize=12)
        P.axes[0].legend(fontsize=10)
        P.fig.savefig(f'tau_comp_sim_{v}.pdf', dpi=300, transparent=True)
        plt.show()

    def tau_comp_dummy_cluster(self, ne=0.01, T=5.0, Z=0.4, L=100, v=200, line_state='w', idx=-1):
        from scipy.stats import chi2
        import matplotlib.pyplot as plt

        L = L * u.kpc
        ne = ne * u.cm**(-3)

        # 上下2段、下段1/2サイズ、x軸共有
        P = PlotManager(subplot_shape=(2, 1), height_ratios=[2, 1], sharex=True)
        S = Simulation(cluster_name="dummy", step_dist=L.to("kpc").value)
        all_photon, scattered_photon = S.return_photon(filename='simulation_dummy_digit.hdf5', idx=idx)

        bins_edge = np.linspace(6690, 6710, 80)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n, _ = np.histogram(scattered_photon, bins=bins_edge)
        bins_center = 0.5 * (bins[1:] + bins[:-1])

        mask = all_photon_n > 0
        prob_sim = np.zeros_like(bins_center)
        prob_sim[mask] = scattered_photon_n[mask] / all_photon_n[mask]

        tau_sim = np.zeros_like(prob_sim)
        delta_tau_sim = np.zeros_like(prob_sim)
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_sim[mask] = -np.log(1 - prob_sim[mask])
            delta_tau_sim[mask] = np.sqrt(prob_sim[mask] / ((1 - prob_sim[mask]) * all_photon_n[mask]))

        delta_tau_sim = np.maximum(delta_tau_sim, 1e-3)

        # === 理論値の計算 ===
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)
        print(f"iz = {self.atom.iz(T)}")
        print(f"f = {self.atom.f}")

        E = bins_center
        delE = self.DeltaE(T=T, E0=self.atom.line_energy, v=v)

        C_delE = ClusterManager(cluster_name="dummy", rmin=0, rmax=100, division=1000,
                                ne=ne, kT=T, Z=Z)
        prob_delE = self.resonance_scattering_probability_field_delE(
            line_state=line_state,
            rmax=1000,
            Emin=6690,
            Emax=6710,
            dn=1000,
            dL=L.to("kpc").value,
            f_ne=C_delE.Cluster.data.f_ne,
            f_T=C_delE.Cluster.data.f_T,
            f_Z=C_delE.Cluster.data.f_Z,
            velocity=0,
            delE=delE
        )
        tau_theory = -np.log(1 - prob_delE((np.full(len(E), 100), E)))

        # === 有効ビンの条件 ===
        valid = (
            mask
            & np.isfinite(tau_theory)
            & (prob_sim < 0.999)
            & (all_photon_n > 10)
        )

        # === 一致評価 ===
        abs_diff = np.abs(tau_sim[valid] - tau_theory[valid])
        non_abs_diff = tau_sim[valid] - tau_theory[valid]
        max_diff = np.max(abs_diff)
        n_within = np.sum(abs_diff < 0.01)
        n_total = np.sum(valid)
        ratio_within = n_within / n_total if n_total > 0 else 0

        print(f"最大差 |τ_sim - τ_theory| = {max_diff:.4f}")
        print(f"0.01以内に一致したビン数: {n_within} / {n_total} ({ratio_within:.1%})")

        # === χ² + p値 ===
        chi_squared = np.sum(((tau_sim[valid] - tau_theory[valid]) / delta_tau_sim[valid]) ** 2)
        dof = np.sum(valid)
        reduced_chi2 = chi_squared / dof if dof > 0 else np.nan
        p_value = chi2.sf(chi_squared, df=dof) if dof > 0 else np.nan
        print(f"Chi² = {chi_squared:.2f}")
        print(f"reduced Chi² = {reduced_chi2:.2f}")
        print(f"p-value = {p_value:.3g}")

        # === 上段プロット：τ本体 ===
        P.ax.errorbar(bins_center[valid], tau_sim[valid], yerr=delta_tau_sim[valid],
                    fmt='o', color="black", label="Simulation", markersize=5)
        lab = rf"$\Delta E = {delE.value:.2f}$ eV" + f"\n$kT={T}$ keV, $v={v}$ km/s"
        P.ax.plot(E, tau_theory, label=lab, color='crimson')

        # === 下段プロット：残差 ===
        P.ax2.errorbar(E[valid], non_abs_diff, yerr=delta_tau_sim[valid],
               fmt='o', color='black', markersize=5)
        P.ax2.axhline(0.01, color='red', linestyle='--', label='0.01 threshold')
        P.ax2.axhline(-0.01, color='red', linestyle='--')
        P.ax2.set_ylabel(r"$|\tau_{\rm sim} - \tau_{\rm theory}|$")
        P.ax2.set_xlabel("Energy (eV)")
        P.ax2.legend()

        # === 軸調整と保存 ===
        P.ax.tick_params(labelbottom=False)
        P.ax.set_ylabel(r'$\tau$', fontsize=12)
        P.ax.legend(fontsize=10)

        P.fig.tight_layout()
        P.fig.savefig(f'tau_comp_sim3_{v}.pdf', dpi=300, transparent=True)
        plt.show()

    def tau_comp_cluster(self, cluster_name='perseus', L=100, v=200, line_state='w', idx=-1):
        from scipy.stats import chi2
        import matplotlib.pyplot as plt

        L = L * u.kpc

        # 上下2段、下段1/2サイズ、x軸共有
        P = PlotManager(subplot_shape=(2, 1), height_ratios=[2, 1], sharex=True)
        S = Simulation(cluster_name=cluster_name, step_dist=L.to("kpc").value)
        all_photon, scattered_photon = S.return_photon(filename='simulation_perseus_test_center.hdf5', idx=idx)

        bins_edge = np.linspace(6690, 6710, 80)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n, _ = np.histogram(scattered_photon, bins=bins_edge)
        bins_center = 0.5 * (bins[1:] + bins[:-1])

        mask = all_photon_n > 0
        prob_sim = np.zeros_like(bins_center)
        prob_sim[mask] = scattered_photon_n[mask] / all_photon_n[mask]

        tau_sim = np.zeros_like(prob_sim)
        delta_tau_sim = np.zeros_like(prob_sim)
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_sim[mask] = -np.log(1 - prob_sim[mask])
            delta_tau_sim[mask] = np.sqrt(prob_sim[mask] / ((1 - prob_sim[mask]) * all_photon_n[mask]))

        delta_tau_sim = np.maximum(delta_tau_sim, 1e-3)

        # === 理論値の計算 ===
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)

        E = bins_center

        C_delE = ClusterManager(cluster_name=cluster_name, rmin=0, rmax=830, division=400)
        prob_delE = self.resonance_scattering_tau_dL_field(
            line_state=line_state,
            rmax=1000,
            Emin=6690,
            Emax=6710,
            dn=1000,
            dL=L.to("kpc").value,
            f_ne=C_delE.Cluster.data.f_ne,
            f_T=C_delE.Cluster.data.f_T,
            f_Z=C_delE.Cluster.data.f_Z,
            velocity=0
        )
        tau_E_func = S.generate_tau_E_function(tau_func=prob_delE, rmin=0, rmax=830,E_array=E)
        tau_theory = tau_E_func(E)

        # === 有効ビンの条件 ===
        valid = (
            mask
            & np.isfinite(tau_theory)
            & (prob_sim < 0.999)
            & (all_photon_n > 10)
        )

        # === 一致評価 ===
        abs_diff = np.abs(tau_sim[valid] - tau_theory[valid])
        non_abs_diff = tau_sim[valid] - tau_theory[valid]
        max_diff = np.max(abs_diff)
        n_within = np.sum(abs_diff < 0.01)
        n_total = np.sum(valid)
        ratio_within = n_within / n_total if n_total > 0 else 0

        print(f"最大差 |τ_sim - τ_theory| = {max_diff:.4f}")
        print(f"0.01以内に一致したビン数: {n_within} / {n_total} ({ratio_within:.1%})")

        # === χ² + p値 ===
        chi_squared = np.sum(((tau_sim[valid] - tau_theory[valid]) / delta_tau_sim[valid]) ** 2)
        dof = np.sum(valid)
        reduced_chi2 = chi_squared / dof if dof > 0 else np.nan
        p_value = chi2.sf(chi_squared, df=dof) if dof > 0 else np.nan
        print(f"Chi² = {chi_squared:.2f}")
        print(f"reduced Chi² = {reduced_chi2:.2f}")
        print(f"p-value = {p_value:.3g}")

        # === 上段プロット：τ本体 ===
        P.ax.errorbar(bins_center[valid], tau_sim[valid], yerr=delta_tau_sim[valid],
                    fmt='o', color="black", label="Simulation", markersize=5)
        lab = rf"{cluster_name}"
        P.ax.plot(E, tau_theory, label=lab, color='crimson')

        # === 下段プロット：残差 ===
        P.ax2.errorbar(E[valid], non_abs_diff, yerr=delta_tau_sim[valid],
               fmt='o', color='black', markersize=5)
        P.ax2.axhline(0.01, color='red', linestyle='--', label='0.01 threshold')
        P.ax2.axhline(-0.01, color='red', linestyle='--')
        P.ax2.set_ylabel(r"$|\tau_{\rm sim} - \tau_{\rm theory}|$")
        P.ax2.set_xlabel("Energy (eV)")
        P.ax2.legend()

        # === 軸調整と保存 ===
        P.ax.tick_params(labelbottom=False)
        P.ax.set_ylabel(r'$\tau$', fontsize=12)
        P.ax.legend(fontsize=10)

        P.fig.tight_layout()
        P.fig.savefig(f'tau_comp_{cluster_name}_{v}.pdf', dpi=300, transparent=True)
        plt.show()
     
    def multi_sigma(self, line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], f_ne=None, f_T=None, f_Z=None, velocity=0):
        combine_prob = []
        for e,line in enumerate(line_list):
            combine_prob.append(self.resonance_scattering_sigma_field_return_sigma(line_state=line,f_ne=f_ne,f_T=f_T,f_Z=f_Z,velocity=velocity))
        return combine_prob

    def calc_theoretical_RS_prob(self, ne, kT, Z, E, line):
        '''
        Calculate the theoretical probability of resonance scattering.
        '''
        # Calculate the mean free path
        tau = self.calc_tau_L(ne, kT, Z, 1)
        # Calculate the probability
        prob = 1 - np.exp(-tau)
        return prob

    def calc_theoretical_TS_prob(self, ne, E, L):
        '''
        Calculate the theoretical probability of Thomson scattering.
        '''

        L = L * u.Mpc
        ne = ne * u.cm**(-3)
        tau = (self.thomson_scattering(E) * ne * L).to('')
        # Calculate the probability
        prob = 1 - np.exp(-tau)
        print(f'tau = {tau}')
        print(f'prob = {prob}')
        return prob

    def _plot_RS_sigma(self, state='w'):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 10
        CM = ClusterManager(cluster_name='perseus', rmin=0.1, rmax=1000, division=50)
        atom = AtomicDataManager(atomic_data_file)
        atom.load_line_data(state=state)
        Emin = atom.line_energy-10
        Emax = atom.line_energy+10
        interp_function = self.resonance_scattering_probability_field_natulal(line_state=state, rmax=1000, Emin=Emin, Emax=Emax, dn=1000, dL=dL, f_ne=CM.Cluster.data.f_ne, f_T=CM.Cluster.data.f_T, f_Z=CM.Cluster.data.f_Z, velocity=300)
        r = np.linspace(0, 1000, 1000)
        e = np.linspace(Emin, Emax, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        prob = interp_function((Ri, Ei))
        R = RadiationField()
        R.plot_style('single')
        print(interp_function((8, 6700.4)), interp_function((200, 6700.4)))
        print(interp_function((8, 6700.4)), interp_function((8, 6710.4)))
        prob[prob <= 0] = 1e-10 
        im = R.ax.imshow(prob, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='cividis', aspect='auto', origin='lower')
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability', norm=LogNorm())

        R.ax.set_xlim(0,1000)
        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {dL} kpc')

        R.fig.savefig('RS_probability.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_RS_probability(self, state='w'):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 1
        CM = ClusterManager(cluster_name='perseus', rmin=0.1, rmax=1000, division=50)
        atom = AtomicDataManager(atomic_data_file)
        atom.load_line_data(state=state)
        Emin = atom.line_energy-10
        Emax = atom.line_energy+10
        interp_function = self.resonance_scattering_tau_field(line_state=state, rmax=1000, Emin=Emin, Emax=Emax, dn=1000, dL=dL, f_ne=CM.Cluster.data.f_ne, f_T=CM.Cluster.data.f_T, f_Z=CM.Cluster.data.f_Z, velocity=0)
        r = np.linspace(0, 1000, 1000)
        e = np.linspace(Emin, Emax, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='ij')
        print(interp_function((8, 6700.4)), interp_function((500, 6700.4)))
        print(interp_function((8, 6700.4)), interp_function((8, 6710.4)))

        print(interp_function((6700.4, 8)), interp_function((6700.4, 500)))
        prob = interp_function((Ri, Ei))
        R = RadiationField(2,8,0.5e-3)
        R.plot_style('single')
        # プロットの描画
        im = R.ax.imshow(prob.T, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='viridis', aspect='auto', origin='lower')

        # カラーバーの追加
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability', norm=LogNorm())

        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {dL} kpc')
        #R.ax.set_xlim(0,200)
        R.fig.savefig('RS_probability.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_RS_tau(self, state='w'):
        '''
        Plot the probability of Thomson scattering (log scale).
        '''
        dL = 1
        CM = ClusterManager(cluster_name='dummy', rmin=0.1, rmax=1000, division=100)
        atom = AtomicDataManager(atomic_data_file)
        atom.load_line_data(state=state)
        R = RadiationField()
        R.plot_style('single')
        Emin = atom.line_energy-10
        Emax = atom.line_energy+10
        for v in [0]:
            interp_function = self.resonance_scattering_tau_field(line_state=state, rmax=1000, Emin=Emin, Emax=Emax, dn=1000, dL=dL, f_ne=CM.Cluster.data.f_ne, f_T=CM.Cluster.data.f_T, f_Z=CM.Cluster.data.f_Z, velocity=v)
            r = np.linspace(0, 1000, 1000)
            e = np.linspace(Emin, Emax, 1000)
            Ri, Ei = np.meshgrid(r, e, indexing='ij')
            prob = interp_function((Ri, Ei))
            sum_p = np.sum(prob, axis=-1)
            R.ax.plot(r, sum_p, label=f'v = {v} km/s')
        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel(r'$\tau_{RS}$')
        R.ax.set_yscale('log')
        R.ax.set_title(r'$\tau_{RS}$ (dL=1 kpc) of Resonant Scattering')
        R.fig.savefig('RS_tau.png', dpi=300)
        plt.show()

    def plot_setting(self):
        from matplotlib.ticker import LogFormatter
        import matplotlib.ticker as ticker
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
        self.ax  = self.fig.add_subplot(221)
        self.ax2  = self.fig.add_subplot(222)
        self.ax3  = self.fig.add_subplot(223)
        self.ax4  = self.fig.add_subplot(224)
        spine_width = 2  # スパインの太さ
        for spine in self.ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in self.ax2.spines.values():
            spine.set_linewidth(spine_width)
        for spine in self.ax3.spines.values():
            spine.set_linewidth(spine_width)
        for spine in self.ax4.spines.values():
            spine.set_linewidth(spine_width)
        self.ax.tick_params(axis='both',direction='in',width=1.5)
        self.ax2.tick_params(axis='both',direction='in',width=1.5)
        self.ax3.tick_params(axis='both',direction='in',width=1.5)
        self.ax4.tick_params(axis='both',direction='in',width=1.5)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax3.grid(linestyle='dashed')
        self.ax4.grid(linestyle='dashed')
        self.fig.align_labels()
        r = np.arange(0,1000,1)
        ne = self.pks_ne(r)
        Te = self.pks_Te(r)
        Z  = np.full(len(r),self.pks_Z(r))
        self.ion_fraction(stage=25)
        iz = self.iz(Te)
        self.resonance_scattering_probability(rmax=1000,Emin=6690, Emax=6710,dL=1)
        contour = self.ax3.contourf(self.Ri, self.Ei, self.prob, levels=100, cmap='viridis')
        cbar = self.fig.colorbar(contour, ax=self.ax3)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        self.ax3.set_ylabel('Energy (eV)')
        self.ax.plot(r, ne, color='blue', lw=2)
        self.ax2.plot(r, Te, color='blue', lw=2)
        #self.ax3.plot(r, Z, color='blue', lw=2)
        self.ax4.plot(r, iz, color='blue', lw=2)
        self.ax.set_ylabel(r'$n_e \ \rm (cm^{-3})$')
        self.ax2.set_ylabel(r'$T_e \ \rm (keV)$')
        #self.ax3.set_ylabel('Abundance')
        self.ax4.set_ylabel('Ion fraction Fe XXV')
        self.ax.set_xlabel(r'$\rm Radius \ (kpc)$')
        self.ax2.set_xlabel(r'$\rm Radius \ (kpc)$')
        self.ax3.set_xlabel(r'$\rm Radius \ (kpc)$')
        self.ax4.set_xlabel(r'$\rm Radius \ (kpc)$')
        self.fig.tight_layout()
        self.fig.savefig('RS_simulation_setting.pdf',dpi=300,transparent=True)
        plt.show()

    def _interp_test(self):
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        X_x, Y_y = np.meshgrid(x, y, indexing='xy')
        data_grid = tuple([x, y])
        def z(x, y):
            return x - 5 * (y-3)
        
        Z = z(X, Y)
        print(Z)
        print(Z.shape)
        interp_function = RegularGridInterpolator(data_grid, Z, bounds_error=False, fill_value=None)
        print(interp_function((10,0)))
        print(interp_function((0,10)))
        Z_interp = interp_function((X_x, Y_y))
        # indexingはijにしないと二次元プロットでおかしくなる
        # しかしinterpの関数と引数はどっちで生成しても同じ
        plt.imshow(Z_interp, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
        plt.colorbar(label='Z')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Interpolated Data')
        plt.show()

    def _plot_mfp(self, Emin=6697.5, Emax=6702.5, rmax=1000, dL=10):
        CM = ClusterManager(cluster_name='perseus', rmin=0, rmax=1000, division=100)
        interp_func = self.resonance_scattering_mfp_field(
            line_state='w',
            rmax=rmax,
            Emin=Emin,
            Emax=Emax,
            dn=1000,
            dL=dL,
            f_ne=CM.Cluster.data.f_ne,
            f_T=CM.Cluster.data.f_T,
            f_Z=CM.Cluster.data.f_Z,
            velocity=0
        )
        # interp_func = self.resonance_scattering_probability_field(
        #     line_state='w',
        #     rmax=rmax,
        #     Emin=Emin,
        #     Emax=Emax,
        #     dn=1000,
        #     dL=dL,
        #     f_ne=CM.Cluster.data.f_ne,
        #     f_T=CM.Cluster.data.f_T,
        #     f_Z=CM.Cluster.data.f_Z,
        #     velocity=0
        # )
        r = np.linspace(0, rmax, 1000)
        e = np.linspace(Emin, Emax, 1000)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        prob = interp_func((Ri, Ei))
        #prob[prob <= 0] = 1e-10
        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        im = R.ax.imshow(prob*1e-3, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                        cmap='plasma_r', aspect='auto', origin='lower')
        # カラーバーの追加
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Mean Free Path (kpc)', norm=LogNorm())
        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        # R.ax.set_xscale('log')
        R.ax.set_title('Mean Free Path of Resonant Scattering')
        R.fig.savefig('RS_mfp.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_tau_E(self, Emin=6690, Emax=6710, rmax=1000, dL=1000):
        CM = ClusterManager(cluster_name='dummy', rmin=0, rmax=1000, division=100)
        # interp_func = self.resonance_scattering_mfp_field(
        #     line_state='w',
        #     rmax=rmax,
        #     Emin=Emin,
        #     Emax=Emax,
        #     dn=1000,
        #     dL=dL,
        #     f_ne=CM.Cluster.data.f_ne,
        #     f_T=CM.Cluster.data.f_T,
        #     f_Z=CM.Cluster.data.f_Z,
        #     velocity=0
        # )
        interp_func = self.resonance_scattering_tau_field(
            line_state='w',
            rmax=rmax,
            Emin=Emin,
            Emax=Emax,
            dn=1000,
            dL=dL,
            f_ne=CM.Cluster.data.f_ne,
            f_T=CM.Cluster.data.f_T,
            f_Z=CM.Cluster.data.f_Z,
            velocity=0
        )

        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        E = np.linspace(Emin, Emax, 1000)
        r = np.full_like(E, 1)
        tau = interp_func((r, E))
        R.ax.plot(E, tau)
        R.ax.set_ylabel('Tau')
        R.ax.set_xlabel('Energy (eV)')
        # R.ax.set_xscale('log')
        #R.ax.set_title('Mean Free Path of Resonant Scattering')
        R.fig.savefig('RS_tau.pdf', dpi=300, transparent=True)
        plt.show()

    def _plot_mfp_E(self, Emin=6690, Emax=6710, rmax=1000, dL=100):
        CM = ClusterManager(cluster_name='dummy', rmin=0, rmax=100, division=100)
        interp_func = self.resonance_scattering_mfp_field(
            line_state='w',
            rmax=rmax,
            Emin=Emin,
            Emax=Emax,
            dn=1000,
            dL=dL,
            f_ne=CM.Cluster.data.f_ne,
            f_T=CM.Cluster.data.f_T,
            f_Z=CM.Cluster.data.f_Z,
            velocity=0
        )

        R = RadiationField()
        R.plot_style('single')
        # プロットの描画
        E = np.linspace(Emin, Emax, 1000)
        r = np.full_like(E, 1)
        mfp = interp_func((r, E))
        R.ax.plot(E, mfp)
        R.ax.set_ylabel('mfp')
        R.ax.set_xlabel('Energy (eV)')
        R.ax.set_yscale('log')
        #R.ax.set_title('Mean Free Path of Resonant Scattering')
        R.fig.savefig('RS_mfp.pdf', dpi=300, transparent=True)
        plt.show()

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

class AtomicDataManager:
    def __init__(self, filename, atomdb_version="3.0.9", load_furukawa=False):
        self.filename = filename
        self.k_Boltzmann_keV = const.k_B.value * const.e.value * 1e-3
        pyatomdb.util.switch_version(atomdb_version)
        self.atomdb_version = atomdb_version
        #! : This is tempolary setting. Please Remove it finnaly.
        self.load_furukawa = load_furukawa
        print('Atomic Data Manager Initialized.')
        print(f'Furukawa setting is {self.load_furukawa}')

    def calculate_all_lines(self):
        for state in ['w', 'x', 'y', 'z', 'Lya2', 'Lya1', 'Heb1', 'Heb2', 'u', 'r', 't', 'q']:
            self.line_manager(Z=26, state=state)
        self.ion_fraction(Z=26, stage=self.z1, Tmin=0, Tmax=20, dT=500)
        for state in ['w', 'x', 'y', 'z', 'Lya2', 'Lya1', 'Heb1', 'Heb2', 'u', 'r', 't', 'q']:
            self.load_line_data(state=state)
            self.load_ion_fraction(Z=26,stage=self.z1)

    def ion_fraction(self, Z=26, stage=25, Tmin=0, Tmax=20, dT=100):
        temperatures_keV = np.linspace(Tmin, Tmax, dT)
        ion_fractions = np.zeros((dT, Z+1))
        return_ionbal = pyatomdb.apec.return_ionbal
        for i, T in enumerate(temperatures_keV):
            ion_fractions[i, :] = return_ionbal(Z, T, teunit='keV', datacache=False)
        for i in range(Z+1):
            self.save_ion_fraction(Z, i, temperatures_keV, ion_fractions[:,i])
        self.iz = interp1d(temperatures_keV, ion_fractions[:, stage-1])

    def save_ion_fraction(self, Z, stage, temperatures_keV, ion_fractions):
        """ion_fractionの結果をHDF5ファイルに保存"""
        with h5py.File(self.filename, 'a') as f:
            group_name = f"{self.atomdb_version}/ion_fraction/Z_{Z}_stage_{stage}"
            if group_name in f:
                del f[group_name]  # 既存のデータセットを削除してから再作成

            grp = f.create_group(group_name)
            grp.create_dataset("temperatures_keV", data=temperatures_keV)
            grp.create_dataset("ion_fractions", data=ion_fractions)
            print(f"Ion fraction data saved in {self.filename} under group {group_name}")

    def load_ion_fraction(self, Z, stage, return_fraction=False):
        """保存されたion_fractionデータをHDF5ファイルから読み込み"""
        with h5py.File(self.filename, 'r') as f:
            group_name = f"{self.atomdb_version}/ion_fraction/Z_{Z}_stage_{stage-1}"
            if group_name in f:
                temperatures_keV = f[group_name]["temperatures_keV"][...]
                ion_fractions = f[group_name]["ion_fractions"][...]

                # インターポレーション関数を再作成
                self.iz = interp1d(temperatures_keV, ion_fractions[:])
                #print(f"Ion fraction data loaded from {self.filename} under group {group_name}")
            else:
                print(f"No ion fraction data found in {self.filename} for Z={Z}, stage={stage-1}")
        #! : This is tempolary setting. Please Remove it finnaly.
        if self.load_furukawa:
            if stage == 26:
                file = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/perseus/atomdb_v3.0.8_angr/fraction_H_file.dat'
            elif stage == 25:
                file = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/perseus/atomdb_v3.0.8_angr/fraction_He_file.dat'
            elif stage == 24:
                file = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/perseus/atomdb_v3.0.8_angr/fraction_Li_file.dat'
            f = np.loadtxt(file)
            self.iz = interp1d(
                                f[:, 0],           # x
                                f[:, 1],           # y
                                kind='linear',     # または 'nearest', 'cubic' など必要に応じて
                                bounds_error=False,
                                fill_value=(f[0, 1], f[-1, 1])  # 左端と右端はそれぞれ最近接のy値を使う
                            )
        if return_fraction:
            return self.iz

    def oscillator_strength(self,Z=26,z1=25,upperlev=7,lowerlev=1):
        self.f = pyatomdb.atomdb.get_oscillator_strength(Z, z1, upperlev, lowerlev, datacache=False)
        print('----------------------')
        print('Oscillator Strength')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {upperlev}, lowerlev = {lowerlev}')
        print(f'f = {self.f}')

    def line_manager(self, Z=26, state='w'):
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(1, 80, 500)
        
        # stateごとのz1とupの組み合わせを辞書で定義
        state_map = {
            'w': (Z-1, 7), 'x': (Z-1, 6), 'y': (Z-1, 5), 'z': (Z-1, 2),
            'Lya2': (Z, 3), 'Lya1': (Z, 4), 'Heb1': (Z-1, 13), 'Heb2': (Z-1, 11),
            'u': (Z-2, 45), 'r': (Z-2, 47), 't': (Z-2, 50), 'q': (Z-2, 48)
        }
        
        # z1とupの設定
        if state in state_map:
            z1, up = state_map[state]
        else:
            print('state is not defined')
            return  # 定義外のstateの場合は処理を終了
        self.z1 = z1
        ldata = sess.return_line_emissivity(kTlist, Z, z1, up, 1)
        self.line_energy = ldata['energy'] * 1e+3
        atom_data = np.array(pyatomdb.atomdb.get_data(Z, z1, 'LA')[1].data)
        level_mask = (atom_data['UPPER_LEV'] == up) & (atom_data['LOWER_LEV'] == 1)
        self.einstein_a = float(atom_data['EINSTEIN_A'][level_mask][0]) / u.s
        self.natural_width = (self.einstein_a * const.h / (2 * np.pi)).to(u.eV)
        print('----------------------')
        print('Line Energy')
        print(f'state = {state}')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {up}')
        print(f'line energy = {self.line_energy} eV')
        print(f'einstein_a = {self.einstein_a}')
        print(f'natural width = {self.natural_width}')
        self.oscillator_strength(Z=Z, z1=z1, upperlev=up, lowerlev=1)
        self.save_line_data(state, Z, z1, up)

    def save_line_data(self, state, Z, z1, upperlev):
        """line_managerとoscillator_strengthの結果をHDF5ファイルに保存"""
        with h5py.File(self.filename, 'a') as f:
            group_name = f"{self.atomdb_version}/line_data/{state}"
            if group_name in f:
                print(f"{group_name} is already exis.")
                print(f'{group_name} is delete, and saved new dataset')
                del f[group_name]  # 既存のデータセットを削除してから再作成

            grp = f.create_group(group_name)
            grp.attrs["Z"] = Z
            grp.attrs["z1"] = z1
            grp.attrs["upperlev"] = upperlev
            grp.create_dataset("line_energy", data=self.line_energy)
            grp.attrs["einstein_a"] = self.einstein_a.value
            grp.attrs["natural_width"] = self.natural_width.value
            grp.attrs["oscillator_strength"] = self.f
            print(f"Line data saved in {self.filename} under group {group_name}")

    def load_line_data(self, state):
        """保存されたline_managerとoscillator_strengthの結果をHDF5ファイルから読み込み、selfに設定"""
        with h5py.File(self.filename, 'r') as f:
            group_name = f"{self.atomdb_version}/line_data/{state}"
            if group_name in f:
                grp = f[group_name]
                self.Z = grp.attrs["Z"]
                self.z1 = grp.attrs["z1"]
                self.upperlev = grp.attrs["upperlev"]
                self.line_energy = grp["line_energy"][()]
                self.einstein_a = grp.attrs["einstein_a"] * u.s**-1
                self.natural_width = grp.attrs["natural_width"] * u.eV
                self.f = grp.attrs["oscillator_strength"]
                #print(f"Line data loaded and set to self from {self.filename} under group {group_name}")
            else:
                print(f"No line data found in {self.filename} for state={state}")

    def plot_ion_fraction(self):
        P = PlotManager()
        temp = np.logspace(0,np.log10(10),100)
        Z = 26
        self.load_ion_fraction(Z, 25)
        stage = [23,24,25,26]
        stage_num = ['Fe XXIII','Fe XXIV','Fe XXV','Fe XXVI']
        self.atomdb_version = "3.0.8"
        for e,stage in enumerate(stage):
            self.load_ion_fraction(Z, stage)
            P.axes[0].plot(temp, self.iz(temp),"-.", label=rf'{stage_num[stage-23]}', color=cm.plasma([e/4]))
        
        stage = [23,24,25,26]
        self.atomdb_version = "3.0.9"
        for e,stage in enumerate(stage):
            self.load_ion_fraction(Z, stage)
            P.axes[0].plot(temp, self.iz(temp), label=rf'{stage_num[stage-23]}', color=cm.plasma([e/4]))
        stage = [23,24,25,26]
        self.atomdb_version = "3.1.0"
        for e,stage in enumerate(stage):
            self.load_ion_fraction(Z, stage)
            P.axes[0].plot(temp, self.iz(temp),"--", label=rf'{stage_num[stage-23]}', color=cm.plasma([e/4]))
        P.axes[0].set_xlabel('Temperature (keV)')
        P.axes[0].set_ylabel('Ion Fraction')
        #P.ax.set_xscale('log')
        #P.axes[0].set_yscale('log')
        #P.axes[0].set_ylim(0.01,1)
        P.axes[0].legend()
        #ax.set_title(f'Ion Fraction of Fe XXV (Z={self.Z}, stage={self.z1})')
        P.fig.savefig('ion_fraction.pdf', dpi=300)
        plt.show()

class RadiationField:
    '''
    Radiation Field class
    This class is used to generate energy spectrum of the cluster of galaxies.
    Now, it only supports CIE plasma of APEC model using pyatomdb.
    For using lpgs abundance set (please see the xspec abudance), I edit abundance file of the pyatomdb.
    So, now, # !:this class only support local environment.
    The bapec spectrum is generated from each emissivity.
    The thermal broadening is valid, and the gaussian sigma is calculated from the setting temperature.
    The photon sampling is done by rejection sampling method.
    The generated photons are saved in HDF5 file.
    In the Simultaion class, the photon generation is done by loading saved HDF5 file. 

    If you want to use this class for the simulation,
    save_apec_spectrum_hdf5() : The spectrum of the cluster in each radius is saved in HDF5 file.

    parameters
    ----------
    cluster_name : str
        The name of the cluster.
    Emin : float
        The minimum energy of the spectrum in keV.
        Emin is used for sampling spectrum generation.
    Emax : float
        The maximum energy of the spectrum in keV.
        Emax is used for sampling spectrum generation.
    dE : float
        The energy bin width in keV.
        dE is used for sampling spectrum generation.
    abundance_name : str
        The name of the abundance set. (default: lpgs)
    
    '''
    def __init__(self, Emin, Emax, dE, abundance_name='lpgs', atomdb_version='3.0.9') -> None:
        self.abundance_name = abundance_name
        self.Emin           = Emin 
        self.Emax           = Emax
        self.dE             = dE
        self.atomdb_version = atomdb_version
        pyatomdb.util.switch_version(atomdb_version)
        # self.linefile = f"$ATOMDB/apec_v{self.atomdb_version}_line.fits"
        # self.cocofile = f"$ATOMDB/apec_v{self.atomdb_version}_coco.fits"
        
    def apec_field(self,kT=5, turbulent_velocity=0, abundance=0.4):
        '''
        Generate APEC spectrum using pyatomdb.
        The BVAPEC model is used to generate the spectrum.

        Parameters
        ----------
        kT : float
            The temperature of the plasma in keV.
        turbulent_velocity : float
            The turbulent velocity in km/s.
        abundance : float
            The abundance of the plasma in solar units.
        '''
        sess = pyatomdb.spectrum.CIESession(abundset=self.abundance_name)
        sess.set_abundset(self.abundance_name)
        el_list = [2, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20]
        # ! : the elements number is defined to vapec. If you use other model, you need to change the number of elements.
        # !  He : 2, C : 6, N : 7, O : 8, Ne : 10, Mg : 12, Al : 13, Si : 14, S : 16, Ar : 18, Ca : 20, Fe : 26, Ni : 28
        sess.set_abund(el_list, 0.4)
        #el_list = [2, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28]
        sess.set_abund([26, 28], abundance)
        sess.set_broadening(thermal_broadening=True,broaden_limit=1e-100,velocity_broadening=turbulent_velocity,thermal_broaden_temperature=kT)
        ebins = np.arange(self.Emin, self.Emax+self.dE, self.dE)
        sess.set_response(ebins, raw=True)
        spec = sess.return_spectrum(kT)
        return sess.ebins_out, spec

    # def apec_field_abund_Fe(self,kT=5, turbulent_velocity=0, abundance=0.4):
    #     '''
    #     Generate APEC spectrum using pyatomdb.
    #     The BVAPEC model is used to generate the spectrum.

    #     Parameters
    #     ----------
    #     kT : float
    #         The temperature of the plasma in keV.
    #     turbulent_velocity : float
    #         The turbulent velocity in km/s.
    #     abundance : float
    #         The abundance of the plasma in solar units.
    #     '''
    #     sess = pyatomdb.spectrum.CIESession(abundset=self.abundance_name)
    #     sess.set_abundset(self.abundance_name)
    #     el_list = [2, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20]
    #     # ! : the elements number is defined to vapec. If you use other model, you need to change the number of elements.
    #     # !  He : 2, C : 6, N : 7, O : 8, Ne : 10, Mg : 12, Al : 13, Si : 14, S : 16, Ar : 18, Ca : 20, Fe : 26, Ni : 28
    #     sess.set_abund(el_list, 0.4)  # Updated to use el_list
    #     #el_list = [2, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28]
    #     sess.set_abund([26,28], abundance)
    #     sess.set_broadening(thermal_broadening=True,broaden_limit=1e-100,velocity_broadening=turbulent_velocity,thermal_broaden_temperature=kT)
    #     ebins = np.arange(self.Emin, self.Emax+self.dE, self.dE)
    #     sess.set_response(ebins, raw=True)
    #     spec = sess.return_spectrum(kT)
    #     return sess.ebins_out, spec

    def integrate_spectrum(self,ebins_out, spec):
        ebin_centers = (ebins_out[:-1] + ebins_out[1:]) / 2.0
        total_flux = simpson(x=ebin_centers, y=spec)
        return total_flux     

    def generate_random_numbers_rejection(self, ebins, spec, sampling_Emin, sampling_Emax, size=1000):
        ebins_center = (ebins[:-1] + ebins[1:]) / 2.0
        spec_mask = (ebins_center >= sampling_Emin) & (ebins_center <= sampling_Emax)

        bins_center = ebins_center[spec_mask]
        spec = spec[spec_mask]

        max_spec = np.max(spec)
        random_numbers = np.empty(size)
        count = 0

        while count < size:
            rand_energy = np.random.uniform(bins_center[0], bins_center[-1], size - count)
            rand_uniform = np.random.uniform(0, max_spec, size - count)

            spec_interp = np.interp(rand_energy, bins_center, spec)
            accepted = rand_uniform < spec_interp
            num_accepted = np.sum(accepted)

            random_numbers[count:count + num_accepted] = rand_energy[accepted]
            count += num_accepted

        return random_numbers

    def generate_apec_photon(self, kT=4, Emin=6.60, Emax=6.75, dE=1e-3, velocity=0, abundance=0.4, size=1000, filename='photon.hdf5', cluster_name='None', method='None', radius=None, save=True):
        size = int(size)
        ebins, spec = self.apec_field(kT=kT, turbulent_velocity=velocity, abundance=abundance)
        # norm = self.integrate_spectrum(ebins, spec)
        photons = self.generate_random_numbers_rejection(ebins=ebins, spec=spec, sampling_Emin=Emin, sampling_Emax=Emax, size=size)
        if save == True:
            self.save_photons_to_hdf5(filename, kT, Emin, Emax, dE, velocity, abundance, size, photons, cluster_name, method, radius)
        return photons

    def generate_apec_spectrum_from_hdf5(self, size=1000, sampling_Emin=2.0, sampling_Emax=8.0, turbulent_velocity=0, radius=None, radial_list=None, seed_filename=None):
        '''
        Generate photons from APEC spectrum saved in HDF5 file.
        If the filename is not specified, it will use the default filename 'apec_spectrum.hdf5'.
        '''
        ebins, spec  = self.load_spectrum_from_hdf5(radius=radius, turbulent_velocity=turbulent_velocity, radial_list=radial_list, seed_filename=seed_filename)
        photons      = self.generate_random_numbers_rejection(ebins=ebins, spec=spec, sampling_Emax=sampling_Emax, sampling_Emin=sampling_Emin, size=size)
        return photons    

    def save_apec_spectrum_hdf5(self, Cluster, turbulent_velocity=0, save_root_name=None):
        '''
        Generate cluster of galaxies spectrum for photon sampling, and save it to HDF5 file.
        The cluster parameters are set in the ClusterManager class.
        # ?: The cluster parameters are defined local file. So, this script only works in local environment. 

        Parameters
        ----------
        Cluster : ClusterManager()
            The cluster manager class.
        turbulent_velocity : float
            The turbulent velocity in km/s.
        Z : float
            The abundance of the plasma in solar units.
        cluster_name : str
            The name of the cluster.
        '''
        CM = Cluster
        if save_root_name is None:
            save_root_name = f'{CM.cluster_name}_apec_spectrum'
        radial_list = CM.divide_cluster_sphere_linear()
        print(radial_list)
        for e,r in enumerate(radial_list):
            print(f'Generating photons at r = {r} kpc')
            print(f'Temperature  {CM.Cluster.data.f_T(r)} keV')
            print(f'Abundance    {CM.Cluster.data.f_Z(r)}')
            ebin, spec = self.apec_field(kT=CM.Cluster.data.f_T(r), turbulent_velocity=turbulent_velocity, abundance=CM.Cluster.data.f_Z(r))   
            self.save_spectrum_to_hdf5(filename=f'{save_root_name}_{turbulent_velocity}.hdf5', spectrum=spec, ebin=ebin, cluster_name=CM.cluster_name, kT=CM.Cluster.data.f_T(r), turbulent_velocity=turbulent_velocity, abundance=CM.Cluster.data.f_Z(r), method='linear', radius=r, num=e)

    def make_spectrum_multi_v(self, cluster_name='perseus', rmax=830, rmin=0, division=400):
        #for v in [0, 100, 150, 200, 250, 300]:
        CM = ClusterManager(cluster_name=cluster_name, rmin=rmin, rmax=rmax, division=division) 
        for v in [0]:
            self.save_apec_spectrum_hdf5(Cluster=CM, turbulent_velocity=v, save_root_name=None)

    def find_nearest_index(self, array, value):
        array = np.array(array)  # NumPy配列に変換
        index = (np.abs(array - value)).argmin()  # 差の絶対値が最小のインデックスを取得
        return index

    def load_spectrum_from_hdf5(self, radius=823.3519, turbulent_velocity=150, seed_filename='apec_spectrum', radial_list=None):
        hdf5_file = f'{seed_filename}_{turbulent_velocity}.hdf5'  
        print(f'Loading spectrum from {hdf5_file}')
        spectrum_index = self.find_nearest_index(radial_list, radius)
        with h5py.File(hdf5_file, 'r') as f:
            print(f'Loading spectrum at r = {radial_list[spectrum_index]} kpc')
            print(f'Loading spectrum at index = {spectrum_index}')
            dataset_name = f"spectrum_{spectrum_index}"
            spec = f[f'{dataset_name}/spectrum'][...]
            ebin = f[f'{dataset_name}/ebin'][...]
        return ebin, spec

    def load_photon_radial_distrubution(self, filename='data.hdf5', attributes={'cluster_name': 'perseus'}):
        links = self.get_matching_datasets(filename=filename, attributes=attributes)
        data = self.load_dataset_by_link(filename=filename, dataset_link=links)
        return data, filename, attributes

    def save_photons_to_hdf5(self, filename, kT, Emin, Emax, dE, velocity, abundance, size, photons, cluster_name='None', method='None', radius=None):
        with h5py.File(filename, 'a') as f:
            dataset_name = f"photons_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset = f.create_dataset(dataset_name, data=photons)
            dataset.attrs['kT']                = kT
            dataset.attrs['kT_unit']           = 'keV' 
            dataset.attrs['Emin']              = Emin
            dataset.attrs['Emax']              = Emax
            dataset.attrs['dE']                = dE
            dataset.attrs['E_unit']            = 'keV' 
            dataset.attrs['velocity']          = velocity
            dataset.attrs['velocity_unit']     = 'km/s'
            dataset.attrs['abundance']         = abundance
            dataset.attrs['size']              = size
            dataset.attrs['cluster_name']      = cluster_name
            dataset.attrs['method']            = method
            dataset.attrs['radius']            = radius
            dataset.attrs['radius_unit']       = 'kpc'
            dataset.attrs['timestamp']         = datetime.now().isoformat()

            print(f"Photons and metadata saved to {filename} in dataset {dataset_name}") 

    def save_spectrum_to_hdf5(self, filename, spectrum, ebin, cluster_name, kT, turbulent_velocity, abundance, method='None', radius=None, num=0):
        file_exists = os.path.exists(filename)

        with h5py.File(filename, 'a') as f:
            if not file_exists:
                # ファイル新規作成時に属性を追加
                f.attrs['abundance_table'] = self.abundance_name
                f.attrs['atomdb_version']  = self.atomdb_version
            else:
                # 既存ファイルなら、属性が一致するかチェック
                abundance_table = f.attrs.get('abundance_table')
                atomdb_version  = f.attrs.get('atomdb_version')

                if abundance_table != self.abundance_name or atomdb_version != self.atomdb_version:
                    raise ValueError(
                        f"❌ 保存中止: ファイルの属性と現在の設定が一致しません。\n"
                        f"  - file: abundance_table={abundance_table}, atomdb_version={atomdb_version}\n"
                        f"  - now : abundance_table={self.abundance_name}, atomdb_version={self.atomdb_version}"
                    )

            # データセットの作成
            dataset_name = f"spectrum_{num}"
            dataset = f.create_group(dataset_name)
            f.create_dataset(f'{dataset_name}/spectrum', data=spectrum)
            f.create_dataset(f'{dataset_name}/ebin', data=ebin)

            # 属性を付与
            dataset.attrs['kT']                = kT
            dataset.attrs['kT_unit']           = 'keV' 
            dataset.attrs['Emin']              = self.Emin
            dataset.attrs['Emax']              = self.Emax
            dataset.attrs['dE']                = self.dE
            dataset.attrs['E_unit']            = 'keV' 
            dataset.attrs['velocity']          = turbulent_velocity
            dataset.attrs['velocity_unit']     = 'km/s'
            dataset.attrs['abundance']         = abundance
            dataset.attrs['cluster_name']      = cluster_name
            dataset.attrs['method']            = method
            dataset.attrs['radius']            = radius
            dataset.attrs['radius_unit']       = 'kpc'
            dataset.attrs['timestamp']         = datetime.now().isoformat()

            print(f"✅ Photons and metadata saved to {filename} in dataset {dataset_name}")

    def generate_rebin_photons(self,cluster_name='perseus', rmin=0.1, rmax=1000, Emin=2, Emax=10, dE=0.5e-3, velocity=0, division=10, size=1000, filename='apec_photon_energy_distribution.hdf5', method='linear'):
        CM = ClusterManager(cluster_name=cluster_name, rmin=rmin, rmax=rmax, division=division)
        if method == 'linear':
            radius = CM.divide_cluster_sphere_linear()
        elif method == 'log':
            radius = CM.divide_cluster_sphere_logscale()
        for r in radius:
            print(f'Generating photons at r = {r} kpc')
            self.generate_apec_photon(kT=CM.Cluster.data.f_T(r), Emin=Emin, Emax=Emax, dE=dE, abundance=CM.Cluster.data.f_Z(r), velocity=velocity, size=size, filename=filename, cluster_name=cluster_name, method='linear', radius=r)

    def get_matching_datasets(self, filename='photon.hdf5', attributes={'Emin': 6.5, 'Emax': 8.0,'velocity': 150}):
        hdf5 = HDF5Manager(filename)
        links = hdf5.get_matching_datasets(attributes=attributes)
        print(links)
        data = hdf5.load_dataset_by_link(dataset_links=links)
        print(data)
        print(len(data))
        plt.hist(data[:], bins=100, histtype='step')
        Detector().fits2xspec(pha = data[:]*1e3, name = 'r0_v150')

    def plot_style(self,style='double'):
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

    def make_fits(self):
        D = Detector(cluster="perseus")
        for i in range(1, 9):
            photon = self.generate_apec_spectrum_from_hdf5(size=int(1e5),sampling_Emin=6.5,sampling_Emax=8.0,radius=0,turbulent_velocity=150,radial_list=[0],seed_filename='perseus_apec_spectrum')
            D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.001, name=f'spectrum_test_{i}', TEStype='TMU542', Datatype='PHA', pha=photon*1e3)

    def abundance_test_and_make_fits(self,abund=0.6):
        '''
        Check the abundance table and effect of the Fe Ni abundance, and others.
        '''
        D = Detector(cluster="dummy")
        photons = self.generate_apec_photon(kT = 5.5, Emin=6.60, Emax=6.75, dE=0.5e-3, velocity=0, abundance=abund, size=1e5, filename='photon.hdf5', cluster_name='None', method='None', radius=None, save=False)
        D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'spectrum_abundance_test3', TEStype='TMU542', Datatype='PHA', pha=photons*1e3)

    def _for_comparison_furukawa_spectrum(self, size=int(1e5)):
        '''
        This script make perseus spectrum divided by 0-0.5arcmin.
        0.5 arcmin is about 10.5 kpc.
        The seed spectrum is 1-11kpc and, the generated spectrum is sampled by ne square distribution. 
        '''
        CM = ClusterManager(cluster_name='perseus', rmin=0, rmax=10.5)
        radial_photon = InitialPhotonGenerator(CM, None).generate_random_ne_squared(size=size)
        rebine_radius_ref = CM.divide_cluster_sphere_linear()
        rebined_radial_photon = CM.assign_nearest_value(radial_photon, rebine_radius_ref)
        # plt.hist(rebined_radial_photon, bins=1000,histtype="step")
        # plt.show()
        # self.save_apec_spectrum_hdf5(Cluster=CM, turbulent_velocity=150, save_root_name='perseus_apec_spectrum_comp_furukawa')
        rebinned_radial = np.unique(rebined_radial_photon)
        energy_list = []
        for r in rebinned_radial:
            r_length = len(np.where(rebined_radial_photon == r)[0])
            energy_e = self.generate_apec_spectrum_from_hdf5(size=r_length,sampling_Emin=6.5,sampling_Emax=8.0,radius=r,turbulent_velocity=150,radial_list=rebinned_radial,seed_filename='perseus_apec_spectrum_comp_furukawa')
            energy_list.append(energy_e)
        energy_list = np.hstack(energy_list)
        plt.hist(energy_list, bins=1000,histtype="step")
        plt.show()
        D = Detector(cluster="dummy")
        D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'spectrum_comp_furukawa', TEStype='TMU542', Datatype='PHA', pha=energy_list*1e3)

    def load_qdp(self, filename):
        f = np.loadtxt(filename)
        energy = f[:,0]
        cnt = f[:,2]
        return energy, cnt

    def generate_random_numbers_rejection_center(self, ebins, spec, size=1000):
        '''
        Do not use bins edge. 
        '''
        bins_center = ebins
        spec = spec

        max_spec = np.max(spec)
        random_numbers = np.empty(size)
        count = 0

        while count < size:
            rand_energy = np.random.uniform(bins_center[0], bins_center[-1], size - count)
            rand_uniform = np.random.uniform(0, max_spec, size - count)

            spec_interp = np.interp(rand_energy, bins_center, spec)
            accepted = rand_uniform < spec_interp
            num_accepted = np.sum(accepted)

            random_numbers[count:count + num_accepted] = rand_energy[accepted]
            count += num_accepted

        return random_numbers
    
    def sampling_from_qdp(self, filename, size):
        '''
        Spectrum sampling by using furukawa spectrum.
        Please see the furukawa master paper for more details.
        The mother spectrum is
        半径 [arcmin]       スペクトルファイル
        ------------------------------------------------
        0.0–0.5            bapec_model_v308_all_vt150_01.qdp
        0.5–1.0            bapec_model_v308_all_vt150_02.qdp
        1.0–1.5            bapec_model_v308_all_vt150_03.qdp
        1.5–2.0            bapec_model_v308_all_vt150_04.qdp
        2.0–2.5            bapec_model_v308_all_vt150_05.qdp
        2.5–3.0            bapec_model_v308_all_vt150_06.qdp
        3.0–3.5            bapec_model_v308_all_vt150_07.qdp
        3.5–4.0            bapec_model_v308_all_vt150_08.qdp
        4.0–4.5            bapec_model_v308_all_vt150_09.qdp
        4.5–5.0            bapec_model_v308_all_vt150_10.qdp
        5.0–10.0           bapec_model_v308_all_vt150_11.qdp
        10.0–20.0          bapec_model_v308_all_vt150_21.qdp
        20.0–30.0          bapec_model_v308_all_vt150_31.qdp
        30.0–40.0          bapec_model_v308_all_vt150_41.qdp
        '''
        energy, cnt = self.load_qdp(filename)
        photon = self.generate_random_numbers_rejection_center(energy, cnt, size) # [keV]
        return photon

    def _link_check(self, filename='data.hdf5'):
        hdf5 = HDF5Manager(filename)
        links = hdf5.get_matching_datasets(attributes={'Emin': 6})
        print(links)
        data = hdf5.load_dataset_by_link(dataset_link=links)
        #hdf5.print_hdf5_contents()
        print(data[2000])

    def _plot_field_cor_sim(self,size=100000):
        self.plot_style('single')
        size = int(size)
        ebins, spec = self.apec_field(Emin=6.5, Emax=8.0)
        bins_center = (ebins[:-1] + ebins[1:]) / 2.0
        norm = self.integrate_spectrum(ebins, spec)
        print(norm)
        photons = self.generate_random_numbers_rejection(ebins, spec, size=size)
        self.ax.hist(photons, bins=ebins, density=True ,histtype='step',color='red', label='Generated Photons')
        self.ax.step(bins_center, spec/norm, linestyle='--' , color='black',where='mid' , label='APEC Model Spectrum', alpha=0.5)
        self.ax.set_xlabel('Energy (keV)')
        self.ax.set_ylabel('Normalized Emissivity and Photon Count')
        self.ax.set_title(f'APEC Spectrum and Generated Photons\n size={size}')
        self.ax.legend()
        #self.ax.set_yscale('log')
        plt.show()
        self.fig.savefig('APEC_spectrum_and_generated_photons.pdf',dpi=300,transparent=True)

    def _plot_apec_field(self,kT=5.5):
        self.plot_style('single')
        ebins, spec = self.apec_field(kT, 0, 0.6)
        bins_center = (ebins[:-1] + ebins[1:]) / 2.0
        norm = self.integrate_spectrum(ebins, spec)
        print(norm)
        self.ax.step(bins_center, spec/norm, where='mid', color='black', label='kT = 5.5 keV, abund All 0.6')
        ebins, spec = self.apec_field(kT, 0, 0.6)
        norm = self.integrate_spectrum(ebins, spec)
        print(norm)
        self.ax.step(bins_center, spec/norm, where='mid', color='red', label='kT = 5.5 keV, abund Fe,Ni 0.6, others 0.4')
        self.ax.set_xlabel('Energy (keV)')
        self.ax.set_ylabel('Emissivity (ph cm$^3$ s$^{-1}$ keV$^{-1}$)')
        self.ax.set_title('APEC Spectrum')
        plt.legend()
        plt.show()

    def _load_and_plot_spec(self, filename='apec_spectrum', radius=0.1, velocity=0, radial_list=[0.1], Emin=6.5, Emax=8.0):
        photon = self.generate_apec_spectrum_from_hdf5(radius=radius, velocity=velocity, radial_list=radial_list, Emin=Emin, Emax=Emax)
        self.plot_style('single')
        self.ax.hist(photon, bins=1000, histtype='step')
        self.ax.set_xlabel('Energy (keV)')
        self.ax.set_ylabel('Photon Count')
        self.ax.set_title(f'Photon Energy Distribution\n radius={radius} kpc, velocity={velocity} km/s')
        plt.show()

class IonField:
    '''
    Ion Field Class
    Ion velocity field is generated by thermal and turbulent velocity.
    Each velocity is sampled from Gaussian distribution.
    The velocity is in the unit of km/s.
    The thermal velocity is calculated by the formula:
    v_thermal = sqrt(kT/(const.u*A))
    where kT is the temperature in keV, A is the atomic mass number.
    The turbulent velocity is sampled from Gaussian distribution with the root mean square (vrms).
    The turbulent velocity is calculated by the formula:
    v_turbulent = vrms
    where vrms is the root mean square velocity.
    The velocity is the sum of the thermal and turbulent velocity.
    The velocity is in the unit of km/s.
    The conversion from the ion rest frame to the observed frame is calculated by the formula:
    E_observed = E_rest * (1 + (v @ photon_vector/const.c))
    where E_rest is the energy in the ion rest frame, v is the velocity vector, photon_vector is the photon vector, const.c is the speed of light.
    The conversion from the observed frame to the ion rest frame is calculated by the formula:
    E_rest = E_observed * (1 - (v @ photon_vector/const.c))
    where E_observed is the energy in the observed frame, v is the velocity vector, photon_vector is the photon vector, const.c is the speed of light.
    The velocity is in the unit of km/s.
    The temperature is in the unit of keV.
    The energy is in the unit of eV.
    The atomic mass number is in the unit of amu.
    '''
    def __init__(self):
        self.iron_atomic_weight = 55.845
        self.thermal_velocity   = None
        self.turbulant_velocity = None
        self.velocity_photon_direction = None
        self.keV2K = const.e.value*1e+3/const.k_B.value
        self.K2keV = 1/self.keV2K
        pass

    def vtoeV(self, v, E0):
        '''
        Convert velocity to energy
        v: velocity (km/s)
        E0: center energy (eV)
        return: sigma energy (eV)
        '''
        E0 = E0 * u.eV
        v = v * u.km / u.s
        return (v*E0/(const.c)).to('eV').value

    def turbulent_velocity_sampling(self, vrms):
        """
        Generate a random sample from a Gaussian distribution
        Parameters
        ----------
        vrms : float
            Root mean square 1D velocity.

        """
        self.turbulant_velocity = np.random.normal(loc=0, scale=vrms, size=3)

    def thermal_velocity_sampling(self, kT, A):
        kT = kT * u.keV
        A  = A
        self.thermal_velocity = np.random.normal(loc=0, scale=np.sqrt(kT/(const.u*A)).to('km/s').value, size=3)

    def thermal_turbulent_velocity_sampling(self, kT, v):
        if hasattr(kT, 'unit') == True:
            kT = kT
        else:
            kT = kT * u.keV
        if hasattr(v, 'unit') == True:
            v = v
        else:
            v  = v * u.km/u.s
        self.velocity = np.random.normal(loc=0, scale=np.sqrt(kT/(const.u*self.iron_atomic_weight) + v**2).to('km/s').value, size=3)

    def sampling_ion_velocity_for_cluster_frame(self, kT, vrms, sigma_nat, A, E0, E):
        G = GeneralFunction()
        E0 = G.ensure_unit(E0, u.eV)
        E  = G.ensure_unit(E, u.eV)
        sigma_nat = G.ensure_unit(sigma_nat, u.eV)
        kT = G.ensure_unit(kT, u.keV)
        vrms = G.ensure_unit(vrms, u.km/u.s)
        # print(f'kT = {kT}, vrms = {vrms}, sigma_nat = {sigma_nat}, A = {A}, E0 = {E0}, E = {E}')
        sigma_v = np.sqrt(kT/(const.u*A) + vrms**2).to('km/s')
        factor_A = 1/sigma_v**2 + (E**2/(const.c**2 * sigma_nat**2)).to((u.km/u.s)**-2)
        factor_B = (E-E0)*E/(const.c*sigma_nat**2)
        center = factor_B/factor_A
        sigma  = 1/np.sqrt(factor_A)
        #print(f'center = {center.to("km/s")}, sigma = {sigma}, sigma_v = {sigma_v}')
        v_photon = np.random.normal(loc=center.to("km/s").value, scale=sigma.to("km/s").value, size=1)
        v_photon_prop = np.random.normal(loc=0, scale=sigma_v.to("km/s").value, size=2)
        #v_photon = ((1-E0/E)*const.c).to('km/s').value
        #v_photon_prop = np.array([90,90])
        self.velocity_photon_direction = np.hstack((v_photon_prop, v_photon))
    
    def maxwell_distribution(self, v, kT, A=1):
        '''
        3d Maxwell-Boltzmann distribution function
        v: velocity (km/s)
        T: temperature (keV)
        A: atomic mass number
        '''
        v  = v * u.km / u.s
        kT = kT * u.keV
        return 4 * np.pi * (v**2) * (const.u*A / (2 * np.pi * kT))**(3/2) * np.exp((-const.u*A * (v**2) / (2 * kT)).to(''))
    
    def maxwell_distribution_1d(self, v, kT, A=1):
        '''
        1d Maxwell-Boltzmann distribution function
        v: velocity (km/s)
        T: temperature (keV)
        A: atomic mass number
        '''
        v = v * u.km / u.s
        kT = kT * u.keV
        return (const.u*A / (2 * np.pi * kT))**(1/2) * np.exp((-const.u*A * (v**2) / (2 * kT)).to(''))

    def velocity_sampling(self, kT, vrms):
        # self.turbulent_velocity_sampling(vrms)
        # self.thermal_velocity_sampling(kT, self.iron_atomic_weight)
        # self.velocity           = self.turbulant_velocity + self.thermal_velocity
        self.thermal_turbulent_velocity_sampling(kT, vrms)
        #self.velocity          = self.turbulant_velocity

    def velocity_sampling_dummy(self):
        self.turbulent_velocity = np.array([0,0,0])
        self.thermal_velocity   = np.array([0,0,0])
        self.velocity           = np.array([0,0,0])
        # self.turbulent_velocity_sampling(0.217/6700.404447285952*const.c.to('km/s').value)
        # self.velocity           = self.turbulant_velocity

    def thermal_turabulant_velocity(self, kT, v):
        if hasattr(kT, 'unit') == True:
            kT = kT
        else:
            kT = kT * u.keV
        if hasattr(v, 'unit') == True:
            v = v
        else:
            v  = v * u.km/u.s
        return np.sqrt(kT/(const.u*self.iron_atomic_weight) + v**2).to('km/s')
    
    def conversion_ion_restframe(self, E, photon_vector):
        if hasattr(E, 'unit') == True:
            E = E
        else:
            E = E * u.eV
        if hasattr(self.velocity, 'unit') == True:
            v = self.velocity
        else:
            v = self.velocity * u.km/u.s
        return (E * (1 - (v @ photon_vector/const.c))).to('eV').value
    
    def conversion_ion_observedframe(self, E, photon_vector):
        if hasattr(E, 'unit') == True:
            E = E
        else:
            E = E * u.eV
        if hasattr(self.velocity, 'unit') == True:
            v = self.velocity
        else:
            v = self.velocity  * u.km/u.s
        return (E * (1 + (v @ photon_vector/const.c))).to('eV').value

    def _plot_turbulent_velocity(self):
        v = np.linspace(-1000, 1000, 1000)
        vrms = np.arange(100,600,100)
        P = PlotManager('single')
        for i in vrms:
            f = self.gaussian_distribution(v, i)
            P.ax.plot(v, f, label=f'vrms = {i}', color=plt.cm.viridis(i/500))
        P.ax.set_xlabel('Velocity (km/s)')
        P.ax.set_ylabel('Probability Density')
        P.ax.set_title('Turbulent Velocity Distribution')
        P.ax.legend()
        P.fig.tight_layout()
        P.fig.savefig('Turbulent_Velocity_Distribution.pdf',dpi=300,transparent=True)
        plt.show()

    def _sampling_test(self):
        photon = Photon()
        G = GeneralFunction()
        for i in range(10000):
            self.sampling_ion_velocity_for_cluster_frame(kT=5, vrms=100, sigma_nat=0.217, A=self.iron_atomic_weight, E0=6700.4, E=6699.6)
            R = photon.make_rotation_matrix(photon.polartocart(photon.random_on_sphere(r=1)))
            if i == 0:
                ion_v = self.velocity_photon_direction
                ion_v_conv = R @ self.velocity_photon_direction
            else:
                ion_v = np.vstack((ion_v, self.velocity_photon_direction))
                ion_v_conv = np.vstack((ion_v_conv, R @ self.velocity_photon_direction))
        ion_v = np.array(ion_v)
        ion_v_conv = np.array(ion_v_conv)
        P = PlotManager((1,3),(12,8))
        G.gaussian_fitting_with_plot(ion_v_conv[:,0],ax=P.axes[0])
        G.gaussian_fitting_with_plot(ion_v_conv[:,1],ax=P.axes[1])
        G.gaussian_fitting_with_plot(ion_v_conv[:,2],ax=P.axes[2])
        plt.show()
        P = PlotManager((1,3),(12,8))
        G.gaussian_fitting_with_plot(ion_v[:,0],ax=P.axes[0])
        G.gaussian_fitting_with_plot(ion_v[:,1],ax=P.axes[1])
        G.gaussian_fitting_with_plot(ion_v[:,2],ax=P.axes[2])
        # plt.hist(ion_v_conv[:,0], bins=100, histtype='step', label='Ion Velocity')
        # plt.hist(ion_v_conv[:,1], bins=100, histtype='step', label='Ion Velocity')
        # plt.hist(ion_v_conv[:,2], bins=100, histtype='step', label='Ion Velocity')
        # plt.xlabel('Velocity (km/s)')
        # plt.ylabel('Counts')
        # plt.title('Ion Velocity Distribution')
        # plt.legend()
        # plt.grid()
        plt.show()

    def gaussian(self, x, a, mu, sigma):
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2)

    def _check_velocity_broadening(self, turb=True, frame="ion", E_rest=None):
        E = 6700.4
        if E_rest is None:
            E_rest = []
            photon = Photon()
            for i in range(0, 5000):
                photon_vector = photon.polartocart(photon.random_on_sphere(r=1))
                if turb == True:
                    self.thermal_turbulent_velocity_sampling(kT=5, v=100)
                else:
                    self.thermal_velocity_sampling(kT=5, A=self.iron_atomic_weight)
                    self.velocity = self.thermal_velocity
                if frame == "ion":
                    E_rest.append(self.conversion_ion_restframe(E, photon_vector))
                elif frame == "observer":
                    E_rest.append(self.conversion_ion_observedframe(E, photon_vector))
        
        E_rest = np.array(E_rest)

        # ヒストグラムの作成（密度として正規化）
        counts, bins = np.histogram(E_rest, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 初期推定値
        a0 = np.max(counts)
        mu0 = np.mean(E_rest)
        sigma0 = np.std(E_rest)

        # フィッティング
        popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[a0, mu0, sigma0])
        a_fit, mu_fit, sigma_fit = popt

        # 描画
        plt.hist(E_rest, bins=100, density=True, histtype="step", label="data")
        x_fit = np.linspace(np.min(E_rest), np.max(E_rest), 1000)
        plt.plot(x_fit, self.gaussian(x_fit, *popt), label=f'Gaussian fit\nσ = {sigma_fit:.2f} eV')
        sigma_fit_km = (sigma_fit * u.eV * const.c / (6700.4 * u.eV)).to('km/s')
        plt.xlabel("E_rest [eV]")
        plt.ylabel("Normalized counts")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Fitted mean (μ): {mu_fit:.2f} eV")
        print(f"Fitted sigma (σ): {sigma_fit:.2f} eV")
        print(f"Fitted σ (km/s): {sigma_fit_km:.2f}")

    def fit_gaussian(self):
        with h5py.File('simulation_pks_digit.hdf5', 'r') as f:
            keyname = list(f.keys())[-1]
            E_rest = f[keyname]['initial_energy'][:]
        # ヒストグラムの作成（密度として正規化）
        counts, bins = np.histogram(E_rest, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 初期推定値
        a0 = np.max(counts)
        mu0 = np.mean(E_rest)
        sigma0 = np.std(E_rest)

        # フィッティング
        popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[a0, mu0, sigma0])
        a_fit, mu_fit, sigma_fit = popt

        # 描画
        plt.hist(E_rest, bins=100, density=True, histtype="step", label="data")
        x_fit = np.linspace(np.min(E_rest), np.max(E_rest), 1000)
        plt.plot(x_fit, self.gaussian(x_fit, *popt), label=f'Gaussian fit\nσ = {sigma_fit:.2f} eV')
        sigma_fit_km = (sigma_fit * u.eV * const.c / (6700 * u.eV)).to('km/s')
        plt.xlabel("E_rest [eV]")
        plt.ylabel("Normalized counts")
        plt.legend()
        plt.grid(linestyle='--')
        #plt.title(f'Gaussian Fit\nμ = {mu_fit:.2f} eV, σ = {sigma_fit:.2f} eV, σ (km/s) = {sigma_fit_km:.2f}')
        plt.tight_layout()
        plt.show()

        return a_fit, mu_fit, sigma_fit

    def _plot_turbulent_velocity_sampling(self, num_samples, vrms):
        samples = self.turbulent_velocity_sampling(num_samples, vrms)
        v = np.linspace(-1000, 1000, 1000)
        f = self.gaussian_distribution(v, vrms)
        norm = simpson(f, v)
        P = PlotManager('single')
        P.ax.hist(samples, bins=v, density=True, histtype='step', label='Samples', color='black')
        P.ax.plot(v, f/norm, label='Target Distribution', color='red')
        P.ax.set_xlabel('Velocity (km/s)')
        P.ax.set_ylabel('Probability Density')
        P.ax.set_title('Turbulent Velocity Sampling')
        P.ax.legend()
        P.fig.tight_layout()
        P.fig.savefig('Turbulent_Velocity_Sampling.pdf',dpi=300,transparent=True)
        plt.show()
    
    def _plot_maxwell_distribution(self):
        P = PlotManager('single')
        v = np.linspace(-1000, 1000, 1000)
        for T in np.arange(3, 9, 1):
            T_keV = T * u.keV
            f = self.maxwell_distribution(v=v, kT=T, A=self.iron_atomic_weight)
            P.ax.plot(v, f, label=f'T = {T_keV}', color=plt.cm.viridis(T_keV/8))
        #f = self.maxwell_distribution(v, T, A=55)
        #plt.plot(v, f)
        P.ax.legend()
        P.ax.set_xlabel('Velocity (km/s)')
        P.ax.set_ylabel('Probability Density')
        P.ax.set_title('Maxwell-Boltzmann Distribution')
        P.fig.savefig('Maxwell_Boltzmann_Distribution.pdf',dpi=300,transparent=True)
        plt.show()  

    def _plot_rejection_sampling(self, num_samples, v_max, kT):
        samples = self.maxwellian_rejection_sampling(num_samples, kT, v_max)
        v = np.linspace(-v_max, v_max, 1000)
        f = self.maxwell_distribution(v, kT, A=self.iron_atomic_weight)
        norm = simpson(f, v)
        P = PlotManager('single')
        P.ax.hist(samples, bins=v, density=True, histtype='step', label='Samples', color='black')
        P.ax.step(v, f/norm, label='Target Distribution', color='red')
        P.ax.set_xlabel('Velocity (m/s)')
        P.ax.set_ylabel('Probability Density')
        P.ax.set_title('Rejection Sampling')
        P.ax.legend()
        P.fig.tight_layout()
        P.fig.savefig('Rejection_Sampling.pdf',dpi=300,transparent=True)
        plt.show()

    def ion_sigma(self, kT, A=55.845):
        kT = kT * u.keV
        return np.sqrt(kT/(const.u*A)).to('km/s')
    
    def calc_turv(self, kT, sigma1D):
        kT = kT * u.keV
        sigma1D = sigma1D * u.km/u.s
        return np.sqrt(sigma1D**2 - (kT/(const.u*self.iron_atomic_weight))).to('km/s')

    def calc_turv_err(self, kT, kT_ep, kT_em, sigma1D, sigma1D_ep, sigma1D_em):
        sigma_th = self.ion_sigma(kT, self.iron_atomic_weight)
        sigma_th_ep = self.ion_sigma(kT+kT_ep, self.iron_atomic_weight)
        sigma_th_em = self.ion_sigma(kT+kT_em, self.iron_atomic_weight)
        sigma_th_str = f'$ {sigma_th.value:.2f}^{{+{sigma_th_ep.value-sigma_th.value:.2f}}}_{{{sigma_th_em.value-sigma_th.value:.2f}}} $ '
        sigma_turv = self.calc_turv(kT, sigma1D)
        sigma_turv_em = self.calc_turv(kT+kT_ep, sigma1D+sigma1D_em)
        sigma_turv_ep = self.calc_turv(kT+kT_em, sigma1D+sigma1D_ep)
        sigma_turv_str = f'$ {sigma_turv.value:.2f}^{{+{sigma_turv_ep.value-sigma_turv.value:.2f}}}_{{{sigma_turv_em.value-sigma_turv.value:.2f}}}$'
        print(sigma_th_str)
        print(sigma_turv_str)

class Simulation:
    '''
    Simulation class for executing the simulation of the cluster.
    Some of the cluster and simulation setting is loaded from initialize parameters.
    
    # ! TODO : Please, adding the docstring for each function.
    '''
    def __init__(self, physics_list=['RS', 'TS'], rmin=0.1, rmax=1000, division=100, Emin = 6500, Emax = 8000, 
                 dn = 1000, size=1000, rs_dE=10, turbulent_velocity=0, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], 
                 cluster_name='perseus', interp_method='linear', rebin_method='linear', step_dist=2, test_mode=False,
                 tau_mode='natural', atomdb_version='3.0.9', abundance_table='lpgs',
                 ne_dummy=0.01, kT_dummy=5.0, Z_dummy=0.4, comparison=False, bulk_velocity=0, bulk_change_radius=100,
                 ):
        '''
        Initialize the simulation parameters.

        Parameters
        ----------
        physics_list : list
            List of physics to be included in the simulation. Default is ['RS', 'TS'].
        rmin : float
            Minimum radius of the cluster in kpc. Default is 0.1.
            rmin is used to photon generation and stopping radius.
            Photon generated between rmin and rmax.
            Each photon stops at the rmax.            
        rmax : float
            Maximum radius of the cluster in kpc. Default is 1000.
            rmax is used to photon generation and stopping radius.
            Photon generated between rmin and rmax.
            Each photon stops at the rmax.
        division : int
            Number of divisions of the cluster. Default is 100.
            The radius is divided into division parts. 
            The cluster parameters, such as electron density, temperature, and metallicity, are calculated at each division.
        Emin : float 
            Minimum energy of the photons in eV. Default is 2000.
        Emax : float
            Maximum energy of the photons in eV. Default is 8000.
        dn : int
            Number of points to interpolate the radial distribution of the . Default is 1000.
        size : int
            Number of photons to be generated. Default is 1000.
        rs_dE : float
            Energy range of the resonance scattering in eV. Default is 5.
        turbulent_velocity : float
            Turbulent velocity of the ions in km/s. Default is 0.
        rs_line_list : list
            List of the resonance scattering lines. Default is ['w','z'].
        cluster_name : str
            Name of the cluster. Default is 'perseus'.
            electron density, temperature, and metallicity functions are defined in the ClusterManager class.
        step_dist : float
            Step distance of the photon in kpc. Default is 10.
        test_mode : bool
            If True, the test mode is activated. Default is False.
        tau_mode : str
            Mode of the optical depth calculation. 
            This parameter is changing the calculation method of the optical depth.
            Default is 'natural'.
            'natural'   : using the gaussian with natural width. The scattering judged in the ion frame. 
                          Thus, the ion velocity is sampling each step. 
            'effective' : using the gaussian with natural width + velocity dispersion. The scattering judged in the cluster frame. 
                          if the photon is scattered, the ion velocity is sampling.
            'mfp'       : using the gaussian with natural width + velocity dispersion. The step size is calculated by the mean free path.
        '''
        self.physics_list       = physics_list
        self.Emin               = Emin
        self.Emax               = Emax
        self.rmin               = rmin
        self.rmax               = rmax
        self.division           = division
        self.dn                 = dn      
        self.step_dist          = step_dist
        self.size               = int(size)
        self.rs_dE              = rs_dE
        self.turbulent_velocity = turbulent_velocity
        self.rs_line_list       = rs_line_list
        self.cluster_name       = cluster_name
        self.interp_method      = interp_method
        self.method             = rebin_method
        self.CM                 = ClusterManager(cluster_name=cluster_name,rmin=rmin,rmax=rmax,division=division,method=interp_method,ne=ne_dummy, kT=kT_dummy, Z=Z_dummy)
        self.attributes         = {'velocity': self.turbulent_velocity}
        self.atomic_number      = 55.845 # !: This script can be used in Iron.
        self.tau_mode           = tau_mode
        self.atomdb_version      = atomdb_version
        self.abundance_table     = abundance_table
        self.spectrum_sampling_file_root = f'{self.cluster_name}_apec_spectrum'

        self.seed_spectrum_Emin = 2.0 # keV
        self.seed_spectrum_Emax = 8.0 # keV
        self.seed_spectrum_dE   = 0.5e-3 # keV
        # ? : for check the each parameter distribution, temporaly
        self.test_mode = test_mode
        self.comparison = comparison
        self.arcmin2kpc_furukawa = 20.818 # from furukawa san s setting
        #if test_mode == True:
        self.velocity_list    = [] # for check the ion velocity distribution, each row is the velocity of direction(x,y,z). 
        self.velocity_prop    = [] # for check the ion velocity of photon direction
        self.energy_ion_rest  = []
        self.photon_direction = []
        self.line_prob_check  = []
        self.random_check     = []

        #! : for bulk velocity setting
        self.bulk_velocity = bulk_velocity
        self.bulk_change_radius = bulk_change_radius

        if self.tau_mode == "mfp":
            #self.mfp_radius = np.geomspace(1e-3, self.rmax, 5000)
            self.mfp_radius = np.linspace(1e-5, self.rmax, 5000)

    def calculate_RS_prob(self):
        '''
        Calculate the optical depth of the resonance scattering.
        '''
        self.initialize_for_tau()
        e = np.linspace(6500,8000,1500)
        e = np.linspace(6675,6725,1500)
        ene = e
        r = np.linspace(0,500,500)
        Ri, Ei = np.meshgrid(r, e, indexing='xy')
        R = RadiationField()
        R.plot_style('single')
        self.rs_line_list = ['w']
        for e,line in enumerate(self.rs_line_list):
            if e == 0:
                line_prob  = self.probability_function_list[line]((Ri,Ei))
            else:
                line_prob += self.probability_function_list[line]((Ri,Ei))
        im = R.ax.imshow(line_prob, extent=[Ri.min(), Ri.max(), Ei.min(), Ei.max()],
                    cmap='plasma', aspect='auto', origin='lower')
        cbar = R.ax.figure.colorbar(im, ax=R.ax, label='Probability', norm=LogNorm())
        atom = AtomicDataManager(atomic_data_file)
        #for state in ['w', 'x', 'y', 'z', 'Lya2', 'Lya1', 'Heb1', 'Heb2', 'u', 'r', 't', 'q']:
        for state in ['w']:
            atom.load_line_data(state)
            #R.ax.axhline(atom.line_energy, color='white', linestyle='--', alpha=0.25)
            #R.ax.text(10, atom.line_energy+1, state, color='white', fontsize=15, va='bottom')

        R.ax.set_xlabel('Radius (kpc)')
        R.ax.set_ylabel('Energy (eV)')
        R.ax.set_title(f'Probability of Thomson Scattering\n dL = {10} kpc')
        #R.ax.set_xscale('log')
        R.ax.set_ylim(6600,6800)
        R.ax.set_xlim(1, 830)
        plt.show()
        print(line_prob.shape)
        plt.plot(ene, np.sum(line_prob, axis=1))
        plt.show()

    def calculate_tau_RS(self):
        '''
        Calculate the optical depth of the resonance scattering.
        '''
        self.initialize_for_tau()
        energy = np.linspace(6690, 6710, 20)
        radius = np.linspace(0.1, 840, 50)
        R = RadiationField()
        R.plot_style('single')
        line_prob_list = Physics().multi_sigma(
            f_ne=self.CM.Cluster.data.f_ne,
            f_T=self.CM.Cluster.data.f_T,
            f_Z=self.CM.Cluster.data.f_Z,
            velocity=self.turbulent_velocity
        )

        combined_results = []
        res = combined_results[0]

        comb = np.array(combined_results)
        plt.show()

#? for thesis tau plot
    def integrate_rs(self):
        turb_list = np.linspace(0, 300, 6)
        for i in range(len(turb_list)):
            self.turbulent_velocity = turb_list[i]
            self.initialize_for_tau_E()
            tau_func_dict = self.probability_function_list_rsmall
            tau_all = self.generate_sum_tau_function(tau_func_dict, self.rmin, self.rmax, self.rs_dE, r_sampling_points=100, kind='linear')
            #E_array = np.linspace(6500, 8000, 10000)
            E_array = np.linspace(6500, 8000, 10000)
            
            plt.plot(E_array, tau_all(E_array), label=f'turb = {self.turbulent_velocity} km/s', color=plt.cm.viridis(i/len(turb_list)))
            plt.xlabel('Energy (eV)')
            plt.ylabel('Optical Depth')
            plt.title('Optical Depth of Resonance Scattering')
            plt.legend()
            plt.grid(linestyle='--')
            plt.semilogy()
            plt.ylim(5e-2, 20)
            #plt.xlim(6680, 6720)
            #plt.show()

    def tau_v_dependence(self, color="black"):
        turb_list = np.linspace(0, 400, 40)
        tau_result = {}
        for i in range(len(turb_list)):
            self.turbulent_velocity = turb_list[i]
            self.initialize_for_tau_E()
            tau_func_dict = self.probability_function_list_rsmall
            tau_all = self.generate_sum_tau_function(tau_func_dict, self.rmin, self.rmax, self.rs_dE, r_sampling_points=100, kind='linear')
            #E_array = np.linspace(6500, 8000, 10000)
            E_array = np.linspace(6500, 8000, 10000)
            for line_list, tau_func_r_e in tau_func_dict.items():
                center_energy = self.line_manager.line_energies[line_list]
                if i == 0:
                    tau_result[line_list] = tau_all(center_energy)
                else:
                    tau_result[line_list] = np.append(tau_result[line_list],tau_all(center_energy))
        for i, line_list in enumerate(tau_result.keys()):
            plt.plot(turb_list, tau_result[line_list], label=f'{self.cluster_name}', color=color) 
        plt.xlabel('Turbulent Velocity (km/s)')
        plt.ylabel('Optical Depth')
        plt.title('Optical Depth of Resonance Scattering')
        plt.legend()
        plt.grid(linestyle='--')
        plt.show()

    def tau_line_center(self, check=False):
        """
        Tau RS of the line center.
        """
        turb_list = [0]
        tau_result = {}
        for i in range(len(turb_list)):
            self.turbulent_velocity = turb_list[i]
            self.initialize_for_tau_E()
            tau_func_dict = self.probability_function_list_rsmall
            if check:
                E_array = np.linspace(6500, 8000, 10000)
                print(tau_func_dict.items())
                for line_list, tau_func_r_e in tau_func_dict.items():
                    tau_f = self.generate_tau_E_function(tau_func_r_e, self.rmin, self.rmax, E_array, r_sampling_points=100, kind='linear')
                    plt.plot(E_array, tau_f(E_array))
                    center_energy = self.line_manager.line_energies[line_list]
                    if i == 0:
                        tau_result[line_list] = tau_f(center_energy)
                    else:
                        tau_result[line_list] = np.append(tau_result[line_list],tau_f(center_energy))
            else:
                tau_all = self.generate_sum_tau_function(tau_func_dict, self.rmin, self.rmax, self.rs_dE, r_sampling_points=100, kind='linear')
                #E_array = np.linspace(6500, 8000, 10000)
                E_array = np.linspace(6500, 8000, 10000)
                plt.plot(E_array, tau_all(E_array), label=f'turb = {self.turbulent_velocity} km/s', color=plt.cm.viridis(i/len(turb_list)))
                #plt.show()
                for line_list, tau_func_r_e in tau_func_dict.items():
                    center_energy = self.line_manager.line_energies[line_list]
                    if i == 0:
                        tau_result[line_list] = tau_all(center_energy)
                    else:
                        tau_result[line_list] = np.append(tau_result[line_list],tau_all(center_energy))
        # for i, line_list in enumerate(tau_result.keys()):
        #     plt.plot(turb_list, tau_result[line_list], label=f'{self.cluster_name}', color=color) 
        print(tau_result)
        plt.xlabel('Turbulent Velocity (km/s)')
        plt.ylabel('Optical Depth')
        plt.title('Optical Depth of Resonance Scattering')
        plt.legend()
        plt.grid(linestyle='--')
        plt.show()

    def ni_integration(self):
        r = np.linspace(0.1, 1000, 1000)
        ion_density = (self.CM.Cluster.data.f_ne(r) *u.cm**(-3) * 1/1.17 * 3.27e-5 * self.CM.Cluster.data.f_Z(r) * self.physics.atom.iz(self.CM.Cluster.data.f_T(r)*u.keV)).to('cm**-3')
        r_cm = (r * u.kpc).to('cm').value
        # 視線方向に沿った積分 → 単位 [cm⁻²]
        Ne_column = np.trapz(ion_density.value, r_cm)  # 単位: cm⁻³ × cm = cm⁻²
        print(f"視線方向の電子カラム密度 Ne_column = {Ne_column:.3e} [cm⁻²]")
        return Ne_column

    def tau_v_ni_dependence_2d(self, line_target="w", r_points=100, show=True):
        """
        For AO2 proposal Figure 1 right. 
        """
        import matplotlib.ticker as ticker
        # 軸定義
        turb_list = np.linspace(0, 1000, 100)
        factor_list = np.linspace(0.1, 1.5, 100)

        tau_map = np.zeros((len(factor_list), len(turb_list)))

        self.initialize_for_tau_E()
        # 中心エネルギー取得
        ni_col = self.ni_integration()
        ni_col_scaled = ni_col * factor_list  # 実際のカラム密度になる
        center_energy = self.line_manager.line_energies[line_target]

        # 先に全 turbulent velocity に対する tau(center_energy) を取得
        tau_column = np.zeros(len(turb_list))
        for j, vturb in enumerate(turb_list):
            self.turbulent_velocity = vturb
            self.initialize_for_tau_E()
            tau_func_dict = self.probability_function_list_rsmall
            tau_all = self.generate_sum_tau_function(
                tau_func_dict, self.rmin, self.rmax, self.rs_dE,
                r_sampling_points=r_points, kind='linear'
            )
            tau_column[j] = tau_all(center_energy)

        # 各 factor に対して tau をスケーリングして 2D マップ生成
        for i, factor in enumerate(factor_list):
            tau_map[i, :] = factor * tau_column

        # -------------------------------
        # 論文向けプロット整形
        # -------------------------------
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.family": "serif",
            "text.usetex": False
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [turb_list[0], turb_list[-1], ni_col_scaled[0], ni_col_scaled[-1]]

        norm = LogNorm(vmin=3e-2, vmax=np.max(tau_map))

        # tau_map 内の 0 以下の値をカットオフ（必要）
        tau_map_clipped = np.clip(tau_map, norm.vmin, norm.vmax)

        # imshow にノルムを渡す
        im = ax.imshow(
            tau_map_clipped, aspect='auto', origin='lower', extent=extent,
            interpolation='nearest', cmap='plasma', norm=norm
        )

        # 軸とタイトル
        ax.set_xlabel("Turbulent Velocity (km/s)")
        ax.set_title(f"Optical Depth of Fe XXV Heα w", pad=10)

        # カラーバー（normに合わせる）
        # カラーバーにノルムを **明示的に** 渡す
        cbar = fig.colorbar(im, ax=ax, pad=0.02, norm=norm)
        #cbar.set_label(f"τ", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # 等高線（linear scale）
        contour_levels = [0.1, 0.2, 0.5, 1.0]
        CS = ax.contour(
            turb_list, ni_col_scaled, tau_map,
            levels=contour_levels, colors='white', linewidths=1.5
        )
        ax.clabel(CS, inline=True, fontsize=10, fmt=lambda x: f"τ={x:.1f}")

        # ↓↓↓ y 軸ラベル変更
        ax.set_ylabel(r"Fe XXV Column Density [cm$^{-2}$]")
        ax.axhline(1.757e+17, color='black', linestyle='--', alpha=0.5)
        ax.axhline(1.227e+17, color='black', linestyle='--', alpha=0.5)
        ax.text(
            turb_list[0] + 10, 1.757e+17, "PKS 0745-191",  # ←適切なラベルに置き換え
            va='bottom', ha='left', fontsize=12, color='black'
        )
        ax.text(
            turb_list[0] + 10, 1.227e+17, "Perseus",  # ←適切なラベルに置き換え
            va='bottom', ha='left', fontsize=12, color='black'
        )
        # 軸調整
        ax.tick_params(direction='in', top=True, right=True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

        fig.tight_layout()
        fig.savefig(f"tau_v_ni_dependence_{line_target}.png", dpi=300, transparent=True)
        if show:
            plt.show()

    def generate_sum_tau_function(self, tau_func_dict, rmin, rmax, rs_dE, r_sampling_points=100, kind='linear'):
        """
        複数のエネルギー範囲で計算された tau(E) を合算して、1つのエネルギーに対する tau(E) を返す関数を生成。

        Parameters
        ----------
        tau_func_dict : dict
            各line_listごとに、エネルギー範囲でtau関数を計算したdict。
            key: line_list, value: tau_func_r_e (RegularGridInterpolator関数)
        rmin, rmax : float
            Rの積分範囲
        rs_dE : float
            エネルギー幅
        r_sampling_points : int
            積分で使用する点数（R方向）
        kind : str
            補間方法（'linear', 'cubic' など）

        Returns
        -------
        sum_tau_func : function
            合算された tau(E) を返す補間関数
        """
        tau_dunc_list = []
        E_all = []

        # 各line_listに対するtau(E)を合算
        for line_list, tau_func_r_e in tau_func_dict.items():
            center_energy = self.line_manager.line_energies[line_list]
            # Eの範囲設定
            E_array = np.linspace(center_energy - rs_dE, center_energy + rs_dE, 100)
            # 各エネルギー範囲に対する tau(E) を生成
            tau_func_e = self.generate_tau_E_function(tau_func_r_e, rmin, rmax, E_array, r_sampling_points, kind)
            tau_dunc_list.append(tau_func_e)
            E_all.extend(E_array)

        # E_allに対してtauを合計
        E_all = np.unique(E_all)  # 重複を排除
        tau_sum_list = []

        for E in E_all:
            # 各エネルギーEについて、全ての関数の値を足し合わせる
            tau_values = [tau_func(E) for tau_func in tau_dunc_list]
            tau_sum = np.sum(tau_values)  # 合計
            tau_sum_list.append(tau_sum)

        # 最終的な tau(E) 関数を作成
        sum_tau_func = interp1d(E_all, tau_sum_list, kind=kind, bounds_error=False, fill_value="extrapolate")

        return sum_tau_func

    def generate_tau_E_function(self, tau_func, rmin, rmax, E_array, r_sampling_points=100, kind='linear'):
        """
        tau(R,E)をRについて積分してtau(E)を得る補間関数を返す。

        Parameters
        ----------
        tau_func : function
            RegularGridInterpolatorなどで定義された(R,E)→tauの関数
        rmin : float
            積分範囲の下限
        rmax : float
            積分範囲の上限
        E_array : ndarray
            エネルギー配列
        r_sampling_points : int
            積分時の評価点の最大数（quadに渡すlimit）
        kind : str
            補間方法（'linear', 'cubic'など）

        Returns
        -------
        tau_E_interp_func : function
            tau(E)を返す補間関数（interp1dオブジェクト）
        """
        tau_E_list = []


        R_array = np.linspace(rmin, rmax, 1000)
        dR = R_array[1] - R_array[0]
        tau_E_list = []
        # for E in E_array:
        #     integrand = lambda R: tau_func([[R, E]])[0]
        #     integral_result, _ = quad(integrand, rmin, rmax)
        #     tau_E_list.append(integral_result)

        for E in E_array:
            points = np.column_stack((R_array, np.full_like(R_array, E)))
            tau_values = tau_func(points)
            tau_sum = np.trapz(tau_values, R_array)
            tau_E_list.append(tau_sum)

        tau_E_interp_func = interp1d(E_array, tau_E_list, kind=kind, bounds_error=False, fill_value="extrapolate")
        return tau_E_interp_func

    def integrate_rs_tau(self, f_ne, f_T, f_Z, v):
        self.initialize_for_tau()
        energy_values = np.linspace(6500,8000, 3000)
        R_min, R_max = 0, 1000
        physics = Physics()
        for line_list in ['w', 'x', 'y', 'z', 'Lya1', 'Lya2', 'Heb1', 'Heb2', 'u', 'r', 't', 'q']:
            compute_prob = physics.resonance_scattering_sigma_field_return_sigma(line_state=line_list, 
                f_ne=f_ne,
                f_T=f_T,
                f_Z=f_Z,
                velocity=v)

            energy_values = physics.atom.line_energy
            E = energy_values
            result = romberg(lambda R: compute_prob(R, E).value, R_min, R_max)
            
        return result

    ## multi-threading, maybe the self.photon manager is not thread safe
    ## please check the thread safety of the photon manager
    # @profile_func
    # def scatter_generated_photons_div(self, plot=True, savefile='simulation.hdf5'):
    #     photon_number = self.size
    #     self.initialize()
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [
    #             executor.submit(self.process_single_photon, num)
    #             for num in range(1, photon_number + 1)
    #         ]
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result() 
    #     if plot:
    #         self.plot_results(photon_manager=self.photon_manager)
    #     self.save_simulation_results(filename=savefile, photon_dataset=self.photon_manager.photon_dataset)
    #     return self.photon_manager.photon_dataset

    @profile_func
    def scatter_generated_photons_div(self, plot=True, savefile='simulation.hdf5'):
        '''
        Execute the simulation of the cluster.
        The photons are generated and scattered by using the initialize function.
        The multi threading is not used in this function. (maybe the self.photon manager is not thread safe, please check the thread safety of the photon manager)
        '''
        photon_number = self.size
        self.initialize()

        for num in range(1, photon_number + 1):
            self.process_single_photon(num)

        if plot:
            self.plot_results(photon_manager=self.photon_manager)

        self.save_simulation_results(
            filename=savefile,
            photon_dataset=self.photon_manager.photon_dataset
        )

        return self.photon_manager.photon_dataset

    def process_single_photon(self, num):
        """
        Process a single photon.
        The photon is generated and scattered by using the initialize function.
        The multi threading is not used in this function. (maybe the self.photon manager is not thread safe, please check the thread safety of the photon manager)
        """
        print(f'photon num = {num}', end='\r', flush=True)
        photon = self.generate_photon(num, self.rs_line_list, self.photon_energy_list, self.radisu_distribution_list)
        
        if len(self.physics_list) != 0:
            if "RS" in self.physics_list:
                self.handle_resonance_scattering(
                    photon=photon,
                    line_manager=self.line_manager,
                    probability_function_list_rsmall=self.probability_function_list_rsmall,
                    probability_function_list_rlarge=self.probability_function_list_rlarge,
                    thomson_scattering_probability_rsmall=self.thomson_scattering_probability_rsmall,
                    thomson_scattering_probability_rlarge=self.thomson_scattering_probability_rlarge,
                    ion=self.ion
                )
            else:
                if "TS" in self.physics_list:
                    print('Processing Thomson Scattering', flush=True)
                    self.handle_thomson_scattering(photon=photon, thomson_scattering_probability_rsmall=self.thomson_scattering_probability_rsmall, thomson_scattering_probability_rlarge=self.thomson_scattering_probability_rlarge)
                else:
                    print('Your selected physics is not setted in fuction', flush=True)

        else:
            #self.handle_no_physics(photon=photon)
            print('No physics is selected', flush=True)
            pass
        self.photon_manager.store_photon_data(photon=photon, num=num)

    def initialize(self):
        """
        Initialize the simulation parameters.
        Some instance variables are initialized here.
        Probability functions for resonance scattering and Thomson scattering are also initialized.
        Probability functions are defined radius is small(rsmall) and large(rlarge).
        This setting is not enought to calculate the large optical depth.
        """
        # store the photon data ex) photon_energy, photon_direction, photon_position
        print('initialize', flush=True)
        print('Set the photon manager', flush=True)
        self.photon_manager = PhotonManager() 
        # set the physics, this class calulate the mean-free path of the resonance scattering and thomson scattering
        print('Set the physics', flush=True)
        self.physics        = Physics(line_state='w',atomdb_version=self.atomdb_version, abundance_table=self.abundance_table,load_furukawa=False)
        # set the line manager, this class manage the resonance scattering lines
        print('Set the line manager', flush=True)
        self.line_manager   = RSLineManager(rs_line_list=self.rs_line_list, dE=self.rs_dE, atomdb_version=self.atomdb_version, abundance_table=self.abundance_table,load_furukawa=False)
        # set the ion field, this class manage the ion velocity
        print('Set the ion field', flush=True)
        self.ion            = IonField()
        print('Set the probability functions', flush=True)
        # definition of the probability or mean-free path function of the resonance scattering and thomson scattering.
        self.probability_function_list_rsmall      = self.line_manager.define_RS_probability_list(rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne, f_T=self.CM.Cluster.data.f_T, f_Z=self.CM.Cluster.data.f_Z, velocity=self.turbulent_velocity,mode=self.tau_mode)
        self.probability_function_list_rlarge      = self.line_manager.define_RS_probability_list(rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne, f_T=self.CM.Cluster.data.f_T, f_Z=self.CM.Cluster.data.f_Z, velocity=self.turbulent_velocity,mode=self.tau_mode)
        if self.tau_mode == "mfp":
            self.thomson_scattering_probability_rsmall = self.physics.thomson_scattering_mfp_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne)
            self.thomson_scattering_probability_rlarge = self.physics.thomson_scattering_mfp_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne)
        else:
            self.thomson_scattering_probability_rsmall = self.physics.thomson_scattering_probability_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne)
            self.thomson_scattering_probability_rlarge = self.physics.thomson_scattering_probability_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne)
        # make the photon energy list, this is the energy of the photon
        self.make_photon_energy_list()

    def initialize_for_tau_E(self):
        """
        Initialize the simulation parameters.
        Some instance variables are initialized here.
        Probability functions for resonance scattering and Thomson scattering are also initialized.
        Probability functions are defined radius is small(rsmall) and large(rlarge).
        This setting is not enought to calculate the large optical depth.
        """
        # store the photon data ex) photon_energy, photon_direction, photon_position
        print('initialize', flush=True)
        print('Set the photon manager', flush=True)
        self.photon_manager = PhotonManager() 
        # set the physics, this class calulate the mean-free path of the resonance scattering and thomson scattering
        print('Set the physics', flush=True)
        self.physics        = Physics(line_state='w',atomdb_version=self.atomdb_version, abundance_table=self.abundance_table,load_furukawa=False)
        # set the line manager, this class manage the resonance scattering lines
        print('Set the line manager', flush=True)
        self.line_manager   = RSLineManager(rs_line_list=self.rs_line_list, dE=self.rs_dE, atomdb_version=self.atomdb_version, abundance_table=self.abundance_table,load_furukawa=False)
        # set the ion field, this class manage the ion velocity
        print('Set the ion field', flush=True)
        self.ion            = IonField()
        print('Set the probability functions', flush=True)
        # definition of the probability or mean-free path function of the resonance scattering and thomson scattering.
        self.probability_function_list_rsmall      = self.line_manager.define_RS_probability_list(rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne, f_T=self.CM.Cluster.data.f_T, f_Z=self.CM.Cluster.data.f_Z, velocity=self.turbulent_velocity,mode="tau_dL")
        self.probability_function_list_rlarge      = self.line_manager.define_RS_probability_list(rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne, f_T=self.CM.Cluster.data.f_T, f_Z=self.CM.Cluster.data.f_Z, velocity=self.turbulent_velocity,mode="tau_dL")
        if self.tau_mode == "mfp":
            self.thomson_scattering_probability_rsmall = self.physics.thomson_scattering_mfp_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne)
            self.thomson_scattering_probability_rlarge = self.physics.thomson_scattering_mfp_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne)
        else:
            self.thomson_scattering_probability_rsmall = self.physics.thomson_scattering_probability_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist, f_ne=self.CM.Cluster.data.f_ne)
            self.thomson_scattering_probability_rlarge = self.physics.thomson_scattering_probability_field(Emin=self.Emin, Emax=self.Emax, rmax=self.rmax, dn=self.dn, dL=self.step_dist*10, f_ne=self.CM.Cluster.data.f_ne)
        # make the photon energy list, this is the energy of the photon

    def generate_photon(self, num, rs_line_list, photon_energy_list, radius_distribution_list):
        """
        Generate a photon with a given number and parameters.
        Parameters
        ----------
        num : int
            Photon number.
        rs_line_list : list
            List of resonance scattering lines.
        photon_energy_list : list
            List of photon energies (eV).
        radius_distribution_list : list
            List of radius distributions.
        Returns
        -------
        Photon
            A Photon object with the generated parameters.
        """
        photon = Photon(rs_line_list=rs_line_list, photon_id=num)
        photon.generate_random_photon(r=radius_distribution_list[num-1], center_energy=photon_energy_list[num-1])
        return photon

    def compare_sampled_and_theoretical_mfp(self,  E=6700.4, deg=0, r0_norm=1e-5, num_samples=10000):
        from scipy.integrate import simpson

        self.Emin = 6690
        self.Emax = 6710
        self.initialize_for_tau_E()
        mfp_func = self.physics.resonance_scattering_mfp_field(
            f_ne=self.CM.Cluster.data.f_ne,
            f_T=self.CM.Cluster.data.f_T,
            f_Z=self.CM.Cluster.data.f_Z,
            velocity=self.turbulent_velocity,
            line_state="w",
            Emin=self.Emin,
            Emax=self.Emax,
            rmax=self.rmax,
            dn=self.dn,
            dL=self.step_dist,
        )
        samples = []
        for _ in range(0, num_samples):
            photon = Photon()
            photon.generate_random_photon(r=r0_norm, center_energy=E)
            mfp_val, tau_s, alpha_s = self.sampling_mfp(mfp_func, photon, test_mode=True)
            samples.append(mfp_val)
        s_array = self.mfp_radius
        P_s = alpha_s * np.exp(-tau_s)
        P_s = P_s / simpson(P_s, s_array)  # 正規化
        # プロット
        plt.hist(samples, bins=s_array, density=True, histtype='step', label='Sampled MFP', color='black')
        plt.plot(s_array, P_s, label='Theoretical $P(s)$', color='red')
        plt.xlabel('Free path $s$ (kpc)')
        plt.ylabel('Probability Density')
        #plt.title('Sampled vs Theoretical Mean Free Path Distribution')
        plt.legend()
        plt.grid(True)
        plt.semilogx()
        plt.tight_layout()
        plt.savefig("mfp_sampling.png",dpi=300)
        plt.show()

    # def sampling_mfp(self, mfp_func, photon, test_mode=False):
    #     '''
    #     sampling the mean free path.
    #     The mean free path (MFP) is calculated by the function mfp_func.
    #     MFP is calculating by the integration of the absorption coefficient along the photon path.
    #     The photon path (s_array) is defined in __init__ function of the Simulation class.
    #     Default is 0 to 1000 kpc.
    #     If the sampling value is larger than the maximum value of the optical depth,
    #     the mean free path is set to inf.

    #     Parameters
    #     ----------
    #     mfp_func : function
    #         The function to calculate the mean free path.
    #         mfp_func should be defined in the class Physics.
    #         ex) mfp_func = self.physics.resonance_scattering_mfp_field
    #         radial and energy dependent function like mfp_func(R, E) 
    #     photon : Photon
    #         The photon object.
    #     test_mode : bool
    #         If True, the test mode is activated. Default is False.
    #     Returns
    #     -------
    #     mfp_val : float
    #         The mean free path.
    #     tau_s : array
    #         The optical depth.
    #     alpha_s : array
    #         The absorption coefficient.
    #     '''
    #     # calulate the photon position and direction from the photon object
    #     r_vec = photon.polartocart(photon.current_position)
    #     r_hat = r_vec / photon.current_position[0]
    #     n_hat = photon.polartocart(photon.direction)
    #     mu = np.dot(r_hat, n_hat)

    #     s_array = self.mfp_radius
    #     E_array = np.full_like(s_array, photon.energy)

    #     r_s = np.sqrt(photon.current_position[0]**2 + s_array**2 + 2 * photon.current_position[0] * s_array * mu)
    #     lambda_s = mfp_func((r_s, E_array))
    #     alpha_s = 1.0 / lambda_s
    #     tau_s = cumtrapz(alpha_s, s_array, initial=0)

    #     delta_tau = -np.log(np.random.uniform())
    #     if delta_tau <= tau_s[-1]: # tau_s[-1] is the maximum value of tau_s
    #         mfp_val = np.interp(delta_tau, tau_s, s_array)
    #     else:
    #         mfp_val = np.inf  # If the sampling value is larger than the maximum value of the optical depth, set to inf

    #     if test_mode == False:
    #         return mfp_val
    #     else:
    #         return mfp_val, tau_s, alpha_s

    def qdp_sampling_counter(self, radial_distribution):
        from collections import defaultdict

        # 初期位置の配列（kpc単位）
        initial_positions_kpc = np.array(radial_distribution)

        # arcmin換算用定数
        arcmin2kpc = self.arcmin2kpc_furukawa
        initial_positions_arcmin = initial_positions_kpc / arcmin2kpc

        # 対応表：radius range（arcmin）→ スペクトルファイル名
        spectrum_mapping = [
            (0.0, 0.5,  "bapec_model_v308_all_vt150_01.qdp"),
            (0.5, 1.0,  "bapec_model_v308_all_vt150_02.qdp"),
            (1.0, 1.5,  "bapec_model_v308_all_vt150_03.qdp"),
            (1.5, 2.0,  "bapec_model_v308_all_vt150_04.qdp"),
            (2.0, 2.5,  "bapec_model_v308_all_vt150_05.qdp"),
            (2.5, 3.0,  "bapec_model_v308_all_vt150_06.qdp"),
            (3.0, 3.5,  "bapec_model_v308_all_vt150_07.qdp"),
            (3.5, 4.0,  "bapec_model_v308_all_vt150_08.qdp"),
            (4.0, 4.5,  "bapec_model_v308_all_vt150_09.qdp"),
            (4.5, 5.0,  "bapec_model_v308_all_vt150_10.qdp"),
            (5.0, 10.0, "bapec_model_v308_all_vt150_11.qdp"),
            (10.0, 20.0,"bapec_model_v308_all_vt150_21.qdp"),
            (20.0, 30.0,"bapec_model_v308_all_vt150_31.qdp"),
            (30.0, 40.0,"bapec_model_v308_all_vt150_41.qdp"),
        ]

        # スペクトルファイルごとの初期位置数をカウント
        spectrum_counts = defaultdict(int)

        for r in initial_positions_arcmin:
            for rmin, rmax, filename in spectrum_mapping:
                if rmin <= r < rmax:
                    spectrum_counts[filename] += 1
                    break

        # ファイル名とカウント数をリストに変換して返す
        spectrum_counts_sorted = sorted(spectrum_counts.items(), key=lambda x: x[0])
        spectrum_files = [item[0] for item in spectrum_counts_sorted]
        counts = [item[1] for item in spectrum_counts_sorted]

        return spectrum_files, counts
    def sampling_mfp(self, mfp_func, photon, test_mode=False):
        """
        Sample mean-free-path without altering the photon direction.

        Parameters
        ----------
        mfp_func : callable
            λ(r,E) を返す補間関数（RegularGridInterpolator など）
        photon : Photon
            現在位置・方向・エネルギーを持つ Photon インスタンス
        test_mode : bool, default False
            True のとき (mfp_val, τ(s), α(s)) を返す（デバッグ用）

        Returns
        -------
        mfp_val : float
            サンプリングされた平均自由行程 [kpc]。散乱しなければ np.inf
        """
        # -------- 単位ベクトルなど -------------------------------------------
        r_vec = photon.polartocart(photon.current_position)   # xyz
        r0    = photon.current_position[0]                    # 現在半径 [kpc]
        r_hat = r_vec / r0                                    # 外向き
        n_hat = photon.polartocart(photon.direction)          # 進行方向
        mu    = float(np.dot(r_hat, n_hat))                   # cos θ

        # -------- 球面 r = rmax までの光路長 s_max ----------------------------
        #  r(s)^2 = r0^2 + s^2 + 2 r0 s μ = rmax^2
        disc = self.rmax**2 - r0**2 * (1.0 - mu**2)           # 判別式
        if disc < 0:
            raise RuntimeError("No intersection with r=rmax (geometry error)")
        s_max = -r0 * mu + np.sqrt(disc)                      # 正の解のみ採用

        # -------- s グリッドを 0 → s_max で生成 -------------------------------
        N_s   = len(self.mfp_radius)                          # 元の分割数を流用
        s_array = np.linspace(0.0, s_max, N_s)
        E_array = np.full_like(s_array, photon.energy)

        # r(s) = √(r0² + s² + 2 r0 s μ)
        r_s = np.sqrt(r0**2 + s_array**2 + 2.0*r0*s_array*mu)

        # -------- λ(s), α(s), τ(s) ------------------------------------------
        lambda_s = mfp_func((r_s, E_array))      # [kpc]
        alpha_s  = 1.0 / lambda_s                # [kpc⁻¹]
        tau_s    = cumtrapz(alpha_s, s_array, initial=0.0)

        # -------- Δτ を逆変換サンプリング ------------------------------------
        delta_tau = -np.log(np.random.uniform())
        if delta_tau <= tau_s[-1]:
            mfp_val = np.interp(delta_tau, tau_s, s_array)
        else:
            mfp_val = np.inf                        # 散乱せずに球外へ

        if test_mode:
            return mfp_val, tau_s, alpha_s
        return mfp_val

    @staticmethod
    def process_radius(args):
        r, rebined_radial_distribution, turbulent_velocity, radial_list, sampling_Emin, sampling_Emax, seed_spectrum_Emin, seed_spectrum_Emax, seed_spectrum_dE, spectrum_file_root, abundance_table, atomdb_version = args
        print(f'radius = {r} kpc', flush=True)
        r_length = np.count_nonzero(rebined_radial_distribution == r)
        if r_length == 0:
            return []  # 光子がない場合は空リストを返す

        print(f'radius = {r} kpc, length = {r_length}')
        energy_list = RadiationField(Emin=seed_spectrum_Emin, Emax=seed_spectrum_Emax, dE=seed_spectrum_dE, abundance_name=abundance_table, atomdb_version=atomdb_version).generate_apec_spectrum_from_hdf5(
            size=r_length,
            sampling_Emin = sampling_Emin,
            sampling_Emax = sampling_Emax,
            turbulent_velocity=turbulent_velocity,
            radius=r,
            radial_list=radial_list,
            seed_filename=spectrum_file_root
        )
        return energy_list

    def make_photon_energy_list(self):
        '''
        Make the photon energy and initial position list.
        If cluster_name is 'dummy', the photon energy list is generated by the function gaussian_energy.
        
        First, the initial position list of the photon is generated by the InitialPhotonGenerator.generate_and_save().
        The photon energy list is generated by the function generate_apec_photon.
        '''
        print('Making initial photon position list')
        print('cluster name:', self.cluster_name)
        if self.test_mode == False and self.comparison == False:
            # make the initial photon position list
            radial_distribution_list = InitialPhotonGenerator(ClusterManager=self.CM,savefilename=None).generate_and_save(size=self.size, save=False)

            # the rebin of the radial distribution list
            if self.method == 'linear':
                self.rebined_radial_distribution = sorted(self.CM.assign_nearest_value(radial_distribution_list, self.CM.divide_cluster_sphere_linear()))
                self.radial_list = np.unique(self.rebined_radial_distribution)
            elif self.method == 'log':
                self.rebined_radial_distribution = sorted(self.CM.rebin_data_logscale(radial_distribution_list))
                self.radial_list = np.unique(self.rebined_radial_distribution)

            print('Making photon energy list')
            print('radial list:', self.radial_list)
            process_args = [
                (r, self.rebined_radial_distribution, self.turbulent_velocity, self.radial_list, self.Emin*1e-3, self.Emax*1e-3, 
                 self.seed_spectrum_Emin, self.seed_spectrum_Emax, self.seed_spectrum_dE, self.spectrum_sampling_file_root, self.abundance_table, self.atomdb_version)
                for r in self.radial_list
            ]
            # !: 参照するspectrum fileがない時に自動で生成するように変更する
            spectrum_file = f'{self.spectrum_sampling_file_root}_{self.turbulent_velocity}.hdf5'
            if os.path.exists(spectrum_file) == False:
                print(f'{spectrum_file} is not found.')
                RadiationField(Emin=self.seed_spectrum_Emin, Emax=self.seed_spectrum_Emax, dE=self.seed_spectrum_dE, abundance_name=self.abundance_table, atomdb_version=self.atomdb_version
                               ).save_apec_spectrum_hdf5(Cluster=self.CM,turbulent_velocity=self.turbulent_velocity,save_root_name=self.spectrum_sampling_file_root)
            with ProcessPoolExecutor() as executor:
                results = executor.map(Simulation.process_radius, process_args)
            self.photon_energy_list = []
            for result in results:
                self.photon_energy_list.extend(result)
            self.radisu_distribution_list = self.rebined_radial_distribution
            self.photon_energy_list = np.array(self.photon_energy_list)*1e3

        elif self.comparison == True:
            radial_distribution_list = sorted(InitialPhotonGenerator(ClusterManager=self.CM,savefilename=None).generate_and_save(size=self.size, save=False))
            spectrum_files, counts = self.qdp_sampling_counter(radial_distribution_list)
            mother_spectrum_dir = '/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/perseus/atomdb_v3.0.8_angr/furukawa_mother_spectrum'
            self.photon_energy_list = []
            for i in range(len(spectrum_files)):
                print(spectrum_files[i], counts[i])
                energy_e = RadiationField(Emin=self.seed_spectrum_Emin, Emax=self.seed_spectrum_Emax, dE=self.seed_spectrum_dE, abundance_name=self.abundance_table, atomdb_version=self.atomdb_version).sampling_from_qdp(f'{mother_spectrum_dir}/{spectrum_files[i]}', counts[i])
                self.photon_energy_list.extend(energy_e)
            self.radisu_distribution_list = radial_distribution_list
            self.rebined_radial_distribution = radial_distribution_list
            self.radial_list = np.unique(self.rebined_radial_distribution)
            self.photon_energy_list = np.array(self.photon_energy_list)*1e3
        else:
            radial_distribution_list = np.full(self.size, 1e-5)
            self.rebined_radial_distribution = radial_distribution_list
            self.radisu_distribution_list    = radial_distribution_list
            self.radial_list = np.unique(self.rebined_radial_distribution)
            # ? : temporaly setting for changing the photon energy
            #self.photon_energy_list = np.array(Photon().gaussian_energy(self.line_manager.line_energies['w'], 0.21735, self.size))
            self.photon_energy_list = np.array(Photon().gaussian_energy(self.line_manager.line_energies['w'], self.physics.DeltaE(T=self.CM.Cluster.data.f_T(0), E0=self.line_manager.line_energies['w'], v=self.turbulent_velocity).to('eV').value/np.sqrt(2), self.size))
            #self.photon_energy_list = np.random.uniform(self.line_manager.line_energies['w']-10,self.line_manager.line_energies['w']+10, self.size)
            #self.photon_energy_list = np.full(self.size, self.line_manager.line_energies['w']-1) 

    def make_load_photon_energy_list(self):
        # HDF5ファイルの読み取り（全体）
        radial_distribution_list_length = self.take_out_hdf5_photons_return_data(
            filename='initial_photon_distribution.hdf5', 
            attributes={'cluster_name': self.cluster_name}, 
            length_mode=True
        )

        if radial_distribution_list_length < self.size:
            print('The number of photons is smaller than the size of the simulation.')
            InitialPhotonGenerator().generate_and_save(
                cluster_name=self.cluster_name, 
                size=self.size, 
                rmax=self.rmax, 
                filename='initial_photon_distribution.hdf5'
            )

        # 分布データを読み込み
        radial_distribution_list = self.take_out_hdf5_photons_return_data(
            filename='initial_photon_distribution.hdf5', 
            attributes={'cluster_name': self.cluster_name},
            length=self.size
        ).flatten()

        # 再ビン分け
        if self.method == 'linear':
            self.rebined_radial_distribution = sorted(self.CM.rebin_data_linear(radial_distribution_list))
            self.radial_list = self.CM.divide_cluster_sphere_linear()
        elif self.method == 'log':
            self.rebined_radial_distribution = sorted(self.CM.rebin_data_logscale(radial_distribution_list))
            self.radial_list = self.CM.divide_cluster_sphere_logscale()

        if len(self.rebined_radial_distribution) != self.size:
            print('The number of radial distribution and the number of radial list do not match.')
            print('The simulation will be terminated.')
            print(len(self.rebined_radial_distribution))
            sys.exit()

        # 順次実行
        photon_energy_list = []
        for r in self.radial_list:
            print(f'radius = {r} kpc', flush=True)
            r_length = np.count_nonzero(self.rebined_radial_distribution == r)
            if r_length == 0:
                continue

            # 属性を設定
            self.attributes['radius'] = r
            energy_list_length = self.take_out_hdf5_photons_return_data(
                filename='apec_photon_energy_distribution.hdf5', 
                attributes=self.attributes, 
                length_mode=True
            )

            # 光子が足りない場合は生成
            if r_length > energy_list_length:
                print(f'radius = {r} kpc, length = {r_length}')
                energy_list = RadiationField().generate_apec_photon(
                    kT=self.CM.Cluster.data.f_T(r), 
                    Emin=self.Emin * 1e-3, 
                    Emax=self.Emax * 1e-3, 
                    dE=0.5e-3, 
                    abundance=self.CM.Cluster.data.f_Z(r), 
                    velocity=self.turbulent_velocity, 
                    size=r_length, 
                    filename='apec_photon_energy_distribution.hdf5', 
                    cluster_name=self.cluster_name, 
                    method='linear', 
                    radius=r
                )
            # 必要なエネルギーリストを取得
            else:
                energy_list = self.take_out_hdf5_photons_return_data(
                    filename='apec_photon_energy_distribution.hdf5', 
                    attributes=self.attributes,
                    length=r_length
                )
            
            photon_energy_list.extend(energy_list)

        # 結果の検証
        self.photon_energy_list = photon_energy_list
        if len(self.photon_energy_list) != len(self.rebined_radial_distribution):
            print(len(self.photon_energy_list))
            print(len(self.rebined_radial_distribution))
            print('The number of photons and the number of radial distribution do not match.')
            print('The simulation will be terminated.')
            sys.exit()

        self.radisu_distribution_list = self.rebined_radial_distribution

    def handle_thomson_scattering(self, photon: Photon, thomson_scattering_probability_rsmall, thomson_scattering_probability_rlarge):
        while photon.current_position[0] < self.rmax:
            if photon.current_position[0] < 100:
                step = np.array([self.step_dist, 1, 1])  # Precompute step vector
                scatter_prob = thomson_scattering_probability_rsmall((photon.current_position[0], photon.energy))
            else:
                step = np.array([self.step_dist*10, 1, 1])
                scatter_prob = thomson_scattering_probability_rlarge((photon.current_position[0], photon.energy))
            if np.random.rand() < scatter_prob:
                # Handle scattering
                photon.direction = photon.random_on_sphere(1)
                photon.flags['scatter'] = True
            photon.previous_position = photon.current_position.copy()
            photon.current_position = photon.add_polar(photon.current_position, photon.direction * step)

    def handle_resonance_scattering(self, photon: Photon, line_manager: RSLineManager,
                                    probability_function_list_rsmall, probability_function_list_rlarge, thomson_scattering_probability_rsmall,  thomson_scattering_probability_rlarge, ion: IonField):
        """
        Handle resonance scattering and Thomson scattering for the photon.
        Parameters
        ----------
        photon : Photon()
            Photon instance. 
        line_manager : RSLineManager
            The line manager for resonance scattering.
        probability_function_list_rsmall : list
            The list of probability functions for resonance scattering for small radius.
        probability_function_list_rlarge : list
            The list of probability functions for resonance scattering for large radius.
        thomson_scattering_probability_rsmall : function
            The function for Thomson scattering probability for small radius.
        thomson_scattering_probability_rlarge : function
            The function for Thomson scattering probability for large radius.
        ion : IonField()
            The ion field for resonance scattering.
        """
        if self.tau_mode == 'natural':
            self._handle_resonance_scattering_natural(photon=photon, line_manager=line_manager,
                                        probability_function_list_rsmall=probability_function_list_rsmall,
                                        probability_function_list_rlarge=probability_function_list_rlarge,
                                        thomson_scattering_probability_rsmall=thomson_scattering_probability_rsmall,
                                        thomson_scattering_probability_rlarge=thomson_scattering_probability_rlarge,
                                        ion=ion)
        elif self.tau_mode == 'effective':
            self._handle_resonance_scattering_effective(photon=photon, line_manager=line_manager,
                                        probability_function_list_rsmall=probability_function_list_rsmall,
                                        probability_function_list_rlarge=probability_function_list_rlarge,
                                        thomson_scattering_probability_rsmall=thomson_scattering_probability_rsmall,
                                        thomson_scattering_probability_rlarge=thomson_scattering_probability_rlarge,
                                        ion=ion)
        elif self.tau_mode == 'mfp':
            self._handle_resonance_scattering_mfp(photon=photon, line_manager=line_manager,
                                        probability_function_list_rsmall=probability_function_list_rsmall,
                                        probability_function_list_rlarge=probability_function_list_rlarge,
                                        thomson_scattering_probability_rsmall=thomson_scattering_probability_rsmall,
                                        thomson_scattering_probability_rlarge=thomson_scattering_probability_rlarge,
                                        ion=ion)

    def _handle_resonance_scattering_natural(self, photon: Photon, line_manager: RSLineManager,
                                    probability_function_list_rsmall, probability_function_list_rlarge, thomson_scattering_probability_rsmall,  thomson_scattering_probability_rlarge, ion: IonField):
        """
        Handle resonance scattering and Thomson scattering for the photon.
        Parameters
        ----------
        photon : Photon()
            Photon instance. 
        line_manager : RSLineManager
            The line manager for resonance scattering.
        probability_function_list_rsmall : list
            The list of probability functions for resonance scattering for small radius.
        probability_function_list_rlarge : list
            The list of probability functions for resonance scattering for large radius.
        thomson_scattering_probability_rsmall : function
            The function for Thomson scattering probability for small radius.
        thomson_scattering_probability_rlarge : function
            The function for Thomson scattering probability for large radius.
        ion : IonField()
            The ion field for resonance scattering.
        """
        while photon.current_position[0] < self.rmax:
            random_number = np.random.rand() # Generate a random number for scattering judgment
            temp = self.CM.Cluster.data.f_T(photon.current_position[0]) # Get the temperature at the current position
            if self.cluster_name != 'dummy':
                ion.velocity_sampling(kT=temp, vrms=self.turbulent_velocity) # Ion velocity sampling
            else:
                #ion.velocity_sampling_dummy()
                ion.velocity_sampling(kT=temp, vrms=self.turbulent_velocity)
            
            # Determine the step size and probability function based on the current position
            # If the current position is smaller than 100 kpc, use a small step size (setted by step_dist)
            # If the current position is larger than 100 kpc, use a large step size (setted by step_dist*10)
            # !: Maybe this is not enough to calculate the large optical depth.
            # ?: Temporaly setting
            if photon.current_position[0] <= 1000:
                step = np.array([self.step_dist, 1, 1])  # Precompute step vector
                probability_function_list = probability_function_list_rsmall
                thomson_scattering_probability = thomson_scattering_probability_rsmall
            else:
                step = np.array([self.step_dist*10, 1, 1])
                probability_function_list = probability_function_list_rlarge
                thomson_scattering_probability = thomson_scattering_probability_rlarge

            photon.reserved_energy = photon.energy # save the photon energy.
            photon.energy = ion.conversion_ion_restframe(photon.energy, photon.polartocart(photon.direction)) # convert the photon energy to the ion rest frame.
            #!: for test mode
            self.energy_ion_rest.append(photon.energy)
            self.velocity_list.append(ion.velocity)
            self.photon_direction.append(photon.direction)
            self.velocity_prop.append(ion.velocity @ photon.polartocart(photon.direction))
            # Calculate cumulative probabilities for resonance scattering
            # ? : temporarily setting 
            RS_line_list = line_manager.is_in_range(photon.energy) # Get the resonance scattering line list. Wether the photon energy is in the range of the resonance scattering line.
            #RS_line_list = ["w"]
            cumulative_prob = 0 # Initialize cumulative probability. this used for the resonance scattering probability.

            for line in RS_line_list:
                line_prob = probability_function_list[line]((photon.current_position[0], photon.energy))
                cumulative_prob += line_prob
                # !: TODO for test mode
                self.line_prob_check.append(random_number)
                self.random_check.append(cumulative_prob)
                if random_number < cumulative_prob:
                    self._resonance_scatter(photon, ion, line, step)
                    break
            else:
                # Handle Thomson scattering
                if 'TS' in self.physics_list:
                    print('Thomson scattering')
                    thomson_prob = thomson_scattering_probability((photon.current_position[0], photon.reserved_energy))
                    cumulative_prob += thomson_prob
                    if random_number < cumulative_prob:
                        photon.previous_position = photon.current_position
                        photon.direction         = photon.random_on_sphere(1)
                        photon.flags['scatter'] = True
                
                photon.energy = photon.reserved_energy
                photon.previous_position = photon.current_position
                photon.current_position = photon.add_polar(photon.previous_position, photon.direction * step)

    # !: Editing Function For New Mode
    def _handle_resonance_scattering_effective(self, photon: Photon, line_manager: RSLineManager,
                                    probability_function_list_rsmall, probability_function_list_rlarge, thomson_scattering_probability_rsmall,  thomson_scattering_probability_rlarge, ion: IonField):
        """
        Handle resonance scattering and Thomson scattering for the photon.
        Parameters
        ----------
        photon : Photon()
            Photon instance. 
        line_manager : RSLineManager
            The line manager for resonance scattering.
        probability_function_list_rsmall : list
            The list of probability functions for resonance scattering for small radius.
        probability_function_list_rlarge : list
            The list of probability functions for resonance scattering for large radius.
        thomson_scattering_probability_rsmall : function
            The function for Thomson scattering probability for small radius.
        thomson_scattering_probability_rlarge : function
            The function for Thomson scattering probability for large radius.
        ion : IonField()
            The ion field for resonance scattering.
        """
        while photon.current_position[0] < self.rmax:
            random_number = np.random.rand() # Generate a random number for scattering judgment
            temp = self.CM.Cluster.data.f_T(photon.current_position[0]) # Get the temperature at the current position
            # Determine the step size and probability function based on the current position
            # If the current position is smaller than 100 kpc, use a small step size (setted by step_dist)
            # If the current position is larger than 100 kpc, use a large step size (setted by step_dist*10)
            # !: Maybe this is not enough to calculate the large optical depth.
            # ?: Temporaly setting
            if photon.current_position[0] <= 1000:
                step = np.array([self.step_dist, 1, 1])  # Precompute step vector
                probability_function_list = probability_function_list_rsmall
                thomson_scattering_probability = thomson_scattering_probability_rsmall
            else:
                step = np.array([self.step_dist*10, 1, 1])
                probability_function_list = probability_function_list_rlarge
                thomson_scattering_probability = thomson_scattering_probability_rlarge

            # Calculate cumulative probabilities for resonance scattering
            RS_line_list = line_manager.is_in_range(photon.energy) # Get the resonance scattering line list. Wether the photon energy is in the range of the resonance scattering line.
            cumulative_prob = 0 # Initialize cumulative probability. this used for the resonance scattering probability.

            for line in RS_line_list:
                line_prob = probability_function_list[line]((photon.current_position[0], photon.energy))
                cumulative_prob += line_prob
                # !: TODO for test mode
                self.line_prob_check.append(random_number)
                self.random_check.append(cumulative_prob)
                if random_number < cumulative_prob:
                    self._resonance_scatter(photon, ion, line, step, mode='effective')
                    break
            else:
                # Handle Thomson scattering
                if 'TS' in self.physics_list:
                    print('Thomson scattering')
                    thomson_prob = thomson_scattering_probability((photon.current_position[0], photon.reserved_energy))
                    cumulative_prob += thomson_prob
                    if random_number < cumulative_prob:
                        photon.previous_position = photon.current_position
                        photon.direction         = photon.random_on_sphere(1)
                        photon.flags['scatter'] = True
                
                photon.previous_position = photon.current_position
                photon.current_position = photon.add_polar(photon.previous_position, photon.direction * step)

    def _handle_resonance_scattering_mfp(self, photon: Photon, line_manager: RSLineManager,
                                        probability_function_list_rsmall, probability_function_list_rlarge,
                                        thomson_scattering_probability_rsmall, thomson_scattering_probability_rlarge,
                                        ion: IonField):
        """
        Handle resonance scattering and Thomson scattering for the photon.
        Parameters
        ----------
        photon : Photon()
            Photon instance. 
        line_manager : RSLineManager
            The line manager for resonance scattering.
        probability_function_list_rsmall : list
            The list of probability functions for resonance scattering for small radius.
        probability_function_list_rlarge : list
            The list of probability functions for resonance scattering for large radius.
        thomson_scattering_probability_rsmall : function
            The function for Thomson scattering probability for small radius.
        thomson_scattering_probability_rlarge : function
            The function for Thomson scattering probability for large radius.
        ion : IonField()
            The ion field for resonance scattering.
        """
        def update_photon_position(step_length):
            step = np.array([step_length, 1, 1])
            photon.previous_position = photon.current_position.copy()
            photon.current_position = photon.add_polar(photon.previous_position, photon.direction * step)

        def terminate_photon():
            photon.previous_position = photon.current_position.copy()
            photon.current_position = photon.propagate_to_rmax(photon.current_position, photon.direction, self.rmax)

        def handle_thomson_scattering(step_length):
            update_photon_position(step_length)
            photon.direction = photon.random_on_sphere(1)
            photon.flags['scatter'] = True

        def handle_resonance_scattering(step_length, line):
            update_photon_position(step_length)
            T = self.CM.Cluster.data.f_T(photon.current_position[0])
            ion.sampling_ion_velocity_for_cluster_frame(
                kT=T, vrms=self.turbulent_velocity,
                sigma_nat=self.line_manager.natural_width[line],
                A=self.atomic_number, E0=self.line_manager.line_energies[line],
                E=photon.energy
            )
            Rotation = photon.make_rotation_matrix(photon.polartocart(photon.direction))
            ion.velocity = Rotation @ ion.velocity_photon_direction
            photon.direction = photon.random_on_sphere(1)
            photon.flags['scatter'] = True
            # !: for test mode
            self.energy_ion_rest.append(photon.energy)
            self.velocity_list.append(ion.velocity)
            self.photon_direction.append(photon.direction)
            self.velocity_prop.append(ion.velocity_photon_direction[0])
            photon.energy = ion.conversion_ion_observedframe(float(self.line_manager.line_energies[line]),
                                                            photon.polartocart(photon.direction))
            if hasattr(photon, 'RS_position'):
                photon.RS_position = np.vstack((photon.RS_position, photon.previous_position))
            else:
                photon.RS_position = photon.previous_position

        probability_function_list = probability_function_list_rsmall 
        thomson_prob = thomson_scattering_probability_rsmall 

        while photon.current_position[0] < self.rmax:
            # Determine the RS line list based on the photon energy
            RS_line_list = line_manager.is_in_range(photon.energy)
            # Calculate the mfp for each line
            mfp_dict = {line: self.sampling_mfp(probability_function_list[line], photon)
                        for line in RS_line_list}
            # Calculate the mfp for Thomson scattering if applicable
            if "TS" in self.physics_list:
                mfp_dict['TS'] = self.sampling_mfp(thomson_prob, photon)
            # If the mfp dictionary is empty, terminate the photon
            if len(mfp_dict) == 0:
                terminate_photon()
                break
            # Find the minimum mfp and its corresponding key
            min_key = min(mfp_dict, key=mfp_dict.get)
            mfp_min = mfp_dict[min_key]
            # If the minimum mfp is infinite, terminate the photon
            if mfp_min == np.inf or photon.current_position[0] + mfp_min >= self.rmax:
                terminate_photon()
                break
            # If the minimum mfp is smaller than the maximum radius, update the photon position
            if min_key == 'TS':
                handle_thomson_scattering(mfp_min)
            else:
                handle_resonance_scattering(mfp_min, min_key)

    def handle_no_physics(self, photon:Photon):
        photon.previous_position   = photon.current_position.copy()
        photon.direction           = photon.random_on_sphere(1)
        photon.current_position    = photon.add_polar(photon.current_position, np.array([photon.direction[0] * self.step_dist, photon.direction[1], photon.direction[2]]))

    # !: Editing Function For New Mode
    def _resonance_scatter(self, photon : Photon, ion: IonField, line, step, mode='natural'):
        """
        Handle the after resonance scattering process.
        """
        if mode == 'natural':
            photon.previous_position   = photon.current_position
            photon.direction           = photon.random_on_sphere(1)
            if hasattr(photon, 'RS_position') == True:
                photon.RS_position = np.vstack((photon.RS_position, photon.previous_position))
            else:
                photon.RS_position = photon.current_position
            photon.current_position    = photon.add_polar(photon.previous_position, photon.direction * step)
            photon.flags['scatter']    = True
            photon.energy = ion.conversion_ion_observedframe(float(self.line_manager.line_energies[line]), photon.polartocart(photon.direction))
        elif mode == 'effective':
            photon.previous_position   = photon.current_position
            if hasattr(photon, 'RS_position') == True:
                photon.RS_position = np.vstack((photon.RS_position, photon.previous_position))
            else:
                photon.RS_position = photon.previous_position
            ## ! : adding the function for the effective scattering
            # sampling the ion velocity.
            ion.sampling_ion_velocity_for_cluster_frame(kT=self.CM.Cluster.data.f_T(photon.current_position[0]), 
                                                        vrms=self.turbulent_velocity, 
                                                        sigma_nat=self.line_manager.natural_width[line], 
                                                        A=self.atomic_number,
                                                        E0=self.line_manager.line_energies[line],
                                                        E=photon.energy)
            # make the Rotation matrix for the ion velocity for photon direction.
            Rotation = photon.make_rotation_matrix(photon.polartocart(photon.direction))
            # conversion the velocity to the cartesian.
            ion.velocity = Rotation.T @ ion.velocity_photon_direction
            photon.direction           = photon.random_on_sphere(1)
            photon.current_position    = photon.add_polar(photon.previous_position, photon.direction * step)
            photon.flags['scatter']    = True
            if photon.energy == self.line_manager.line_energies['w']-1:
                self.energy_ion_rest.append(photon.energy)
                self.velocity_list.append(ion.velocity)
                self.photon_direction.append(photon.direction)
                self.velocity_prop.append(ion.velocity_photon_direction[0])
            # energy conversion
            photon.energy = ion.conversion_ion_observedframe(float(self.line_manager.line_energies[line]), photon.polartocart(photon.direction))

    def noRS_spectrum(self):
        self.initialize()
        with h5py.File('noRS_spectrum.hdf5', 'a') as f:
            f.create_dataset('noRS_spectrum/energy', data=self.photon_energy_list)
            f.create_dataset('noRS_spectrum/radius', data=self.radisu_distribution_list)

    def plot_results(self, photon_manager):
        fig, ax = plt.subplots()
        scatter_flag = photon_manager.photon_dataset['scatter']
        print(scatter_flag)
        all_photons = photon_manager.photon_dataset['energy']
        unscattered_photons = all_photons[scatter_flag == False]
        scattered_photons   = all_photons[scatter_flag == True]
        # print(f'Number of all photons = {len(all_photons)}')
        # print(f'Number of unscattered photons = {len(unscattered_photons)}')
        # print(f'Number of scattered photons = {len(scattered_photons)}')
        ax.hist(all_photons, bins=300, histtype='step', color='black', label='all photons')
        ax.hist(unscattered_photons, bins=300, histtype='step', color='blue', label='unscattered photons')
        ax.hist(scattered_photons, bins=300, histtype='step', color='red', label='scattered photons')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.grid(linestyle='dashed')
        fig.savefig(f'scatter_test_{self.turbulent_velocity}.pdf', dpi=300, transparent=True)
        return all_photons, unscattered_photons
    
    def return_photon(self, filename=None, keyname=None, idx=-1):
        if keyname == None:
            with h5py.File(filename, 'r') as f:
                d_list = list(f.keys())[idx]
        keyname = d_list
        with h5py.File(filename, 'r') as f:
            all_photons = f[keyname]['initial_energy'][:]
            scatter = f[keyname]['scatter'][:]
            unscattered_photons = all_photons[scatter==False]
            scattered_photons = all_photons[scatter==True]
        print(f'Number of all photons = {len(all_photons)}')
        print(f'Number of unscattered photons = {len(unscattered_photons)}')
        print(f'Number of scattered photons = {len(scattered_photons)}')
        return all_photons, scattered_photons

    def plot_simulation_result(self,filename=None, keyname=None):
        if keyname == None:
            with h5py.File(filename, 'r') as f:
                d_list = list(f.keys())[-1]
        keyname = d_list
        with h5py.File(filename, 'r') as f:
            all_photons = f[keyname]['initial_energy'][:]
            scatter = f[keyname]['scatter'][:]
            unscattered_photons = all_photons[scatter==False]
            scattered_photons = all_photons[scatter==True]
        print(f'Number of all photons = {len(all_photons)}')
        print(f'Number of unscattered photons = {len(unscattered_photons)}')
        print(f'Number of scattered photons = {len(scattered_photons)}')
        fig, ax = plt.subplots()
        ax.hist(all_photons, bins=1000, histtype='step', color='black', label='all photons')
        ax.hist(unscattered_photons, bins=1000, histtype='step', color='blue', label='unscattered photons')
        ax.hist(scattered_photons, bins=1000, histtype='step', color='red', label='scattered photons')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Counts')
        atom = AtomicDataManager(atomic_data_file)
        for state in ['w', 'x', 'y', 'z', 'Lya2', 'Lya1', 'Heb1', 'Heb2', 'u', 'r', 't', 'q']:
            atom.load_line_data(state)
            ax.axvline(atom.line_energy, color='black', linestyle='--', alpha=0.25)
            ax.text(atom.line_energy+1, 20, state)
        ax.legend()
        ax.grid(linestyle='dashed')
        #ax.set_xlim(6690,6710)
        plt.show()
        fig.savefig(f'scatter_test.pdf', dpi=300, transparent=True)

    def scatter_info(self, keyname=None):

        with h5py.File('simulation_pks_digit.hdf5', 'r') as f:
            if keyname == None:
                keyname = list(f.keys())[-1]
            velocity = f[keyname].attrs['velocity']
            scatter = f[keyname]['scatter'][:]
            all_photon_num = len(f[keyname]['scatter'])
            scatter_photon_num = len(f[keyname]['scatter'][scatter==True])
        print(f'Number of all photons = {all_photon_num}')
        print(f'Number of scattered photons = {scatter_photon_num}')
        print(f'Scattering rate = {scatter_photon_num/all_photon_num*100:.2f}%')

    def take_out_hdf5_photons_return_data(self,filename='photon.hdf5',attributes={'cluster_name': 'None'},length=1000,length_mode=False):
        self.hdf5 = HDF5Manager(filename)
        links = self.hdf5.get_matching_datasets(attributes=attributes)
        data = self.hdf5.load_dataset_by_link(dataset_links=links,length=length,length_mode=length_mode)
        return data

    def take_out_hdf5_photons(self,filename='photon.hdf5',attributes={'cluster_name': 'None'},length=1000,length_mode=False):
        self.hdf5 = HDF5Manager(filename)
        links = self.hdf5.get_matching_datasets(attributes=attributes)
        self.hdf5.load_dataset_by_link(dataset_links=links,length=length,length_mode=length_mode)
        
    def save_simulation_results(self, filename, photon_dataset):
        with h5py.File(filename, 'a') as f:
            dataset_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            group = f.create_group(dataset_name)
            group.create_dataset('position', data=photon_dataset['position'], compression="gzip")
            group.create_dataset('energy', data=photon_dataset['energy'], compression="gzip")
            group.create_dataset('initial_position', data=photon_dataset['initial_position'], compression="gzip")
            group.create_dataset('initial_energy', data=photon_dataset['initial_energy'], compression="gzip")
            group.create_dataset('initial_direction', data=photon_dataset['initial_direction'], compression="gzip")
            group.create_dataset('previous_position', data=photon_dataset['previous_position'], compression="gzip")
            group.create_dataset('scatter', data=photon_dataset['scatter'], compression="gzip")
            group.create_dataset('direction', data=photon_dataset['direction'], compression="gzip")
            print(photon_dataset['RS_position'].keys())
            for photon_id in photon_dataset['RS_position'].keys():
                group.create_dataset(f'RS_position/{photon_id}', data=photon_dataset[f'RS_position'][f'{photon_id}'][:], compression="gzip")

            group.attrs['Emin']          = self.Emin
            group.attrs['Emax']          = self.Emax
            group.attrs['dE']            = self.rs_dE
            group.attrs['E_unit']        = 'keV' 
            group.attrs['velocity']      = self.turbulent_velocity
            group.attrs['velocity_unit'] = 'km/s'
            group.attrs['cluster_name']  = self.cluster_name
            group.attrs['timestamp']     = datetime.now().isoformat()

            print(f"Photons and metadata saved to {filename} in dataset {dataset_name}")

    def test_mode_plot(self, filename):
        P = PlotManager((2,6),(12.8,8))
        G = GeneralFunction()
        v_ion_arr = np.vstack(self.velocity_list)
        photon_direction = np.vstack(self.photon_direction)
        G.gaussian_fitting_with_plot(data=v_ion_arr[:,0],ax=P.axes[0],title=r"$v_{ion,x}$")
        G.gaussian_fitting_with_plot(data=v_ion_arr[:,1],ax=P.axes[1],title=r"$v_{ion,y}$")
        G.gaussian_fitting_with_plot(data=v_ion_arr[:,2],ax=P.axes[2],title=r"$v_{ion,z}$")
        G.gaussian_fitting_with_plot(data=np.array(self.velocity_prop),ax=P.axes[3],title=r"$v_{ion,photon}$")
        G.gaussian_fitting_with_plot(data=np.array(self.energy_ion_rest),ax=P.axes[4],title=r"$E_{ion}$")
        G.gaussian_fitting_with_plot(data=np.array(self.photon_energy_list),ax=P.axes[5],title=r"$E_{input,photon}$")
        theta_x = np.linspace(0,np.pi,100)
        theta_y = np.sin(theta_x)
        P.axes[6].set_title(r"$r$")
        P.axes[7].set_title(r"$\theta$")
        P.axes[8].set_title(r"$\phi$")
        
        v_ion = Photon().carttopolar_array(v_ion_arr)
        v_ion_r = v_ion[:,0]
        v_ion_theta = v_ion[:,1]
        v_ion_phi = v_ion[:,2]
        P.axes[6].hist(v_ion_r, bins=100, color="purple", label="ion v")
        P.axes[7].hist(v_ion_theta, bins=100, density=True, color="purple",histtype="step")
        P.axes[8].hist(v_ion_phi, bins=100, density=True, color="purple",histtype="step")

        P.axes[6].hist(photon_direction[:,0], bins=100, color="darkblue", label="last direction")
        P.axes[7].hist(photon_direction[:,1], bins=100, density=True, color="darkblue",histtype="step")
        P.axes[7].plot(theta_x, theta_y*0.5, color="crimson")
        P.axes[8].hist(photon_direction[:,2], bins=100, density=True, color="darkblue",histtype="step")

        Result = SimulationResult(filename=filename)
        Result.load_result()
        P.axes[6].hist(Result.initial_direction[:,0], bins=100, color="orange",histtype="step", label="initial direction")
        P.axes[7].hist(Result.initial_direction[:,1], bins=100, density=True, color="orange",histtype="step")
        P.axes[8].hist(Result.initial_direction[:,2], bins=100, density=True, color="orange",histtype="step")
        P.axes[6].hist(Result.initial_position[:,0], bins=100, color="green",histtype="step", label="initial position")
        P.axes[7].hist(Result.initial_position[:,1], bins=100, density=True, color="green",histtype="step")
        P.axes[8].hist(Result.initial_position[:,2], bins=100, density=True, color="green",histtype="step")
        P.axes[6].hist(Result.position[:,0], bins=100, color="cyan",histtype="step", range=(0,101), label="last position")
        P.axes[7].hist(Result.position[:,1], bins=100, density=True, color="cyan",histtype="step")
        P.axes[8].hist(Result.position[:,2], bins=100, density=True, color="cyan",histtype="step")
        print("-"*50)

        P.axes[6].legend(fontsize=5)
        line_state = 'w'
        P.axes[9].set_title("tau")
        self.atom = AtomicDataManager(atomic_data_file)
        self.atom.load_line_data(state=line_state)
        self.atom.load_ion_fraction(Z=26, stage=self.atom.z1)
        self.f  = self.atom.f
        self.iz = self.atom.iz
        E = np.linspace(self.Emin,self.Emax,self.dn)
        prob = self.probability_function_list_rsmall["w"]
        tau = - np.log(1 - prob((np.full(len(E),100),E))) 
        P.axes[9].plot(E,tau)

        phys = Physics()
        for delE in [self.physics.deltaE]:
            C = ClusterManager(cluster_name='dummy', rmin=self.rmin, rmax=self.rmax, division=self.division)
            prob = phys.resonance_scattering_probability_field_delE(line_state=line_state, rmax=self.rmax, Emin=self.Emin, Emax=self.Emax, dn=self.dn, dL=self.rmax, f_ne=C.Cluster.data.f_ne, f_T=C.Cluster.data.f_T, f_Z=C.Cluster.data.f_Z, velocity=0, delE=delE)
            tau = - np.log(1 - prob((np.full(len(E),100),E)))
            P.axes[9].plot(E,tau,label=rf'$\Delta E = {delE}$')

        all_photon, scattered_photon = self.return_photon(filename=filename)
        ratio = len(scattered_photon)/len(all_photon)
        bins_edge = np.linspace(6690,6710,100)
        all_photon_n, bins = np.histogram(all_photon, bins=bins_edge)
        scattered_photon_n , bins = np.histogram(scattered_photon, bins=bins_edge)

        prob_sim = scattered_photon_n/all_photon_n
        bins_center = 0.5 * (bins[1:] + bins[:-1])
        tau_sim = - np.log(1 - prob_sim)
        P.axes[9].step(bins_center,tau_sim,color='black',where='mid',label='Simulation')
        print(f'Ratio = {ratio}')

        P.axes[10].hist(Result.initial_energy, bins=1000, histtype='step', color='black', label='all')
        P.axes[10].hist(Result.initial_energy[Result.scatter==True], bins=1000, histtype='step', color='crimson', label='scattered')
        P.axes[10].hist(Result.initial_energy[Result.scatter==False], bins=1000, histtype='step', color='blue', label='unscattered')
        P.axes[11].hist(Result.energy, bins=1000, histtype='step', color='red', label='after scatter')
        P.axes[11].hist(Result.initial_energy, bins=1000, histtype='step', color='black', label='initial energy')

        plt.show()
        P.fig.savefig(f'simulation_profile_{self.turbulent_velocity}.png', dpi=300, transparent=False)
        
class SimulationResult:

    def __init__(self, filename='simulation_pks_digit.hdf5'):
        self.filename = filename
        self.photon_dataset = None
    
    def load_keyname(self, keyname=None):
        if keyname is None:
            with h5py.File(self.filename, 'r') as f:
                self.keyname = list(f.keys())[-1]
        else:
            self.keyname = keyname
            with h5py.File(self.filename, 'r') as f:
                if keyname not in f.keys():
                    raise ValueError(f"Keyname '{keyname}' not found in the file.")
        return self.keyname
    
    def load_result(self, keyname=None):
        self.keyname = self.load_keyname(keyname=keyname)
        with h5py.File(self.filename, 'r') as f:
            self.position = f[self.keyname]['position'][:]
            self.energy = f[self.keyname]['energy'][:]
            self.initial_position = f[self.keyname]['initial_position'][:]
            self.initial_energy = f[self.keyname]['initial_energy'][:]
            self.initial_direction = f[self.keyname]['initial_direction'][:]
            self.previous_position = f[self.keyname]['previous_position'][:]
            self.scatter = f[self.keyname]['scatter'][:]
            self.RS_position = {}
            for photon_id in f[self.keyname]['RS_position']:
                self.RS_position[photon_id] = f[self.keyname]['RS_position'][photon_id][:]
            self.Emin = f[self.keyname].attrs['Emin']
            self.Emax = f[self.keyname].attrs['Emax']
            self.dE = f[self.keyname].attrs['dE']
            self.E_unit = f[self.keyname].attrs['E_unit']
            self.velocity = f[self.keyname].attrs['velocity']
            self.velocity_unit = f[self.keyname].attrs['velocity_unit']
            self.cluster_name = f[self.keyname].attrs['cluster_name']
            self.timestamp = f[self.keyname].attrs['timestamp']

        print(f"Loaded simulation results from {self.filename} in dataset {self.keyname}")

class VectorCalculator:
    def __init__(self):
        pass

    def angle_between_spherical_vectors(self, theta1, phi1, theta2, phi2, degrees=False):
        """
        2つの極座標ベクトル配列（θ, φ）同士のなす角（±90度 or ±π/2）を計算する。
        
        Parameters
        ----------
        theta1, phi1 : ndarray
            ベクトルAの極角・方位角（shape = (N,)）
        theta2, phi2 : ndarray
            ベクトルBの極角・方位角（shape = (N,)）
        degrees : bool
            Trueなら角度を度で返す。Falseならラジアンで返す。
            
        Returns
        -------
        angle : ndarray
            各ペアのなす角（±90度 or ±π/2）配列（shape = (N,)）
        """
        cos_alpha = (
            np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) +
            np.cos(theta1) * np.cos(theta2)
        )
        #cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        angle = np.arccos(cos_alpha)
        if degrees:
            angle = np.degrees(angle)
        
        return angle

class Detector:
    '''
    Detector class for the simulation.
    This class calculates the angular diameter distance and the photon division list.
    '''
    def __init__(self, cluster='perseus', sphere_radius=825, savefile='simulation_perseus_digit.hdf5'):
        omega_s=0.73 
        H0=70 
        self.redshift     = ClusterManager(cluster).Cluster.redshift
        self.cluster_name = cluster
        self.omega_s      = omega_s                                   # Cosmological parameters of the dark energy
        self.omega_m      = 1 - self.omega_s                          # Cosmological parameters of the matter
        self.H0           = H0*u.km/u.s/u.Mpc                         # Hubble constant in km/s/Mpc
        self.sphere_radius_value = sphere_radius                      # Radius of the sphere in kpc
        self.shere_radius = sphere_radius * u.kpc                     # Radius of the sphere in kpc
        self.photon_div_list = np.array([0.5,1,2,3,4,5,10,15,20])     # arcmin
        self.savefile     = savefile
        self.calculate_cosmological_distance()
        self.resolve_pixel_cluster_deg = 30.5/2                                 # resolve 1pixel FOV in arcsec
        self.center_pixel_cluster_deg  = self.resolve_pixel_cluster_deg * 2     # resolve 4pixel FOV in degrees
        self.resolve_3by3_cluster_deg  = self.resolve_pixel_cluster_deg * 3 * np.sqrt(2)    # resolve 9pixel FOV in degrees
        self.outer_pixel_cluster_deg   = self.resolve_pixel_cluster_deg * 6     # resolve 9pixel FOV in degrees
        self.arcsec2kpc        = (self.angular_diameter_distance *2 * np.pi / (360*60*60)).to('kpc')
        self.resolve_pixel_kpc = self.resolve_pixel_cluster_deg * self.arcsec2kpc
        self.resolve_pixel_deg = np.arctan(self.resolve_pixel_kpc.value/self.shere_radius.value) * 180/np.pi
        self.center_pixel_kpc  = self.center_pixel_cluster_deg * self.arcsec2kpc
        self.resolve_3by3_kpc  = self.resolve_3by3_cluster_deg * self.arcsec2kpc
        self.resolve_3by3_deg  = np.arctan(self.resolve_3by3_kpc.value/self.shere_radius.value)  * 180/np.pi
        self.center_pixel_deg  = np.arctan(self.center_pixel_kpc.value/self.shere_radius.value)  * 180/np.pi
        self.outer_pixel_kpc   = self.outer_pixel_cluster_deg * self.arcsec2kpc
        self.outer_pixel_deg   = np.arctan(self.outer_pixel_kpc.value/self.shere_radius.value)   * 180/np.pi
        self.photon_div_list_kpc = self.photon_div_list * 60 * self.arcsec2kpc
        self.photon_div_list_deg = np.arctan(self.photon_div_list_kpc.value/self.shere_radius.value) * 180/np.pi
        print(f'arcsec2kpc         = {self.arcsec2kpc}')
        print(f'Resolve FOV        = {self.resolve_pixel_deg} degrees')
        print(f'Resolve 3by3 FOV   = {self.resolve_3by3_deg} degrees')
        print(f'Resolve center FOV = {self.center_pixel_deg} degrees')
        print(f'Resolve outer FOV  = {self.outer_pixel_deg} degrees')
        print(f'Photon Div List    = {self.photon_div_list_deg} degrees')

    def deg2arcmin(self, deg):
        #return np.tan(deg*np.pi/180) * (self.shere_radius.value/self.arcsec2kpc.value) * 60
        return np.sin(deg*np.pi/180) * (self.shere_radius.value/self.angular_diameter_distance.to("kpc").value) * 180/np.pi * 60

    def calculate_cosmological_distance(self):
        self.luminosity_distance       = (self.integrate_inverse_E_z() * const.c / self.H0 * (1 + self.redshift)).to('Mpc')
        self.angular_diameter_distance = (self.luminosity_distance/(1+self.redshift)**2).to('Mpc')
        print(f'Luminosity Distance       = {self.luminosity_distance}')
        print(f'Angular Diameter Distance = {self.angular_diameter_distance}')

    def E_z(self, z):
        return np.sqrt(self.omega_m * (1 + z)**3 + self.omega_s)

    def integrate_inverse_E_z(self):
        integrand = lambda z: 1 / self.E_z(z)
        result, error = quad(integrand, 0, self.redshift)
        return result

    def calculate_photon_degree(self,keyname='None'):
        """
        Calculate the angle between the photon direction and the normal vector of the sphere.
        Parameters
        ----------
        filename : str
            The name of the HDF5 file to read.
        keyname : str
            The name of the dataset to read from the HDF5 file.
        sphere_radius : float
            The radius of the sphere in kpc.
        Returns
        -------
        tuple
            A tuple containing the following elements:
            - deg_hist: The histogram of angles between the photon direction and the normal vector.
            - energy: The energy of the photons.
            - velocity: The velocity of the photons.
            - scatter: The scatter flag of the photons.
            - deg_hist_noRS: The histogram of angles between the photon direction and the normal vector without RS.
            - initial_energy: The initial energy of the photons.
            - initial_position: The initial position of the photons.
            - RS_position: The position of the photons after RS.
            - position: The final position of the photons.
        """
        filename = self.savefile
        with h5py.File(filename, 'r') as hdf5_file:
            if keyname == 'None':
                keyname = list(hdf5_file.keys())[-1]

        hdf5 = HDF5Manager(filename)
        photon = Photon()
        (
            energy, position, previous_position, scatter, velocity,
            initial_position, initial_direction, initial_energy,
            RS_position, direction
        ) = hdf5.load_simulation_data(filename=filename, keyname=keyname)

        # === 各ベクトルの極座標成分抽出 ===
        _, theta_last_position, phi_last_position   = position[:, 0], position[:, 1], position[:, 2]
        _, theta_last_direction, phi_last_direction = direction[:, 0], direction[:, 1], direction[:, 2]
        _, theta_initial_direction, phi_initial_direction = initial_direction[:, 0], initial_direction[:, 1], initial_direction[:, 2]

        # === RSがない場合の最終位置を計算 ===
        no_RS_position = photon.propagate_to_rmax_array(initial_position, initial_direction, self.sphere_radius_value)
        _, theta_last_position_noRS, phi_last_position_noRS = no_RS_position[:, 0], no_RS_position[:, 1], no_RS_position[:, 2]

        # === ベクトル間のなす角を計算 ===
        vector_calculator = VectorCalculator()
        deg_hist = vector_calculator.angle_between_spherical_vectors(theta_last_position, phi_last_position, theta_last_direction, phi_last_direction, degrees=True)
        deg_hist_noRS = vector_calculator.angle_between_spherical_vectors(theta_initial_direction, phi_initial_direction, theta_last_position_noRS, phi_last_position_noRS, degrees=True)
        return (
            deg_hist, energy, velocity, scatter, deg_hist_noRS,
            initial_energy, initial_position, RS_position, position
        )

    def make_spectrum_divided_angle(self, keyname='None', angle_spec=False, plotting=True,idx=-4):
        filename = self.savefile
        hdf5_file = h5py.File(filename, 'r')
        if keyname == 'None':
            keyname = list(hdf5_file.keys())[idx]
        else:
            keyname = keyname
        v_in=hdf5_file[keyname].attrs["velocity"]
        print(v_in)
        self.deg_hist, self.energies, self.velocity, self.scatter, self.deg_hist_noRS, initial_energy, initial_position, RS_position, position = self.calculate_photon_degree(keyname)

        center_mask       = (self.deg_hist < self.center_pixel_deg) & (self.deg_hist > 0)
        outer_mask        = (self.deg_hist < self.outer_pixel_deg) & (self.deg_hist > self.center_pixel_deg)
        center_mask_noRS  = (self.deg_hist_noRS < self.center_pixel_deg) & (self.deg_hist_noRS > 0)
        outer_mask_noRS   = (self.deg_hist_noRS < self.outer_pixel_deg) & (self.deg_hist_noRS > self.center_pixel_deg)
        resolve_3by3_mask = (self.deg_hist < self.resolve_3by3_deg) & (self.deg_hist > 0) 
        resolve_3by3_mask_noRS = (self.deg_hist_noRS < self.resolve_3by3_deg) & (self.deg_hist_noRS > 0)
        resolve_FOV_mask = (self.deg_hist < self.outer_pixel_deg)
        resolve_FOV_mask_noRS = (self.deg_hist_noRS < self.outer_pixel_deg)

        print(f'Center 4 pixel FOV counts   = {len(self.deg_hist[center_mask])}')
        print(f'Outer FOV counts            = {len(self.deg_hist[outer_mask])}')
        print(f'Resolve 3by3 FOV counts     = {len(self.deg_hist[resolve_3by3_mask])}')
        print(f'Center 4 pixel FOV fraction = {len(self.deg_hist[center_mask]) / len(self.deg_hist)}')
        print(f'Outer FOV fraction          = {len(self.deg_hist[outer_mask]) / len(self.deg_hist)}')
        print(f'Resolve 3by3 FOV fraction   = {len(self.deg_hist[resolve_3by3_mask]) / len(self.deg_hist)}')
        if plotting:
            self._plot_angle()
        self.arcmin_hist = self.deg2arcmin(self.deg_hist)
        self.arcmin_hist_noRS = self.deg2arcmin(self.deg_hist_noRS)
        # if angle_spec:
        #     furukawa_arcmin = [0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]

        #     # 辞書形式でマスクを自動生成
        #     arcmin_masks = {
        #         f"{low:.1f}_{high:.1f}": (self.arcmin_hist > low) & (self.arcmin_hist <= high)
        #         for low, high in zip(furukawa_arcmin[:-1], furukawa_arcmin[1:])
        #     }
        #     arcmin_masks_noRS = {
        #         f"{low:.1f}_{high:.1f}": (self.arcmin_hist_noRS > low) & (self.arcmin_hist_noRS <= high)
        #         for low, high in zip(furukawa_arcmin[:-1], furukawa_arcmin[1:])
        #     }

        #     for key, mask in arcmin_masks.items():
        #         print(f'Arcmin {key} counts = {len(self.arcmin_hist[mask])}')
        #         self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'arcmin_{key}', TEStype='TMU542', Datatype='PHA', pha=self.energies[mask])
        #     for key, mask in arcmin_masks_noRS.items():
        #         print(f'Arcmin {key} counts = {len(self.arcmin_hist_noRS[mask])}')
        #         self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'arcmin_{key}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[mask])

        if angle_spec:
            # 0–2.0 (0.5刻み), 2.0–5.0 (1.0刻み), 5.0–20.0 (5.0刻み)
            if self.cluster_name == 'perseus':
                arcmin_bins = (
                list(np.arange(0, 2.0, 0.5)) +
                list(np.arange(2.0, 5.0, 1.0)) +
                list(np.arange(5.0, 20.0, 5.0)) +
                [20.0]
                )
            elif self.cluster_name == 'pks':
                arcmin_bins = (
                list(np.arange(0, 2.0, 0.5)) +
                list(np.arange(2.0, 5.0, 1.0)) +
                list(np.arange(5.0, 10.0, 5.0)) +
                [10.0]
                )

            arcmin_masks = {
                f"{low:.1f}_{high:.1f}": (self.arcmin_hist > low) & (self.arcmin_hist <= high)
                for low, high in zip(arcmin_bins[:-1], arcmin_bins[1:])
            }
            arcmin_masks_noRS = {
                f"{low:.1f}_{high:.1f}": (self.arcmin_hist_noRS > low) & (self.arcmin_hist_noRS <= high)
                for low, high in zip(arcmin_bins[:-1], arcmin_bins[1:])
            }
            os.makedirs('./output_divided_arcmin_fits', exist_ok=True)
            os.makedirs(f'./output_divided_arcmin_fits/{v_in}', exist_ok=True)
            for key, mask in arcmin_masks.items():
                print(f'Arcmin {key} counts = {len(self.arcmin_hist[mask])}')
                # exposure tiem setted to adapt hitomi/perseus observation. 
                self.fits2xspec(binsize=0.5, exptime=1e7, fwhm=0.0001, name=f'./output_divided_arcmin_fits/{v_in}/arcmin_{key}', TEStype='TMU542', Datatype='PHA', pha=self.energies[mask])

            for key, mask in arcmin_masks_noRS.items():
                print(f'Arcmin {key} counts = {len(self.arcmin_hist_noRS[mask])}')
                self.fits2xspec(binsize=0.5, exptime=1e7, fwhm=0.0001, name=f'./output_divided_arcmin_fits/{v_in}/arcmin_{key}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[mask])

        if self.cluster_name == 'perseus':
            self.fits2xspec(binsize=0.5, exptime=1e5, fwhm=0.0001, name=f'3by3_pixel_{self.velocity}', TEStype='TMU542', Datatype='PHA', pha=self.energies[resolve_3by3_mask])
            self.fits2xspec(binsize=0.5, exptime=1e5, fwhm=0.0001, name=f'3by3_pixel_{self.velocity}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[resolve_3by3_mask_noRS])
            if plotting:
                plt.hist(self.energies[resolve_3by3_mask], bins=500, density=True, histtype='step', color='black', label='All Data')
                plt.hist(self.energies[resolve_3by3_mask_noRS], bins=500, density=True, histtype='step', color='red', label='Scattered photon')
                plt.show()
        elif self.cluster_name == 'pks':
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'center_pixel_{self.velocity}', TEStype='TMU542', Datatype='PHA', pha=self.energies[center_mask])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'center_pixel_{self.velocity}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[center_mask_noRS])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'outer_pixel_{self.velocity}', TEStype='TMU542', Datatype='PHA', pha=self.energies[outer_mask])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'outer_pixel_{self.velocity}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[outer_mask_noRS])
        elif self.cluster_name == 'abell478':
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'center_pixel_{self.velocity}', TEStype='TMU542', Datatype='PHA', pha=self.energies[center_mask])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'center_pixel_{self.velocity}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[center_mask_noRS])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'all_pixel_{self.velocity}', TEStype='TMU542', Datatype='PHA', pha=self.energies[resolve_FOV_mask])
            self.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'all_pixel_{self.velocity}_noRS', TEStype='TMU542', Datatype='PHA', pha=self.energies[resolve_FOV_mask_noRS])

    def _plot_angle(self):
        P = PlotManager((2, 1), (12.8, 8), sharex=True)
        deg_hist_bins = np.logspace(-1, np.log10(90), 100)
        P.axes[0].hist(self.deg_hist_noRS, bins=deg_hist_bins, histtype='step', color='blue', label='All Data(RS exclude)')
        P.axes[0].hist(self.deg_hist, bins=deg_hist_bins, histtype='step', color='black', label='All Data(RS include)')
        P.axes[0].hist(self.deg_hist[self.scatter], bins=deg_hist_bins, histtype='step', color='red', label='Scattered photon')
        deg_hist_bin, bins = np.histogram(self.deg_hist, bins=deg_hist_bins)
        deg_hist_noRS_bin, _ = np.histogram(self.deg_hist_noRS, bins=deg_hist_bins)
        P.axes[1].step(deg_hist_bins[:-1],deg_hist_bin/deg_hist_noRS_bin, color='black') 
        P.axes[1].set_xlabel('Angle with Normal Vector (degrees)')
        P.ax.set_ylabel('Counts')
        P.axes[1].set_ylabel('RS include/RS exclude')
        P.axes[0].set_title(f'Angle with Normal Vector {self.cluster_name}')
        P.axes[0].set_xscale('log')
        P.axes[1].set_xscale('log')
        P.axes[0].grid(linestyle='dashed')
        P.axes[1].grid(linestyle='dashed')
        P.axes[0].axvline(self.center_pixel_deg, color='red', linestyle='--')
        P.axes[0].axvline(self.outer_pixel_deg, color='blue', linestyle='--')
        P.axes[0].axvline(self.resolve_3by3_deg, color='green', linestyle='--')
        P.axes[1].axvline(self.center_pixel_deg, color='red', linestyle='--', label='Resolve center 4 pixel FOV')
        P.axes[1].axvline(self.outer_pixel_deg, color='blue', linestyle='--', label='Resolve FOV')
        P.axes[1].axvline(self.resolve_3by3_deg, color='green', linestyle='--', label='Resolve 3x3 FOV')
        P.axes[0].legend()
        P.axes[1].legend()
        P.fig.savefig('Angle_with_Tangent_Plane.png', dpi=300, transparent=False)
        plt.show()
        arcmin_hist = self.deg2arcmin(self.deg_hist)
        arcmin_hist_noRS = self.deg2arcmin(self.deg_hist_noRS)
        bins = np.logspace(-1, np.log10(40), 100)
        P = PlotManager((2, 1), (12.8, 8), sharex=True)
        P.axes[0].hist(arcmin_hist, bins=bins, histtype='step', color='black', label='All Data')
        P.axes[0].hist(arcmin_hist_noRS, bins=bins, histtype='step', color='blue', label='All Data(RS exclude)')
        P.axes[0].set_xlabel('Angle with Normal Vector (arcmin)')
        P.axes[0].set_ylabel('Counts')
        P.axes[0].set_title(f'Angle with Normal Vector {self.cluster_name}')
        P.axes[0].set_xscale('log')
        #P.axes[0].set_yscale('log')
        P.axes[0].grid(linestyle='dashed')
        P.axes[1].step(bins[:-1], deg_hist_bin/deg_hist_noRS_bin, color='black')
        P.fig.savefig('Angle_with_Tangent_Plane_arcmin.png', dpi=300, transparent=False)
        plt.show()

#? : Not  
    def _plot_angle_with_dummy_angle(self, num_bins=50, angle="degree"):
        if angle == "degree":
            original_deg = np.linspace(0, 90, len(self.deg_hist))  # 0〜90度対応の半径
            rebinned_deg = np.linspace(0, 90, num_bins)
        elif angle == "arcmin":
            original_deg = np.linspace(0, 60*90, len(self.deg_hist))  # 0〜90度対応の半径
            rebinned_deg = np.linspace(0, 60*90, num_bins)

        rebinned_hist = np.interp(rebinned_deg, original_deg, self.deg_hist)  # 線形補間

        # === θ方向（2πを一定分割）===
        theta_vals = np.linspace(0, 2 * np.pi, 360)
        r_vals = rebinned_deg

        # === グリッド作成 ===
        theta_grid, r_grid = np.meshgrid(theta_vals, r_vals)

        # === ヒストグラムをθ方向に複製 ===
        hist_grid = np.tile(rebinned_hist[:, np.newaxis], (1, len(theta_vals)))

        # === プロット ===
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        c = ax.pcolormesh(theta_grid, r_grid, hist_grid, cmap='viridis', shading='auto')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Angle-dependent Distribution (binned to {num_bins})")
        plt.colorbar(c, ax=ax, label='Counts')
        plt.show()

    def make_spec_multi(self):
        keynames = h5py.File(self.savefile, 'r').keys()
        for keyname in keynames:
            self.make_spectrum_divided_angle(keyname=keyname,angle_spec=True,plotting=False)

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

    def histogram(self, pha, binsize=0.5):
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

    def fits2xspec(self,binsize=0.5, exptime=1, fwhm=0.0001, name='test', TEStype='TMU542', Datatype='PHA', pha=None):
        n, bins = self.histogram(pha[pha>0], binsize=binsize)
        filename = name + ".fits"
        rmfname = name + ".rmf"
        Exposuretime = exptime
        tc = int(n.sum())
        chn = len(bins)-1
        #x = (bins[:-1]+bins[1:])/2
        x = np.arange(0, (len(bins)-1), 1)
        y = n
        
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
        
        # make fits
        col_x = fits.Column(name='CHANNEL', format='J', array=np.asarray(x))
        col_y = fits.Column(name='COUNTS', format='J', unit='count', array=np.asarray(y))
        cols  = fits.ColDefs([col_x, col_y])
        tbhdu = fits.BinTableHDU.from_columns(cols)

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
        exthdr['USER']     = ('Keita Tanaka', 'User name of creator')
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

        hdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([hdu, tbhdu])
        thdulist.writeto(filename, overwrite=True)

    def reconstruct_from_bin_centers(self, bin_centers, counts, method='center'):
        """
        binの中心とカウントから擬似的な元データを復元する

        Parameters
        ----------
        counts : array-like
            各ビンのカウント数
        bin_centers : array-like
            各ビンの中心座標
        method : str
            'center'：中心をそのまま複製して復元（離散）
            'uniform'：隣接ビンの中心を使って範囲を推定し、一様乱数で復元（連続）

        Returns
        -------
        reconstructed_data : ndarray
            擬似的な元データ配列
        """
        counts = np.asarray(counts).astype(int)  # ← intに変換！
        bin_centers = np.asarray(bin_centers)

        if method == 'center':
            # 各ビンの中心値をカウント回数だけ繰り返す
            data = np.repeat(bin_centers, counts)

        elif method == 'uniform':
            # ビン幅を推定（中心間の距離から）
            bin_edges_left = np.empty_like(bin_centers)
            bin_edges_right = np.empty_like(bin_centers)

            # 左端と右端（両端は外挿）
            bin_edges_left[1:] = 0.5 * (bin_centers[1:] + bin_centers[:-1])
            bin_edges_left[0] = bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2

            bin_edges_right[:-1] = 0.5 * (bin_centers[1:] + bin_centers[:-1])
            bin_edges_right[-1] = bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2

            data = []
            for c, left, right in zip(counts, bin_edges_left, bin_edges_right):
                data.append(np.random.uniform(left, right, size=c))
            data = np.concatenate(data)
        else:
            raise ValueError("method must be 'center' or 'uniform'")

        return data

class ForHeasim:
    """
    The analysis of the Heasim spectrum.
    
    """
    def __init__(self) -> None:
        pass
    
    def fit_data_bapec(self, spec, rmf):
        basename = os.path.splitext(os.path.basename(spec))[0]
        commands = f"""
xspec << EOF
xsel
statistic cstat
data {spec}
resp {rmf}
model bapec
3.5, 1
0.4, 1
0,1
150, 1

fit 200
fit 200

data none
energy 6.5 8.0 1500
cpd /xs
setplot comm wdata {basename}.qdp
plot model
exit
no
EOF
"""
        subprocess.call(commands, shell=True)

    def multi_fit_data_bapec(self, keyname):
        file_list = glob.glob(f'{keyname}*.fits')
        for file in file_list:
            rmf = file.replace('.fits', '.rmf')
            self.fit_data_bapec(file, rmf)


    def fit2qdp(self, fits, rmf, save_dir='./'):
        """
        Convert fits file to QDP format using xspec for heasim.
        Parameters:
            fits (str): Path to the fits file.
            rmf (str): Path to the RMF file.
            save_dir (str): Directory to save the QDP file.
        """
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(fits))[0]
        commands = f"""
xspec << EOF
data {fits}
resp {rmf}
plot data
iplot
wdata {save_dir}{basename}.qdp

exit
no
EOF
"""
        subprocess.run(commands, shell=True, check=True)

    def multi_fit2qdp(self, save_dir='./'):
        file_list = glob.glob(f'*.fits')
        for file in file_list:
            rmf = file.replace('.fits', '.rmf')
            self.fit2qdp(file, rmf, save_dir)

    def make_qdp_all(self):
        os.chdir("./output_divided_arcmin_fits")
        dirs = glob.glob('*')
        for dir in dirs:
            os.chdir(dir)
            last_dir = os.path.basename(dir.rstrip("/"))
            self.multi_fit2qdp(save_dir=f'{dir}/../../../output_divided_arcmin_qdp/{last_dir}/')
            os.chdir('../')


    def select_3by3_reg_for_perseus(self, spectrum_dir: str):
        """
        Divide the FoV into a 3×3 grid (Hitomi/Perseus obs-23 centre)
        and extract spectra with xselect.

        *Input files*  
        perseus_RS_heasim_[with|no]RS_vt150_001.fits  ← 例

        *Output files*  
        ./
        """
        # --- resolve & create paths ------------------------------------------------
        spec_dir = Path(spectrum_dir).expanduser().resolve()

        # region file (=現作業ディレクトリに置いている想定)
        region_file = Path.cwd() / "3by3_demo.reg"

        # --- scan FITS files -------------------------------------------------------
        for fits_path in spec_dir.glob("*.fits"):
            fname = fits_path.name                       # perseus_RS_heasim_… .fits

            # ▽ vt###_### を丸ごと抜き出す（vt150_001 など）
            m_vt = re.search(r"(vt\d+_\d+)", fname)
            if not m_vt:
                print(f"[SKIP] vt???_??? が見つからない: {fname}")
                continue
            vt_tag = m_vt.group(1)                       # 'vt150_001'

            # ▽ RSフラグ
            rs_flag = "wRS"  if "withRS" in fname else "woRS"

            # ▽ 出力 PI ファイル
            out_pi = f"./3by3_{vt_tag}_{rs_flag}.pi"

            # --- xselect -----------------------------------------------------------
            cmd = f"""
xselect << EOF
xsel
no
read events
{spec_dir}
{fname}
y
filter region {region_file}
extract spectrum
save spectrum {out_pi}
exit
n
EOF
"""
            subprocess.run(cmd, shell=True, check=True)

        print(f"[DONE] Spectra saved")

    def select_3by3_multidir(self, base_dir):
        """
        Divide the FoV into a 3×3 grid (Hitomi/Perseus obs-23 centre)
        and extract spectra with xselect.
        """
        dirs = glob.glob(f"{base_dir}/vt*")
        for direc in dirs:
            target_dir = f"{direc}/evt"
            self.select_3by3_reg_for_perseus(target_dir)
        

    def select_reg_for_pks(self, region, idx=None):
        """
        For analysis of the heasim data of PKS0745.
        simulated event file is divided by the region.
        """
        import re
        # spec_dir = "../../analysis/"
        # obs_dir = '../spec'
        spec_dir = "."
        obs_dir = 'RS_sim_v309_all_vt100_vc0'
        if region == "center":
            regfile = 'center_sky.reg'
        elif region == "outer":
            regfile = 'outer_sky.reg'
        cdir = os.getcwd()
        if idx == None:
            dirs = glob.glob(f'{obs_dir}')
            #dirs = glob.glob(f'{obs_dir}/*')
        else:
            dirs = glob.glob(f'../spec/*_{idx}')
        print(cdir)
        for dir in dirs:
            print(dir)
            os.chdir(dir)
            print(os.getcwd())
            last_dir = os.path.basename(dir.rstrip("/"))
            match = re.search(r'vt(\d+)', last_dir)
            if match:
                vt_value = int(match.group(1))  # 数値として取り出す
                print(vt_value)  # 出力: 100
        # 共鳴散乱あり
        input_text = f"""xsel
no
read events
./obs_event
pks0745_ICM_resolve_withRS_obs.fits
yes
filter region {cdir}/{regfile}
extr spec
save spec ../../analysis/{region}_v{vt_value}_wRS.pi
yes
    """

        subprocess.run("xselect", input=input_text, text=True, check=True)

        # 共鳴散乱なし
        input_text = f"""xsel
no
read events
./obs_event
pks0745_ICM_resolve_noRS_obs.fits
yes
filter region {cdir}/{regfile}
extr spec
save spec ../../analysis/{region}_v{vt_value}_woRS.pi
yes
    """

        subprocess.run("xselect", input=input_text, text=True, check=True)

        os.chdir(cdir)

    def bootstrap_res(self, filename, ci=68):
        ratio_list = []
        with h5py.File(filename, "r") as f:
            for key in f.keys():
                ratio = f[f"{key}/fitting_result"]["2/dzgaus"]["ratio"]["value"][...]
                ratio_list.append(ratio)
                print(f"{key}: {ratio}")

        ratio_list = np.asarray(ratio_list)

        # ---- ベスト値 & 誤差（中央値 ± 1σ 相当） ----
        best = np.median(ratio_list)
        lo, hi = np.percentile(ratio_list, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        err_minus = best - lo
        err_plus  = hi   - best

        print(f"ratio = {best:.4f}  (+{err_plus:.4f} / -{err_minus:.4f})  [{ci}% CI]")
        P = PlotManager(figsize=(6, 4), label_size=15)
        # ---- 描画 ----
        P.axes[0].hist(ratio_list, bins=40, color="lightgray", edgecolor="k")
        P.axes[0].axvline(best, color="crimson", linewidth=2, label=f"median")
        P.axes[0].axvline(lo,   color="royalblue", linestyle="--")
        P.axes[0].axvline(hi,   color="royalblue", linestyle="--", label=f"{ci}% CI")
        P.axes[0].set_xlabel("w/z ratio")
        P.axes[0].set_ylabel("count")
        P.axes[0].set_title(rf"ratio = {best:.4f}  (+{err_plus:.4f} / -{err_minus:.4f})")
        P.axes[0].legend()
        P.fig.tight_layout()
        P.fig.savefig(f"figure/bootstrap_ratio.pdf")
        plt.show()

class PlotManager:
    def __init__(self, subplot_shape=(1, 1), figsize=(8, 6), spine_width=1.5,
                 label_size=10, height_ratios=None, sharex=False):
        self._apply_rcparams(label_size=label_size)
        self.spine_width = spine_width
        self.subplot_shape = subplot_shape
        self.fig = plt.figure(figsize=figsize)
        self.axes = []

        rows, cols = subplot_shape
        share_ax = None  # 基準となる軸

        if height_ratios is not None:
            gs = gridspec.GridSpec(rows, cols, height_ratios=height_ratios, figure=self.fig)
            for i in range(rows * cols):
                ax = self.fig.add_subplot(gs[i], sharex=share_ax if sharex else None)
                if i == 0 and sharex:
                    share_ax = ax  # 最初の軸を共有対象として記憶
                self._set_ax_style(ax)
                self.axes.append(ax)
        else:
            for i in range(1, rows * cols + 1):
                ax = self.fig.add_subplot(rows, cols, i, sharex=share_ax if sharex else None)
                if i == 1 and sharex:
                    share_ax = ax
                self._set_ax_style(ax)
                self.axes.append(ax)

        for i, ax in enumerate(self.axes):
            setattr(self, f'ax{i+1}' if i > 0 else 'ax', ax)

        self.fig.align_labels()

    def _apply_rcparams(self, label_size=10):
        rc = {
            'axes.labelsize': label_size,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 10,
            'font.weight': 500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'xtick.labelsize': label_size,
            'ytick.labelsize': label_size,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'text.usetex': False,
            'font.family': 'serif',
        }
        mpl.rcParams.update(rc)

    def _set_ax_style(self, ax):
        ax.tick_params(axis='both', direction='in', width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

    def set_spine_width_all_axes(self, width=1.5):
        for ax in self.fig.axes:
            for spine in ax.spines.values():
                spine.set_linewidth(width)

class ToyBox:
    def __init__(self):
        self.keV2K = const.e.value*1e+3/const.k_B.value
        self.K2keV = 1/self.keV2K
        self.iron_atomic_weight = 55.845

    def Doppler_broadening(self, v, E0):
        return E0 * np.sqrt((1+v/const.c)/(1-v/const.c))

    def thomson_scattering(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 定数
        r0 = 2.817e-15  # 古典的電子半径 (m)

        # 角度（入射方向と放出方向のなす角、ラジアン）
        theta = np.linspace(0, np.pi, 100)  # thetaは0からπ
        phi = np.linspace(0, 2 * np.pi, 100)  # phiは0から2π
        theta, phi = np.meshgrid(theta, phi)

        # 微分断面積（入射角度 θ に依存する）
        d_sigma_d_omega = (r0**2 / 2) * (1 + np.cos(theta)**2)

        # 球座標系をデカルト座標系に変換
        x = d_sigma_d_omega * np.sin(theta) * np.cos(phi)
        y = d_sigma_d_omega * np.sin(theta) * np.sin(phi)
        z = d_sigma_d_omega * np.cos(theta)
        norm = z.max()
        y = y / z.max()  # 正規化
        x = x / z.max()
        z = z / z.max()

        # プロットの作成
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # サーフェスプロット（元のデータ）
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', edgecolor='none', alpha=0.7, label='Original')

        # グラフのラベルと視点の設定
        ax.set_title('Thomson Scattering Differential Cross-Section (Unpolarized Photon)', fontsize=15)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.view_init(elev=30, azim=120)  # 視点の角度調整

        # カラーバーを追加
        cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.5)  # shrinkを追加してサイズを小さく
        cbar.set_label(f'Differential Cross-Section (m²/sr/{norm})', fontsize=12)

        # 座標軸を点線にする
        ax.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
        ax.yaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)
        ax.zaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5)

        # 軸を非表示にする
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.zaxis.set_visible(True)

        # X, Y, Z軸の範囲を取得してアスペクト比を揃える
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_box_aspect([1, 1, 1])  # 1:1:1 のアスペクト比に設定
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        fig.savefig('thomson_scattering.pdf',dpi=300,transparent=True)
        plt.show()

    def plot_positions_3d(self):
        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # カラーマップと正規化
        cmap = cm.get_cmap('plasma')
        norm = mcolors.Normalize(vmin=2000, vmax=8000)  # 6600から6750に正規化
        with h5py.File('simulation.hdf5', 'r') as f:
            key = 'simulation_20241028_193942'
            keyname = f[key].keys()
            for e, photon_id in enumerate(keyname):
                positions = np.array(f[key +'/'+ photon_id + '/positions'])
                energies  = np.array(f[key +'/'+ photon_id + '/energies'])
                scatter   = f[key +'/'+ photon_id].attrs['scatter']
                max_energy = np.max(energies)
                cmap = plt.get_cmap('plasma')
                if scatter == False:
                    lw = 1
                else:
                    lw = 2
                if e < 1000:
                    x, y, z = Photon().polartocart_array(positions)
                    ax.plot3D(x, y, z, color=cmap(norm(max_energy)), alpha=0.5, linewidth=lw)  # グラデーションカラーを適用
                    ax.scatter(x[0], y[0], z[0], color='white', s=5)  # 点をプロット

        radius = 1000
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color='cyan', alpha=0.05, rstride=5, cstride=5, edgecolor='white')

        # 軸ラベルと色の変更
        ax.set_xlabel('X (kpc)', color='white')
        ax.set_ylabel('Y (kpc)', color='white')
        ax.set_zlabel('Z (kpc)', color='white')

        # 軸の背景色や目盛りの色を変更
        # ax.w_xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        # ax.w_yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        # ax.w_zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        # 視点の変更
        ax.view_init(elev=20, azim=120)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm,ax=ax, label='Energy (eV)')
        cbar.ax.yaxis.label.set_color('white')
        ax.scatter(0, 0, 0, color='white', s=100)  # 球の中心にハイライトを追加
        fig.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.savefig('positions_3d.pdf',dpi=300)
        plt.show()

    def plot_ene(self):
        P = PlotManager('single')
        rng = (6500,6800)
        # カラーマップと正規化
        cmap = cm.get_cmap('plasma')

        key_list = ['simulation_20241108_165808', 'simulation_20241108_172207', 'simulation_20241108_174706', 'simulation_20241108_183017']
        with h5py.File('simulation.hdf5', 'r') as f:
            for ee,key in enumerate(key_list):
                keyname = f[key].keys()
                ver = f[key].attrs['velocity']
                ene = []
                ene_RS = []
                ene_nonRS = []
                for e, photon_id in enumerate(keyname):
                    positions = np.array(f[key +'/'+ photon_id + '/positions'])
                    energies  = np.array(f[key +'/'+ photon_id + '/energies'])
                    scatter   = f[key +'/'+ photon_id].attrs['scatter']
                    ene.append(energies[-1])
                    if scatter == False:
                        ene_nonRS.append(energies[-1])
                    else:
                        ene_RS.append(energies[-1])
                #P.ax.hist(ene, bins=300,range=rng, histtype='step', color=cm.plasma([ee/4]), label=f'{ver} km/s')
                #P.ax.hist(ene_RS, bins=300,range=rng, histtype='step', color=cm.plasma([ee/4]), label=f'{ver} km/s')
                #P.ax.hist(ene_nonRS, bins=300,range=rng, histtype='step', color=cm.plasma([ee/4]), label=f'{ver} km/s')
                print(f'{ver} km/s')
                print(f'all photons = {len(ene)}')
                print(f'RS photons = {len(ene_RS)}')
                print(f'nonRS photons = {len(ene_nonRS)}')
                print(f'RS fraction = {len(ene_RS)/len(ene)}')

                #P.ax.hist(ene_nonRS, bins=150,range=rng, histtype='step', color=cm.plasma([ee/6]), label=f'{ver} km/s')
            #P.ax.hist(ene_RS, bins=150,range=rng, histtype='step', color='black')
            #P.ax.hist(ene_nonRS, bins=150,range=rng, histtype='step', color='red')
        P.ax.grid(linestyle='dashed')
        P.ax.legend()
        P.fig.tight_layout()
        P.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        P.fig.savefig('RS_RS_photons.pdf',dpi=300)
        plt.show()

    def calc_deltheta(self,D,z):
        H0 = 70
        return (D*u.kpc/(const.c*z/(H0*u.km/u.s/u.Mpc*(1+z)**2))).to('')

    def calculate_tau(self):
        pass

class ForComp:
    def __init__(self):
        pass
    
    def load_qdp(self, filename):
        f = np.loadtxt(filename)
        energy = f[:,0]
        cnt = f[:,2]
        return energy, cnt

    def generate_random_numbers_rejection(self, ebins, spec, size=1000):
        bins_center = ebins
        spec = spec

        max_spec = np.max(spec)
        random_numbers = np.empty(size)
        count = 0

        while count < size:
            rand_energy = np.random.uniform(bins_center[0], bins_center[-1], size - count)
            rand_uniform = np.random.uniform(0, max_spec, size - count)

            spec_interp = np.interp(rand_energy, bins_center, spec)
            accepted = rand_uniform < spec_interp
            num_accepted = np.sum(accepted)

            random_numbers[count:count + num_accepted] = rand_energy[accepted]
            count += num_accepted

        return random_numbers
    
    def sampling(self, filename, size):
        energy, cnt = self.load_qdp(filename)
        photon = self.generate_random_numbers_rejection(energy, cnt, size) # [keV]
        plt.step(energy, cnt)
        plt.hist(photon, bins=1000, histtype="step")
        plt.show()
        D = Detector()
        D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'spectrum_furukawa_0.5arcmin', TEStype='TMU542', Datatype='PHA', pha=photon*1e3) 

    def tanaka_spec(self, rmin, rmax, size):
        pass
        CM = ClusterManager('perseus', rmin, rmax)
        rad_dist = InitialPhotonGenerator(CM, None).generate_random_ne_squared(size=size)
        #RadiationField(6.5,8.0,0.5e-3,'AG89','3.0.8').generate_apec_spectrum_from_hdf5()

    def make_output_spectrum(self):
        '''
        from furukawa output spectrum, make fits file
        '''
        import re
        D = Detector()
        wRS_files = glob.glob('../furukawa_output_spectrum/*wRS*.qdp')
        woRS_files = glob.glob('../furukawa_output_spectrum/*woRS*.qdp')
        for file in wRS_files:
            ene,cnt = self.load_qdp(file)
            data = D.reconstruct_from_bin_centers(ene, cnt)
            filename = os.path.basename(file) 
            match = re.search(r'_(\d+)_([0-9]+)_arcmin', filename)
            if match:
                x = int(match.group(1)) / 10
                y = int(match.group(2)) / 10
                arcmin = f"{x:.1f}_{y:.1f}"
            D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'arcmin_{arcmin}', TEStype='TMU542', Datatype='PHA', pha=data*1e3)
        for file in woRS_files:
            ene,cnt = self.load_qdp(file)
            data = D.reconstruct_from_bin_centers(ene, cnt)
            filename = os.path.basename(file) 
            match = re.search(r'_(\d+)_([0-9]+)_arcmin', filename)
            if match:
                x = int(match.group(1)) / 10
                y = int(match.group(2)) / 10
                arcmin = f"{x:.1f}_{y:.1f}"
            D.fits2xspec(binsize=0.5, exptime=1, fwhm=0.0001, name=f'arcmin_{arcmin}_noRS', TEStype='TMU542', Datatype='PHA', pha=data*1e3)

def main1():
    for v in [0, 100]:
        s = Simulation(physics_list=['RS', 'TS'],rmin=0.1, rmax=840, division=10, Emin = 6500, Emax = 8000, dn = 1000,  size=int(1e5), rs_dE=10, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='perseus')
        s.scatter_generated_photons_div(savefile='simulation_test_seed3.hdf5')
    # I.velocity_sampling(5,100)

def main():
    for v in [250]:
        # s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='perseus', step_dist=5, test_mode=False, tau_mode="mfp")
        s = Simulation(physics_list=['RS'], rmin=0, rmax=825, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e5), rs_dE=20, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='perseus', step_dist=2, test_mode=False, tau_mode="mfp", atomdb_version='3.0.8', abundance_table='AG89', comparison=False)
        s.scatter_generated_photons_div(savefile='simulation_perseus_digit.hdf5', plot=False)
        # s.test_mode_plot(filename='simulation_perseus_digit.hdf5')

def main_309():
    for v in [0, 100, 150, 200, 250, 300, 350]:
        # s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='perseus', step_dist=5, test_mode=False, tau_mode="mfp")
        s = Simulation(physics_list=['RS'], rmin=0, rmax=825, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e6), rs_dE=20, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='perseus', step_dist=2, test_mode=False, tau_mode="mfp", atomdb_version='3.0.9', abundance_table='lpgs', comparison=False)
        s.scatter_generated_photons_div(savefile='simulation_perseus_digit.hdf5', plot=False)
        # s.test_mode_plot(filename='simulation_perseus_digit.hdf5')

def main_pks():
    for v in [0, 50, 100, 150, 200, 250, 300]:
        # s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='perseus', step_dist=5, test_mode=False, tau_mode="mfp")
        s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e6), rs_dE=20, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='pks', step_dist=2, test_mode=False, tau_mode="mfp", atomdb_version='3.0.9', abundance_table='lpgs', comparison=False)
        s.scatter_generated_photons_div(savefile='simulation_pks_digit.hdf5', plot=False)
        # s.test_mode_plot(filename='simulation_perseus_digit.hdf5')

def main_abell478():
    for v in [0, 50, 100, 150, 200, 250, 300]:
        # s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='perseus', step_dist=5, test_mode=False, tau_mode="mfp")
        s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e6), rs_dE=20, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='abell478', step_dist=2, test_mode=False, tau_mode="mfp", atomdb_version='3.0.9', abundance_table='wilm', comparison=False)
        s.scatter_generated_photons_div(savefile='simulation_abell478.hdf5', plot=False)
        # s.test_mode_plot(filename='simulation_perseus_digit.hdf5')

def main2():
    for v in [150]:
        # s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='perseus', step_dist=5, test_mode=False, tau_mode="mfp")
        s = Simulation(physics_list=['RS'], rmin=0, rmax=825, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e6), rs_dE=20, turbulent_velocity=v, rs_line_list=['w','x','y','z','u','r','t','q','Lya1','Lya2','Heb1','Heb2'], cluster_name='perseus', step_dist=2, test_mode=False, tau_mode="mfp", atomdb_version='3.0.8', abundance_table='AG89', comparison=True)
        s.scatter_generated_photons_div(savefile='simulation_perseus_furukawa.hdf5', plot=False)
        # s.test_mode_plot(filename='simulation_perseus_digit.hdf5')

def tau_plot():
    s = Simulation(physics_list=['RS'], rmin=0, rmax=830, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=20, turbulent_velocity=0, rs_line_list=['w'], cluster_name='perseus', step_dist=1, test_mode=False, tau_mode="effective", atomdb_version='3.0.9', abundance_table='wilm', comparison=False)
    s.tau_v_dependence("black")
    s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=20, turbulent_velocity=0, rs_line_list=['w'], cluster_name='pks', step_dist=1, test_mode=False, tau_mode="effective", atomdb_version='3.0.9', abundance_table='wilm', comparison=False)
    s.tau_v_dependence("red")
    s = Simulation(physics_list=['RS'], rmin=0, rmax=1000, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=20, turbulent_velocity=0, rs_line_list=['w'], cluster_name='abell478', step_dist=1, test_mode=False, tau_mode="effective", atomdb_version='3.0.9', abundance_table='wilm', comparison=False)
    s.tau_v_dependence("blue")
    plt.show()
    plt.savefig('tau_v_dependence.png', dpi=300)

def tau_2d_plot():
    s = Simulation(physics_list=['RS'], rmin=0, rmax=830, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(1e4), rs_dE=20, turbulent_velocity=0, rs_line_list=['w'], cluster_name='pks', step_dist=1, test_mode=False, tau_mode="effective", atomdb_version='3.0.9', abundance_table='lpgs', comparison=False)
    s.tau_v_ni_dependence_2d()

def main_test(): 
    for v in [200] :
        s = Simulation(physics_list=['RS'], rmin=0, rmax=100, division=400, Emin = 6500, Emax = 6800, dn = 1000,  size=int(2e5), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='dummy', step_dist=2, test_mode=False, tau_mode="mfp")
        s.scatter_generated_photons_div(savefile='simulation_dummy_digit.hdf5')
        s.test_mode_plot(filename='simulation_dummy_digit.hdf5')

def main_test2():
    for v in [100] :
        s = Simulation(physics_list=['RS'] ,rmin=0, rmax=99.999, division=10, Emin = 6690, Emax = 6710, dn = 1000,  size=int(1e5), rs_dE=10, turbulent_velocity=v, rs_line_list=['w'], cluster_name='dummy', step_dist=5, test_mode=True, ne_dummy=0.01, tau_mode="effective")
        s.scatter_generated_photons_div(savefile='simulation_pks_digit.hdf5', plot=False)
        #plt.show()
        s.test_mode_plot()
        line_prob_check = np.array(s.line_prob_check)
        random_check = np.array(s.random_check)
        print(line_prob_check)
        print(random_check)
        print(len(line_prob_check))
        print(len(line_prob_check[line_prob_check<random_check[:]]))
        print(len(line_prob_check[line_prob_check>random_check[:]]))
        print(len(line_prob_check[line_prob_check<random_check[:]])/len(line_prob_check))
        print(len(line_prob_check[line_prob_check>random_check[:]])/len(line_prob_check))
        #s.scatter_info()
        #s.plot_simulation_result()

def pierson_chi2_test(num_trials, calculated_result, simulated_event_count):
    '''
    ピアソンのカイ二乗検定を行う関数
    Perform Pearson's chi-squared test for a given simulated event count and calculated result.

    Parameters
    ----------
    num_trials : int
        The number of trials in the simulation.
    calculated_result : float
        The calculated result of the simulation (i.e. the expected number of events).
    simulated_event_count : int
        The number of events that occurred in the simulation.

    Returns
    -------
    p_value : float
        The p-value of the test.

    Notes
    -----
    The chi-squared statistic is calculated as:
    chisq = (simulated_event_count - calculated_result)^2 / calculated_result + (num_trials - simulated_event_count - (num_trials - calculated_result))^2 / (num_trials - calculated_result)
    The p-value is then calculated using scipy.stats.chi2.cdf.
    num_trials: シミュレーション回数
    calculated_result: 計算結果（事象が発生する回数）
    simulated_event_count: 実際にシミュレーションで事象が発生した回数
    '''
    from scipy.stats import chi2
    chisq = (simulated_event_count - calculated_result)**2 / calculated_result + (num_trials - simulated_event_count - (num_trials - calculated_result))**2 / (num_trials - calculated_result)
    p_value = 1 - chi2.cdf(chisq, 1)
    print(f"chi-square: {chisq}")
    print(f"pvalue: {p_value}")
    return p_value

def profile_function(func):
    prof = LineProfiler()
    prof.add_function(func)
    prof.runcall(func)
    prof.print_stats()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <function_name>")
        sys.exit(1)

    func_name = sys.argv[1]
    func = globals().get(func_name)

    if func is None:
        print(f"Function '{func_name}' not found.")
        sys.exit(1)

    prof = LineProfiler()
    prof.add_function(func)
    prof.runcall(func)
    prof.print_stats()


