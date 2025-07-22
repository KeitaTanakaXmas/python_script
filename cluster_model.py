import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
from line_profiler import LineProfiler, profile

def profile_func(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.runcall(func, *args, **kwargs)
        profiler.print_stats()
    return wrapper

class NFW:
    def __init__(self, r_200, consent_param, H0=70):
        self.r_200 = r_200 * u.Mpc
        self.consent_param = consent_param
        self.H0 = H0
        self.rho_c = 3 * (H0 * u.km / u.s / u.Mpc) ** 2 / (8 * np.pi * const.G)
        self.rs = self.r_200 / consent_param  # 修正
        self.delta_c = (200 / 3) * (consent_param ** 3) / (np.log(1 + consent_param) - consent_param / (1 + consent_param))
        self.f_b = 0.16  # バリオン質量分率（通常は約0.16）

    def has_unit_and_is_mpc(self, var):
        if isinstance(var, u.Quantity) and var.unit.is_equivalent(u.Mpc):
            return True
        return False

    def nfw_density(self, r):
        r = np.atleast_1d(r.to(u.Mpc))  # Mpcの単位に変換し、配列に変換
        return (self.delta_c * self.rho_c / ((r / self.rs) * (1 + r / self.rs) ** 2)).to('g/cm^3')

    def nfw_analytical(self, r, r_200, c, H0=70):  # selfを追加
        r = r * u.Mpc
        r_200 = r_200 * u.Mpc
        rs = r_200 / c
        x = r / rs
        return (np.log(1 + x) - x / (1 + x)) / x ** 2

    def nfw_mass(self, rmax):
        if isinstance(rmax, u.Quantity):
            rmax = np.atleast_1d(rmax.to(u.Mpc))  # Mpcの単位に変換し、配列に変換

        # 積分する関数: ρ(r) * r^2
        def integrand(r):
            rho = self.nfw_density(r * u.Mpc)  # NFW密度プロファイル
            return (rho * (r * u.Mpc)**2).to('g * Mpc^2 / cm^3').value

        # rmaxの各要素に対して質量を計算
        masses = []
        for r in rmax:
            mass, _ = integrate.quad(integrand, 0, r.value)  # 0からrまで積分
            total_mass = 4 * np.pi * mass * (u.Mpc.to('cm')**3)  # 質量 [g]
            solar_mass = total_mass / const.M_sun.to('g').value  # 太陽質量に換算
            masses.append(solar_mass)

        return masses if len(masses) > 1 else masses[0]  # 単一の値ならそれを返す

    def plot_style(self, style='double'):
        params = {
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight': 500,
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
        self.fig = plt.figure(figsize=(12, 8))  # 高さを調整して3つのプロットを表示
        spine_width = 2  # スパインの太さ

        self.ax = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax)
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax)  # 3番目の軸

        for ax in [self.ax, self.ax2, self.ax3]:
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis='both', direction='in', width=1.5)

        self.fig.align_labels()


    # 電子密度の計算
    def electron_density(self, r):
        from scipy.integrate import cumtrapz
        """
        静水圧平衡を仮定して電子密度分布を計算
        """
        # 重力ポテンシャルの勾配 dΦ/dr を計算
        def potential_gradient(r):
            r = r.to(u.Mpc)  # 単位をMpcに統一
            x = r / self.rs
            dphi_dr = -4 * np.pi * const.G * self.rho_c * self.delta_c * (
                np.log(1 + x) / x**2 - 1 / (x * (1 + x))
            )
            return dphi_dr.to('m^2/s^2/Mpc')

        # 重力ポテンシャルの勾配を計算
        dphi_dr = potential_gradient(r)
        
        # 静水圧平衡の式を用いて電子密度を計算
        # n_e(r) = n_e(r0) * exp(-μmp/kT * ∫ dΦ/dr dr)
        prefactor = (mu * m_p / (k_B * T_const))  # 係数
        integral = cumtrapz(dphi_dr.value / T_const, r.value, initial=0)  # 積分
        ne_0 = 1e-3  # 任意の基準密度 (cm^-3)
        n_e = ne_0 * np.exp(-prefactor.value * integral)

        return n_e * u.cm**-3  # 単位をつけて返す

    @profile_func
    def plotting(self):
        self.plot_style('triple')  # tripleに変更

        r = np.linspace(5e-3, 0.5, 100) * u.Mpc
        density = self.nfw_density(r)
        self.ax.plot(r * 1e+3, density)

        mass = self.nfw_mass(r)
        self.ax2.plot(r * 1e+3, mass)

        electron_density = self.electron_density(r)  # 電子密度を計算
        self.ax3.plot(r * 1e+3, electron_density, label='Electron Density')

        # スケールとラベルを設定
        self.ax.set_yscale('log')
        self.ax.set_xscale('log')
        self.ax.set_xlabel('r [kpc]')
        self.ax.set_ylabel('ρ(r) [g/cm^3]')
        self.ax.set_title('NFW Density Profile')

        self.ax2.set_yscale('log')
        self.ax2.set_xscale('log')
        self.ax2.set_xlabel('r [kpc]')
        self.ax2.set_ylabel('M(r) [M_sun]')

        self.ax3.set_yscale('log')
        self.ax3.set_xscale('log')
        self.ax3.set_xlabel('r [kpc]')
        self.ax3.set_ylabel('n_e(r) [cm^-3]')
        self.ax3.set_title('Electron Density Profile')
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax3.grid(linestyle='dashed')
        self.ax.set_xlim(5e-3 * 1e+3, 0.5 * 1e+3)
        self.fig.tight_layout()
        plt.show()



    # def electron_density(self, r):
    #     # 電子密度の計算: n_e = f_b * rho / m_e
    #     rho = self.nfw_density(r)
    #     return (self.f_b * rho / const.m_p).to('cm**-3')  # プロトン質量で規格化