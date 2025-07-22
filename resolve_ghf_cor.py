from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess
import glob
import os

"""
This script is used to correct the effective temperature of the pixels in the XRISM/Resolve data.
The effective temperature is corrected by fitting a linear function to the temperature data of each pixel.
The corrected temperature data is then used to create a new FITS file with the corrected temperature data.
"""

plot_params = {#'backend': 'pdf',
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
plt.rcParams.update(plot_params)

def ghf_linear_cor(inputfile='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/xa000112000rsl_000_fe55.ghf.gz', outputfile='xa000112000rsl_000_fe55_linear_cor.ghf', endpoint='header'):

    with fits.open(inputfile, mode='readonly') as hdul:
        data = hdul[1].data  # index 1 のデータ部分
        header1 = hdul[1].header  # index 1 のヘッダー
        header2 = hdul[2].header  # index 2 のヘッダー
        data2 = hdul[2].data  # index 2 のデータ部分
        time = data['TIME']
        pixel = data['PIXEL']
        temp_fit = data['TEMP_FIT']

        original_columns = data.columns.names

    tstart = 153138136
    time_th = 1.533293e9
    #time_th = 1.533293e8
    if endpoint == 'header':
        g_start = header1['TSTART']
    else:
        g_start = endpoint
    interp_time = np.linspace(tstart, g_start, 100)
    interp_temp = []

    for pixel_id in np.unique(pixel):
        mask = (pixel == pixel_id) & (time < time_th)
        temp_pix = temp_fit[mask]
        time_pix = time[mask]

        # NaNとinfを除外
        valid_indices = ~np.isnan(temp_pix) & ~np.isinf(temp_pix) & ~np.isnan(time_pix) & ~np.isinf(time_pix)
        temp_pix = temp_pix[valid_indices]
        time_pix = time_pix[valid_indices]

        if len(temp_pix) > 1:
            try:
                # 線形フィット
                popt, _ = curve_fit(lambda x, a, b: a * x + b, time_pix, temp_pix)
                a_fit, b_fit = popt

                # interp_time における温度を計算
                temp_interp = a_fit * interp_time + b_fit
                interp_temp.extend(temp_interp)

            except RuntimeError:
                print(f"Warning: Fit failed for pixel {pixel_id}")
                interp_temp.extend([np.nan] * len(interp_time))

    # 配列を1次元に変換
    interp_time_repeated = np.tile(interp_time, len(np.unique(pixel)))
    interp_pixel_repeated = np.repeat(np.unique(pixel), len(interp_time))

    # FITSファイルを更新するための新しいデータ

    new_time = np.concatenate((time, interp_time_repeated))
    new_pixel = np.concatenate((pixel, interp_pixel_repeated))
    new_temp_fit = np.concatenate((temp_fit, interp_temp))

    # new_timeの順番でソート
    sorted_indices = np.argsort(new_time)
    new_time = new_time[sorted_indices]
    new_pixel = new_pixel[sorted_indices]
    new_temp_fit = new_temp_fit[sorted_indices]

    # 元のデータから他のカラムを取得し、次元を考慮してNaNデータを作成
    new_columns_data = {}
    for col_name in original_columns:
        if col_name not in ['TIME', 'PIXEL', 'TEMP_FIT']:
            original_col_data = data[col_name]
            
            # カラムの次元数を取得
            if len(original_col_data.shape) == 1:
                # 1次元データの処理
                new_col_data = np.concatenate((original_col_data, np.full(len(interp_time_repeated), np.nan)))
            elif len(original_col_data.shape) == 2:
                # 2次元データの処理（例：行数 x 列数 の形を保持）
                new_shape = (len(interp_time_repeated), original_col_data.shape[1])
                new_col_data = np.concatenate((original_col_data, np.full(new_shape, np.nan)))

            # 追加データを辞書に保存
            new_columns_data[col_name] = new_col_data

    # FITSファイルの新しいカラムリスト
    cols = [
        fits.Column(name='TIME', format='D', array=new_time),
        fits.Column(name='PIXEL', format='I', array=new_pixel),
        fits.Column(name='TEMP_FIT', format='D', array=new_temp_fit)
    ]

    new_hdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(cols), header=header1)
    new_hdu2 = fits.BinTableHDU(data2, header=header2)

    # プライマリHDU（ヘッダーのみの空データ）
    primary_hdu = fits.PrimaryHDU(header=hdul[0].header)

    # FITSファイルを新規作成（プライマリHDU + 更新データ + 2つ目のHDU）
    new_hdul = fits.HDUList([primary_hdu, new_hdu1, new_hdu2])

    new_hdul.writeto(outputfile, overwrite=True)

    print(f"Updated FITS file saved to: {outputfile}")

def plot_cor(file_path='xa000111000rsl_000_fe55.ghf'):

    tstart = 153138136
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        time = (data['TIME']-tstart)*1e-3
        pixel = data['PIXEL']
        temp_fit = data['TEMP_FIT']
        header = hdul[1].header
        g_start = header['TSTART']

    with fits.open('/Volumes/SUNDISK_SSD/Abell2319/repro/55Fe_gh_liner_cor/xa000111000rsl_000_pxcal.ghf.gz') as hdul:
        data_cal = hdul[1].data
        time_cal = (data_cal['TIME']-tstart)*1e-3
        pixel_cal = data_cal['PIXEL']
        temp_fit_cal = data_cal['TEMP_FIT']

    # TSTARTの設定
    tstart = 153138136

    # 対象のピクセルIDリスト
    pixel_ids = np.arange(0, 36, 1)

    # 時間の閾値でフィルタリング
    time_th = 1.533293e8
    time_th = 1.533293e9
    filtered_indices = time < time_th
    time_filtered = time[filtered_indices]
    pixel_filtered = pixel[filtered_indices]
    temp_fit_filtered = temp_fit[filtered_indices]

    # 線形フィット関数の定義
    def linear_func(x, a, b):
        return a * x + b

    # メインプロットと残差プロットの設定
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    for pixel_id in pixel_ids:
        mask = pixel_filtered == pixel_id
        temp_pix = temp_fit_filtered[mask]
        time_pix = time_filtered[mask]

        # NaNとinfを除外
        valid_indices = ~np.isnan(temp_pix) & ~np.isinf(temp_pix) & ~np.isnan(time_pix) & ~np.isinf(time_pix)
        temp_pix = temp_pix[valid_indices]
        time_pix = time_pix[valid_indices]

        if len(temp_pix) > 1:
            try:
                # 線形フィット
                popt, _ = curve_fit(linear_func, time_pix, temp_pix)

                # フィット結果のパラメータ
                a_fit, b_fit = popt

                # TSTART までのデータを予測
                # time_fit = np.linspace(tstart, time_pix.max(), 500)
                time_fit = np.linspace(time_cal.min(), time_cal.max(), 500)
                temp_fit_pred = linear_func(time_fit, a_fit, b_fit)

                # 元データのプロット
                axes[0].scatter(time_pix, temp_pix*1e3, label=f'Pixel {pixel_id}', alpha=1, s=20, color=cm.jet(pixel_id/36))

                # フィット曲線をプロット
                axes[0].plot(time_fit, temp_fit_pred*1e3, linestyle='--',alpha=0.5, label=f'Fit {pixel_id}', color=cm.jet(pixel_id/36))
                axes[0].text(time_fit[-1]+0.3e4, temp_fit_pred[-1]*1e3, f'{pixel_id}', fontsize=10, color=cm.jet(pixel_id/36))

                # 残差の計算とプロット
                temp_pred_actual = linear_func(time_pix, a_fit, b_fit)
                residuals = temp_pix*1e3 - temp_pred_actual*1e3
                axes[1].scatter(time_pix, residuals, alpha=0.3, s=10, label=f'Residual {pixel_id}', color=cm.jet(pixel_id/36))

            except RuntimeError:
                print(f"Warning: Fit failed for pixel {pixel_id}")

    # メインプロット設定
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Temperature (mK)')
    axes[0].set_title('Fitted Effective Temperature for each pixel')
    #axes[0].axvline(tstart, color='red', linestyle='--', label='TSTART')
    #axes[0].axvline(g_start, color='red', linestyle='--', label='TSTART')
    #axes[0].legend()

    # 残差プロット設定
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Residuals (mK)')
    axes[1].set_ylim(-0.005, 0.005)
    #axes[1].legend()
    axes[0].scatter(time_cal, temp_fit_cal*1e3, label='Pixel cal', alpha=1, s=20, color='black')

    plt.tight_layout()

    fig.savefig(f'./figure/ghf_fit.png',dpi=300,transparent=True)
    # file_path1 = './xa000112000rsl_p0px4000_cl.evt'
    # with fits.open(file_path1) as hdul:
    #     data = hdul[1].data
    #     itype = data['ITYPE']
    #     time = data['TIME'][itype == 0]
    #     temp = np.array(data['TEMP'])[itype == 0]
    # axes[0].scatter(time, temp*1e3)
    # plt.tight_layout()
    # fig.savefig(f'./figure/ghf_fit_w_data.png',dpi=300,transparent=True)
    plt.show()

def plot_ghf(file_path='/Volumes/SUNDISK_SSD/PKS_XRISM/ghf_test/xa000112000rsl_000_fe55.ghf.gz'):
    '''
    This function plots the temperature data from a GHF file for each pixel.
    file_path : str
        Path to the GHF file.
    '''
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        time = data['TIME']
        pixel = data['PIXEL']
        temp_fit = data['TEMP_FIT']
        header = hdul[1].header
        tstart = header['TSTART']

    pixel_ids = np.arange(0, 36, 1)
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    for pixel_id in pixel_ids:
        mask = pixel == pixel_id
        temp_pix = np.array(temp_fit[mask])
        time_pix = np.array(time[mask])
        if len(temp_pix) > 0:
            temp_pix = temp_pix[np.argsort(time_pix)]
            time_pix = np.array(sorted(time_pix))
            axes.scatter((time_pix-tstart)*1e-3, temp_pix*1e3, label=f'Pixel {int(pixel_id)}, default process', alpha=1, s=20, color=cm.jet(pixel_id/36))
            axes.plot((time_pix-tstart)*1e-3, temp_pix*1e3, alpha=1, color=cm.jet(pixel_id/36))
            axes.text((time_pix[-1]-tstart)*1e-3+3, temp_pix[-1]*1e3, f'{pixel_id}', fontsize=10, color=cm.jet(pixel_id/36))

    axes.set_xlabel('Time (ksec)')
    axes.set_ylabel('Temperature (mK)')
    plt.tight_layout()
    plt.show()
    fig.savefig(f'ghf_plot.png', dpi=300, transparent=True)

def plot_ghf_cor(file_path1='ghf_before_recycle.ghf', file_path2='ghf_after_recycle.ghf'):
    pix_mask = [0, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 34, 35]
    # TSTARTの設定
    tstart = 153138136
    adr1_time = tstart + 35341
    adr_filter = 0 
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    files = [file_path1, file_path2]
    for e,file_path in enumerate(files):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time_filter = (data['TIME'] > tstart)
            time = (data['TIME'][time_filter] - tstart) * 1e-3
            pixel = data['PIXEL'][time_filter]
            temp_fit = data['TEMP_FIT'][time_filter]
            header = hdul[1].header
            g_start = header['TSTART']


        # 対象のピクセルIDリスト
        pixel_ids = np.arange(0, 36, 1)


        for pixel_id in pixel_ids:
            mask = pixel == pixel_id
            if pixel_id == 12:
                pass
            elif pixel_id in pix_mask:
                temp_pix = temp_fit[mask]
                time_pix = time[mask]

                axes.plot(time_pix, temp_pix*1e3, label=f'Pixel {pixel_id}', alpha=0.5, color=cm.jet(pixel_id/36))
                axes.scatter(time_pix, temp_pix*1e3, alpha=0.5, s=20, color=cm.jet(pixel_id/36))

                #axes.text(time_pix[-1]+0.3e4, temp_pix[-1]*1e3, f'{pixel_id}', fontsize=10, color=cm.jet(pixel_id/36))




    # メインプロット設定
    axes.set_xlabel('Time (ks)')
    axes.set_ylabel('Temperature (mK)')
    plt.tight_layout()
    axes.axvspan(94140e-3, 96540e-3, color='gray', alpha=0.1)
    axes.set_xlim(0,150)
    axes.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=5)
    #fig.tight_layout()
    plt.subplots_adjust(right=0.75, left=0.15)
    plt.show()
    fig.savefig(f'ghf_cor_plot.png', dpi=300, transparent=True)

def cor_pi():
    file_path1 = './xa000103000rsl_p0px5000_uf_rslpha2pi.evt'
    file_path2 = './xa000103000rsl_p0px5000_uf.evt.gz'

    col = ['black', 'red']
    lab = ['ghf liner cor', 'ghf no cor']

        # メインプロットと残差プロットの設定
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    for e,file_path in enumerate([file_path1, file_path2]):
        # FITSファイルを開く
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            itype = data['ITYPE']
            pi = np.array(data['PI'])[itype == 0]/2000
            pi = pi[pi < 8]
            pi = pi[pi > 2]



        # メインプロット設定

        axes.hist(pi, bins=60000, histtype='step', color=col[e], label=lab[e])
        axes.set_xlim(5,7)
        axes.legend()


        plt.tight_layout()




    plt.show()

def check_evt_temp():
    file_path = glob.glob('./*.evt')
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    for file in file_path:
        with fits.open(file) as hdul:
            data = hdul[1].data
            itype = data['ITYPE']
            time = data['TIME'][itype == 0]
            temp = np.array(data['TEMP'])[itype == 0]
        axes.scatter(time, temp*1e3, label=file)
    axes.legend()
    plt.tight_layout()
    fig.savefig(f'./figure/evt_temp.png',dpi=300,transparent=True)
    plt.show()


def make_ghf_from_scott(scottfile='/Users/keitatanaka/Dropbox/xrism_inf/PKS_sub_items/scott_ghf/Gain_history/*.txt', 
                         outputfile='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/xa000112000rsl_000_cra.ghf', 
                         rootghf='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_000_cra.ghf'):
    '''
    This function creates a new GHF file based on Scott's text data.
    
    Parameters:
    -----------
    scottfile : str
        Path to Scott's GHF text data.
    outputfile : str
        Path to save the output GHF file.
    rootghf : str
        Path to the root GHF file (typically CrKa GHF).

    Returns:
    --------
    None
    '''
    # Load Scott's data
    files = glob.glob(scottfile)
    times, temps, pixs = np.array([]), np.array([]), np.array([])

    for file in files:
        data = np.loadtxt(file, dtype=float)
        temp = data[:, 2]
        
        # Filter out temperature values that are too small
        temp_mask = temp > 0.01
        time = data[:, 0][temp_mask]
        pix = data[:, 1][temp_mask]
        temp = temp[temp_mask]

        times = np.concatenate((times, time))
        temps = np.concatenate((temps, temp))
        pixs = np.concatenate((pixs, pix))

    # Read the root GHF file (CrKa)
    with fits.open(rootghf, mode='readonly') as hdul:
        data = hdul[1].data  # Data from HDU 1
        header1 = hdul[1].header  # Header from HDU 1
        header2 = hdul[2].header  # Header from HDU 2
        data2 = hdul[2].data  # Data from HDU 2

        # Extract existing data
        time = data['TIME']
        pixel = data['PIXEL']
        temp_fit = data['TEMP_FIT']

    # Sort new data by time
    sorted_indices = np.argsort(times)
    new_time = times[sorted_indices]
    new_pixel = pixs[sorted_indices]
    new_temp_fit = temps[sorted_indices]

    # Create new columns for the updated FITS file
    cols = [
        fits.Column(name='TIME', format='D', array=new_time),
        fits.Column(name='PIXEL', format='I', array=new_pixel),
        fits.Column(name='TEMP_FIT', format='D', array=new_temp_fit)
    ]

    # Create new HDUs
    new_hdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(cols), header=header1)
    new_hdu2 = fits.BinTableHDU(data2, header=header2)

    # Create primary HDU (empty data with header only)
    primary_hdu = fits.PrimaryHDU(header=hdul[0].header)

    # Create HDU list and write the new FITS file
    new_hdul = fits.HDUList([primary_hdu, new_hdu1, new_hdu2])
    new_hdul.writeto(outputfile, overwrite=True)

    print(f"Updated FITS file saved to: {outputfile}")


def adapt_ghf_rslpha2pi(infile='xa000112000rsl_p0px4000_cl.evt.gz', outfile='rslpha2pi_4000.out', ghffile='xa000112000rsl_000_fe55_linear_cor.ghf'):
    caldb = os.getenv('CALDB')
    print('CALDB:', caldb)
    gainfile = f'{caldb}/data/xrism/resolve/bcf/gain/xa_rsl_gainpix_20190101v006.fits'
    command = f'rslpha2pi infile={infile} outfile={outfile} calcpi=yes calcupi=yes driftfile={ghffile} gainfile={gainfile}  tempidx=-1 pxphaoffset=0 secphacol=PHA addepicol=EPI2 method=FIT itypecol=ITYPE binwidth=0.5 offset=0.5 tlmax=59999 gapdt=-1 ntemp=3 writetemp=yes extrap=no randomize=yes seed=7 buffer=-1 clobber=no chatter=2 debug=no history=yes mode=hl'
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)

def adapt_ghf_rslpha2pi_each_fw(identifiers=['1000', '2000', '3000', '4000']):
    for identifier in identifiers:
        infile = f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/xa000112000rsl_p0px{identifier}_cl.evt'
        outfile = f'./xa000112000rsl_p0px{identifier}_cl.evt'
        ghffile = f'./xa000112000rsl_000_cra.ghf'
        adapt_ghf_rslpha2pi(infile=infile, outfile=outfile, ghffile=ghffile)