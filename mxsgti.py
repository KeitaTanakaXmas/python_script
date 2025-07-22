import os
from astropy.io import fits 
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.table import Table, Column
import datetime
import numpy as np
import pandas as pd

def make_finegti(outfile, hkfile, timfile, led_thresh_ms, tmargin=10., spwtick=1/64, ti_spw_delay=1, ledid=1, overwrite=True):
    """
    HK fileのMXS LEDパラメータに基づいて、fine GTIを作成する
    
    :param outfile: 出力fine gtiファイルのパス
    :param hkfile:  HKファイルのパス
    :param timfile: TIMファイルのパス, L32TIとTIMEの変換に使用する 
    :param led_thresh_ms: LEDのしきい値 (ミリ秒単位), LEDの照射ONからこの閾値時間までは
    :param tmargin: 時間の余白 (デフォルトは10秒)
    :param spwtick: スポットワークティック (デフォルトは1/64)
    :param ti_spw_delay: TI遅延 (デフォルトは1)
    :param ledid: LED識別番号 (デフォルトは1)
    :param overwrite: 上書き設定 (デフォルトはTrue)
    """
    
    # hkfileを読み込み
    fw = fits.open(hkfile)['HK_SXS_FWE']
    mxs_time = fw.data['TIME']
    t_spc_ms = fw.data['FWE_LED%d_PLS_SPC_CAL' % ledid]
    t_len_ms = fw.data['FWE_LED%d_PLS_LEN_CAL' % ledid]
    i_set = fw.data['FWE_I_LED%d_SET' % ledid]
    i_set_ma = fw.data['FWE_I_LED%d_SET_CAL' % ledid]
    t_spc = t_spc_ms / 1e3

    # LED ON/OFF時のTI取得とL32TIへの変換
    ti_led_on = fw.data['FWE_TI_LED%d_ON' % ledid]
    ti_led_off = fw.data['FWE_TI_LED%d_OFF' % ledid]
    l32ti_led_on = ti_led_on & (2**32-1)
    l32ti_led_off = ti_led_off & (2**32-1)

    # timfileを使ってTIの対応するTIMEを取得
    tim = fits.open(timfile)['TIM_LOOKUP']
    timtime = tim.data['TIME']
    timl32ti = tim.data['L32TI']

    # L32TIとTIME間の補間関数を作成
    ti2time = interpolate.interp1d(timl32ti, timtime, fill_value='extrapolate')
    time2ti = interpolate.interp1d(timtime, timl32ti, fill_value='extrapolate')

    # LED ON/OFFの時刻を計算
    time_led_on  = np.unique(ti2time(l32ti_led_on))
    time_led_off = np.unique(ti2time(l32ti_led_off))
    time_led_off = np.append(time_led_off[1:], time_led_off[-1] + 1e6)

    # タイムスタンプを計算
    timestamp_led_on  = datetime.datetime(2019, 1, 1, 0, 0, 0) + np.array([datetime.timedelta(seconds=t) for t in time_led_on])
    timestamp_led_off = datetime.datetime(2019, 1, 1, 0, 0, 0) + np.array([datetime.timedelta(seconds=t) for t in time_led_off])

    # マスクインデックスの作成
    mxs_time_int = mxs_time.astype(np.int64)
    time_led_on_int = time_led_on.astype(np.int64)
    time_led_off_int = time_led_off.astype(np.int64)
    mask_index = np.where(np.isin(mxs_time_int, time_led_on_int))[0] + int(tmargin)  # 10sの余白

    # パラメータの抽出と挿入
    spc_ms = np.insert(t_spc_ms[mask_index], 0, t_spc_ms[0])
    len_ms = np.insert(t_len_ms[mask_index], 0, t_len_ms[0])
    i_set = np.insert(i_set[mask_index], 0, i_set[0])
    i_set_ma = np.insert(i_set_ma[mask_index], 0, i_set_ma[0])

    # DataFrame作成
    seq_num = np.arange(1, len(time_led_on) + 1)
    df = pd.DataFrame({
        'timestamp_led_on': timestamp_led_on,
        'timestamp_led_off': timestamp_led_off,
        'time_led_on': time_led_on,
        'spc_ms': spc_ms,
        'len_ms': len_ms,
        'i_set': i_set,
        'i_set_ma': i_set_ma,
        'seq_num': seq_num,
    })
    df['timestamp_led_on'] = df['timestamp_led_on'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(df.to_string(index=False))

    led_thres_s = led_thresh_ms / 1e3
    spc_s = spc_ms / 1e3

    # GTI作成
    for seq in range(len(time_led_on)):
        print(f'Creating fine GTI for sequence {seq+1}...')
        cycle_number = int((time_led_off[seq] - time_led_on[seq]) // spc_s[seq])
        cycle_array = np.arange(cycle_number)
        
        print(f"  LED ON: {timestamp_led_on[seq]}")
        print(f"  LED OFF: {timestamp_led_off[seq]}")
        print(f"  Cycle number: {cycle_number}")

        gti_start = (time_led_on[seq] + ti_spw_delay * spwtick + cycle_array * spc_s[seq] + led_thres_s)[:-1]
        gti_stop  = (time_led_on[seq] + ti_spw_delay * spwtick + (cycle_array + 1) * spc_s[seq])[:-1]

        if any(gti_stop > time_led_off[seq]):
            print('  WARNING: GTI stop exceeds LED OFF time')

        if seq == 0:
            create_gti_column(outfile, gti_start, gti_stop, overwrite=False)
        else:
            create_gti_column(outfile, gti_start, gti_stop, overwrite=True)


def create_gti_column(fits_file, start_times, stop_times, overwrite=False):
    """
    FITSファイルにGTIインデックスを追加または更新する関数
    
    :param fits_file: 出力するFITSファイル
    :param start_times: GTI開始時刻のリスト
    :param stop_times: GTI終了時刻のリスト
    :param overwrite: True の場合、既存のファイルに追加。False の場合、元のファイルを削除して新しいファイルを作成。
    """
    # GTIデータを作成
    gti_data = np.array(list(zip(start_times, stop_times)), dtype=[('START', 'f8'), ('STOP', 'f8')])

    if overwrite == False:
        if os.path.exists(fits_file):
            print(f"Remove existing FITS file: {fits_file}")
            os.remove(fits_file)  # ファイルが存在すれば削除
        print(f"Creating a new FITS file: {fits_file}")
        primary_hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([primary_hdu])
        gti_hdu = fits.BinTableHDU(gti_data, name='GTI')
        hdul.append(gti_hdu)
        hdul.writeto(fits_file)
    
    else:
        print(f"Appending to existing FITS file: {fits_file}")
        hdul = fits.open(fits_file, mode='update')
        
        # 既存のGTI拡張があるかを確認
        if 'GTI' in hdul:
            existing_gti_data = hdul['GTI'].data
            new_gti_data = np.append(existing_gti_data, gti_data)
            hdul['GTI'].data = new_gti_data  # 新しいデータで更新
        else:
            gti_hdu = fits.BinTableHDU(gti_data, name='GTI')
            hdul.append(gti_hdu)
        
        hdul.flush()  # 更新を保存
        hdul.close()

    print("GTI column update complete.")
       

def add_mxs_phase(infile, outfile, hkfile, timfile, tmargin=10., spwtick=1/64, ti_spw_delay=1, ledid=1):
    """
    HK fileのMXS LEDパラメータに基づいて、event fileにMXS phaseを追加する
    
    :param outfile: 出力fine gtiファイルのパス
    :param hkfile:  HKファイルのパス
    :param timfile: TIMファイルのパス, L32TIとTIMEの変換に使用する 
    :param tmargin: 時間の余白 (デフォルトは10秒)
    :param spwtick: スポットワークティック (デフォルトは1/64)
    :param ti_spw_delay: TI遅延 (デフォルトは1)
    :param ledid: LED識別番号 (デフォルトは1)
    """
    
    # hkfileを読み込み
    fw = fits.open(hkfile)['HK_SXS_FWE']
    mxs_time = fw.data['TIME']
    t_spc_ms = fw.data['FWE_LED%d_PLS_SPC_CAL' % ledid]
    t_len_ms = fw.data['FWE_LED%d_PLS_LEN_CAL' % ledid]
    i_set = fw.data['FWE_I_LED%d_SET' % ledid]
    i_set_ma = fw.data['FWE_I_LED%d_SET_CAL' % ledid]
    t_spc = t_spc_ms / 1e3

    # LED ON/OFF時のTI取得とL32TIへの変換
    ti_led_on = fw.data['FWE_TI_LED%d_ON' % ledid]
    ti_led_off = fw.data['FWE_TI_LED%d_OFF' % ledid]
    l32ti_led_on = ti_led_on & (2**32-1)
    l32ti_led_off = ti_led_off & (2**32-1)

    # timfileを使ってTIの対応するTIMEを取得
    tim = fits.open(timfile)['TIM_LOOKUP']
    timtime = tim.data['TIME']
    timl32ti = tim.data['L32TI']

    # L32TIとTIME間の補間関数を作成
    ti2time = interpolate.interp1d(timl32ti, timtime, fill_value='extrapolate')
    time2ti = interpolate.interp1d(timtime, timl32ti, fill_value='extrapolate')

    # LED ON/OFFの時刻を計算
    time_led_on  = np.unique(ti2time(l32ti_led_on))
    time_led_off = np.unique(ti2time(l32ti_led_off))
    time_led_off = np.append(time_led_off[1:], time_led_off[-1] + 1e6)

    # タイムスタンプを計算
    timestamp_led_on  = datetime.datetime(2019, 1, 1, 0, 0, 0) + np.array([datetime.timedelta(seconds=t) for t in time_led_on])
    timestamp_led_off = datetime.datetime(2019, 1, 1, 0, 0, 0) + np.array([datetime.timedelta(seconds=t) for t in time_led_off])

    # マスクインデックスの作成
    mxs_time_int = mxs_time.astype(np.int64)
    time_led_on_int = time_led_on.astype(np.int64)
    time_led_off_int = time_led_off.astype(np.int64)
    mask_index = np.where(np.isin(mxs_time_int, time_led_on_int))[0] + int(tmargin)  # 10sの余白

    # パラメータの抽出と挿入
    spc_ms = np.insert(t_spc_ms[mask_index], 0, t_spc_ms[0])
    len_ms = np.insert(t_len_ms[mask_index], 0, t_len_ms[0])
    i_set = np.insert(i_set[mask_index], 0, i_set[0])
    i_set_ma = np.insert(i_set_ma[mask_index], 0, i_set_ma[0])

    # DataFrame作成
    seq_num = np.arange(1, len(time_led_on) + 1)
    df = pd.DataFrame({
        'timestamp_led_on': timestamp_led_on,
        'timestamp_led_off': timestamp_led_off,
        'time_led_on': time_led_on,
        'spc_ms': spc_ms,
        'len_ms': len_ms,
        'i_set': i_set,
        'i_set_ma': i_set_ma,
        'seq_num': seq_num,
    })
    df['timestamp_led_on'] = df['timestamp_led_on'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(df.to_string(index=False))

    spc_s = spc_ms / 1e3

    evt = fits.open(infile)['EVENTS']
    evt_time = evt.data['TIME']
    evt_seq = []

    # search for sequence number
    for i in range(len(evt_time)):
        matched = False  # フラグを追加
        for j in range(len(time_led_on)):
            if evt_time[i] >= time_led_on[j] and evt_time[i] < time_led_off[j]:
                evt_seq.append(j)
                matched = True
                break
        if not matched:
            evt_seq.append(-1)  # 条件に合わない場合は -1 を追加

    evt_seq = np.array(evt_seq)
    mxs_phase = np.full_like(evt_time, -1.0)  # 初期値を -1 で設定

    # evt_seqが-1でない場合のみ計算を行う
    valid_indices = evt_seq != -1  # evt_seqが-1でないインデックスを取得
    mxs_phase[valid_indices] = (evt_time[valid_indices] - time_led_on[evt_seq[valid_indices]] - ti_spw_delay * spwtick) % spc_s[evt_seq[valid_indices]]

    from pyfits_addons import add_new_columns


    add_new_columns(infile, outfile, 1, 'MXSPHASE', mxs_phase, '1D')
    return mxs_phase, evt_time

