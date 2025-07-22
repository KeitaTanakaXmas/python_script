#!/usr/bin/env python

import os, sys, glob, datetime
import numpy as np
from scipy import interpolate
import pandas as pd
from astropy.io import fits

import pyfits_addons

def assign_mxsphase(infile, outfile, time2ti, time_led_on, l32ti_led_on, t_spc, spwtick=1/64, ti_spw_delay=1):

    newhdulist = fits.HDUList()
    hdulist = fits.open(infile)

    ti_spc = np.int64(t_spc / spwtick)

    for hdu in hdulist:

        newhdu = hdu

        if hdu.name == 'EVENTS':

            # back projection of event TIME into L32TI space
            l32ti = time2ti(hdu.data['TIME'])

            # to handle l32ti wrap around
            # this assumes that l32ti never runs one complete cycle in one observation
            # otherwise the following condition is not sufficient to catch all the wrap around cases
            if l32ti[0] > l32ti[-1]:

                if l32ti[0] < l32ti_led_on:

                    # the case where LED-on happened before the wrap around of l32ti
                    # increase l32ti values after the wrap around by 2**32
                    l32ti = np.where(l32ti < l32ti[0], l32ti + 2 ** 32, l32ti)

                else:
                    # the other case where LED-on happened after the wrap around of l32ti
                    # actually this case does not matter as MXS phase only needs to be assigned after LED-on
                    # thus no MXS pulse-on durations should be there before the wrap around
                    # but anyway, just for symmetry
                    l32ti = np.where(l32ti < l32ti[0], l32ti, l32ti - 2 ** 32)
                #
            else:

                pass

            #

            time = hdu.data['TIME']

            # psuedo-TIME, which is equal to TIME at time_led_on but whose interval is strictly proportional to TI
            ptime = time_led_on + (l32ti - l32ti_led_on) * spwtick
            # note that this should be the same as hdu.data['TIME'] for the GPS-sync mode

            # MXS phase using TI-based time axis (ptime)
            #mxsphase = (ptime - time_led_on - ti_spw_delay * spwtick) % t_spc
            #mxscycle = np.int64((ptime - time_led_on - ti_spw_delay * spwtick) // t_spc)

            # equivalently,
            mxsphase = (l32ti - l32ti_led_on - ti_spw_delay) % ti_spc * spwtick
            mxscycle = np.int64((l32ti - l32ti_led_on - ti_spw_delay) // ti_spc)

            # MXS phase using TIME-based time axis
            mxsphase2 = (time - time_led_on - ti_spw_delay * spwtick) % t_spc
            mxscycle2 = np.int64((time - time_led_on - ti_spw_delay * spwtick) // t_spc)
            # again, in the GPS-sync mode, mxsphase should match mxsphase2

            coldef_l32ti     = fits.Column(name='PSUEDO_L32TI', format='1J', array=l32ti)
            coldef_ptime     = fits.Column(name='PSUEDO_TIME', format='1D', array=ptime, unit='s')
            coldef_mxsphase  = fits.Column(name='MXS_PHASE', format='1D', array=mxsphase, unit='s')
            coldef_mxscycle  = fits.Column(name='MXS_CYCLE', format='1J', array=mxscycle)
            coldef_mxsphase2 = fits.Column(name='MXS_PHASE2', format='1D', array=mxsphase2, unit='s')
            coldef_mxscycle2 = fits.Column(name='MXS_CYCLE2', format='1J', array=mxscycle2)
            coldefs = [coldef_l32ti, coldef_ptime, coldef_mxsphase, coldef_mxscycle, coldef_mxsphase2, coldef_mxscycle2]
            newhdu = pyfits_addons.insert_columns(newhdu, hdu.columns.names[-1], 'after', coldefs)

        else:
            pass
        #
        newhdulist.append(newhdu)
    #
    newhdulist.writeto(outfile, overwrite=True)
    newhdulist.close()
    hdulist.close()

    return(hdu)

def update_header(outfile, newkeydict):

    comm = 'assign_mxsphase'

    unzipped = outfile.replace('.gz', '')

    os.system('gunzip %s' % outfile)

    for key, value in newkeydict.items():

        print('run fparkey to append %s keyword' % key)
        cmd = 'fparkey %s %s[1] %s comm=%s add=yes' % (value, unzipped, key, comm)
        os.system(cmd)
    #

    os.system('gzip %s' % unzipped)

    return()

def main(infile, outfile, hkfile, timfile, tmargin=10., ledid=1, overwrite=True):

    # MXS phase should be assigned in TI-based time coordinate rather than TIME-based one
    # The first step is to assign TI-based time to individual events
    # To do so, we first define the origin of TIME at TSTART
    px = fits.open(infile)['EVENTS']
    tstart = px.header['TSTART']

    # To find TI_LED_ON relevant to the current data, look-up the value in HK
    # just in case, add a margin w.r.t. tstart, not to pick a value
    # for the previous LED on
    t0 = tstart + tmargin

    fw = fits.open(hkfile)['HK_SXS_FWE']
    fw_overlap_px = (fw.data['TIME'] > t0)

    # a few values to be added to the header
    t_spc_ms = fw.data['FWE_LED%d_PLS_SPC_CAL' % ledid][fw_overlap_px][0]
    t_len_ms = fw.data['FWE_LED%d_PLS_LEN_CAL' % ledid][fw_overlap_px][0]
    i_set    = fw.data['FWE_I_LED%d_SET' % ledid][fw_overlap_px][0]
    i_set_ma = fw.data['FWE_I_LED%d_SET_CAL' % ledid][fw_overlap_px][0]

    t_spc = t_spc_ms / 1e3

    # get TI at LED on and convert it to L32TI
    ti_led_on = fw.data['FWE_TI_LED%d_ON' % ledid][fw_overlap_px][0]
    l32ti_led_on = ti_led_on & (2**32-1)

    # use timfile to find corresponding TIME
    tim = fits.open(timfile)['TIM_LOOKUP']
    timtime = tim.data['TIME']
    timl32ti = tim.data['L32TI']

    print(os.path.basename(__file__))

    print('Input file: %s' % infile)
    print('Output file: %s' % outfile)
    print('HK1 file: %s' % hkfile)
    print('TIM file: %s' % timfile)
    print('FWE_TI_LED%d_ON: %s' % (ledid, ti_led_on))
    print('L32TI at LED%d ON: %s' % (ledid, l32ti_led_on))
    print('FWE_LED%d_PLS_SPC_CAL: %s (ms)' % (ledid, t_spc_ms))
    print('FWE_LED%d_PLS_LEN_CAL: %s (ms)' % (ledid, t_len_ms))
    print('FWE_I_LED%d_SET_CAL: %s (mA)' % (ledid, i_set_ma))

    # forward and reverse conversions between L32TI and TIME in TIM_LOOKUP for LED on
    ti2time = interpolate.interp1d(timl32ti, timtime, fill_value='extrapolate')
    time2ti = interpolate.interp1d(timtime, timl32ti, fill_value='extrapolate')

    time_led_on = float(ti2time(l32ti_led_on))
    timestamp_led_on = datetime.datetime(2019, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=time_led_on)
    date_led_on = timestamp_led_on.strftime('%Y-%m-%dT%H:%M:%S')

    print('L32TI at LED on      : %d' % l32ti_led_on)
    print('L32TI coverage by TIM: %d -- %d' % (timl32ti.min(), timl32ti.max()))
    print('TIME  coverage by TIM: %d -- %d' % (timtime.min(), timtime.max()))
    print('Timestamp at LED on  : %s' % date_led_on)

    newkeydict = {
    'TI_LEDON':ti_led_on,
    'TIME_LEDON':time_led_on,
    'DATE_LEDON':date_led_on,
    'I_SET':i_set,
    'T_SPC':int(round(t_spc_ms / 15.625)),
    'T_LEN':int(round(t_len_ms / 0.125)),
    'I_SET_mA':i_set_ma,
    'T_SPC_ms':t_spc_ms,
    'T_LEN_ms':t_len_ms,
    }

    if not overwrite and os.path.exists(outfile):
        print('%s already exists. Skipped.' % outfile)
    else:
        assign_mxsphase(infile, outfile, time2ti, time_led_on, l32ti_led_on, t_spc)
        update_header(outfile, newkeydict)
    #

    print('')

    return()

if __name__ == '__main__':

    try:
        infile, outfile, hkfile, timfile = sys.argv[1:5]
    except:
        print('%s infile outfile hkfile timfile' % (os.path.basename(__file__)))
        exit()
    #
    main(infile, outfile, hkfile, timfile)