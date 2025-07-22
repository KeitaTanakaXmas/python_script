#!/usr/bin/env python

import sys, time, glob, warnings
from numpy import *
from scipy.io import loadmat
import astropy.io.fits as pf

if __name__ == '__main__':
    
    import sys, getopt
    
    opts, params = getopt.getopt(sys.argv[1:], 'h?n:c:r:b:t:a:', ['help', 'reverse'])
    
    # Default values
    ch = 1
    dsr = 1
    rebin = 1
    tlevel = None
    reverse = False
    ab = 'A'
    
    
    def usage():
        print("""Usage: {0} [options] (output filename) (mat converted psdata folders)

        Options:
            -c ch
                Channel number (Default: {1})
            -r rate
                Decimation rate (Default: {2})
            -b bins
                Rebin (Default: {3})
            -t level
                Apply after trigger (negative edge) (Default: None)
            --reverse
                Use positive edge for after trigger
            -h
                Show this usage.
            -a
                Channel 'A' or 'B' (Default: {4})
            """.format(sys.argv[0], ch, dsr, rebin, ab))
    
    for o, a in opts:
        if o == '-c':
            ch = int(a)
        if o == '-r':
            dsr = int(a)
        if o == '-b':
            rebin = int(a)
        if o == '-t':
            tlevel = eval(a)
        if o == '--reverse':
            reverse = True
        if o in ("-h", "-?", "--help"):
            usage()
            exit()
        if o in '-a':
            ab = a

    if len(params) < 2:
        usage()
        exit()

    ofile = params[0]
    idirs = params[1:]
    # Prepare variables
    w = []

    # Open input directory and search mat files
    for dc, idir in enumerate(idirs):
        ifile = glob.glob(idir + '/*.mat')
        ifile.sort()
    
        if len(ifile) == 0:
            print("No matlab file found")
            exit()
    
        for fc, f in enumerate(ifile):
            print("{0} ({1}/{2}): {3} ({4}/{5})\r".format(idir, dc+1, len(idirs), f.split('/')[-1], fc+1, len(ifile)),)
            sys.stdout.flush()
            dat = loadmat(f)
            #hres, l, w = dat['Tinterval'][0][0], dat['Length'][0][0], dat['B'].T[ch-1]
            hres, l, w, rl = dat['Tinterval'][0][0], dat['Length'][0][0], dat[ab].T[ch-1], dat['RequestedLength']
            if isinf(w).any():
                continue
            
            _w = w[:int(rl)]

            # Decimation (this does not perform averaging intentionally)
            if dsr > 1:
                _w = _w[::dsr]

            # Rebin
            if rebin > 1:
                #_p = _p[:(_p.shape[-1]//rebin)*rebin].reshape(_p.shape[0], -1, rebin).mean(axis=-1)
                #_n = _n[:(_n.shape[-1]//rebin)*rebin].reshape(_n.shape[0], -1, rebin).mean(axis=-1)
                # old one is too long and has mistake in reshape()
                _w = _w[:(_w.shape[-1]//rebin)*rebin]
                _w = _w.reshape(_w.shape[-1]//rebin, 1, rebin).mean(axis=-1)

            # After trigger
            if tlevel is not None:
                if reverse:
                    if _w.max() > tlevel:
                        w.append(_w)
                else:
                    if _w.min() < tlevel:
                        w.append(_w)
            else:
                w.append(_w)

            
        #print

    print("Generating fits files...")

    w = asarray(w)  #.reshape(fc+1, -1)
        
    vres = max(abs(p).max(), abs(n).max())/2.**15
    p = vectorize(int)(p/vres)
    n = vectorize(int)(n/vres)
    
    hres *= dsr
    hres *= rebin

    for data, colname, filename in zip((p, n), ('PulseRec', 'NoiseRec'), (ofile + 'p.fits', ofile + 'n.fits')):
    
        # Columns
        col_t = pf.Column(name='TIME', format='1D', unit='s', array=zeros(data.shape[0], dtype=int))
        col_data = pf.Column(name=colname, format='%dI' % data.shape[1], unit='V', array=data)

        cols = pf.ColDefs([col_t, col_data])
        tbhdu = pf.BinTableHDU.from_columns(cols)

        # Name of extension
        exthdr = tbhdu.header
        exthdr['EXTNAME'] = ('Record', 'name of this binary table extension')
        exthdr['EXTVER'] = (1, 'extension version number')

        # Add more attributes
        exthdr['TSCAL2'] = (vres, '[V/ch]')
        exthdr['TZERO2'] = (0, '[V]')
        exthdr['THSCL2'] = (hres, '[s/bin] horizontal resolution of record')
        exthdr['THZER2'] = (0, '[s] horizontal offset of record')
        exthdr['THSAM2'] = (data.shape[1], 'sample number of record')
        exthdr['THUNI2'] = ('s', 'physical unit of sampling step of record')
        exthdr['TRMIN2'] = (-2**15, '[channel] minimum number of each sample')
        exthdr['TRMAX2'] = (2**15, '[channel] maximum number of each sample')
        exthdr['TRBIN2'] = (1, '[channel] default bin number of each sample')

        # More attributes
        exthdr['TSTART'] = (0, 'start time of experiment in total second')
        exthdr['TSTOP'] = (0, 'end time of experiment in total second')
        exthdr['TEND'] = (0, 'end time of experiment (obsolete)')
        exthdr['DATE'] = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')

        # We anyway need Primary HDU
        hdu = pf.PrimaryHDU()

        # Write to FITS
        thdulist = pf.HDUList([hdu, tbhdu])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            thdulist.writeto(filename)
