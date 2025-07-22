#!/usr/bin/env python

import sys, time, glob, warnings
from numpy import *
from scipy.io import loadmat
import astropy.io.fits as pf
import h5py

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
    p = []
    n = []

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
            
            w = w[:int(rl)]

            _p = w[-int(w.shape[-1]/2):]
            _n = w[:int(w.shape[-1]/2)]

            # Decimation (this does not perform averaging intentionally)
            if dsr > 1:
                _p = _p[::dsr]
                _n = _n[::dsr]

            # Rebin
            if rebin > 1:
                #_p = _p[:(_p.shape[-1]//rebin)*rebin].reshape(_p.shape[0], -1, rebin).mean(axis=-1)
                #_n = _n[:(_n.shape[-1]//rebin)*rebin].reshape(_n.shape[0], -1, rebin).mean(axis=-1)
                # old one is too long and has mistake in reshape()
                _p = _p[:(_p.shape[-1]//rebin)*rebin]
                _p = _p.reshape(_p.shape[-1]//rebin, 1, rebin).mean(axis=-1)
                _n = _n[:(_n.shape[-1]//rebin)*rebin]
                _n = _n.reshape(_n.shape[-1]//rebin, 1, rebin).mean(axis=-1)

            # After trigger
            if tlevel is not None:
                if reverse:
                    if _p.max() > tlevel:
                        p.append(_p)
                else:
                    if _p.min() < tlevel:
                        p.append(_p)
            else:
                p.append(_p)

            n.append(_n)
            
        #print

    print("Generating hdf5 files...")

    p = asarray(p)  #.reshape(fc+1, -1)
    n = asarray(n)  #.reshape(fc+1, -1)
        
    vres = max(abs(p).max(), abs(n).max())/2.**15
    p = vectorize(int)(p/vres)
    n = vectorize(int)(n/vres)
    
    hres *= dsr
    hres *= rebin

    with h5py.File(ofile,"a") as f:
        f.create_dataset('pulse',data=p)
        f.create_dataset('noise',data=n)