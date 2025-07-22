from pytes import Util,Analysis,Filter
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import astropy.io.fits as pf
import time, struct
import re
from xspec import *
from scipy.stats import chi2
import types
import subprocess

dataset = "1"
afn = "data1_ThLine.hdf5"
ch2 = "data_set_"+ str(dataset) +"-CH2.hdf5"
ch3 = "data_set_"+ str(dataset) +"-CH3.hdf5"
ch4 = "data_set_"+ str(dataset) +"-CH4.hdf5"

def make_Th():
    dir = os.curdir
    fall = [path for path in os.listdir(dir) if path.endswith(".hdf5")]
    fl = sorted(fall)
    for fn in fl:
        with h5py.File(fn) as f:
            i = 1
            if i == 1:
                d = f["energy/gen"][:]
            else :
                d = np.append(f[d,f["energy/gen"][:]])
            i = i + 1
    print(d)
    with h5py.File(afn) as f:
        if "ThLine" in f.keys():
            del f["ThLine"]
        f.create_dataset("ThLine",data=d)

def make_Th_sin(data=1,ch=2):
    if ch == 2:
        dn = "data_set_"+ str(data) +"-CH2.hdf5"
    if ch == 3:
        dn = "data_set_"+ str(data) +"-CH3.hdf5"
    if ch == 4:
        dn = "data_set_"+ str(data) +"-CH4.hdf5"
    afn="data1_CH"+str(ch)+"_ThLine.hdf5"
    fn = dn
    with h5py.File(fn) as f:
        d = f["energy/gen"][:]
        print(d)
    with h5py.File(afn) as f:
        if "ThLine" in f.keys():
            del f["ThLine"]
        f.create_dataset("ThLine",data=d)
        
def make_resp(rmfname='test.rmf', bmin=1000., bmax=30000.):
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

def histogram(pha, binsize=1.0):
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

def fits2xspec(binsize=1, exptime=1, fwhm=0.0001, gresp=False, garf=False, filename='test.fits', rmfname='test.rmf', arfname='test.arf', TEStype='TMU524', Datatype='PHA', chan='ch65',gen=True,ch=2):
    afn="data1_CH"+str(ch)+"_ThLine.hdf5"
    with h5py.File(afn) as f:
        if gen == True:
            pha = f["ThLine"][:]*1e+3
        else:
            pha = f["analysis/se/pha"][:]*1e+4
        # separate bins
        if Datatype=='PHA':
            n, bins = histogram(pha[pha>0], binsize=binsize)
        else:
            n, bins = histogram(pha[pha>0.05], binsize=binsize)
        # py.figure()
        # py.hist(pha, bins=bins, histtype='stepfilled', color='k')
        # py.show()

        # par of fits
        filename = filename
        Exposuretime = exptime
        tc = int(n.sum())
        chn = len(bins)-1
        #x = (bins[:-1]+bins[1:])/2
        x = np.arange(0, (len(bins)-1), 1)
        y = n

        # print('#fits file name : %s' %(filename))
        # print('#info to make responce')
        # print('genrsp')
        # print('none')
        # print('%s' %rmfname)
        # print('constant')
        # print('no')
        # print('%f' %fwhm)
        # print('linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( 'linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( '5000000')
        # print( 'DUMMY')
        # print( 'DUMMY')
        # print( 'none')
        # print( 'none')
        # print( 'none')
        # print( '\n')
        
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
        
        if gresp==True:
            pass
            # if TEStype == 'Th229':
            #     mrTh._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
            # else:
            #     mr._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
        
        if garf==True:
            arf._make_arf_(pha, binsize=1, aname=arfname, chan=chan)

        # make fits
        col_x = pf.Column(name='CHANNEL', format='J', array=np.asarray(x))
        col_y = pf.Column(name='COUNTS', format='J', unit='count', array=np.asarray(y))
        cols  = pf.ColDefs([col_x, col_y])
        tbhdu = pf.BinTableHDU.from_columns(cols)

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
        # exthdr['USER']     = ('tasuku', 'User name of creator')
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

        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu, tbhdu])
        thdulist.writeto(filename, clobber=False)

