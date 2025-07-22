#!/usr/bin/env python

import time, glob
import numpy as np
from scipy.io import loadmat
import h5py

#memo
#2023.04.07 Changing the data type form float64 to float32.

class mat2hdf5:
    def __init__(self):
        self.wa = []
        self.wb = []
        self.rebin = 1
    
    def mopen(self,channel='A',idir=None):
        if idir:
            idir=idir
        else:
            idir='.'
        
        self.channel = channel #channel = A or B
        ifile = glob.glob(idir+'/*.mat')
        ifile.sort()

        ch = 1
        if len(ifile) == 0:
            print("No matlab file found")
    
        for fc, f in enumerate(ifile):
            #print( "%s (%d/%d)\r" % (f.split('/')[-1], fc+1, len(ifile)))
            hoge = f.split('/')[-1]
            print(f'{hoge} ({fc+1:02}/{len(ifile)}), ({self.dc+1:04}/{self.ic:04}) \r',end="")
            dat = loadmat(f)
            self.hres, l, _w, rl = dat['Tinterval'][0][0], dat['Length'][0][0], dat[self.channel].T[ch-1], dat['RequestedLength']
            if np.isinf(_w).any():
                continue

            _w = _w[:int(rl)]

            # Rebin
            if self.rebin > 1:
                _w = _w[:(_w.shape[-1]//self.rebin)*self.rebin]
                _w = _w.reshape(_w.shape[-1]//self.rebin, 1, self.rebin).mean(axis=-1)
                _w = _w.reshape(1,len(_w))[0]

            if self.channel=='A':
                self.wa.append(_w)
            else:
                self.wb.append(_w)
                self.hresb = self.hres

    def mat2hdf5(self,dathdf5,idirs,datatype='pulse',rebin=1,inverce=False):
        '''
        dathdf5  is savefile name
        idirs    is folder name (ex. '20230209*')
        datatype is 'pulse' or 'phiv'
        rebin    is down sample rate
        '''
        self.dathdf5 = dathdf5
        if self.checkhdf5():
            pass
        else:
            self.Inithdf5()
            self.wa    = []
            self.wb    = []
            self.rebin = rebin
            idirs      = glob.glob(idirs)
            idirs.sort()
            self.ic = len(idirs)
            self.idirs = idirs

        if datatype=='pulse':            
            for self.dc, idir in enumerate(idirs):
                self.mopen(channel='A',idir=idir)

            self.wa = np.asarray(self.wa)

            self.vres = np.float32(abs(self.wa).max()/2**15)
            self.wa = np.vectorize(int)(self.wa/self.vres)
            self.hres *= rebin
            self.SetPulseData()

        elif datatype=='phiv':
            for self.dc, idir in enumerate(idirs):
                self.idir = idir
                self.wa   = []
                self.wb   = []
                #x=A=I, y=B=V
                self.mopen(channel='A',idir=idir)
                self.mopen(channel='B',idir=idir)
                self.wa = np.asarray(self.wa)
                self.wb = np.asarray(self.wb)
                if inverce==True:
                    wa = self.wb
                    wb = self.wa
                    self.wa = wa
                    self.wb = wb
                self.vres = np.float32(abs(self.wa).max()/2**15)
                self.vresb = np.float32(abs(self.wb).max()/2**15)
                self.wa = np.vectorize(int)(self.wa/self.vres)
                self.xa = np.average(self.wa*self.vres, axis=0)
                self.wb = np.vectorize(int)(self.wb/self.vresb)
                self.ya = np.average(self.wb*self.vresb, axis=0)
                self.hres *= rebin
                self.SetPhiVData() 

    def SetPulseData(self):
        print("Generating HDF5 files...")
        with h5py.File(self.dathdf5, 'a') as f:
            f['waveform'].create_dataset('wave', data=self.wa,   dtype=np.float32)
            f['waveform'].create_dataset('vres', data=self.vres, dtype=np.float32)
            f['waveform'].create_dataset('hres', data=self.hres, dtype=np.float32)
            f['header'].create_dataset('UseDirectory', data=self.idirs)

    def SetPhiVData(self):
        print("Generating HDF5 files...")
        with h5py.File(self.dathdf5, 'a') as f:
            f['waveform'].create_group(self.idir)
            f['waveform'][self.idir].create_dataset('x', data=self.xa)
            f['waveform'][self.idir].create_dataset('y', data=self.ya)
            f['waveform'][self.idir].create_dataset('vres', data=self.vres)
            f['waveform'][self.idir].create_dataset('vresb', data=self.vresb)
            f['waveform'][self.idir].create_dataset('hres', data=self.hres)
            f['waveform'][self.idir].create_dataset('hresb', data=self.hresb)

    def Inithdf5(self):
        with h5py.File(self.dathdf5, 'a') as f:
            if any(['waveform' == u for u in f.keys()]):
                print(f'Overwrite {self.dathdf5}')
                del f['waveform']

            f.create_group('waveform')

            if any(['header' == v for v in f.keys()]):
                del f['header']

            f.create_group('header')
            f['header'].create_dataset('MakeDate', data=time.strftime('%Y.%m.%d-%H:%M:%S'))

    def checkhdf5(self):
        hdf5file = glob.glob('*hdf5')
        hdf5file = np.asarray(hdf5file)
        if len(hdf5file)==0:
            return False
        else:
            if any([self.dathdf5 == fi for fi in hdf5file]):
                print(f'there is already {self.dathdf5} file.')
                return True
            else:
                return False

class ReadHDF5:
    def __init__(self,dathdf5):
        self.dathdf5 = dathdf5

    def ReadHdf5(self,sp=True):
        with h5py.File(self.dathdf5, 'a') as f:
            wave = f['waveform']['wave'][()]
            vres = f['waveform']['vres'][()]
            hres = f['waveform']['hres'][()]
            wave = wave*vres
            if sp:
                if wave[0].shape[0] % 2 != 0:
                    wave = wave.T[:-1]
                    wave = wave.T
                    
                w = wave.reshape(int(wave.shape[0]*2), int(wave.shape[-1]/2))
                p = w[1::2]
                n = w[::2]
                t = np.arange(p.shape[-1]) * hres
                return t, p, n
            else:
                t = np.arange(wave.shape[-1]) * hres
                return t, wave

    def ReadPhiVHdf5(self):
        self.x = {}
        self.y = {}
        with h5py.File(self.dathdf5, 'a') as f:
            for i in f['waveform'].keys():
                self.x[i]  = f['waveform'][i]['x'][()]
                self.y[i]  = f['waveform'][i]['y'][()]
                self.hres  = f['waveform'][i]['hres'][()]
                self.hresb = f['waveform'][i]['hresb'][()]
                self.xt = np.arange(self.x[i].shape[-1]) * self.hres
                self.xt = self.xt - (self.xt[-1]/2)
                self.yt = np.arange(self.x[i].shape[-1]) * self.hres
                self.yt = self.yt - (self.yt[-1]/2)

        return self.xt, self.x, self.yt, self.y

















