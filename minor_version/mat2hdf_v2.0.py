#!/usr/bin/env python

import time, glob
import numpy as np
from scipy.io import loadmat
import h5py

#memo
#2023.04.07 Changing the data type form float64 to float32.
#2023.06.20 ver2.0

__author__ =  'Tasuku Hayashi'
__version__=  '1.0' #2023.04.07
__version__=  '2.0' #2023.06.20

print('========================================================')
print(f' mat2hdf version {__version__}')
print('========================================================')

class mat2hdf5:
    def __init__(self, winpath=False, channel='A', max_counts_num=4000):
        self.channel = channel #channel of pico-scope = A or B 
        self.wa = []
        self.wb = []
        self.rebin = 1
        self.winpath = winpath
        self.max_counts_num = max_counts_num

    def SeparateDirectory(self, idirs):
        '''
        
        '''
        import math

        self.idirs_list = []
        self.pulse_num_list = []

        self.idirs = glob.glob(idirs)
        self.idirs.sort()
        self.num_dc = len(self.idirs) #number of directory

        ifile = glob.glob(self.idirs[0]+'/*.mat')
        ifile.sort()
        self.num_fc = len(ifile) #number of file 

        total_p_num = self.num_dc * self.num_fc #total pulse counts
        sp_num = total_p_num//self.max_counts_num #sp_num = number of split counts
        self.sp_num = sp_num
        print(f'number of files to create(sp_num) = {sp_num}')

        if sp_num == 0:
            sp_d_num = self.num_dc
        else:
            sp_d_num = math.ceil(total_p_num//sp_num/self.num_fc) #sp_d_num = number of split directory

        print(f'number of unit-directory={sp_d_num}')

        set_num = total_p_num//sp_d_num//self.num_fc
        print(f'Set num={set_num}')

        remainder_fc = total_p_num - (sp_d_num*self.num_fc*set_num)
        remainder_dc = remainder_fc // self.num_fc
        print(f'number of remainder directory={remainder_dc}')
        for i in range(set_num):
            self.idirs_list.append(sp_d_num)
            self.pulse_num_list.append(sp_d_num*self.num_fc)

        if sp_num != 0:
            if remainder_fc < self.max_counts_num/8:
                self.idirs_list.append(sp_d_num+remainder_dc)
                self.pulse_num_list.append((sp_d_num+remainder_dc)*self.num_fc)
            else:
                self.idirs_list.append(remainder_dc)
                self.pulse_num_list.append(remainder_dc*self.num_fc)

        print(self.idirs_list)
        print(self.pulse_num_list)
        print(sum(self.idirs_list))

    def getdirs(self):
        ifile = glob.glob(idir+'/*.mat')
        ifile.sort()

    def mopen(self,idir=None):
        if idir:
            idir=idir
        else:
            idir='.'
        
        ifile = glob.glob(idir+'/*.mat')
        ifile.sort()
        self.total_fc = len(ifile)

        ch = 1
        if len(ifile) == 0:
            print("No matlab file found")
    
        for fc, f in enumerate(ifile):
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

    def mat2hdf5(self,dathdf5,idirs,datatype='pulse',rebin=1):
        '''
        dathdf5  is savefile name
        idirs    is folder name (ex. '20230209*')
        datatype is 'pulse' or 'phiv'
        rebin    is down sample rate
        '''
        self.dathdf5 = dathdf5
        self.datatype = datatype
        
        self.data_name_head = self.dathdf5[:6]

        self.SeparateDirectory(idirs)

        #self.checkhdf5()

        self.wa    = []
        self.wb    = []
        self.rebin = rebin
        self.ic = len(self.idirs)

        if self.datatype=='pulse': 
            for e, num in enumerate(self.idirs_list):
                self.pulse_counts = self.pulse_num_list[e]
                self.wa           = []
                self.idirs_cut    = self.idirs[e*num:(e+1)*num]
                self.dathdf5_new  = self.data_name_head+f'pn-{(e+1)}.hdf5'
                print(self.dathdf5_new)
                self.Inithdf5()
                self.makeheder()
                for self.dc, idir in enumerate(self.idirs_cut):
                    self.mopen(idir=idir)

                self.wa = np.asarray(self.wa)

                self.vres = np.float32(abs(self.wa).max()/2**15)
                self.wa = np.vectorize(int)(self.wa/self.vres)
                self.hres *= rebin

                #
                if self.wa[0].shape[0] % 2 != 0:
                    self.wa = self.wa.T[:-1]
                    self.wa = self.wa.T
                    
                w = self.wa.reshape(int(self.wa.shape[0]*2), int(self.wa.shape[-1]/2))
                self.p = w[1::2]
                self.n = w[::2]
                self.SetPulseData()

        elif self.datatype=='phiv':
            for self.dc, idir in enumerate(idirs):
                self.idir = idir
                self.wa   = []
                self.wb   = []
                #x=A=I, y=B=V
                self.channel = 'A'
                self.mopen(idir=idir)
                self.channel = 'B'
                self.mopen(idir=idir)

                self.wa = np.asarray(self.wa)
                self.wb = np.asarray(self.wb)

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
        with h5py.File(self.dathdf5_new, 'a') as f:
            f['waveform'].create_dataset('pulse', data=self.p, dtype=np.float32)
            f['waveform'].create_dataset('noise', data=self.n, dtype=np.float32)
            f['waveform'].create_dataset('vres',  data=self.vres, dtype=np.float32)
            f['waveform'].create_dataset('hres',  data=self.hres, dtype=np.float32)

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
        with h5py.File(self.dathdf5_new, 'a') as f:
            if any(['waveform' == u for u in f.keys()]):
                print(f'Overwrite {self.dathdf5_new}')
                del f['waveform']

            f.create_group('waveform')

    def checkhdf5(self):
        hdf5file = glob.glob('*hdf5')
        hdf5file = np.asarray(hdf5file)
        if len(hdf5file)==0:
            return False
        else:
            if any([self.dathdf5 == fi for fi in hdf5file]):
                print(f'there is already {self.dathdf5} file.')
                exit()
            else:
                return False

    def Convert2pn(self):
        with h5py.File(self.dathdf5, 'r') as f:
            wave = f['waveform']['wave'][()]
            vres = f['waveform']['vres'][()]
            hres = f['waveform']['hres'][()]
            if wave[0].shape[0] % 2 != 0:
                wave = wave.T[:-1]
                wave = wave.T
                
            w = wave.reshape(int(wave.shape[0]*2), int(wave.shape[-1]/2))
            p = w[1::2]
            n = w[::2]
            t = np.arange(p.shape[-1]) * hres
        
        dathdf5_new = self.dathdf5[:-5]+'pn.hdf5'
        print("Generating HDF5 files...")
        with h5py.File(dathdf5_new, 'a') as f:
            f.create_group('waveform')
            f['waveform'].create_dataset('pulse', data=p, dtype=np.float32)
            f['waveform'].create_dataset('noise', data=n, dtype=np.float32)
            f['waveform'].create_dataset('vres',  data=vres, dtype=np.float32)
            f['waveform'].create_dataset('hres',  data=hres, dtype=np.float32)

    def makeheder(self):
        import pandas as pd

        if self.datatype=='pulse':

            if self.winpath:
                settingfilepath = 'C:\\hoge\\hoge\\'+'SQUID_settingfile.xlsx'
            else:
                settingfilepath = '/Users/otaryo/OneDrive - ISAS Yamasaki-Lab/A1613/CryogenicData/SQUID_settingfile.xlsx'

            df = pd.read_excel(settingfilepath, skiprows=2, sheet_name='Pulse setting')

            print('loading SQUID_settingfile')
            print(f'')

            self.runid = self.dathdf5_new[0:6]
            print(f'Checked id that matches {self.runid}...')
            run_num = np.argwhere(df['ID'].values==self.runid)[0][0]
            
            with h5py.File(self.dathdf5_new, 'a') as f:
                if any(['header' == v for v in f.keys()]):
                    del f['header']

                f.create_group('header')
                f['header'].create_dataset('MakeDate', data=time.strftime('%Y.%m.%d-%H:%M:%S'))
                f['header'].create_dataset('mat2hdf_Ver', data=__version__)
                f['header'].create_dataset('MeasurementsDate', data=df['date'][run_num])
                f['header'].create_dataset('UseDirectory', data=self.idirs_cut)
                f['header'].create_dataset('AllDirectory', data=self.idirs)
                f['header'].create_dataset('NumOfPData', data=self.pulse_counts)
                f['header'].create_dataset('SpDataNum', data=self.sp_num)
                #TES information
                f['header'].create_dataset('MeasurementsID', data=df['ID'][run_num])
                f['header'].create_dataset('LOT-No', data=df['LOT No.'][run_num])
                f['header'].create_dataset('ChipID', data=df['ChipID'][run_num])
                f['header'].create_dataset('Pix.ID', data=df['Pix.ID'][run_num])
                f['header'].create_dataset('DataType', data=df['type'][run_num])
                #magnicon
                f['header'].create_dataset('Magnicon_Ch', data=df['Channel'][run_num])
                f['header'].create_dataset('SQUID_Bias', data=df['Ib [uA](**)'][run_num])
                f['header'].create_dataset('SQUID_Vb', data=df['Vb [uV]'][run_num])
                f['header'].create_dataset('SQUID_Phib', data=df['Phib [uA]'][run_num])
                f['header'].create_dataset('TES_Bias', data=df['I TES[uA]'][run_num])
                f['header'].create_dataset('Rfb', data=df['Rfb[kOhm]'][run_num])
                f['header'].create_dataset('GainBW', data=df['Gain BW[GHz]'][run_num])
                #PicoScope
                f['header'].create_dataset('V_div', data=df['div A[V/div]'][run_num])
                f['header'].create_dataset('time_div', data=df['time div[ms/div]'][run_num])
                f['header'].create_dataset('sample_kS', data=df['sample[kS]'][run_num])
                f['header'].create_dataset('bit', data=df['bit'][run_num])
                f['header'].create_dataset('trigger_V', data=df['trigger [V]'][run_num])
                f['header'].create_dataset('edge_point_percent', data=df['%'][run_num])
                f['header'].create_dataset('Edge', data=df['Edge'][run_num])

class ReadHDF5:
    def __init__(self):
        hdf5List = glob.glob('*.hdf5')
        hdf5List.sort()
        for e, hdf5name in enumerate(hdf5List):
            print(f'{e}: {hdf5name}')

    def ReadHDF5(self, dathdf5):
        self.dathdf5 = dathdf5
        print('Checking mat2hdf version...')
        with h5py.File(self.dathdf5, 'a') as f:
            ver = f['header']['mat2hdf_Ver'][()].decode('utf-8')

        if ver == __version__:
            print('OK.')
            self.GetHeader()
            self.OpenHDF5()
        else:
            print('mat2hdf version not match.')
            self.GetHeader()
            self.OpenHDF5()

    def OpenHDF5(self):
        self.f = h5py.File(self.dathdf5, 'r')
        self.pulse = self.f['waveform']['pulse']
        self.noise = self.f['waveform']['noise']
        self.vres  = self.f['waveform']['vres'][()]
        self.hres  = self.f['waveform']['hres'][()]
        self.time  = np.arange(self.pulse.shape[-1]) * self.hres

    def CloseHDF5(self):
        self.f.close()

    def ReadPhiVHDF5(self, dathdf5):
        self.dathdf5 = dathdf5
        x = {}
        y = {}
        with h5py.File(self.dathdf5, 'a') as f:
            for i in f['waveform'].keys():
                x[i]  = f['waveform'][i]['x'][()]
                y[i]  = f['waveform'][i]['y'][()]
                hres  = f['waveform'][i]['hres'][()]
                hresb = f['waveform'][i]['hresb'][()]
                xt = np.arange(x[i].shape[-1]) * hres
                xt = xt - (xt[-1]/2)
                yt = np.arange(x[i].shape[-1]) * hres
                yt = yt - (yt[-1]/2)

        return xt, x, yt, y

    def GetHeader(self):
        with h5py.File(self.dathdf5, 'a') as f:
            self.MeasurementsDate = f['header']['MeasurementsDate'][()].decode('utf-8')
            self.mat2hdf_Ver      = f['header']['mat2hdf_Ver'][()].decode('utf-8')
            self.MakeDate         = f['header']['MakeDate'][()].decode('utf-8')
            self.NumOfPulseData   = f['header']['NumOfPData'][()]
            self.NumOfFileCreate  = f['header']['SpDataNum'][()]
            self.LOTNo            = f['header']['LOT-No'][()].decode('utf-8')
            self.ChipID           = f['header']['ChipID'][()].decode('utf-8')
            self.PixID            = f['header']['Pix.ID'][()].decode('utf-8')
            self.DataType         = f['header']['DataType'][()].decode('utf-8')
            self.MagniconCh       = int(f['header']['Magnicon_Ch'][()])
            self.SQUID_Bias       = f['header']['SQUID_Bias'][()]
            self.SQUID_Vb         = f['header']['SQUID_Vb'][()]
            self.SQUID_Phib       = f['header']['SQUID_Phib'][()]
            self.TES_Bias         = f['header']['TES_Bias'][()]
            self.Rfb              = f['header']['Rfb'][()]
            self.GainBW           = f['header']['GainBW'][()]
            self.V_div            = f['header']['V_div'][()]
            self.time_div         = f['header']['time_div'][()]
            self.sample_kS        = f['header']['sample_kS'][()]
            self.bit              = f['header']['bit'][()]
            self.trigger_V        = f['header']['trigger_V'][()]
            self.edge_point       = f['header']['edge_point_percent'][()]
            self.Edge             = f['header']['Edge'][()].decode('utf-8')
            self.UseDirectory     = f['header']['UseDirectory'][()]
            self.AllDirectory     = f['header']['AllDirectory'][()]
            self.dircounts        = len(self.UseDirectory)

    def info(self):
        print( ' ==============================================')
        print( ' HDF5 header Information ')
        print(f' mat2hdf version {self.mat2hdf_Ver} ')
        print( ' ==============================================')
        print(f' File name            : {self.dathdf5}')
        print(f' Measurements Date    : {self.MeasurementsDate}')
        print(f' Make Date            : {self.MakeDate}')
        print(f' Num of File Create   : {self.NumOfFileCreate}')
        print(f' Number of Pulse Data : {self.NumOfPulseData}')
        print( ' --------------------------------------------')
        print( ' TES information')
        print(f' LOT No.              : {self.LOTNo}')
        print(f' Chip ID              : {self.ChipID}')
        print(f' Pixel ID             : {self.PixID}')
        print(f' Data Type            : {self.DataType}')
        print( ' --------------------------------------------')
        print( ' Magnicon Setting')
        print(f' Magnicon Channel     : {self.MagniconCh}')
        print(f' SQUID Bias           : {self.SQUID_Bias} (uA)')
        print(f' SQUID Vb             : {self.SQUID_Vb} (uV)')
        print(f' SQUID Phib           : {self.SQUID_Phib} (uA)')
        print(f' TES_Bias             : {self.TES_Bias} (uA)')
        print(f' RFB                  : {self.Rfb} (kOhm)')
        print(f' Gain BW              : {self.GainBW} (GHz)')
        print( ' --------------------------------------------')
        print( ' PicoScope Setting')
        print(f' V div                : {self.V_div} (V/div)')
        print(f' time div             : {self.time_div} (ms/div)')
        print(f' sample               : {self.sample_kS} (kS)')
        print(f' bit                  : {self.bit} (bit)')
        print(f' trigger              : {self.trigger_V}')
        print(f' edge point           : {self.edge_point} (%)')
        print(f' Edge type            : {self.Edge}')
        print( ' --------------------------------------------')
        print(f' Number of used directory : {self.dircounts}')
        print( ' Used Directory : ')
        print(f' {self.UseDirectory}')
        print( ' ALL Directory : ')
        print(f' {self.AllDirectory}')
        print('==============================================')

