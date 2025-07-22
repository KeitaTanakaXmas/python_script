import numpy as np
import h5py


__author__ =  'Tasuku Hayashi'
__version__=  '1.0'
__version__=  '2.0' #2023.05.09
__version__=  '2.1' #2023.05.10


class ReadHDF5:
    def __init__(self,dathdf5):
        self.dathdf5 = dathdf5
        
    def OpenHDF5(self):
        self.f = h5py.File(self.dathdf5, 'r')
        self.pulse = self.f['waveform']['pulse']
        self.noise = self.f['waveform']['noise']
        self.vres = self.f['waveform']['vres'][()]
        self.hres = self.f['waveform']['hres'][()]
        self.time = np.arange(self.pulse.shape[-1]) * self.hres

    def CloseHDF5(self):
        self.f.close()

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
            f['waveform'].create_dataset('vres', data=vres, dtype=np.float32)
            f['waveform'].create_dataset('hres', data=hres, dtype=np.float32)