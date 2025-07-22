import numpy as np
import h5py
from sqana import SQUID
import matplotlib.pyplot as plt

class MakeSettingFile:
    def __init__(self,setting_file_name='setting.hdf5'):
        self.setting_file_name = setting_file_name
        self.Min_dir           = '../data/PhiV/Min/Min.hdf5'
        self.Mfb_dir           = '../data/PhiV/Mfb/Mfb.hdf5'
        self.Excel_dir         = '../../SQUID_settingfile.xlsx'

    def write_mutual_inductance(self):
        s = SQUID(self.Min_dir)
        s.plotPhiV()
        plt.savefig('./figure/Min.png',dpi=300)
        with h5py.File(self.setting_file_name,'a') as f:
            if 'Min' in f.keys():
                del f['Min']
            for sbname in s.sbias_list:        
                f.create_dataset(f'Min/{sbname}/average',data=s.__dict__[sbname].M_H)
                f.create_dataset(f'Min/{sbname}/error',data=s.__dict__[sbname].Me_H)

        s = SQUID(self.Mfb_dir)
        s.plotPhiV()
        plt.savefig('./figure/Mfb.png',dpi=300)
        with h5py.File(self.setting_file_name,'a') as f:
            if 'Mfb' in f.keys():
                del f['Mfb']
            for sbname in s.sbias_list:        
                f.create_dataset(f'Mfb/{sbname}/average',data=s.__dict__[sbname].M_H)
                f.create_dataset(f'Mfb/{sbname}/error',data=s.__dict__[sbname].Me_H)

    def load_excel_file(self,attribute='pulse'):
        import pandas as pd
        if attribute == 'pulse':
            self.df = pd.read_excel(self.Excel_dir, sheet_name='Pulse setting', skiprows=2)
            self.df = self.df[self.df['ID'].notnull()]
            self.df = self.df.dropna(subset=['ID'])
            print(self.df)
            with h5py.File(self.setting_file_name,'a') as f:
                if 'Pulse' in f.keys():
                    del f['Pulse']
                for ID in self.df['ID']:
                    f.create_dataset(f'Pulse/{ID}/SQUID_current_bias',data=self.df[self.df['ID']==ID]['Ib [uA](**)'].values[0])
                    f.create_dataset(f'Pulse/{ID}/Rfb',data=self.df[self.df['ID']==ID]['Rfb[kOhm]'].values[0])
        elif attribute == 'measurement':
            self.df = pd.read_excel(self.Excel_dir, sheet_name='IV,Z', skiprows=2)
            self.df = self.df[self.df['ID'].notnull()]
            self.df = self.df.dropna(subset=['ID'])
            print(self.df)
            with h5py.File(self.setting_file_name,'a') as f:
                if 'Measurement' in f.keys():
                    del f['Measurement']
                for ID in self.df['ID']:
                    f.create_dataset(f'Measurement/{ID}/SQUID_current_bias',data=self.df[self.df['ID']==ID]['Ib [uA](**)'].values[0])
                    f.create_dataset(f'Measurement/{ID}/Rfb',data=self.df[self.df['ID']==ID]['Rfb[kOhm]'].values[0])


    def all_process(self):
        self.write_mutual_inductance()
        self.load_excel_file()