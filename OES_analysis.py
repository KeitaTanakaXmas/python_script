import numpy as np
import pandas as pd
import h5py
import sys
import os
from astropy.io  import fits
from xspec_tools import UnitConverter
import matplotlib.pyplot as plt
import plotly.express as px
from xspec import *

class Analysis:

    def __init__(self) -> None:
        home = os.environ['HOME']
        self.savehdf5 = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/Halosat_OES.hdf5'
        self.UC = UnitConverter()

    def set_data(self,obsid_list:list,word='fit/param'):
        with h5py.File(self.savehdf5,'r') as f:
            self.data = {
                'obsid':[obsid for obsid in obsid_list],
                'nh_gal':[f[f'{obsid}'][f'{word}/TBabs/nH'][0] for obsid in obsid_list],
                'nh_OES':[f[f'{obsid}'][f'{word}/TBabs_4/nH'][0] for obsid in obsid_list],
                'CXB':[self.UC.PowNorm2SB(f[f'{obsid}'][f'{word}/powerlaw/norm'][0],'halosat') for obsid in obsid_list],
                'CXB_ep':[self.UC.PowNorm2SB(f[f'{obsid}'][f'{word}/powerlaw/norm'][2],'halosat') for obsid in obsid_list],
                'CXB_em':[self.UC.PowNorm2SB(f[f'{obsid}'][f'{word}/powerlaw/norm'][1],'halosat') for obsid in obsid_list],
                'Warm_kT':[f[f'{obsid}'][f'{word}/apec_5/kT'][0] for obsid in obsid_list],
                'Warm_kT_ep':[f[f'{obsid}'][f'{word}/apec_5/kT'][2] for obsid in obsid_list],
                'Warm_kT_em':[f[f'{obsid}'][f'{word}/apec_5/kT'][1] for obsid in obsid_list],
                'Warm_EM':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_5/norm'][0],'halosat') for obsid in obsid_list],
                'Warm_EM_ep':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_5/norm'][2],'halosat') for obsid in obsid_list],
                'Warm_EM_em':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_5/norm'][1],'halosat') for obsid in obsid_list],
                'Hot_kT':[f[f'{obsid}'][f'{word}/apec_6/kT'][0] for obsid in obsid_list],
                'Hot_kT_ep':[f[f'{obsid}'][f'{word}/apec_6/kT'][2] for obsid in obsid_list],
                'Hot_kT_em':[f[f'{obsid}'][f'{word}/apec_6/kT'][1] for obsid in obsid_list],
                'Hot_EM':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_6/norm'][0],'halosat') for obsid in obsid_list],
                'Hot_EM_ep':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_6/norm'][2],'halosat') for obsid in obsid_list],
                'Hot_EM_em':[self.UC.ApecNorm2EM(f[f'{obsid}'][f'{word}/apec_6/norm'][1],'halosat') for obsid in obsid_list],
                'bg_s14_low':[f[f'{obsid}'][f'{word}/powerlaw_2_bg1/norm'][0] for obsid in obsid_list],
                'bg_s38_low':[f[f'{obsid}'][f'{word}/powerlaw_2_bg2/norm'][0] for obsid in obsid_list],
                'bg_s54_low':[f[f'{obsid}'][f'{word}/powerlaw_2_bg3/norm'][0] for obsid in obsid_list],
                'bg_s14_high':[f[f'{obsid}'][f'{word}/powerlaw_bg1/norm'][0] for obsid in obsid_list],
                'bg_s38_high':[f[f'{obsid}'][f'{word}/powerlaw_bg2/norm'][0] for obsid in obsid_list],
                'bg_s54_high':[f[f'{obsid}'][f'{word}/powerlaw_bg3/norm'][0] for obsid in obsid_list],
                'chi_sq':[f[f'{obsid}']['fit/stat'][0] for obsid in obsid_list],
                'dof':[f[f'{obsid}']['fit/stat'][1] for obsid in obsid_list]
                }
            self.df = pd.DataFrame(self.data)
            print(self.df)

    def load_nh(self):
        pass

    def hdf5tocsv(self,obsid_list:list,savecsv:str='OES_result.csv'):
        self.set_data(obsid_list)
        self.df.to_csv(savecsv)

    def Warm_Hot_correlation(self,obsid_list:list):
        self.set_data(obsid_list)
        Warm = self.data['Warm_EM']
        Warm_ep = self.data['Warm_EM_ep']
        Warm_em = self.data['Warm_EM_em']
        Hot = self.data['Hot_EM']
        Hot_ep = self.data['Hot_EM_ep']
        Hot_em= self.data['Hot_EM_em']
        fig = px.scatter(data_frame=self.df,x='Warm_EM',y='Hot_EM')
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        fig.show()

class XPLOT:

    def __init__(self) -> None:
        pass

    def plotting(self):
        pass
        

class Information:

    def __init__(self):
        home            = os.environ['HOME']
        self.HalosatDir = f"{home}/Dropbox/share/work/astronomy/Halosat/raw_data/products/"
        self.hdf5_inf = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_Halosat_information.hdf5'

    def readdata(self,obsid):
        self.obsid      = obsid
        self.s14_pi    = self.HalosatDir + f"hs{self.obsid}_s14.pi.gz"
        self.s38_pi    = self.HalosatDir + f"hs{self.obsid}_s38.pi.gz"
        self.s54_pi    = self.HalosatDir + f"hs{self.obsid}_s54.pi.gz"
        self.s14_hdu = fits.open(self.s14_pi)[1]
        self.s38_hdu = fits.open(self.s38_pi)[1]
        self.s54_hdu = fits.open(self.s54_pi)[1]
        print(f'Halosat Data Directory : {self.HalosatDir}')
        print(f'Observation Number : {self.obsid}')
        print(f's14 pi file : {self.s14_pi}')
        print(f's38 pi file : {self.s38_pi}')
        print(f's54 pi file : {self.s54_pi}')

    def read_hardrate(self):
        self.s14_hardrate = self.s14_hdu.header['HARDRATE']
        self.s38_hardrate = self.s38_hdu.header['HARDRATE']
        self.s54_hardrate = self.s54_hdu.header['HARDRATE']
        print('-----------------------------------------------------------')
        print(f's14 Hardrate = {self.s14_hardrate} counts/sec')
        print(f's38 Hardrate = {self.s38_hardrate} counts/sec')
        print(f's54 Hardrate = {self.s54_hardrate} counts/sec')

    def read_exposure(self):
        self.s14_exposure = self.s14_hdu.header['EXPOSURE']
        self.s38_exposure = self.s38_hdu.header['EXPOSURE']
        self.s54_exposure = self.s54_hdu.header['EXPOSURE']
        print('-----------------------------------------------------------')
        print(f's14 Exposure = {self.s14_exposure} counts/sec')
        print(f's38 Exposure = {self.s38_exposure} counts/sec')
        print(f's54 Exposure = {self.s54_exposure} counts/sec')

    def read_ra_dec(self):
        self.ra = self.s14_hdu.header['RA_NOM']
        self.dec = self.s38_hdu.header['DEC_NOM']
        print('-----------------------------------------------------------')
        print(f'RA = {self.ra} deg')
        print(f'DEC = {self.dec} deg')
    
    def read_all(self,obsid):
        self.readdata(obsid)
        self.read_hardrate()
        self.read_exposure()
        self.read_ra_dec()

    def savehdf5(self,obsid_list:list):
        with h5py.File(self.hdf5_inf,"a") as f:
            for obsid in obsid_list:
                if obsid in f.keys():
                    if "inf" in f[f'{obsid}'].keys():
                        del f[f'{obsid}']["inf"]
                self.read_all(obsid)
                f.create_dataset(f"{obsid}/inf/s14/hardrate",data=self.s14_hardrate)
                f.create_dataset(f"{obsid}/inf/s14/exposure",data=self.s14_exposure)
                f.create_dataset(f"{obsid}/inf/s38/hardrate",data=self.s38_hardrate)
                f.create_dataset(f"{obsid}/inf/s38/exposure",data=self.s38_exposure)
                f.create_dataset(f"{obsid}/inf/s54/hardrate",data=self.s54_hardrate)
                f.create_dataset(f"{obsid}/inf/s54/exposure",data=self.s54_exposure)
                f.create_dataset(f"{obsid}/inf/ra",data=self.ra)
                f.create_dataset(f"{obsid}/inf/dec",data=self.dec)
        print(f'Data saved in {self.hdf5_inf}')

    def setdata(self,obsid_list:list):
        with h5py.File(self.hdf5_inf,"a") as f:
            self.data = {
                'obsid':[obsid for obsid in obsid_list],
                's14_exposure':[f[f'{obsid}'][f'inf/s14/exposure'][...] for obsid in obsid_list],
                's38_exposure':[f[f'{obsid}'][f'inf/s38/exposure'][...] for obsid in obsid_list],
                's54_exposure':[f[f'{obsid}'][f'inf/s54/exposure'][...] for obsid in obsid_list],
                'ra':[f[f'{obsid}'][f'inf/ra'][...] for obsid in obsid_list],
                'dec':[f[f'{obsid}'][f'inf/dec'][...] for obsid in obsid_list],
            }

    def inftocsv(self,obsid_list:list,savecsv:str='OES_information.csv'):
        self.savehdf5(obsid_list)
        self.setdata(obsid_list)
        self.df = pd.DataFrame(self.data)
        print(self.df)
        self.df.to_csv(savecsv)                

if __name__ == '__main__':
    target_id = str(sys.argv[1])
    print(f'target ID = {target_id}')
    home = os.environ['HOME']
    d = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_fuller_txt.txt'
    f = pd.read_table(d,dtype='str')
    obsid = list(f['Obsid'])
    print(obsid)
    if target_id in obsid:
        OES_nh = float(f[f.Obsid == target_id]['nh'])
    else:
        print("ID check not over")
        print('Error: configuration failed', file=sys.stderr)
        sys.exit(1)
    with open('fuller_nh.txt','w') as o:
        print(OES_nh,file=o)



