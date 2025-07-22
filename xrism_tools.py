from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from xspec import *
import h5py
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import astropy.units as u
from astropy import units as u
from astropy import constants as c
from datetime import datetime
import pyatomdb
import os
import glob
import socket
import re

from Resonance_Scattering_Simulation import PlotManager
# from matplotlib.ticker import FuncFormatter
# from matplotlib.colors import LogNorm
# from astropy.wcs import WCS
# import matplotlib.patches as patches
# import glob
# import matplotlib.cm as cm
# from astropy.cosmology import FlatLambdaCDM

from astropy.time import Time
from IPython.display import display
import time

from matplotlib.patches import Rectangle
class Cluster:
    def __init__(self):
        pass

    def oscillator_strength(self,Z=26,z1=25,upperlev=7,lowerlev=1):
        import pyatomdb
        self.f = pyatomdb.atomdb.get_oscillator_strength(Z, z1, upperlev, lowerlev, datacache=False)
        print('----------------------')
        print('Oscillator Strength')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {upperlev}, lowerlev = {lowerlev}')
        print(f'f = {self.f}')

    def line_manager(self,Z=26,state='w',Tmin=1, Tmax=10, dT=1000):
        import pyatomdb
        # declare the Collisional Ionization Equilibrium session
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(Tmin, Tmax, dT)
        if state == 'w':
            z1 = Z-1
            up = 7
        elif state == 'x':
            z1 = Z-1
            up = 6
        elif state == 'y':
            z1 = Z-1
            up = 5
        elif state == 'z':
            z1 = Z-1
            up = 2
        elif state == 'Lya2':
            z1 = Z
            up = 3
        elif state == 'Lya1':
            z1 = Z
            up = 4
        elif state == 'Heb1':
            z1 = Z-1
            up = 13
        elif state == 'Heb2':
            z1 = Z-1
            up = 11
        elif state == 'u':
            z1 = Z-2
            up = 45
        elif state == 'r':
            z1 = Z-2
            up = 47
        elif state == 't':
            z1 = Z-2
            up = 48
        elif state == 'q':
            z1 = Z-2
            up = 50
        else:
            print('state is not defined')
        self.z1 = z1
        ldata = sess.return_line_emissivity(kTlist, Z, z1, up, 1)
        self.line_energy = ldata['energy'] * 1e+3
        print('----------------------')
        print('Line Energy')
        print(f'state = {state}')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {up}')
        print(f'line energy = {self.line_energy} eV')
        self.oscillator_strength(Z=Z,z1=z1,upperlev=up,lowerlev=1)

    def line_manager_emm(self,Z=26,state='w',Tmin=1, Tmax=10, dT=1000):
        import pyatomdb
        # declare the Collisional Ionization Equilibrium session
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(Tmin, Tmax, dT)
        if state == 'w':
            z1 = Z-1
            up = 7
        elif state == 'x':
            z1 = Z-1
            up = 6
        elif state == 'y':
            z1 = Z-1
            up = 5
        elif state == 'z':
            z1 = Z-1
            up = 2
        elif state == 'Lya2':
            z1 = Z
            up = 3
        elif state == 'Lya1':
            z1 = Z
            up = 4
        elif state == 'Heb1':
            z1 = Z-1
            up = 13
        elif state == 'Heb2':
            z1 = Z-1
            up = 11
        elif state == 'u':
            z1 = Z-2
            up = 45
        elif state == 'r':
            z1 = Z-2
            up = 47
        elif state == 't':
            z1 = Z-2
            up = 48
        elif state == 'q':
            z1 = Z-2
            up = 50
        else:
            print('state is not defined')
        self.z1 = z1
        self.ldata = sess.return_line_emissivity(kTlist, Z, z1, up, 1)
        self.line_energy = ldata['energy'] * 1e+3
        print('----------------------')
        print('Line Energy')
        print(f'state = {state}')
        print(f'Z = {Z}, z1 = {z1}, upperlev = {up}')
        print(f'line energy = {self.line_energy} eV')
        self.oscillator_strength(Z=Z,z1=z1,upperlev=up,lowerlev=1)

    def emissivity(self,T,linefile='default', cocofile='default'):
        # declare the Collisional Ionization Equilibrium session
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,8.0,1000)

        fig = plt.figure()

        ax= fig.add_subplot(111)

        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1, apply_abund=False)
            if up == 2:
                z = ldata['epsilon']
                print('z')
                print(ldata['epsilon'])
            if up == 7:
                w = ldata['epsilon']
                print('w')
                print(ldata['epsilon'])

        ax.plot(kTlist, w/z, label='z', color='darkblue')
        ax.set_xlabel('Temperature (keV)')
        ax.set_ylabel('w/z flux ratio')
        ax.set_title('atomdb v3.0.9')
        plt.show()
        fig.savefig('wz_ratio.png',dpi=300,transparent=True)
        return w, z

class Local:
    '''This is Local class for PKS 0745-191 analysis'''
    def __init__(self):
        self.obsid = '000112000'
        self.t_start, self.t_end = '2023-11-08 10:21:00', '2023-11-11 12:01:04'
        self.datas = {
            'Open': '2023-11-08 10:21:00',
            'ND': '2023-11-08 23:51:31',
            'Be': '2023-11-09 05:26:31',
            'OBF': '2023-11-09 13:02:51',
            '55Fe': '2023-11-09 16:47:51',
            'ADR1': '2023-11-08 20:10:01',
            'ADR2': '2023-11-10 15:10:00'
            #'LED_BRIGHT': '2023-11-10 12:01:01'
        }

        self.t_start = datetime.strptime(self.t_start, '%Y-%m-%d %H:%M:%S')
        self.t_end = datetime.strptime(self.t_end, '%Y-%m-%d %H:%M:%S')

        # 各辞書の値をdatetimeに変換して差を秒数で計算
        date_diffs = {}
        for label, date_str in self.datas.items():
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            diff_seconds = (date_obj - self.t_start).total_seconds()  # t_startからの差
            date_diffs[label] = diff_seconds


        self.data_xr_time ={}
        for label, data in self.datas.items():
            date_obj = datetime.strptime(data, '%Y-%m-%d %H:%M:%S')
            self.data_xr_time[label] = self.XRISMtime(Time(date_obj))

        for label, diff in date_diffs.items():
            print(f"{label}: {diff:.0f}秒")

        self.data_diffs = date_diffs

    def XRISMtime(self, t):
        return t.cxcsec - Time('2019-01-01 00:00:00.000').cxcsec

class XspecFit:
    def __init__(self,savefile,atomdb_version='3.0.9',abundance_table='lpgs'):
        print('initialize')
        cdir = os.getcwd()
        self.savefile=f"{cdir}/{savefile}"
        self.plot_params = {#'backend': 'pdf',
            'axes.labelsize': 15,
            'axes.linewidth': 1.0,
            'axes.labelweight': 500,
            'font.size': 15,
            'font.weight':500,
            'legend.fontsize': 12,
            'legend.borderpad': 0.5,
            'legend.framealpha': 1,
            'legend.fancybox': False,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'font.family': 'serif',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True
            }

        plt.rcParams.update(self.plot_params)
        Plot.device = "/xs"
        Plot.xAxis="keV"
        Fit.query = 'yes'
        Fit.method = ["leven", 999, 1e-4] 
        Fit.criticalDelta = 1e-4
        Plot.add = True
        Xset.abund=abundance_table
        Fit.statMethod = "cstat"
        self.atomdb_version = atomdb_version
        self.thermal_broadening("yes")
        dropbox = os.getenv("DROPBOX")
        print(dropbox)
        self.dropbox = dropbox 
        hostname = socket.gethostname()
        print(hostname)
        if hostname == 'KeitaTanakaMBpro1.local' or hostname == 'note192.astro.isas.jaxa.jp':
            AllModels.lmod("mymodel",f"{dropbox}/xspec_local_model_for_macbook")
            Xset.parallel.leven = 3
            Xset.parallel.error = 3
            Xset.parallel.steppar = 3
        else:
            AllModels.lmod("mymodel",f"{dropbox}/xspec_local_model")
            Xset.parallel.leven = 6
            Xset.parallel.error = 6
            Xset.parallel.steppar = 6
        if os.path.exists(self.savefile):
            key_list = h5py.File(self.savefile, 'r').keys()
            print('-'*50)
            print('Saved Model List')
            for key in key_list:
                print(key)

    def add_model(self, model_str:str, add_model_str:str, model_name:str='', sourceNum=1):
        '''Add model to the existing model
        model_str: str 
            Existing Model string
        add_model_str: str
            Add model string
        model_name: str
            Adding Model name

        Returns
        -------
        new_model: Model
            New Model
        new_model_str: str
            New Model string
        '''
        new_model_str = model_str + '+' + add_model_str
        new_model = Model(new_model_str, model_name, sourceNum)
        return new_model, new_model_str
    
    def load_spectrum(self,spec,rmf,arf,multiresp=None,multiresp2=None,bgd=None,rng=(2.0,17.0),rebin=False,datagroup=1,spectrumgroup=1):
        AllData(f'{datagroup}:{spectrumgroup} {spec}')
        s = AllData(spectrumgroup)
        s.response = rmf
        s.response.arf = arf
        if multiresp != None:
            s.multiresponse[1] = multiresp
        if multiresp2 != None:
            s.multiresponse[2] = multiresp2
        if bgd != None:
            s.background = bgd
        self.set_data_range(rng, datagroup)
        self.set_xydata(plotgroup=spectrumgroup,rebin=rebin)

    def load_data_ssm(self, xcm="dummy", spec_num=8, rng=(  )):
        '''Load data from SSM file
        xcm: str
            SSM file path
        '''
        Xset.restore(xcm)
        print(f"Loaded SSM file: {xcm}")
        for i in range(0,spec_num):
            self.set_data_range(rng=rng,datagroup=i+1)
            self.set_xydata(subject='data', plotgroup=i+1, rebin=1)

    def thermal_broadening(self, setting="yes"):
        Xset.addModelString("APECTHERMAL", setting)

    def set_xydata(self,subject='data',plotgroup=1,rebin=False):
        if rebin != False:
            Plot.setRebin(rebin, 1000, plotgroup)
        Plot(subject)
        self.xs=Plot.x(plotgroup,1)
        self.ys=Plot.y(plotgroup,1)
        if subject == 'data':
            self.xe=Plot.xErr(plotgroup,1)
            self.ye=Plot.yErr(plotgroup,1)
            return self.xs,self.ys,self.xe,self.ye
        else:
            return self.xs,self.ys,[],[]  

    def set_data_range(self,rng=(2.0,17.0),datagroup=1):
        AllData.notice(f"{datagroup}:{rng[0]}-{rng[1]}")
        AllData.ignore(f"{datagroup}:**-{rng[0]} {rng[1]}-**")

    def load_apecroot(self,line):
        if line == 'w':
            apecroot=f"/Users/keitatanaka/apec_modify/v{self.atomdb_version}/del_w/apec_v{self.atomdb_version}"
        elif line == 'wz':
            apecroot=f"/Users/keitatanaka/apec_modify/v{self.atomdb_version}/del_w_z/apec_v{self.atomdb_version}"
        elif line == 'Heab':
            apecroot=f"/Users/keitatanaka/apec_modify/v{self.atomdb_version}/del_Heab/apec_v{self.atomdb_version}"
        elif line == 'all':
            apecroot=f"/Users/keitatanaka/apec_modify/v{self.atomdb_version}/del_Heab_Lya/apec_v{self.atomdb_version}"
        else:
            apecroot='/Users/keitatanaka/heasoft-6.34/spectral/modelData/apec_v3.0.9'
        Xset.addModelString("APECROOT",apecroot)

    def load_nxb_model(self):
        Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/rsl_nxb_model_v1.mo')
        #self.nxb_model = AllModels(1,'nxb1')

    def model_bvapec(self,line='w',load_nxb=False,load_FeMXS=False,sigma_fix=False,noMXS=False,redshift=0.1031,multi=False,z_fix=False,spec_num=5,line_model='zgaus'):
        C = Cluster()
        self.model_str = "TBabs*bvapec"
        self.model = Model(self.model_str)
        self.line = line
        if load_nxb == True:
            Xset.restore(f'{self.dropbox}/SSD_backup/PKS_XRISM/model_xcm/rsl_nxb_model_v1.mo')
            self.nxb_model = AllModels(1,'nxb1')
        if load_FeMXS == True:
            Xset.restore(f'{self.dropbox}/SSD_backup/PKS_XRISM/model_xcm/model_MnKab_MXS.xcm')
            self.FeMXS_model = AllModels(1,'FeMXS')
            if noMXS == True:
                self.FeMXS_model.powerlaw.norm = 0
                self.FeMXS_model.powerlaw.PhoIndex = 0
                self.FeMXS_model.powerlaw.norm.frozen = True
                self.FeMXS_model.powerlaw.PhoIndex.frozen = True
        if line == 'w':
            self.model, self.model_str = self.add_model(self.model_str, line_model)
            C.line_manager(state='w')
            self.model.zgauss.LineE = C.line_energy*1e-3
            self.model.zgauss.LineE.frozen = True
            self.error_str = '1.0 2 14 15 16 17 18 20'
        elif line == 'wz':
            if line_model == 'zgaus':
                self.model, self.model_str = self.add_model(self.model_str, 'zgaus+zgaus')
                C.line_manager(state='z')
                self.model.zgauss.LineE = C.line_energy*1e-3
                self.model.zgauss.LineE.frozen = True
                self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
                C.line_manager(state='w')
                self.model.zgauss_4.LineE = C.line_energy*1e-3
                self.model.zgauss_4.LineE.frozen = True
                self.model.zgauss_4.Redshift.link = self.model.bvapec.Redshift
                self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26'
            elif line_model == 'dzgaus':
                self.model, self.model_str = self.add_model(self.model_str, line_model)
                C.line_manager(state='z')
                self.model.dzgaus.LineE_z = C.line_energy*1e-3
                self.model.dzgaus.LineE_z.frozen = True
                self.model.dzgaus.Redshift_z.link = self.model.bvapec.Redshift
                C.line_manager(state='w')
                self.model.dzgaus.LineE_w = C.line_energy*1e-3
                self.model.dzgaus.LineE_w.frozen = True
                self.model.dzgaus.Redshift_w.link = self.model.bvapec.Redshift
                self.model.dzgaus.norm = 1
                self.model.dzgaus.norm.frozen = True
                # self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 42'
                self.error_str = '1.0 2 14 15 16 17 18 22 24 26'
                #self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26'

        elif line == 'None':
            self.error_str = '1.0 2,14,15,16,17,18'
        elif line == 'Heab':
            self.model = Model("TBabs*bvapec+zgauss+zgauss+zgauss+zgauss+zgauss")
            C.line_manager(state='z')
            self.model.zgauss.LineE = C.line_energy*1e-3
            self.model.zgauss.LineE.frozen = True
            self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='w')
            self.model.zgauss_4.LineE = C.line_energy*1e-3
            self.model.zgauss_4.LineE.frozen = True
            self.model.zgauss_4.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='x')
            self.model.zgauss_5.LineE = C.line_energy*1e-3
            self.model.zgauss_5.LineE.frozen = True
            self.model.zgauss_5.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='y')
            self.model.zgauss_6.LineE = C.line_energy*1e-3
            self.model.zgauss_6.LineE.frozen = True
            self.model.zgauss_6.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='Heb1')
            self.model.zgauss_7.LineE = C.line_energy*1e-3
            self.model.zgauss_7.LineE.frozen = True
            self.model.zgauss_7.Redshift.link = self.model.bvapec.Redshift
        elif line == 'All':
            self.model = Model("TBabs*bvapec+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss+zgauss")
            C.line_manager(state='z')
            self.model.zgauss.LineE = C.line_energy*1e-3
            self.model.zgauss.LineE.frozen = True
            self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='w')
            self.model.zgauss_4.LineE = C.line_energy*1e-3
            self.model.zgauss_4.LineE.frozen = True
            self.model.zgauss_4.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='x')
            self.model.zgauss_5.LineE = C.line_energy*1e-3
            self.model.zgauss_5.LineE.frozen = True
            self.model.zgauss_5.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss_5.Sigma.link = self.model.zgauss.Sigma
            C.line_manager(state='y')
            self.model.zgauss_6.LineE = C.line_energy*1e-3
            self.model.zgauss_6.LineE.frozen = True
            self.model.zgauss_6.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss_6.Sigma.link = self.model.zgauss.Sigma
            C.line_manager(state='Lya2')
            self.model.zgauss_7.LineE = C.line_energy*1e-3
            self.model.zgauss_7.LineE.frozen = True
            self.model.zgauss_7.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='Lya1')
            self.model.zgauss_8.LineE = C.line_energy*1e-3
            self.model.zgauss_8.LineE.frozen = True
            self.model.zgauss_8.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss_8.Sigma.link = self.model.zgauss_7.Sigma
            C.line_manager(state='Heb2')
            self.model.zgauss_9.LineE = C.line_energy*1e-3
            self.model.zgauss_9.LineE.frozen = True
            self.model.zgauss_9.Redshift.link = self.model.bvapec.Redshift
            C.line_manager(state='Heb1')
            self.model.zgauss_10.LineE = C.line_energy*1e-3
            self.model.zgauss_10.LineE.frozen = True
            self.model.zgauss_10.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss_10.Sigma.link = self.model.zgauss_9.Sigma
            self.error_str = '1.0 16,66,116,166,2,14,15,17,18,20,22,24,26,30,34,36,38,42,44,50'
        elif line == 'double':
            self.model = Model("TBabs*(bvapec+bvapec)")
        self.line = line
        self.model.TBabs.nH = 0.409
        self.model.TBabs.nH.frozen = True
        self.model.bvapec.kT = 5.2
        self.model.bvapec.He = 0.4
        self.model.bvapec.C  = 0.4
        self.model.bvapec.N  = 0.4
        self.model.bvapec.O  = 0.4
        self.model.bvapec.Ne = 0.4
        self.model.bvapec.Mg = 0.4
        self.model.bvapec.Al = 0.4
        self.model.bvapec.Si = 0.4
        self.model.bvapec.S  = 0.4
        self.model.bvapec.Ar = 0.4
        self.model.bvapec.Ca = 0.4
        self.model.bvapec.Fe = 0.5
        self.model.bvapec.Ni = 0.5
        self.model.bvapec.Redshift = redshift
        self.model.bvapec.Velocity = 150
        self.model.bvapec.norm = 7e-2
        self.model.bvapec.Fe.frozen = False
        self.model.bvapec.Ni.frozen = False

        if line == 'double':
            self.model.bvapec_3.kT = 5.2
            self.model.bvapec_3.He = 0.4
            self.model.bvapec_3.C  = 0.4
            self.model.bvapec_3.N  = 0.4
            self.model.bvapec_3.O  = 0.4
            self.model.bvapec_3.Ne = 0.4
            self.model.bvapec_3.Mg = 0.4
            self.model.bvapec_3.Al = 0.4
            self.model.bvapec_3.Si = 0.4
            self.model.bvapec_3.S  = 0.4
            self.model.bvapec_3.Ar = 0.4
            self.model.bvapec_3.Ca = 0.4
            self.model.bvapec_3.Fe.link = self.model.bvapec.Fe
            self.model.bvapec_3.Ni.link = self.model.bvapec.Ni
            self.model.bvapec_3.Redshift.link = self.model.bvapec.Redshift
            self.model.bvapec_3.Velocity.link = self.model.bvapec.Velocity
            self.model.bvapec_3.norm = 7e-2
            self.error_str = '1.0 18,20,22,24,26,30,34,36,38,42,44,46,50'
        if redshift !=0:
            self.model.bvapec.Redshift.frozen = False
        else:
            self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity.frozen = False  
        if line == 'None':
            if spec_num >= 2:
                AllModels(2).bvapec.Redshift = 0.1028
                AllModels(2).bvapec.Redshift.frozen = False
                self.error_str = '1.0 2 14 15 16 17 18 34'
            if spec_num >= 4:
                AllModels(3).bvapec.Redshift = 0.1028
                AllModels(3).bvapec.Redshift.frozen = False
                AllModels(4).bvapec.Redshift = 0.1028
                AllModels(4).bvapec.Redshift.frozen = False
                self.error_str = '1.0 2 14 15 16 17 18 34 52 70'
            if spec_num >= 5:
                AllModels(5).bvapec.Redshift = 0.1028
                AllModels(5).bvapec.Redshift.frozen = False
                self.error_str = '1.0 2 14 15 16 17 18 34 52 70 88'

        if line == 'w':
            self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss.Sigma = 2e-3
            self.model.zgauss.norm = 1e-4
            self.error_str = '1.0 2 14 15 16 17 18 20 22'
            for si in range(2,spec_num+1):
                AllModels(si).bvapec.Redshift = 0.1028
                AllModels(si).bvapec.Redshift.frozen = False
                AllModels(si).zgauss.Redshift.link = AllModels(si).bvapec.Redshift
            if spec_num >= 2:
                self.error_str = '1.0 2 14 15 16 17 18 20 22 38'
            if spec_num >= 4:
                self.error_str = '1.0 2 14 15 16 17 18 20 22 38 60 82' 
            if spec_num >= 5:
                self.error_str = '1.0 2 14 15 16 17 18 20 22 38 60 82 104' 
            if multi == True:
                print('='*50)
                print('Multi Loaded')
                AllModels(2).bvapec.Redshift = 0.1028
                AllModels(2).bvapec.Redshift.frozen = False
                AllModels(2).zgauss.Redshift.link = AllModels(2).bvapec.Redshift
                AllModels(3).bvapec.Redshift = 0.1028
                AllModels(3).bvapec.Redshift.frozen = False
                AllModels(3).zgauss.Redshift.link = AllModels(3).bvapec.Redshift
                AllModels(4).bvapec.Redshift = 0.1028
                AllModels(4).bvapec.Redshift.frozen = False
                AllModels(4).zgauss.Redshift.link = AllModels(4).bvapec.Redshift
                AllModels(5).bvapec.Redshift = 0.1028
                AllModels(5).bvapec.Redshift.frozen = False
                AllModels(5).zgauss.Redshift.link = AllModels(5).bvapec.Redshift
                AllModels(6).bvapec.Redshift = 0.1028
                AllModels(6).bvapec.Redshift.frozen = False
                AllModels(6).zgauss.Redshift.link = AllModels(6).bvapec.Redshift

                self.error_str = '1.0 2,14,15,16,17,18,20,22,38,60,82'
        elif line == 'wz':
            if line_model == 'zgaus':
                self.model.zgauss.Redshift.link   = self.model.bvapec.Redshift
                self.model.zgauss_4.Redshift.link = self.model.bvapec.Redshift
                self.model.zgauss.Sigma   = 2e-3
                self.model.zgauss.norm    = 1e-5
                self.model.zgauss_4.Sigma = 2e-3
                self.model.zgauss_4.norm  = 1e-4
                for si in range(2,spec_num+1):
                    AllModels(si).bvapec.Redshift = 0.1028
                    AllModels(si).bvapec.Redshift.frozen = False
                    AllModels(si).zgauss.Redshift.link = AllModels(si).bvapec.Redshift
                    AllModels(si).zgauss_4.Redshift.link = AllModels(si).bvapec.Redshift
                if spec_num >= 2:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 42'
                if spec_num >= 4:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 42 68 94'
                if spec_num >= 5:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 42 68 94 120'
            elif line_model == 'dzgaus':
                self.model.dzgaus.Redshift_w.link   = self.model.bvapec.Redshift
                self.model.dzgaus.Redshift_z.link   = self.model.bvapec.Redshift
                self.model.dzgaus.Sigma_w     = 2e-3
                self.model.dzgaus.norm        = 1
                self.model.dzgaus.norm.frozen = True
                self.model.dzgaus.Sigma_z     = 2e-3
                self.model.dzgaus.lnorm_z     = 1e-4
                self.model.dzgaus.ratio       = 2.5
                if z_fix == False:
                    for si in range(2,spec_num+1):
                        AllModels(si).bvapec.Redshift = 0.1028
                        AllModels(si).bvapec.Redshift.frozen = False
                        AllModels(si).dzgaus.Redshift_w.link = AllModels(si).bvapec.Redshift
                        AllModels(si).dzgaus.Redshift_z.link = AllModels(si).bvapec.Redshift
                    if spec_num >= 2:
                        self.error_str = '1.0 2 14 15 16 17 18 22 24 26 43'
                    if spec_num >= 4:
                        self.error_str = '1.0 2 14 15 16 17 18 22 24 26 43 70 97'
                    if spec_num >= 5:
                        self.error_str = '1.0 2 14 15 16 17 18 22 24 26 43 70 97 114'
            if multi == True:
                print('='*50)
                print('Multi Loaded')
                AllModels(1,'FeMXS').constant.factor = 0
                AllModels(1,'FeMXS').constant.factor.frozen = True
                AllModels(1,'FeMXS').constant_14.factor = 0
                AllModels(1,'FeMXS').constant_14.factor.frozen = True
                AllModels(1,'FeMXS').constant_14.factor = 0
                AllModels(1,'FeMXS').constant_14.factor.frozen = True
                AllModels(1,'FeMXS').powerlaw.norm = 0
                AllModels(1,'FeMXS').powerlaw.norm.frozen = True
                AllModels(2).bvapec.Redshift = 0.1028
                AllModels(2).bvapec.Redshift.frozen = False
                AllModels(2).zgauss.Redshift.link = AllModels(2).bvapec.Redshift
                AllModels(2).zgauss_4.Redshift.link = AllModels(2).bvapec.Redshift
                AllModels(2,'FeMXS').constant.factor = 0
                AllModels(2,'FeMXS').constant.factor.frozen = True
                AllModels(2,'FeMXS').constant_14.factor = 0
                AllModels(2,'FeMXS').constant_14.factor.frozen = True
                AllModels(2,'FeMXS').constant_14.factor = 0
                AllModels(2,'FeMXS').constant_14.factor.frozen = True
                AllModels(2,'FeMXS').powerlaw.norm = 0
                AllModels(2,'FeMXS').powerlaw.norm.frozen = True
                AllModels(3).bvapec.Redshift = 0.1028
                AllModels(3).bvapec.Redshift.frozen = False
                AllModels(3).zgauss.Redshift.link = AllModels(3).bvapec.Redshift
                AllModels(3).zgauss_4.Redshift.link = AllModels(3).bvapec.Redshift
                AllModels(3,'FeMXS').constant.factor = 0
                AllModels(3,'FeMXS').constant.factor.frozen = True
                AllModels(3,'FeMXS').constant_14.factor = 0
                AllModels(3,'FeMXS').constant_14.factor.frozen = True
                AllModels(3,'FeMXS').constant_14.factor = 0
                AllModels(3,'FeMXS').constant_14.factor.frozen = True
                AllModels(3,'FeMXS').powerlaw.norm = 0
                AllModels(3,'FeMXS').powerlaw.norm.frozen = True
                AllModels(4).bvapec.Redshift = 0.1028
                AllModels(4).bvapec.Redshift.frozen = False
                AllModels(4).zgauss.Redshift.link = AllModels(4).bvapec.Redshift
                AllModels(4).zgauss_4.Redshift.link = AllModels(4).bvapec.Redshift
                AllModels(4,'FeMXS').constant.factor = 0
                AllModels(4,'FeMXS').constant.factor.frozen = True
                AllModels(4,'FeMXS').constant_14.factor = 0
                AllModels(4,'FeMXS').constant_14.factor.frozen = True
                AllModels(4,'FeMXS').constant_14.factor = 0
                AllModels(4,'FeMXS').constant_14.factor.frozen = True
                AllModels(4,'FeMXS').powerlaw.norm = 0
                AllModels(4,'FeMXS').powerlaw.norm.frozen = True
                AllModels(5).bvapec.Redshift = 0.1028
                AllModels(5).bvapec.Redshift.frozen = False
                AllModels(5).zgauss.Redshift.link = AllModels(5).bvapec.Redshift
                AllModels(5).zgauss_4.Redshift.link = AllModels(5).bvapec.Redshift
                AllModels(5,'FeMXS').constant.factor = 1.2
                AllModels(5,'FeMXS').constant.factor.frozen = False
                AllModels(5,'FeMXS').constant_14.factor = 0.6
                AllModels(5,'FeMXS').constant_14.factor.frozen = False
                AllModels(5,'FeMXS').constant_14.factor = 0.1
                AllModels(5,'FeMXS').constant_14.factor.frozen = False
                AllModels(5,'FeMXS').powerlaw.norm = 0.1
                AllModels(5,'FeMXS').powerlaw.norm.frozen = False
                AllModels(6).bvapec.Redshift = 0.1028
                AllModels(6).bvapec.Redshift.frozen = False
                AllModels(6).zgauss.Redshift.link = AllModels(6).bvapec.Redshift
                AllModels(6).zgauss_4.Redshift.link = AllModels(6).bvapec.Redshift
                AllModels(6,'FeMXS').constant.factor = 1.2
                AllModels(6,'FeMXS').constant.factor.frozen = False
                AllModels(6,'FeMXS').constant_14.factor = 0.6
                AllModels(6,'FeMXS').constant_14.factor.frozen = False
                AllModels(6,'FeMXS').constant_14.factor = 0.1
                AllModels(6,'FeMXS').constant_14.factor.frozen = False
                AllModels(6,'FeMXS').powerlaw.norm = 0.1
                AllModels(6,'FeMXS').powerlaw.norm.frozen = False
            if sigma_fix == True:
                self.model.zgauss_4.Sigma.link = self.model.zgauss.Sigma
        elif line == 'Heab':
            self.model.zgauss.Redshift.link   = self.model.bvapec.Redshift
            self.model.zgauss_4.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss.Sigma = 2e-3
            self.model.zgauss.norm   = 1e-5
            self.model.zgauss_4.Sigma = 2e-3
            self.model.zgauss_4.norm = 1e-4
            self.model.zgauss_5.Sigma = 2e-3
            self.model.zgauss_5.norm = 1e-5
            self.model.zgauss_6.Sigma = 2e-3
            self.model.zgauss_6.norm = 1e-5
            self.model.zgauss_7.Sigma = 2e-3
            self.model.zgauss_7.norm = 1e-5
        elif line == 'All':
            self.model.zgauss.Sigma = 2e-3
            self.model.zgauss.norm   = 1e-5
            self.model.zgauss_4.Sigma = 2e-3
            self.model.zgauss_4.norm = 1e-4
            self.model.zgauss_5.Sigma = 2e-3
            self.model.zgauss_5.norm = 1e-5
            self.model.zgauss_6.Sigma = 2e-3
            self.model.zgauss_6.norm = 1e-5
            self.model.zgauss_7.Sigma = 2e-3
            self.model.zgauss_7.norm = 1e-5
            self.model.zgauss_8.Sigma = 2e-3
            self.model.zgauss_8.norm = 1e-5
            self.model.zgauss_9.Sigma = 2e-3
            self.model.zgauss_9.norm = 1e-5
            self.model.zgauss_10.Sigma = 2e-3
            self.model.zgauss_10.norm = 1e-5
            self.model.zgauss_5.Sigma.link = self.model.zgauss.Sigma
            self.model.zgauss_6.Sigma.link = self.model.zgauss.Sigma
            self.model.zgauss_7.Redshift.link = self.model.bvapec.Redshift
            self.model.zgauss_8.Sigma.link = self.model.zgauss_7.Sigma
            self.model.zgauss_10.Sigma.link = self.model.zgauss_9.Sigma
            for si in range(2,spec_num+1):
                AllModels(si).bvapec.Redshift = 0.1028                
                AllModels(si).bvapec.Redshift.frozen = False
                AllModels(si).zgauss.Redshift.link   = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_4.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_5.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_6.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_7.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_8.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_9.Redshift.link = AllModels(si).bvapec.Redshift
                AllModels(si).zgauss_10.Redshift.link = AllModels(si).bvapec.Redshift

            self.error_str = '1.0 16,66,116,166,2,14,15,17,18,20,22,24,26,30,34,36,38,42,44,50'
            if sigma_fix == True:
                self.model.zgauss_4.Sigma.link = self.model.zgauss.Sigma
                self.model.zgauss_5.Sigma.link = self.model.zgauss.Sigma
                self.model.zgauss_6.Sigma.link = self.model.zgauss.Sigma
                self.model.zgauss_7.Sigma.link = self.model.zgauss.Sigma
        if load_nxb == True:
            self.nxb_model.gaussian_9.Sigma.frozen = True # par19
            self.nxb_model.gaussian_18.Sigma.frozen = True # par46
        AllModels.show()      

    def _init_models(self, model_str: str, spec_num=8):
        """Build center / outer / exMXS models and store in self.models."""
        self.model_center = Model(model_str, modName="", sourceNum=1)
        self.model_outer = Model(model_str, modName="outer", sourceNum=2)
        self.model_exMXS = Model(model_str, modName="exMXS", sourceNum=3)
        self.models = {}
        self.models["center"] = {}
        self.models["outer"]  = {}
        self.models["exMXS"]  = {}
        for i in range(1, spec_num+1):
            self.models["center"][str(i)] = AllModels(i)
            self.models["outer"][str(i)]  = AllModels(i, "outer")
            self.models["exMXS"][str(i)]  = AllModels(i, "exMXS")

    def _freeze_line(
        self,
        energy_keV: float,
        *,
        sigma: float = 2e-3,
        norm: float = 1e-4,
        comp: str = "zgauss",
        regions=None,
    ) -> None:
        """各リージョン・各スペクトルで Gaussian を固定する。"""
        for i in range(1,9):
            if regions is None:
                regions = ["", "outer", "exMXS"]
                for region in regions:
                    m = AllModels(i, modName=region)
                    if hasattr(m, comp):
                        if i == 1:
                            getattr(m, comp).LineE        = energy_keV
                            getattr(m, comp).LineE.frozen = True
                            getattr(m, comp).Sigma        = sigma
                            getattr(m, comp).norm         = norm

    def _freeze_line_dzgaus(
        self,
        w_energy_keV: float,
        z_energy_keV: float,
        *,
        sigma: float = 2e-3,
        norm: float = 1e-4,
        comp: str = "dzgaus",
        regions=None,
        spec_num = 8
    ) -> None:
        """dzgausのLineEを固定する。"""
        for i in range(1,spec_num+1):
            if regions is None:
                regions = ["", "outer", "exMXS"]
                for region in regions:
                    m = AllModels(i, modName=region)
                    if hasattr(m, comp):
                        if i == 1:
                            getattr(m, comp).LineE_z        = z_energy_keV
                            getattr(m, comp).LineE_w        = w_energy_keV
                            getattr(m, comp).LineE_z.frozen = True
                            getattr(m, comp).LineE_w.frozen = True
                            getattr(m, comp).Sigma_w        = sigma
                            getattr(m, comp).Sigma_z        = sigma
                            getattr(m, comp).lnorm_z        = norm
                            getattr(m, comp).norm           = 1
                            getattr(m, comp).norm.frozen    = True

    def _link_redshift(self, comp: str = "zgauss") -> None:
        """comp.Redshift を対応する bvapec.Redshift にリンク。"""
        for region_models in self.models.values():
            for mdl in region_models.values():
                if hasattr(mdl, comp):
                    mdl.bvapec.Redshift.frozen = False
                    mdl.bvapec.Redshift.values = 0.1028
                    getattr(mdl, comp).Redshift.link = mdl.bvapec.Redshift

    def _freeze_redshift(self, comp: str = "zgauss") -> None:
        """comp.Redshift を対応する bvapec.Redshift にリンク。"""
        for region_models in self.models.values():
            for mdl in region_models.values():
                if hasattr(mdl, comp):
                    mdl.bvapec.Redshift.frozen = True

    def _free_redshift(self, comp: str = "zgauss") -> None:
        """comp.Redshift を対応する bvapec.Redshift にリンク。"""
        for region_models in self.models.values():
            for mdl in region_models.values():
                if hasattr(mdl, comp):
                    mdl.bvapec.Redshift.frozen = False

    def _link_redshift_dzgaus(self, comp: str = "dzgaus") -> None:
        """comp.Redshift を対応する bvapec.Redshift にリンク。"""
        for region_models in self.models.values():
            for mdl in region_models.values():
                if hasattr(mdl, comp):
                    mdl.bvapec.Redshift.frozen = False
                    mdl.bvapec.Redshift.values = 0.1028
                    getattr(mdl, comp).Redshift_w.link = mdl.bvapec.Redshift
                    getattr(mdl, comp).Redshift_z.link = mdl.bvapec.Redshift

    def _link_sigma(self, comp_mother: str = "zgauss", comp_sun: str = "zgauss_4") -> None:
        """comp.Redshift を対応する bvapec.Redshift にリンク。"""
        for region_models in self.models.values():
            for mdl in region_models.values():
                if hasattr(mdl, comp_mother):
                    getattr(mdl, comp_sun).Sigma.link = getattr(mdl, comp_mother).Sigma

    def _refresh_models(self) -> None:
        """self.models を *今 XSPEC に存在する* Model で作り直す。"""
        self.models = {"center": {}, "outer": {}, "exMXS": {}}
        for i in range(1, 9):                # ← 8 データセット想定
            self.models["center"][str(i)] = AllModels(i)          # modName=""
            self.models["outer"][str(i)]  = AllModels(i, "outer")
            self.models["exMXS"][str(i)]  = AllModels(i, "exMXS")

    def _init_bvapec(
        self,
        *,
        kT: float = 4.56278,
        Z: float = 0.4,
        Fe: float = 0.454,
        Ni: float = 0.4826,
        redshift: float = 0.1028,
        regions = None
    ) -> None:
        """各リージョン・各スペクトルの bvapec に初期値を流し込む。"""
        elems = ("He", "C", "N", "O", "Ne", "Mg", "Al",
                 "Si", "S", "Ar", "Ca")

        for i in range(1,9):
            if regions is None:
                regions = ["", "outer", "exMXS"]
                for region in regions:
                    m = AllModels(i, modName=region)
                    if i == 1:
                        m.TBabs.nH        = 0.409
                        m.TBabs.nH.frozen = True

                        m.bvapec.kT = kT
                        for el in elems:
                            setattr(m.bvapec, el, Z)

                        m.bvapec.Fe       = Fe
                        m.bvapec.Ni       = Ni
                        m.bvapec.Redshift = redshift
                        m.bvapec.Velocity = 100

                        m.bvapec.Fe.frozen       = False
                        m.bvapec.Ni.frozen       = False
                        m.bvapec.Redshift.frozen = False
                        m.bvapec.Velocity.frozen = False
                        m.bvapec.norm = 3.26e-2

                        if region == "outer" or region == "exMXS":
                            m.bvapec.kT = 7.5


    def _freeze_double_bvapec(self, Z: float = 0.4, regions = None):
        elems = ("He", "C", "N", "O", "Ne", "Mg", "Al",
                 "Si", "S", "Ar", "Ca")
        for i in range(1,9):
            if regions is None:
                regions = ["", "outer", "exMXS"]
                for region in regions:
                    m = AllModels(i, modName=region)
                    for el in elems:
                        setattr(m.bvapec_3, el, Z)

                    m.bvapec_3.Fe.link       = m.bvapec.Fe
                    m.bvapec_3.Ni.link       = m.bvapec.Ni
                    m.bvapec_3.Redshift.link = m.bvapec.Redshift
                    m.bvapec_3.Velocity.link = m.bvapec.Velocity
                    m.bvapec_3.norm = 3.26e-3

    def _freeze_exMXS(self,spec_num=8):
        """exMXS モデルのすべてのパラメータを outer モデルにリンクする"""
        for i in range(1, spec_num+1):
            m_ex = AllModels(i, modName="exMXS")
            m_outer = AllModels(i, modName="outer")
            for comp_name in m_ex.componentNames:
                try:
                    comp_ex = getattr(m_ex, comp_name)
                    comp_outer = getattr(m_outer, comp_name)
                except AttributeError:
                    continue 

                for par_name in comp_ex.parameterNames:
                    try:
                        par_ex = getattr(comp_ex, par_name)
                        par_outer = getattr(comp_outer, par_name)
                        par_ex.link = par_outer
                    except AttributeError:
                        continue
                
    def model_bvapec_ssm(self,line='wz',load_nxb=False,load_FeMXS=False,sigma_fix=False,noMXS=False,redshift=0.1031,multi=False,spec_num=8,line_model='zgauss'):
        C = Cluster()
        self.model_str = "TBabs*bvapec"
        self.line = line
        if line == "None":
            self._init_models(self.model_str)
            self.error_str = '1.0 2 14 15 16 17 18 34 52 70 88 106 124 142 outer:2 outer:14  outer:15  outer:16  outer:17  outer:18  outer:34  outer:52  outer:70  outer:88  outer:106  outer:124  outer:142'
            self._init_bvapec(redshift=redshift)
        elif line == 'double':
            # double-bvapec model
            self.model_str = "TBabs*(bvapec+bvapec)"
            self._init_models(self.model_str)
            self.error_str = '1.0 2 14 15 16 17 18 19 35 51 86 121 156 191 226 261 outer:2 outer:14  outer:15  outer:16  outer:17  outer:18  outer:19  outer:35  outer:51  outer:86  outer:121  outer:156  outer:191  outer:226  outer:261'
            self._init_bvapec(redshift=redshift)
            self._freeze_double_bvapec()
        elif line == 'w':
            self.model_str += f"+{line_model}"
            self._init_models(self.model_str)
            C.line_manager(state='w')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-4,comp=line_model)
            self._link_redshift(line_model)
            self.error_str = '1.0 16 38 60 82 104 126 148 170 2 14 15 17 18 20 22 outer:16 outer:38 outer:60 outer:82 outer:104 outer:126 outer:148 outer:170 outer:2 outer:14  outer:15 outer:17  outer:18  outer:20  outer:22 '
            self._init_bvapec(redshift=redshift)
        elif line == 'wz':
            if line_model == 'zgauss':
                self.model_str += f"+{line_model}+{line_model}"
                self._init_models(self.model_str,spec_num)
                C.line_manager(state='z')
                self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=line_model)
                self._link_redshift(line_model)
                C.line_manager(state='w')
                self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_4')
                self._link_redshift(comp=f'{line_model}_4')
                if spec_num == 8:
                    self.error_str = '1.0 16 42 68 94 120 146 172 198 2 14 15 17 18 20 22 24 26 outer:16 outer:42  outer:68  outer:94  outer:120  outer:146  outer:172  outer:198 outer:2  outer:14  outer:15 outer:17  outer:18  outer:20  outer:22  outer:24  outer:26 '
                else:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 42 outer:2  outer:14  outer:15  outer:16  outer:17  outer:18  outer:20  outer:22  outer:24  outer:26  outer:42'
                self._init_bvapec(redshift=redshift)
            if line_model == 'dzgaus':
                self.model_str += f"+{line_model}"
                self._init_models(self.model_str,spec_num)
                C.line_manager(state='z')
                z_line_e = C.line_energy*1e-3
                C.line_manager(state='w')
                w_line_e = C.line_energy*1e-3
                self._freeze_line_dzgaus(w_energy_keV=w_line_e,z_energy_keV=z_line_e,sigma=2e-3,norm=1e-5,comp=line_model,spec_num=spec_num)
                self._link_redshift_dzgaus(line_model)
                if spec_num == 8:
                    self.error_str = '1.0 16 43 70 97 124 151 178 205 2 14 15 17 18 20 22 24 26 outer:16  outer:43  outer:70  outer:97  outer:124  outer:151  outer:178  outer:205  outer:2  outer:14  outer:15  outer:17  outer:18  outer:20  outer:22  outer:24  outer:26'
                elif spec_num == 6:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 43 70 97 124 151 outer:2  outer:14  outer:15  outer:16  outer:17  outer:18  outer:20  outer:22  outer:24  outer:26  outer:43  outer:70  outer:97  outer:124  outer:151'
                else:
                    self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 43 outer:2 outer:14 outer:15 outer:16 outer:17 outer:18 outer:20 outer:22 outer:24 outer:26 outer:43'
                self._init_bvapec(redshift=redshift)
        elif line == "all":
            self.model_str += f"+{line_model}+{line_model}+{line_model}+{line_model}+{line_model}+{line_model}+{line_model}+{line_model}"
            self._init_models(self.model_str,spec_num)
            C.line_manager(state='z')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=line_model)
            self._link_redshift(line_model)
            C.line_manager(state='w')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_4')
            self._link_redshift(comp=f'{line_model}_4')
            C.line_manager(state='x')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_5')
            self._link_sigma(comp_mother=line_model, comp_sun=f'{line_model}_5')
            C.line_manager(state='y')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_6')
            self._link_sigma(comp_mother=line_model, comp_sun=f'{line_model}_6')
            C.line_manager(state='Lya2')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_7')
            C.line_manager(state='Lya1')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_8')
            self._link_sigma(comp_mother=f'{line_model}_7', comp_sun=f'{line_model}_8')
            C.line_manager(state='Heb2')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_9')
            C.line_manager(state='Heb1')
            self._freeze_line(C.line_energy*1e-3,sigma=2e-3,norm=1e-5,comp=f'{line_model}_10')
            self._link_sigma(comp_mother=f'{line_model}_9', comp_sun=f'{line_model}_10')
            self._init_bvapec(redshift=redshift)
            self.error_str = '1.0 2 14 15 16 17 18 20 22 24 26 30 34 36 38 42 44 46 50 66 116 166 216 266 316 366 outer:2 outer:14 outer:15 outer:16 outer:17 outer:18 outer:20 outer:22 outer:24 outer:26 outer:30 outer:34 outer:36 outer:38 outer:42 outer:44 outer:46 outer:50 outer:66 outer:116 outer:166 outer:216 outer:266 outer:316 outer:366'
            # if sigma_fix == True:
            #     for si in range(1,9):
            #         for region in ['', 'outer', 'exMXS']:
            #             AllModels(si,modName=region).zgauss_4.Sigma.link = AllModels(si,modName=region).zgauss.Sigma
            #             AllModels(si,modName=region).zgauss_5.Sigma.link = AllModels(si,modName=region).zgauss.Sigma
            #             AllModels(si,modName=region).zgauss_6.Sigma.link = AllModels(si,modName=region).zgauss.Sigma
            #             AllModels(si,modName=region).zgauss

        red_center = [0.102757, 0.102593, 0.10296, 0.10256, 0.10309, 0.10280, 0.10319, 0.10310]
        red_outer = [0.10318, 0.103369, 0.1026, 0.10317, 0.1026, 0.10318, 0.1026, 0.10283]
        if spec_num == 8:
            for si in range(1,9):
                for region in ['', 'outer', 'exMXS']:
                    if region == '':
                        redshift = red_center[si-1]
                    else:
                        redshift = red_outer[si-1]
                    AllModels(int(si),modName=region).bvapec.Redshift = redshift
                    AllModels(int(si),modName=region).bvapec.Redshift.frozen = False
                    AllModels(int(si),modName=region).bvapec.Redshift.link = AllModels(int(si),modName=region).bvapec.Redshift
                    AllModels(int(si),modName=region).bvapec.Redshift.link = AllModels(int(si),modName=region).bvapec.Redshift
        else:
            for si in range(1,3):
                for region in ['', 'outer', 'exMXS']:
                    AllModels(int(si),modName=region).bvapec.Redshift = redshift
                    AllModels(int(si),modName=region).bvapec.Redshift.frozen = False
                    AllModels(int(si),modName=region).bvapec.Redshift.link = AllModels(int(si),modName=region).bvapec.Redshift
                    AllModels(int(si),modName=region).bvapec.Redshift.link = AllModels(int(si),modName=region).bvapec.Redshift

        if load_nxb == True:
            Xset.restore(f'{self.dropbox}/SSD_backup/PKS_XRISM/model_xcm/rsl_nxb_model_v1_num4.mo')
            self.nxb_model_center = AllModels(1,'nxb1')
            Xset.restore(f'{self.dropbox}/SSD_backup/PKS_XRISM/model_xcm/rsl_nxb_model_v1_num5.mo')
            self.nxb_model_outer = AllModels(2,'nxb2')
            self.nxb_model_center.gaussian_9.Sigma.frozen  = True   # par19
            self.nxb_model_center.gaussian_18.Sigma.frozen = True   # par46
            self.nxb_model_outer.gaussian_9.Sigma.frozen   = True   # par19
            self.nxb_model_outer.gaussian_18.Sigma.frozen  = True   # par46

        self._freeze_exMXS(spec_num)
        AllModels.show()      

    def model_bvapec_for_simulation(self, Nickel=False, gaus='dzgaus', line='wz'):
        C = Cluster()
        if gaus == 'dzgaus':
            self.model = Model("bvapec+dzgaus")
            C.line_manager(state='z')
            self.model.dzgaus.LineE_z = C.line_energy*1e-3
            self.model.dzgaus.LineE_z.frozen = True
            self.model.dzgaus.Redshift_z.link = self.model.bvapec.Redshift
            C.line_manager(state='w')
            self.model.dzgaus.LineE_w = C.line_energy*1e-3
            self.model.dzgaus.LineE_w.frozen = True
            self.model.dzgaus.Redshift_w.link = self.model.bvapec.Redshift
            self.model.dzgaus.norm = 1
            self.model.dzgaus.norm.frozen = True
        elif gaus == 'zgaus':
            if line == 'w':
                self.model = Model("bvapec+zgauss")
                C.line_manager(state='w')
                self.model.zgauss.LineE = C.line_energy*1e-3
                self.model.zgauss.LineE.frozen = True
                self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
            elif line == 'wz':
                self.model = Model("bvapec+zgauss+zgauss")
                C.line_manager(state='z')
                self.model.zgauss.LineE = C.line_energy*1e-3
                self.model.zgauss.LineE.frozen = True
                self.model.zgauss.Redshift.link = self.model.bvapec.Redshift
                C.line_manager(state='w')
                self.model.zgauss_3.LineE = C.line_energy*1e-3
                self.model.zgauss_3.LineE.frozen = True
                self.model.zgauss_3.Redshift.link = self.model.bvapec.Redshift
        self.model.bvapec.kT = 3.5
        self.model.bvapec.He = 0.4
        self.model.bvapec.C  = 0.4
        self.model.bvapec.N  = 0.4
        self.model.bvapec.O  = 0.4
        self.model.bvapec.Ne = 0.4
        self.model.bvapec.Mg = 0.4
        self.model.bvapec.Al = 0.4
        self.model.bvapec.Si = 0.4
        self.model.bvapec.S  = 0.4
        self.model.bvapec.Ar = 0.4
        self.model.bvapec.Ca = 0.4
        self.model.bvapec.Fe = 0.5
        self.model.bvapec.Ni = 0.4
        self.model.bvapec.Velocity = 150
        self.model.bvapec.norm = 7e-2
        self.model.bvapec.Fe.frozen = False
        if Nickel == True:
            self.model.bvapec.Ni.frozen = False
            self.error_str = '1.0 1 13 14 16 17 19 21 23 25'
            if gaus == "zgaus":
                self.error_str = '1.0 1 13 14 16 17 19 21'

        else:
            self.model.bvapec.Ni.frozen = True
            self.error_str = '1.0 1 13 16 17 19 21 23 25'
            if gaus == "zgaus":
                self.error_str = '1.0 1 13 16 17 19 21'
        
        self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity.frozen = False  
        if gaus == 'dzgaus':
            self.model.dzgaus.Sigma_w     = 2e-3
            self.model.dzgaus.Sigma_z     = 2e-3
            self.model.dzgaus.lnorm_z     = 1e-4
            self.model.dzgaus.ratio       = 2.5
        elif gaus == 'zgaus':
            self.model.zgauss.Sigma = 2e-3
            self.model.zgauss.norm   = 1e-5
        AllModels.show()      

    def model_55Fe(self,load_nxb=False,load_FeMXS=False,noMXS=False):
        if load_FeMXS == True:
            Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKab_MXS_raw.xcm')
            self.model = AllModels(1)
            if noMXS == True:
                self.model.powerlaw.norm = 0
                self.model.powerlaw.PhoIndex = 0
                self.model.powerlaw.norm.frozen = True
                self.model.powerlaw.PhoIndex.frozen = True
            self.model.expcutoff.tau.frozen = False
        if load_nxb == True:
            Xset.restore('/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/rsl_nxb_model_v1.mo')
            self.nxb_model = AllModels(1,'nxb1')
            self.nxb_model.gaussian_9.Sigma.frozen = True  # par19
            self.nxb_model.gaussian_18.Sigma.frozen = True # par46
        AllModels.show()      

    def bvapec_fix_some_param(self,fix=True):
        if fix == True:
            self.model.bvapec.kT.frozen = True
            self.model.bvapec.Fe.frozen = True
            self.model.bvapec.Ni.frozen = True
            self.model.bvapec.Redshift.frozen = True
            self.model.bvapec.Velocity.frozen = True
        else:
            self.model.bvapec.kT.frozen = False
            self.model.bvapec.Fe.frozen = False
            self.model.bvapec.Ni.frozen = False
            self.model.bvapec.Redshift.frozen = False
            self.model.bvapec.Velocity.frozen = False

    def nxb_fix_parameter(self,fix=True):
        if fix == True:
            self.nxb_model.constant.factor.frozen = True
            self.nxb_model.powerlaw.PhoIndex.frozen = True
            self.nxb_model.gaussian_19.Sigma.frozen = True
            self.nxb_model.gaussian_20.Sigma.frozen = True
            self.nxb_model.gaussian_21.Sigma.frozen = True
        else:
            self.nxb_model.constant.factor.frozen = False
            self.nxb_model.powerlaw.PhoIndex.frozen = False
            self.nxb_model.gaussian_19.Sigma.frozen = False
            self.nxb_model.gaussian_20.Sigma.frozen = False
            self.nxb_model.gaussian_21.Sigma.frozen = False

    def nxb_fix_parameter_ssm(self,fix=True):
        if fix == True:
            self.nxb_model_center.constant.factor.frozen   = True
            self.nxb_model_center.powerlaw.PhoIndex.frozen = True
            self.nxb_model_center.gaussian_19.Sigma.frozen = True
            self.nxb_model_center.gaussian_20.Sigma.frozen = True
            self.nxb_model_center.gaussian_21.Sigma.frozen = True

            self.nxb_model_outer.constant.factor.frozen   = True
            self.nxb_model_outer.powerlaw.PhoIndex.frozen = True
            self.nxb_model_outer.gaussian_19.Sigma.frozen = True
            self.nxb_model_outer.gaussian_20.Sigma.frozen = True
            self.nxb_model_outer.gaussian_21.Sigma.frozen = True
        else:
            self.nxb_model_center.constant.factor.frozen   = False
            self.nxb_model_center.powerlaw.PhoIndex.frozen = False
            self.nxb_model_center.gaussian_19.Sigma.frozen = False
            self.nxb_model_center.gaussian_20.Sigma.frozen = False
            self.nxb_model_center.gaussian_21.Sigma.frozen = False

            self.nxb_model_outer.constant.factor.frozen   = False
            self.nxb_model_outer.powerlaw.PhoIndex.frozen = False
            self.nxb_model_outer.gaussian_19.Sigma.frozen = False
            self.nxb_model_outer.gaussian_20.Sigma.frozen = False
            self.nxb_model_outer.gaussian_21.Sigma.frozen = False

    def model_rsapec(self):
        import rsapec_xspec
        self.model = Model('TBabs*pyvapecrs')
        # self.model.TBabs.nH = 0.409
        # self.model.TBabs.nH.frozen = True
        # self.model.pyvapecrs.kT = 5.2
        # self.model.pyvapecrs.He = 0.4
        # self.model.pyvapecrs.C  = 0.4
        # self.model.pyvapecrs.N  = 0.4
        # self.model.pyvapecrs.O  = 0.4
        # self.model.pyvapecrs.Ne = 0.4
        # self.model.pyvapecrs.Mg = 0.4
        # self.model.pyvapecrs.Al = 0.4
        # self.model.pyvapecrs.Si = 0.4
        # self.model.pyvapecrs.S  = 0.4
        # self.model.pyvapecrs.Ar = 0.4
        # self.model.pyvapecrs.Ca = 0.4
        # self.model.pyvapecrs.Fe = 0.5
        # self.model.pyvapecrs.Ni = 0.5
        # self.model.pyvapecrs.Redshift = 0.1031
        # self.model.pyvapecrs.Velocity = 150
        # self.model.pyvapecrs.norm = 7e-2
        # self.model.pyvapecrs.Fe.frozen = False
        # self.model.pyvapecrs.Ni.frozen = False
        # self.model.pyvapecrs.Redshift.frozen = True
        # self.model.pyvapecrs.Velocity.frozen = False

    def fitting_spectrum_mod(self,spec, rmf, arf, multiresp, modname, line='wz', rng=(2.0,17.0), rebin=1, line_model="zgauss"):
        '''
        単一データのfitting用の関数
        '''
        #modname = f'{identifier}_{region}_{line}_rebin_{rebin}'
        modname = str(modname)
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,rng=rng, rebin=rebin)
        self.model_bvapec(line=line,load_nxb=True,sigma_fix=False,spec_num=1,line_model=line_model)
        self.load_apecroot(line)
        self.thermal_broadening('yes')
        self.bvapec_fix_some_param(fix=True)
        self.fit_error(error=self.error_str,error_calc=False,detector='nxb1')
        self.bvapec_fix_some_param(fix=False)
        self.nxb_fix_parameter(fix=True)
        self.fit_error(error=self.error_str,error_calc=True,detector='nxb1')
        Plot('data delc')
        self.set_xydata_multi([modname])
        self.result_pack()
        self.savemod_multi(modname)
        self.plotting_multi(modname)



    def fitting_pixel_by_pixel(self, pixel_num=["00", "17", "18", "35"], line_model="dzgaus"):
        for pix in pixel_num:
            spec      = f"1000_PIXEL_{pix}_merged_b1.pi"
            rmf       = f"1000_pix{pix}_L_without_Lp.rmf"
            arf       = f"1000_pix{pix}_image_1p8_8keV_1e7.arf"
            multiresp = f'{self.dropbox}/SSD_backup/PKS_XRISM/rmfarf/newdiag60000.rmf'
            modname = f"1000_pixel_{pix}_rebin_1_{line_model}"
            self.fitting_spectrum_mod(spec, rmf, arf,  multiresp, modname, line_model=line_model)

    def fitting_spectrum(self,identifier='1000', region='center', line='wz', rng=(2.0,17.0), rebin=1):
        '''
        単一データのfitting用の関数
        '''
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}'
        self.load_spectrum(spec=f'{identifier}_{region}_merged_b1.pi',rmf=f'{identifier}_{region}_L_without_Lp.rmf',arf=f'{identifier}_{region}_image_1p8_8keV_1e7.arf',multiresp=f'{self.dropbox}/SSD_backup/PKS_XRISM/rmfarf/newdiag60000.rmf',rng=rng, rebin=rebin)
        self.model_bvapec(line=line,load_nxb=True,sigma_fix=False,spec_num=1)
        self.load_apecroot(line)
        self.thermal_broadening('yes')
        self.bvapec_fix_some_param(fix=True)
        self.fit_error(error=self.error_str,error_calc=False,detector='nxb1')
        self.bvapec_fix_some_param(fix=False)
        self.nxb_fix_parameter(fix=True)
        self.fit_error(error=self.error_str,error_calc=True,detector='nxb1')
        self.set_xydata_multi([modname])
        self.result_pack()
        self.savemod_multi(modname)
        self.plotting_multi(modname)

    def fitting_spectrum_each_reg(self,identifier='1000',rng=(2.0,17.0), rebin=1):
        for region in ['all','center','outer']:
            self.load_spectrum(spec=f'{identifier}_{region}_merged_b1.pi',rmf=f'{identifier}_{region}_L_without_Lp.rmf',arf=f'{identifier}_{region}_image_1p8_8keV_1e7.arf',multiresp=f'{self.dropbox}/SSD_backup/rmfarf/newdiag60000.rmf',rng=rng, rebin=rebin)
            for line in ['None','w','wz']:
                modname = f'{identifier}_{region}_{line}_rebin_{rebin}'
                self.model_bvapec(line=line,load_nxb=True,sigma_fix=False,spec_num=1)
                self.load_apecroot(line)
                self.bvapec_fix_some_param(fix=True)
                self.fit_error(error=self.error_str,error_calc=False,detector='nxb1')
                self.bvapec_fix_some_param(fix=False)
                self.nxb_fix_parameter(fix=True)
                self.fit_error(error=self.error_str,error_calc=True,detector='nxb1')
                self.result_pack()
                self.savemod(modname)
                self.plotting(modname)

    def simulation_fitting(self, spec, rmf, line='wz', modname='test', v=0, Nickel=False, gaus='dzgaus'):
        self.load_spectrum(spec=spec,rmf=rmf,arf=None,rng=(6.5,8.0))
        self.model_bvapec_for_simulation(Nickel=Nickel, gaus=gaus, line=line)
        self.load_apecroot(line)
        self.model.bvapec.Fe.frozen = True
        self.model.bvapec.Ni.frozen = True
        self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity = v
        self.model.bvapec.Velocity.frozen = True
        if gaus == 'dzgaus':
            self.model.dzgaus.Sigma_w = 2e-3
            self.model.dzgaus.Sigma_z = 2e-3
            self.model.dzgaus.Sigma_w.frozen = True
            self.model.dzgaus.Sigma_z.frozen = True
        elif gaus == 'zgaus':
            self.model.zgauss.Sigma = 2e-3
            self.model.zgauss.norm   = 1e-5
            self.model.zgauss.norm.frozen = False
            self.model.zgauss.Sigma.frozen = True
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.bvapec.Fe.frozen = False
        if Nickel == True:
            self.model.bvapec.Ni.frozen = False
        self.model.bvapec.Velocity.frozen = False
        if gaus == 'dzgaus':
            self.model.dzgaus.Sigma_w.frozen = False
            self.model.dzgaus.Sigma_z.frozen = False
        elif gaus == 'zgaus':
            self.model.zgauss.Sigma.frozen = False
        self.line = line
        self.fit_error(error=self.error_str,error_calc=True,detector=False)
        self.result_pack()
        self.set_xydata_multi([modname])
        self.savemod_multi(modname)
        self.plotting_multi(modname, x_rng=(6.611,6.732))

    def simulation_fitting_dzgaus(self, spec, rmf, line='wz', modname='test', v=0, Nickel=False):
        self.load_spectrum(spec=spec,rmf=rmf,arf=None,rng=(6.5,8.0))
        self.model_bvapec_for_simulation(Nickel=Nickel, gaus="zgaus", line=line)
        self.load_apecroot(line)
        self.model.bvapec.Fe.frozen = True
        self.model.bvapec.Ni.frozen = True
        self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity = v
        self.model.bvapec.Velocity.frozen = True
        self.model.zgauss.Sigma = 2e-3
        self.model.zgauss.norm   = 1e-5
        self.model.zgauss.norm.frozen = False
        self.model.zgauss.Sigma.frozen = True
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.bvapec.Fe.frozen = False
        if Nickel == True:
            self.model.bvapec.Ni.frozen = False
        self.model.bvapec.Velocity.frozen = False
        self.model.zgauss.Sigma.frozen = False
        self.line = line
        self.fit_error(error=self.error_str,error_calc=True,detector=False)
        ratio = self.model.zgauss.norm.value / self.model.zgauss_4.norm.value
        kT = self.model.bvapec.kT.value
        apec_norm = self.model.bvapec.norm.value
        Fe = self.model.bvapec.Fe.value
        Ni = self.model.bvapec.Ni.value
        velocity = self.model.bvapec.Velocity.value
        z_sigma = self.model.zgauss.Sigma.value
        z_norm = self.model.zgauss.norm.value
        w_sigma = self.model.zgauss_4.Sigma.value
        self.model_bvapec_for_simulation(Nickel=Nickel, gaus="dzgaus", line=line)
        self.model.bvapec.kT = kT
        self.model.bvapec.norm = apec_norm
        self.model.bvapec.Fe = Fe
        self.model.bvapec.Ni = Ni
        self.model.bvapec.Velocity = velocity
        self.model.dzgaus.Sigma_z = z_sigma
        self.model.dzgaus.Sigma_w = w_sigma
        self.model.dzgaus.lnorm_z = z_norm
        self.model.dzgaus.ratio = ratio
        self.model.bvapec.Redshift.frozen = True
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.result_pack()
        self.set_xydata_multi([modname])
        self.savemod_multi(modname)
        self.plotting_multi(modname, x_rng=(6.611,6.732))

    def fit_heasim_spec_perseus(self, spec, modname, v=150):
        rmf = "/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/heasim_response/obs234_HP_fov_all_det.rmf"
        arf = "/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/heasim_response/HP_obs23_all_det_diffuse_fudge.arf"
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,rng=(6.51,6.79))
        Plot("data")
        self.model_bvapec_for_simulation(Nickel=False, gaus="zgaus", line="wz")
        self.load_apecroot("wz")
        self.line = "wz"
        ratio = 2.3
        kT = 3.8
        apec_norm = 88
        Fe = 0.52
        Ni = 0.4
        velocity = v
        z_sigma = 2e-3
        z_norm = 4e-2
        w_sigma = 5e-3
        self.model_bvapec_for_simulation(Nickel=False, gaus="dzgaus", line="wz")
        self.model.bvapec.kT = kT
        self.model.bvapec.norm = apec_norm
        self.model.bvapec.Fe = Fe
        self.model.bvapec.Ni = Ni
        self.model.bvapec.Velocity = velocity
        self.model.dzgaus.Sigma_z = z_sigma
        self.model.dzgaus.Sigma_w = w_sigma
        self.model.dzgaus.lnorm_z = z_norm
        self.model.dzgaus.ratio = ratio
        self.model.dzgaus.ratio.frozen = False
        self.model.bvapec.Redshift.frozen = False
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        #self.fit_error(error=self.error_str,error_calc=True,detector=False)
        self.result_pack()
        self.set_xydata_multi([modname])
        self.savemod_multi(modname)
        self.plotting_multi(modname, x_rng=(6.611,6.732))

    def fit_heasim_perseus_test(self):
        t0 = time.perf_counter() 
        files = sorted(glob.glob('../spec/*_vt150_*wRS.pi'))
        for f in files:
            self.fit_heasim_spec_perseus(f, f.split("/")[-1].split(".")[0], v=150)
        elapsed = time.perf_counter() - t0       # ---- 測定終了 ----
        print(f"[fit_heasim_perseus_test] elapsed: {elapsed:.2f} s")
    def fit_heasim_spec_multi_perseus(self):
        v_list = [150, 200, 300]
        for v in v_list:
            self.fit_heasim_spec(f'3by3_v{v}_wRS.pi', f'3by3_v_{v}', v)
            self.fit_heasim_spec(f'3by3_v{v}_woRS.pi', f'3by3_v_{v}_noRS', v)

    def fit_heasim_spec_pks(self, spec, modname, v=0):
        """
        Fitting heasim spectrum of the PKS0745-191
        Center region expected
        """
        rmf = "/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/heasim_response_xrism/1000_center_L_without_Lp.rmf"
        arf = "/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/rs_simulation/heasim_response_xrism/1000_center_image_1p8_8keV_1e7.arf"
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,rng=(6.51,6.79))
        Plot("data")
        self.model_bvapec_for_simulation(Nickel=False, gaus="zgaus", line="wz")
        self.load_apecroot("wz")
        self.model.bvapec.Fe.frozen = True
        self.model.bvapec.Ni.frozen = True
        self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity = v
        self.model.bvapec.Velocity.frozen = True
        self.model.zgauss.Sigma = 2e-3
        self.model.zgauss.norm   = 1e-5
        self.model.zgauss.norm.frozen = False
        self.model.zgauss.Sigma.frozen = True
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.bvapec.Fe.frozen = False
        self.model.bvapec.Velocity.frozen = False
        self.model.zgauss.Sigma.frozen = False
        self.model.bvapec.Redshift.frozen = False
        self.line = "wz"
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        ratio = self.model.zgauss_3.norm.values[0] / self.model.zgauss.norm.values[0]
        kT = self.model.bvapec.kT.values[0]
        apec_norm = self.model.bvapec.norm.values[0]
        Fe = self.model.bvapec.Fe.values[0]
        Ni = self.model.bvapec.Ni.values[0]
        velocity = self.model.bvapec.Velocity.values[0]
        z_sigma = self.model.zgauss.Sigma.values[0]
        z_norm = self.model.zgauss.norm.values[0]
        w_sigma = self.model.zgauss_3.Sigma.values[0]
        self.model_bvapec_for_simulation(Nickel=False, gaus="dzgaus", line="wz")
        self.model.bvapec.kT = kT
        self.model.bvapec.norm = apec_norm
        self.model.bvapec.Fe = Fe
        self.model.bvapec.Ni = Ni
        self.model.bvapec.Velocity = velocity
        self.model.dzgaus.Sigma_z = z_sigma
        self.model.dzgaus.Sigma_w = w_sigma
        self.model.dzgaus.lnorm_z = z_norm
        self.model.dzgaus.ratio = ratio
        self.model.dzgaus.ratio.frozen = True
        self.model.bvapec.Redshift.frozen = False
        # self.model.dzgaus.Sigma_z.frozen = True
        # self.model.dzgaus.Sigma_w.frozen = True
        
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.dzgaus.ratio.frozen = False
        self.fit_error(error=self.error_str,error_calc=True,detector=False)
        self.result_pack()
        self.set_xydata_multi([modname])
        self.savemod_multi(modname)
        self.plotting_multi(modname, x_rng=(6.611,6.732),line="wz")

    def fit_heasim_spec_pks2(self, spec, modname, v=0):
        rmf = "/Users/keitatanaka/spec_sim/heasim/heasimfiles/xrism/resolve/response/resolve_h5ev_2019a.rmf"
        arf = "/Users/keitatanaka/spec_sim/heasim/heasimfiles/xrism/resolve/response/resolve_pnt_heasim_withGV_20190701.arf"
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,rng=(6.51,6.79))
        Plot("data")
        self.model_bvapec_for_simulation(Nickel=False, gaus="zgaus", line="wz")
        self.load_apecroot("wz")
        self.model.bvapec.Fe.frozen = True
        self.model.bvapec.Ni.frozen = True
        self.model.bvapec.Redshift.frozen = True
        self.model.bvapec.Velocity = v
        self.model.bvapec.Velocity.frozen = True
        self.model.zgauss.Sigma = 2e-3
        self.model.zgauss.norm   = 1e-5
        self.model.zgauss.norm.frozen = False
        self.model.zgauss.Sigma.frozen = True
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.bvapec.Fe.frozen = False
        self.model.bvapec.Velocity.frozen = False
        self.model.zgauss.Sigma.frozen = False
        self.model.bvapec.Redshift.frozen = False
        self.line = "wz"
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        ratio = self.model.zgauss_3.norm.values[0] / self.model.zgauss.norm.values[0]
        kT = self.model.bvapec.kT.values[0]
        apec_norm = self.model.bvapec.norm.values[0]
        Fe = self.model.bvapec.Fe.values[0]
        Ni = self.model.bvapec.Ni.values[0]
        velocity = self.model.bvapec.Velocity.values[0]
        z_sigma = self.model.zgauss.Sigma.values[0]
        z_norm = self.model.zgauss.norm.values[0]
        w_sigma = self.model.zgauss_3.Sigma.values[0]
        self.model_bvapec_for_simulation(Nickel=False, gaus="dzgaus", line="wz")
        self.model.bvapec.kT = kT
        self.model.bvapec.norm = apec_norm
        self.model.bvapec.Fe = Fe
        self.model.bvapec.Ni = Ni
        self.model.bvapec.Velocity = velocity
        self.model.dzgaus.Sigma_z = z_sigma
        self.model.dzgaus.Sigma_w = w_sigma
        self.model.dzgaus.lnorm_z = z_norm
        self.model.dzgaus.ratio = ratio
        self.model.dzgaus.ratio.frozen = True
        self.model.bvapec.Redshift.frozen = False
        # self.model.dzgaus.Sigma_z.frozen = True
        # self.model.dzgaus.Sigma_w.frozen = True
        
        self.fit_error(error=self.error_str,error_calc=False,detector=False)
        self.model.dzgaus.ratio.frozen = False
        self.fit_error(error=self.error_str,error_calc=True,detector=False)
        self.result_pack()
        self.set_xydata_multi([modname])
        self.savemod_multi(modname)
        self.plotting_multi(modname, x_rng=(6.611,6.732))

    def fit_heasim_spec_multi_pks(self,region='center'):
        v_list = [0, 100, 150, 200, 300]
        for v in v_list:
            self.fit_heasim_spec_pks(f'{region}_v{v}_wRS.pi', f'{region}_v_{v}', v)
            self.fit_heasim_spec_pks(f'{region}_v{v}_woRS.pi', f'{region}_v_{v}_noRS', v)

    def simulation_multi(self, region='3by3', v=None, gaus='dzgaus',Nickel=True):
        velocity_list = [0, 100, 150, 200, 300]
        if v is not None:
            velocity_list = [v]
        for velocity in velocity_list:
            spec = f'./{region}_pixel_{velocity}.fits'
            rmf  = f'./{region}_pixel_{velocity}.rmf'
            self.simulation_fitting(spec, rmf, line='wz', modname=f'{region}_v_{velocity}',v=velocity,gaus=gaus,Nickel=Nickel)
            spec = f'./{region}_pixel_{velocity}_noRS.fits'
            rmf  = f'./{region}_pixel_{velocity}_noRS.rmf'
            self.simulation_fitting(spec, rmf, line='wz', modname=f'{region}_v_{velocity}_noRS',v=velocity,gaus=gaus,Nickel=Nickel)

    def simulation_multi_for_w(self, v=150):
        rad_name_list = glob.glob(f'./arcmin_*_noRS.fits')
        print(rad_name_list)
        for filename in rad_name_list:
            basename = os.path.basename(filename)
            match = re.match(r"(arcmin_\d+\.\d+_\d+\.\d+)", basename)
            base_name = match.group(1)         
            spec = f'./{base_name}.fits'
            rmf  = f'./{base_name}.rmf'
            self.simulation_fitting(spec, rmf, line='w', modname=f'{base_name}_v_{v}',v=v,gaus="zgaus")
            spec = f'./{base_name}_noRS.fits'
            rmf  = f'./{base_name}_noRS.rmf'
            self.simulation_fitting(spec, rmf, line='w', modname=f'{base_name}_v_{v}_noRS',v=v,gaus="zgaus")

    def w_woRS_flux_ratio(self, v):
        with h5py.File(self.savefile,'r') as f:
            keynames = list(f.keys())
            print(keynames)
            center_rad = []
            ratio_list = []
            ratio_err_list = []
            filtered = [k for k in keynames if 'noRS' not in k and f'v_{v}' in k]
            for region in filtered:
                wRS = f[f'{region}/fitting_result']['2/zgauss/norm/value'][...]
                wRS_ep = f[f'{region}/fitting_result']['2/zgauss/norm/ep'][...]
                wRS_em = f[f'{region}/fitting_result']['2/zgauss/norm/em'][...]
                woRS = f[f'{region}_noRS/fitting_result']['2/zgauss/norm/value'][...]
                woRS_ep = f[f'{region}_noRS/fitting_result']['2/zgauss/norm/ep'][...]
                woRS_em = f[f'{region}_noRS/fitting_result']['2/zgauss/norm/em'][...]
                ratio = woRS/wRS 
                ratio_err_em = ratio - (woRS + woRS_em) / (wRS + wRS_ep) 
                ratio_err_ep = (woRS + woRS_ep) / (wRS + wRS_em) - ratio
                print(ratio, ratio_err_em, ratio_err_ep)
                match = re.search(r'arcmin_(\d+\.\d+)_(\d+\.\d+)', region)
                if match:
                    low, high = float(match.group(1)), float(match.group(2))
                    print(f'arcmin {low} - {high} : {ratio:.3f}')
                center = (low + high) / 2
                center_rad.append(center)
                ratio_list.append(ratio)
                ratio_err_list.append((ratio_err_em, ratio_err_ep))
            #f = np.loadtxt(f'furukawa_g4_w_ratio_v{v}.txt')
            #plt.scatter(center_rad, ratio_list, color='red', label=f'tanaka, v = {v} km/s')
            ratio_err_array = np.array(ratio_err_list).T  # shape: (2, N)
            plt.errorbar(center_rad, ratio_list, yerr=ratio_err_array, fmt='o', color='red', capsize=5)
            #plt.scatter(f[:,0], f[:,1], color='black', label=f'Geant4 (furukawa), v = {v} km/s')
            plt.legend()
            plt.xlabel('Radius [arcmin]')
            plt.ylabel('witout RS w / with RS w')
            plt.axhline(1, color='black', lw=1, ls='--')
            plt.semilogx()
            plt.xlim(0.2, 20)
            #plt.ylim(0.5, 2.2)
            plt.savefig(f'flux_ratio_w_woRS_v{v}.png', dpi=300, transparent=False)
            plt.show()
    
    def result_plot_perseus(self, region='3by3'):
        velocity_list = [0, 100, 150, 200, 300]
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.grid(linestyle='dashed')
        wz_list = []
        with h5py.File("perseus_noPSF.hdf5",'r') as f:
            for v in velocity_list:
                wz = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='darkred', label='RS simulation (tanaka, PSF uncorrected)')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='darkred')
                wz_list.append(wz)
            self.ax.plot(velocity_list, wz_list, '-.', color='darkred')
        wz_list = []

        with h5py.File("heasim_perseus.hdf5",'r') as f:
            for v in velocity_list:
                wz = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red', label='RS simulation (tanaka, PSF corrected)')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red')
                wz_list.append(wz)
            self.ax.plot(velocity_list, wz_list, '-.', color='red')
            for v in velocity_list:
                wz = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5, label='optically thin (tanaka, PSF corrected)')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5)
        file = np.loadtxt('perseus_akimoto_digit.txt')
        self.ax.scatter(file[:,0], file[:,1], color='blue', label='Geant4 (akimoto)')
        self.ax.plot(file[:,0], file[:,1],'-.', color='blue')
        file = np.loadtxt('perseus_MCCM_digit.txt')
        self.ax.plot(file[:,0], file[:,1], color='orange', lw=5,alpha=0.75 ,label='MCCM simulation (Hitomi Col 2018)')
        self.ax.axhline(2.34, color='black', lw=1, ls='--', label='Observation (Hitomi Col 2018, atomdb v3.0.9)')
        self.ax.axhspan(2.34-0.1, 2.34+0.1, color='gray', alpha=0.5)
        self.ax.set_xlabel('Turbulent Velocity [km/s]')
        self.ax.set_ylabel('Integrated flux ratio w/z')
        self.ax.legend(loc='lower right')
        self.ax.set_title('Perseus Cluster (3x3 pixel region)')
        plt.show()                
        self.fig.savefig('simulation_result.png', dpi=300, transparent=False)

    def result_plot_pks(self, region='center'):
        velocity_list = [0, 100, 150, 200, 300]
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ax.grid(linestyle='dashed')
        wz_list = []
        with h5py.File(f'dummy2.hdf5','r') as f:
            for v in velocity_list:
                wz = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red', label='RS simulation result')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red')
                wz_list.append(wz)
            self.ax.plot(velocity_list, wz_list, '-.', color='red')
            # for v in velocity_list:
            #     wz = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/value'][...]
            #     wz_ep = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/ep'][...]
            #     wz_em = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/em'][...]
            #     if v == 0:
            #         self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5, label='optically thin (tanaka)')
            #     else:
            #         self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5)
        if region == 'center':
            wz_obs = 2.1
            wz_err_p = 2.7
            wz_err_m = 1.8
        elif region == 'outer':
            wz_obs = 7.1
            wz_err_p = 4.0
            wz_err_m = 9.0
        self.ax.axhline(wz_obs, color='black', lw=1, ls='--', label='Observation')
        if region == "center":
            self.ax.axvline(78, color='black', lw=1, ls='--')
            self.ax.axvspan(78-45, 78+34, color='blue', alpha=0.3)
        if region == "outer":
            self.ax.axvline(110, color='black', lw=1, ls='--')
            self.ax.axvspan(110-41, 110+32, color='blue', alpha=0.3)
        self.ax.axhspan(wz_err_m, wz_err_p, color='gray', alpha=0.5)
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,8.0,1000)
        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
            if up == 2:
                z = ldata['epsilon']
                print(ldata['energy'])
            if up == 7:
                w = ldata['epsilon']
                print(ldata['energy'])
        #self.ax.plot(kTlist, w/z, '-.',color='orange', label='apec')
        from scipy.interpolate import interp1d
        wz_ratio_kT = interp1d(kTlist, w/z)
        print(wz_ratio_kT(7.7), wz_ratio_kT(4.59))
        self.ax.axhline(wz_ratio_kT(7.7), color='orange', lw=3, ls='--', label='optically thin (atomdb v3.0.9)')
        self.ax.set_xlabel('Turbulent Velocity [km/s]')
        self.ax.set_ylabel('Integrated flux ratio w/z')
        #self.ax.legend(loc='lower right')
        self.ax.legend(loc='upper right')
        self.ax.set_title(f'PKS 0745-191 ({region} region)')
        self.fig.savefig(f'simulation_result_{region}.pdf', dpi=300, transparent=True)
        plt.show()                

    def result_plot_abell478(self, region='center'):
        velocity_list = [0, 50, 100, 150, 200, 250]
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ax.grid(linestyle='dashed')
        wz_list = []
        with h5py.File(f'dummy.hdf5','r') as f:
            for v in velocity_list:
                wz = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red', label='RS simulation (tanaka)')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], markersize=5, capsize=5, fmt='o', color='red')
                wz_list.append(wz)
            self.ax.plot(velocity_list, wz_list, '-.', color='red')
            for v in velocity_list:
                wz = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/value'][...]
                wz_ep = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/ep'][...]
                wz_em = f[f'{region}_v_{v}_noRS/fitting_result']['2/dzgaus/ratio/em'][...]
                if v == 0:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5, label='optically thin (tanaka)')
                else:
                    self.ax.errorbar(v, wz, yerr=[[-wz_em], [wz_ep]], fmt='o', color='black', markersize=5, capsize=5)
        if region == 'center':
            wz_obs = 2.49
            wz_err_p = 2.72
            wz_err_m = 2.26
        elif region == 'outer':
            wz_obs = 3.16
            wz_err_p = 3.88
            wz_err_m = 2.56
        #self.ax.axhline(wz_obs, color='black', lw=1, ls='--', label='Observation')
        #self.ax.axhspan(wz_err_m, wz_err_p, color='gray', alpha=0.5)
        self.ax.set_xlabel('Velocity [km/s]')
        self.ax.set_ylabel('Integrated flux ratio w/z')
        self.ax.legend(loc='lower right')
        self.ax.set_title(f'Abell 478 ({region} region)')
        self.fig.savefig(f'simulation_result_{region}.pdf', dpi=300, transparent=True)
        plt.show() 

    def fitting_spectrum_w(self,identifier='1000', region='center', line='w', rng=(2.0,17.0)):
        '''
        単一データのfitting用の関数
        '''
        modname = f'{identifier}_{region}_{line}'
        self.load_spectrum(spec=f'{identifier}_{region}_merged_b1.pi',rmf=f'{identifier}_{region}_L_without_Lp.rmf',arf=f'{identifier}_{region}_image_1p8_8keV_1e7.arf',multiresp='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf',rng=rng)
        self.model_bvapec(line=line,load_nxb=True,sigma_fix=False)
        self.load_apecroot(line)
        self.bvapec_fix_some_param(fix=True)
        self.fit_error(error=self.error_str,error_calc=False,detector='nxb1')
        self.bvapec_fix_some_param(fix=False)
        self.nxb_fix_parameter(fix=True)
        self.fit_error(error=self.error_str,error_calc=False,detector='nxb1')
        self.result_pack()
        self.savemod(modname)
        self.load_apecroot(line=None)
        Plot.device = '/xs'
        self.model.zgauss.norm = 0
        Plot('data delc')
        Plot.add = True
        self.y_apec=Plot.model()
        self.plotting_for_w(modname)

    def load_xcm_and_set(self, xcm="./xcm/ssm_w_for_ratio_plot.xcm"):
        Xset.restore(xcm)
        self.set_xydata_multi(['1','2'])
        self.load_apecroot(line=None)
        Plot.device = '/xs'
        comp = 'zgauss'
        for i in range(1,3):
            regions = ["", "outer", "exMXS"]
            for region in regions:
                m = AllModels(i, modName=region)
                if hasattr(m, comp):
                    getattr(m, comp).norm = 0


        cols = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        comp_col = ['orange', 'blue', 'green',  'brown', 'salmon', 'darkgreen', 'green']
        colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    
        for e, key in enumerate(list(self.xs.keys())):
            xs = self.xs[key]
            ys = self.ys[key]
            xe = self.xe[key]
            ye = self.ye[key]
            ys_comps = self.ys_comps
            y   = self.y[key]
            fig = plt.figure(figsize=(9,6))
            gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
            ax  = fig.add_subplot(gs1[:,:])
            ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
            ls  = 15
            ps  = 1.5
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e])
            ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
            ax.plot(xs,y,label="w include as gaus",color='black',lw=ps)
            Plot('data delc')
            Plot.add = True
            self.y_apec=Plot.model(plotGroup=int(key)+1)
            self.y_apec = np.array(self.y_apec)
            y = np.array(y)
            ax.plot(xs,self.y_apec,label="w exclude",color='red',lw=ps)
            # center_mod = np.sum(ys_comps[0:2],axis=0)
            # outer_mod = np.sum(ys_comps[2:4],axis=0)
            # ex12pix_mod = np.sum(ys_comps[4:6],axis=0)
            # ax.plot(xs,center_mod,'-',label=f"Center",lw=ps,color=colors[1])
            # ax.plot(xs,outer_mod,'-',label=f"Outer",lw=ps,color=colors[2])
            # ax.plot(xs,ex12pix_mod,'-.',label=f"ex12pix",lw=ps,color=colors[3])
            
            ax.set_xlim(5.9, 6.2)
            #ax.set_ylim(3e-4,np.max(yrng)+0.1)
            ax2.set_ylim(0.5, 1.5)
            y_apec_sum = np.sum(self.y_apec,axis=0)
            y_sum = np.sum(y,axis=0)
            ax2.plot(xs, y/self.y_apec,'-',label=f"sum",lw=ps,color='black')
            ax2.hlines(0,1.8,10,linestyle='-.',color='green')
            ax2.set_xlabel("Energy[keV]",fontsize=ls)
            ax2.set_ylabel("Model ratio",fontsize=ls)
            ax.grid(linestyle='dashed')
            ax2.grid(linestyle='dashed')
            spine_width = 2  # スパインの太さ
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            for spine in ax2.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis='both',direction='in',width=1.5)
            ax2.tick_params(axis='both',direction='in',width=1.5)
            ax.legend(fontsize=12,loc="upper right")
            #ax.set_title(f"{modname}")
            fig.align_labels()
            fig.tight_layout()
            #plt.show()
            fig.savefig(f"./figure/w_ratio_{key}.pdf",dpi=300,transparent=True)        

    def fitting_each_fwdata(self,region='center',line='w',rebin=1):
        '''
        全フィルターのfitting
        '''
        identifiers = ['2000','3000','4000']
        for identifier in identifiers:
            if identifier == '2000':
                self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,15.0), rebin=rebin)
            else:
                self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,17.0), rebin=rebin)
        # self.simultaneous_some_fwdata(region=region,line=line)

    def fitting_each_fwdata_reg(self,rebin=1):
        '''
        全フィルターのfitting
        '''
        identifiers = ['1000','2000','3000','4000']
        lines       = ['w', 'wz']
        for line in lines:
            # for region in ['center','outer','all']:
            for region in ['center']:
                for identifier in identifiers:
                    if identifier == '2000':
                        self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,15.0), rebin=rebin)
                    else:
                        self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,17.0), rebin=rebin)
        # self.simultaneous_some_fwdata(region=region,line=line)

    def Fe_data_fitting(self,region='center',line='wz',optbin=True,noMXS=False):
        AllData.clear()

        modname = f'5000_{region}_{line}'
        Plot.device = '/xs'
        identifier = '5000'
        spec = f'{identifier}_{region}_merged_b1.pi'
        rmf  = f'{identifier}_{region}_X_without_Lp_comb.rmf'
        arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'

        spectrum = Spectrum(spec)
        spectrum.response = rmf
        spectrum.response.arf = arf
        spectrum.multiresponse[1] = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
        spectrum.multiresponse[2] = f'{identifier}_{region}_X_without_Lp_comb.rmf'
        self.set_data_range(rng=(2.0,17.0))
        AllData.ignore("1:6.2-6.4")
        self.set_xydata()
        self.model_bvapec(line,load_nxb=True,load_FeMXS=True,sigma_fix=False,noMXS=noMXS)
        self.load_apecroot(line)
        self.bvapec_fix_some_param(fix=True)
        self.FeMXS_model.zashift.Redshift.frozen = True
        self.FeMXS_model.gsmooth.Sig_6keV.frozen = True
        # self.FeMXS_model.zashift_12.Redshift.frozen = True
        # self.FeMXS_model.gsmooth_13.Sig_6keV.frozen = True
        # self.FeMXS_model.zashift_21.Redshift.frozen = True
        # self.FeMXS_model.gsmooth_22.Sig_6keV.frozen = True
        # self.FeMXS_model.zashift_31.Redshift.frozen = True
        # self.FeMXS_model.gsmooth_32.Sig_6keV.frozen = True
        # self.FeMXS_model.zashift_39.Redshift = 0
        # self.FeMXS_model.zashift_39.Redshift.frozen = True
        # self.FeMXS_model.gsmooth_40.Sig_6keV.frozen = True
        # self.FeMXS_model.gsmooth_47.Sig_6keV.frozen = True
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter(fix=True)
        self.bvapec_fix_some_param(fix=False)
        # self.FeMXS_model.powerlaw.PhoIndex.frozen = True
        # self.FeMXS_model.powerlaw.norm.frozen = True
        # self.FeMXS_model.gsmooth.Sig_6keV.frozen = True
        self.fit_error(self.error_str,True,'nxb1')
        self.result_pack()
        self.savemod(modname=modname)
        self.plotting(modname=modname)
        #fig.savefig(f"./figure/{modname}_{region}_{line}_fitting.pdf",dpi=300,transparent=True)
        # Xset.save('center_for_cont.xcm')

    def occulation_data_fitting(self,region='all',line='MnKab',noMXS=True,rebin=5):
        AllData.clear()
        identifier = '5000'
        spec = f'{identifier}_{region}_merged_b1.pi'
        rmf  = f'{identifier}_{region}_X_without_Lp_comb.rmf'

        spectrum = Spectrum(spec)
        spectrum.response = rmf
        spectrum.multiresponse[1] = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
        self.set_data_range(rng=(2.0,17.0))
        # AllData.ignore("1:6.2-6.4")
        self.set_xydata(rebin=rebin)
        self.model_55Fe(load_nxb=True,load_FeMXS=True,noMXS=noMXS)
        self.load_apecroot(line)
        # self.bvapec_fix_some_param(fix=True)
        err = '1.0 1,4,29,32,49,50'
        self.fit_error(err,False,'nxb1')
        self.model.gsmooth.Sig_6keV.frozen = True
        # self.nxb_fix_parameter(fix=True)
        self.fit_error(err,False,'nxb1')
        self.result_pack()
        self.savemod(f'{region}_{line}')
        Xset.save('Fe_exp.xcm')
        self.plotting_55Fe_line(modname=f'{region}_{line}',logging=True)
        # fig.savefig(f"./figure/{region}_simultaneous_fitting.pdf",dpi=300,transparent=True)
        # Xset.save('center_for_cont.xcm')

    def occulation_pixel_fitting(self,region='center',line='wz',optbin=True,noMXS=True,rebin=1):
        AllData.clear()
        identifier = '5000'
        spec = f'{identifier}_PIXEL_{region}_merged_b1.pi'
        rmf  = f'{identifier}_pix{region}_X_without_Lp_comb.rmf'

        spectrum = Spectrum(spec)
        spectrum.response = rmf
        spectrum.multiresponse[1] = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
        self.set_data_range(rng=(2.0,17.0))
        # AllData.ignore("1:6.2-6.4")
        self.set_xydata(rebin=rebin)
        self.model_55Fe(load_nxb=True,load_FeMXS=True,noMXS=noMXS)
        self.load_apecroot(line)
        # self.bvapec_fix_some_param(fix=True)
        err = '1.0 1,4,29,32,49,50'
        self.fit_error(err,False,'nxb1')
        self.model.gsmooth.Sig_6keV.frozen = True
        self.nxb_fix_parameter(fix=True)
        self.fit_error(err,False,'nxb1')
        self.result_pack()
        self.savemod(f'{region}_{line}')
        self.plotting_55Fe_line(modname=f'{region}_{line}',logging=True)
        # fig.savefig(f"./figure/{region}_simultaneous_fitting.pdf",dpi=300,transparent=True)
        # Xset.save('center_for_cont.xcm')

    def fitting_pixel_by_pixel_55Fe_line(self,line='MnKa'):
        # self.all_pixels = ["00", "17", "18", "35"]
        # self.all_pixels  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        #self.all_pixels = ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        #self.all_pixels = ['00', '07', '08', '09', '10', '11']
        # self.all_pixels = ['21', '22', '23', '24', '25', '26', '33', '34', '35']
        # self.all_pixels  = ['21']
        if line == 'MnKa':
            rng = (5.855,5.935)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKa_gsmooth_f1.xcm'
        if line == 'MnKb':
            rng = (6.4,6.6)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_MnKb.xcm'
        if line == 'CrKa':
            rng = (5.3,5.5)
            model_xcm = '/Volumes/SUNDISK_SSD/PKS_XRISM/model_xcm/model_CrKa.xcm'
        for num,pixel in enumerate(self.all_pixels):
            AllData.clear()
            spec = f"5000_PIXEL_{pixel}_b1.pi"
            rmf = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
            self.load_spectrum(spec=spec,rmf=rmf,arf=None,multiresp='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf',rng=rng)
            Xset.restore(model_xcm)
            Fit.statMethod = "cstat"
            self.set_data_range(rng=rng)
            model = AllModels(1)
            model.gsmooth.Sig_6keV.frozen = True
            self.fit_error(error='1.0 1,2,4',error_calc=False,detector=False)
            model.zashift.Redshift.frozen = False
            model.gsmooth.Sig_6keV.frozen = False
            self.fit_error(error='1.0 1 2',error_calc=True,detector=False)
            self.set_xydata_multi(['5000'])
            self.result_pack()
            self.savemod_multi(modname=f'pixel{pixel}')
            #rng = (5.375,5.450)
            self.plotting_55Fe_line(f'pixel{pixel}',x_rng=rng,logging=False,bgplot=False,line=line)

    def fitting_all_region(self,identifiers=['1000','2000','3000','4000']):
        regions = ['all', 'center', 'outer']
        for line in ['None']:
            self.line = line
            for region in regions:
                for identifier in identifiers:
                    if identifier == '2000':
                        self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,15.0))
                    else:
                        self.fitting_spectrum(identifier=identifier, region=region, line=line, rng=(2.0,17.0))
                self.simultaneous_some_fwdata(region=region,line=line)

    def open_data_fitting(self,identifier='1000',region='center',line='wz', rebin=1, line_model='dzgaus'):
        AllData.clear()
        spec = f'{identifier}_{region}_merged_b1.pi'
        rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
        arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
        multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=1,spectrumgroup=1)
        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        Plot.setRebin(rebin,1000)
        Plot.device = '/xs'
        Plot('data')
        self.model_bvapec(line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=1,line_model=line_model)
        self.load_apecroot(line)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter(fix=True)
        self.fit_error(self.error_str,True,'nxb1')
        Plot('data delc')
        self.set_xydata_multi([identifier])
        self.result_pack()
        self.savemod_multi(f'{identifier}_{region}_{line}_rebin_{rebin}_{line_model}')
        self.plotting_simult(f'{identifier}_{region}_{line}_rebin_{rebin}_{line_model}',line=line,bg_plot=False)

    def fe_data_exbgd_fitting(self,mode="brt",region='center',line='wz', rebin=1, line_model='dzgaus'):
        AllData.clear()
        identifier = '5000'
        spec = f'{identifier}_{region}_merged_b1.pi'
        rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
        arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
        bgd  = f'./bgd/{identifier}_{region}.pi'
        self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=None,multiresp2=None,bgd=bgd,datagroup=1,spectrumgroup=1)
        AllData.notice("2.0-10.0")
        AllData.ignore("1:**-2.0 10.0-**")
        Plot.setRebin(rebin,1)
        Plot.device = '/xs'
        Plot('data')
        self.model_bvapec(line,load_nxb=False,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=1,line_model=line_model)
        self.load_apecroot(line)
        self.fit_error(self.error_str,False,None)
        self.fit_error(self.error_str,True,None)
        Plot('data delc')
        self.set_xydata_multi([identifier])
        self.result_pack()
        self.savemod_multi(f'{mode}_{region}_{line}_rebin_{rebin}_{line_model}')
        self.plotting_simult(f'{mode}_{region}_{line}_rebin_{rebin}_{line_model}',line=line,bg_plot=False,x_rng=[5.9, 6.2])

    def each_fe_fitting(self):
        dir_list = ['brt', 'max', 'brmax']
        #dir_list = ['brmax']
        cdir = os.getcwd()
        for d in dir_list:
            os.chdir(f'./{d}')
            for region in ['outer']:
                self.fe_data_exbgd_fitting(mode=d,region=region,line='wz',rebin=1,line_model='zgaus')
            os.chdir(cdir)

    def simultaneous_fitting(self,identifiers=['1000', '2000', '3000', '4000'],region='all',line='None', rebin=5, line_model='zgaus', thermal=False, z_fix=False):
        AllData.clear()
        for e,identifier in enumerate(identifiers): 
            spec = f'{identifier}_{region}_merged_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
            multiresp = f'{self.dropbox}/SSD_backup/PKS_XRISM/rmfarf/newdiag60000.rmf'
            self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=e+1,spectrumgroup=e+1)
        if identifiers == ['2000', '3000', '4000']:
            sim_idnt = 'S_other'
        elif identifiers == ['1000', '2000', '3000', '4000']:
            sim_idnt = 'S_all'
        else:
            sim_idnt = 'other'
        if z_fix == True:
            z_name = "z_fix"
        else:
            z_name = "z_nofix"
        if thermal == True:
            modname = f'{sim_idnt}_{region}_{line}_rebin_{rebin}_{line_model}_therm_{z_name}'
            self.thermal_broadening("no")
        else:
            modname = f'{sim_idnt}_{region}_{line}_rebin_{rebin}_{line_model}_{z_name}'
            self.thermal_broadening("yes")
        AllData.notice("2.0-17.0")
        for i in range(1,len(identifiers)+1):
            AllData.ignore(f"{i}:**-2.0 17.0-**")
        Plot.setRebin(rebin,1000)
        Plot.device = '/xs'
        Plot('data')
        self.model_bvapec(line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=len(identifiers),line_model=line_model,z_fix=z_fix)
        self.load_apecroot(line)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter(fix=True)
        self.fit_error(self.error_str,True,'nxb1')
        Plot('data delc')
        self.set_xydata_multi(identifiers)
        self.result_pack()
        if z_fix == False:
            for i in range(2,len(identifiers)+1):
                self.result_pack_only_z(model=AllModels(i),group=i)
        self.savemod_multi(modname)
        self.plotting_simult(modname,line=line,bg_plot=False,identifiers=identifiers)

    def fitting_ssm(self, xcm, line='wz', line_model='zgauss', fix_bvapec=False, rng=(2.0,17.0)):
        self.load_data_ssm(xcm=xcm,rng=rng)
        Plot.device = '/xs'
        Plot('data')
        identifiers = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.model_bvapec_ssm(line=line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=8,line_model=line_model)
        if fix_bvapec == True:
            self.bvapec_parameter_fix('ssm_w_rebin_1_zgauss')
            fix_str = "param_fix"
            self.error_str = '1.0 20 22 24 26 outer:20 outer:22 outer:24 outer:26'
        else:
            fix_str = "param_nofix"
        self.load_apecroot(line=line)
        #self._freeze_redshift(comp=line_model)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter_ssm(fix=True)
        #self._free_redshift(comp=line_model)
        self.fit_error(self.error_str,True,'nxb1')
        Plot('data delc')
        self.set_xydata_multi(identifiers)
        self.result_pack_ssm()
        for i in range(2,len(identifiers)+1):
            self.result_pack_only_z_ssm(model=self.model_center,group=i,region='center',modname='')
            self.result_pack_only_z_ssm(model=self.model_outer,group=i,region='outer',modname='outer')
        self.save_mod_ssm(f'ssm_{line}_rebin_1_{line_model}_{fix_str}')
        self.plotting_ssm(f'ssm_{line}_rebin_1_{line_model}_{fix_str}',x_rng=[5.9, 6.2])

    def add_rebin_data(self, modname, rebin=1):
        self.load_data_ssm(xcm=f"./xcm/{modname}.xcm",rng=(2.0,17.0))
        Plot.device = '/xs'
        Plot('data')
        identifiers = ["1", "2", "3", "4", "5", "6", "7", "8"]        
        self.set_xydata_multi(identifiers=identifiers, rebin=rebin)
        self.save_rebined_data(modname, rebin, identifiers=identifiers)

    def add_some_rebin(self, modname):
        self.load_data_ssm(xcm=f"./xcm/{modname}.xcm",rng=(2.0,17.0))
        Plot.device = '/xs'
        Plot('data')
        identifiers = ["1", "2", "3", "4", "5", "6", "7", "8"]      
        self.models = {}
        self.models["center"] = {}
        self.models["outer"]  = {}
        self.models["exMXS"]  = {}
        for i in range(1, 9):
            self.models["center"][str(i)] = AllModels(i)
            self.models["outer"][str(i)]  = AllModels(i, "outer")
            self.models["exMXS"][str(i)]  = AllModels(i, "exMXS")
        self.model_center = AllModels(1)
        self.model_outer  = AllModels(1, "outer")
        self.model_exMXS  = AllModels(1, "exMXS")
        for i in range(1,6):
            self.set_xydata_multi(identifiers=identifiers, only_data=False, rebin=i)
            self.result_pack_ssm()
            new_modname = f"{modname}_rebin{i}"
            f = h5py.File(self.savefile, 'a')
            if new_modname in f.keys():
                del f[new_modname]
            self.savemod_multi(new_modname)

    def save_rebined_data(self, modname, rebin, identifiers=["1", "2", "3", "4", "5", "6", "7", "8"]):
        with h5py.File(self.savefile, 'a') as f:
            if modname not in f.keys():
                print(f"{modname} is not in {self.savefile}")
                return
            else:
                if f"xs_rebin{rebin}" in f[modname].keys():
                    print(f"xs_rebin{rebin} is already in {self.savefile}")
                    del f[modname][f"xs_rebin{rebin}"], f[modname][f"ys_rebin{rebin}"], f[modname][f"xe_rebin{rebin}"], f[modname][f"ye_rebin{rebin}"]
                
                for e,i in enumerate(identifiers):
                    f[modname].create_dataset(f"xs_rebin{rebin}/{e}", data=self.xs[f"{e}"])
                    f[modname].create_dataset(f"ys_rebin{rebin}/{e}", data=self.ys[f"{e}"])
                    f[modname].create_dataset(f"xe_rebin{rebin}/{e}", data=self.xe[f"{e}"])
                    f[modname].create_dataset(f"ye_rebin{rebin}/{e}", data=self.ye[f"{e}"])
                        

    def fitting_ssm_some_line(self, xcm, fix_bvapec=False, rng=(2.0,17.0)):
        self.fitting_ssm(xcm=xcm, line="wz", line_model="zgauss", fix_bvapec=fix_bvapec, rng=rng)
        self.fitting_ssm(xcm=xcm, line="wz", line_model="dzgaus", fix_bvapec=fix_bvapec, rng=rng)
        self.fitting_ssm(xcm=xcm, line="w", line_model="zgauss", fix_bvapec=fix_bvapec, rng=rng)

    def bvapec_parameter_fix(self, modname):
        regions = ["", "outer", "exMXS"]
        reg_name = ["center", "outer", "outer"]
        with h5py.File(self.savefile, 'a') as f:
            for i in range(1,9):
                for e,region in enumerate(regions):
                    m = AllModels(i, modName=region)                        
                    reg = reg_name[e]
                    if i == 1:
                        print(reg)
                        m.bvapec.kT = f[f'{modname}/fitting_result/{reg}/2/bvapec/kT/value'][()]

                        m.bvapec.Fe       = f[f'{modname}/fitting_result/{reg}/2/bvapec/Fe/value'][()]
                        m.bvapec.Ni       = f[f'{modname}/fitting_result/{reg}/2/bvapec/Ni/value'][()]
                        m.bvapec.Redshift = f[f'{modname}/fitting_result/{reg}/2/bvapec/Redshift/value'][()]
                        m.bvapec.Velocity = f[f'{modname}/fitting_result/{reg}/2/bvapec/Velocity/value'][()]
                        m.bvapec.norm = f[f'{modname}/fitting_result/{reg}/2/bvapec/norm/value'][()]

                        m.bvapec.Fe.frozen       = True
                        m.bvapec.Ni.frozen       = True
                        m.bvapec.Redshift.frozen = True
                        m.bvapec.Velocity.frozen = True
                        m.bvapec.norm.frozen     = True
                        m.bvapec.kT.frozen       = True
                        bg_model = AllModels(1, modName='nxb1')
                        bg_model.constant.factor          = 0.244742
                        bg_model.constant.factor.frozen   = True
                        bg_model.powerlaw.PhoIndex        = 0.17723
                        bg_model.powerlaw.PhoIndex.frozen = True
                        bg_model.gaussian_19.Sigma           = 0.00751946
                        bg_model.gaussian_19.Sigma.frozen = True
                        bg_model.gaussian_20.Sigma        = 0.00896934
                        bg_model.gaussian_20.Sigma.frozen = True
                        bg_model.gaussian_21.Sigma        = 0.0132287
                        bg_model.gaussian_21.Sigma.frozen = True
                        
                    elif i == 2:
                        bg_model = AllModels(2, modName='nxb2')
                        bg_model.constant.factor        = 0.611156
                        bg_model.constant.factor.frozen = True
                        bg_model.powerlaw.PhoIndex        = 0.17723
                        bg_model.powerlaw.PhoIndex.frozen = True
                        bg_model.gaussian_19.Sigma           = 0.0060404
                        bg_model.gaussian_19.Sigma.frozen = True
                        bg_model.gaussian_20.Sigma        = 0.00731091
                        bg_model.gaussian_20.Sigma.frozen = True
                        bg_model.gaussian_21.Sigma        = 0.0194877
                        bg_model.gaussian_21.Sigma.frozen = True
                        m.bvapec.Redshift  = f[f'{modname}/fitting_result/{reg}/2/bvapec/Redshift_{i}/value'][()]
                        m.bvapec.Redshift.frozen = True
                    else:
                        m.bvapec.Redshift  = f[f'{modname}/fitting_result/{reg}/2/bvapec/Redshift_{i}/value'][()]
                        m.bvapec.Redshift.frozen = True

    def fitting_ssm_exBe(self, xcm, line='wz', line_model='zgauss'):
        self.load_data_ssm(xcm=xcm,spec_num=6)
        Plot.device = '/xs'
        Plot('data')
        identifiers = ["1", "2", "3", "4", "5", "6"]
        self.model_bvapec_ssm(line=line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=6,line_model=line_model)
        self.load_apecroot(line=line)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter_ssm(fix=True)
        self.fit_error(self.error_str,True,'nxb1')
        Plot('data delc')
        self.set_xydata_multi(identifiers)
        self.result_pack_ssm()
        for i in range(2,len(identifiers)+1):
            self.result_pack_only_z_ssm(model=self.model_center,group=i,region='center',modname='')
            self.result_pack_only_z_ssm(model=self.model_outer,group=i,region='outer',modname='outer')
        self.save_mod_ssm(f'ssm_{line}_rebin_1_{line_model}_exBe')
        self.plotting_ssm(f'ssm_{line}_rebin_1_{line_model}_exBe',x_rng=[5.9, 6.2])

    def fitting_ssm_single(self, fwnum="4000", line='wz', line_model='zgauss'):
        AllData.clear()
        xcm = f"{fwnum}_input.xcm"
        self.load_data_ssm(xcm=xcm,spec_num=2,rng=(2.0,17.0))
        Plot.device = '/xs'
        Plot('data')
        identifiers = ["1", "2"]
        self.model_bvapec_ssm(line=line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=2,line_model=line_model)
        self.load_apecroot(line=line)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter_ssm(fix=True)
        self.fit_error(self.error_str,True,'nxb1')
        Plot('data delc')
        self.set_xydata_multi(identifiers)
        self.result_pack_ssm()
        for i in range(2,len(identifiers)+1):
            self.result_pack_only_z_ssm(model=self.model_center,group=i,region='center',modname='')
            self.result_pack_only_z_ssm(model=self.model_outer,group=i,region='outer',modname='outer')
        self.save_mod_ssm(f'ssm_{fwnum}_{line}_rebin_1_{line_model}')
        self.plotting_ssm(f'ssm_{fwnum}_{line}_rebin_1_{line_model}',x_rng=[5.9, 6.2])

    def multi_ssm(self):
        fws = ["4000", "3000"]
        for fw in fws:
            self.fitting_ssm_single(fwnum=fw)

    def ssm_multi(self):
        xcm = "ssm_data.xcm"
        for line in ["all"]:
            self.fitting_ssm(xcm=xcm, line=line)

        # for i in range(2,len(identifiers)+1):
        #     X.result_pack_only_z_ssm(model=X.model_center,group=i,region='center',modname='')
        #     X.result_pack_only_z_ssm(model=X.model_outer,group=i,region='outer',modname='outer')

        # X.save_mod_ssm(f'ssm_4000_wz_rebin_1_dzgaus')
        # X.plotting_ssm(f'ssm_4000_wz_rebin_1_dzgaus',x_rng=[5.9, 6.2])

    def ssm_savetest(self):
        self.result_pack_ssm()
        self.save_mod_ssm(f'ssm_wz_rebin_1_zgauss')
        #self.plotting_ssm(f'ssm_wz_rebin_1_zgauss',x_rng=[5.9, 6.2])

    def simultaneous_fe_ex_bgd(self,identifiers=['max', 'brt', 'brmax'],region='outer',line='wz', rebin=1, line_model='dzgaus', thermal=False):
        AllData.clear()
        cdir = os.getcwd()
        for e,d in enumerate(identifiers):
            os.chdir(f'./{d}')
            spec = f'5000_{region}_merged_b1.pi'
            rmf  = f'5000_{region}_L_without_Lp.rmf'
            arf  = f'5000_{region}_image_1p8_8keV_1e7.arf'
            bgd  = f'./bgd/5000_{region}.pi'
            self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=None,multiresp2=None,bgd=bgd,datagroup=e+1,spectrumgroup=e+1)
            os.chdir(cdir)
        if identifiers == ['max', 'brt', 'brmax']:
            sim_idnt = 'Fe_all'
        else:
            sim_idnt = 'other'
        if thermal == True:
            modname = f'{sim_idnt}_{region}_{line}_rebin_{rebin}_{line_model}_therm'
            self.thermal_broadening("no")
        else:
            modname = f'{sim_idnt}_{region}_{line}_rebin_{rebin}_{line_model}'
            self.thermal_broadening("yes")
        AllData.notice("2.0-17.0")
        for i in range(1,len(identifiers)+1):
            AllData.ignore(f"{i}:**-2.0 17.0-**")
        Plot.setRebin(rebin,100)
        Plot.device = '/xs'
        Plot('data')
        self.model_bvapec(line,load_nxb=False,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=1,line_model=line_model)
        self.load_apecroot(line)
        self.fit_error(self.error_str,False,None)
        self.fit_error(self.error_str,True,None)
        Plot('data delc')
        self.set_xydata_multi(identifiers)
        self.result_pack()
        self.savemod_multi(modname)
        self.plotting_simult(modname,line=line,bg_plot=False,identifiers=identifiers)

    def run_fit(self):
        for region in ['center', 'outer', 'all']:
            self.simultaneous_fitting(identifiers=['1000', '2000', '3000', '4000'],region=region,line='wz', rebin=1)
            self.simultaneous_fitting(identifiers=['2000', '3000', '4000'],region=region,line='wz', rebin=1)
        for region in ['center','outer', 'all']:
            self.open_data_fitting(identifier='1000',region=region,line='wz', rebin=1)

    def print_ratio_zgaus(self):
        pass

    def set_xydata_multi(self,identifiers=['1000','2000','3000','4000'],only_data=False,rebin=1):
        xs_data = {}
        ys_data = {}
        xe_data = {}
        ye_data = {}
        y_data  = {}
        xres_data = {}
        yres_data = {}
        xres_e_data = {}
        yres_e_data = {}
        yscomps_data = {}
        if only_data == False:
            for group_num, identifier in enumerate(identifiers):
                self.set_xydata(plotgroup=group_num+1, rebin=rebin)
                self.get_model_comps(plotgroup=group_num+1)
                self.get_delc(spectrumnum=group_num+1)
                self.get_del(spectrumnum=group_num+1)
                xs_data[f'{group_num}'] = self.xs
                ys_data[f'{group_num}'] = self.ys
                xe_data[f'{group_num}'] = self.xe
                ye_data[f'{group_num}'] = self.ye
                y_data[f'{group_num}']  = self.y
                xres_data[f'{group_num}'] = self.xres
                yres_data[f'{group_num}'] = self.yres
                xres_e_data[f'{group_num}'] = self.xres_e
                yres_e_data[f'{group_num}'] = self.yres_e
                yscomps_data[f'{group_num}'] = self.ys_comps 
            self.xs = xs_data
            self.ys = ys_data
            self.xe = xe_data
            self.ye = ye_data
            self.y  = y_data
            self.xres = xres_data
            self.yres = yres_data
            self.xres_e = xres_e_data
            self.yres_e = yres_e_data
            self.yscomps = yscomps_data
        else:
            for group_num, identifier in enumerate(identifiers):
                self.set_xydata(plotgroup=group_num+1,rebin=rebin)
                xs_data[f'{group_num}'] = self.xs
                ys_data[f'{group_num}'] = self.ys
                xe_data[f'{group_num}'] = self.xe
                ye_data[f'{group_num}'] = self.ye
            self.xs = xs_data
            self.ys = ys_data
            self.xe = xe_data
            self.ye = ye_data

    def simultaneous_some_fwdata_rsapec(self,region='center',line='None',optbin=True,range=(2.0,17.0)):
        AllData.clear()
        identifiers = ['1000', '2000', '3000', '4000']
        # identifiers = ['1000']
        col_list    = ['black', 'red', 'green', 'blue'] 
        for identifier in identifiers:
            if optbin == True:
                spec = f'{identifier}_{region}_merged_b1.pi'
            else:
                spec = f'{identifier}_{region}_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'

            spectrum = Spectrum(spec)
            spectrum.response = rmf
            spectrum.response.arf = arf
            spectrum.multiresponse[1] = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
       
        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        if '2000' in identifiers:
            AllData.ignore("2:**-2.0 15.0-**")
        if '3000' in identifiers:
            AllData.ignore("3:**-2.0 17.0-**")
        if '4000' in identifiers:
            AllData.ignore("4:**-2.0 17.0-**")
        Plot.setRebin(1,1000)
        self.model_bvapec(line='w',load_nxb=True,sigma_fix=False)
        self.load_apecroot(line='w')
        Plot.device = '/xs'
        Plot('data') 
        err = '1.0 2'
        self.fit_error(err,False,False)
        self.nxb_fix_parameter(fix=True)
        print('------------------------')
        nh = self.model.TBabs.nH.values[0]
        kT = self.model.bvapec.kT.values[0]
        Fe = self.model.bvapec.Fe.values[0]
        Ni = self.model.bvapec.Ni.values[0]
        Redshift = self.model.bvapec.Redshift.values[0]
        Velocity = self.model.bvapec.Velocity.values[0]
        norm     = self.model.bvapec.norm.values[0]
        print('------------------------')
        self.model_rsapec()
        self.model.TBabs.nH        = nh
        self.model.TBabs.nH.frozen = True
        self.model.pyvapecrs.norm  = norm
        self.model.pyvapecrs.kT = kT
        self.model.pyvapecrs.H  = 0.4
        self.model.pyvapecrs.He = 0.4
        self.model.pyvapecrs.C  = 0.4
        self.model.pyvapecrs.N  = 0.4
        self.model.pyvapecrs.O  = 0.4
        self.model.pyvapecrs.Ne = 0.4
        self.model.pyvapecrs.Mg = 0.4
        self.model.pyvapecrs.Al = 0.4
        self.model.pyvapecrs.Si = 0.4
        self.model.pyvapecrs.S  = 0.4
        self.model.pyvapecrs.Ar = 0.4
        self.model.pyvapecrs.Ca = 0.4
        self.model.pyvapecrs.Fe = 0.5
        self.model.pyvapecrs.Ni = 0.5
        self.model.pyvapecrs.Fe = Fe
        self.model.pyvapecrs.Ni = Ni
        self.model.pyvapecrs.Redshift = Redshift
        self.model.pyvapecrs.Velocity = Velocity
        self.model.pyvapecrs.norm.frozen = False
        self.model.pyvapecrs.kT.frozen = False
        self.model.pyvapecrs.H.frozen  = True
        self.model.pyvapecrs.He.frozen = True
        self.model.pyvapecrs.C.frozen  = True
        self.model.pyvapecrs.N.frozen  = True
        self.model.pyvapecrs.O.frozen  = True
        self.model.pyvapecrs.Ne.frozen = True
        self.model.pyvapecrs.Mg.frozen = True
        self.model.pyvapecrs.Al.frozen = True
        self.model.pyvapecrs.Si.frozen = True
        self.model.pyvapecrs.S.frozen  = True
        self.model.pyvapecrs.Ar.frozen = True
        self.model.pyvapecrs.Ca.frozen = True
        self.model.pyvapecrs.Fe.frozen = False
        self.model.pyvapecrs.Ni.frozen = False
        self.model.pyvapecrs.Redshift.frozen = False
        self.model.pyvapecrs.Velocity.frozen = False
        self.model.pyvapecrs.nL.values = [4.8E+21,1e+19,0,0,1e+24,1e+24]
        self.load_apecroot(line='None')
        Fit.perform()
        Fit.error('1.0 19')
        Fit.error('1.0 2')
        Fit.error('1.0 15')
        Fit.error('1.0 16')
        Fit.error('1.0 18')
        Fit.error('1.0 19')
        Fit.error('1.0 20')
        Fit.error('1.0 19')

        self.set_xydata_multi(identifiers)
        self.result_pack()
        self.savemod_multi(f'simultaneous_{region}_{line}_rsapec')
        self.plotting_multi(modname=f'simultaneous_{region}_{line}_rsapec')

    def Fe_data_fitting_simult(self,region='center',line='wz'):
        AllData.clear()
        current_dir = os.getcwd()
        identifiers = ['1000', '2000', '3000', '4000', 'max', 'bright']
        col_list    = ['black', 'red', 'green', 'blue'] 
        for e,identifier in enumerate(identifiers):
            if identifier == 'max' or identifier == 'bright':
                if identifier == 'max':
                    Fe_dir     = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/max'
                elif identifier == 'bright':
                    Fe_dir     = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/bright'
                os.chdir(Fe_dir)
                spec = f'5000_{region}_merged_b1.pi'
                rmf  = f'5000_{region}_X_without_Lp_comb.rmf'
                arf  = f'5000_{region}_image_1p8_8keV_1e7.arf'    
                multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                multiresp2 = f'5000_{region}_X_without_Lp_comb.rmf'   
            else:
                os.chdir(current_dir)
                spec = f'{identifier}_{region}_merged_b1.pi'
                rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
                arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
                multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                multiresp2 = f'{identifier}_{region}_L_without_Lp.rmf'


            self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=multiresp2,datagroup=e+1,spectrumgroup=e+1)
        os.chdir(current_dir)
        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        AllData.ignore("2:**-2.0 15.0-**")
        AllData.ignore("3:**-2.0 17.0-**")
        AllData.ignore("4:**-2.0 17.0-**")
        AllData.ignore("5:**-2.0 17.0-**")
        AllData.ignore("5:6.2-6.4")
        AllData.ignore("6:**-2.0 17.0-**")
        AllData.ignore("6:6.2-6.4")
        Plot.setRebin(1,1000)
        Plot.device = '/xs'
        Plot('data')
        # self.set_data_range(rng=(2.0,17.0))
        # self.set_xydata()
        self.model_bvapec(line,load_nxb=True,load_FeMXS=True,sigma_fix=False,multi=True)
        self.load_apecroot(line)
        # self.bvapec_fix_some_param(fix=True)
        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter(fix=True)
        #self.bvapec_fix_some_param(fix=False)
        self.fit_error(self.error_str,True,'nxb1')
        self.result_pack()
        self.savemod(f'bright_{line}')
        #self.plotting(modname=f'bright_{line}')
        # fig.savefig(f"./figure/{region}_simultaneous_fitting.pdf",dpi=300,transparent=True)
        # Xset.save('center_for_cont.xcm')

# ? : plotつかう
    def result_plot_each_fw(self,error_calc=False):
        self.identifiers = ['1000','2000','3000','4000','simultaneous']
        self.fwname      = ['OPEN', 'OBF', 'ND', 'Be', 'All']
        self.fig = plt.figure(figsize=(12, 8))
        self.ax  = self.fig.add_subplot(431) # redshift
        self.ax2 = self.fig.add_subplot(432) # temperature
        self.ax3 = self.fig.add_subplot(433) # abundance
        self.ax4 = self.fig.add_subplot(434) # velocity
        self.ax5 = self.fig.add_subplot(435) # w norm
        self.ax6 = self.fig.add_subplot(436) # z norm
        self.ax7 = self.fig.add_subplot(437) # w fwhm
        self.ax8 = self.fig.add_subplot(438) # z fwhm
        self.ax9 = self.fig.add_subplot(439) # w/z ratio
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax3.grid(linestyle='dashed')
        self.ax4.grid(linestyle='dashed')
        self.ax5.grid(linestyle='dashed')
        self.ax6.grid(linestyle='dashed')
        self.ax7.grid(linestyle='dashed')
        self.ax8.grid(linestyle='dashed')
        self.ax9.grid(linestyle='dashed')
        self.ax.set_title(r'Redshift')
        self.ax2.set_title(r'Temperature')
        self.ax3.set_title(r'Abundance')
        self.ax4.set_title(r'Velocity')
        self.ax5.set_title(r'W norm')
        self.ax6.set_title(r'Z norm')
        self.ax7.set_title(r'W FWHM')
        self.ax8.set_title(r'Z FWHM')
        self.ax9.set_title(r'W/Z ratio')
        self.ax.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax2.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax3.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax4.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax5.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax6.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax7.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax8.set_xticks([0,1,2,3,4],labels=self.fwname)
        self.ax9.set_xticks([0,1,2,3,4],labels=self.fwname)

        with h5py.File("55Fe.hdf5", 'r') as f:
            for num,identifier in enumerate(self.identifiers):
                if error_calc==True:
                    self.ax.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5,label='55Fe')
                    self.ax2.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['kT']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['kT']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax3.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax4.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax5.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax6.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'{identifier}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='black',fmt="o",markersize=5, capsize=5)
                    if f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...] < 0:
                        print(identifier)
                        print(f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...])
                    if -f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...] < 0:
                        print(identifier)
                        print(-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...])
                    else:
                        self.ax7.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                    print(np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)))
                    if f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...] < 0:
                        pass
                    if -f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...] < 0:
                        pass
                    else:
                        self.ax8.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                else:
                    self.ax.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],color='black')
                    self.ax2.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],color='black')
                    self.ax3.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='black')
                    self.ax4.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],color='black')
                    self.ax5.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='black')
                    self.ax6.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],color='black')
                    self.ax7.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='black')
                    self.ax8.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='black')
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        with h5py.File("mxs.hdf5", 'r') as f:
            for num,identifier in enumerate(self.identifiers):
                if error_calc==True:
                    self.ax.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5,label='mxs')
                    self.ax2.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['kT']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['kT']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax3.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax4.errorbar(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['em'][...],f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax5.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['em'][...],f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax6.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['norm']['em'][...],f[f'{identifier}/fitting_result']['3/zgauss']['norm']['ep'][...])),color='blue',fmt="o",markersize=5, capsize=5)
                    self.ax7.errorbar(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)
                    if f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...] < 0:
                        pass
                    if -f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...] < 0:
                        pass
                    else:
                        self.ax8.errorbar(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='blue',fmt="o",markersize=5, capsize=5)
                else:
                    self.ax.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Redshift']['value'][...],color='blue')
                    self.ax2.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['kT']['value'][...],color='blue')
                    self.ax3.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Abundanc']['value'][...],color='blue')
                    self.ax4.scatter(num,f[f'{identifier}/fitting_result']['2/bapec']['Velocity']['value'][...],color='blue')
                    self.ax5.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['norm']['value'][...],color='blue')
                    self.ax6.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['norm']['value'][...],color='blue')
                    self.ax7.scatter(num,f[f'{identifier}/fitting_result']['4/zgauss_4']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='blue')
                    self.ax8.scatter(num,f[f'{identifier}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,color='blue')

            self.fig.align_labels()
            self.fig.tight_layout()
            self.fig.savefig(f'./figure/pixel_by_pixel.png',dpi=300,transparent=True)
            plt.show()

    def result_plot_each_fw2(self, error_calc=True, region='center', line='wz', rebin=1):
        self.line = line
        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}', f'2000_{region}_{line}_rebin_{rebin}', f'3000_{region}_{line}_rebin_{rebin}', f'4000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN','OPEN', 'OBF', 'ND', 'Be', 'All']
        self._setup_figure_grid()
        self._set_titles_and_labels()

        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN(before recycle)']
        with h5py.File('ghf_no_cor_before_recycle.hdf5', 'r') as f:
            self._plot_data(f, color='black', label='no cor', error_calc=error_calc)

        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}', f'2000_{region}_{line}_rebin_{rebin}', f'3000_{region}_{line}_rebin_{rebin}', f'4000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN(after recycle)', 'OBF', 'ND', 'Be']
        with h5py.File('ghf_no_cor_after_recycle.hdf5', 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=1)

        self.identifiers = [f'simultaneous_{region}_{line}_rebin_{rebin}']
        self.fwname = ['All']
        with h5py.File('ghf_no_cor_sim.hdf5', 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=5, z_plot=False)


        offset = 0.2
        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN(before recycle)']
        with h5py.File('ghf_lin_before_recycle.hdf5', 'r') as f:
            self._plot_data(f, color='red', label='gh linear', error_calc=error_calc, start_num=0+offset)

        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}', f'2000_{region}_{line}_rebin_{rebin}', f'3000_{region}_{line}_rebin_{rebin}', f'4000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN(after recycle)', 'OBF', 'ND', 'Be']
        with h5py.File('ghf_lin_after_recycle.hdf5', 'r') as f:
            self._plot_data(f, color='red', label='None', error_calc=error_calc, start_num=1+offset)

        self.identifiers = [f'simultaneous_{region}_{line}_rebin_{rebin}']
        self.fwname = ['All']
        with h5py.File('ghf_lin_sim.hdf5', 'r') as f:
            self._plot_data(f, color='red', label='None', error_calc=error_calc, start_num=5+offset, z_plot=False)



        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}', f'2000_{region}_{line}_rebin_{rebin}', f'3000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN(after recycle)', 'OBF', 'ND', 'Be']
        with h5py.File('mxs_low_cnt.hdf5', 'r') as f:
            self._plot_data(f, color='blue', label='None', error_calc=error_calc, start_num=1+offset+0.2)


        self.identifiers = [f'4000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['Be']
        with h5py.File('mxs_low_cnt.hdf5', 'r') as f:
            self._plot_data(f, color='blue', label='MXS', error_calc=error_calc, start_num=4+offset+0.2)

        self.identifiers = [f'simultaneous_{region}_{line}_rebin_{rebin}']
        self.fwname = ['All']
        with h5py.File('mxs_low_cnt.hdf5', 'r') as f:
            self._plot_data(f, color='blue', label='None', error_calc=error_calc, start_num=5+offset+0.2, z_plot=False)
        # with h5py.File("mxs_cal_without_55Fe_filter.hdf5", 'r') as f:
        #     self._plot_data(f, color='blue', label='mxs', error_calc=error_calc)

        self.identifiers = ['bright_wz']
        self.fwname.append('Max')
        with h5py.File("max_center.hdf5", 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=6)

        self.identifiers = ['bright_wz']
        self.fwname.append('Brt')
        with h5py.File("bright_center.hdf5", 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=7)
        self.fwname = ['OPEN 1','OPEN 2', 'OBF', 'ND', 'Be', 'All', 'Max', 'Brt']
        self._set_titles_and_labels()
        self.fig.align_labels()
        self.fig.tight_layout()
        self.ax[0].legend(loc='lower right',framealpha=0.5)
        if region == 'center':
            #self.ax[5].set_ylim(0, 1e-4)
            #self.ax[7].set_ylim(-0.9, 15)
            pass
        if region == 'all':
            self.ax[7].set_ylim(-0.9, 15)
        plt.show()
        self.fig.savefig(f'./figure/fitting_result_{region}_{line}.png', dpi=300, transparent=True)

    def result_plot_each_fw4(self, error_calc=True, region='center', line='wz', rebin=1):
        self.line = line
        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}',f'S_other_{region}_{line}_rebin_{rebin}', f'S_all_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN', 'other', 'All', 'brmax', 'max', 'brt']
        self._setup_figure_grid()
        self._set_titles_and_labels()

        self.fwname = ['OPEN', 'other', 'All']
        with h5py.File(self.savefile, 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=0, z_plot=False)


        self.identifiers = [f'brmax_{region}_{line}_rebin_{rebin}_zgaus',f'max_{region}_{line}_rebin_{rebin}_zgaus', f'brt_{region}_{line}_rebin_{rebin}_zgaus']
        self.fwname = ["brmax", "max", "brt"]
        with h5py.File("fe_data.hdf5", 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=3, z_plot=True, z_add=1)

        self.identifiers = [f'S_all_{region}_{line}_rebin_{rebin}']
        self.fwname = ['All']
        with h5py.File(self.savefile, 'r') as f:
            self._z_plot(f,ax=self.ax[0])

        # self.identifiers = [f'brmax_{region}_{line}_rebin_{rebin}_zgaus',f'max_{region}_{line}_rebin_{rebin}_zgaus', f'brt_{region}_{line}_rebin_{rebin}_zgaus']
        # with h5py.File("fe_data.hdf5", 'r') as f:
        #     self._z_plot(f,ax=self.ax[0])
        self.fig.align_labels()
        self.fig.tight_layout()
        plt.show()
        self.fig.savefig(f'./figure/fitting_result_{region}_{line}.png', dpi=300, transparent=True)

    def result_plot_each_red(self, error_calc=True, region='center', line='wz', rebin=1):
        self.line = line
        self.fwname = ['All']
        cols = ['black', 'red', 'blue', 'green']
        start_num = 0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4))
        for e,region in enumerate(['center', 'outer', 'all']):
            self.identifiers = [f'S_all_{region}_{line}_rebin_{rebin}']
            with h5py.File(self.savefile, 'r') as f:
                self._z_plot(f,start_num=start_num,color=cols[e],label=region,ax=self.ax)
                start_num += 0.2
        self.fig.align_labels()
        self.fig.tight_layout()
        self.ax.grid(linestyle='dashed')
        self.ax.legend(loc='lower right',framealpha=0.5)

        plt.show()
        self.fig.savefig(f'./figure/fitting_result_redshift.png', dpi=300, transparent=True)

    def _z_plot(self, file, color='black', label='None', error_calc=True, start_num=0, ax=None):
        labels = ['OPEN', 'OBF', 'ND', 'Be', 'brmax', 'max', 'brt']
        
        ax.set_xticks(range(0, 7), labels=labels, rotation=45)
        
        for num, identifier in enumerate(self.identifiers):
            print(f'Plotting {identifier}...')
            if self.line == 'wz':
                redshift = file[f'{identifier}/fitting_result/2/bvapec/Redshift']
                redshift_2 = file[f'{identifier}/fitting_result/2/bvapec/Redshift_2']
                redshift_3 = file[f'{identifier}/fitting_result/2/bvapec/Redshift_3']
                redshift_4 = file[f'{identifier}/fitting_result/2/bvapec/Redshift_4']
                data_pairs = [
                    (ax, redshift, 'Redshift'), (ax, redshift_2, 'Redshift_2'),
                    (ax, redshift_3, 'Redshift_3'), (ax, redshift_4, 'Redshift_4')
                ]
        
        for e, (ax, data, name) in enumerate(data_pairs):
            
            if 'em' not in data:
                value = data['value'][...]
                em, ep = np.zeros_like(value), np.zeros_like(value)
            else:
                value, em, ep = data['value'][...], -data['em'][...], data['ep'][...]

            if name == 'W FWHM' or name == 'Z FWHM':
                value = value * 2 * np.sqrt(2 * np.log(2)) * 1e3
                em = em * 2 * np.sqrt(2 * np.log(2)) * 1e3
                ep = ep * 2 * np.sqrt(2 * np.log(2)) * 1e3

            # Replace negative errors with zero
            em = np.where(em < 0, 0, em)
            ep = np.where(ep < 0, 0, ep)
            

            if error_calc:
                yerr = np.vstack((em, ep))
                if e == 0:
                    ax.errorbar(num + start_num, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5, label=label)
                else:
                    ax.errorbar(num + start_num, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5)
                num = num + 1
            else:
                ax.scatter(num + start_num, value, color=color)
        # 各redshiftの最大値と最小値を比較して系統誤差を計算
        values = [redshift['value'][...], redshift_2['value'][...], redshift_3['value'][...], redshift_4['value'][...]]
        max_value = np.max(values)
        min_value = np.min(values)
        sys_em = max_value - min_value  # 最大値と最小値の差を系統誤差として定義

        # 系統誤差をプリント
        print(values)
        print(f"Systematic error for redshift: {sys_em}")
        print(f"Systematic error for redshift: {sys_em*5900} eV @5.9 keV")
        print(f"bulk motion: {sys_em*2.998e5} km/s @6.7 keV")

    def _setup_figure_grid(self):
        """Set up figure grid with specific subplot layout."""
        if self.line == 'wz':
            self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 10))
        if self.line == 'w':
            self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 6.6))
        if self.line == 'None':
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6.6))
        self.ax = self.axes.ravel()
        for ax in self.ax:
            ax.grid(linestyle='dashed')

    def _set_titles_and_labels(self):
        """Set titles and x-axis labels for each subplot."""

        if self.line == 'wz':
            titles = ['Redshift', 'Temperature (keV)', r'$Z_{Fe}$', r'$\sigma_v \rm \ (km/s)$', 
                        'W norm', 'Z norm', 'W FWHM (eV)', 'Z FWHM (eV)', 'W/Z ratio']
        if self.line == 'w':
            titles = ['Redshift', 'Temperature (keV)', r'$Z_{Fe}$', r'$\sigma_v \rm \ (km/s)$', 
                        'W norm', 'W FWHM (eV)']
        if self.line == 'None':
            titles = ['Redshift', 'Temperature (keV)', r'$Z_{Fe}$', r'$\sigma_v \rm \ (km/s)$']
        for i, ax in enumerate(self.ax):
            ax.set_title(titles[i])
            ax.set_xticks(range(len(self.fwname)), labels=self.fwname, rotation=45)

    def _plot_data(self, file, color, label, error_calc, start_num=0, z_plot=True, z_add=1):
        """Plot data from HDF5 file with error bars if requested."""
        for num, identifier in enumerate(self.identifiers):
            print(f'Plotting {identifier}...')
            if self.line == 'wz':
                redshift = file[f'{identifier}/fitting_result/2/bvapec/Redshift']
                kT = file[f'{identifier}/fitting_result/2/bvapec/kT']
                abundance = file[f'{identifier}/fitting_result/2/bvapec/Fe']
                velocity = file[f'{identifier}/fitting_result/2/bvapec/Velocity']
                apec_norm = file[f'{identifier}/fitting_result/2/bvapec/norm']
                w_norm = file[f'{identifier}/fitting_result/4/zgauss_4/norm']
                z_norm = file[f'{identifier}/fitting_result/3/zgauss/norm']
                w_sigma = file[f'{identifier}/fitting_result/4/zgauss_4/Sigma']
                z_sigma = file[f'{identifier}/fitting_result/3/zgauss/Sigma']
                
                # Calculate W norm / Z norm ratio and its error
                w_z_ratio_value = w_norm['value'][...] / np.where(z_norm['value'][...] != 0, z_norm['value'][...], np.nan)
                w_z_ratio_err_ep = (w_norm['value'][...] + w_norm['ep'][...])/(z_norm['value'][...] + z_norm['em'][...]) - w_z_ratio_value
                w_z_ratio_err_em = w_z_ratio_value - (w_norm['value'][...] + w_norm['em'][...])/(z_norm['value'][...] + z_norm['ep'][...])
                print('#'*50)
                print('identifier:', identifier)
                print('W/Z ratio:', w_z_ratio_value, w_z_ratio_err_ep, w_z_ratio_err_em)
                print('w:', w_norm['value'][...], w_norm['ep'][...], w_norm['em'][...])
                print('z:', z_norm['value'][...], z_norm['ep'][...], z_norm['em'][...])
                print('#'*50)
                if z_plot == True:
                    data_pairs = [
                        (self.ax[0], redshift, 'Redshift'), (self.ax[1], kT, 'Temperature'),
                        (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity'),
                        (self.ax[4], w_norm, 'W norm'), (self.ax[5], z_norm, 'Z norm'),
                        (self.ax[6], w_sigma, 'W FWHM'), (self.ax[7], z_sigma, 'Z FWHM'),
                        (self.ax[8], (w_z_ratio_value, w_z_ratio_err_em, w_z_ratio_err_ep), 'W/Z ratio')  # Adding W/Z ratio plot
                        # (self.ax[9], apec_norm, 'APEC norm')
                    ]
                if z_plot == False:
                    data_pairs = [
                        (self.ax[1], kT, 'Temperature'),
                        (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity'),
                        (self.ax[4], w_norm, 'W norm'), (self.ax[5], z_norm, 'Z norm'),
                        (self.ax[6], w_sigma, 'W FWHM'), (self.ax[7], z_sigma, 'Z FWHM'),
                        (self.ax[8], (w_z_ratio_value, w_z_ratio_err_em, w_z_ratio_err_ep), 'W/Z ratio')  # Adding W/Z ratio plot
                        #(self.ax[9], apec_norm, 'APEC norm')
                    ]

            if self.line == 'w':
                redshift = file[f'{identifier}/fitting_result/2/bvapec/Redshift']
                kT = file[f'{identifier}/fitting_result/2/bvapec/kT']
                abundance = file[f'{identifier}/fitting_result/2/bvapec/Fe']
                velocity = file[f'{identifier}/fitting_result/2/bvapec/Velocity']
                w_norm = file[f'{identifier}/fitting_result/3/zgauss/norm']
                w_sigma = file[f'{identifier}/fitting_result/3/zgauss/Sigma']
                
                if z_plot == True:
                    data_pairs = [
                    (self.ax[0], redshift, 'Redshift'), (self.ax[1], kT, 'Temperature'),
                    (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity'),
                    (self.ax[4], w_norm, 'W norm'),
                    (self.ax[5], w_sigma, 'W FWHM')
                ]
                if z_plot == False:
                    data_pairs = [
                    (self.ax[1], kT, 'Temperature'),
                    (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity'),
                    (self.ax[4], w_norm, 'W norm'),
                    (self.ax[5], w_sigma, 'W FWHM')
                ]

            if self.line == 'None':
                redshift = file[f'{identifier}/fitting_result/2/bvapec/Redshift']
                kT = file[f'{identifier}/fitting_result/2/bvapec/kT']
                abundance = file[f'{identifier}/fitting_result/2/bvapec/Fe']
                velocity = file[f'{identifier}/fitting_result/2/bvapec/Velocity']
                
                if z_plot == True:
                    data_pairs = [
                    (self.ax[0], redshift, 'Redshift'), (self.ax[1], kT, 'Temperature'),
                    (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity')
                ]
                if z_plot == False:
                    data_pairs = [
                    (self.ax[1], kT, 'Temperature'),
                    (self.ax[2], abundance, 'Abundance'), (self.ax[3], velocity, 'Velocity')
                ]

            
            for ax, data, name in data_pairs:
                if name == 'W/Z ratio':
                    value, em, ep = data  # Special case for W/Z ratio with manual error
                
                else:
                    if 'em' not in data:
                        value = data['value'][...]
                        em, ep = np.zeros_like(value), np.zeros_like(value)
                    else:
                        value, em, ep = data['value'][...], -data['em'][...], data['ep'][...]

                if name == 'W FWHM' or name == 'Z FWHM':
                    value = value * 2 * np.sqrt(2*np.log(2)) * 1e3
                    em = em * 2 * np.sqrt(2*np.log(2)) * 1e3
                    ep = ep * 2 * np.sqrt(2*np.log(2)) * 1e3

                # Replace negative errors with zero
                em = np.where(em < 0, 0, em)
                ep = np.where(ep < 0, 0, ep)
                
                if error_calc:
                    yerr = np.vstack((em, ep))
                    if z_plot == True and name == 'Redshift':
                        if label != 'None':
                            ax.errorbar(num+start_num+z_add, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5, label=label)
                        else:
                            ax.errorbar(num+start_num+z_add, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5)
                    else:
                        if label != 'None':
                            ax.errorbar(num+start_num, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5, label=label)
                        else:
                            ax.errorbar(num+start_num, value, yerr=yerr, color=color, fmt="o", markersize=5, capsize=5)
                else:
                    ax.scatter(num+start_num, value, color=color)

    def result_plot_each_fw3(self, error_calc=True, region='center', line='wz', rebin=1):
        self.line = line
        self.identifiers = [f'1000_{region}_{line}_rebin_{rebin}', f'2000_{region}_{line}_rebin_{rebin}', f'3000_{region}_{line}_rebin_{rebin}', f'4000_{region}_{line}_rebin_{rebin}']
        self.fwname = ['OPEN', 'other filter', 'all']
        self._setup_figure_grid()
        self._set_titles_and_labels()

        self.identifiers = ['bright_wz']
        with h5py.File(self.savefile, 'r') as f:
            self._plot_data(f, color='black', label='None', error_calc=error_calc, start_num=7)
        self.fwname = ['OPEN 1','OPEN 2', 'OBF', 'ND', 'Be', 'All', 'Max', 'Brt']
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6.6))
        self._set_titles_and_labels()
        self.fig.align_labels()
        self.fig.tight_layout()
        self.ax[0].legend(loc='lower right',framealpha=0.5)
        self.ax.grid(linestyle='dashed')
        if region == 'center':
            #self.ax[5].set_ylim(0, 1e-4)
            #self.ax[7].set_ylim(-0.9, 15)
            pass
        if region == 'all':
            self.ax[7].set_ylim(-0.9, 15)
        plt.show()
        self.fig.savefig(f'./figure/fitting_result_{region}_{line}.png', dpi=300, transparent=True)

    def result_plot_pixel_by_pixel_cal_line(self):
        self.lines = ["CrKa"]
        
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        self.ax.set_xticks([0],labels=self.lines)
        self.ax2.set_xticks([0],labels=self.lines)
        self.ax.set_ylabel('Shift (eV) (@5.9keV)')
        self.ax2.set_ylabel('FWHM (eV) (@6.0keV)')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        E = 5.9e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File(self.savefile, 'r') as f:
            for num,line in enumerate(self.lines):

                if f'{line}' in f.keys():
                    self.ax.errorbar(num,f[f'{line}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'{line}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'{line}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    #self.ax2.errorbar(num,f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
                print(line)
                print(f'shift = {f[f'{line}/fitting_result']['1/zashift']['Redshift']['value'][...]*E} + {f[f'{line}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E} - {f[f'{line}/fitting_result']['1/zashift']['Redshift']['em'][...]*E} eV')
                #print(f'fwhm = {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm} + {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm} - {f[f'{line}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm} eV')
        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_pixel_by_pixel_cal(self,line='MnKa'):
        #self.all_pixels = ["00", "17", "18", "35"]
        #self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.cnt_rate = np.array([3515, 3507, 3209, 19032, 14191, 18636,14525,15297,8378,8805,8371,9280,18888,13686,19192,13417,18715,12774,8626,8140,3194,2826,3653])/155.48e+3
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        # self.ax.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        # self.ax2.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        if line == 'MnKa':
            self.ax.set_ylabel('Shift (eV)\n @5.9 keV')
            E = 5.9e+3
        elif line == 'MnKb':
            self.ax.set_ylabel('Shift (eV)\n @6.4 keV')
            E = 6.4e+3
        elif line == 'CrKa':
            self.ax.set_ylabel('Shift (eV)\n @6.4 keV')
            E = 5.4e+3
        self.ax2.set_ylabel('FWHM (eV)\n @6.0 keV')
        self.ax2.set_xlabel('Pixel ID')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File(self.savefile, 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    # print(f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...],f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...], f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...])
                    if f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...] > 0:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                        #self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,color='black',marker='o',label='MnKa')
                    else:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                        #self.ax.scatter(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,color='black',marker='o',label='MnKa')
                    print(f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...], f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...])
                    if num == 0:
                        self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5, label='Fitting result')
                        #self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,color='black',marker='o',label='Fitting result')
                    else:
                        #self.ax2.scatter(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,color='black',marker='o',label='Fitting result')
                        self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')

        # file = '/Users/keitatanaka/xraydata/caldb/data/xrism/resolve/bcf/response/xa_rsl_rmfparam_20190101v006.fits.gz'
        # f = fits.open(file)
        # energy = f[1].data['ENERGY']
        # fwhm_MnKa = []
        # pixel = []
        # for i in range(36):
        #     fwhm_e = f[1].data[f'PIXEL{i}'][:]
        #     #plt.plot(energy,fwhm,label=f'pixel {i}')
        #     fwhm_MnKa.append(fwhm_e[np.abs(energy-6000).argmin()])
        #     pixel.append(i)
        # self.ax2.scatter(pixel,fwhm_MnKa,color='red',marker='s',alpha=0.5, label='CALDB 8')
        #plt.xlim(2000,8000)
        # self.ax2.legend()
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='red')

        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel_{line}.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_pixel_by_pixel_cal_for_mxs(self,line='MnKa'):
        #self.all_pixels = ["00", "17", "18", "35"]
        #self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.cnt_rate = np.array([3515, 3507, 3209, 19032, 14191, 18636,14525,15297,8378,8805,8371,9280,18888,13686,19192,13417,18715,12774,8626,8140,3194,2826,3653])/155.48e+3
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(211)
        self.ax.grid(linestyle='dashed')
        if line == 'MnKa':
            self.ax.set_ylabel('Shift (eV)\n @5.9 keV')
            E = 5.9e+3
        elif line == 'MnKb':
            self.ax.set_ylabel('Shift (eV)\n @6.4 keV')
            E = 6.4e+3
        elif line == 'CrKa':
            self.ax.set_ylabel('Shift (eV)\n @6.4 keV')
            E = 5.4e+3
        fwhm = np.array([])
        shift = np.array([])
        y1 = np.array([23,21,19,11,9])
        y2 = np.array([24, 22, 20, 10, 13, 14])
        y3 = np.array([25, 26, 18, 17, 15, 16])
        y4 = np.array([0, 7, 8, 33, 34, 35])
        y5 = np.array([2, 4, 6, 32, 31, 28])
        y6 = np.array([30, 29, 1, 3, 5, 27])

        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File(self.savefile, 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    if int(pixel) in y1:
                        self.ax.errorbar(1,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                    if int(pixel) in y2:
                        self.ax.errorbar(2,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    if int(pixel) in y3:
                        self.ax.errorbar(3,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    if int(pixel) in y4:
                        self.ax.errorbar(4,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    if int(pixel) in y5:
                        self.ax.errorbar(5,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    if int(pixel) in y6:
                        self.ax.errorbar(6,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)


        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='red')

        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel_{line}.png',dpi=300,transparent=True)
        plt.show()

    def result_plot_pixel_by_pixel_cal_MnKa_MnKb(self):
        self.all_pixels = ["00", "17", "18", "35"]
        #self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.all_pixels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax.grid(linestyle='dashed')
        self.ax2.grid(linestyle='dashed')
        # self.ax.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        # self.ax2.set_xticks([0,1,2,3],labels=['00', '17', '18', '35'])
        self.ax.set_ylabel('Shift (eV)\n @5.9 keV,6.4 keV')
        self.ax2.set_ylabel('FWHM (eV)\n @6.0 keV')
        self.ax2.set_xlabel('Pixel ID')
            # for spine in self.ax.spines.values():
            #     spine.set_linewidth(1.5)
            # self.ax.tick_params(axis='both',direction='in',width=1.5)
        fwhm = np.array([])
        shift = np.array([])
        E = 5.9e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File('pixel_by_pixel_MnKa.hdf5', 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    if num == 0:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5,label='MnKa')
                    else:    
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),color='black',fmt="o",markersize=5, capsize=5)
                    self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),color='black',fmt="o",markersize=5, capsize=5)
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='black')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='black')

        fwhm = np.array([])
        shift = np.array([])
        E = 6.4e+3
        sigma2fwhm = 2*np.sqrt(2*np.log(2))*1e3
        with h5py.File('pixel_by_pixel_MnKb.hdf5', 'r') as f:
            for num,pixel in enumerate(self.all_pixels):

                if f'pixel{pixel}' in f.keys():
                    num = int(pixel)
                    shift = np.append(shift,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E)
                    fwhm = np.append(fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm)
                    if num == 0:
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),fmt="o",markersize=5, capsize=5,color='blue',label='MnKb')
                    else:    
                        self.ax.errorbar(num,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['value'][...]*E,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['em'][...]*E,f[f'pixel{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][...]*E)),fmt="o",markersize=5, capsize=5,color='blue')
                    self.ax2.errorbar(num,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['value'][...]*sigma2fwhm,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['em'][...]*sigma2fwhm,f[f'pixel{pixel}/fitting_result']['2/gsmooth']['Sig_6keV']['ep'][...]*sigma2fwhm)),fmt="o",markersize=5, capsize=5,color='blue')
                #self.ax8.errorbar(num,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['value'][...]*2*np.sqrt(2*np.log(2))*1e3,yerr=np.vstack((-f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['em'][...]*2*np.sqrt(2*np.log(2))*1e3,f[f'pixel{pixel}/fitting_result']['3/zgauss']['Sigma']['ep'][...]*2*np.sqrt(2*np.log(2))*1e3)),color='black',fmt="o",markersize=5, capsize=5)
        print(f'shift avg = {np.average(shift)}')
        print(f'fwhm avg = {np.average(fwhm)}')
        print(f'shift std = {np.std(shift)}')
        print(f'fwhm std = {np.std(fwhm)}')
        self.ax.hlines(np.average(shift),-1,36,linestyle='-.',color='blue')
        # self.ax.hlines(np.average(shift)+np.std(shift),-1,36,linestyle='-.',color='red')
        # self.ax.hlines(np.average(shift)-np.std(shift),-1,36,linestyle='-.',color='red')
        self.ax2.hlines(np.average(fwhm),-1,36,linestyle='-.',color='blue')
        self.ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        self.fig.align_labels()
        self.fig.tight_layout()
        self.fig.savefig(f'./figure/pixel_by_pixel_MnKa_MnKb.png',dpi=300,transparent=True)
        plt.show()

    def get_parameters_name(self,model):
        parameters = {}
        m = model
        model_components = m.componentNames
        for counter,name in enumerate(model_components):
            print('-----------------')
            print(f'model name: {name}')
            parameters[counter+1]       = {}
            parameters[counter+1][name] = {} 
            model_parameters            = m.__getattribute__(name).parameterNames
            for p in model_parameters:
                print(f'parameter name: {p}')
                parameters[counter+1][name][p] = {}
        return parameters

    def result_pack(self,detector='nxb1',model=None,group=1,modname=""):
        if model == None:
            model=self.model
        fit_result = self.get_parameters_name(model)
        counter = 0
        for e in range(0,len(fit_result.keys())):
            mod = str(list(fit_result[e+1].keys())[0])
            print(mod)
            for param in list(fit_result[e+1][mod].keys()):
                print(param)
                fit_result[e+1][f'{mod}'][param]['value'] = AllModels(group,modname)(counter+1).values[0]
                print(model.__getattribute__(mod).__getattribute__(param).link)
                if model.__getattribute__(mod).__getattribute__(param).frozen == False and model.__getattribute__(mod).__getattribute__(param).link == '':
                    fit_result[e+1][mod][param]['em']    = AllModels(group,modname)(counter+1).error[0] - AllModels(group,modname)(counter+1).values[0]
                    fit_result[e+1][mod][param]['ep']    = AllModels(group,modname)(counter+1).error[1] - AllModels(group,modname)(counter+1).values[0]
                    print(fit_result[e+1][mod][param]['value'],fit_result[e+1][mod][param]['em'],fit_result[e+1][mod][param]['ep'])
                else:
                    print(fit_result[e+1][mod][param]['value'])
                counter += 1
        if hasattr(self, 'model_background'):
            fit_result['bg'] = {}
            fit_result['bg']['pow'] = {}
            fit_result['bg']['pow']['norm'] = {}
            fit_result['bg']['pow']['norm']['value'] = AllModels(group,detector)(1).values[0]
            fit_result['bg']['pow']['norm']['em'] =AllModels(group,detector)(1).error[0] - AllModels(1,detector)(1).values[0]
            fit_result['bg']['pow']['norm']['ep'] = AllModels(group,detector)(1).error[1] - AllModels(1,detector)(1).values[0]
        self.fit_result = fit_result
        self.statistic = Fit.statistic
        self.dof = Fit.dof

    def result_pack_ssm(self):
        results = {}
        self.result_pack(detector='nxb1', model=self.model_center, group=1, modname="")
        res1 = self.fit_result
        self.result_pack(detector='nxb1', model=self.model_outer, group=1, modname="outer")
        res2 = self.fit_result
        results["center"] = res1
        results["outer"]  = res2
        self.fit_result_all = results
        print(results)

    def result_pack_only_z_ssm(self,model,region,modname,group=1):
        fit_result = self.get_parameters_name(model)
        counter = 0
        for e in range(0,len(fit_result.keys())):
            mod = str(list(fit_result[e+1].keys())[0])
            print(mod)
            for param in list(fit_result[e+1][mod].keys()):
                print(param)
                fit_result[e+1][f'{mod}'][param]['value'] = AllModels(group,modname)(counter+1).values[0]
                print(model.__getattribute__(mod).__getattribute__(param).link)
                if model.__getattribute__(mod).__getattribute__(param).frozen == False and model.__getattribute__(mod).__getattribute__(param).link == '':
                    fit_result[e+1][mod][param]['em']    = AllModels(group,modname)(counter+1).error[0] - AllModels(group,modname)(counter+1).values[0]
                    fit_result[e+1][mod][param]['ep']    = AllModels(group,modname)(counter+1).error[1] - AllModels(group,modname)(counter+1).values[0]
                    print(fit_result[e+1][mod][param]['value'],fit_result[e+1][mod][param]['em'],fit_result[e+1][mod][param]['ep'])
                else:
                    print(fit_result[e+1][mod][param]['value'])
                counter += 1
        self.fit_result_all[f"{region}"][2]['bvapec'].setdefault(f'Redshift_{group}', {})
        keys = ['value', 'ep', 'em']
        for key in keys:
            self.fit_result_all[f"{region}"][2]['bvapec'][f'Redshift_{group}'][key] = fit_result[2]['bvapec'][f'Redshift'][key]

    def result_pack_only_z(self,model=None,group=1):
        if model == None:
            model=self.model
        fit_result = self.get_parameters_name(model)
        counter = 0
        for e in range(0,len(fit_result.keys())):
            mod = str(list(fit_result[e+1].keys())[0])
            print(mod)
            for param in list(fit_result[e+1][mod].keys()):
                print(param)
                fit_result[e+1][f'{mod}'][param]['value'] = AllModels(group)(counter+1).values[0]
                print(model.__getattribute__(mod).__getattribute__(param).link)
                if model.__getattribute__(mod).__getattribute__(param).frozen == False and model.__getattribute__(mod).__getattribute__(param).link == '':
                    fit_result[e+1][mod][param]['em']    = AllModels(group)(counter+1).error[0] - AllModels(group)(counter+1).values[0]
                    fit_result[e+1][mod][param]['ep']    = AllModels(group)(counter+1).error[1] - AllModels(group)(counter+1).values[0]
                    print(fit_result[e+1][mod][param]['value'],fit_result[e+1][mod][param]['em'],fit_result[e+1][mod][param]['ep'])
                else:
                    print(fit_result[e+1][mod][param]['value'])
                counter += 1
        self.fit_result[2]['bvapec'].setdefault(f'Redshift_{group}', {})
        keys = ['value', 'ep', 'em']
        for key in keys:
            self.fit_result[2]['bvapec'][f'Redshift_{group}'][key] = fit_result[2]['bvapec'][f'Redshift'][key]

    def get_delc(self,spectrumnum=1):
        Plot("delc")
        self.xres=Plot.x(spectrumnum)
        self.yres=Plot.y(spectrumnum)
        self.xres_e=Plot.xErr(spectrumnum)
        self.yres_e=Plot.yErr(spectrumnum)
        return self.xres, self.yres, self.xres_e, self.yres_e

    def get_del(self,spectrumnum=1):
        Plot("del")
        self.xdel=Plot.x(spectrumnum)
        self.ydel=Plot.y(spectrumnum)
        self.xdel_e=Plot.xErr(spectrumnum)
        self.ydel_e=Plot.yErr(spectrumnum)
        return self.xdel, self.ydel, self.xdel_e, self.ydel_e

    def get_model(self):
        Plot('data delc')
        Plot.add = True
        self.y=Plot.model()
        self.ys_comps=[]
        comp_N=1
        while(True):
            try:
                ys_tmp = Plot.addComp(comp_N,1,1)
                comp_N += 1
                # execlude components with only 0
                if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                    continue
                self.ys_comps.append(ys_tmp)
            except:
                break  

    def fit_error(self,error='1.0 2,3,4,5,6,8,10,det:1',error_calc=True,detector='nxb1'):
        Fit.perform()
        self.model = AllModels(1)
        if detector != False and detector != None:
            self.model_background = AllModels(1,detector)
        if error_calc == True:
            Fit.error(error)
        self.get_model()
        self.get_delc(1)
        return self.xres, self.yres, self.xres_e, self.yres_e

    def get_model_comps(self,plotgroup=1):
        self.y = Plot.model(plotgroup)
        self.ys_comps=[]
        comp_N=1
        while(True):
            try:
                ys_tmp = Plot.addComp(comp_N,plotgroup,1)
                comp_N += 1
                # execlude components with only 0
                if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                    continue
                self.ys_comps.append(ys_tmp)
            except:
                break  
        Plot("delc")
        self.xres=Plot.x(plotgroup,1)
        self.yres=Plot.y(plotgroup,1)
        self.xres_e=Plot.xErr(plotgroup,1)
        self.yres_e=Plot.yErr(plotgroup,1)
        return self.ys_comps, self.xres, self.yres, self.xres_e, self.yres_e  

    def savexcm(self,modname):
        if os.path.exists('xcm'):
            pass
        else:
            os.makedirs('xcm')
        if os.path.exists(f'./xcm/{modname}.xcm'):
            os.remove(f'./xcm/{modname}.xcm')
        Xset.save(f'./xcm/{modname}')

    def savemod_multi(self,modname):
        print(modname)
        with h5py.File(self.savefile,'a') as f:
            if modname in f.keys():
                del f[modname]
                print('model is deleted')
            f.create_group(modname)
            f.create_group(f'{modname}/fitting_result')
            f[modname].create_dataset("statistic",data=self.statistic)
            f[modname].create_dataset("dof",data=self.dof)
            self.savexcm(modname)
            f[modname].attrs['xcm_path'] = f'./xcm/{modname}.xcm'
            for multi_key in self.xs.keys():
                f[modname].create_dataset(f"xs/{multi_key}",data=self.xs[multi_key])
                f[modname].create_dataset(f"ys/{multi_key}",data=self.ys[multi_key])
                f[modname].create_dataset(f"xe/{multi_key}",data=self.xe[multi_key])
                f[modname].create_dataset(f"ye/{multi_key}",data=self.ye[multi_key])
                f[modname].create_dataset(f"y/{multi_key}",data=self.y[multi_key])
                f[modname].create_dataset(f"yscomps/{multi_key}",data=self.yscomps[multi_key])
                f[modname].create_dataset(f"xres/{multi_key}",data=self.xres[multi_key])
                f[modname].create_dataset(f"yres/{multi_key}",data=self.yres[multi_key])
                f[modname].create_dataset(f"xres_e/{multi_key}",data=self.xres_e[multi_key])
                f[modname].create_dataset(f"yres_e/{multi_key}",data=self.yres_e[multi_key])

            for model_number in self.fit_result.keys():
                model_components = list(self.fit_result[model_number].keys())
                for model_component in model_components:
                    for k,v in self.fit_result[model_number][model_component].items():
                        print(model_number,model_component,k,v)
                        if str(model_number) not in f[f'{modname}/fitting_result'].keys():
                            f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}')
                        f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}/{k}')
                        f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset('value',data=v['value'])
                        if f'em' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("em",data=v['em'])
                            print('em =', v['em'])
                        if f'ep' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("ep",data=v['ep'])   
                        print(f[modname]['fitting_result'][str(model_number)][model_component].keys()) 

    def save_mod_ssm(self, modname):
        print(modname)
        with h5py.File(self.savefile,'a') as f:
            if modname in f.keys():
                del f[modname]
                print('model is deleted')
            f.create_group(modname)
            f.create_group(f'{modname}/fitting_result')
            f[modname].create_dataset("statistic",data=self.statistic)
            f[modname].create_dataset("dof",data=self.dof)
            self.savexcm(modname)
            f[modname].attrs['xcm_path'] = f'./xcm/{modname}.xcm'
            for multi_key in self.xs.keys():
                f[modname].create_dataset(f"xs/{multi_key}",data=self.xs[multi_key])
                f[modname].create_dataset(f"ys/{multi_key}",data=self.ys[multi_key])
                f[modname].create_dataset(f"xe/{multi_key}",data=self.xe[multi_key])
                f[modname].create_dataset(f"ye/{multi_key}",data=self.ye[multi_key])
                f[modname].create_dataset(f"y/{multi_key}",data=self.y[multi_key])
                f[modname].create_dataset(f"yscomps/{multi_key}",data=self.yscomps[multi_key])
                f[modname].create_dataset(f"xres/{multi_key}",data=self.xres[multi_key])
                f[modname].create_dataset(f"yres/{multi_key}",data=self.yres[multi_key])
                f[modname].create_dataset(f"xres_e/{multi_key}",data=self.xres_e[multi_key])
                f[modname].create_dataset(f"yres_e/{multi_key}",data=self.yres_e[multi_key])

            for reg in ["center", "outer"]:
                f.create_group(f'{modname}/fitting_result/{reg}')
                for model_number in self.fit_result_all[reg].keys():
                    model_components = list(self.fit_result_all[reg][model_number].keys())
                    for model_component in model_components:
                        for k,v in self.fit_result_all[reg][model_number][model_component].items():
                            print(model_number,model_component,k,v)
                            if str(model_number) not in f[f'{modname}/fitting_result/{reg}'].keys():
                                f.create_group(f'{modname}/fitting_result/{reg}/{str(model_number)}/{model_component}')
                            f.create_group(f'{modname}/fitting_result/{reg}/{str(model_number)}/{model_component}/{k}')
                            f[modname]['fitting_result'][f"{reg}"][str(model_number)][model_component][k].create_dataset('value',data=v['value'])
                            if f'em' in v.keys():
                                f[modname]['fitting_result'][f"{reg}"][str(model_number)][model_component][k].create_dataset("em",data=v['em'])
                                print('em =', v['em'])
                            if f'ep' in v.keys():
                                f[modname]['fitting_result'][f"{reg}"][str(model_number)][model_component][k].create_dataset("ep",data=v['ep'])   
                            print(f[modname]['fitting_result'][f"{reg}"][str(model_number)][model_component].keys()) 

    def plotting(self,modname,error=True,x_rng=[5.9,6.2],logging=False,line='None'):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls  = 15
        ps  = 2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.step(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        #ax.plot(xs,ys_comps[1],'-.',label="bapec(Hot)",lw=ps,color='orange')
        ax.plot(xs,ys_comps[0],'-.',label=r"bapec",lw=ps,color='orange')
        #ax.plot(xs,ys_comps[0],'-.',label=r"bapec(Hot)",lw=ps,color='orange')
        if self.line == 'wz':
            ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color='darkblue')
            ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ y",lw=ps,color='mediumblue')
        #ax.plot(xs,ys_comps[3],'-.',label=r"Fe XXV He $\alpha$ x",lw=ps,color='blue')
        if self.line == 'w':
            ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        #ax.plot(xs,ys_comps[2],label=r"$\rm Fe\ He\alpha \ z$",lw=1,color='green')
        #ax.plot(xs,ys_comps[5],'-.',label=r"$\rm Fe\ H \ Ly\alpha 2$",lw=ps,color='brown')
        #ax.plot(xs,ys_comps[6],'-.',label=r"$\rm Fe\ H \ Ly\alpha 1$",lw=ps,color='salmon')
        #ax.plot(xs,ys_comps[7],'-.',label=r"Fe XXV He $\beta$ 2",lw=ps,color='darkgreen')
        #ax.plot(xs,ys_comps[8],'-.',label=r"Fe XXV He $\beta$ 1",lw=ps,color='green')
        
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-2,np.max(yrng)+0.1)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        #ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_multi(self,modname,error=True,x_rng=[5.975, 6.125],logging=False,line='None',bg_plot=False):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        cols = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        with h5py.File(self.savefile,'a') as f:
            for e, key in enumerate(list(f[modname]['xs'].keys())):
                print(key)
                xs = f[f"{modname}/xs/{key}"][:]
                ys = f[f"{modname}/ys/{key}"][:]
                xe = f[f"{modname}/xe/{key}"][:]
                ye = f[f"{modname}/ye/{key}"][:]
                y = f[f"{modname}/y/{key}"][:]
                ys_comps = f[f"{modname}/yscomps/{key}"][:]
                xres = f[f"{modname}/xres/{key}"][:]
                yres = f[f"{modname}/yres/{key}"][:]
                xres_e = f[f"{modname}/xres_e/{key}"][:]
                yres_e = f[f"{modname}/yres_e/{key}"][:]

                fig = plt.figure(figsize=(9,6))
                gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
                gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
                gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
                ax  = fig.add_subplot(gs1[:,:])
                ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
                ls=15
                ps=2
                if error == True:
                    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label="data")
                else:
                    ax.step(xs,ys,color=cols[e],label="data",lw=ps)
                ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
                ax.plot(xs,y,label="All model",color="red",lw=ps)
                if self.line != 'None':
                    ax.plot(xs,ys_comps[0],'-.',label=r"bvapec",lw=ps,color='orange')
                if self.line == 'wz':
                    ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color='darkblue')
                    ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
                if self.line == 'w':
                    ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
                if self.line == 'double':
                    ax.plot(xs,ys_comps[1],'-.',label=r"bvapec 2",lw=ps,color='blue')
                    bg_count = 2
                if self.line == 'None':
                    bg_count = 0
                if self.line == 'All':
                    ax.plot(xs,ys_comps[1],':',label=r"Fe XXV He $\alpha$ z",lw=ps,color='black')
                    ax.plot(xs,ys_comps[2],':',label=r"Fe XXV He $\alpha$ w",lw=ps,color='black')
                    ax.plot(xs,ys_comps[3],':',label=r"Fe XXV He $\alpha$ y",lw=ps,color='black')
                    ax.plot(xs,ys_comps[4],':',label=r"Fe XXV He $\alpha$ x",lw=ps,color='black')
                    ax.plot(xs,ys_comps[5],':',label=r"Fe XXVI Ly $\alpha$2 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[6],':',label=r"Fe XXVI Ly $\alpha$1 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[7],':',label=r"Fe XXV He $\beta$2 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[8],':',label=r"Fe XXV Ly $\beta$1 ",lw=ps,color='black')
                    C = Cluster()
                    for lines in ['w','z','y','x','Heb2','Heb1','Lya2','Lya1']:
                        C.line_manager(state=lines)
                        if lines == 'w' or lines == 'z' or lines == 'y' or lines == 'x':
                            ax.text(C.line_energy*1e-3/(1+0.1031), 1.02, f"{lines}", fontsize=15, color='blue', fontname='Arial')
                            ax.vlines(C.line_energy*1e-3/(1+0.1031),0.98,1.0,linestyle='-',color='blue',lw=2)
                        else:
                            if lines == 'Heb1':
                                lines = r'He $\beta$1'
                            if lines == 'Lya2':
                                lines = r'Ly $\alpha$2'
                            if lines == 'Lya1':
                                lines = r'Ly $\alpha$1'
                            ax.text(C.line_energy*1e-3/(1+0.1031), 0.5, lines, fontsize=15, color='blue', fontname='Arial')
                            ax.vlines(C.line_energy*1e-3/(1+0.1031),0.46,0.48,linestyle='-',color='blue',lw=2)

                print(len(ys_comps))
                for i in range(ys_comps.shape[0]):
                    ax.plot(xs,ys_comps[i],':',label=r"Fe XXV He $\alpha$",lw=ps,color='black')
                if bg_plot == True:
                    bg = sum([ys_comps[i] for i in range(bg_count,len(ys_comps))])
                    ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='black')
                ax.set_xlim(x_rng[0],x_rng[1])
                x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
                yrng = ys[x_rng_mask]
                ax.set_ylim(3e-4,np.max(yrng)+0.3)
                #ax.set_ylim(3e-4,np.max(yrng)+10)
                ax2.set_ylim(-0.42, 0.2)
                #ax.set_yscale('log')
                #ax.set_xscale('log')
                #ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
                ax2.errorbar(xs,ys-y,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
                
                ax2.hlines(0,1.8,10,linestyle='-.',color='green')
                ax2.set_xlabel("Energy (keV)",fontsize=ls)
                ax2.set_ylabel("Data - Model",fontsize=ls)
                # ax.grid(linestyle='dashed')
                # ax2.grid(linestyle='dashed')
                spine_width = 2  # スパインの太さ
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                for spine in ax2.spines.values():
                    spine.set_linewidth(spine_width)
                ax.tick_params(axis='both',direction='in',width=1.5)
                ax.tick_params(axis='x', labelbottom=False, direction='in',width=1.5)
                ax2.tick_params(axis='both',direction='in',width=1.5)
                txpos = 0.1
                fs = 20
                cl = "red"
                # 短い線の長さ（例：上下に ±0.01）
                dy = 0.05

                # 各テキストのx座標
                x_positions = [6.017, 6.043, 6.057, 6.073]
                line_list = ["z", "y", "x", "w"]
                # テキストと縦線を描画
                # for ee,x in enumerate(x_positions):
                #     ax.text(x, txpos, line_list[ee], fontsize=fs, color=cl, fontname='Arial', va='center', ha='center')  # "?"はあとで置き換え
                #     ax.vlines(x, txpos - dy, txpos - 2*dy, color='red', linewidth=2)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

                #ax.text(6.0, 0.9, r"$\rm Fe XXV\ He \alpha$", fontsize=fs, color="black", va='center', ha='center', fontname="serif") 
                # ax.legend(fontsize=12,loc="upper right")
                #ax.set_title(f"{modname}")
                if logging==True:
                    ax.set_yscale('log')
                    # ax.set_xscale('log')
                fig.align_labels()
                fig.tight_layout()
                plt.show()
                fig.savefig(f"./figure/{base_name}_{modname}_{key}.pdf",dpi=300,transparent=True)


    def plotting_for_PASJ_fig3(self,modname="S_all_all_None_rebin_5_zgaus_z_nofix",error=True,x_rng=[2, 8],logging=True,line='None',bg_plot=False):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        cols = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        with h5py.File(self.savefile,'a') as f:
            for e, key in enumerate(list(f[modname]['xs'].keys())):
                if e == 0:
                    print(key)
                    xs = f[f"{modname}/xs/{key}"][:]
                    ys = f[f"{modname}/ys/{key}"][:]
                    xe = f[f"{modname}/xe/{key}"][:]
                    ye = f[f"{modname}/ye/{key}"][:]
                    y = f[f"{modname}/y/{key}"][:]
                    ys_comps = f[f"{modname}/yscomps/{key}"][:]
                    xres = f[f"{modname}/xres/{key}"][:]
                    yres = f[f"{modname}/yres/{key}"][:]
                    xres_e = f[f"{modname}/xres_e/{key}"][:]
                    yres_e = f[f"{modname}/yres_e/{key}"][:]

                    fig = plt.figure(figsize=(8,5.5))
                    gs  = GridSpec(nrows=1,ncols=1)
                    gs1 = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs[0,:])
                    ax  = fig.add_subplot(gs1[:,:])
                    ls=15
                    ps=1.5
                    if error == True:
                        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label="data")
                    else:
                        ax.step(xs,ys,color=cols[e],label="data",lw=ps)
                    ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
                    ax.plot(xs,y,label="All model",color="red",lw=ps)
                    if self.line != 'None':
                        ax.plot(xs,ys_comps[0],'-.',label=r"bvapec",lw=ps,color='orange')
                    if self.line == 'wz':
                        ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color='darkblue')
                        ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
                    if self.line == 'w':
                        ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
                    if self.line == 'double':
                        ax.plot(xs,ys_comps[1],'-.',label=r"bvapec 2",lw=ps,color='blue')
                        bg_count = 2
                    if self.line == 'None':
                        bg_count = 0
                    if self.line == 'All':
                        ax.plot(xs,ys_comps[1],':',label=r"Fe XXV He $\alpha$ z",lw=ps,color='black')
                        ax.plot(xs,ys_comps[2],':',label=r"Fe XXV He $\alpha$ w",lw=ps,color='black')
                        ax.plot(xs,ys_comps[3],':',label=r"Fe XXV He $\alpha$ y",lw=ps,color='black')
                        ax.plot(xs,ys_comps[4],':',label=r"Fe XXV He $\alpha$ x",lw=ps,color='black')
                        ax.plot(xs,ys_comps[5],':',label=r"Fe XXVI Ly $\alpha$2 ",lw=ps,color='black')
                        ax.plot(xs,ys_comps[6],':',label=r"Fe XXVI Ly $\alpha$1 ",lw=ps,color='black')
                        ax.plot(xs,ys_comps[7],':',label=r"Fe XXV He $\beta$2 ",lw=ps,color='black')
                        ax.plot(xs,ys_comps[8],':',label=r"Fe XXV Ly $\beta$1 ",lw=ps,color='black')
                        C = Cluster()
                        for lines in ['w','z','y','x','Heb2','Heb1','Lya2','Lya1']:
                            C.line_manager(state=lines)
                            if lines == 'w' or lines == 'z' or lines == 'y' or lines == 'x':
                                ax.text(C.line_energy*1e-3/(1+0.1031), 1.02, f"{lines}", fontsize=15, color='blue', fontname='Arial')
                                ax.vlines(C.line_energy*1e-3/(1+0.1031),0.98,1.0,linestyle='-',color='blue',lw=2)
                            else:
                                if lines == 'Heb1':
                                    lines = r'He $\beta$1'
                                if lines == 'Lya2':
                                    lines = r'Ly $\alpha$2'
                                if lines == 'Lya1':
                                    lines = r'Ly $\alpha$1'
                                ax.text(C.line_energy*1e-3/(1+0.1031), 0.5, lines, fontsize=15, color='blue', fontname='Arial')
                                ax.vlines(C.line_energy*1e-3/(1+0.1031),0.46,0.48,linestyle='-',color='blue',lw=2)

                    print(len(ys_comps))
                    if bg_plot == True:
                        bg = sum([ys_comps[i] for i in range(bg_count,len(ys_comps))])
                        ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='black')
                    ax.set_xlim(x_rng[0],x_rng[1])
                    x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
                    yrng = ys[x_rng_mask]
                    ax.set_ylim(1e-2,4)
                    ax.text(6.0, 2, "Fe XXV"+"\n"+r"He$\alpha$", fontsize=12, ha='center', va='center')
                    ax.text(6.5, 0.5, "Fe XXVI"+"\n"+r"Ly$\alpha$", fontsize=12, ha='center', va='center')
                    ax.text(7.2, 0.5, "Fe XXV"+"\n"+r"He$\beta$", fontsize=12, ha='center', va='center')
                    #ax.set_yscale('log')
                    #ax.set_xscale('log')
                    ax.set_xlabel("Energy (keV)",fontsize=ls)
                    #ax.grid(linestyle='dashed')
                    spine_width = 2  # スパインの太さ
                    for spine in ax.spines.values():
                        spine.set_linewidth(spine_width)
                    ax.tick_params(axis='both',direction='in',width=1.5)
                    # ax.legend(fontsize=12,loc="upper right")
                    #ax.set_title(f"{modname}")
                    if logging==True:
                        ax.set_yscale('log')
                        # ax.set_xscale('log')
                    fig.align_labels()
                    fig.tight_layout()
                    plt.show()
                    fig.savefig(f"./figure/{base_name}_{modname}_{key}.pdf")



    def plotting_ssm(self, modname, error=True, x_rng=[5.975, 6.125], logging=False, line='wz', rebin=False):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        cols = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        comp_col = ['orange', 'blue', 'green',  'brown', 'salmon', 'darkgreen', 'green']
        colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
        with h5py.File(self.savefile,'a') as f:
            for e, key in enumerate(list(f[modname]['xs'].keys())):
                print(key)
                if rebin == False:
                    xs = f[f"{modname}/xs/{key}"][:]
                    ys = f[f"{modname}/ys/{key}"][:]
                    xe = f[f"{modname}/xe/{key}"][:]
                    ye = f[f"{modname}/ye/{key}"][:]
                else:
                    xs = f[f"{modname}/xs_rebin{rebin}/{key}"][:]
                    ys = f[f"{modname}/ys_rebin{rebin}/{key}"][:]
                    xe = f[f"{modname}/xe_rebin{rebin}/{key}"][:]
                    ye = f[f"{modname}/ye_rebin{rebin}/{key}"][:]
                y = f[f"{modname}/y/{key}"][:]
                ys_comps = f[f"{modname}/yscomps/{key}"][:]
                xres = f[f"{modname}/xres/{key}"][:]
                yres = f[f"{modname}/yres/{key}"][:]
                xres_e = f[f"{modname}/xres_e/{key}"][:]
                yres_e = f[f"{modname}/yres_e/{key}"][:]

                fig = plt.figure(figsize=(9,6))
                gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
                gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
                gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
                ax  = fig.add_subplot(gs1[:,:])
                ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
                ls=15
                ps=2
                if error == True:
                    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e])
                else:
                    ax.step(xs,ys,color=cols[e],label="data",lw=ps)
                ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
                ax.plot(xs,y,label="All model",color=colors[0],lw=ps)
                print('comps')
                print(len(ys_comps))
                print(ys_comps.shape)
                if line == 'wz':
                    center_mod = np.sum(ys_comps[0:3],axis=0)
                    outer_mod = np.sum(ys_comps[3:6],axis=0)
                    ex12pix_mod = np.sum(ys_comps[6:9],axis=0)
                else:
                    center_mod = np.sum(ys_comps[0:2],axis=0)
                    outer_mod = np.sum(ys_comps[2:4],axis=0)
                    ex12pix_mod = np.sum(ys_comps[4:6],axis=0)
                ax.plot(xs,center_mod,'-',label=f"Center",lw=ps,color=colors[1])
                ax.plot(xs,outer_mod,'--',label=f"Outer",lw=ps,color=colors[2])
                ax.plot(xs,ex12pix_mod,'-.',label=f"ex12pix",lw=ps,color=colors[3])
                for i in range(len(ys_comps)):
                    print(i)
                # ax.plot(xs,ys_comps[0],'--',lw=ps,label=f"center temp1")
                # ax.plot(xs,ys_comps[1],'--',lw=ps,label=f"center temp2")
                # ax.plot(xs,ys_comps[2],'--',lw=ps,label=f"outer temp1")
                # ax.plot(xs,ys_comps[3],'--',lw=ps,label=f"outer temp2")
                # ax.plot(xs,ys_comps[4],'--',lw=ps,label=f"outer temp2")
                # ax.plot(xs,ys_comps[5],'--',lw=ps,label=f"outer temp2")
                bg = np.sum(ys_comps[6:],axis=0)
                ax.plot(xs,bg,'-',lw=ps,color="gray",label=f"NXB")
                
                ax.set_xlim(x_rng[0],x_rng[1])
                x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
                yrng = ys[x_rng_mask]
                ax.set_ylim(3e-4,np.max(yrng)+0.1)
                ax2.set_ylim(-3, 3)
                ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
                ax2.hlines(0,1.8,10,linestyle='-.',color='green')
                ax2.set_xlabel("Energy (keV)",fontsize=ls)
                ax2.set_ylabel("Residual",fontsize=ls)
                # ax.grid(linestyle='dashed')
                # ax2.grid(linestyle='dashed')
                spine_width = 2  # スパインの太さ
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                for spine in ax2.spines.values():
                    spine.set_linewidth(spine_width)
                ax.tick_params(axis='both',direction='in',width=1.5)
                ax.tick_params(axis='x', labelbottom=False, direction='in',width=1.5)
                ax2.tick_params(axis='both',direction='in',width=1.5)
                #ax.legend(fontsize=13,loc="upper right")
                #ax.set_title(f"{modname}")
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
                if logging==True:
                    ax.set_yscale('log')
                    ax.set_ylim(1e-4,np.max(yrng)+1)   
                    # ax.set_xscale('log')
                fig.align_labels()
                fig.tight_layout()
                #plt.show()
                fig.savefig(f"./figure/{base_name}_{modname}_{key}_{x_rng[0]}_{x_rng[1]}.pdf",transparent=True)        

    def plotting_simult(self,modname,error=True,x_rng=[5.9, 6.2],logging=False,line='None',bg_plot=False,identifiers='None'):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        cols = ['black', 'red', 'blue', 'green', 'orange']
        with h5py.File(self.savefile,'a') as f:
            for e, key in enumerate(list(f[modname]['xs'].keys())):
                print(key)
                xs = f[f"{modname}/xs/{key}"][:]
                ys = f[f"{modname}/ys/{key}"][:]
                xe = f[f"{modname}/xe/{key}"][:]
                ye = f[f"{modname}/ye/{key}"][:]
                y = f[f"{modname}/y/{key}"][:]
                ys_comps = f[f"{modname}/yscomps/{key}"][:]
                xres = f[f"{modname}/xres/{key}"][:]
                yres = f[f"{modname}/yres/{key}"][:]
                xres_e = f[f"{modname}/xres_e/{key}"][:]
                yres_e = f[f"{modname}/yres_e/{key}"][:]

                if e == 0:
                    fig = plt.figure(figsize=(9,6))
                    gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
                    gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
                    gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
                    ax  = fig.add_subplot(gs1[:,:])
                    ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
                ls=15
                ps=1.5
                if error == True:
                    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label=identifiers[e])
                else:
                    ax.step(xs,ys,color=cols[e],label="data",lw=ps)
                ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
                ax.plot(xs,y,color=cols[e],lw=ps)
                if self.line == 'wz':
                    ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color=cols[e])
                    #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color=cols[e])
                if self.line == 'w':
                    ax.plot(xs,ys_comps[1],'-.',lw=ps,color=cols[e])
                if self.line == 'double':
                    ax.plot(xs,ys_comps[1],'-.',label=r"bvapec 2",lw=ps,color='blue')
                    bg_count = 2
                if self.line == 'None':
                    bg_count = 0
                if self.line == 'All':
                    ax.plot(xs,ys_comps[1],':',label=r"Fe XXV He $\alpha$ z",lw=ps,color='black')
                    ax.plot(xs,ys_comps[2],':',label=r"Fe XXV He $\alpha$ w",lw=ps,color='black')
                    ax.plot(xs,ys_comps[3],':',label=r"Fe XXV He $\alpha$ y",lw=ps,color='black')
                    ax.plot(xs,ys_comps[4],':',label=r"Fe XXV He $\alpha$ x",lw=ps,color='black')
                    ax.plot(xs,ys_comps[5],':',label=r"Fe XXVI Ly $\alpha$2 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[6],':',label=r"Fe XXVI Ly $\alpha$1 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[7],':',label=r"Fe XXV He $\beta$2 ",lw=ps,color='black')
                    ax.plot(xs,ys_comps[8],':',label=r"Fe XXV Ly $\beta$1 ",lw=ps,color='black')
                    C = Cluster()
                    for lines in ['w','z','y','x','Heb2','Heb1','Lya2','Lya1']:
                        C.line_manager(state=lines)
                        if lines == 'w' or lines == 'z' or lines == 'y' or lines == 'x':
                            ax.text(C.line_energy*1e-3/(1+0.1031), 1.02, f"{lines}", fontsize=15, color='blue', fontname='Arial')
                            ax.vlines(C.line_energy*1e-3/(1+0.1031),0.98,1.0,linestyle='-',color='blue',lw=2)
                        else:
                            if lines == 'Heb1':
                                lines = r'He $\beta$1'
                            if lines == 'Lya2':
                                lines = r'Ly $\alpha$2'
                            if lines == 'Lya1':
                                lines = r'Ly $\alpha$1'
                            ax.text(C.line_energy*1e-3/(1+0.1031), 0.5, lines, fontsize=15, color='blue', fontname='Arial')
                            ax.vlines(C.line_energy*1e-3/(1+0.1031),0.46,0.48,linestyle='-',color='blue',lw=2)

                print(len(ys_comps))
                if bg_plot == True:
                    bg = sum([ys_comps[i] for i in range(bg_count,len(ys_comps))])
                    ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='black')
                ax.set_xlim(x_rng[0],x_rng[1])
                x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
                yrng = ys[x_rng_mask]
                ax.set_ylim(3e-4,np.max(yrng)+0.4)
                ax2.set_ylim(-3, 3)
                ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label="data")
                ax2.hlines(0,1.8,10,linestyle='-.',color='green')
                ax2.set_xlabel("Energy[keV]",fontsize=ls)
                ax2.set_ylabel("Residual",fontsize=ls)
                ax.grid(linestyle='dashed')
                ax2.grid(linestyle='dashed')
                spine_width = 2  # スパインの太さ
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                for spine in ax2.spines.values():
                    spine.set_linewidth(spine_width)
                ax.tick_params(axis='both',direction='in',width=1.5)
                ax2.tick_params(axis='both',direction='in',width=1.5)
                ax.legend(fontsize=12,loc="upper right")
                #ax.set_title(f"{modname}")
                if logging==True:
                    ax.set_yscale('log')
                    # ax.set_xscale('log')
                fig.align_labels()
                fig.tight_layout()
            #plt.show()
            fig.savefig(f"./figure/{base_name}_{modname}_{key}.pdf",dpi=300,transparent=True)

    def plotting_single(self,modname,error=True,x_rng=[5.9,6.2],logging=False,line='None'):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(15,6))
        ax  = fig.add_subplot(111)
        ls=15
        ps=2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.step(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        #ax.plot(xs,ys_comps[1],'-.',label="bapec(Hot)",lw=ps,color='orange')
        ax.plot(xs,ys_comps[0],'-.',label=r"bapec",lw=ps,color='orange')
        #ax.plot(xs,ys_comps[0],'-.',label=r"bapec(Hot)",lw=ps,color='orange')
        if self.line == 'wz':
            ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ z",lw=ps,color='darkblue')
            ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ y",lw=ps,color='mediumblue')
        #ax.plot(xs,ys_comps[3],'-.',label=r"Fe XXV He $\alpha$ x",lw=ps,color='blue')
        if self.line == 'w':
            ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        #ax.plot(xs,ys_comps[2],label=r"$\rm Fe\ He\alpha \ z$",lw=1,color='green')
        #ax.plot(xs,ys_comps[5],'-.',label=r"$\rm Fe\ H \ Ly\alpha 2$",lw=ps,color='brown')
        #ax.plot(xs,ys_comps[6],'-.',label=r"$\rm Fe\ H \ Ly\alpha 1$",lw=ps,color='salmon')
        #ax.plot(xs,ys_comps[7],'-.',label=r"Fe XXV He $\beta$ 2",lw=ps,color='darkgreen')
        #ax.plot(xs,ys_comps[8],'-.',label=r"Fe XXV He $\beta$ 1",lw=ps,color='green')
        z = 0.1028
        c = Cluster()
        for state in ['z', 'y', 'x', 'w']:
            c.line_manager(state=state)
            ax.text(c.line_energy*1e-3/(1+z), 1.05, state)
            ax.vlines(c.line_energy*1e-3/(1+z), 1.01, 1.05, color='black')
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-2,1.2)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_xlabel("Energy[keV]",fontsize=ls)
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        #ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_bgd(self,modname,error=True,x_rng=[1.8,12.0],logging=True):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.step(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="black",lw=ps)
        print(ys_comps)
        #ax.plot(xs,ys_comps[0],'-.',label=r"bvapec",lw=ps,color='orange')

        bg = sum([ys_comps[i] for i in range(0,len(ys_comps))])

        ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='black')
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-5,np.max(yrng)+1)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Residual",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        #ax.legend(fontsize=12,loc="center left", bbox_to_anchor=(1, 0.5))
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        nxb_dir = '/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/OBF_ND_Be/nxb'
        self.load_spectrum(spec=f'{nxb_dir}/1000_center_rslnxb.pi',rmf='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf',arf=None)
        
        self.set_xydata(plotgroup=2)
        ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=ps,elinewidth=ps,color="red",label="nxb")
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_55Fe_line(self,modname,error=True,x_rng=[5.41, 7.0],logging=False,line='MnKa',bgplot=True):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            print(list(f[modname]['xs'].keys()))
            for e, key in enumerate(list(f[modname]['xs'].keys())):
                print(key)
                xs = f[f"{modname}/xs/{key}"][:]
                ys = f[f"{modname}/ys/{key}"][:]
                xe = f[f"{modname}/xe/{key}"][:]
                ye = f[f"{modname}/ye/{key}"][:]
                y = f[f"{modname}/y/{key}"][:]
                ys_comps = f[f"{modname}/yscomps/{key}"][:]
                xres = f[f"{modname}/xres/{key}"][:]
                yres = f[f"{modname}/yres/{key}"][:]
                xres_e = f[f"{modname}/xres_e/{key}"][:]
                yres_e = f[f"{modname}/yres_e/{key}"][:]

            fig = plt.figure(figsize=(9,6))
            gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
            gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
            gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
            ax  = fig.add_subplot(gs1[:,:])
            ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
            ls=15
            ps=1.5
            if error == True:
                ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
            else:
                ax.plot(xs,ys,color="black",label="data",lw=ps)
            ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
            ax.plot(xs,y,label="All model",color="red",lw=ps)
            bgs = 5
            if line == 'MnKa':
                ax.plot(xs,ys_comps[0],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[1],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[2],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[3],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[4],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[5],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[6],'-.',lw=ps,color='blue')
                ax.plot(xs,ys_comps[7],'-.',lw=ps,color='blue')
                bgs = 8
        
            elif line == 'MnKab':
                ax.plot(xs,sum(ys_comps[0:8]),lw=ps,color='blue', label='MnKa')
                for i in range(0,8):
                    ax.plot(xs,ys_comps[i],':',lw=ps,color='blue')
                ax.plot(xs,sum(ys_comps[8:15]),lw=ps,color='green', label='MnKb')
                for i in range(8,15):
                    ax.plot(xs,ys_comps[i],':',lw=ps,color='green')
                bgs = 14
            if bgplot == True:
                bg = sum([ys_comps[i] for i in range(bgs,len(ys_comps))])

                ax.plot(xs,bg,'-.',label=r"NXB",lw=ps,color='orange')

            ax.set_xlim(x_rng[0],x_rng[1])
            x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
            yrng = ys[x_rng_mask]
            if logging==True:
                ax.set_ylim(1e-5,np.max(yrng)+100.0)
            else:
                ax.set_ylim(1e-3,np.max(yrng)+10.0)
                #ax.set_ylim(1e-3,np.max(yrng)+0.3)
            ax2.set_ylim(-5,5)
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
            ax2.hlines(0,1.8,10,linestyle='-.',color='green')
            ax2.set_xlabel("Energy[keV]",fontsize=ls)
            ax2.set_ylabel("Residual",fontsize=ls)
            ax.grid(linestyle='dashed')
            ax2.grid(linestyle='dashed')
            spine_width = 2  # スパインの太さ
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            for spine in ax2.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis='both',direction='in',width=1.5)
            ax2.tick_params(axis='both',direction='in',width=1.5)
            #ax.legend(fontsize=12,loc='upper right')
            #ax.set_title(f"{modname}")
            if logging==True:
                ax.set_yscale('log')
            fig.align_labels()
            fig.tight_layout()
            #plt.show()
            fig.savefig(f"./figure/{base_name}_{modname}.png",dpi=300,transparent=True)

    def plotting_multi_cor(self,modname,error=True,x_rng=[5.9, 6.2],logging=False,line='None',bg_plot=False):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        
        with h5py.File(self.savefile,'a') as f:
            for key in list(f[modname]['xs'].keys())[0]:
                print(key)
                xs = f[f"{modname}/xs/{key}"][:]
                ys = f[f"{modname}/ys/{key}"][:]
                xe = f[f"{modname}/xe/{key}"][:]
                ye = f[f"{modname}/ye/{key}"][:]
                y = f[f"{modname}/y/{key}"][:]
                ys_comps = f[f"{modname}/yscomps/{key}"][:]
                xres = f[f"{modname}/xres/{key}"][:]
                yres = f[f"{modname}/yres/{key}"][:]
                xres_e = f[f"{modname}/xres_e/{key}"][:]
                yres_e = f[f"{modname}/yres_e/{key}"][:]

                fig = plt.figure(figsize=(9,6))
                gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
                gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
                gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
                ax  = fig.add_subplot(gs1[:,:])
                ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
                ls=15
                ps=2
                if error == True:
                    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
                else:
                    ax.step(xs,ys,color="black",label="data",lw=ps)
                ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
                ax.plot(xs,y,label="All model",color="red",lw=ps)
                ax.plot(xs,ys_comps[0],'-.',label=r"bapec",lw=ps,color='orange')
                ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='darkblue')
                #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
                mask = (xs > 6.055) & (xs < 6.10)
                print(self.y_apec)
                self.y_apec = np.array(self.y_apec)
                ax.plot(xs[mask],self.y_apec[:][mask],'--',label=r"apec expected",lw=ps,color='gray')
                

                ax.set_xlim(x_rng[0],x_rng[1])
                x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
                yrng = ys[x_rng_mask]
                ax.set_ylim(1e-2,np.max(yrng)+1)
                #ax.set_yscale('log')
                #ax.set_xscale('log')
                ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
                ax2.hlines(0,1.8,10,linestyle='-.',color='green')
                ax2.set_xlabel("Energy[keV]",fontsize=ls)
                ax2.set_ylabel("Residual",fontsize=ls)
                ax.grid(linestyle='dashed')
                ax2.grid(linestyle='dashed')
                spine_width = 2  # スパインの太さ
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                for spine in ax2.spines.values():
                    spine.set_linewidth(spine_width)
                ax.tick_params(axis='both',direction='in',width=1.5)
                ax2.tick_params(axis='both',direction='in',width=1.5)
                ax.legend(fontsize=12,loc="upper left")
                #ax.set_title(f"{modname}")
                if logging==True:
                    ax.set_yscale('log')
                    # ax.set_xscale('log')
                fig.align_labels()
                fig.tight_layout()
                plt.show()
                fig.savefig(f"./figure/{base_name}_{modname}_com.pdf",dpi=300,transparent=True)

    def plotting_for_w(self,modname,error=True,x_rng=[5.9,6.2],logging=False):
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        with h5py.File(self.savefile,'a') as f:
            xs = f[f"{modname}/xs"][:]
            ys = f[f"{modname}/ys"][:]
            xe = f[f"{modname}/xe"][:]
            ye = f[f"{modname}/ye"][:]
            y = f[f"{modname}/y"][:]
            ys_comps = f[f"{modname}/yscomps"][:]
            xres = f[f"{modname}/xres"][:]
            yres = f[f"{modname}/yres"][:]
            xres_e = f[f"{modname}/xres_e"][:]
            yres_e = f[f"{modname}/yres_e"][:]

        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls=15
        ps=2
        if error == True:
            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color="black",label="data")
        else:
            ax.step(xs,ys,color="black",label="data",lw=ps)
        ax.set_ylabel("Count s$^{-1}$ keV$^{-1}$",fontsize=ls)
        ax.plot(xs,y,label="All model",color="red",lw=ps)
        ax.plot(xs,ys_comps[0],'-.',label=r"bvapec",lw=ps,color='orange')
        ax.plot(xs,ys_comps[1],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='darkblue')
        #ax.plot(xs,ys_comps[2],'-.',label=r"Fe XXV He $\alpha$ w",lw=ps,color='green')
        mask = (xs > 6.055) & (xs < 6.10)
        print(self.y_apec)
        self.y_apec = np.array(self.y_apec)
        ax.plot(xs[mask],self.y_apec[:][mask],'--',label=r"bvapec expected",lw=ps,color='gray')
        

        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
        yrng = ys[x_rng_mask]
        ax.set_ylim(1e-5,1.0)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax2.plot(xs,self.y_apec/y,color="black")
        #ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Ratio",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        ax.legend(fontsize=12,loc="upper left")
        ax2.set_ylim(0.5,1.5)
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}_com.pdf",dpi=300,transparent=True)

    def plotting_raw_data(self,spec, rmf, bgd=None, error=True,x_rng=[5,7],logging=False):
        fig, ax = plt.subplots(1, figsize=(8,6))
        ls  = 15
        ps  = 2
        self.load_spectrum(spec=spec,rmf=rmf,arf=None,bgd=bgd)

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.xe = np.array(self.xe)
        self.ye = np.array(self.ye)

        if error == True:
            ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=ps,elinewidth=ps,color='black')
        else:
            ax.step(self.xs,self.ys,color='black')
        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < self.xs) & (self.xs < x_rng[1])
        yrng = self.ys[x_rng_mask]
        ax.set_ylim(1e-5,np.max(yrng)+100)
        ax.set_xlabel("Energy[keV]",fontsize=ls)
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax.legend(fontsize=12,loc="best")
        if logging==True:
            ax.set_yscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/raw_data.png",dpi=300,transparent=True)

    def plotting_raw_data_cor(self,spec, rmf, bgd, spec2, rmf2, bgd2, dir2=None , error=True,x_rng=[5,7],logging=False):
        fig, ax = plt.subplots(1, figsize=(8,6))
        ls  = 15
        ps  = 2
        cdir = os.getcwd()
        self.load_spectrum(spec=spec,rmf=rmf,arf=None,bgd=bgd)

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.xe = np.array(self.xe)
        self.ye = np.array(self.ye)

        if error == True:
            ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=ps,elinewidth=ps,color='black')
        else:
            ax.step(self.xs,self.ys,color='black')

        if dir2 != None:
            os.chdir(dir2)
        self.load_spectrum(spec=spec2,rmf=rmf2,arf=None,bgd=bgd2)

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.xe = np.array(self.xe)
        self.ye = np.array(self.ye)

        if error == True:
            ax.errorbar(self.xs,self.ys,yerr=self.ye,xerr=self.xe,fmt="o",markersize=ps,elinewidth=ps,color='red')
        else:
            ax.step(self.xs,self.ys,color='red')

        ax.set_xlim(x_rng[0],x_rng[1])
        x_rng_mask = (x_rng[0] < self.xs) & (self.xs < x_rng[1])
        yrng = self.ys[x_rng_mask]
        ax.set_ylim(1e-5,np.max(yrng)+100)
        ax.set_xlabel("Energy[keV]",fontsize=ls)
        ax.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        #ax.legend(fontsize=12,loc="best")
        if logging==True:
            ax.set_yscale('log')
        fig.align_labels()
        fig.tight_layout()
        os.chdir(cdir)
        plt.show()
        fig.savefig(f"./figure/raw_data.png",dpi=300,transparent=True)

    def plotting_raw_cor(self,modname,error=True,x_rng=[5.9,6.2],logging=False,line='None'):
        self.line = line
        base_name = os.path.splitext(os.path.basename(self.savefile))[0]
        files = [self.savefile, '55Fe_cal_without_55Fe_filter.hdf5']
        modname = [modname, '4000_center_w']
        labs = ['55Fe cal', 'gh linear cor']
        cols = ['black','red']
        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls  = 15
        ps  = 2
        for e, file in enumerate(files):
            with h5py.File(file,'a') as f:
                xs = f[f"{modname[e]}/xs"][:]
                ys = f[f"{modname[e]}/ys"][:]
                xe = f[f"{modname[e]}/xe"][:]
                ye = f[f"{modname[e]}/ye"][:]
                y = f[f"{modname[e]}/y"][:]
                ys_comps = f[f"{modname[e]}/yscomps"][:]
                xres = f[f"{modname[e]}/xres"][:]
                yres = f[f"{modname[e]}/yres"][:]
                xres_e = f[f"{modname[e]}/xres_e"][:]
                yres_e = f[f"{modname[e]}/yres_e"][:]


            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label=labs[e])
            ax.plot(xs,y,color=cols[e],lw=ps)
            ax.set_xlim(x_rng[0],x_rng[1])
            x_rng_mask = (x_rng[0] < xs) & (xs < x_rng[1])
            yrng = ys[x_rng_mask]
            ax.set_ylim(1e-2,np.max(yrng)+0.1)
            ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label="data")
            ax2.hlines(0,1.8,10,linestyle='-.',color='green')
            ax2.set_xlabel("Energy[keV]",fontsize=ls)
            ax2.set_ylabel("Residual",fontsize=ls)
            ax.grid(linestyle='dashed')
            ax2.grid(linestyle='dashed')
            spine_width = 2  # スパインの太さ
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            for spine in ax2.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis='both',direction='in',width=1.5)
            ax2.tick_params(axis='both',direction='in',width=1.5)
            ax.legend(fontsize=12,loc="best")
            #ax.set_title(f"{modname}")
            if logging==True:
                ax.set_yscale('log')
                # ax.set_xscale('log')
            fig.align_labels()
            fig.tight_layout()
            #plt.show()
            fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def plotting_raw_cor_ratio(self,modname,x_rng=[2,10],logging=True,line='None'):
        self.line = line
        files = ['./5000_center.pi', './55Fe_bgd/5000_center.pi']
        labs = ['bright', '55Fe template']
        cols = ['black','red']
        fig = plt.figure(figsize=(9,6))
        gs  = GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
        gs1 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
        gs2 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
        ax  = fig.add_subplot(gs1[:,:])
        ax2 = fig.add_subplot(gs2[:,:],sharex=ax)
        ls  = 15
        ps  = 2
        for e, file in enumerate(files):
            if e == 0:
                self.load_spectrum(spec=file,rmf='./5000_center_X_without_Lp_comb.rmf',arf=None)
                self.set_xydata(plotgroup=1)
            else:
                self.load_spectrum(spec=file,rmf='./55Fe_bgd/5000_center_X_without_Lp_comb.rmf',arf=None)
                self.set_xydata(plotgroup=1)

            if e == 1:
                self.ratio_x = np.array(self.xs)
                self.ratio_y = np.array(self.ys)/ys

            xs = np.array(self.xs)
            ys = np.array(self.ys)
            xe = np.array(self.xe)
            ye = np.array(self.ye)

            ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=ps,elinewidth=ps,color=cols[e],label=labs[e])
            ax.set_xlim(x_rng[0],x_rng[1])
            if e == 1:
                ax2.scatter(self.ratio_x,self.ratio_y,color=cols[e],label="data")
            #ax2.hlines(0,1.8,10,linestyle='-.',color='green')
        ax2.set_xlabel("Energy[keV]",fontsize=ls)
        ax2.set_ylabel("Ratio (data/template)",fontsize=ls)
        ax.grid(linestyle='dashed')
        ax2.grid(linestyle='dashed')
        spine_width = 2  # スパインの太さ
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        for spine in ax2.spines.values():
            spine.set_linewidth(spine_width)
        ax.tick_params(axis='both',direction='in',width=1.5)
        ax2.tick_params(axis='both',direction='in',width=1.5)
        ax.legend(fontsize=12,loc="best")
        #ax.set_title(f"{modname}")
        if logging==True:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        fig.align_labels()
        fig.tight_layout()
        plt.show()
        fig.savefig(f"./figure/{base_name}_{modname}.pdf",dpi=300,transparent=True)

    def formatter(self, values, significant_digits):
        def format_to_significant_figures(value, significant_digits):
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        def format_to_significant_digit(value, significant_digits):
            if value == 0:
                return 0
            return significant_digits - int(np.floor(np.log10(abs(value)))) - 1

        decimals1 = format_to_significant_digit(values[1], significant_digits)
        decimals2 = format_to_significant_digit(values[2], significant_digits)
        min_decimals = min(decimals1, decimals2)
        # ここでは、有効数字に基づいて数値をフォーマットした後の値をそのまま使用します。
        # 小数点以下の桁数を調整する必要がある場合は、別のアプローチを検討する必要があります。
        formatted_num1 = round(values[0], min_decimals)
        formatted_num2 = round(values[1], min_decimals)
        formatted_num3 = round(values[2], min_decimals)
        return np.array([formatted_num1, formatted_num2, formatted_num3])

    def steppar(self, xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep):
        '''
        run steppar command by pyxspec

        '''
        input_par = np.array([xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep])

        Plot.device = '/xs'
        command = f'{xnum} {xmin} {xmax} {xstep} {ynum} {ymin} {ymax} {ystep}'
        print(command)
        Fit.steppar(command)
        Plot('contour')
        delta = Fit.stepparResults('delstat')
        statistic = Fit.stepparResults('statistic')
        return input_par, delta, statistic

    def steppar_data_reshape(self, input_par, delta, statistic):
        self.delta_1d = np.array(delta)
        print(self.delta_1d.shape)
        self.xnum, self.xmin, self.xmax, self.xstep = input_par[0], float(input_par[1]), float(input_par[2]), int(input_par[3])
        self.ynum, self.ymin, self.ymax, self.ystep = input_par[4], float(input_par[5]), float(input_par[6]), int(input_par[7])
        self.delta_2d = self.delta_1d.reshape((self.ystep + 1, self.xstep + 1))
        self.x_values = np.linspace(self.xmin, self.xmax, self.xstep + 1)
        self.y_values = np.linspace(self.ymin, self.ymax, self.ystep + 1)
        self.X, self.Y = np.meshgrid(self.x_values, self.y_values)
        self.minx_index, self.miny_index = np.unravel_index(np.argmin(self.delta_2d), self.delta_2d.shape)
        self.minx = self.X[self.minx_index]
        self.miny = self.Y[self.miny_index]
        self.all_model = AllModels(1)
        # self.best_x = self.all_model(self.xnum).values[0]
        # self.best_y = self.all_model(self.ynum).values[0]
        self.best_x = 7.735
        self.best_y = 7.09257

    def plot_contour(self, xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep):
        input_par, delta, statistic = self.steppar(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.steppar_data_reshape(input_par, delta, statistic)
        # CRange68 = 2.295815160785974337606
        # CRange90 = 4.605170185988091368036
        # CRange99 = 9.210340371976182736072
        CRange68 = 1.0	
        CRange90 = 2.295815160785974337606
        CRange99 = 4.605170185988091368036	
        contour_levels = [CRange68, CRange90, CRange99]
        col = ['red', 'green', 'blue']
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        #CS = ax.contourf(self.X, self.Y, self.delta_2d, 20, cmap='Greys_r')
        #fig.colorbar(CS)
        contours = self.ax.contour(self.X, self.Y, self.delta_2d, levels=contour_levels, colors=col)
        self.ax.set_xlabel(f'kT (keV)')
        self.ax.set_ylabel(f'w/z flux ratio')
        self.ax.scatter(self.best_x,self.best_y, marker='+', color='red', s=30)
        #ax.set_title('Z, W Sigma fix')

        # sess = pyatomdb.spectrum.CIESession()
        # kTlist = np.linspace(4.0,8.0,1000)
        # for up in [2,3,4,5,6,7]:
        #     ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
        #     if up == 2:
        #         z = ldata['epsilon']
        #         print(ldata['energy'])
        #     if up == 7:
        #         w = ldata['epsilon']
        #         print(ldata['energy'])
        # # ax.plot(kTlist, w/z,color='darkblue', label='apec')

        # ax.plot(kTlist, w/z,color='darkblue', label='apec')
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        # ax.set_xlim(min(self.x_values), max(self.x_values))
        # ax.set_ylim(min(self.y_values), max(self.y_values))
        self.fig.tight_layout()
        self.fig.savefig(f'steppar_{xnum}_{ynum}.png', dpi=300, transparent=True)
        #plt.show()

    def plot_contour_load(self, modname, parameter_set):
        self.load_steppar_result(modname, parameter_set)
        CRange1  = 1.0
        CRange68 = 2.295815160785974337606
        CRange90 = 4.605170185988091368036
        CRange99 = 9.210340371976182736072		
        contour_levels = [CRange68, CRange90, CRange99]
        contour_levels = [CRange1 ,CRange68, CRange90]
        col = ['red', 'green', 'blue']
        self.fig = plt.figure(figsize=(10, 8))
        self.ax  = self.fig.add_subplot(111)
        contours = self.ax.contour(self.X, self.Y, self.delta_2d, levels=contour_levels, colors=col)
        #self.ax.scatter(self.best_x,self.best_y, marker='+', color='red', s=30)
        # self.ax.set_xlabel(f'kT (keV)')
        # self.ax.set_ylabel(f'w/z flux ratio')
        self.fig.tight_layout()
        #plt.show()

    def plot_contour_fake_multi(self):
        import matplotlib.lines as mlines
        from Resonance_Scattering_Simulation import PlotManager
        P = PlotManager((1,1),(8,6),label_size=15)

        modnames = ["fake_100ksec", "S_all_center_wz_rebin_1_dzgaus"]
        parameter_set = "2_26"

        contour_levels = [2.30, 4.61, 9.21]  # 68%, 90%, 99% confidence levels
        colors = ['red', 'black']
        savefile_l = self.savefile
        for modname, color in zip(modnames, colors):
            if modname == "S_all_center_wz_rebin_1_dzgaus":
                self.savefile = '240225_thesis.hdf5'
            # データ読み込み
            self.load_steppar_result(modname, parameter_set)

            # 等高線の描画
            contours = P.axes[0].contour(self.X, self.Y, self.delta_2d, levels=contour_levels, colors=color, linestyles=['dashed', 'solid', 'dashdot'])
            labels = ['Comissioning 44 ksec', '1σ', '2σ', '3σ', 'Simulated 100 ksec', '1σ', '2σ', '3σ']
            legend_lines = [
                mlines.Line2D([], [], color=c, linestyle=ls, label=lab)
                for c, ls, lab in zip(['white','black', 'black', 'black', 'white', 'red', 'red', 'red'], ['solid','dashed', 'solid', 'dashdot','solid', 'dashed', 'solid', 'dashdot'], labels)
            ]
            # 凡例を作成（contours.collectionsとlabelsを対応付ける）
            # for i, (line, label) in enumerate(zip(contours.collections, labels)):
            #     line.set_label(label)
            # best fit マーカー
            if modname == "S_all_center_wz_rebin_1_dzgaus":
                P.axes[0].scatter(self.best_x, self.best_y, marker='+', color='black', s=30)
            else:
                P.axes[0].scatter(self.best_x, self.best_y, marker='+', color='red', s=30)

            # モデル名から表示用文字列を取得（例: "100ksec"）
            label_text = modname.replace("fake_", "")
            if modname == "S_all_center_wz_rebin_1_dzgaus":
                label_text = "PV 44.1 ksec"

            # 各 level の contour の最初の点にラベルを付ける
            # for i, collection in enumerate(contours.collections):
            #     paths = collection.get_paths()
            #     if paths:
            #         # 最初の path の座標を取得
            #         v = paths[0].vertices
            #         x, y = v[len(v) // 2]  # 中央付近の点を選んでそこにラベルを置く
            #         P.axes[0].text(x, y, label_text, color=color, fontsize=10, weight='bold')

        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,8.0,1000)
        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
            if up == 2:
                z = ldata['epsilon']
                print(ldata['energy'])
            if up == 7:
                w = ldata['epsilon']
                print(ldata['energy'])
        P.axes[0].plot(kTlist, w/z, '-',color='orange', label='atomdb v3.0.9', lw=3)
        P.axes[0].set_xlim(np.min(self.X), np.max(self.X))
        P.axes[0].set_ylim(np.min(self.Y), np.max(self.Y))
        P.axes[0].set_xlabel('kT (keV)')
        P.axes[0].set_ylabel('w/z flux ratio')
        P.axes[0].text(5.6, 3.25, 'Atomdb v3.0.9', fontsize=12, color='orange', weight='bold')
        P.axes[0].legend(loc='lower right', handles=legend_lines)
        P.fig.tight_layout()
        P.fig.savefig(f'steppar_fake_multi.png', dpi=300)
        self.savefile = savefile_l
        plt.show()

    def oplot_contour(self,new=True,line_style='-',label='None'):
        if new == True:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111)
        CRange68 = 2.295815160785974337606
        CRange90 = 4.605170185988091368036
        CRange99 = 9.210340371976182736072		
        contour_levels = [CRange68, CRange90, CRange99]
        col = ['red', 'green', 'blue']
        lines = ['-', '-', '-']
        if line_style == '-.':
            col = ['blue', 'blue', 'blue']
        contours = self.ax.contour(self.X, self.Y, self.delta_2d, levels=contour_levels, colors=col, linestyles=lines)
        # self.ax.legend([contours.collections[0], contours.collections[1], contours.collections[2]],
        #        ['68% Contour', '90% Contour', '99% Contour'],
        #        loc='upper right', fontsize=12)
        self.ax.set_xlabel(f'kT (keV)')
        self.ax.set_ylabel(f'w/z integrated flux ratio')
        self.ax.scatter(self.best_x,self.best_y, marker='+', color=col[0], s=50, label=label)
        self.fig.tight_layout()
        self.fig.savefig(f'steppar.png', dpi=300)
        #plt.show()

    def save_steppar_result(self, modname, parameter_set):
        with h5py.File(self.savefile, 'a') as f:
            if modname not in f.keys():
                f.create_group(modname)
            if f'steppar/{parameter_set}' in f[f'{modname}'].keys():
                del f[f'{modname}/steppar/{parameter_set}']
            f[modname].create_group(f'steppar/{parameter_set}')
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('x_values', data=self.x_values)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('y_values', data=self.y_values)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('delta_2d', data=self.delta_2d)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('minx', data=self.minx)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('miny', data=self.miny)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('minx_index', data=self.minx_index)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('miny_index', data=self.miny_index)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('delta_1d', data=self.delta_1d)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('best_x', data=self.best_x)
            f[f'{modname}/steppar/{parameter_set}'].create_dataset('best_y', data=self.best_y)

    def load_steppar_result(self, modname, parameter_set):
        with h5py.File(self.savefile, 'r') as f:
            self.x_values = f[f'{modname}/steppar/{parameter_set}/x_values'][:]
            self.y_values = f[f'{modname}/steppar/{parameter_set}/y_values'][:]
            self.delta_2d = f[f'{modname}/steppar/{parameter_set}/delta_2d'][:]
            self.minx = f[f'{modname}/steppar/{parameter_set}/minx'][()]
            self.miny = f[f'{modname}/steppar/{parameter_set}/miny'][()]
            self.minx_index = f[f'{modname}/steppar/{parameter_set}/minx_index'][()]
            self.miny_index = f[f'{modname}/steppar/{parameter_set}/miny_index'][()]
            self.delta_1d = f[f'{modname}/steppar/{parameter_set}/delta_1d'][:]
            self.best_x = f[f'{modname}/steppar/{parameter_set}/best_x'][()]
            self.best_y = f[f'{modname}/steppar/{parameter_set}/best_y'][()]
        self.X, self.Y = np.meshgrid(self.x_values, self.y_values)

    def run_steppar(self, identifier='S_all', region='center', line='wz', rebin=1, line_model='dzgaus', xnum=2, xmin=5.0, xmax=6.0, xstep=20, ynum=26, ymin=2.0, ymax=3.0, ystep=20):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}_{line_model}'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}_{line_model}.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        Xset.restore(filename)
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def run_steppar_ssm(self, identifier='ssm', line='wz', rebin=1, line_model='dzgaus', xnum=2, xmin=4.0, xmax=5.0, xstep=20, ynum=26, ymin=2.0, ymax=4.0, ystep=20):
        # zgauss: 22 z, 26 w
        modname = f'{identifier}_{line}_rebin_{rebin}_{line_model}'
        filename = f'./xcm/{identifier}_{line}_rebin_{rebin}_{line_model}.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        Xset.restore(filename)
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def re_run_steppar(self, identifier='ssm', line='wz', rebin=1, line_model='dzgaus', xnum="outer:2", xmin=6, xmax=9, xstep=20, ynum="outer:26", ymin=3.0, ymax=9.0, ystep=20):
        modname = f'{identifier}_{line}_rebin_{rebin}_{line_model}'
        #parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        parameter_set = f'{str(xnum)}_{str(ynum)}'
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def re_run_steppar2(self, identifier='ssm', line='wz', rebin=1, line_model='dzgaus', xnum=2, xmin=4.15, xmax=5.0, xstep=30, ynum=26, ymin=1.8, ymax=3.5, ystep=30):
        modname = f'{identifier}_{line}_rebin_{rebin}_{line_model}'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def run_fake(self, spec, exposure=100, model_xcm="dzgaus_model.xcm", xnum=2, xmin=5.15, xmax=5.7, xstep=20, ynum=26, ymin=2.0, ymax=3.4, ystep=20):
        #xnum=2, xmin=5.1, xmax=5.7, xstep=20, ynum=26, ymin=2.25, ymax=3.25, ystep=20
        modname = f'fake_{exposure}ksec'
        filename = model_xcm
        AllData.clear()
        Spectrum(dataFile=spec, respFile="1000_center_L_without_Lp.rmf", arfFile="1000_center_image_1p8_8keV_1e7.arf")
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        Xset.restore(filename)
        self.set_data_range((2.0, 17.0))
        self.fit_error(error_calc=False,detector=False)
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def run_steppar_zgaus(self, identifier='S_all', region='center', line='wz', rebin=1, xnum=22, xmin=2.5e-5, xmax=5.5e-5, xstep=20, ynum=26, ymin=7e-5, ymax=12e-5, ystep=20):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}_zgaus'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        Xset.restore(filename)
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.plot_error(identifier, region, line, rebin)
        self.save_steppar_result(modname, parameter_set)

    def run_steppar_dzgaus(self, identifier='S_all', region='center', line='wz', rebin=1, xnum=2, xmin=2.5, xmax=5.5, xstep=20, ynum=26, ymin=7e-5, ymax=12e-5, ystep=20):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}_dzgaus'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}_dzgaus.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour_load(modname, parameter_set)
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,8.0,1000)
        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
            if up == 2:
                z = ldata['epsilon']
                print(ldata['energy'])
            if up == 7:
                w = ldata['epsilon']
                print(ldata['energy'])
        self.ax.plot(kTlist, w/z, '-.',color='orange', label='apec')
        self.ax.set_xlim(np.min(self.X), np.max(self.X))
        self.ax.set_ylim(np.min(self.Y), np.max(self.Y))
        self.fig.savefig(f'steppar_wapec_{region}.png', dpi=300, transparent=True)
        plt.show()
        #self.save_steppar_result(modname, parameter_set)

    def run_steppar_dzgaus_v(self, identifier='S_all', region='center', line='wz', rebin=1, xnum=17, xmin=2.5, xmax=5.5, xstep=20, ynum=26, ymin=7e-5, ymax=12e-5, ystep=20):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}_dzgaus'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}_dzgaus.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour_load(modname, parameter_set)
        self.fig.savefig(f'steppar_wv_{region}.png', dpi=300, transparent=True)
        plt.show()

    def run_steppar_zgaus_load(self, identifier='1000', region='center', line='wz', rebin=1, xnum=22, xmin=1e-5, xmax=6e-5, xstep=10, ynum=26, ymin=7e-5, ymax=13e-5, ystep=10):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour_load(modname, parameter_set)
        self.plot_error(identifier, region, line, rebin)
        self.save_steppar_result(modname, parameter_set)

    def run_steppar_zgaus_load_ssm(self, identifier='ssm', line='wz', rebin=1, region='center', xnum=22, xmin=1e-5, xmax=6e-5, xstep=10, ynum=26, ymin=7e-5, ymax=13e-5, ystep=10):
        modname = f'{identifier}_{line}_rebin_{rebin}_zgauss'
        filename = f'./xcm/{identifier}_{line}_rebin_{rebin}_zgauss.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour_load(modname, parameter_set)
        #self.plot_error_ssm(identifier, line, rebin, region)
        #self.save_steppar_result(modname, parameter_set)

    def run_steppar_dzgaus_load_ssm(self, identifier='ssm', line='wz', rebin=1, region='center', xnum=2, xmin=1e-5, xmax=6e-5, xstep=10, ynum=26, ymin=7e-5, ymax=13e-5, ystep=10):
        modname = f'{identifier}_{line}_rebin_{rebin}_dzgaus'
        filename = f'./xcm/{identifier}_{line}_rebin_{rebin}_dzgaus.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        self.plot_contour_load(modname, parameter_set)

    def run_steppar_v(self, identifier='S_all', region='center', line='wz', rebin=1, line_model='dzgaus', xnum=17, xmin=0, xmax=300, xstep=10, ynum=26, ymin=1.5, ymax=4.0, ystep=10):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}_{line_model}'
        filename = f'./xcm/{identifier}_{region}_{line}_rebin_{rebin}_{line_model}.xcm'
        parameter_set = f'{str(int(xnum))}_{str(int(ynum))}'
        Xset.restore(filename)
        self.plot_contour(xnum, xmin, xmax, xstep, ynum, ymin, ymax, ystep)
        self.save_steppar_result(modname, parameter_set)

    def steppar_plot_test(self):
        # self.load_steppar_result('1000_center_wz_rebin_1_dzgaus', '2_26')
        # self.oplot_contour(new=True, label='Open')
        # self.load_steppar_result('S_other_center_wz_rebin_1_dzgaus', '2_26')
        # self.oplot_contour(new=False,line_style='-.',label='Other filter')
        self.load_steppar_result('ssm_wz_rebin_1_dzgaus', 'outer:2_outer:26')
        self.oplot_contour(new=True,line_style='-',label='Other filter')
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,9.0,1000)
        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
            if up == 2:
                z = ldata['epsilon']
                print(ldata['energy'])
            if up == 7:
                w = ldata['epsilon']
                print(ldata['energy'])
        self.ax.plot(kTlist, w/z,color='orange', label='apec')
        self.ax.set_xlim(np.min(self.X), np.max(self.X))
        self.ax.set_ylim(np.min(self.Y), np.max(self.Y))
        self.fig.savefig(f'steppar_wapec.png', dpi=300, transparent=True)
        plt.show()

    def wz_temp_pre(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        sess = pyatomdb.spectrum.CIESession()
        kTlist = np.linspace(4.0,8.0,1000)
        for up in [2,3,4,5,6,7]:
            ldata = sess.return_line_emissivity(kTlist, 26, 25, up, 1)
            if up == 2:
                z = ldata['epsilon']
                print(ldata['energy'])
            if up == 7:
                w = ldata['epsilon']
                print(ldata['energy'])
        self.ax.plot(kTlist, w/z,color='darkblue', label='CIE (atomdb v3.0.9)')
        self.ax.errorbar(4.53, 2.07, yerr=np.vstack(([0.6], [0.6])),xerr=np.vstack(([0.13], [0.13])), fmt='o', color='red', label='center')
        self.ax.errorbar(7.75, 7.12, yerr=np.vstack(([3.75], [23.88])),xerr=np.vstack(([0.5], [0.5])), fmt='o', color='blue', label='outer')
        self.ax.set_xlabel('kT (keV)', fontsize=15)
        self.ax.set_ylabel('w/z flux ratio', fontsize=15)
        self.ax.set_ylim(1.2,7.5)
        self.ax.legend(fontsize=15)
        plt.show()
        self.fig.savefig(f'wz_apec_pre.png', dpi=300, transparent=True)

    def plot_error(self, identifier='1000', region='center', line='wz', rebin=1):
        modname = f'{identifier}_{region}_{line}_rebin_{rebin}'
        ratiom = f'{modname}_dzgaus'
        zgausm = f'{modname}'
        with h5py.File(self.savefile, 'a') as f:
            if zgausm not in f.keys():
                print(f'The group {zgausm} is not included in the file.')
            else:
                w = f[f'{zgausm}/fitting_result']['4/zgauss_4']['norm']['value'][...]
                w_ep = f[f'{zgausm}/fitting_result']['4/zgauss_4']['norm']['ep'][...]
                w_em = f[f'{zgausm}/fitting_result']['4/zgauss_4']['norm']['em'][...]
                z = f[f'{zgausm}/fitting_result']['3/zgauss']['norm']['value'][...]
                z_ep = f[f'{zgausm}/fitting_result']['3/zgauss']['norm']['ep'][...]
                z_em = f[f'{zgausm}/fitting_result']['3/zgauss']['norm']['em'][...]

            if ratiom not in f.keys():
                print(f'The group {ratiom} is not included in the file.')
            else:
                ratio = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['value'][...]
                ratio_ep = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['ep'][...]
                ratio_em = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['em'][...]

        print(w, w_ep, w_em)
        print(z, z_ep, z_em)
        print(ratio, ratio_ep, ratio_em)
        xrng = np.linspace(1e-5, 10e-5, 1000)
        y_cen = ratio * xrng
        y_ep = (ratio + ratio_ep) * xrng
        y_em = (ratio + ratio_em) * xrng
        self.ax.plot(xrng, y_cen, color='black')
        print("ber")
        self.ax.fill_between(xrng, y_ep, y_em, color='gray', alpha=0.25)
        r, r_ep, r_em = self.calc_wz_err(w, z, w_ep, z_ep, -w_em, -z_em)
        print(r, r_ep, r_em)
        self.ax.fill_between(xrng, (r+r_ep)*xrng, (r-r_em)*xrng, color='blue', alpha=0.25)
        self.ax.errorbar(z, w, xerr=np.vstack((-z_em,z_ep)), yerr=np.vstack((-w_em,w_ep)), markersize=5, capsize=5, fmt='o', color='red', )

        x_min = np.min(z + z_em)
        x_max = np.max(z + z_ep)
        y_min = np.min(w + w_em)
        y_max = np.max(w + w_ep)

        big_rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--'
        )
        self.ax.add_patch(big_rect)
        plt.show()
        self.fig.savefig(f'./figure/{modname}_steppar.pdf')

    def calc_wz_err(self, w, z, w_ep, z_ep, w_em, z_em):
        """
        w/z の比と非対称誤差を返す。
        誤差入力の符号に関係なく、内部では絶対値を用いて計算する。

        Parameters
        ----------
        w, z : float
            値
        w_ep, z_ep : float
            +側（上側）誤差。符号は気にしなくて良い。
        w_em, z_em : float
            -側（下側）誤差。符号は気にしなくて良い。

        Returns
        -------
        ratio : float
            w/z
        ratio_ep : float
            +側誤差（必ず正）
        ratio_em : float
            -側誤差（必ず正）
        """
        # 誤差は絶対値で扱う
        w_ep, z_ep = abs(w_ep), abs(z_ep)
        w_em, z_em = abs(w_em), abs(z_em)

        ratio = w / z

        # +側：w が大きく & z が小さく振れる組合せ
        ratio_plus  = (w + w_ep) / (z - z_em)
        ratio_ep    = ratio_plus - ratio

        # −側：w が小さく & z が大きく振れる組合せ
        ratio_minus = (w - w_em) / (z + z_ep)
        ratio_em    = ratio - ratio_minus

        return ratio, ratio_ep, ratio_em

    def plot_error_ssm(self, identifier='ssm', line='wz', rebin=1, region='center'):
        modname = f'{identifier}_{line}_rebin_{rebin}'
        ratiom = f'{modname}_dzgaus'
        zgausm = f'{modname}_zgauss'
        with h5py.File(self.savefile, 'a') as f:
            if zgausm not in f.keys():
                print(f'The group {zgausm} is not included in the file.')
            else:
                w = f[f'{zgausm}/fitting_result/{region}']['4/zgauss_4']['norm']['value'][...]
                w_ep = f[f'{zgausm}/fitting_result/{region}']['4/zgauss_4']['norm']['ep'][...]
                w_em = f[f'{zgausm}/fitting_result/{region}']['4/zgauss_4']['norm']['em'][...]
                z = f[f'{zgausm}/fitting_result/{region}']['3/zgauss']['norm']['value'][...]
                z_ep = f[f'{zgausm}/fitting_result/{region}']['3/zgauss']['norm']['ep'][...]
                z_em = f[f'{zgausm}/fitting_result/{region}']['3/zgauss']['norm']['em'][...]

            if ratiom not in f.keys():
                print(f'The group {ratiom} is not included in the file.')
            else:
                ratio = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['value'][...]
                ratio_ep = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['ep'][...]
                ratio_em = f[f'{ratiom}/fitting_result']['3/dzgaus']['ratio']['em'][...]

        print(w, w_ep, w_em)
        print(z, z_ep, z_em)
        xrng = np.linspace(1e-5, 10e-5, 1000)
        y_cen = ratio * xrng
        y_ep = (ratio + ratio_ep) * xrng
        y_em = (ratio + ratio_em) * xrng
        self.ax.plot(xrng, y_cen, color='black')
        self.ax.fill_between(xrng, y_ep, y_em, color='gray', alpha=0.5)
        #self.ax.errorbar(z, w, xerr=np.vstack((-z_em,z_ep)), yerr=np.vstack((-w_em,w_ep)), markersize=5, capsize=5, fmt='o', color='red', )
        
        plt.show()

    def make_latex_table(self, keyname):
        import h5py
        from jinja2 import Template
        import subprocess
        import numpy as np

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマットする"""
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        # HDF5ファイルの読み込み
        file_path = self.savefile
        keyname = keyname

        with h5py.File(file_path, 'r') as f:
            dataset = f[keyname]['fitting_result']
            number_keys = list(dataset.keys())

            # データの抽出と整理
            data = {}
            for number in number_keys:
                comp = list(dataset[number].keys())[0]
                keys = list(dataset[number][comp].keys())
                for key in keys:
                    v_e = list(dataset[number][comp][key].keys())
                    if 'em' in v_e:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        value = dataset[number][comp][base_key]['value'][()]
                        ep = dataset[number][comp][base_key]['ep'][()]
                        em = dataset[number][comp][base_key]['em'][()]
                        values = self.formatter([value, ep, em], 2)
                        data[modname][base_key] = {
                            'value': values[0],
                            'ep': values[1],
                            'em': values[2]
                        }
                    else:
                        base_key = key
                        modname = f'{number}:{comp}'
                        if modname not in data.keys():
                            data[modname] = {}
                        data[modname][base_key] = {
                            'value': format_significant_figures(dataset[number][comp][base_key]['value'][()], 6)
                        }

            # statisticとdofの読み込み
            statistic = format_significant_figures(f[keyname]['statistic'][()],6)
            dof = f[keyname]['dof'][()]

        print('-------------')
        print(statistic, dof)
        
        # 各key_numに対するパラメータ数の計算
        param_counts = {key_num: len(values) for key_num, values in data.items()}

        latex_template = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{multirow}
        \renewcommand{\arraystretch}{r}
        \pagestyle{empty}
        \begin{document}
        \renewcommand{\arraystretch}{0.5}\begin{table}[htbp]
            \centering
            \scalebox{1.5}{\begin{tabular}{|c|c|c|}
            \hline
            Model Number & Parameter & Value \\
            \hline
            {% for key_num, values in data.items() %}
                {% set row_count = param_counts[key_num] %}
                {% for key, value in values.items() %}
                    {% if loop.first %}
                        \multirow {{row_count}}{}{}{{ key_num }} & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% else %}
                        & {{ key }} & ${{ value['value']}}{% if 'ep' in value and 'em' in value %}^{+{{value['ep']}}}_{ {{value['em']}}}{% endif %}$ \\
                    {% endif %}
                {% endfor %}
            \hline
            {% endfor %}
            Statistic & \multicolumn{2}{c|}{${{ statistic }}$} \\
            \hline
            dof & \multicolumn{2}{c|}{${{ dof }}$} \\
            \hline
            \end{tabular}
            }
        \end{table}
        \end{document}
        """



        # Jinja2テンプレートのレンダリング
        template = Template(latex_template)
        latex_content = template.render(data=data, param_counts=param_counts, statistic=statistic, dof=dof)

        # LaTeXソースファイルの書き込み
        with open(f'{keyname}.tex', 'w') as f:
            f.write(latex_content)

        # LaTeXコンパイルを実行してPDFを生成（非対話モードで）
        subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{keyname}.tex'])

        print("PDF table has been generated as table.pdf")

    def make_latex_table_format(self, keyname, fix=False):
        import h5py
        import numpy as np

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマットする"""
            if value == 0:
                return 0
            return round(value, significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        # HDF5ファイルの読み込み
        file_path = self.savefile
        keyname = keyname

        with h5py.File(file_path, 'r') as f:
            dataset = f[keyname]['fitting_result']
            number_keys = list(dataset.keys())

            # データの抽出と整理
            data = []
            for number in number_keys:
                comp = list(dataset[number].keys())[0]
                keys = list(dataset[number][comp].keys())
                for key in keys:
                    v_e = list(dataset[number][comp][key].keys())
                    if fix == False:
                        if 'em' in v_e:
                            base_key = key
                            modname = f'{number}:{comp}'
                            value = dataset[number][comp][base_key]['value'][()]
                            ep = dataset[number][comp][base_key]['ep'][()]
                            em = dataset[number][comp][base_key]['em'][()]
                            values = self.formatter([value, ep, em], 2)
                            data.append({
                                'model': modname,
                                'parameter': base_key,
                                'value': f'${values[0]}^{{+{values[1]}}}_{{{values[2]}}}$'
                            })
                        else:
                            pass
                    else:
                        if "em" in v_e:
                            base_key = key
                            modname = f'{number}:{comp}'
                            value = dataset[number][comp][base_key]['value'][()]
                            ep = dataset[number][comp][base_key]['ep'][()]
                            em = dataset[number][comp][base_key]['em'][()]
                            if "Sigma" in base_key:
                                values = self.formatter([value*1e3, ep*1e3, em*1e3], 2)
                            else:
                                values = self.formatter([value, ep, em], 2)
                            data.append({
                                'model': modname,
                                'parameter': base_key,
                                'value': f'${values[0]}^{{+{values[1]}}}_{{{values[2]}}}$'
                            })

                        else:
                            base_key = key
                            modname = f'{number}:{comp}'
                            value = dataset[number][comp][base_key]['value'][()]
                            values = self.formatter([value, 1.1111, 1.1111], 5)
                            data.append({
                                'model': modname,
                                'parameter': base_key,
                                'value': f'${values[0]}$'
                            })


            # statisticとdofの読み込み
            statistic = format_significant_figures(f[keyname]['statistic'][()], 6)
            dof = f[keyname]['dof'][()]

        # pandas DataFrame に変換
        df = pd.DataFrame(data)

        # 結果を表示
        print(f'Statistic: {statistic}, DOF: {dof}')
        print(df)

        return df

    def make_latex_table_format_ssm(self, keyname, fix=False):
        import h5py
        import numpy as np
        import pandas as pd      # ← DataFrame 生成に必要

        def format_significant_figures(value, significant_digits):
            """数値を指定された有効数字でフォーマット"""
            if value == 0:
                return 0
            return round(value,
                        significant_digits - int(np.floor(np.log10(abs(value)))) - 1)

        file_path = self.savefile
        data = []  # ← region も含めた結果を順次追加

        with h5py.File(file_path, "r") as f:
            # 統計量と自由度は従来どおり
            statistic = format_significant_figures(f[keyname]["statistic"][()], 6)
            dof       = f[keyname]["dof"][()]

            # center・outer の 2 領域を順に処理
            for region in ("center", "outer"):
                print(f[keyname].keys())
                dataset = f[keyname]["fitting_result"][region]
                number_keys = list(dataset.keys())

                for number in number_keys:
                    comp = list(dataset[number].keys())[0]
                    for base_key in dataset[number][comp]:
                        v_e = list(dataset[number][comp][base_key].keys())

                        # ------ 誤差付きパラメータ ------
                        if "em" in v_e:
                            value = dataset[number][comp][base_key]["value"][()]
                            ep    = dataset[number][comp][base_key]["ep"][()]
                            em    = dataset[number][comp][base_key]["em"][()]

                            # Sigma だけ keV→eV 換算
                            if fix and "Sigma" in base_key:
                                scale = 1e3
                                value, ep, em = value*scale, ep*scale, em*scale

                            values = self.formatter([value, ep, em], 2 if not fix else 2)
                            latex_val = f"${values[0]}^{{+{values[1]}}}_{{{values[2]}}}$"

                        # ------ 固定パラメータ ------
                        else:
                            value = dataset[number][comp][base_key]["value"][()]
                            values = self.formatter([value, 1.0, 1.0], 5)
                            latex_val = f"${values[0]}$"

                        data.append({
                            "region"   : region,           # ← 追加
                            "model"    : f"{number}:{comp}",
                            "parameter": base_key,
                            "value"    : latex_val
                        })

        # DataFrame 化
        df = pd.DataFrame(data)

        # 結果表示（任意）
        print(f"Statistic: {statistic}, DOF: {dof}")
        display(df[:60])
        display(df[60:])

        return df

    def make_latex_table_format_ssm2(self, keyname):
        """
        2:bvapec      → kT, Velocity, Fe, N, Redshift, Redshift_2–8, norm
        3:zgauss      → すべて（“3:” 表示を外す）
        4:zgauss_4    → すべて
        Σ(zgauss*)    → keV → eV 換算（×1e3）
        小/大さい数   → ( … )×10^{n} 形式
        """

        import h5py, numpy as np, pandas as pd

        # ---------- 設定 ----------
        bvapec_param_order = (
            ["kT", "Velocity", "Fe", "Ni", "Redshift"]
            + [f"Redshift_{i}" for i in range(2, 9)]
            + ["norm"]
        )
        wanted_bvapec = set(bvapec_param_order)
        #target_models  = {"2:bvapec", "3:zgauss", "4:zgauss_4"}
        target_models  = {"2:bvapec", "3:bvapec_3"}
        region_order   = ["center", "outer"]
        model_display  = {"2:bvapec": "bvapec", "3:zgauss": "zgauss", "4:zgauss_4": "zgauss_4"}
        model_display  = {"2:bvapec":"bvapec", "3:bvapec_3":"bvapec_2"}
        model_sort_key = ["bvapec", "zgauss", "zgauss_4"]
        model_sort_key  = ["2:bvapec", "3:bvapec_3"]
        # ---------- 数値 → LaTeX ----------
        def latex_triplet(val, ep=None, em=None, sig=2):
            """
            数値 → LaTeX 形式
            • ep / em が None → 固定値
            • ep / em がある  → 誤差付き
            • |exp| ≥ 3       → 10^n 表記
            小数点以下桁は「誤差の桁を見て」自動決定
            """
            import numpy as np

            def _decimals(x, s=sig):
                """誤差 x に対し、有効数字 s 桁が残るだけ小数点以下を確保"""
                if x == 0:
                    return 2          # 保険
                exp = int(np.floor(np.log10(abs(x))))
                return max(0, s - 1 - exp)

            # 10^n を使うかどうか判定
            exp_val = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
            use_exp = (exp_val <= -3) or (exp_val >= 3)

            # ----- 10^n 表記 -----
            if use_exp:
                fac = 10**exp_val
                v   = val / fac
                if ep is None:
                    body = f"{v:.{5}f}"
                else:
                    ep_s = abs(ep) / fac
                    em_s = abs(em) / fac
                    d    = max(_decimals(ep_s), _decimals(em_s))
                    body = (f"{v:.{d}f}^{{+{ep_s:.{d}f}}}"
                            f"_{{-{em_s:.{d}f}}}")
                return rf"$({body})\times10^{{{exp_val}}}$"

            # ----- 通常表記 -----
            if ep is None:
                d = max(_decimals(val), 2)
                return rf"$ {val:.{5}f} $"
            else:
                d = max(_decimals(ep), _decimals(em))
                return rf"$ {val:.{d}f}^{{+{abs(ep):.{d}f}}}_{{-{abs(em):.{d}f}}}$"

        def is_digit_key(k):
            s = k.decode() if isinstance(k, bytes) else str(k)
            return s.isdigit()

        # ---------- 走査 ----------
        rows, file_path = [], self.savefile
        with h5py.File(file_path, "r") as f:
            stat = round(f[keyname]["statistic"][()], 6)
            dof  = int(f[keyname]["dof"][()])

            for region in region_order:
                dset = f[keyname]["fitting_result"][region]

                for num in sorted(filter(is_digit_key, dset), key=lambda x: int(x.decode() if isinstance(x, bytes) else x)):
                    comp  = next(iter(dset[num]))
                    comp  = comp.decode() if isinstance(comp, bytes) else comp
                    mfull = f"{num}:{comp}"
                    if mfull not in target_models:
                        continue

                    for par in dset[num][comp]:
                        # --- bvapec パラメータ選別 ---
                        if (mfull == "2:bvapec" or mfull == "3:bvapec_3") and par not in wanted_bvapec:
                            continue

                        ve = dset[num][comp][par]
                        if "em" in ve:                       # 可変
                            v, ep, em = (ve[k][()] for k in ("value", "ep", "em"))
                        else:                                # 固定
                            v, ep, em = ve["value"][()], None, None

                        # Σ → eV 換算
                        if num in ("3", "4") and "Sigma" in par:
                            v, ep, em = v * 1e3, (ep or 0) * 1e3, (em or 0) * 1e3

                        rows.append(
                            dict(region=region,
                                model=model_display[mfull],
                                parameter=par,
                                value=latex_triplet(v, ep, em, sig=2))
                        )

        df = pd.DataFrame(rows)

        # ---------- 並べ替え ----------
        df["region"]   = pd.Categorical(df["region"],   region_order,   ordered=True)
        df["model"]    = pd.Categorical(df["model"],    model_sort_key, ordered=True)
        def param_rank(r):
            if r["model"] == "bvapec":
                return bvapec_param_order.index(r["parameter"])
            return 100 + hash(r["parameter"])            # bvapec 以外は適当で OK
        df["p_rank"] = df.apply(param_rank, axis=1)
        df = df.sort_values(["region", "model", "p_rank"]).drop(columns="p_rank")

        # ---------- 参考表示 ----------
        print(f"Statistic = {stat},  DOF = {dof}")
        display(df.head(80))

        return df

    def make_latex_table_all(self):
        f = h5py.File(self.savefile, 'r')
        keynames = list(f.keys())
        f.close()
        for keyname in keynames:
            self.make_latex_table(keyname)

    def make_apec_model(self, del_line='all', output_dir = '/Users/keitatanaka/apec_modify', atomdb_version='3.0.9'):
        # re: cp coco_file from opt/heasoft
        from astropy.io import fits
        from astropy import constants
        from astropy import units
        import numpy as np
        from pathlib import Path
        import shutil

        atomdb = os.environ['ATOMDB'] 
        line_file_path = f'{atomdb}/apec_v{atomdb_version}_line.fits'
        coco_file_path = f'{atomdb}/apec_v{atomdb_version}_coco.fits'
        output_path = f'{output_dir}/v{atomdb_version}'
        
        f = fits.open(line_file_path)
        for i in range(2,len(f)):
            # データを取得
            data = f[i].data
            element = np.array(data['Element'])
            ion = np.array(data['Ion'])
            up = np.array(data['UpperLev'])
            down = np.array(data['LowerLev'])
            Lambda = np.array(data['Lambda'])

            # エネルギー計算
            E = constants.c * constants.h / (Lambda * units.angstrom)

            # 条件に合致するマスクを作成
            if del_line == 'all':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))| ((ion == 25) & (element == 26) & (up == 2) & (down == 1)) | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  | ((ion == 26) & (element == 26) & (up == 4) & (down == 1))  | ((ion == 25) & (element == 26) & (up == 6) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 5) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 11) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))
                output_path_del = f'{output_path}/del_Heab_Lya'

            elif del_line == 'Heab':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))| ((ion == 25) & (element == 26) & (up == 2) & (down == 1))  | ((ion == 25) & (element == 26) & (up == 6) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 5) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 11) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))
                output_path_del = f'{output_path}/del_Heab'           

            elif del_line == 'w_z_Lya':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))| ((ion == 25) & (element == 26) & (up == 2) & (down == 1)) | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  | ((ion == 26) & (element == 26) & (up == 4) & (down == 1)) 
                output_path_del = f'{output_path}/del_w_z_Lya'

            elif del_line == 'w_z':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1)) | ((ion == 25) & (element == 26) & (up == 2) & (down == 1))
                output_path_del = f'{output_path}/del_w_z'

            elif del_line == 'w':
                mask = ((ion == 25) & (element == 26) & (up == 7) & (down == 1))
                output_path_del = f'{output_path}/del_w'
            # |   (ion == 25) & (element == 26) & (up == 7) & (down == 1))  w
            # | ((ion == 25) & (element == 26) & (up == 2) & (down == 1))  z
            # | ((ion == 25) & (element == 26) & (up == 6) & (down == 1))  x
            # | ((ion == 25) & (element == 26) & (up == 5) & (down == 1))  y
            # | ((ion == 25) & (element == 26) & (up == 11) & (down == 1))  He b2
            # | ((ion == 25) & (element == 26) & (up == 13) & (down == 1))  He b1
            # | ((ion == 26) & (element == 26) & (up == 3) & (down == 1))  Lya2 
            # | ((ion == 26) & (element == 26) & (up == 4) & (down == 1)   Lya1
            new_data = data[~mask]

            # 新しいHDUを作成（既存のHDUヘッダーを再利用）
            f[i] = fits.BinTableHDU(data=new_data, header=f[i].header)
            print(E[mask==True].to('keV'))
        
        Path(output_path_del).mkdir(exist_ok=True)
        # 新しいファイルに保存
        new_file_path = f'{output_path_del}/apec_v{atomdb_version}_line.fits'
        print(new_file_path)
        f.writeto(new_file_path, overwrite=True)

        # ファイルを閉じる
        f.close()
        shutil.copy(coco_file_path, f'{output_path_del}/apec_v{atomdb_version}_coco.fits')

    def fake_spec(self,model_xcm,resp,arf,exposure,savefile):
        Xset.restore(model_xcm)
        fs = FakeitSettings(response=resp, arf=arf, exposure=exposure, fileName=savefile)
        AllData.fakeit(nSpectra=1, settings=fs, applyStats=True, filePrefix='', noWrite=False)

    def rebin_spec(self,spec_file,savefile,rebin):
        import subprocess
        commands = f"""
ftgrouppha infile={spec_file} outfile={savefile} grouptype=min groupscale={rebin} clobber=yes
"""
        subprocess.run(commands, shell=True, check=True)
    
    def rebin_all(self):
        for i in glob.glob("*.fak"):
            basename = os.path.basename(i)
            name_without_extension = os.path.splitext(basename)[0]
            self.rebin_spec(spec_file=i,savefile=f'./{name_without_extension}_b1.pi',rebin=1)

    def fit_fake_spec(self, spec, model_xcm, savemod):
        AllData.clear()
        Xset.restore(model_xcm)
        Spectrum(spec)
        self.set_data_range(rng=(2.0,17.0))
        self.fit_error(error='1.0 2,26',error_calc=False,detector=False)
        self.set_xydata_multi([savemod])
        self.result_pack(model=AllModels(1))
        self.savemod_multi(savemod)

    def fit_fake_spec_all(self):
        for i in glob.glob("*_b1.fak"):
            basename = os.path.basename(i)
            name_without_extension = os.path.splitext(basename)[0]
            self.fit_fake_spec(spec=i,model_xcm='fakeit_model_ratio.xcm',savemod=name_without_extension)

    def _fake_test(self, resp='1000_center_L_without_Lp.rmf', arf='1000_center_image_1p8_8keV_1e7.arf', exposure=44.13e3,  rebin=1, model_xcm='dzgaus_model.xcm', num=30, savedir = f'./fakeit_result_ratio_300ksec'):
        import time
        start = time.time()
        for i in range(num):
            self.fake_spec(model_xcm=model_xcm,
                       resp=resp,
                       arf=arf,
                       exposure=exposure,
                       savefile=f'{savedir}/fakeit_spec_{i}.pi')
        for i in range(num):
            self.rebin_spec(spec_file=f'{savedir}/fakeit_spec_{i}.pi',savefile=f'{savedir}/fakeit_spec_{i}_b1.pi',rebin=rebin)
        for i in range(num):
            self.fit_fake_spec(spec=f'{savedir}/fakeit_spec_{i}_b1.pi',model_xcm="fakeit_model_ratio.xcm",savemod=f'fake_{i}')
        end = time.time()  # 終了時刻
        print(f"実行時間: {end - start:.2f} 秒")

    def search_most_match_number(self, value=2.5):
        ratio_list = []
        key_list = []

        # HDF5 から値を読み出し
        with h5py.File(f'{self.savefile}', 'r') as f:
            for key in f.keys():
                ratio = f[f'{key}/fitting_result/3/dzgaus/ratio/value'][...]
                ratio_list.append(ratio)
                key_list.append(key)

        ratio_list = np.array(ratio_list)
        ratio_abs = np.abs(ratio_list - value)

        # ソートされたインデックスを取得（最小に近い順）
        sorted_indices = np.argsort(ratio_abs)

        results = []
        for i in range(min(2, len(sorted_indices))):  # 最大2件
            idx = sorted_indices[i]
            ratio_val = ratio_list[idx]
            key_val = key_list[idx]
            fake_num = re.search(r'\d+', key_val).group()
            results.append((key_val, fake_num, ratio_val))

            print(f"{i+1} 番目に近い値: {ratio_val}, キー: {key_val}")

        return results  # [(key1, num1, val1), (key2, num2, val2)]

    def plot_fake_result(self):
        v_list = []
        w_list = []
        z_list = []
        ratio_list = []
        with h5py.File(f'{self.savefile}', 'r') as f:
            for key in f.keys():
                v = f[f'{key}/fitting_result/2/bvapec/Velocity/value'][...]
                w = f[f'{key}/fitting_result/4/zgauss_4/norm/value'][...]
                z = f[f'{key}/fitting_result/3/zgauss/norm/value'][...]
                ratio = w / z
                v_list.append(v)
                ratio_list.append(ratio)
        v_list = np.array(v_list)
        ratio_list = np.array(ratio_list)
        print(ratio_list)
        # plt.hist(v_list, bins=50, histtype='step', color='k')
        # plt.axvline(np.mean(v_list), color='r', linestyle='dashed')
        # plt.axvspan(np.mean(v_list) - np.std(v_list), np.mean(v_list) + np.std(v_list), color='r', alpha=0.2)
        # plt.xlabel('velocity [km/s]')
        # plt.ylabel('count')

        plt.hist(ratio_list, bins=50, histtype='step', color='k')
        plt.axvline(np.mean(ratio_list), color='r', linestyle='dashed')
        #plt.axvspan(np.mean(ratio_list) - np.std(ratio_list), np.mean(ratio_list) + np.std(ratio_list), color='r', alpha=0.2)
        plt.xlabel('ratio')
        plt.ylabel('count')


        param_values = ratio_list
        sorted_values = np.sort(param_values)

        # 信頼区間の関数
        def get_confidence_interval(data, level=0.683):
            lower = (1 - level) / 2 * 100
            upper = (1 + level) / 2 * 100
            return np.percentile(data, [lower, upper])

        # 各信頼区間
        ci_1sigma = get_confidence_interval(param_values, level=0.683)
        ci_90 = get_confidence_interval(param_values, level=0.90)
        ci_95 = get_confidence_interval(param_values, level=0.95)

        print(f"68.3% (1σ) CI: {ci_1sigma[0]:.3f} – {ci_1sigma[1]:.3f}")
        print(f"90% CI       : {ci_90[0]:.3f} – {ci_90[1]:.3f}")
        print(f"95% CI       : {ci_95[0]:.3f} – {ci_95[1]:.3f}")    

        plt.axvspan(ci_1sigma[0], ci_1sigma[1], color='r', alpha=0.2)

        plt.show()

    def plot_fake_result_ratio(self):
        ratio_list = []
        with h5py.File(f'{self.savefile}', 'r') as f:
            for key in f.keys():
                ratio = f[f'{key}/fitting_result/3/dzgaus/ratio/value'][...]
                ratio_list.append(ratio)
        ratio_list = np.array(ratio_list)
        print(ratio_list)
        # plt.hist(v_list, bins=50, histtype='step', color='k')
        # plt.axvline(np.mean(v_list), color='r', linestyle='dashed')
        # plt.axvspan(np.mean(v_list) - np.std(v_list), np.mean(v_list) + np.std(v_list), color='r', alpha=0.2)
        # plt.xlabel('velocity [km/s]')
        # plt.ylabel('count')

        plt.hist(ratio_list, bins=50, histtype='step', color='k')
        plt.axvline(np.mean(ratio_list), color='r', linestyle='dashed')
        #plt.axvspan(np.mean(ratio_list) - np.std(ratio_list), np.mean(ratio_list) + np.std(ratio_list), color='r', alpha=0.2)
        plt.xlabel('ratio')
        plt.ylabel('count')


        param_values = ratio_list
        sorted_values = np.sort(param_values)

        # 信頼区間の関数
        def get_confidence_interval(data, level=0.683):
            lower = (1 - level) / 2 * 100
            upper = (1 + level) / 2 * 100
            return np.percentile(data, [lower, upper])

        # 各信頼区間
        ci_1sigma = get_confidence_interval(param_values, level=0.683)
        ci_90 = get_confidence_interval(param_values, level=0.90)
        ci_95 = get_confidence_interval(param_values, level=0.95)

        print(f"68.3% (1σ) CI: {ci_1sigma[0]:.3f} – {ci_1sigma[1]:.3f}")
        print(f"90% CI       : {ci_90[0]:.3f} – {ci_90[1]:.3f}")
        print(f"95% CI       : {ci_95[0]:.3f} – {ci_95[1]:.3f}")    

        plt.axvspan(ci_1sigma[0], ci_1sigma[1], color='r', alpha=0.2)

        plt.show()

    def result_plot_ssm_z(self, group_name):
        """
        Parameters
        ----------
        group_name : str
            HDF5 内の最上位グループ名（例: 'pks0745' など）
        """
        # ------------ 1. データ読み込み＆辞書化 ------------
        zdata = {reg: {} for reg in ('center', 'outer')}  # {reg: {idx: (val, +err, -err)}}
        with h5py.File(self.savefile, "r") as f:
            if group_name not in f:
                raise KeyError(f"'{group_name}' not found in {self.savefile}")

            for reg in ('center', 'outer'):
                for idx in range(1, 9):                       # 1〜8 番まで
                    suffix = "" if idx == 1 else f"_{idx}"
                    base   = f"{group_name}/fitting_result/{reg}/2/bvapec/Redshift{suffix}"
                    z_val  = f[f"{base}/value"][()]
                    z_ep   = f[f"{base}/ep"][()]
                    z_em   = f[f"{base}/em"][()]
                    zdata[reg][idx] = (z_val, z_ep, z_em)

        # ------------ 2. （例）辞書を使った計算 ------------
        #   red1 の差分を取りたいなら：
        delta_red1 = zdata["center"][1][0] - zdata["outer"][1][0]
        print(f"Δz (center−outer) for Redshift1 = {delta_red1:.4e}")

        # ------------ 3. プロット ------------
        P = PlotManager((2, 2), (12.8, 8), sharex=False)
        colors = {"center": "darkred", "outer": "darkblue"}

        for reg in ("center", "outer"):
            col = colors[reg]
            for idx in range(1, 9):
                x = idx                                   # x 軸位置
                z_val, z_ep, z_em = zdata[reg][idx]
                P.axes[0].errorbar(
                    x, z_val, np.vstack((-z_em, z_ep)),
                    fmt="o", color=col, markersize=5, capsize=5,
                    label=reg.capitalize() if idx == 1 else None  # 凡例は1回だけ
                )

        reg = 'center'
        col = colors[reg]
        datas = np.array([])
        for idx in [1, 3, 5, 7]:
            x = idx                                   # x 軸位置
            z_val, z_ep, z_em = zdata[reg][idx]
            P.axes[1].errorbar(
                x, z_val, np.vstack((-z_em, z_ep)),
                fmt="o", color=col, markersize=5, capsize=5,
                label=reg.capitalize() if idx == 1 else None  # 凡例は1回だけ
            )
            datas = np.append(datas, z_val)
        datas_max = np.max(datas)
        datas_min = np.min(datas)
        datas_avg = np.average(datas)
        delz = datas_max - datas_min
        delz_e = delz * 6e3
        print(f'delz center = {delz}')
        print(f'delz center = {delz_e} eV')
        P.axes[1].axhline(datas_avg, color='k', linestyle='--', linewidth=1)
        #P.axes[1].text(1.5, datas_avg, f'{datas_avg:.4e}', fontsize=10)



        reg = 'outer'
        col = colors[reg]
        datas = np.array([])
        for idx in [2, 4, 6, 8]:
            x = idx                                   # x 軸位置
            z_val, z_ep, z_em = zdata[reg][idx]
            P.axes[2].errorbar(
                x, z_val, np.vstack((-z_em, z_ep)),
                fmt="o", color=col, markersize=5, capsize=5,
                label=reg.capitalize() if idx == 1 else None  # 凡例は1回だけ
            )
            datas = np.append(datas, z_val)
        datas_max = np.max(datas)
        datas_min = np.min(datas)
        datas_avg = np.average(datas)
        delz = datas_max - datas_min
        delz_e = delz * 6e3
        print(f'delz center = {delz}')
        print(f'delz center = {delz_e} eV')
        P.axes[2].axhline(datas_avg, color='k', linestyle='--', linewidth=1)
        #P.axes[2].text(1.5, datas_avg, f'{datas_avg:.4e}', fontsize=10)

        for i in range(0,4):
            P.axes[i].set_xticks(
            range(1, 9),
            [
                "Open\ncenter", "Open\nouter",
                "OBF\ncenter",  "OBF\nouter",
                "ND\ncenter",   "ND\nouter",
                "Be\ncenter",   "Be\nouter"
            ]
            )
        P.axes[0].set_ylabel("Redshift")
        P.axes[0].legend()
        plt.show()
        P.fig.savefig("figure/redshift.png", dpi=300)

        # 必要なら zdata を返すと後続処理でも使い回せる
        return zdata

    def result_plot_pixel(self):
        import matplotlib.pyplot as plt
        P = PlotManager()  # P.axes[0] が使える前提

        datas = {}
        pix_map = {0: "00", 1: "17", 2: "18", 3: "35"}

        with h5py.File(self.savefile, "r") as f:
            for e, hdfkey in enumerate(f.keys()):
                pix = pix_map.get(e)
                if pix is None:
                    continue
                red = f[hdfkey]["fitting_result"]["2/bvapec"]["Redshift"]["value"][()]
                red_ep = f[hdfkey]["fitting_result"]["2/bvapec"]["Redshift"]["ep"][()]
                red_em = f[hdfkey]["fitting_result"]["2/bvapec"]["Redshift"]["em"][()]
                err = np.vstack((-red_em, red_ep))

                datas[f"pixel{pix}"] = {
                    "redshift": {
                        "value": red,
                        "err": err
                    }
                }

        # MnKaの補正ずれデータ読み込み
        E = 6e3
        # with h5py.File('/Users/keitatanaka/Dropbox/SSD_backup/PKS_XRISM/repro_analysis/mxs/nominal/55Fe_cl_data/pixel_by_pixel/pixel_by_pixel_MnKa.hdf5', "r") as f:
        #     for pixel in f.keys():
        #         if pixel not in datas:
        #             continue
        #         shift = f[f'{pixel}/fitting_result']['1/zashift']['Redshift']['value'][()]
        #         shift_ep = f[f'{pixel}/fitting_result']['1/zashift']['Redshift']['ep'][()]
        #         shift_em = f[f'{pixel}/fitting_result']['1/zashift']['Redshift']['em'][()]
        #         datas[pixel]["cal_shift"] = {
        #             "value": shift,
        #             "err": np.vstack((-shift_em, shift_ep))
        #         }
        datas["pixel00"]["cal_shift"] = {"value": 0.9,"err": np.vstack((0.1,0.1))}
        datas["pixel17"]["cal_shift"] = {"value": 0.7,"err": np.vstack((0.1,0.1))}
        datas["pixel18"]["cal_shift"] = {"value": 0.6,"err": np.vstack((0.1,0.1))}
        datas["pixel35"]["cal_shift"] = {"value": 0.75,"err": np.vstack((0.1,0.1))}
        # プロット
        ax = P.axes[0]
        z_a = 0.1030
        for pixel, d in datas.items():
            if "cal_shift" not in d or "redshift" not in d:
                continue  # 必須データがなければ skip
            x = d["cal_shift"]["value"] 
            y = (d["redshift"]["value"])
            xerr = d["cal_shift"]["err"]
            yerr = d["redshift"]["err"]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=5,label=pixel)

        ax.set_xlabel("Calibration shift [eV]")
        ax.set_ylabel("Fitted redshift energy [eV]")
        ax.legend()
        plt.show()

    def result_plot_ssm(self):
        """
        Plot each FW ssm result
        Parameters
        ----------
        group_name : str
            HDF5 内の最上位グループ名（例: 'pks0745' など）
        """
        P = PlotManager((2,4),(12.8,8))
        # ------------ 1. データ読み込み＆辞書化 ------------
        zdata = {reg: {} for reg in ('center', 'outer')}  # {reg: {idx: (val, +err, -err)}}
        with h5py.File(self.savefile, "r") as f:
            for ee,group_name in enumerate(["ssm_1000_wz_rebin_1_dzgaus","ssm_2000_wz_rebin_1_dzgaus","ssm_4000_wz_rebin_1_dzgaus","ssm_wz_rebin_1_dzgaus"]):
                if group_name not in f:
                    raise KeyError(f"'{group_name}' not found in {self.savefile}")
    # kT, v, shift, Fe, norm, ratio, w_width, z_width
                for reg in ('center', 'outer'):
                    for num,target in enumerate(["2/bvapec/kT", "2/bvapec/Velocity", "2/bvapec/Fe", "2/bvapec/norm", "3/dzgaus/ratio", "3/dzgaus/Sigma_w", "3/dzgaus/Sigma_z","2/bvapec/Redshift", "2/bvapec/Redshift_2"]):
                        base   = f"{group_name}/fitting_result/{reg}/{target}"
                        z_val  = f[f"{base}/value"][()]
                        z_ep   = f[f"{base}/ep"][()]
                        z_em   = f[f"{base}/em"][()]
                        if "Sigma" in target:
                            z_val *= 1e3
                            z_ep  *= 1e3
                            z_em  *= 1e3
                        print(z_em, z_ep)
                        if reg == "center":
                            col = "black"
                        else:
                            col = "red"
                        if num == 8:
                            num = 7
                        if z_em < 0 and z_ep > 0:
                            P.axes[num].errorbar(ee, z_val,  np.vstack((-z_em, z_ep)),fmt='o', capsize=5, color=col)
                        elif z_em > 0 and z_ep > 0:
                            P.axes[num].errorbar(ee, z_val,  np.vstack((z_em, z_ep)),fmt='o', capsize=5, color=col)
                        elif z_ep < 0:
                            P.axes[num].errorbar(ee, z_val,  np.vstack((-z_em, -z_ep)),fmt='o', capsize=5, color=col)

        P.axes[0].set_ylabel("kT")
        P.axes[1].set_ylabel("Turbulent Velocity")
        P.axes[2].set_ylabel("Fe")
        P.axes[3].set_ylabel("apec norm")
        P.axes[4].set_ylabel("w/z ratio")
        P.axes[5].set_ylabel("Sigma w")
        P.axes[6].set_ylabel("Sigma z")
        P.axes[7].set_ylabel("Redshift")
        for i in range(0,8):
            P.axes[i].set_xticks(
            range(0, 4),
            [
                "Open", 
                "OBF", 
                "Be", 
                "Simult"
            ]
            )
        P.fig.tight_layout()
        plt.show()
        P.fig.savefig("figure/ssm_res.png", dpi=300)

        # 必要なら zdata を返すと後続処理でも使い回せる
        return zdata

    def solve_red_gain(zdata, anchor='gain_mean_zero'):
        """
        Parameters
        ----------
        zdata : dict
            result_plot_ssm_z() の戻り値
            zdata['center'][idx] = (val, +err, -err)
            zdata['outer' ][idx] = (val, +err, -err)

        anchor : {'gain_mean_zero', ('red_cen', value)}
            - 'gain_mean_zero' : 全ゲインの平均 = 0 を仮定（デフォルト）
            - ('red_cen', value): red_cen を外部値 value に固定

        Returns
        -------
        param, cov : (ndarray, ndarray)
            param = [red_cen, red_out,
                    gain_cen1, gain_out1,
                    gain_cen2, gain_out2, …, gain_out4]
            cov   = 10×10 共分散行列（1σ = √diag(cov)）
        """
        # ---------- 1. 16 本の観測値ベクトル ----------
        y, sigma = [], []
        pairs = [(1,2), (3,4), (5,6), (7,8)]
        for p, (i, j) in enumerate(pairs, 1):
            # 奇数 idx = gain_cen_p, 偶数 idx = gain_out_p
            for idx, gain_type in zip((i, i, j, j),
                                    ('cen', 'cen', 'out', 'out')):
                reg = 'center' if len(y)%2==0 else 'outer'  # C,O,C,O,...
                val, ep, em = zdata[reg][idx]
                y.append(val)
                sigma.append(0.5*(ep+em))
        y      = np.array(y)
        W_inv  = np.diag(sigma**2)          # 観測共分散（対角近似）

        # ---------- 2. 設計行列 (16×10) ----------
        A = np.zeros((16, 10))
        # 列 0,1 = red_cen, red_out
        A[0::4, 0] = 1          # center rows
        A[1::4, 1] = 1          # outer  rows
        A[2::4, 0] = 1          # center rows (idx even)
        A[3::4, 1] = 1          # outer  rows (idx even)
        # 列 2..9 = gain
        for p in range(4):
            A[4*p+0:4*p+2, 2+2*p]   = 1     # gain_cen_p
            A[4*p+2:4*p+4, 3+2*p]   = 1     # gain_out_p

        # ---------- 3. アンカー条件を追加 ----------
        if anchor == 'gain_mean_zero':
            # 1×10 行を下に足す
            A  = np.vstack([A, np.r_[0,0, np.ones(8)]/8])
            y  = np.append(y, 0.0)
            W_inv = np.pad(W_inv, ((0,1),(0,1)), constant_values=0)  # 重み小→強制
            W_inv[-1, -1] = 1e-12
        elif isinstance(anchor, tuple) and anchor[0]=='red_cen':
            A  = np.vstack([A, np.r_[1, np.zeros(9)]])
            y  = np.append(y, anchor[1])
            W_inv = np.pad(W_inv, ((0,1),(0,1)), constant_values=0)
            W_inv[-1, -1] = 1e-12
        else:
            raise ValueError("anchor must be 'gain_mean_zero' or ('red_cen', value)")

        # ---------- 4. 加重最小二乗解 ----------
        # (Aᵀ W⁻¹ A) x = Aᵀ W⁻¹ y
        WT = np.linalg.inv(W_inv)
        AtW = A.T @ WT
        cov = np.linalg.inv(AtW @ A)
        param = cov @ (AtW @ y)

        return param, cov

    def test_ssm(self):
        zdata = self.result_plot_ssm_z('ssm_w_rebin_1_zgauss')
        fig, ax = plt.subplots(figsize=(6, 4))

        x        = np.arange(1, 9)
        delta_z  = []
        err_lo   = []
        err_hi   = []

        for i in x:
            # 1σ誤差は ± の平均を絶対値で受け取る
            zc, ep_c, em_c = zdata['center'][i]
            zo, ep_o, em_o = zdata['outer' ][i]

            dz   = zc - zo
            # ▼ ここで abs() を取るのがポイント
            elo  = abs(em_c) + abs(ep_o)   # 下側誤差
            ehi  = abs(ep_c) + abs(em_o)   # 上側誤差

            delta_z.append(dz)
            err_lo.append(elo)
            err_hi.append(ehi)

        # errorbar の yerr は shape = (2, N) の “正の” 配列
        yerr = np.vstack([err_lo, err_hi])

        ax.errorbar(
            x, delta_z, yerr=yerr,
            fmt='o', capsize=5, color='k'
        )

        ax.set_xticks(x)
        ax.set_xlabel('Configuration index (1–8)')
        ax.set_ylabel(r'$\Delta z = z_{\mathrm{center}} - z_{\mathrm{outer}}$')
        ax.set_title('Center–Outer Redshift Difference (1σ Errors)')
        plt.tight_layout()
        plt.show()

    def plasma_diagnostic(self, modname="S_all_center_All_rebin_1_zgaus"):
        with h5py.File(self.savefile, 'r') as f:
            z = f[modname]["fitting_result"]["3/zgauss"]["norm"]["value"][...]
            z_ep = f[modname]["fitting_result"]["3/zgauss"]["norm"]["ep"][...]
            z_em = f[modname]["fitting_result"]["3/zgauss"]["norm"]["em"][...]
            w = f[modname]["fitting_result"]["4/zgauss_4"]["norm"]["value"][...]
            w_ep = f[modname]["fitting_result"]["4/zgauss_4"]["norm"]["ep"][...]
            w_em = f[modname]["fitting_result"]["4/zgauss_4"]["norm"]["em"][...]
            x = f[modname]["fitting_result"]["5/zgauss_5"]["norm"]["value"][...]
            x_ep = f[modname]["fitting_result"]["5/zgauss_5"]["norm"]["ep"][...]
            x_em = f[modname]["fitting_result"]["5/zgauss_5"]["norm"]["em"][...]
            y = f[modname]["fitting_result"]["6/zgauss_6"]["norm"]["value"][...]
            y_ep = f[modname]["fitting_result"]["6/zgauss_6"]["norm"]["ep"][...]
            y_em = f[modname]["fitting_result"]["6/zgauss_6"]["norm"]["em"][...]
            Lya2 = f[modname]["fitting_result"]["7/zgauss_7"]["norm"]["value"][...]
            Lya2_ep = f[modname]["fitting_result"]["7/zgauss_7"]["norm"]["ep"][...]
            Lya2_em = f[modname]["fitting_result"]["7/zgauss_7"]["norm"]["em"][...]
            Lya1 = f[modname]["fitting_result"]["8/zgauss_8"]["norm"]["value"][...]
            Lya1_ep = f[modname]["fitting_result"]["8/zgauss_8"]["norm"]["ep"][...]
            Lya1_em = f[modname]["fitting_result"]["8/zgauss_8"]["norm"]["em"][...]
            Heb2 = f[modname]["fitting_result"]["9/zgauss_9"]["norm"]["value"][...]
            Heb2_ep = f[modname]["fitting_result"]["9/zgauss_9"]["norm"]["ep"][...]
            Heb2_em = f[modname]["fitting_result"]["9/zgauss_9"]["norm"]["em"][...]
            Heb1 = f[modname]["fitting_result"]["10/zgauss_10"]["norm"]["value"][...]
            Heb1_ep = f[modname]["fitting_result"]["10/zgauss_10"]["norm"]["ep"][...]
            Heb1_em = f[modname]["fitting_result"]["10/zgauss_10"]["norm"]["em"][...]

        print("z:", z, z_ep, z_em)
        print("w:", w, w_ep, w_em)
        print("x:", x, x_ep, x_em)
        print("y:", y, y_ep, y_em)
        print("Lya2:", Lya2, Lya2_ep, Lya2_em)
        print("Lya1:", Lya1, Lya1_ep, Lya1_em)
        print("Heb2:", Heb2, Heb2_ep, Heb2_em)
        print("Heb1:", Heb1, Heb1_ep, Heb1_em)

        

def fake_test_multi():    
    # X = XspecFit('fake_ratio_0ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # X._fake_test(resp='1000_center_L_without_Lp.rmf', arf='1000_center_image_1p8_8keV_1e7.arf', exposure=44.13e3,  rebin=1, model_xcm='fakeit_model_ratio.xcm', num=30, savedir = f'./fakeit_result_ratio_0ksec')
    X = XspecFit('fake_ratio_100ksec_sim.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    X._fake_test(resp='1000_center_L_without_Lp.rmf', arf='1000_center_image_1p8_8keV_1e7.arf', exposure=100.0e3,  rebin=1, model_xcm='fakeit_model_zgaus.xcm', num=30, savedir = f'./fakeit_result_ratio_100ksec_sim')
    # X = XspecFit('fake_ratio_200ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # X._fake_test(resp='1000_center_L_without_Lp.rmf', arf='1000_center_image_1p8_8keV_1e7.arf', exposure=244.13e3,  rebin=1, model_xcm='fakeit_model_ratio.xcm', num=30, savedir = f'./fakeit_result_ratio_200ksec')
    # X = XspecFit('fake_ratio_300ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # X._fake_test(resp='1000_center_L_without_Lp.rmf', arf='1000_center_image_1p8_8keV_1e7.arf', exposure=344.13e3,  rebin=1, model_xcm='fakeit_model_ratio.xcm', num=30, savedir = f'./fakeit_result_ratio_300ksec')

def run_fake_multi():
    # X = XspecFit('fake_ratio_0ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # min_key, fake_num = X.search_most_match_number(value=2.5)
    # X.savefile = f'fake_ratio_result.hdf5'
    # X.run_fake(spec=f'fakeit_result_ratio_0ksec/fakeit_spec_{fake_num}_b1.pi', model_xcm='fakeit_model_ratio.xcm', exposure=0)
    X = XspecFit('fake_ratio_100ksec_sim.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    result = X.search_most_match_number(value=2.5)
    fake_num = result[0][1]
    #fake_num = result[1][1]
    #fake_num = 22
    #print(result)
    # 100 ksec, num = 22 is best
    X.savefile = f'fake_ratio_result_sim.hdf5'
    X.run_fake(spec=f'fakeit_result_ratio_100ksec_sim/fakeit_spec_{fake_num}_b1.pi', model_xcm='fakeit_model_ratio.xcm', exposure=100)
    # X = XspecFit('fake_ratio_200ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # min_key, fake_num = X.search_most_match_number(value=2.5)
    # X.savefile = f'fake_ratio_result.hdf5'
    # X.run_fake(spec=f'fakeit_result_ratio_200ksec/fakeit_spec_{fake_num}_b1.pi', model_xcm='fakeit_model_ratio.xcm', exposure=200)
    # X = XspecFit('fake_ratio_300ksec.hdf5',atomdb_version='3.0.9',abundance_table='lpgs')
    # min_key, fake_num = X.search_most_match_number(value=2.5)
    # X.savefile = f'fake_ratio_result.hdf5'
    # X.run_fake(spec=f'fakeit_result_ratio_300ksec/fakeit_spec_{fake_num}_b1.pi', model_xcm='fakeit_model_ratio.xcm', exposure=300)

class Calc:

    def __init__(self):
        pass

    def sigma2v(self, sigma, E):
        """
        Convert sigma to velocity dispersion
        """
        sigma = sigma * u.eV
        E = E * u.eV
        return (sigma * c.c / E).to(u.km / u.s)

class OutPut:
    def __init__(self):
        self.X = XspecFit()