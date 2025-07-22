import numpy as np
from astropy import constants as const
from astropy.io  import fits
import re
import h5py
import pandas as pd
import csv
import sys
import os

class UnitConverter:

    def __init__(self,pixel_number=36,FOV_manual=0):
        self.FOV_suzaku = 2 * np.pi * (1 - np.cos(17.8 / 60 * np.pi / 180))
        self.arcmin2deg = 1/60
        self.str2deg2 = 3282.80635
        self.deg22str = 1 / self.str2deg2
        # self.FOV_halosat = np.pi * (5 * np.pi / 180)**2 # 10deg diameter full respomse.
        self.FOV_halosat = 0.035 # 10deg diameter full respomse, tapering 14deg.
        self.FOV_xrism = 2 * np.pi * (1 - np.cos(17.8 / 60 * np.pi / 180)) * 3
        self.FOV_resolve = 3 * 3  * self.arcmin2deg**2 / self.str2deg2 * pixel_number / 36
        self.FOV_manual = FOV_manual

    def FOV_selecter(self,satellite:str):
        if satellite == "suzaku":
            self.FOV = self.FOV_suzaku   #[str]
        elif satellite == "halosat":
            self.FOV = self.FOV_halosat  #[str]
        elif satellite == "xrism":
            self.FOV = self.FOV_xrism
        elif satellite == "resolve":
            self.FOV = self.FOV_resolve
        elif satellite == "manual":
            self.FOV = self.FOV_manual
        elif satellite == "xrism_4pix":
            self.FOV = 8.461594994110499e-08
        print(f'{satellite} FOV = {self.FOV} str')

    def ApecNorm2EM(self,norm:float,satellite:str):
        """
        Apec normalization to emission measure.
        Return    : emission measure [cm-6 pc]
        norm      : apec normalization [cm-5]
        satellite : suzaku or halosat
        """
        self.FOV_selecter(satellite)
        return norm * 1e+14 * 4*np.pi  / (self.FOV * const.pc.cgs.value) # [cm-6 pc]

    def EM2ApecNorm(self,EM:float,satellite:str):
        """
        Emission measure to apec normalization.
        Return    : apec normalization [cm-5]
        EM        : emission measure [cm-6 pc]
        satellite : suzaku or halosat
        """
        self.FOV_selecter(satellite)
        return EM * 1e-14 * self.FOV * const.pc.cgs.value / (4*np.pi) # [cm-5 FOV-1]

    def PowNorm2SB(self,norm:float,satellite:str):
        """
        Powerlaw normalization to Surface Brightness.
        Return : Surface Brightness [photons/keV/cm-2/s/str]
        norm   : Powerlaw normalization [photons/keV/cm-2/s]
        satellite : suzaku or halosat
        """
        self.FOV_selecter(satellite)
        return norm / self.FOV # [photons/keV/cm-2/s/str]

    def SB2PowNorm(self,SB:float,satellite:str):
        """
        Powerlaw normalization to Surface Brightness.
        Return : Surface Brightness [photons/keV/cm-2/s/str]
        norm   : Powerlaw normalization [photons/keV/cm-2/s/str]
        satellite : suzaku or halosat
        """
        self.FOV_selecter(satellite)
        return SB * self.FOV # [photons/keV/cm-2/s]

    def calc_tapering_omega(self,full_rad:float,tap_rad:float):
        """_summary_

        Args:
            full_rad (float): _description_
            tap_rad (float): _description_

        Returns:
            _type_: _description_
        """
        pass

    def z2dist(self,z,d):
        from scipy.integrate import quad

        # 定数の設定
        c = 3.0e5  # 光速 (km/s)
        H0 = 70.0  # ハッブル定数 (km/s/Mpc)
        Omega_m = 0.3
        Omega_lambda = 0.7

        # 積分の関数
        def integrand(z):
            return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)

        # 積分の計算
        z = z
        DC, _ = quad(integrand, 0, z)

        # 共動距離の計算
        DC = (c / H0) * DC

        # 光学的距離の計算
        DL = (1 + z) * DC * 1e+6 # pc


        theta_rad = np.arcsin(d / DL)
        theta_arcmin = theta_rad * 3437.75
        print(theta_rad)

        print(f'z = {z}, DL = {DL*1e-6} Mpc, ', f'theta = {theta_arcmin} arcmin')



class LoadLog:

    def __init__(self,filename:str):
        self.filename = filename
        self.numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        self.rx = re.compile(self.numeric_const_pattern, re.VERBOSE)
        self.idx_range = 100
        print('-------------------------')
        print(f'Log filename = {self.filename}')

    def keysearch(self,keyword:str):
        """
        Output specific word column from logfile.
        keyword : keyword
        """
        for sline in self.target_list:
            if keyword in sline:
                print("--------------------")
                print(f'Searching {keyword}')
                self.target = sline
                print(self.target)

    def keysearch_idx(self,keyword:str):
        """
        Output specific word column from logfile.
        keyword : keyword
        """
        f = open(self.filename, mode='r')
        data = f.readlines()
        for e,sline in enumerate(data):
            if keyword in sline:
                print("--------------------")
                print(f'Searching {keyword}')
                self.target_idx = e
                print(e)
                print(sline)
        self.target_list = data[self.target_idx:self.target_idx+self.idx_range]

    def cut_float(self):
        self.float_words = self.rx.findall(self.target)
        print(self.float_words)

    def wordcut(self,start:int,end:int):
        print(self.target[start:end])
        self.value = float(self.target[start:end])

    def fileout(self):
        filename = "nh_result.txt"
        np.savetxt(filename,np.array([self.value/1e+22]))

class FitResultOut(LoadLog):

    def __init__(self,filename:str):
        super().__init__(filename)
        self.UC = UnitConverter()
        home    = os.environ['HOME']
        self.hdf5 = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_Halosat_result.hdf5'
        self.hdf5_inf = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_Halosat_information.hdf5'

    def result_out(self,base_keyword:str,keywords:list,**kwargs):
        self.keysearch_idx(base_keyword)
        for e,keyword in enumerate(keywords):
            self.keysearch(keyword)
            self.cut_float()
            print(e)
            if len(self.float_words) <= 4:
                self.float_words = np.hstack((self.float_words,np.zeros(5-len(self.float_words))))
            if e == 0:
                self.float_array = self.float_words
            else:
                self.float_array = np.vstack((self.float_array,self.float_words))

    def extract_fit(self):
        for e,target in enumerate(self.float_array):
            if e == 0:
                self.fit = float(target[2])
            else:
                self.fit = np.vstack((self.fit,float(target[2])))

    def extract_fix_nh(self):
        self.fix_nh = np.array([float(self.float_array[4]),float(self.float_array[4]),float(self.float_array[4])])

    def extract_fix_kT(self):
        self.fix_kT = np.array([float(self.float_array[2]),float(self.float_array[2]),float(self.float_array[2])])

    def extract_bgd(self):
        for e,target in enumerate(self.float_array):
            if e == 0:
                res = float(target[2])
            else:
                res = np.vstack((res,float(target[2])))
        return res

    def extract_error(self):
        for e,target in enumerate(self.float_array):
            if e == 0:
                self.error = np.array([float(target[1]),float(target[2])])
            else:
                self.error = np.vstack((self.error,np.array([float(target[1]),float(target[2])])))

    def extract_error_bgd(self):
        for e,target in enumerate(self.float_array):
            if e == 0:
                error = np.array([float(target[1]),float(target[2])])
            else:
                error = np.vstack((error,np.array([float(target[1]),float(target[2])])))
        return error

    # def extract_BGD(self):
    #     for e,target in enumerate(self.float_array):
    #         if e == 0:
    #             self.BGD = np.array([float(target[2]),float(target[2]),float(target[2])])
    #         else:
    #             self.BGD = np.vstack((self.BGD,np.array([float(target[1]),float(target[2])])))

    def stack_fit_error(self):
        self.result = np.hstack((self.fit,self.error))
        self.result = np.vstack((self.result,self.fix_nh))

    def stack_bgd_v1(self,low,high,high_err):
        bgd = np.hstack((high,high_err))
        low_bgd = np.hstack((low,low))
        low_bgd = np.hstack((low_bgd,low))
        bgd = np.vstack((bgd,low_bgd))
        self.bgd = bgd
        self.bgd[:,1] -= self.bgd[:,0]
        self.bgd[:,2] -= self.bgd[:,0] 
        print('---------------------')
        print(self.bgd)

    def stack_bgd(self,low,high):
        high_bgd = np.hstack((high,high))
        high_bgd = np.hstack((high_bgd,high))
        low_bgd = np.hstack((low,low))
        low_bgd = np.hstack((low_bgd,low))
        bgd = np.vstack((high_bgd,low_bgd))
        self.bgd = bgd
        self.bgd[:,1] -= self.bgd[:,0]
        self.bgd[:,2] -= self.bgd[:,0] 
        print('---------------------')
        print(self.bgd)


    def OES_result_out_v1(self,obsid:str):
        """_summary_
        Used until 2022.02.20
        Free params
        CXB norm, MWH EM, LHB norm, Hot kt, EM
        Args:
            obsid (str): halosat observation ID
        """
        self.result_out('#Model (phabs',['#   3    2   powerlaw   norm','#   7    3','#  11    4','#  12    5','#  28    5'])
        self.extract_fit()
        self.result_out('#   1    1   phabs',['#   1    1   phabs'])
        self.extract_fix_nh()
        self.result_out('#Model (phabs',['#   4    3'])
        self.extract_fix_kT()
        self.result_out('#Total fit statistic',['#Total fit statistic'])
        print(self.float_array)
        chi_sq = float(self.float_array[0])
        dof    = float(self.float_array[1])
        self.filename = 'error.log'
        self.result_out('Confidence',['#     3','#     7','#    11','#    12','#    28'])
        self.extract_error()
        #self.result_out('#Model back',['#   1    1','#   2    1','#   3    1','#   4    1','#   5    1','#   6    1'])
        self.stack_fit_error()
        print(self.result)
        self.result[:,1] -= self.result[:,0]
        self.result[:,2] -= self.result[:,0] 
        CXB_SB = self.UC.PowNorm2SB(self.result[0],'halosat')
        MWH_EM = self.UC.ApecNorm2EM(self.result[1],'halosat')
        LHB_EM = self.UC.ApecNorm2EM(self.result[2],'halosat')
        HTC_kT = self.result[3]
        HTC_EM = self.UC.ApecNorm2EM(self.result[4],'halosat')
        nh     = self.result[5]
        MWH_kT = self.result[6]
        data = {}
        data['nh']     = nh
        data['CXB_SB'] = CXB_SB
        data['MWH_kT'] = MWH_kT
        data['MWH_EM'] = MWH_EM
        data['LHB_EM'] = LHB_EM
        data['HTC_EM'] = HTC_EM
        data['HTC_kT'] = HTC_kT
        data['chi_sq'] = chi_sq
        data['dof']    = dof
        print(data)
        with h5py.File(self.hdf5,"a") as f:
            if obsid in f.keys():
                del f[obsid]
            f.create_dataset(f'{obsid}/CXB_SB',data=CXB_SB)
            f.create_dataset(f'{obsid}/MWH_kT',data=MWH_kT)
            f.create_dataset(f'{obsid}/MWH_EM',data=MWH_EM)
            f.create_dataset(f'{obsid}/LHB_EM',data=LHB_EM)
            f.create_dataset(f'{obsid}/HTC_EM',data=HTC_EM)
            f.create_dataset(f'{obsid}/HTC_kT',data=HTC_kT)
            f.create_dataset(f'{obsid}/nh',data=nh)
            f.create_dataset(f'{obsid}/chi_sq',data=chi_sq)
            f.create_dataset(f'{obsid}/dof',data=dof)

    def OES_result_out_v2(self,obsid:str):
        """_summary_
        Use from 2022.02.21
        Free params
        MWH kT, EM, Hot kt, EM
        Args:
            obsid (str): halosat observation ID
        """
        self.result_out('#Model (phabs',['#   4    3','#   7    3','#  12    5','#  15    5'])
        self.extract_fit()
        self.result_out('#   1    1   phabs',['#   1    1   phabs'])
        self.extract_fix_nh()
        self.result_out('#Model bg:powerlaw<1>',['#   2    1','#   6    1','#  10    1'])
        res_high = self.extract_bgd()
        self.result_out('#Model bg:powerlaw<1>',['#   4    2','#   8    2','#  12    2'])
        res_low = self.extract_bgd()
        self.result_out('#Total fit statistic',['#Total fit statistic'])
        print(self.float_array)
        chi_sq = float(self.float_array[0])
        dof    = float(self.float_array[1])
        self.filename = 'error.log'
        self.result_out('Confidence',['#     4','#     7','#    12','#    15'])
        self.extract_error()
        #self.result_out('#Model back',['#   1    1','#   2    1','#   3    1','#   4    1','#   5    1','#   6    1'])
        self.result_out('Confidence',['#     2','#     6','#     8'])
        error_bgd = self.extract_error_bgd()
        self.stack_bgd(res_low,res_high,error_bgd)
        self.stack_fit_error()
        print(self.result)
        self.result[:,1] -= self.result[:,0]
        self.result[:,2] -= self.result[:,0] 
        MWH_kT = self.result[0]
        MWH_EM = self.UC.ApecNorm2EM(self.result[1],'halosat')
        HTC_kT = self.result[2]
        HTC_EM = self.UC.ApecNorm2EM(self.result[3],'halosat')
        nh     = self.result[4]
        bg_h_s14 = self.bgd[0]
        bg_h_s38 = self.bgd[1]
        bg_h_s54 = self.bgd[2]
        bg_l_s14 = self.bgd[3]
        bg_l_s38 = self.bgd[4]
        bg_l_s54 = self.bgd[5]
        data = {}
        data['nh']          = nh
        data['MWH_kT']      = MWH_kT
        data['MWH_EM']      = MWH_EM
        data['HTC_EM']      = HTC_EM
        data['HTC_kT']      = HTC_kT
        data['chi_sq']      = chi_sq
        data['dof']         = dof
        data['bg_s14_high'] = bg_h_s14
        data['bg_s38_high'] = bg_h_s38
        data['bg_s54_high'] = bg_h_s54
        data['bg_s14_low']  = bg_l_s14
        data['bg_s38_low']  = bg_l_s38
        data['bg_s54_low']  = bg_l_s54
        print(data)
        with h5py.File(self.hdf5,"a") as f:
            if obsid in f.keys():
                del f[obsid]
            f.create_dataset(f'{obsid}/MWH_kT',data=MWH_kT)
            f.create_dataset(f'{obsid}/MWH_EM',data=MWH_EM)
            f.create_dataset(f'{obsid}/HTC_EM',data=HTC_EM)
            f.create_dataset(f'{obsid}/HTC_kT',data=HTC_kT)
            f.create_dataset(f'{obsid}/nh',data=nh)
            f.create_dataset(f'{obsid}/chi_sq',data=chi_sq)
            f.create_dataset(f'{obsid}/dof',data=dof)
            f.create_dataset(f'{obsid}/bg_s14_low',data=bg_l_s14)
            f.create_dataset(f'{obsid}/bg_s38_low',data=bg_l_s38)
            f.create_dataset(f'{obsid}/bg_s54_low',data=bg_l_s54)
            f.create_dataset(f'{obsid}/bg_s14_high',data=bg_h_s14)
            f.create_dataset(f'{obsid}/bg_s38_high',data=bg_h_s38)
            f.create_dataset(f'{obsid}/bg_s54_high',data=bg_h_s54)

    def OES_result_out(self,obsid:str):
        """_summary_
        Use from 2023.03.22
        Free params
        Warm kT, EM, Hot kt, EM
        Args:
            obsid (str): halosat observation ID
        """
        self.result_out('#Model (TBabs<1>',['#   3    2','#   9    5','#  12    5','#  13    6','#  16    6'])
        self.extract_fit()
        self.result_out('#   1    1   TBabs',['#   1    1   TBabs'])
        self.extract_fix_nh()
        self.result_out('#Model bg:powerlaw<1>',['#   2    1','#   6    1','#  10    1'])
        res_high = self.extract_bgd()
        self.result_out('#Model bg:powerlaw<1>',['#   4    2','#   8    2','#  12    2'])
        res_low = self.extract_bgd()
        self.result_out('#Total fit statistic',['#Total fit statistic'])
        print(self.float_array)
        chi_sq = float(self.float_array[0])
        dof    = float(self.float_array[1])
        self.filename = 'error.log'
        print('test')
        self.result_out('AAAAAAAA',['0000'])
        print(self.float_array)
        self.result_out('Confidence',['#     3','#     9','#    12','#    13','#    16'])
        self.result_out('***Warning: New best fit found',['#     3','#     9','#    12','#    13','#    16'])
        self.extract_error()
        #self.result_out('#Model back',['#   1    1','#   2    1','#   3    1','#   4    1','#   5    1','#   6    1'])
        #self.result_out('Confidence',['#     2','#     6','#     8'])
        # error_bgd = self.extract_error_bgd()
        self.stack_bgd(res_low,res_high)
        self.stack_fit_error()
        print(self.result)
        self.result[:,1] -= self.result[:,0]
        self.result[:,2] -= self.result[:,0] 
        Warm_kT = self.result[1]
        Warm_EM = self.UC.ApecNorm2EM(self.result[2],'halosat')
        Hot_kT = self.result[3]
        Hot_EM = self.UC.ApecNorm2EM(self.result[4],'halosat')
        CXB_SB = self.UC.PowNorm2SB(self.result[0],'halosat')
        nh     = self.result[5]
        bg_h_s14 = self.bgd[0]
        bg_h_s38 = self.bgd[1]
        bg_h_s54 = self.bgd[2]
        bg_l_s14 = self.bgd[3]
        bg_l_s38 = self.bgd[4]
        bg_l_s54 = self.bgd[5]
        data = {}
        data['nh']          = nh
        data['CXB_SB']      = CXB_SB
        data['Warm_kT']     = Warm_kT
        data['Warm_EM']     = Warm_EM
        data['Hot_EM']      = Hot_EM
        data['Hot_kT']      = Hot_kT
        data['chi_sq']      = chi_sq
        data['dof']         = dof
        data['bg_s14_high'] = bg_h_s14
        data['bg_s38_high'] = bg_h_s38
        data['bg_s54_high'] = bg_h_s54
        data['bg_s14_low']  = bg_l_s14
        data['bg_s38_low']  = bg_l_s38
        data['bg_s54_low']  = bg_l_s54
        print(data)
        with h5py.File(self.hdf5,"a") as f:
            if obsid in f.keys():
                del f[obsid]
            f.create_dataset(f'{obsid}/CXB_SB',data=CXB_SB)
            f.create_dataset(f'{obsid}/Warm_kT',data=Warm_kT)
            f.create_dataset(f'{obsid}/Warm_EM',data=Warm_EM)
            f.create_dataset(f'{obsid}/Hot_EM',data=Hot_EM)
            f.create_dataset(f'{obsid}/Hot_kT',data=Hot_kT)
            f.create_dataset(f'{obsid}/nh',data=nh)
            f.create_dataset(f'{obsid}/chi_sq',data=chi_sq)
            f.create_dataset(f'{obsid}/dof',data=dof)
            f.create_dataset(f'{obsid}/bg_s14_low',data=bg_l_s14)
            f.create_dataset(f'{obsid}/bg_s38_low',data=bg_l_s38)
            f.create_dataset(f'{obsid}/bg_s54_low',data=bg_l_s54)
            f.create_dataset(f'{obsid}/bg_s14_high',data=bg_h_s14)
            f.create_dataset(f'{obsid}/bg_s38_high',data=bg_h_s38)
            f.create_dataset(f'{obsid}/bg_s54_high',data=bg_h_s54)

    def rate_out(self,obsid):
        self.idx_range = 30
        self.result_out('!XSPEC12> # 0.4-0.5keV count rates',['#Net count rate (cts/s) for Spectrum:1','#Net count rate (cts/s) for Spectrum:2','#Net count rate (cts/s) for Spectrum:3'])
        rate_low = np.array(self.float_array[:,1],dtype=float)
        self.result_out('!XSPEC12> # 0.5-0.7keV count rates',['#Net count rate (cts/s) for Spectrum:1','#Net count rate (cts/s) for Spectrum:2','#Net count rate (cts/s) for Spectrum:3'])
        rate_mid = np.array(self.float_array[:,1],dtype=float)
        self.result_out('!XSPEC12> # 0.7-1.0keV count rates',['#Net count rate (cts/s) for Spectrum:1','#Net count rate (cts/s) for Spectrum:2','#Net count rate (cts/s) for Spectrum:3'])
        rate_high = np.array(self.float_array[:,1],dtype=float)
        with h5py.File(self.hdf5_inf,"a") as f:
            if obsid in f.keys():
                if "rate" in f[obsid].keys():
                    del f[obsid]["rate"]
            f.create_dataset(f"{obsid}/rate/s14/rate_low",data=rate_low[0])
            f.create_dataset(f"{obsid}/rate/s14/rate_mid",data=rate_mid[0])
            f.create_dataset(f"{obsid}/rate/s14/rate_high",data=rate_high[0])
            f.create_dataset(f"{obsid}/rate/s38/rate_low",data=rate_low[1])
            f.create_dataset(f"{obsid}/rate/s38/rate_mid",data=rate_mid[1])
            f.create_dataset(f"{obsid}/rate/s38/rate_high",data=rate_high[1])
            f.create_dataset(f"{obsid}/rate/s54/rate_low",data=rate_low[2])
            f.create_dataset(f"{obsid}/rate/s54/rate_mid",data=rate_mid[2])
            f.create_dataset(f"{obsid}/rate/s54/rate_high",data=rate_high[2])


    def loadhdf5_v1(self,obsid_list:list):
        with h5py.File(self.hdf5,'r') as f:
            data = {
                'obsid':[obsid for obsid in obsid_list],
                'nh':[f[f'{obsid}']['nh'][0] for obsid in obsid_list],
                'CXB_SB':[f[f'{obsid}']['CXB_SB'][0] for obsid in obsid_list],
                'CXB_SB_ep':[f[f'{obsid}']['CXB_SB'][2] for obsid in obsid_list],
                'CXB_SB_em':[f[f'{obsid}']['CXB_SB'][1] for obsid in obsid_list],
                'MWH_kT':[f[f'{obsid}']['MWH_kT'][0] for obsid in obsid_list],
                'MWH_EM':[f[f'{obsid}']['MWH_EM'][0] for obsid in obsid_list],
                'MWH_EM_ep':[f[f'{obsid}']['MWH_EM'][2] for obsid in obsid_list],
                'MWH_EM_em':[f[f'{obsid}']['MWH_EM'][1] for obsid in obsid_list],
                'HTC_kT':[f[f'{obsid}']['HTC_kT'][0] for obsid in obsid_list],
                'HTC_EM':[f[f'{obsid}']['HTC_EM'][0] for obsid in obsid_list],
                'HTC_EM_ep':[f[f'{obsid}']['HTC_EM'][2] for obsid in obsid_list],
                'HTC_EM_em':[f[f'{obsid}']['HTC_EM'][1] for obsid in obsid_list],
                'chi_sq':[f[f'{obsid}']['chi_sq'][...] for obsid in obsid_list],
                'dof':[f[f'{obsid}']['dof'][...] for obsid in obsid_list]
                }
            df = pd.DataFrame(data)
            print(df)
            df.to_csv('OES_HS_result.csv')

    def loadhdf5(self,obsid_list:list,savecsv:str='OES_result.csv'):
        with h5py.File(self.hdf5,'r') as f:
            data = {
                'obsid':[obsid for obsid in obsid_list],
                'nh':[f[f'{obsid}']['nh'][0] for obsid in obsid_list],
                'CXB':[f[f'{obsid}']['CXB_SB'][0] for obsid in obsid_list],
                'CXB_ep':[f[f'{obsid}']['CXB_SB'][2] for obsid in obsid_list],
                'CXB_em':[f[f'{obsid}']['CXB_SB'][1] for obsid in obsid_list],
                'Warm_kT':[f[f'{obsid}']['Warm_kT'][0] for obsid in obsid_list],
                'Warm_kT_ep':[f[f'{obsid}']['Warm_kT'][2] for obsid in obsid_list],
                'Warm_kT_em':[f[f'{obsid}']['Warm_kT'][1] for obsid in obsid_list],
                'Warm_EM':[f[f'{obsid}']['Warm_EM'][0] for obsid in obsid_list],
                'Warm_EM_ep':[f[f'{obsid}']['Warm_EM'][2] for obsid in obsid_list],
                'Warm_EM_em':[f[f'{obsid}']['Warm_EM'][1] for obsid in obsid_list],
                'Hot_kT':[f[f'{obsid}']['Hot_kT'][0] for obsid in obsid_list],
                'Hot_kT_ep':[f[f'{obsid}']['Hot_kT'][2] for obsid in obsid_list],
                'Hot_kT_em':[f[f'{obsid}']['Hot_kT'][1] for obsid in obsid_list],
                'Hot_EM':[f[f'{obsid}']['Hot_EM'][0] for obsid in obsid_list],
                'Hot_EM_ep':[f[f'{obsid}']['Hot_EM'][2] for obsid in obsid_list],
                'Hot_EM_em':[f[f'{obsid}']['Hot_EM'][1] for obsid in obsid_list],
                'bg_s14_low':[f[f'{obsid}']['bg_s14_low'][0] for obsid in obsid_list],
                'bg_s38_low':[f[f'{obsid}']['bg_s38_low'][0] for obsid in obsid_list],
                'bg_s54_low':[f[f'{obsid}']['bg_s54_low'][0] for obsid in obsid_list],
                'bg_s14_high':[f[f'{obsid}']['bg_s14_high'][0] for obsid in obsid_list],
                'bg_s14_high_ep':[f[f'{obsid}']['bg_s14_high'][2] for obsid in obsid_list],
                'bg_s14_high_em':[f[f'{obsid}']['bg_s14_high'][1] for obsid in obsid_list],
                'bg_s38_high':[f[f'{obsid}']['bg_s38_high'][0] for obsid in obsid_list],
                'bg_s38_high_ep':[f[f'{obsid}']['bg_s38_high'][2] for obsid in obsid_list],
                'bg_s38_high_em':[f[f'{obsid}']['bg_s38_high'][1] for obsid in obsid_list],
                'bg_s54_high':[f[f'{obsid}']['bg_s54_high'][0] for obsid in obsid_list],
                'bg_s54_high_ep':[f[f'{obsid}']['bg_s54_high'][2] for obsid in obsid_list],
                'bg_s54_high_em':[f[f'{obsid}']['bg_s54_high'][1] for obsid in obsid_list],
                'chi_sq':[f[f'{obsid}']['chi_sq'][...] for obsid in obsid_list],
                'dof':[f[f'{obsid}']['dof'][...] for obsid in obsid_list]
                }
            df = pd.DataFrame(data)
            print(df)
            df.to_csv(savecsv)

    def loadhdf5_inf(self,obsid_list:list):
        with h5py.File(self.hdf5_inf,'r') as f:
            print(f.keys())
            data = {
                'obsid':[obsid for obsid in obsid_list],
                # 's14_hardrate':[f[f'{obsid}']['inf']['s14']['hardrate'][...] for obsid in obsid_list],
                # 's38_hardrate':[f[f'{obsid}']['inf']['s38']['hardrate'][...] for obsid in obsid_list],
                # 's54_hardrate':[f[f'{obsid}']['inf']['s54']['hardrate'][...] for obsid in obsid_list],
                's14_exposure':[f[f'{obsid}']['inf']['s14']['exposure'][...] for obsid in obsid_list],
                's38_exposure':[f[f'{obsid}']['inf']['s38']['exposure'][...] for obsid in obsid_list],
                's54_exposure':[f[f'{obsid}']['inf']['s54']['exposure'][...] for obsid in obsid_list],
                # 's14_rate_low':[f[f'{obsid}']['rate']['s14']['rate_low'][...] for obsid in obsid_list],
                # 's14_rate_mid':[f[f'{obsid}']['rate']['s14']['rate_mid'][...] for obsid in obsid_list],
                # 's14_rate_high':[f[f'{obsid}']['rate']['s14']['rate_high'][...] for obsid in obsid_list],
                # 's38_rate_low':[f[f'{obsid}']['rate']['s38']['rate_low'][...] for obsid in obsid_list],
                # 's38_rate_mid':[f[f'{obsid}']['rate']['s38']['rate_mid'][...] for obsid in obsid_list],
                # 's38_rate_high':[f[f'{obsid}']['rate']['s38']['rate_high'][...] for obsid in obsid_list],
                # 's54_rate_low':[f[f'{obsid}']['rate']['s54']['rate_low'][...] for obsid in obsid_list],
                # 's54_rate_mid':[f[f'{obsid}']['rate']['s54']['rate_mid'][...] for obsid in obsid_list],
                # 's54_rate_high':[f[f'{obsid}']['rate']['s54']['rate_high'][...] for obsid in obsid_list],
                'ra':[f[f'{obsid}']['inf']['ra'][...] for obsid in obsid_list],
                'dec':[f[f'{obsid}']['inf']['dec'][...] for obsid in obsid_list]
                }
            df = pd.DataFrame(data)
            print(df)
            df.to_csv('OES_HS_inf.csv')
            return df


    def test(self):
        df = pd.read_csv('data.csv')
        # データを追加します
        df.at  = ["jiro" ,11,"Blue" ,True ] 
        df.at = ["sabro",12,"Green",False ] 
        # CSVファイルを出力します
        df.to_csv("addData.csv", index=False )

class LoadFits:
    def __init__(self,obsid):
        home            = os.environ['HOME']
        self.HalosatDir = f"{home}/Dropbox/share/work/astronomy/Halosat/raw_data/products/"
        self.hdf5_inf = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_Halosat_information.hdf5'
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


    def savehdf5(self,obsid):
        with h5py.File(self.hdf5_inf,"a") as f:
            if obsid in f.keys():
                if "inf" in f[f'{obsid}'].keys():
                    del f[f'{obsid}']["inf"]
            f.create_dataset(f"{obsid}/inf/s14/hardrate",data=self.s14_hardrate)
            f.create_dataset(f"{obsid}/inf/s14/exposure",data=self.s14_exposure)
            f.create_dataset(f"{obsid}/inf/s38/hardrate",data=self.s38_hardrate)
            f.create_dataset(f"{obsid}/inf/s38/exposure",data=self.s38_exposure)
            f.create_dataset(f"{obsid}/inf/s54/hardrate",data=self.s54_hardrate)
            f.create_dataset(f"{obsid}/inf/s54/exposure",data=self.s54_exposure)
            f.create_dataset(f"{obsid}/inf/ra",data=self.ra)
            f.create_dataset(f"{obsid}/inf/dec",data=self.dec)
        print(f'Data saved in {self.hdf5_inf}')

    def HS_out(self,obsid):
        L = LoadFits(obsid)
        L.read_hardrate()
        L.read_exposure()
        L.read_ra_dec()
        L.savehdf5(obsid)

    def s14_BGD_photoindex(self,hardrate:float):
        """
        Hard band(3-7 keV) cut level = 0.16
        """
        return -4.642*(hardrate-0.05)+0.885

    def s38_BGD_photoindex(self,hardrate:float):
        """
        Hard band(3-7 keV) cut level = 0.16
        """
        return -5.569*(hardrate-0.05)+0.850

    def s54_BGD_photoindex(self,hardrate:float):
        """
        Hard band(3-7 keV) cut level = 0.16
        """
        return -5.443*(hardrate-0.05)+0.837

    def cal_BGD_idx(self):
        """
        Calcurate BGD photo index for halosat.
        s{num} : Hard rate in 3-7 keV.
        You can check hard rate in *.pi file.
        """
        self.read_hardrate()
        self.s14_BGD_idx = self.s14_BGD_photoindex(self.s14_hardrate)
        self.s38_BGD_idx = self.s38_BGD_photoindex(self.s38_hardrate)
        self.s54_BGD_idx = self.s54_BGD_photoindex(self.s54_hardrate)
        print('-----------------------------------------------------------')
        print(f"s14 BGD photoindex = {self.s14_BGD_idx}")
        print(f"s38 BGD photoindex = {self.s38_BGD_idx}")
        print(f"s54 BGD photoindex = {self.s54_BGD_idx}")

    def fileout(self,filename):
        BGD_list = [str(self.s14_BGD_idx),str(self.s38_BGD_idx),str(self.s54_BGD_idx)]
        with open(filename, mode='w') as f:
            f.write(' '.join(BGD_list))


if __name__ == '__main__':
    if sys.argv[1] == "fit_result":
        obsid = str(sys.argv[2])
        F = FitResultOut('fit_result.log')
        F.OES_result_out(obsid)
    elif sys.argv[1] == "rate_out":
        obsid = str(sys.argv[2])
        F = FitResultOut('rate.log')
        F.rate_out(obsid)
        L = LoadFits(obsid)
        L.HS_out(obsid)
