#!/usr/bin/env python
#python3.7 or higher

#SB**uAのフォルダがあるところにいく
#import mat2hdf
#m = mat2hdf.mat2hdf5()
#m.mat2hdf5('Min.hdf5','SB*',datatype='phiv')

#SQUIDの解析
# import sqana
# sq = sqana.SQUID('Min.hdf5')
# sq.ShowMutual()
# sq.plotPhiV()

import numpy as np
from scipy.constants import physical_constants
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from dataclasses import field, dataclass
import mat2hdf

__author__ =  'Tasuku Hayashi'
__version__=  '1.0' #2023.12.22

print('========================================================')
print(f' SQUID analysis version {__version__}')
print('========================================================')

@dataclass
class SQBiasData:
    xt:          list  = field(default_factory=list)
    x:           list  = field(default_factory=list)
    y:           list  = field(default_factory=list)
    Iscale:      float = 60.e-6
    Ioffset:     float = 0
    Vscale:      float = 2000
    Voffset:     float = 0
    I:           list  = field(default_factory=list)
    V:           list  = field(default_factory=list)
    Ip:          list  = field(default_factory=list)
    In:          list  = field(default_factory=list)
    Vp:          list  = field(default_factory=list)
    Vn:          list  = field(default_factory=list)
    M_A:         float = 0
    Me_A:        float = 0
    M_H:         float = 0
    Me_H:        float = 0
    phiofs_A:    float = 0
    phiofs_phi0: float = 0

    def __post_init__(self):
        """
        I-V curve
        """
        mask = (self.xt.min()/2<self.xt)&(self.xt<self.xt.max()/2)
        self.I  = self.x * self.Iscale - self.Ioffset
        self.V  = self.y / self.Vscale - self.Voffset
        self.Ip = self.I[mask]
        self.In = self.I[mask==False]
        self.Vp = self.V[mask]
        self.Vn = self.V[mask==False]


@dataclass
class AllData:
    sbbiaslist: list  = field(default_factory=list)
    M_H_list:   list  = field(default_factory=list)
    Me_H_list:  list  = field(default_factory=list)
    M_H_ave:    float = 0
    Me_H_ave:   float = 0

class SQUID:
    def __init__(self, dathdf5, Iscale=60e-6, Ioffset=0, Vscale=2000, Voffset=0, verbose=False):
        # More constants
        self.phi0    = physical_constants["mag. flux quantum"][0]
        self.verbose = verbose
        self.sbias_list = []
        
        #read hdf5 file
        r = mat2hdf.ReadHDF5(dathdf5)
        r.ReadPhiVHdf5()
        for keyname in r.x.keys():
            self.sbias_list.append(keyname)
            setattr(self, f'{keyname}', SQBiasData(r.xt,r.x[keyname],r.y[keyname],Iscale,Ioffset,Vscale,Voffset))

        #Cal mutual
        self._mutual()
        self.alldata = AllData(self.sbias_list)
        self._SetAllData()

    def _mutual(self):
        """
        Calculate mutual inductance
        """
        def find_nearest(array, value):
            idx = (abs(array-value)).argmin()
            return array[idx]

        for sbname in self.sbias_list:
            _Vp = self.__dict__[sbname].Vp
            _Vn = self.__dict__[sbname].Vn
            _Ip = self.__dict__[sbname].Ip
            _In = self.__dict__[sbname].In
            Vp = _Vp - (max(_Vp) + min(_Vp)) / 2
            Vn = _Vn - (max(_Vn) + min(_Vn)) / 2
            Vsm_p = savgol_filter(Vp, 10, 3)
            Vsm_n = savgol_filter(Vn, 10, 3)
            dp = np.sort((_In[:-1])[np.diff(np.sign(Vsm_n)) > 0])
            dn = np.sort((_Ip[:-1])[np.diff(np.sign(Vsm_p)) < 0])
        
            if self.verbose:
                print("Positive zero-crossing: " + str(dp))
                print("And their differences: " + str(np.diff(dp)))
                print("Negative zero-crossing: " + str(dn))
                print("And their differences: " + str(np.diff(dn)))
            
                print("Calculating mutual inductance...")
            
            self.__dict__[sbname].M_A = np.mean(np.append(np.diff(dp), np.diff(dn)))
            self.__dict__[sbname].Me_A = np.std(np.append(np.diff(dp), np.diff(dn)), ddof=1)
            self.__dict__[sbname].M_H = self.phi0 / np.mean(np.append(np.diff(dp), np.diff(dn)))
            self.__dict__[sbname].Me_H = self.phi0 / np.mean(np.append(np.diff(dp), np.diff(dn))) * np.std(np.append(np.diff(dp), np.diff(dn)), ddof=1) / np.mean(np.append(np.diff(dp), np.diff(dn)))

            if self.verbose:
                print("M = %f +/- %f uA = %f +/- %f pH" % (M_A*1e6, Me_A*1e6, M_H*1e12, Me_H*1e12))
            
            pe = find_nearest(dp, 0)
            ne = find_nearest(dn, 0)
            
            if abs(pe) < abs(ne):
                ne = find_nearest(dn[dn<pe], pe)
            else:
                pe = find_nearest(dp[dp>ne], ne)
            
            self.__dict__[sbname].phiofs_A = (pe+ne)/2
            self.__dict__[sbname].phiofs_phi0 = (pe+ne)/2 / ((dp[-1] - dp[0]) / (len(dp) - 1))
        
            if self.verbose:
                print("Phi-offset = %f uA = %f phi0" % (self.__dict__.phiofs_A*1e6, self.__dict__.phiofs_phi0))

    def _SetAllData(self):
        """
        Calculate mutual inductance from picoscope hdf5
        """
        for sbname in self.sbias_list:
            self.alldata.M_H_list.append(self.__dict__[sbname].M_H)
            self.alldata.Me_H_list.append(self.__dict__[sbname].Me_H)
            print(f'{sbname}: M = {self.__dict__[sbname].M_H*1e12:.2f} +/- {self.__dict__[sbname].Me_H*1e12:.2f} pH')

        self.alldata.M_H_list = np.asarray(self.alldata.M_H_list)
        self.alldata.Me_H_list = np.asarray(self.alldata.Me_H_list)
        self.alldata.M_H_ave = np.mean(self.alldata.M_H_list)
        if self.alldata.Me_H_list.shape[0] == 0:
            self.alldata.Me_H_ave = 0
        else:
            self.alldata.Me_H_ave = np.sqrt(sum(self.alldata.Me_H_list**2)/self.alldata.Me_H_list.shape[0])
        print(f'Average: M = {self.alldata.M_H_ave*1e12:.2f} +/- {self.alldata.Me_H_ave*1e12:.2f} pH')

    def ShowMutual(self):
        for sbname in self.sbias_list:
            print(f'{sbname}: M = {self.__dict__[sbname].M_H*1e12:.2f} +/- {self.__dict__[sbname].Me_H*1e12:.2f} pH')

        print(f'Average: M = {self.alldata.M_H_ave*1e12:.2f} +/- {self.alldata.Me_H_ave*1e12:.2f} pH')

    def plotPhiV(self):
        plt.figure()
        for sbname in self.sbias_list:
            plt.plot(self.__dict__[sbname].Ip*1e6, self.__dict__[sbname].Vp, label=sbname)

        plt.legend(loc=1)
        plt.grid(ls='--')
        plt.xlabel(r'$I\ (\mu A)$', fontsize=16)
        plt.ylabel(r'$V_{\rm out}\ (\rm V)$', fontsize=16)
        plt.tight_layout()
