import numpy as np
import h5py
from sqana import SQUID
import matplotlib.pyplot as plt

class TES_Analysis:
    def __init__(self) -> None:
        pass

    def convert_hdf5(self,savefile='Min.hdf5'):
        import mat2hdf
        m = mat2hdf.mat2hdf5()
        m.mat2hdf5(savefile,'SB*',datatype='phiv')

    def plotPV(self,filename='Min.hdf5'):
        import sqana
        sq = sqana.SQUID(filename)
        sq.ShowMutual()
        sq.plotPhiV()
        plt.show()