import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import glob
import sys

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2022.12.17

print('===============================================================================')
print(f"Output observation information ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')


class Information:

    def __init__(self,file):
        self.file = file
        self.hdu = fits.open(self.file)[1]
        self.Ra = self.hdu.header['RA_OBJ']
        self.Dec = self.hdu.header['DEC_OBJ']
        print(self.Ra,self.Dec)

    def RaDec(self):
        return self.Ra,self.Dec

    def fileout(self):
        filename = "RaDec.txt"
        RaDec_list = [str(self.Ra),str(self.Dec)]
        with open(filename, mode='w') as f:
            f.write(' '.join(RaDec_list))

fitsfile = str(sys.argv[1])
IF = Information(fitsfile)
IF.fileout()