
import visa
import numpy as np
import time
from pylab import *
import AC370_AC372_para as acpara
from AutoRT import ContLakeShore

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.09.16

rm = visa.ResourceManager()

print('===============================================================================')
print('----------------------------------')
print(f'     Resistance and Temperature measurement ver{__version__}')
print(f'          by {__author__}')
print('----------------------------------')
print('*GPIB List')
print(rm.list_resources())
print('*Use GPIB Instrument')
print('===============================================================================')


class ContLakeShore:
    def __init__(self):
        for i in rm.list_resources():
            check = int(i[7:9])
            if check==20:
                self.LS370 = rm.open_resource(i)
            elif check==12:
                self.LS372 = rm.open_resource(i)
            else:
                self.LS370 = None
                self.LS372 = None

    def GetValue(self, LS, channel, keynames='RDGK?'):
        '''
        Get Resistance or Temperature
        channel 
        keynames : RDGK?=T[K], RDGR?=R[Ohm],  
        '''
        ch = channel
        keyn = keynames
        v = float(LS.query(f'{keyn} {ch}'))
        return v

    def GetTemperature(self, channel, ncounts=0):
        ch = channel
        if ncounts==0:
            T_a = self.GetValue(self.LS370, ch, 'RDGK?')
            T_s = 0
        else:
            for e, i in enumerate(range(ncounts)):
                temp = self.GetValue(self.LS370, ch, 'RDGK?')
                pause(0.09)
                print(temp)
                if e==0:
                    Temp = temp
                else:
                    Temp = np.hstack((Temp, temp))

            T_a = np.average(Temp)
            T_s = np.std(Temp)

        return T_a, T_s

    def GetTemperature_withAverage(self, channel=9, ncounts=100):
        '''
        Get Resistance and Temperature
        return T_a, T_s, R_a, R_s
        T_a is average of 'ncounts' data

        '''
        ch = channel
        for e, i in enumerate(range(ncounts)):
            temp = float(self.LS.query(f'RDGK? {ch}'))
            r    = float(self.LS.query(f'RDGR? {ch}'))
            if e == 0:
                Temp = temp
                R    = r
            else:
                Temp = np.hstack((Temp, temp))
                R    = np.hstack((R, r))

        T_a = np.average(Temp)
        R_a = np.average(R)
        T_s = np.std(Temp)
        R_s = np.std(R)

        return T_a, T_s, R_a, R_s, ch, ncounts

    def SetTemp(self, setT):
        '''
        setT (mK)
        '''
        if setT<1:
            print('unit is mK')
        else:
            setp = f'SETP {setT/1e3:.4f}'
            print(f'Set Temperature {setT:.4f} mK')
        ## (3.0.0).2f -> (3.0.1).4f

        self.LS370.query(setp)
        return setT

    def SetScanCh_LS370(self, channel, t_pause=5):
        ch = channel
        self.LS370.write(f'SCAN {ch},0')
        print(f'scanning ch{ch}...')
        pause(t_pause)

class ContKEITHLY:
    def _init_(self):
        self.KEI = rm.open_resource('GPIB1::24::INSTR') #TES bias source
        self.KEI.write(":SOUR:FUNC VOLT")
        self.KEI.write(":SOUR:VOLT:RANG 20")
        self.KEI.write(":SOUR:VOLT:LEV 15")
        self.KEI.write(":OUTP:SMOD HIMP")
        self.KEI.write(":CURR:PROT:LEV 2.1e-3")   
        self.KEI.write(":OUTP ON")

    def SetVoltage(self,V=0)
        setV = ':SOUR:VOLT:LEV %.4f' %(V)
        KEI.write(setV)

class ContKEYSIGHT:
    def _init_(self):
        self.DCV = rm.open_resource('USB0::0x2A8D::0x1301::MY57226010::INSTR') #measure TES bias voltage

    def MesVolt(self):
        return float(DCV.query('MEAS:VOLT:DC? 10,1E-3'))

class Measurement:
    def _init_(self):
        pass

    def RT_SQUID(self,Ts,Te,dT,Vl=0,Vh=40e-6):
        CLS = ContLakeShore()
        SOM = ContKEITHLY()
        DVM = ContKEYSIGHT()
        T = Ts
        while T < Te
            CLS.setTemp(setT=T)
            SOM.SetVoltage(V=Vl)

            SOM.SetVoltage(V=Vh)
            T -= dT







