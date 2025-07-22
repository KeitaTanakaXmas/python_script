
import visa
import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import *

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2021.09.08

rm = visa.ResourceManager()

print('===============================================================================')
print('----------------------------------')
print(f'     Auto Current-Voltage measurement{__version__}')
print(f'          by {__author__}')
print('----------------------------------')
print('*GPIB List')
print(rm.list_resources())
print('*Use GPIB Instrument')
print('===============================================================================')


class ContKEITHLY:
    def _init_(self):
        self.KEI = rm.open_resource('GPIB1::24::INSTR') #TES bias source
        self.KEI.write(":SOUR:FUNC VOLT")
        self.KEI.write(":SOUR:VOLT:RANG 20")
        self.KEI.write(":SOUR:VOLT:LEV 15")
        self.KEI.write(":OUTP:SMOD HIMP")
        self.KEI.write(":CURR:PROT:LEV 2.1e-3")   
        self.KEI.write(":OUTP ON")

    def SetVoltage(self,V=0):
        setV = ':SOUR:VOLT:LEV %.4f' %(V)
        KEI.write(setV)

class ContKEYSIGHT:
    def _init_(self):
        self.KEY = rm.open_resource('USB0::0x2A8D::0x1301::MY57226010::INSTR') #measure TES bias voltage

    def MVolt(self,range=10.,res=1E-3):
        return KEI.query(f'MEAS:VOLT:DC? {range},{res}')

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

    def CheckTemp(self,channel=9,ncounts=100):
        T_s = 10
        while T_s < 1.:
            T_a,T_s = CLS.GetTemperature(channel=channel, ncounts=ncounts)
        print("Temperature reached setting value")
        time.sleep(60.)

    def SetScanCh_LS370(self, channel, t_pause=5):
        ch = channel
        self.LS370.write(f'SCAN {ch},0')
        print(f'scanning ch{ch}...')
        pause(t_pause)

    def SetScanCh_LS372(self, channel, t_pause=30):
        ch = channel
        self.LS372.write(f'SCAN {ch},0')
        print(f'scanning ch{ch}...')
        pause(t_pause)

    def SetExvRnG_LS372(self, channel, setExV, setRnG):
        ch = channel
        if type(setExV)==str:
            setExV = acpara.exv2num(setExV)
        if type(setRnG)==str:
            setRnG = acpara.rng2num(setRnG)
        print(f'Set Excitation and Range : ch{ch}...')
        self.LS372.write(f'INTYPE {ch},0,{setExV},0,{setRnG}')

class SaveData:
    def _init_(self):
        self.hipname = datetime.datetime.now().strftime("_%Y%m%d%H%M%S")+'.txt'

    def savename(self,headname):
        self.savefile  = headname + self.hipname


class StackData:
    def _init_(self,):
        pass

    def SetDataList(self,num,data,SetV,Ibias,Vout):
        if num == 0:
            data = np.array([SetV,Ibias,Vout])
        else: 
            data = np.vstack((data,np.array([SetV,Ibias,Vout])))

        return data

class IVMeasurment:
    def _init_(self,):
        pass

    def semiauto_IV(self,Vwt=0.5,Is=1500.,Ie=0.,dI=1.,Rtesb=1e+4,setT):
        DVS = ContactKEITHLEY()
        DVM = ContactKEYSIGHT()
        CLS = ContLakeShore()
        '''
         Is start fo TES bias UNIT = uA (Defalt : 1500 uA)
         Ie end of TES bias UNIT = uA   (Defalt : 0 uA)
         dI step of TES bias UNIT = uA   (Defalt : 1 uA)
         Rtesb Resistance of the TES bias line = Ω (Default : 10kΩ)
        '''
        #Rtesb = 10561.23058402594 + 1000 #TES bias Resistance
        CLS.SetTemp(setT=setT)
        CLS.CheckTemp(channel=9,ncounts=100)
        Vs = Is*1e-6*Rtesb
        Ve = Ie*1e-6*Rtesb
        dV = dI*1e-6*Rtesb
        data = np.array([])
        print("SetV[V], Ibias[A], Vout[V]")        
        for num, SetV in enumerate(np.arange(Vs,Ve,dV)):
            DVS.SetVoltage(SetV)
            Ibias = SetV/Rtesb
            time.sleep(wt)
            Vout = DVM.MVolt()
            data = StackData.SetDataList(num=num,data=data,SetV=SetV,Ibias=Ibias,Vout=Vout)
            print('%.3f %.6f %.6f' %(SetV, Ibias, Vout))
            plt.plot(data[:,1],data[:,2])

        return data

    def auto_IV(self,Twt=600.,Is=1500,Ie=0,dI=1,Rtesb=1e+4,Ts,Te,dT,Tb):
        for T in np.arange(Ts,Te,dT):
            data = self.semiauto_IV(Is=Is,Ie=Ie,dI=dI,Rtesb=Rtesb,setT=T)
            time.sleep(Twt)








