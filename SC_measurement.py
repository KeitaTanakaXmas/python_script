import pyvisa
import numpy as np
import time,datetime

class DeviceManeger:

    def __init__(self):
        self.rm          = pyvisa.ResourceManager()
        self.device_name = []
        print(self.rm.list_resources())

    def read_devices(self):
        if 'GPIB0::24::INSTR' in self.rm:
            self.KEITHLEY = self.rm.open_resource('GPIB0::24::INSTR')
            self.device_name.append('KEITHLEY')
        if 'USB0::0x2A8D::0x1301::MY57225762::INSTR' in self.rm:
            self.KEYSIGHT = self.rm.open_resource('USB0::0x2A8D::0x1301::MY57225762::INSTR')
            self.device_name.append('KEYSIGHT')

    def initialize(self):
        print(self.device_name)
        if 'KEITHLEY' in self.device_name:
            self.KEITHLEY.write(':SYST:REM')
            self.KEITHLEY.write('*RST')
            self.KEITHLEY.write(':SOUR:CURR:MODE')
            self.KEITHLEY.write(':SOUR:CURR:RANG:AUTO ON')
            self.KEITHLEY.write(":SOUR:CURR:LEV 0")
            self.KEITHLEY.write(":OUTP:SMOD HIMP")
            self.KEITHLEY.write(":SOUR:VOLT:PROT:LEV 1")  

        if 'KEYSIGHT' in self.device_name:
            pass

class Measurement(DeviceManeger):

    def __init__(self):
        self.read_devices()
        self.initialize()

    def SC_calibration(self,name='test',wt=0.1,Is=0,Ie=40e-3,dI=1e-3):
        I = np.arange(Is,Ie+dI,dI)
        Vout = []
        # Voff = []
        for i in I:
            self.KEITHLEY.write(f':SOUR:CURR:LEV {i}')
            Vout.append(self.KEYSIGHT.query('MEAS:VOLT:DC? 10 V'))
            time.sleep(wt)
        Vout = np.array(Vout)
        # Voff = np.array(Voff)
        self.Data = np.vstack((I,Vout))
        # self.Data = np.vstack((I,Vout,Voff))
        SaveFile.savefile(name)



class SaveFile:
    def __init__(self,sfilename):
        self.date      = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.st        = time.time()
        self.Tbath     = None
        self.sfilename = sfilename
        self.Ts        = None
        self.Te        = None
        self.dT        = None
        self.ncounts   = None
        self.wt        = None
        self.Rfbfll    = None
        self.Data      = None
        self.Vac       = None
        self.Rac       = None
        self.Ibias     = None
        self.cTemp     = None
        self.smode     = None
        self.sfhz      = None
        self.efhz      = None
        self.state     = None

    def MakeHeader(self):

        header = f'######################################\n'
        header = header+f'Date {self.date[0:4]}.{self.date[4:6]}.{self.date[6:8]} {self.date[8:10]}:{self.date[10:12]}:{self.date[12:14]}\n'
        header = header+f'START TIME:{self.st} \n'
        header = header+f'Measurement mode:{self.smode} \n'        
        header = header+f'\n######################################\n\n\n'

        if self.smode == 'SC_cal':
            header = header+f'Icoil(uA) Vout(uV)\n'

        self.headerIV = header

    def savefile(self):
        self.MakeHeader()
        X = self.Data
        np.savetxt(self.sfilename,X.T,header=self.header)