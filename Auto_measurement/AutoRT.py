#!/usr/bin/env python

import visa
import numpy as np
import time
from pylab import *
import AC370_AC372_para as acpara

__author__ =  'Tasuku Hayashi, Keita Tanaka'
__version__=  '3.0.1' #2021.08.23

rm = visa.ResourceManager()

print('===============================================================================')
print('----------------------------------')
print(f'     Temperature Log ver{__version__}')
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

    # def GetGPIB(self):
    #     LS370 = None
    #     LS372 = None
    #     for i in rm.list_resources():
    #         check = int(i[7:9])
    #         if check==20:
    #             LS370 = rm.open_resource(i)
    #         elif check==12:
    #             LS372 = rm.open_resource(i)
    
    #     return LS370, LS372


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

    def GetResistance(self, channel, ncounts=0):
        ch = channel
        if ncounts==0:
            R_a = self.GetValue(self.LS372, ch, 'RDGR?')
            R_s = 0
        else:
            for e, i in enumerate(range(ncounts)):
                r  = self.GetValue(self.LS372, ch, 'RDGR?')
                pause(0.09)
                print(r)
                if e==0:
                    R = r
                else:
                    R = np.hstack((R, r))

            R_a = np.average(R)
            R_s = np.std(R)

        return R_a, R_s

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


class ShowInfo:
    def __init__(self, ):
        pass

    def getInfo_370(self):
        CLS = ContLakeShore()
        hr   = 100.0
        hrng = CLS.LS370.query(f'HTRRNG?')
        HRnG = acpara.num2hrng(int(hrng))
        Hpw  = math.ceil(hr * (HRnG**2) * 1e6) #uW
        info = CLS.LS370.query(f'RDGRNG? 9')
        para = info.split(',')
        mode = acpara.num2mode(int(para[0]))
        exv  = acpara.num2exv(int(para[1]))
        rng  = acpara.num2rng(int(para[2]))
        return mode, exv, rng, HRnG, Hpw

    def getInfo_372(self, channel):
        CLS = ContLakeShore()
        ch = channel
        info = CLS.LS372.query(f'INTYPE? {ch}')
        para = info.split(',')
        mode = acpara.num2mode(int(para[0]))
        exv  = acpara.num2exv(int(para[1]))
        rng  = acpara.num2rng(int(para[3]))
        return mode, exv, rng

    def soutput(self, Ts=None,Te=None,dT=None,wt=None,rc=None,contch=None,Rch=[1,2,3,4,5,6],setExV='200uV',setRnG='2.00Ohm',st='test'):
        CLS = ContLakeShore()
        mode, exv, rng, HRnG, Hpw = self.getInfo_370()
        print('''######################################\n
        Auto measurment
        ''')
        print(datetime.datetime.now().strftime("    Date %Y.%m.%d %H:%M:%S \n"))
        if Ts:
            print(f'    START Temperature {Ts} mK')
        if Te:
            print(f'    END Temperature {Te} mK')
        if dT:
            print(f'    STEP Temperature {dT} mK')
        if wt:
            print(f'    Wait time {wt} s')
        if rc:
            print(f'    Average of {rc}')

        print('    #### LS370 ####\n    Temperature control')
        if contch:
            print(f'    Control channel : ch{contch}')
        print(f'    mode : {mode}')
        print(f'    Excitation : {exv}')
        print(f'    Range : {rng}')
        print(f'    Heater Range (uW): {Hpw}')
        print(f'    Heater Range (A): {HRnG}\n')

        print('    #### LS372 ####\n    Resistance')
        for i in Rch:
            Mode, Exv, Rng = self.getInfo_372(channel=i)
            print(f'    set Excitation and Range : ch{i}...')
            print(f'    mode : {Mode}')
            print(f'    Excitation : {Exv}')
            print(f'    Range : {Rng}')
            #LS372.write(f'INTYPE {i},0,{setExV},0,{setRnG}')
        print(f'    start time {st}\n')
        print('######################################')


class RealtimePlot(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(10.6,6))
        self.initialize()

    def initialize(self):
        nt = datetime.datetime.now().strftime("%Y.%m.%d\ %H:%M:%S")
        self.fig.suptitle(rf'$\rm Temperature\ LOG\ Date\ {nt}$', size=12)
        self.ax2 = subplot(2,1,2)
        self.ax1 = subplot(2,1,1,sharex=self.ax2)
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax1.set_title(r'$\rm LS370$')
        self.ax2.set_title(r'$\rm LS372$')
        self.ax1.set_ylabel(r'$\rm Temperature\ (K)$',fontsize=14)
        self.ax2.set_xlabel(r'$\rm Time\ (s)$',fontsize=14)
        self.ax2.set_ylabel(r'$\rm Resistance\ (\Omega)$',fontsize=14)

        # プロットの初期化
        self.lines101, = self.ax1.plot([],[],'.')
        self.lines102, = self.ax1.plot([],[],'.')
        self.lines103, = self.ax1.plot([],[],'.')
        self.lines104, = self.ax1.plot([],[],'.')
        self.lines105, = self.ax1.plot([],[],'.')
        self.lines201, = self.ax2.plot([],[],'.',color='C0')
        self.lines202, = self.ax2.plot([],[],'.',color='C1')
        self.lines203, = self.ax2.plot([],[],'.',color='C2')
        self.lines204, = self.ax2.plot([],[],'.',color='C3')
        self.lines205, = self.ax2.plot([],[],'.',color='C6')
        self.lines206, = self.ax2.plot([],[],'.',color='C9')

    def search_minmax(self,data):
        cmin = np.array([])
        cmax = np.array([])
        for j in data.keys():
            cmin = np.hstack((cmin,data[j].min()))
            cmax = np.hstack((cmax,data[j].max()))
        return cmin.min(),cmax.max()

    def setData(self,t,data_LS370,data_LS372,LS370chList,LS372chList):

        #Top
        self.lines101.set_data(t, data_LS370[1])
        self.lines102.set_data(t, data_LS370[2])
        self.lines103.set_data(t, data_LS370[3])
        self.lines104.set_data(t, data_LS370[9])            
        self.ax1.set_xlim(t.min()-10.,t.max()+10)
        cmin,cmax = self.search_minmax(data_LS370)
        self.ax1.set_ylim(cmin*0.9,cmax*1.1)
        #self.ax1.set_ylim(0.001, 325)


        #set label for 370
        labeln_370 = [f'ch1 mixing: {data_LS370[1][-1]:.2f} (K)', f'ch2 still:    {data_LS370[2][-1]:.2f} (K)', f'ch3 conc.:  {data_LS370[3][-1]:.2f} (K)', f'ch9 stage:  {data_LS370[9][-1]:.2f} (K)']
        if any([13==ch for ch in LS370chList]):
            self.lines105.set_data(t, data_LS370[13])
            labeln_370.append(f'ch13 stage: {data_LS370[13][-1]:.2f} (K)')

        #Bottom
        labeln_372 = []
        if any([1==ch for ch in LS372chList]):
            self.lines201.set_data(t,data_LS372[1])
            labeln_372.append(rf'$\rm ch1: {data_LS372[1][-1]:.2f} (\Omega)$')
        if any([2==ch for ch in LS372chList]):
            self.lines202.set_data(t,data_LS372[2])
            labeln_372.append(rf'$\rm ch2: {data_LS372[2][-1]:.2f} (\Omega)$')
        if any([3==ch for ch in LS372chList]):
            self.lines203.set_data(t,data_LS372[3])
            labeln_372.append(rf'$\rm ch3: {data_LS372[3][-1]:.2f} (\Omega)$')
        if any([4==ch for ch in LS372chList]):
            self.lines204.set_data(t,data_LS372[4])
            labeln_372.append(rf'$\rm ch4: {data_LS372[4][-1]:.2f} (\Omega)$')
        if any([5==ch for ch in LS372chList]):
            self.lines205.set_data(t,data_LS372[5])
            labeln_372.append(rf'$\rm ch5: {data_LS372[5][-1]:.2f} (\Omega)$')
        if any([6==ch for ch in LS372chList]):
            self.lines206.set_data(t,data_LS372[6])
            labeln_372.append(rf'$\rm ch6: {data_LS372[6][-1]:.2f} (\Omega)$')

        cmin,cmax = self.search_minmax(data_LS372)
        self.ax2.set_xlim(t.min()-10,t.max()+10)
        self.ax2.set_ylim(cmin*0.9,cmax*1.1)
        self.ax1.set_yscale('log')
        #self.ax1.legend(labeln_370,frameon=False,loc='upper right')
        self.ax1.legend(labeln_370,loc='upper right',fontsize=10)
        #self.ax2.legend(frameon=False,loc='upper right')
        self.ax2.legend(labeln_372,loc='upper right',fontsize=10)

    def pause(self,second):
        pause(second)

class RealtimePlot_RT(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(10.6,6))
        self.initialize()

    def initialize(self):
        nt = datetime.datetime.now().strftime("%Y.%m.%d\ %H:%M:%S")
        #self.fig.suptitle(rf'$\rm R-T\ {nt}$', size=12)
        self.ax1 = subplot(1,1,1)
        self.ax1.grid(True)
        self.ax1.set_title(rf'$\rm R-T\ Date\ {nt}$',size=12)
        self.ax1.set_xlabel(r'$\rm Temperature\ (K)$',fontsize=14)
        self.ax1.set_ylabel(r'$\rm Resistance\ (\Omega)$',fontsize=14)

        self.lines101, = self.ax1.plot([],[],'.',label='ch1')
        self.lines102, = self.ax1.plot([],[],'.',label='ch2')
        self.lines103, = self.ax1.plot([],[],'.',label='ch3')
        self.lines104, = self.ax1.plot([],[],'.',label='ch4')
        self.lines105, = self.ax1.plot([],[],'.',label='ch5')
        self.lines106, = self.ax1.plot([],[],'.',label='ch6')

    def search_minmax(self,data):
        cmin = np.array([])
        cmax = np.array([])
        for j in data.keys():
            cmin = np.hstack((cmin,data[j].min()))
            cmax = np.hstack((cmax,data[j].max()))
        return cmin.min(),cmax.max()

    def setData(self,data_LS370,data_LS372,LS372chList):

        if any([1==ch for ch in LS372chList]):
            self.lines101.set_data(data_LS370[9],data_LS372[1])
        if any([2==ch for ch in LS372chList]):
            self.lines102.set_data(data_LS370[9],data_LS372[2])
        if any([3==ch for ch in LS372chList]):
            self.lines103.set_data(data_LS370[9],data_LS372[3])
        if any([4==ch for ch in LS372chList]):
            self.lines104.set_data(data_LS370[9],data_LS372[4])
        if any([5==ch for ch in LS372chList]):
            self.lines105.set_data(data_LS370[9],data_LS372[5])
        if any([6==ch for ch in LS372chList]):
            self.lines106.set_data(data_LS370[9],data_LS372[6])

        self.ax1.set_xlim((data_LS370[9].min()*0.9,data_LS370[9].max()*1.1))
        cmin,cmax = self.search_minmax(data_LS372)
        self.ax1.set_ylim(cmin*0.9, cmax*1.1)
        #self.ax1.set_yscale('log')
        self.ax1.legend(frameon=False,loc='upper right',bbox_to_anchor=(1, 1))

    def pause(self,second):
        pause(second)

class StackData:
    def __init__(self, ):
        pass

    def SetDataList(self, Tch_LS370_list, Rch_LS372_list):
        t     = np.array([])
        Tdata = {}
        T_sig = {}
        Rdata = {}
        R_sig = {}
        for i in Tch_LS370_list:
            Tdata[i] = np.array([])
            T_sig[i] = np.array([])

        for j in Rch_LS372_list:
            Rdata[j] = np.array([])
            R_sig[j] = np.array([])

        return t, Tdata, T_sig, Rdata, R_sig

    def stackdata(self, data, channel, add_data):
        if channel:
            data[channel] = np.hstack((data[channel], add_data))
        else:
            data = np.hstack((data, add_data))
        return data

class SaveData:
    def __init__(self):
        self.date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        

    def make_header_init(self,Ts=None,Te=None,dT=None,wt=None,rc=None,contch=None,Rch=[1,2,3,4,5,6],setExV=None,setRnG=None,st=None):
        si = ShowInfo()
        mode, exv, rng, HRnG, Hpw = si.getInfo_370()
        
        header_RT = f'######################################\n'
        header_RT = header_RT+f'Date {self.date[0:4]}.{self.date[4:6]}.{self.date[6:8]} {self.date[8:10]}:{self.date[10:12]}:{self.date[12:14]}\n'
        header_RT = header_RT+f'START TIME:{st} \n'
        if Ts:
            header_RT = header_RT+f'START Temperature {Ts} mK\n'
        if Te:
            header_RT = header_RT+f'END Temperature {Te} mK\n'
        if dT:
            header_RT = header_RT+f'STEP Temperature {dT} mK\n'
        if wt:
            header_RT = header_RT+f'Wait time {wt} s\n'
        if rc:
            header_RT = header_RT+f'Average of {rc}\n'

        header_RT = header_RT+'\n'
        header_RT = header_RT+'LS370 \n'
        header_RT = header_RT+f'mode : {mode}  \n'
        header_RT = header_RT+f'Excitation : {exv}  \n'
        header_RT = header_RT+f'Range : {rng}  \n'
        header_RT = header_RT+f'Heater Range (uW): {Hpw}\n'
        header_RT = header_RT+f'Heater Range (A): {HRnG}\n\n'
        header_RT = header_RT+'LS372 \n'
        header_RT = header_RT+'Channel, Mode, Excitation, Range\n'
        for i in Rch:
            Mode, Exv, Rng = si.getInfo_372(channel=i)
            header_RT = header_RT+f'ch{i}, {Mode}, {Exv}, {Rng}\n'
        
        header_RT = header_RT+f'\n######################################\n\n\n'
        
        return header_RT

    def make_header_Tlogwith372(self, wt,Rch,setExV,setRnG,st):
        header = self.make_header_init(wt=wt,Rch=Rch,setExV=setExV,setRnG=setRnG,st=st)
        return header

    def make_header_RT(self,Ts,Te,dT,wt,rc,contch,Rch,setExV,setRnG,st):
        header = self.make_header_init(Ts,Te,dT,wt,rc,contch,Rch,setExV,setRnG,st)
        return header

    def packing_data(self, t, Tdata, Rdata, Tsig=None, Rsig=None):
        header_ch = f'#0:time (s), '
        data = np.array([t])
        num = 1
        for i in Tdata.keys():
            data = np.vstack((data,Tdata[i]))
            header_ch = header_ch+f'{num}:T_ch{i} (K), '
            num += 1
            if Tsig:
                data = np.vstack((data,Tsig[i]))
                header_ch = header_ch+f'{num}:T_sig_ch{i} (K), '
                num += 1

        for j in Rdata.keys():
            data = np.vstack((data,Rdata[j]))
            header_ch = header_ch+f'{num}:R_ch{j} (Ohm), '
            num += 1
            if Rsig:
                data = np.vstack((data,Rsig[j]))
                header_ch = header_ch+f'{num}:R_sig_ch{j} (Ohm), '
                num += 1

        return data, header_ch

    def SetSaveFileName(self, nn, Tlogtype='Log'):
        path = '/Users/boss/TES/Experiments/Tlog/'
        if Tlogtype=='Log':
            sfn = path+f'Tlog_{self.date}_{nn}.txt'
        elif Tlogtype=='RT':
            sfn = path+f'RT_{self.date}_{nn}.txt'
        elif Tlogtype=='RRR':
            sfn = path+f'RRR_{self.date}_{nn}.txt'

        return sfn

    def SaveData(self,sfn,wt,Rch,setExV,setRnG,st,t,Tdata,Rdata,Tsig=None,Rsig=None):
        header_init = self.make_header_Tlogwith372(wt,Rch,setExV,setRnG,st)
        data, header_ch = self.packing_data(t,Tdata,Rdata,Tsig,Rsig)
        header = header_init+header_ch
        np.savetxt(sfn, data.T, header=header)

    def SaveData_RT(self,sfn,Ts,Te,dT,wt,rc,contch,Rch,setExV,setRnG,st,t,Tdata,Rdata,Tsig,Rsig):
        header_init = self.make_header_RT(Ts,Te,dT,wt,rc,contch,Rch,setExV,setRnG,st)
        data, header_ch = self.packing_data(t,Tdata,Rdata,Tsig,Rsig)
        header = header_init+header_ch
        np.savetxt(sfn, data.T, header=header)

class Measurement:
    def __init__(self, ):
        pass

    def RRR(rc=10000,nn='test',Rch_LS372_list=[1,2,3,4,5,6],Exv_Rng=['20uV','2.00 Ohm']):
        sd = StackData()
        LS = ContLakeShore()
        s  = SaveData()

        nn = nn
        Rch_LS372_list = Rch_LS372_list
        Exv_LS372 = Exv_Rng[0]
        Rng_LS372 = Exv_Rng[1]

        for ch in Rch_LS372_list:
            LS.SetExvRnG_LS372(channel=ch, setExV=Exv_LS372, setRng=Rng_LS372)

        # save folder
        sfn = s.SetSaveFileName(nn=nn, Tlogtype='RRR')    
        # start time
        st = time.time()
        # set read channel of LS370
        Tch_LS370_list = [9]
        t, Tdata, T_s_data, Rdata, R_s_data = sd.SetDataList(Tch_LS370_list, Rch_LS372_list)

        t = sd.stackdata(t,None,time.time())

        for LS370ch in Tch_LS370_list:
            LS.SetScanCh_LS370(LS370ch)
            T_a, T_s = LS.GetTemperature(LS370ch, ncounts=rc)
            Tdata = sd.stackdata(Tdata,LS370ch,T_a)
            Tsig = sd.stackdata(Tsig,LS370ch,T_s)

        for LS372ch in Rch_LS372_list:
            LS.SetScanCh_LS372(LS372ch)
            R_a, R_s = LS.GetResistance(LS372ch, ncounts=rc)
            Rdata = sd.stackdata(Rdata,LS372ch,R_a)
            Rsig = sd.stackdata(Rsig,LS372ch,R_s)

        s.SaveData(sfn,wt,Rch_LS372_list,Exv_LS372,Rng_LS372,st,t,Tdata,Rdata,Tsig,Rsig)

    def RT(self,Ts=80,Te=300.,dT=10.,wt=10.,rc=100,Exv_Rng=['20uV','2.00 Ohm'],nn='test',contch=9,Rch=[1,2,3,4,5,6]):
        sd = StackData()
        LS = ContLakeShore()
        s  = SaveData()
        si = ShowInfo()

        wt = wt
        nn = nn
        Rch_LS372_list = Rch
        Exv_LS372 = Exv_Rng[0]
        Rng_LS372 = Exv_Rng[1]
        monTch = contch

        #Set Exv and Rng
        for ch in Rch_LS372_list:
            LS.SetExvRnG_LS372(channel=ch, setExV=Exv_LS372, setRnG=Rng_LS372)

        # save folder
        sfn = s.SetSaveFileName(nn=nn, Tlogtype='RT')    

        # set read channel of LS370
        Tch_LS370_list = [monTch]
        LS.SetScanCh_LS370(contch)

        #Set figure
        RP = RealtimePlot_RT()

        #Set Data List
        t, Tdata, Tsig, Rdata, Rsig = sd.SetDataList(Tch_LS370_list, Rch_LS372_list)
        print(Tdata.keys())
        print(Tsig.keys())
        print(Rdata.keys())
        print(Rsig.keys())
        t_init = True
        # start time
        st = time.time()
        #Show Info
        si.soutput(Ts,Te,dT,wt,rc,monTch,Rch_LS372_list,Exv_LS372,Rng_LS372,st)        
        try:
            for e, i in enumerate(np.arange(Ts, Te, dT)):
                setT = LS.SetTemp(i)
                RP.pause(wt)

                if t_init:
                    t = sd.stackdata(t,None,st)
                    t_init = False
                else:
                    t = sd.stackdata(t,None,time.time())

                for LS370ch in Tch_LS370_list:
                    #LS.SetScanCh_LS370(LS370ch)
                    T_a, T_s = LS.GetTemperature(LS370ch, ncounts=rc)
                    Tdata    = sd.stackdata(Tdata,LS370ch,T_a)
                    Tsig    = sd.stackdata(Tsig,LS370ch,T_s)

                for LS372ch in Rch_LS372_list:
                    LS.SetScanCh_LS372(LS372ch)
                    R_a, R_s = LS.GetResistance(LS372ch, ncounts=rc)
                    Rdata    = sd.stackdata(Rdata,LS372ch,R_a)
                    Rsig    = sd.stackdata(Rsig,LS372ch,R_s)

                RP.setData(Tdata, Rdata, Rch_LS372_list)
                s.SaveData_RT(sfn,Ts,Te,dT,wt,rc,monTch,Rch_LS372_list,Exv_LS372,Rng_LS372,st,t,Tdata,Rdata,Tsig,Rsig)

        except KeyboardInterrupt:
            path = '/Users/boss/TES/Experiments/Tlog_png/'
            sf = f'{path}RT_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{nn}.png'
            savefig(sf)
            print('\n============= FINISH =============')

        path = '/Users/boss/TES/Experiments/Tlog_png/'
        sf = f'{path}RT_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{nn}.png'
        savefig(sf)
        print('\n============= FINISH =============')



    def LogTemp():
        return None


    def LogTemp_with372(self,wt=10.,nn='test',ch13=False,Rch_LS372_list=[1,2,3,4,5,6],Exv_Rng=['200uV','2.00 Ohm']):
        sd = StackData()
        LS = ContLakeShore()
        s  = SaveData()
        si = ShowInfo()

        wt = wt
        nn = nn
        Rch_LS372_list = Rch_LS372_list
        Exv_LS372 = Exv_Rng[0]
        Rng_LS372 = Exv_Rng[1]

        #Set Exv and Rng
        for ch in Rch_LS372_list:
            LS.SetExvRnG_LS372(channel=ch, setExV=Exv_LS372, setRnG=Rng_LS372)

        # save folder
        sfn = s.SetSaveFileName(nn=nn, Tlogtype='Log')
        # set read channel of LS370
        if ch13:
            Tch_LS370_list = [1,2,3,9,13]
        else:
            Tch_LS370_list = [1,2,3,9]

        #Set figure
        RP = RealtimePlot()

        #Set Data List
        t, Tdata, T_sig, Rdata, R_sig = sd.SetDataList(Tch_LS370_list, Rch_LS372_list)
        t_init = True
        # start time
        st = time.time()
        #Show Info
        si.soutput(None,None,None,wt,None,None,Rch_LS372_list,Exv_LS372,Rng_LS372,st)  
        try:
            while True:
                if t_init:
                    t = sd.stackdata(t,None,st)
                    t_init = False
                else:
                    t = sd.stackdata(t,None,time.time())

                for LS370ch in Tch_LS370_list:
                    LS.SetScanCh_LS370(LS370ch)
                    T_a, T_s = LS.GetTemperature(LS370ch, ncounts=0)
                    Tdata    = sd.stackdata(Tdata,LS370ch,T_a)

                for LS372ch in Rch_LS372_list:
                    LS.SetScanCh_LS372(LS372ch)
                    R_a, R_s = LS.GetResistance(LS372ch, ncounts=0)
                    Rdata    = sd.stackdata(Rdata,LS372ch,R_a)

                dt = t-st
                RP.setData(dt, Tdata, Rdata, Tch_LS370_list, Rch_LS372_list)
                s.SaveData(sfn,wt=wt,Rch=Rch_LS372_list,setExV=Exv_LS372,setRnG=Rng_LS372,st=st,t=t,Tdata=Tdata,Rdata=Rdata)
                RP.pause(wt)

        except KeyboardInterrupt:
            path = '/Users/boss/TES/Experiments/Tlog_png/'
            sf = sf = f'{path}TLog_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{nn}.png'
            savefig(sf)
            print('\n============= FINISH =============')
