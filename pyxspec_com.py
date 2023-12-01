from xspec import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from Basic import Plotter
import re
import h5py
import numpy as np
import sys
import pandas as pd
import os

# git test
# ver2

class PyXspec:
    def __init__(self):
        pass

    def output(self):
        DT = DataTreat()
        DT.loadspec(file1="xis0_grp80.pha",file2="xis1_grp80.pha",file3="xis3_grp80.pha")


class PyXspecModel:
    def __init__(self,model:str):
        self.model_name = model
        self.model      = Model(self.model_name)
        self.P          = Plotter()

    def setparameters(self,**kwargs):
        if self.model_name == 'apec':
            if 'T' in kwargs:
                self.T     = kwargs['T']
            else : 
                self.T     = 1.
            if 'abund' in kwargs:
                self.abund     = kwargs['abund']
            else : 
                self.abund     = 1.
            if 'z' in kwargs:
                self.z     = kwargs['z']
            else : 
                self.z     = 1.
            if 'norm' in kwargs:
                self.norm     = kwargs['norm']
            else : 
                self.norm     = 1.
            self.model.setPars(self.T,self.abund,self.z,self.norm)

    def model_xyout(self):
        Plot('model')
        self.model_y = Plot.model(1,1)
        self.E       = Plot.x(1,1)

    def model_plot(self):
        plt.plot(self.E,self.model_y,lw=0.5)
        plt.semilogx()
        plt.semilogy()
        plt.xlim(0.1,50)
        plt.ylim(1,2000)
        plt.show()

    def model_valT(self,T,new_window):
        self.setparameters(T=T,abund=1,z=0,norm=1)
        self.model_xyout()
        xname = r'$\rm Energy \ (keV)$'
        yname = r'$\rm photons\ cm^{-2} \ s^{-1} \ keV^{-1}$'
        xlim  = [0.5,10]
        ylim  = [1,2000]
        if new_window == True:
            self.P.plotting(x=self.E,y=self.model_y,lw=0.5,xlog=True,ylog=True,label=f'{T} keV',xname=xname,yname=yname)
        else:
            self.P.plotting(x=self.E,y=self.model_y,new_window=False,lw=0.5,xlog=True,ylog=True,label=f'{T} keV',color='Red')

    def apec_multi_temp(self,T):
        for e,i in enumerate(T):
            if e == 0:
                self.model_valT(T=i,new_window=True)
            else:
                self.model_valT(T=i,new_window=False)
        plt.show()


class DataTreat:
    def __init__(self):
        pass

    def loadspec(self,file1,file2,file3):
        """
        load pha files.
        file{num[1-3]} : pha files include BGD, RMF, ARF 
        """
        self.s1 = Spectrum(file1)
        self.s2 = Spectrum(file2)
        self.s3 = Spectrum(file3)

    def ignore(self,s:str,range:str):
        """
        s : spectrum number
        range : "**-0.4 5.0-**" ()
        """
        pass


    def stack_fit(self):
        result = []
        mod_name = AllModels(1).componentNames
        for name in mod_name:
            par_name = eval(f'AllModels(1).{name}.parameterNames')
            for par in par_name:
                val = eval(f'AllModels(1).{name}.{par}.values[0]')
                err = eval(f'AllModels(1).{name}.{par}.error')
                l = [name, par, val, err[0]-val, err[1]-val]
                result.append(l)
        return result

    def stack_statistic(self):
        dof = Fit.dof
        stat = Fit.statistic
        test = Fit.testStatistic
        stat_list = [stat, stat, stat, dof, test]
        return stat_list

    def stack_bgd(self,mod_word='bg'):
        result = []
        model_index = [1, 2, 3]
        mod_name = AllModels(1,mod_word).componentNames
        print(mod_name)
        for index in model_index:
            for name in mod_name:
                par_name = eval(f'AllModels({index},\'{mod_word}\').{name}.parameterNames')
                for par in par_name:
                    val = eval(f'AllModels({index},\'{mod_word}\').{name}.{par}.values[0]')
                    err = eval(f'AllModels({index},\'{mod_word}\').{name}.{par}.error')
                    l = [f'{name}_bg{index}', par, val, err[0]-val, err[1]-val]
                    result.append(l)
        return result


    def stack_all(self):
        fit = self.stack_fit()
        bgd = self.stack_bgd()
        sta = self.stack_statistic()
        for i in bgd:
            fit.append(i)
        fit.append(sta)
        return fit

    def calc_error(self,com):
        Fit.perform()
        Fit.show()
        Fit.query = 'yes'
        Fit.error(com)

    def load_xcm(self,filename):
        '''loads xcm file into pyxspec env'''
        model_flag=False
        model_param_counter=1
        model_num=1
        for cmd in open(filename):
            cmd=cmd.replace("\n","")
            print(cmd)
            if model_flag==True:
                cmd=re.sub("\s+([\.|\d|\-|\w|\+]+)\s+([\.|\d|\-|\w|\+]+)\s+([\.|\d|\-|\w|\+]+)\s+([\.|\d|\-|\w|\+]+)\s+([\.|\d|\-|\w|\+]+)\s+([\.|\d|\-|\w|\+]+)","\g<1> \g<2> \g<3> \g<4> \g<5> \g<6>",cmd).split(" ") 
                print(cmd)
                p=AllModels(model_num,mod_name)(model_param_counter)
                if "/" in cmd:
                    model_param_counter+=1
                    if model_param_counter>m.nParameters:
                        model_num+=1
                        model_param_counter=1
                        if model_num>AllData.nGroups:
                            model_flag=False
                    continue
                elif "=" in cmd:
                    p.link="".join(cmd).replace("=","")
                else:
                    p.values=list(map(float,[ z for z in cmd if not z==" "]))
                    print('model parameters')
                    print(AllModels(model_num,mod_name).nParameters)

                model_param_counter+=1
                if model_param_counter>AllModels(model_num,mod_name).nParameters:
                    model_num+=1
                    model_param_counter=1

                    if model_num>AllData.nGroups:
                        model_flag=False
            else:
                cmd=cmd.split(" ")
                if cmd[0]=="statistic":
                    Fit.statMethod=cmd[1]
                elif cmd[0]=="method":
                    Fit.method=cmd[1]
                    Fit.nIterations=int(cmd[2])
                    Fit.criticalDelta=float(cmd[3])
                elif cmd[0]=="abund":
                    Xset.abund=cmd[1]
                elif cmd[0]=="xsect":
                    Xset.xsect=cmd[1]
                elif cmd[0]=="xset":
                    if cmd[1]=="delta":
                        Fit.delta=float(cmd[2])
                elif cmd[0]=="systematic":
                    AllModels.systematic=float(cmd[1])
                elif cmd[0]=="data":
                    AllData(" ".join(cmd[1:]))
                elif cmd[0]=="ignore":
                    AllData.ignore(" ".join(cmd[1:]))
                elif cmd[0]=="response":
                    print(cmd)
                    d = int(cmd[2][:cmd[2].index(":")])-1
                    s = int(cmd[2][cmd[2].index(":")+1:])
                    sp = AllData(s)
                    print(d,s)
                    sp.multiresponse[d] = cmd[3]
                elif cmd[0]=='bayes':
                    print(cmd)
                    Fit.bayes = cmd[1]
                elif cmd[0]=='gain':
                    print(cmd)
                    d = int(cmd[1][:cmd[1].index(":")])-1
                    s = int(cmd[1][cmd[1].index(":")+1:])
                    sp = AllData(s).response
                    sp.gain.slope = f'{cmd[3]},-1'
                    sp.gain.offset = f'{cmd[5]},-1'

                elif cmd[0]=="model":
                    model_flag=True
                    print(cmd[2:])
                    if cmd[2] == '2:bg':
                        print('BGD setting')
                        # AllModels += (" ".join(cmd[3:]),'bg', 2)
                        Model(" ".join(cmd[3:]), 'bg', 2)
                        print("OK")
                        model_num = 1
                        model_param_counter = 1
                        mod_name = 'bg'
                    else:    
                        Model(" ".join(cmd[1:]),'', 1)
                        mod_name = ''
                
                elif cmd[0]=="newpar":
                    m=AllModels(1)
                    npmodel=m.nParameters #number of params in model
                    group=int(np.ceil((float(cmd[1]))/npmodel))

                    if not int(cmd[1])/npmodel==float(cmd[1])/npmodel:
                        param=int(cmd[1])-(int(cmd[1])/npmodel)*npmodel # int div so effectivly p-floor(p/npmodel)*npmodel
                    else:
                        param=npmodel
                    
                    print(group,param)
                    
                    m=AllModels(group)
                    p=m(param)
                    
                    if "=" in cmd[2] :
                        p.link="".join(cmd[2:]).replace("=","")
                    else:
                        p.values=map(float,cmd[2:])
        Xset.save(fileName='23_test.xcm',info='a')


class Plotting:

    def __init__(self):
        pass



class Save:

    def __init__(self,savehdf5:str) -> None:
        self.savehdf5 = savehdf5

    def savefit(self,data:list,obsid:str):
        with h5py.File(self.savehdf5, 'a') as f:
            if obsid in f:
                if 'fit' in f[obsid]:
                    del f[obsid]['fit']
            for idx in range(0,len(data)-1): 
                f.create_dataset(f'{obsid}/fit/param/{data[idx][0]}/{data[idx][1]}',data=data[idx][2:])
            f.create_dataset(f'{obsid}/fit/stat',data=data[-1][2:])

    def output(self,savecsv:str,obsid_list):
        with h5py.File(self.hdf5,'r') as f:


            data = {}
            for name in f[f'{obsid}/fit/param']:
                pass

            df = pd.DataFrame(data)
            print(df)
            df.to_csv(savecsv)
            




if __name__ == '__main__':
    obsid = str(sys.argv[1])
    D = DataTreat()
    D.load_xcm('test.xcm')
    D.calc_error(com='1.0 3,9,12,13,16')
    result = D.stack_all()
    home   = os.environ['HOME']
    Save(f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/Halosat_OES.hdf5').savefit(data=result,obsid=obsid)

