import importlib as im
from pytes import Util,Analysis,Filter
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import astropy.io.fits as pf
import time, struct
from xspec import *

h5_fn = "/Users/tanakakeita/work/microcalorimeters/experiment/Th229/20181014_No1/No1_fitting_result.hdf5"
class PySpec:
#READ SPECTRUM DATA BY XSPEC
    def __init__(self,file):
        self.file = file
#        AllData.clear()
#        spec = Spectrum(self.file)
#        #Plot.device = "/xs"
#        Plot.xAxis="keV"
#        Plot("data")
# s = PySpec("xspec_spectrum.pi")
#PLOT SPECTRUM DATA
    def splot(self):
        AllData.clear()
        #Plot.setRebin(maxBins=None)
        spec = Spectrum(self.file)
        Plot.xAxis="keV"
        #spec.ignore("**-"+str(ran[0])+" "+str(ran[1])+"-**")
        Plot("data")
        xs=Plot.x(1,1)
        ys=Plot.y(1,1)
        xe=Plot.xErr(1,1)
        ye=Plot.yErr(1,1)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=1,color="black",label=self.file)
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.set_xlabel("PHA*1e+4[arb.unit]",fontsize=16)
        ax.set_ylabel("Counts",fontsize=16)
        plt.show()
        #s.splot = splot(self)

#FITTING PROCESS
    def pfit(self,line,ran,ch):
        AllData.clear()
        #Plot.setRebin(maxBins=None)
        spec = Spectrum(self.file)
        Plot.xAxis="keV"
        spec.ignore("**-"+str(ran[0])+" "+str(ran[1])+"-**")
        Plot("data")
        xs=Plot.x(1,1)
        ys=Plot.y(1,1)
        xe=Plot.xErr(1,1)
        ye=Plot.yErr(1,1)
        AllModels.tclLoad("/Users/tanakakeita/xspec_local_model/libmypackage.dylib")
        AllModels.lmod("mypackage","/Users/tanakakeita/xspec_local_model")
        if line == "Am241":
            allmod = "powerlaw+pgauss"
            LE = 26.3446
        if line == "Sn":
            allmod = "powerlaw+gsmooth*plorentz"
            LE = 29.10964
            NW = 33.9e-3
#            LE2 = 28.486
#            NW2 = 11.8e-3
#            IT = 0.671/0.346
        if line == "CsKa1":
            allmod = "powerlaw+gsmooth*plorentz"
            LE = 30.973
            NW = 15.6e-3
            print("Fitting Model = ",allmod)
        if line == "CsKa3":
            allmod = "powerlaw+gsmooth*plorentz"
            LE = 30.27
            NW = 15.42e-3
        m = Model(allmod)
        ac = np.max(xs)/LE
        al = ac - 0.1
        ah = ac + 0.1
#        ac = 0.52
#        al = 0.40
#        ah = 0.60
        
        if "Am" in line:
            m.setPars({1:"0.0,-1,-3,-2,9,10"},{2:"200,1,0,0,800,800"},{3:str(LE)+",-1,0,0,1e+20,1e+24"},{4:str(ac)+",1e-6,"+str(al)+","+str(al)+","+str(ah)+","+str(ah)},{5:"0,1e-3,-10,-10,10,10"},{6:"1.5600e-2,1,1e-3,1e-3,10,10"},{7:"1e+3,1e-3,0,0,1e+5,1e+5"})
        if "Cs" in line:
            m.setPars({1:"0.0,-1,-3,-2,9,10"},{2:"200,1,0,0,800,800"},{3:"1.26482e-2,1,1e-4,1e-4,5e-2,5e-2"},{4:"0.0,-1,-10,-10,10,10"},{5:str(LE)+",-1,0,0,1e+20,1e+24"},{6:str(ac)+",1e-3,"+str(al)+","+str(al)+","+str(ah)+","+str(ah)},{7:"0,1e-3,-10,-10,10,10"},{8:str(NW)+",-1,1e-3,1e-3,10,10"},{9:"0.1,1,0,0,10,10"})
        if "Sn" in line:
            m.setPars({1:"0.0,-1,-3,-2,9,10"},{2:"200,1,0,0,800,800"},{3:"1.26482e-2,1,1e-4,1e-4,5e-2,5e-2"},{4:"0.0,-1,-10,-10,10,10"},{5:str(LE)+",-1,0,0,1e+20,1e+24"},{6:str(ac)+",1e-3,"+str(al)+","+str(al)+","+str(ah)+","+str(ah)},{7:"0,1e-3,-10,-10,10,10"},{8:str(NW)+",-1,1e-3,1e-3,10,10"},{9:"0.1,1,0,0,10,10"})
#        if "Sn" in line:
#            m.setPars({1:"0.0,-1,-3,-2,9,10"},{2:"229.44,1,0,0,800,800"},{3:"1.26482e-2,1,0,0,1,1"},{4:"0.0,-1,-10,-10,10,10"},{5:str(LE)+",-1,0,0,1e+20,1e+24"},{6:str(ac)+",1e-6,"+str(al)+","+str(al)+","+str(ah)+","+str(ah)},{7:str(NW)+",-1,1e-3,1e-3,10,10"},{8:"1,1e-3,0,0,1e+5,1e+5"},{9:str(LE2)+",-1,0,0,1e+20,1e+24"},{10:str(ac)+",1e-6,"+str(al)+","+str(al)+","+str(ah)+","+str(ah)},{11:str(NW2)+",-1,1e-3,1e-3,10,10"},{12:"1,1e-3,0,0,1e+5,1e+5"})
            
        Fit.query = "yes"
        Plot.add = True
        Fit.bayes = "off"
        Fit.statMethod = "cstat 1"
        #y=Plot.model()
        Fit.perform()
        Fit.show()
        Plot.device = "/xs"
        Plot("ld,delc")
        #Plot("model")
        #Plot("resid")
        #Plot.show()
        #s.pfit = pfit(self,line,ran)
        print(np.max(xs))
        if "Am" in line:
            Fit.error("1.0 2 4 5 6 7")
        if "Cs" in line:
            Fit.error("1.0 2 3 6 7 9")
        if "Sn" in line:
            Fit.error("1.0 2 3 6 7 9")
        Plot()
#        print(AllModels(1)(2).values[0])
#        print(AllModels(1)(2).error)
#        print(AllModels(1)(4).values[0])
#        print(AllModels(1)(4).error)
#        print(AllModels(1)(5).values[0])
#        print(AllModels(1)(5).error)
#        print(AllModels(1)(6).values[0])
#        print(AllModels(1)(6).error)
        Plot.addCommand("label top " + line)
        Plot.addCommand("label x \"PHA*1e+4\"")
        Plot.addCommand("time off")
        Plot.addCommand("font ro")
        Plot.addCommand("plot")
        Plot.addCommand("hardcopy " + line + ".ps/cps")
        Plot.addCommand("wdata " + line + "_fit")
        Plot.addCommand("whead " + line + "_fit")
        Plot.addCommand("ps2pdf " + line + ".ps")
        Plot()
        print(Plot.commands)
        FR = []
        #print(Fit.nVarPars)
        FR.append(Fit.nullhyp)
        FR.append(Fit.statistic)
        FR.append(Fit.testStatistic)
        pow_norm = []
        pow_norm.append(AllModels(1)(2).values[0])
        pow_norm.append(AllModels(1)(2).error[0])
        pow_norm.append(AllModels(1)(2).error[1])
        if line == "Am241":
            pgauss_a = []
            pgauss_a.append(AllModels(1)(4).values[0])
            pgauss_a.append(AllModels(1)(4).error[0])
            pgauss_a.append(AllModels(1)(4).error[1])
            pgauss_b = []
            pgauss_b.append(AllModels(1)(5).values[0])
            pgauss_b.append(AllModels(1)(5).error[0])
            pgauss_b.append(AllModels(1)(5).error[1])
            pgauss_sig = []
            pgauss_sig.append(AllModels(1)(6).values[0])
            pgauss_sig.append(AllModels(1)(6).error[0])
            pgauss_sig.append(AllModels(1)(6).error[1])
            pgauss_norm = []
            pgauss_norm.append(AllModels(1)(7).values[0])
            pgauss_norm.append(AllModels(1)(7).error[0])
            pgauss_norm.append(AllModels(1)(7).error[1])
            with h5py.File(h5_fn) as f:
                if "CH"+str(ch) in f.keys():
                    if line in f["CH"+str(ch)].keys():
                        del f["CH"+str(ch)+"/"+line]
                f.create_group("CH"+str(ch)+"/"+line)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pow_norm",data=pow_norm)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pgauss_a",data=pgauss_a)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pgauss_b",data=pgauss_b)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pgauss_sig",data=pgauss_sig)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pgauss_norm",data=pgauss_norm)
                f.create_dataset("CH"+str(ch)+"/"+line+"/fitting_result",data=FR)
        
        if "Cs" in line:
            gsmooth_sig = []
            gsmooth_sig.append(AllModels(1)(3).values[0])
            gsmooth_sig.append(AllModels(1)(3).error[0])
            gsmooth_sig.append(AllModels(1)(3).error[0])
            plorentz_a = []
            plorentz_a.append(AllModels(1)(6).values[0])
            plorentz_a.append(AllModels(1)(6).error[0])
            plorentz_a.append(AllModels(1)(6).error[1])
            plorentz_b = []
            plorentz_b.append(AllModels(1)(7).values[0])
            plorentz_b.append(AllModels(1)(7).error[0])
            plorentz_b.append(AllModels(1)(7).error[1])
            plorentz_norm = []
            plorentz_norm.append(AllModels(1)(9).values[0])
            plorentz_norm.append(AllModels(1)(9).error[0])
            plorentz_norm.append(AllModels(1)(9).error[1])
            with h5py.File(h5_fn) as f:
                if "CH"+str(ch) in f.keys():
                    if line in f["CH"+str(ch)].keys():
                        del f["CH"+str(ch)+"/"+line]
                f.create_group("CH"+str(ch)+"/"+line)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pow_norm",data=np.array(pow_norm))
                f.create_dataset("CH"+str(ch)+"/"+line+"/gsmooth_sig",data=np.array(gsmooth_sig))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_a",data=np.array(plorentz_a))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_b",data=np.array(plorentz_b))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_norm",data=np.array(plorentz_norm))
                f.create_dataset("CH"+str(ch)+"/"+line+"/fitting_result",data=FR)
                
                Fit.steppar("6 "+str(plorentz_a[1])+" "+str(plorentz_a[2])+" 100 7 "+str(plorentz_b[1])+" "+str(plorentz_b[1])+" 100")
                
        if "Sn" in line:
            gsmooth_sig = []
            gsmooth_sig.append(AllModels(1)(3).values[0])
            gsmooth_sig.append(AllModels(1)(3).error[0])
            gsmooth_sig.append(AllModels(1)(3).error[0])
            pslorentz_a = []
            pslorentz_a.append(AllModels(1)(6).values[0])
            pslorentz_a.append(AllModels(1)(6).error[0])
            pslorentz_a.append(AllModels(1)(6).error[1])
            pslorentz_norm = []
            pslorentz_norm.append(AllModels(1)(8).values[0])
            pslorentz_norm.append(AllModels(1)(8).error[0])
            pslorentz_norm.append(AllModels(1)(8).error[1])
            with h5py.File(h5_fn) as f:
                if "CH"+str(ch) in f.keys():
                    if line in f["CH"+str(ch)].keys():
                        del f["CH"+str(ch)+"/"+line]
                f.create_group("CH"+str(ch)+"/"+line)
                f.create_dataset("CH"+str(ch)+"/"+line+"/pow_norm",data=np.array(pow_norm))
                f.create_dataset("CH"+str(ch)+"/"+line+"/gsmooth_sig",data=np.array(gsmooth_sig))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_a",data=np.array(plorentz_a))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_b",data=np.array(plorentz_b))
                f.create_dataset("CH"+str(ch)+"/"+line+"/plorentz_norm",data=np.array(plorentz_norm))
                f.create_dataset("CH"+str(ch)+"/"+line+"/fitting_result",data=FR)

            




#s = PySpec("data1_ch4_20_include.pi")
#s.pfit(line="Am241",ran=[14.0,14.6],ch=4)


