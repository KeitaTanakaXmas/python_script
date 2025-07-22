import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D
import glob
import re
import lmfit as lf
from lmfit import Model
from scipy.optimize import curve_fit
import datetime
import os

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2022.02.16

print('===============================================================================')
print(f"Analysis of SEED vapor deposition ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class VAnalysis:
    def __init__(self):
        self.data                  = {}
        self.pre_evaporate_time_Ti = 180 #[s]
        self.pre_evaporate_time_Au = 25 #[s]
        self.shutter_up_time = 10 #[s] 8sec?
        

    def read_log(self,filename):
        self.filename = filename
        l = np.genfromtxt(filename,skip_header=13,delimiter=",",encoding="gbk",dtype="str")
        d = np.genfromtxt(filename,skip_header=13,delimiter=",",encoding="gbk")
        for i in range(0,len(l)):
            if i == 0:
                t = datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S")
            else:
                t = np.append(t,datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S"))
        self.time         = t
        self.pressure     = d[:,2]
        self.Ti_rate      = d[:,5]
        self.Ti_power     = d[:,6]
        self.Ti_thickness = d[:,7]
        self.Au_rate      = d[:,9]
        self.Au_power     = d[:,10]
        self.Au_thickness = d[:,11]

    def cal_Ti_evaporate_time(self):
        self.st_Ti = np.where(self.Ti_power != 0)[0][0] + self.pre_evaporate_time_Ti
        #self.st_Ti = np.where(self.Ti_rate != 0)[0][0]
        self.ed_Ti = np.where(self.Ti_thickness != 0)[0][-1] - 1
        self.evaporate_time_Ti = self.ed_Ti-self.st_Ti

    def cal_Au_evaporate_time(self):
        check_time = 30 
        self.st_Au = np.where(self.Au_power != 0)[0][0] + self.pre_evaporate_time_Au
        if len(np.where(self.Au_power[self.st_Au:self.st_Au+check_time] == 0)[0]) != 0:
            self.st_Au = self.st_Au + check_time + np.where(self.Au_power[self.st_Au+check_time:] != 0)[0][0] + self.pre_evaporate_time_Au
            print("Happend Error in Au Vaporing")
        self.ed_Au = np.where(self.Au_thickness != 0)[0][-1]
        self.evaporate_time_Au = self.ed_Au-self.st_Au

    def cal_Average_STD(self):
        self.Ti_shutter_up_pressure = self.pressure[self.st_Ti-self.pre_evaporate_time_Ti-self.shutter_up_time]
        self.Ti_start_pressure = self.pressure[self.st_Ti]
        self.Ti_end_pressure   = self.pressure[self.ed_Ti]
        self.Ti_pressure_avg   = np.average(self.pressure[self.st_Ti:self.ed_Ti])
        self.Ti_power_avg      = np.average(self.Ti_power[self.st_Ti:self.ed_Ti])
        self.Ti_rate_avg       = np.average(self.Ti_rate[self.st_Ti:self.ed_Ti])
        self.Ti_power_std      = np.std(self.Ti_power[self.st_Ti:self.ed_Ti])
        self.Ti_rate_std       = np.std(self.Ti_rate[self.st_Ti:self.ed_Ti])
        self.Au_start_pressure = self.pressure[self.st_Au-self.pre_evaporate_time_Au]
        self.Au_end_pressure   = self.pressure[self.ed_Au]
        self.Au_pressure_avg   = np.average(self.pressure[self.st_Au:self.ed_Au])
        self.Au_power_avg      = np.average(self.Au_power[self.st_Au:self.ed_Au])
        self.Au_rate_avg       = np.average(self.Au_rate[self.st_Au:self.ed_Au])
        self.Au_power_std      = np.std(self.Au_power[self.st_Au:self.ed_Au])
        self.Au_rate_std       = np.std(self.Au_rate[self.st_Au:self.ed_Au])
        self.max_pressure      = np.max(self.pressure)
        self.properties()

    def save_properties(self,file,name):
        print('---------------------')
        print(f"Save {name} properties")
        with h5py.File(file,'a') as f:
            if f'{name}' in f.keys():
                del f[f'{name}']
            f.create_dataset(f'{name}/Ti_shutter_up_pressure',data=self.Ti_shutter_up_pressure)
            f.create_dataset(f'{name}/Ti_start_pressure',data=self.Ti_start_pressure)
            f.create_dataset(f'{name}/Ti_end_pressure',data=self.Ti_end_pressure)
            f.create_dataset(f'{name}/Ti_pressure_avg',data=self.Ti_pressure_avg)
            f.create_dataset(f'{name}/Ti_power_avg',data=self.Ti_power_avg)
            f.create_dataset(f'{name}/Ti_power_std',data=self.Ti_power_std)
            f.create_dataset(f'{name}/Ti_rate_avg',data=self.Ti_rate_avg)
            f.create_dataset(f'{name}/Ti_rate_std',data=self.Ti_rate_std)
            f.create_dataset(f'{name}/Au_start_pressure',data=self.Au_start_pressure)
            f.create_dataset(f'{name}/Au_end_pressure',data=self.Au_end_pressure)
            f.create_dataset(f'{name}/Au_pressure_avg',data=self.Au_pressure_avg)
            f.create_dataset(f'{name}/Au_power_avg',data=self.Au_power_avg)
            f.create_dataset(f'{name}/Au_power_std',data=self.Au_power_std)
            f.create_dataset(f'{name}/Au_rate_avg',data=self.Au_rate_avg)
            f.create_dataset(f'{name}/Au_rate_std',data=self.Au_rate_std)
            f.create_dataset(f'{name}/max_pressure',data=self.max_pressure)


    def cal_all(self):
        self.cal_Ti_evaporate_time()
        self.cal_Au_evaporate_time()
        self.cal_Average_STD()

    def properties(self):
        print("---------------------------------------------------------------------")
        print("Electron Beam Ti properities")
        print(f"Ti Evaporate time = {self.evaporate_time_Ti} s")
        print(f"Pressure : {self.Ti_start_pressure} Pa - {self.Ti_end_pressure} Pa")
        print(f"Pressure average = {self.Ti_pressure_avg} Pa")
        print(f"Ti power average = {self.Ti_power_avg} % ± {self.Ti_power_std} %")
        print(f"Ti rate average  = {self.Ti_rate_avg} um/s ± {self.Ti_rate_std} um/s")
        print("---------------------------------------------------------------------")
        print("Resistance heat Au properities")
        print(f"Au Evaporate time = {self.evaporate_time_Au} s")
        print(f"Pressure : {self.Au_start_pressure} Pa - {self.Au_end_pressure} Pa")
        print(f"Pressure average = {self.Au_pressure_avg} Pa")
        print(f"Au power average = {self.Au_power_avg} % ± {self.Au_power_std} %")
        print(f"Au rate average  = {self.Au_rate_avg} um/s ± {self.Au_rate_std} um/s")

    def stack_properties(self,name):
        self.data[name] = {}
        self.data[name]["Ti_start_pressure"] = self.Ti_start_pressure
        self.data[name]["Ti_end_pressure"]   = self.Ti_end_pressure
        self.data[name]["Ti_pressure_avg"]   = self.Ti_pressure_avg
        self.data[name]["Ti_power_avg"]      = self.Ti_power_avg
        self.data[name]["Ti_power_std"]      = self.Ti_power_std
        self.data[name]["Ti_rate_avg"]       = self.Ti_rate_avg
        self.data[name]["Ti_rate_std"]       = self.Ti_rate_std
        self.data[name]["Au_start_pressure"] = self.Au_start_pressure
        self.data[name]["Au_end_pressure"]   = self.Au_end_pressure
        self.data[name]["Au_pressure_avg"]   = self.Au_pressure_avg
        self.data[name]["Au_power_avg"]      = self.Au_power_avg
        self.data[name]["Au_power_std"]      = self.Au_power_std
        self.data[name]["Au_rate_avg"]       = self.Au_rate_avg
        self.data[name]["Au_rate_std"]       = self.Au_rate_std

    def single_out(self,filename):
        """Process on the single file
        """
        self.read_log(filename)
        self.cal_all()

    def all_out(self):
        dir_list = sorted(glob.glob("*"))
        print(dir_list)
        for dirname in dir_list:
            file = glob.glob(f"{dirname}/*.CSV")[0]
            print(file)
            self.single_out(filename=file)
            self.stack_properties(name=dirname)
        print(self.data)

    def plot_init(self):
        self.fig = plt.figure(figsize=(15,12))
        # plt.subplots_adjust(wspace=15, hspace=12)
        plt.rcParams['image.cmap']            = 'jet'
        plt.rcParams['font.family']           = 'Times New Roman' # font familyの設定
        plt.rcParams['mathtext.fontset']      = 'stix' # math fontの設定
        plt.rcParams["font.size"]             = 12 # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize']       = 15 # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize']       = 15 # 軸だけ変更されます
        plt.rcParams['xtick.direction']       = 'in' # x axis in
        plt.rcParams['ytick.direction']       = 'in' # y axis in 
        plt.rcParams['axes.linewidth']        = 1.0 # axis line width
        plt.rcParams['axes.grid']             = True # make grid
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['scatter.edgecolors']    = 'black'
        self.fs = 30
        self.ps = 50
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.grid(linestyle="dashed")

    def plot_log(self,filename,savepath='./'):
        self.read_log(filename=filename)
        self.plot_init()
        # self.cal_Ti_evaporate_time()
        # self.cal_Au_evaporate_time()
        # self.cal_Average_STD()
        self.ax1.plot(self.time,self.pressure,color="black")
        self.ax2 = self.ax1.twinx()
        self.ax2.plot(self.time,self.Ti_power,label=r"$\rm Ti \ output$")
        self.ax2.plot(self.time,self.Au_power,label=r"$\rm Au \ output$")
        #self.ax2.plot(self.time,self.Ti_thickness,label=r"$\rm Ti \ rate$")
        #self.ax1.scatter(self.time[self.st_Ti],self.pressure[self.st_Ti],color="red",s=20)
        #self.ax1.scatter(self.time[self.ed_Ti],self.pressure[self.ed_Ti],color="red",s=20)
        #self.ax1.scatter(self.time[self.st_Au],self.pressure[self.st_Au],color="red",s=20)
        #self.ax1.scatter(self.time[self.ed_Au],self.pressure[self.ed_Au],color="red",s=20)
        self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=15)
        self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=15)
        self.ax2.set_ylabel(r"$\rm Output power \ (\%)$",fontsize=15)
        self.ax1.set_yscale("log")
        self.ax2.legend()
        #plt.show()
        file_name = os.path.basename(filename)
        new_file_name = file_name.lower().replace(".csv", "")
        sfn =f"Vlog_{new_file_name}.png"
        self.fig.savefig(f'{savepath}/{sfn}',dpi=300)

    def plot_inst(self,filename,savepath='./'):
        self.read_log(filename=filename)
        self.plot_init()
        self.ax1.plot(self.time,self.pressure,color="black")
        self.ax2 = self.ax1.twinx()
        self.ax2.plot(self.time,self.Ti_power,label=r"$\rm Ti \ output$")
        self.ax2.plot(self.time,self.Au_power,label=r"$\rm Au \ output$")
        #self.ax2.plot(self.time,self.Ti_thickness,label=r"$\rm Ti \ rate$")
        self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=15)
        self.ax1.set_ylabel(r"$\rm Pressure \ (Pa)$",fontsize=15)
        self.ax2.set_ylabel(r"$\rm Output power \ (\%)$",fontsize=15)
        self.ax1.set_yscale("log")
        self.ax2.legend()
        self.ax1.set_title(savepath,fontsize=15)
        #plt.show()
        file_name = os.path.basename(filename)
        new_file_name = file_name.lower().replace(".csv", "")
        sfn =f"Vlog_{new_file_name}.png"
        plt.show()
        self.fig.savefig(f'{savepath}/{sfn}',dpi=300)

    def plot_thickness_rate(self,filename,savepath='./'):
        self.read_log(filename=filename)
        self.plot_init()
        self.ax1.plot(self.time,self.pressure,color="black")
        self.ax2 = self.ax1.twinx()
        self.ax2.plot(self.time,self.Au_thickness,color="Red",label=r"$\rm Ti \ thickness$")
        self.ax2.plot(self.time,self.Ti_thickness,color="Blue",label=r"$\rm Au \ thickness$")
        self.ax2.plot(self.time,self.Ti_rate,label=r"$\rm Ti \ rate$")
        self.ax2.plot(self.time,self.Au_rate,label=r"$\rm Au \ rate$")
        self.ax2.plot(self.time,self.Ti_power,label=r"$\rm Ti \ output$")
        self.ax2.plot(self.time,self.Au_power,label=r"$\rm Au \ output$")
        #self.ax2.plot(self.time,self.Ti_thickness,label=r"$\rm Ti \ rate$")
        self.ax1.set_xlabel(r"$\rm Time \ (s) $",fontsize=15)
        self.ax1.set_ylabel(r"$\rm Thickness \ (\mu m)$",fontsize=15)
        self.ax2.set_ylabel(r"$\rm Rate \ (\%)$",fontsize=15)
        #self.ax1.set_yscale("log")
        self.ax2.legend()
        #plt.show()
        self.ax1.set_title(savepath,fontsize=15)
        file_name = os.path.basename(filename)
        new_file_name = file_name.lower().replace(".csv", "")
        sfn =f"Vlog_rate_{new_file_name}.png"
        self.fig.savefig(f'{savepath}/{sfn}',dpi=300)

    def plot_cor(self,xname):
        self.plot_init()
        for i in list(self.data.keys()):
            self.ax1.scatter(i,self.data[i][xname],color="Blue")
        self.ax1.set_yscale("log")
        self.ax1.set_title(xname)
        sfn = f"../graph/{xname}.png"
        self.fig.savefig(sfn,dpi=300)
        plt.show()

    def plot_cor_all(self):
        f = list(self.data.keys())
        name = list(self.data[str(f[0])].keys())
        for i in name:
            self.plot_cor(xname=i)

    def append_Tc(self):
        Tc = np.array([0.0,0.0])
        with h5py.File('../Vlog.hdf5','r') as f:
            for i in sorted(list(f.keys())):
                self.data[i]["Tc"] = f[i]["Tc"][()]
                self.data[i]["Tc_std"] = f[i]["Tc_std"][()]
                self.data[i]["Tc_time"] = f[i]["Tc_time"][()]
                self.data[i]["Tc_time_std"] = f[i]["Tc_time_std"][()]

    def multi_ana(self):
        folder = glob.glob("*")
        for i in folder:
            basename = os.path.basename(i)
            for j in glob.glob(f"{i}/*_Vlog.CSV"):
                self.plot_thickness_rate(filename=j,savepath=f"./{basename}")
            for j in glob.glob(f"{i}/*_Vlog.csv"):
                self.plot_thickness_rate(filename=j,savepath=f"./{basename}")
    
    def analysis_Ti(self, file):
        self.plot_init()
        fig, axs = plt.subplots(4, 2, figsize=(12, 8))

        with h5py.File(file, 'r') as f:
            keys = sorted(list(f.keys()))
            print(keys)
            for i, key in enumerate(keys):
                print(key)
                if key == 'SEED220616' or key == 'SEED220714' or key == 'SEED221121':
                    color = 'red'
                else:
                    color = 'blue'
                axs[0, 0].plot(key, f[key]["Ti_start_pressure"][...], "o", color=color)
                axs[0, 0].set_title("Ti_start_pressure")
                axs[0, 0].grid(linestyle='dashed')

                axs[1, 0].plot(key, f[key]["Ti_end_pressure"][...], "o", color=color)
                axs[1, 0].set_title("Ti_end_pressure")
                axs[1, 0].grid(linestyle='dashed')

                axs[2, 0].plot(key, f[key]["Ti_pressure_avg"][...], "o", color=color)
                axs[2, 0].set_title("Ti_pressure_avg")
                axs[2, 0].grid(linestyle='dashed')

                axs[3, 0].plot(key, f[key]["Ti_power_avg"][...], "o", color=color)
                axs[3, 0].set_title("Ti_power_avg")
                axs[3, 0].grid(linestyle='dashed')

                axs[0, 1].plot(key, f[key]["Ti_power_std"][...], "o", color=color)
                axs[0, 1].set_title("Ti_power_std")
                axs[0, 1].grid(linestyle='dashed')

                axs[1, 1].plot(key, f[key]["Ti_rate_avg"][...], "o", color=color)
                axs[1, 1].set_title("Ti_rate_avg")
                axs[1, 1].grid(linestyle='dashed')

                axs[2, 1].plot(key, f[key]["Ti_rate_std"][...], "o", color=color)
                axs[2, 1].set_title("Ti_rate_std")
                axs[2, 1].grid(linestyle='dashed')

                axs[3, 1].plot(key, f[key]["max_pressure"][...], "o", color=color)
                axs[3, 1].set_title("max_pressure")
                axs[3, 1].grid(linestyle='dashed')

        plt.tight_layout()
        plt.show()

    def analysis_Au(self, file):
        self.plot_init()
        fig, axs = plt.subplots(4, 2, figsize=(12, 8))

        with h5py.File(file, 'r') as f:
            keys = sorted(list(f.keys()))
            print(keys)
            for i, key in enumerate(keys):
                print(key)
                if key == 'SEED220616' or key == 'SEED220714' or key == 'SEED221121':
                    color = 'red'
                else:
                    color = 'blue'
                axs[0, 0].plot(key, f[key]["Au_start_pressure"][...], "o", color=color)
                axs[0, 0].set_title("Au_start_pressure")
                axs[0, 0].grid(linestyle='dashed')

                axs[1, 0].plot(key, f[key]["Au_end_pressure"][...], "o", color=color)
                axs[1, 0].set_title("Au_end_pressure")
                axs[1, 0].grid(linestyle='dashed')

                axs[2, 0].plot(key, f[key]["Au_pressure_avg"][...], "o", color=color)
                axs[2, 0].set_title("Au_pressure_avg")
                axs[2, 0].grid(linestyle='dashed')

                axs[3, 0].plot(key, f[key]["Au_power_avg"][...], "o", color=color)
                axs[3, 0].set_title("Au_power_avg")
                axs[3, 0].grid(linestyle='dashed')

                axs[0, 1].plot(key, f[key]["Au_power_std"][...], "o", color=color)
                axs[0, 1].set_title("Au_power_std")
                axs[0, 1].grid(linestyle='dashed')

                axs[1, 1].plot(key, f[key]["Au_rate_avg"][...], "o", color=color)
                axs[1, 1].set_title("Au_rate_avg")
                axs[1, 1].grid(linestyle='dashed')

                axs[2, 1].plot(key, f[key]["Au_rate_std"][...], "o", color=color)
                axs[2, 1].set_title("Au_rate_std")
                axs[2, 1].grid(linestyle='dashed')

                # axs[3, 1].plot(key, f[key]["Ti_shutter_up_pressure"][...], "o", color=color)
                # axs[3, 1].set_title("Ti_shutter_up_pressure")
                # axs[3, 1].grid(linestyle='dashed')

        plt.tight_layout()
        plt.show()

    def out(self):
        folder = glob.glob("*")
        for i in folder:
            basename = os.path.basename(i)
            print('basename:', basename)
            for j in glob.glob(f"{i}/*_Vlog.CSV"):
                self.plot_inst(filename=j,savepath=f"../analysis")
            for j in glob.glob(f"{i}/*_Vlog.csv"):
                self.plot_inst(filename=j,savepath=f"../analysis")
            #self.save_properties('../Vlog.hdf5',basename)