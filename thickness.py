#thickness.py
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import datetime
import glob

h5fn = "/Users/tanakakeita/work/microcalorimeters/experiment/SEED_vapor_equipment/thickness/analysis/SEED_vapor_thickness.hdf5"
dlist = "/Users/tanakakeita/work/microcalorimeters/experiment/SEED_vapor_equipment/thickness/data/SEED/"

fig = plt.figure(figsize=(15,12))
plt.subplots_adjust(wspace=15, hspace=12)
#plt.style.use('classic')
#plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます

plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.rcParams['scatter.edgecolors'] = 'black'
fs = 25
ps = 50




def xyout(num="01",Hsf=0.25,rp="./"):
    fn=np.genfromtxt(f"{rp}W{num}_cor.txt")
    y=fn[:]*0.1
    x=np.arange(0,Hsf*len(y),Hsf)
    return x,y

def read_data(fn=h5fn,name=None,mat="Au",num="01",number=None):
    with h5py.File(fn,"a") as f:
        y = f[f"No{number}/{name}/{mat}/W{num}/data"][:]
        x = f[f"No{number}/{name}/{mat}/xdata"][:]
        return x,y

def mask(fn=h5fn,name=None,mat="Au",num="01",w=100,number=None):
    with h5py.File(fn,"a") as f:
        x,y = read_data(fn=fn,name=name,mat=mat,num=num,number=number)
        n = np.ones(w)/w
        ycon = np.convolve(y,n,mode="same")
        hf = (np.max(ycon) - np.min(ycon))/2
        print(hf)
        x1 = x[(100<x)&(x<300)][np.argmin(np.abs(ycon[(100<x)&(x<300)]-hf))]
        x2 = x[(300<x)&(x<500)][np.argmin(np.abs(ycon[(300<x)&(x<500)]-hf))]
        x3 = x[(500<x)&(x<700)][np.argmin(np.abs(ycon[(500<x)&(x<700)]-hf))]
        x4 = x[(700<x)&(x<900)][np.argmin(np.abs(ycon[(700<x)&(x<900)]-hf))]
        x5 = x[(900<x)&(x<1100)][np.argmin(np.abs(ycon[(900<x)&(x<1100)]-hf))]
        x6 = x[(1100<x)&(x<1300)][np.argmin(np.abs(ycon[(1100<x)&(x<1300)]-hf))]
        return float(x1),float(x2),float(x3),float(x4),float(x5),float(x6)
    
def hight(fn=h5fn,name=None,mat="Au",num="01",w=100,jg=30,number=None):
    x,y = read_data(fn=fn,name=name,mat=mat,num=num,number=number)
    x1,x2,x3,x4,x5,x6 = mask(fn=fn,name=name,mat=mat,num=num,w=w,number=number)
    o1=y[(x<x1-jg) | ((x2+jg<x) & (x<x3-jg))]
    o2=y[((x2+jg<x) & (x<x3-jg)) | ((x4+jg<x) & (x<x5-jg))]
    o3=y[((x4+jg<x) & (x<x5-jg)) | (x6+jg<x)]
    h1=y[(x1+jg<x) & (x<x2-jg)]
    h2=y[(x3+jg<x) & (x<x4-jg)]
    h3=y[(x5+jg<x) & (x<x6-jg)]
    all_mask= ((x<x1-jg) | ((x2+jg<x) & (x<x3-jg))) | (((x2+jg<x) & (x<x3-jg)) | ((x4+jg<x) & (x<x5-jg))) | ((x4+jg<x) & ((x<x5-jg)) | (x6+jg<x)) | ((x1+jg<x) & (x<x2-jg)) | ((x3+jg<x) & (x<x4-jg)) | ((x5+jg<x) & (x<x6-jg))
    L1=np.nanmean(h1)-np.nanmean(o1)
    L2=np.nanmean(h2)-np.nanmean(o2)
    L3=np.nanmean(h3)-np.nanmean(o3)
    L1_sig=np.sqrt(np.nanstd(h1,ddof=1)**2+np.nanstd(o1,ddof=1)**2)
    L2_sig=np.sqrt(np.nanstd(h2,ddof=1)**2+np.nanstd(o2,ddof=1)**2)
    L3_sig=np.sqrt(np.nanstd(h3,ddof=1)**2+np.nanstd(o3,ddof=1)**2)
    L=[L1,L1_sig,L2,L2_sig,L3,L3_sig]
    o=[np.nanmean(o1),np.nanmean(o2),np.nanmean(o3),np.nanstd(o1,ddof=1),np.nanstd(o2,ddof=1),np.nanstd(o3,ddof=1)]
    h=[np.nanmean(h1),np.nanmean(h2),np.nanmean(h3),np.nanstd(h1,ddof=1),np.nanstd(h2,ddof=1),np.nanstd(h3,ddof=1)]
    print("----------------------------------------------------------------------")
    print("W"+num+"status")
    print("offset=",o)
    print("hight=",h)
    return L,o,h,all_mask

def plot_thick(fn=h5fn,name=None,mat="Au",num="01",w=100,jg=30,bin=False,number=None):
    x,y = read_data(fn=fn,name=name,mat=mat,num=num,number=number)
    L,o,h,all_mask=hight(fn=fn,name=name,mat=mat,num=num,w=w,jg=jg,number=number)
    width = 100
    if bin:
        x_bin = x[:(x.size // width) * width].reshape(-1, width).mean(axis=1)
        y_bin = y[:(y.size // width) * width].reshape(-1, width).mean(axis=1)
    else:
        x_bin = x
        y_bin = y
    fig = plt.figure(figsize=(8.0,6.0))
    ax = fig.add_subplot(111)
    ax.plot(x_bin,y_bin,".")
    # print(x1,x2,x3,x4,x5,x6)
    # print(np.min(y_bin),np.max(y_bin))
    # print(y_bin)
    ax.plot(x_bin[all_mask],y_bin[all_mask],".")
    ax.set_xlabel("length[nm]",fontsize=16)
    ax.set_ylabel("hight[nm]",fontsize=16)
    ax.set_title("W"+num,fontsize=16)
    boxdic = {
    "facecolor" : "white",
    "edgecolor" : "black",
    "boxstyle" : "Round",
    "linewidth" : 1
}
    fig.text(0.72, 0.84, 'L1=%.1f±%.1f[nm]\nL2=%.1f±%.1f[nm]\nL3=%.1f±%.1f[nm]' %(L[0],L[1],L[2],L[3],L[4],L[5]), ha='left',fontsize=16,bbox=boxdic)
    plt.show()
    fig.savefig(f"{dlist}No{number}/{name}/figure/{mat}_W{num}.png")
    return None

def all_plot_thick(fn=h5fn,name=None,mat="Au",w=100,jg=30,bin=False,number=None):
    for num in ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]:
        plot_thick(fn=fn,name=name,mat=mat,num=num,w=w,jg=jg,bin=bin,number=number)

def save_avg(fn=h5fn,name=None,mat="Au",w=100,jg=30,number=None):
    with h5py.File(fn,"a") as f:
        for num in ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]:
            L,o,h,all_mask = hight(fn=fn,name=name,mat=mat,num=num,w=w,jg=jg,number=number)
            if "average" in f[f"No{number}/{name}/{mat}/W{num}"].keys():
                del f[f"No{number}/{name}/{mat}/W{num}/average"],f[f"{name}/{mat}/W{num}/error"]    
            f.create_dataset(f"No{number}/{name}/{mat}/W{num}/average",data=[L[0],L[2],L[4]])
            f.create_dataset(f"No{number}/{name}/{mat}/W{num}/error",data=[L[1],L[3],L[5]])

def cal_Ti(fn=h5fn,name=None,number=None):
    with h5py.File(fn,"a") as f:
        if "Ti" in f[f"No{number}/{name}"].keys():
            del f[f"No{number}/{name}/Ti"]
        for num in ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]:
            Au = f[f"No{number}/{name}/Au/W{num}/average"][:]
            AuTi = f[f"No{number}/{name}/Au+Ti/W{num}/average"][:]
            Au_err = f[f"No{number}/{name}/Au/W{num}/error"][:]
            AuTi_err = f[f"No{number}/{name}/Au+Ti/W{num}/error"][:]
            Ti = AuTi - Au
            Ti_err = np.sqrt(AuTi_err**2 + Au_err**2)
            f.create_dataset(f"No{number}/{name}/Ti/W{num}/average",data=Ti)
            f.create_dataset(f"No{number}/{name}/Ti/W{num}/error",data=Ti_err)
            print("----------------------------------------------------------------------")
            print(f"W{num} Ti thickness")
            print(f"Average = {Ti}")
            print(f"Error = {Ti_err}")

def dist_plot(fn=h5fn,name=None,bar=1,number=None,nl=np.array(["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"])):
    with h5py.File(fn,"a") as f:
        for i,num in enumerate(nl):
            if i == 0: 
                Au = f[f"No{number}/{name}/Au/W{num}/average"][bar]
                Ti = f[f"No{number}/{name}/Ti/W{num}/average"][bar]
                Au_err = f[f"No{number}/{name}/Au/W{num}/error"][bar]
                Ti_err = f[f"No{number}/{name}/Ti/W{num}/error"][bar]
            else:
                Au = np.append(Au,f[f"No{number}/{name}/Au/W{num}/average"][bar])
                Ti = np.append(Ti,f[f"No{number}/{name}/Ti/W{num}/average"][bar])
                Au_err = np.append(Au_err,f[f"No{number}/{name}/Au/W{num}/error"][bar])
                Ti_err = np.append(Ti_err,f[f"No{number}/{name}/Ti/W{num}/error"][bar])
        
        fig=plt.figure(figsize=(14.2,10.7))
        gs_master=GridSpec(nrows=2,ncols=3,width_ratios=[2,1,1])
        ax1=fig.add_subplot(gs_master[1,0])
        ax2=fig.add_subplot(gs_master[1,1])
        ax3=fig.add_subplot(gs_master[1,2])
        ax4=fig.add_subplot(gs_master[0,0])
        ax5=fig.add_subplot(gs_master[0,1])
        ax6=fig.add_subplot(gs_master[0,2])
        ax1.errorbar(nl,Au,yerr=Au_err,capsize=5, fmt='o', markersize=10,color='red',label="Au")
        Au_avg = np.average(Au,weights=1/Au_err)
        Ti_avg = np.average(Ti,weights=1/Ti_err)
        Au_std = np.std(Au,ddof=1)
        Ti_std = np.std(Ti,ddof=1)
        j1 = (nl == "01") | (nl == "04") | (nl == "08") | (nl == "12") | (nl == "15")
        j2 = (nl == "07") | (nl == "08") | (nl == "09")
        Au_vper = (Au[j1][np.argmax(Au[j1])]-Au[j1][np.argmin(Au[j1])])*100/Au[j1][np.argmin(Au[j1])]
        Ti_vper = (Ti[j1][np.argmax(Ti[j1])]-Ti[j1][np.argmin(Ti[j1])])*100/Ti[j1][np.argmin(Ti[j1])]
        Au_hper = (Au[j2][np.argmax(Au[j2])]-Au[j2][np.argmin(Au[j2])])*100/Au[j2][np.argmin(Au[j2])]
        Ti_hper = (Ti[j2][np.argmax(Ti[j2])]-Ti[j2][np.argmin(Ti[j2])])*100/Ti[j2][np.argmin(Ti[j2])]
        ax2.errorbar(nl[j1],Au[j1],yerr=Au_err[j1],capsize=5, fmt='o', markersize=10,color='red')
        ax2.plot(nl[j1],Au[j1],label="Difference:%.1f" %(Au_vper),color='red')
        ax3.errorbar(nl[j2],Au[j2],yerr=Au_err[j2],capsize=5, fmt='o', markersize=10,color='red')
        ax3.plot(nl[j2],Au[j2],label="Difference:%.1f" %(Au_hper),color='red')
        ax4.errorbar(nl,Ti,yerr=Ti_err,capsize=5, fmt='o', markersize=10,color='blue',label="Ti")
        ax5.errorbar(nl[j1],Ti[j1],yerr=Ti_err[j1],capsize=5, fmt='o', markersize=10,color='blue')
        ax5.plot(nl[j1],Ti[j1],label="Difference:%.1f" %(Ti_vper),color='blue')
        ax6.errorbar(nl[j2],Ti[j2],yerr=Ti_err[j2],capsize=5, fmt='o', markersize=10,color='blue')
        ax6.plot(nl[j2],Ti[j2],label="Difference:%.1f" %(Ti_hper),color='blue')
        ax1.set_xlabel("Chip ID",fontsize=16)
        ax1.set_ylabel("Thickness[nm]",fontsize=16)
        ax4.set_xlabel("Chip ID",fontsize=16)
        ax4.set_ylabel("Thickness[nm]",fontsize=16)
        ax1.axhspan(Au_avg*0.95,Au_avg*1.05,color="lightgray")
        ax1.hlines(Au_avg,nl[0],nl[-1],linestyle="dashed",color="black",label="Average : %.1fnm" %(Au_avg))
        ax4.axhspan(Ti_avg*0.95,Ti_avg*1.05,color="lightgray")
        ax4.hlines(Ti_avg,nl[0],nl[-1],linestyle="dashed",color="black",label="Average : %.1fnm" %(Ti_avg))
        ax2.hlines(np.average(Au[j1]),nl[j1][0],nl[j1][-1],linestyle="dashed",color="black")
        ax3.hlines(np.average(Au[j2]),nl[j2][0],nl[j2][-1],linestyle="dashed",color="black")
        ax5.hlines(np.average(Ti[j1]),nl[j1][0],nl[j1][-1],linestyle="dashed",color="black")
        ax6.hlines(np.average(Ti[j2]),nl[j2][0],nl[j2][-1],linestyle="dashed",color="black")
        hans, labs = ax1.get_legend_handles_labels()
        ax1.legend(handles=hans[::-1], labels=labs[::-1], fontsize=13)
        ax2.legend(handlelength=0,fontsize=13)
        ax3.legend(handlelength=0,fontsize=13)
        hans, labs = ax4.get_legend_handles_labels()
        ax4.legend(handles=hans[::-1], labels=labs[::-1], fontsize=13)
        ax5.legend(handlelength=0,fontsize=13)
        ax6.legend(handlelength=0,fontsize=13)
        fig.text(0.5,0.95,name, ha='center',fontsize=20)
        # 1024 * 768
        # 1pt = 1/72 inch 
        #ax.set_title(f"{name}",fontsize=16)
        ps="/Users/tanakakeita/work/microcalorimeters/experiment/SEED_vapor_equipment/thickness/analysis/graph/"
        fig.savefig(f"{ps}{name}_dist.png",dpi=300)
        print("Au Average = ",Au_avg)
        print("Au Error = ",Au_std)
        print("Ti Average = ",Ti_avg)
        print("Ti Error = ",Ti_std)
        if "result" in f[f"No{number}/{name}"].keys() :
            del f[f"No{number}/{name}/result"]
        f.create_dataset(f"No{number}/{name}/result/Au_average",data=Au_avg)
        f.create_dataset(f"No{number}/{name}/result/Au_error",data=Au_err)
        f.create_dataset(f"No{number}/{name}/result/Ti_average",data=Ti_avg)
        f.create_dataset(f"No{number}/{name}/result/Ti_error",data=Ti_err)


def ratio_plot(fn=h5fn,name=None,bar=1,number=None,nl = np.array(["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"])):
    with h5py.File(fn,"a") as f:
        for i,num in enumerate(nl):
            if i == 0: 
                Au = f[f"No{number}/{name}/Au/W{num}/average"][bar]
                Ti = f[f"No{number}/{name}/Ti/W{num}/average"][bar]
                Au_err = f[f"No{number}/{name}/Au/W{num}/error"][bar]
                Ti_err = f[f"No{number}/{name}/Ti/W{num}/error"][bar]
            else:
                Au = np.append(Au,f[f"No{number}/{name}/Au/W{num}/average"][bar])
                Ti = np.append(Ti,f[f"No{number}/{name}/Ti/W{num}/average"][bar])
                Au_err = np.append(Au_err,f[f"No{number}/{name}/Au/W{num}/error"][bar])
                Ti_err = np.append(Ti_err,f[f"No{number}/{name}/Ti/W{num}/error"][bar])
        err = np.sqrt((Au_err/Ti)**2 + (Au*Ti_err/Ti**2)**2)
        plt.errorbar(nl,Au/Ti,yerr=err,capsize=5, fmt='o', markersize=10,color='blue',label="Ti")
        plt.xlabel("Chip ID",fontsize=16)    
        plt.ylabel("Au/Ti ratio",fontsize=16)
        plt.title(name,fontsize=16)
        plt.show()
        print("median = ",nl[np.argsort(Au/Ti)[len(Au/Ti)//2]])
        print("max = ",nl[np.argmax(Au/Ti)])
        print("min = ",nl[np.argmin(Au/Ti)])
        print(Au/Ti,err)
        ps="/Users/tanakakeita/work/microcalorimeters/experiment/SEED_vapor_equipment/thickness/analysis/graph/"
        #fig.savefig(f"{ps}{name}_ratio.png",dpi=300)

    

#nl=["01","04","08","12","15"]


def fitter():
    Au_m_a = [16.7,98.8,77.0,76.0,103.7,99.1,115.3]
    Au_m_b = [101.6,78.0,75.1,103.9,99.0]
    Ti_m_a = [101,108.5,106.3,113.9]
    Ti_m_b = [99,99.3,113.0,112.9]
    Au_c_a = [100,500,400,400,540,540]
    Au_c_b = [0.5,0.4,0.4,0.54,0.54]
    Ti_c_a = [0.36,0.36,0.36,0.36]
    Ti_c_b = [0.36,0.36,0.36,0.36]

    def linear(x,a,b):
        return a*x + b

    popt,pcov = curve_fit(linear,Au_c_a,Au_m_a)
    print(popt,pcov)
    popt = np.array(popt)
    x = np.linspace(100,1000.0,20)
    fitres = linear(100,*popt)
    fitres = np.append(fitres,linear(500,*popt))
    fitres = np.append(fitres,linear(400,*popt))
    fitres = np.append(fitres,linear(400,*popt))
    fitres = np.append(fitres,linear(540,*popt))
    fitres = np.append(fitres,linear(540,*popt))
    residuals = Au_m_a - fitres
    fig=plt.figure(figsize=(8,6))
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    ax.plot(Au_c_a,Au_m_a,".",color="black")
    ax.plot(x,linear(x,*popt))
    ax.legend(fontsize=14)
    ax2=fig.add_subplot(gs2[:,:])
    ax2.plot(Au_c_a,residuals,".",color="black")
    ax2.plot(x,np.zeros(len(x)),color="red")
    ax2.set_xlabel("setting thickness[nm]",fontsize=ls)
    ax2.set_ylabel("data-model",fontsize=ls)
    ax.set_ylabel("measured thickness[nm]",fontsize=ls)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    plt.show()
    fig.savefig("test.png",dpi=300)

def make_set(fn=h5fn,name="def",fle="_cor.txt",nl=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"],Hsf=0.25,number=None):
    with h5py.File(fn,"a") as f:
        name = str(name)
        fp = f"{dlist}/No{number}/{name}"
        if f"No{number}" in f.keys():
            if name in f[f"No{number}"].keys():
                del f[f"No{number}/{name}"]
        f.create_group(f"No{number}/{name}")
        f.create_group(f"No{number}/{name}/Au")
        f.create_group(f"No{number}/{name}/Au+Ti")
        for num in nl:
            Au_ef = np.genfromtxt(f"{fp}/After_Au/W{num}{fle}") 
            AuTi_ef = np.genfromtxt(f"{fp}/After_Ti/W{num}{fle}")
            Au_nID = np.where(np.isnan(Au_ef))[0][0]
            AuTi_nID = np.where(np.isnan(AuTi_ef))[0][0]
            f.create_dataset(f"No{number}/{name}/Au/W{num}/data",data=Au_ef[0:Au_nID-1]*0.1)
            f.create_dataset(f"No{number}/{name}/Au+Ti/W{num}/data",data=AuTi_ef[0:AuTi_nID-1]*0.1)
        f.create_dataset(f"No{number}/{name}/Au/xdata",data=np.arange(0,Hsf*len(Au_ef[0:Au_nID-1]),Hsf))
        f.create_dataset(f"No{number}/{name}/Au+Ti/xdata",data=np.arange(0,Hsf*len(AuTi_ef[0:AuTi_nID-1]),Hsf))



def conv_test(fn=h5fn,name=None,num="15",w=100):
    with h5py.File(fn,"a") as f:
        y = f[f"{name}/Au/W{num}/data"]
        x = f[f"{name}/Au/xdata"]
        plt.plot(x,y,".")
        n = np.ones(w)/w
        ycon = np.convolve(y,n,mode="same")
        plt.plot(x,ycon,".")
        plt.show()
        hf = (np.max(y) - np.min(y))/2
        print(hf)

def process(fn=h5fn,w=100,jg=30,bin=False,name=None,number=None):
    make_set(fn=fn,name=name,number=number)
    all_plot_thick(fn=h5fn,name=name,mat="Au",w=w,jg=jg,bin=bin,number=number)
    all_plot_thick(fn=h5fn,name=name,mat="Au+Ti",w=w,jg=jg,bin=bin,number=number)
    save_avg(fn=fn,name=name,mat="Au",w=w,jg=jg,number=number)
    save_avg(fn=fn,name=name,mat="Au+Ti",w=w,jg=jg,number=number)
    cal_Ti(fn=fn,name=name,number=number)
    dist_plot(fn=fn,name=name,bar=1,number=number)
# def cal_thick(fn=h5fn,name="def",fle="_cor.txt",nl=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]):
#     with h5py.File(fn,"a") as f:
#         name = str(name)
#         for i in range():


def make_log(fn=h5fn,number=None,file=None):
    with h5py.File(fn,"a") as f:
        pn = f"{dlist}No{number}/log/{file}"
        l = np.genfromtxt(pn,skip_header=13,delimiter=",",encoding="gbk",dtype="str")
        d = np.genfromtxt(pn,skip_header=13,delimiter=",",encoding="gbk")
        for i in range(0,len(l)):
            if i == 0:
                t = datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S")
            else:
                t = np.append(t,datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S"))
        #print(t)
        if "Setting" in f[f"No{number}"].keys():
            del f[f"No{number}/Setting"]
        f.create_dataset(f"No{number}/Setting/RH_voltage",data=d[:,16])
        f.create_dataset(f"No{number}/Setting/RH_current",data=d[:,17])
        f.create_dataset(f"No{number}/Setting/RH_power",data=d[:,16]*d[:,17])
        plt.plot(d[:,14]*d[:,15])

def read_log(filename=None):
    l = np.genfromtxt(filename,skip_header=13,delimiter=",",encoding="gbk",dtype="str")
    d = np.genfromtxt(filename,skip_header=13,delimiter=",",encoding="gbk")
    for i in range(0,len(l)):
        if i == 0:
            t = datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S")
        else:
            t = np.append(t,datetime.datetime.strptime(l[i,0], "%Y/%m/%d %H:%M:%S"))
    return t,d[:,2],d[:,6],d[:,10]    

def vapor_plot(filename1=None,filename2=None):
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle="dashed")
    t,d,Ti,Au = read_log(filename=filename1)
    t2,d2,Ti2,Au2 = read_log(filename=filename2)
    ax1.plot(t,d,color="black")
    ax1.plot(t2,d2,color="black")
    ax2 = ax1.twinx()
    ax2.plot(t2,Ti2,label="Ti thickness")
    ax2.plot(t2,Au2,label="Au thickness")
    ax1.set_xlabel("Time",fontsize=20)
    ax1.set_ylabel(r"$\rm Pressure \ [Pa]$",fontsize=20)
    ax1.set_yscale("log")
    ax1.legend(fontsize=20)
    plt.show()





def fit(func="linear",setting=200):
    Au_set = np.array([0.54,1.03,0.4,1.16,0.795,0.599])
    Au_cal = np.array([103.4,169.8,75.2,193.7,153.4,115.6])
    Au_err = np.array([4.4,2.7,3.5,5.3,3.8,3.1])
    mask = [False,True,False,True,False,False]
    def linear(x,a):
        return a*x

    def square(x,a,b,c):
        return a*x**2 + b*x + c
    popt,pcov = curve_fit(eval(func),Au_set[mask],Au_cal[mask])
    print(popt)
    print(np.sqrt(pcov))
    popt = np.array(popt)
    x = np.linspace(0,1.3,100)
    fitres = eval(func)(Au_set,*popt)
    residuals = Au_cal - fitres
    fig=plt.figure(figsize=(8,6))
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    ax.errorbar(Au_set,Au_cal,yerr=Au_err,capsize=5, fmt='o', markersize=8,color="black")
    ax.plot(x,eval(func)(x,*popt),color="red")
    #ax.legend(fontsize=14)
    ax2=fig.add_subplot(gs2[:,:])
    ax2.errorbar(Au_set,residuals,yerr=Au_err,color="black",capsize=5, fmt='o', markersize=8)
    ax2.plot(x,np.zeros(len(x)),color="red")
    ax2.set_xlabel("Setting thickness[μm]",fontsize=ls)
    ax2.set_ylabel("data-model[nm]",fontsize=ls)
    ax.set_ylabel("Measured thickness[nm]",fontsize=ls)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    plt.show()
    print(setting/popt[0])
    fig.savefig("Au_setting.png",dpi=300)

def fit_Ti(func="linear",setting=30,crt=0.093):
    Au_set = np.array([0.36,0.131,0.094])
    Au_cal = np.array([112.9,46.7,33.4])
    Au_err = np.array([2.5,3.5,3.86])
    mask = [False,True,True]
    def linear(x,a):
        return a*x

    def square(x,a,b,c):
        return a*x**2 + b*x + c
    popt,pcov = curve_fit(eval(func),Au_set[mask],Au_cal[mask])
    print(popt)
    print(np.sqrt(pcov))
    popt = np.array(popt)
    x = np.linspace(0,0.5,100)
    fitres = eval(func)(Au_set,*popt)
    residuals = Au_cal - fitres
    fig=plt.figure(figsize=(8,6))
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    ax.errorbar(Au_set,Au_cal,yerr=Au_err,capsize=5, fmt='o', markersize=8,color="black")
    ax.plot(x,eval(func)(x,*popt),color="red")
    #ax.legend(fontsize=14)
    ax2=fig.add_subplot(gs2[:,:])
    ax2.errorbar(Au_set,residuals,yerr=Au_err,color="black",capsize=5, fmt='o', markersize=8)
    ax2.plot(x,np.zeros(len(x)),color="red")
    ax2.set_xlabel("Setting thickness[μm]",fontsize=ls)
    ax2.set_ylabel("data-model[nm]",fontsize=ls)
    ax.set_ylabel("Measured thickness[nm]",fontsize=ls)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    plt.show()
    print(setting/popt[0])
    print(crt*popt[0])
    fig.savefig("Ti_setting.png",dpi=300)

def setting(fn=h5fn,number=None):
    with h5py.File(fn,"a") as f:
        if "setting" in f[f"No{number}/"].keys():
            del f[f"No{number}/setting"]
        print("Please input setting of crystal unit(Au[um])")
        Au_set = input()
        print("Please input setting of crystal unit(Ti[um])")
        Ti_set = input()
        f.create_dataset(f"No{number}/setting/crystal_unit/Au_thickness",data=Au_set)
        f.create_dataset(f"No{number}/setting/crystal_unit/Ti_thickness",data=Ti_set)

def setting_measure(fn=h5fn,mat=None):
    with h5py.File(fn,"a") as f:
        num = [6,8,10,11]
        arf = ["b","b","b","c"]
        for i,number in enumerate(num):
            if i == 0:
                setting = np.array(f[f"No{number}/setting/crystal_unit/{mat}_thickness"][...])
                measure = np.array(f[f"No{number}/*{arf[i]}/setting/crystal_unit/{mat}_thickness"][...])
            else :
                setting = np.append(setting,f[f"No{number}/setting/crystal_unit/{mat}_thickness"][...])
                measure = np.append(measure,f[f"No{number}/*{arf[i]}/setting/crystal_unit/{mat}_thickness"][...])
    return setting,measure




