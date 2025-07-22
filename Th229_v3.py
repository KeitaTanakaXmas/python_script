from pytes import Util,Analysis,Filter
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import astropy.io.fits as pf
import time, struct
import re
from xspec import *
from scipy.stats import chi2
import types
import subprocess
import glob
import datetime
from scipy.optimize import curve_fit
import imp

#h5_fn = "data_set_1-CH4.hdf5"

#h5_fn = "default.hdf5"
h5_fn = "data_set_1-CH4_allreg.hdf5"

def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            print (sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print (sep+'\t','-',key,':',obj.attrs[key])

def h5dump(path=h5_fn,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])


## Definition of the main function ##

def h5fn(fn=None):
    global h5_fn
    h5_fn = fn
    return h5_fn

def get_fl(s="p"):
    fl = glob.glob("./../rawdata/*"+s+".fits")
    return sorted(fl)
    
def noise_def():
    fl = get_fl(s="n")
    for i,fn in enumerate(fl):
        print("Now loading ",fn)
        if i == 0:
            t,n=Util.fopen(fn)
        else:
            t,n_e=Util.fopen(fn)
            n=np.append(n,n_e,axis=0)
    return t,n
    
def pulse_def():
    fl = get_fl(s="p")
    for i,fn in enumerate(fl):
        print("Now loading ",fn)
        if i == 0:
            t,p=Util.fopen(fn)
        else:
            t,p_a=Util.fopen(fn)
            p=np.append(p,p_a,axis=0)
    return t,p
    

def ofs_def():
    nfl = get_fl(s="n")
    for i,fn in enumerate(nfl):
        print(fn)
        t,n = Util.fopen(fn)
        if i == 0 :
            ofs = np.median(n,axis=1)
            ofs_std = np.std(n,axis=1)
        else :
            ofs = np.append(ofs,np.median(n,axis=1),axis=0)
            ofs_std = np.append(ofs_std,np.std(n,axis=1),axis=0)
    ofs_all_avg = np.median(ofs)
    ofs_all_std = np.std(ofs)
    return ofs,ofs_std,ofs_all_avg,ofs_all_std
     

def ph_def():
    pfl = get_fl(s="p")
    for i,fn in enumerate(pfl):
        t,p = Util.fopen(fn)
        if i == 0 :
            pmin = np.min(p,axis=1)
        else :
            pmin = np.append(pmin,np.min(p,axis=1),axis=0)
    ofs,ofs_std,ofs_all_avg,ofs_all_std = ofs_def()
    ph = ofs - pmin
    return ph

## Definition of the Mask ## 

def omask(sig=3,ofs=None,ofs_all_avg=None,ofs_all_std=None):
    omask = (ofs_all_avg-sig*ofs_all_std < ofs) & (ofs < ofs_all_avg+sig*ofs_all_std)
    return omask

def imask(itrig=None,length=None):
    imask = (np.array(range(length)) < itrig)
    return imask   

def ph_mask(ph=None,rl=None,rh=None):
    ph_mask = (rl < ph) & (ph < rh)
    return ph_mask
    
def rmask(length=None,sID=None,eID=None):
    rmask = (sID < np.array(range(length))) & (np.array(range(length)) < eID)
    return rmask

def dmask(psig=20,nsig=5,tsig=5,order=50,trig=0.1):
    pfl = get_fl(s="p")
    ap_mask = []
    for fl in pfl:
        t,wave_p=Util.fopen(fl)
        length=len(wave_p[0])
        from scipy.signal import argrelextrema
        pulse_cl = []
        p_j      = []
        count_t = int(len(wave_p))
        if count_t == 0:
            print('There is not pulse data.')
            p = []
            pulse_cl = p
            p_j = p
            pass
        else:
            p  = wave_p.reshape((1, len(wave_p)*length))[0]
            p -= np.median(p)
            p  = p.reshape((int(len(p)/length), length))
            p_mask   = []
            print('checking double pulse...')
            for e, i in enumerate(p):
                p_b  = np.correlate(i,[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1], mode='same')
                flag = argrelextrema(-p_b, np.greater_equal, order=order)[0]
                threshold = -(np.median(p_b[0:90])+(psig*np.std(p_b[0:90])))
                check = p_b[flag]<threshold
                if len(p_b[flag][check==True]) > 1:
                    p_mask.append(False)
                    p_j.append(i)
                else:
                    if length==1000:
                        tmin = 80
                        tmax = 120
                    else:
                        tmin = trig*length*0.80
                        tmax = trig*length*1.20
                    if (tmin<flag[check])&(flag[check]<tmax):
                        pulse_cl.append(i)
                        p_mask.append(True)
                    else:
                        p_mask.append(False)
                        p_j.append(i)
                print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)
            count_j = int(len(p_j))
            count_cl = int(len(pulse_cl))
            #print('\n')
            print(f'Number of Junck pulse  {count_j}')
            print(f'Number of clean events {count_cl}')
            if count_t == 0:
                pass
            else:
                print(f'Pulse removal ratio {count_cl/count_t*100:.2f}%\n')
            #print('\n')
        pulse_cl = np.asarray(pulse_cl)
        p_j = np.asarray(p_j)
        ap_mask = np.append(ap_mask,p_mask)
    return ap_mask

## Make hdf5 file & data group ##

def make_hd(fn=h5_fn):
    print("Now Makimg hdf5 file ...")
    print("File name = ",fn)
    with h5py.File(fn) as f:
        ofs,ofs_std,ofs_all_avg,ofs_all_std = ofs_def()
        ph = ph_def()
        if "data" in f.keys():
            del f["data"]
        dgrp=f.create_group("data")
        dgrp.attrs["description"] = "All Data exist in this directory"
        dgrp.attrs["creates_at"] = datetime.datetime.now().isoformat()         
        f.create_dataset("data/all/offset",data=ofs)
        f.create_dataset("data/all/offset_error",data=ofs_std)
        f.create_dataset("data/all/offset_all_average",data=ofs_all_avg)
        f.create_dataset("data/all/offset_all_error",data=ofs_all_std)
        f.create_dataset("data/all/ph",data=ph)
        f.create_dataset("data/all/length",data=len(ofs))


def make_mask(fn=h5_fn,sID=None,eID=None,sig=3,reg=1,dm=True,rm=True,om=True,fm=True):
    with h5py.File(fn) as f:
        length = f["data/all/length"][...]
        dmasks = dmask(psig=20,nsig=5,tsig=5,order=50,trig=0.1)[sID:length]
        rmasks = rmask(length=length,sID=sID,eID=eID)
        rdmask = (dmasks>0.9) & (rmasks)
        ofs = f["data/all/offset"][:]
        ofs_std_e = f["data/all/offset_error"][:][rdmask]
        ofs_e = ofs[rdmask]
        ofs_all_avg = np.median(ofs_e)
        ofs_all_std = np.std(ofs_e)
        ph = f["data/all/ph"][:][rdmask]
        ID = [sID,eID]  
        omasks = omask(sig=sig,ofs=ofs,ofs_all_avg=ofs_all_avg,ofs_all_std=ofs_all_std)
        if "data_reg"+str(reg) in f.keys():
            del f["data/reg"+str(reg)]
            del f["mask/reg"+str(reg)]
        f.create_group("data_reg"+str(reg))         
        f.create_dataset("data/reg"+str(reg)+"/offset",data=ofs_e)
        f.create_dataset("data/reg"+str(reg)+"/offset_error",data=ofs_std_e)
        f.create_dataset("data/reg"+str(reg)+"/offset_all_average",data=ofs_all_avg)
        f.create_dataset("data/reg"+str(reg)+"/offset_all_error",data=ofs_all_std)
        f.create_dataset("data/reg"+str(reg)+"/ph",data=ph)
        f.create_dataset("data/reg"+str(reg)+"/length",data=len(ofs))
        f.create_dataset("data/reg"+str(reg)+"/ID",data=ID)
        f.create_dataset("mask/reg"+str(reg)+"/rmask",data=rmasks)
        f.create_dataset("mask/reg"+str(reg)+"/dmask",data=dmasks)
        f.create_dataset("mask/reg"+str(reg)+"/omask",data=omasks)
        print(ID)


def make_Am241(rl=None,rh=None,reg=1):
    ph = f["data/reg"+str(reg)+"ph"][:]        
    phmask = ph_mask(ph=ph,rl=rl,rh=rh)
    f.create_dataset("mask/reg"+str(reg)+"/phmask",data=phmask)
    print("All Data Number = ",len(ph))
    print("Reducted Data Number = ",len(ph)-len(ph[phmask]))

#tmpl generate
def tmpl_apsel(fn=h5_fn):
    fl = get_fl(s="p")
    with h5py.File(fn) as f:
        if "analysis" in f.keys():
            del f["analysis"]
        f.create_group("analysis/tmpl")
        ID = f["ID"][:]
        t,p = pulse_def()
        for reg in range(0,len(ID)-1):
            print("Resion "+str(reg)+" pulse selection")
            ph_mask = f["mask/Am241/reg"+str(reg)][:]
            ofs_mask = f["mask/offset/reg"+str(reg)][:]
            mask = (ph_mask) & (ofs_mask)
            up = p[ID[reg]:ID[reg+1]]
            print(len(up))
            print(len(mask))
            print(up.shape)
            print(len(up[mask]))
            f.create_dataset("analysis/tmpl/pulse/sel_pulse/reg"+str(reg),data=up[mask])
        
def tmpl_ansel(fn=h5_fn):
    fl = get_fl(s="n")
    with h5py.File(fn) as f:
        if "noise" in f["analysis/tmpl"].keys():
            del f["analysis/tmpl/noise"]
        ID = f["ID"][:]
        t,n = noise_def()
        for reg in range(0,len(ID)-1):        
            ph_mask = f["mask/Am241/reg"+str(reg)][:]
            ofs_mask = f["mask/offset/reg"+str(reg)][:]             
            mask = (ph_mask) & (ofs_mask)
            un = n[ID[reg]:ID[reg+1]]
            print(len(un))
            print(len(mask))
            print(un.shape)
            print(len(un[mask]))
            f.create_dataset("analysis/tmpl/noise/sel_noise/reg"+str(reg),data=un[mask])
        if "time" in f["data"].keys():
            del f["data/time"]
        f.create_dataset("data/time",data=t)
        
def tmpl_sn(ms="Am241",fn=h5_fn):
    with h5py.File(fn) as f:
        if "template" in f["analysis/tmpl"].keys():
            del f["analysis/tmpl/template"]
            del f["analysis/tmpl/sn"]
        ID = f["ID"][:]
        for reg in range(0,len(ID)-1):
            n = f["analysis/tmpl/noise/sel_noise/reg"+str(reg)]
            ofs = np.median(n,axis=1)
            p = f["analysis/tmpl/pulse/sel_pulse/reg"+str(reg)]
            print(p)
            print(n)
            tmpl,sn = Filter.generate_template(p,n,max_shift=10)
            f.create_dataset("analysis/tmpl/template/reg"+str(reg),data=tmpl)
            f.create_dataset("analysis/tmpl/sn/reg"+str(reg),data=sn)

def pha_aopt(fn=h5_fn):
    fl = get_fl(s="p")
    with h5py.File(fn) as f:
        if "pha" in f["analysis"].keys():
            del f["analysis/pha"]
            del f["analysis/ps"]
        ID = f["ID"][:]
        t,p=pulse_def()
        for reg in range(0,len(ID)-1):
            up = p[ID[reg]:ID[reg+1]]
            tmpl = f["analysis/tmpl/template/reg"+str(reg)]
            pha, ps = Filter.optimal_filter(up, tmpl, max_shift=10)
            f.create_dataset("analysis/pha/reg"+str(reg),data=pha)
            f.create_dataset("analysis/ps/reg"+str(reg),data=ps)
        
def gen_tmpl(fn=h5_fn):
    print("Start Pulse Selection")
    tmpl_apsel(fn=h5_fn)
    print("Start Noise Selection")    
    tmpl_ansel(fn=h5_fn)
    print("Generate Template")
    tmpl_sn(fn=h5_fn)
    print("Optimal Filter")
    pha_aopt(fn=h5_fn)
    print("End tmpl process!!")
                            
##PLOT DATASET
    
def ofs_hist():
    with h5py.File(fn) as f:
        ofs = f["data/offset"]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.hist(ofs,bins=1000)
        ax.set_title("offset distribution",fontsize=16)
        ax.set_xlabel("Offset[V]",fontsize=16)
        ax.set_ylabel("Counts",fontsize=16)
        fig.savefig("./graph/offset_hist.png",dpi=300)
        plt.show()
        
def ofs_plot(fn=h5_fn,sig=3,reg=1):
    with h5py.File(fn) as f:
        ofs = f["data/reg"+str(reg)+"/offset"][:]
        mask = f["mask/reg"+str(reg)+"/omask"][:]
        ofs_avg = f["data/reg"+str(reg)+"/offset_all_average"][...]
        ofs_std = f["data/reg"+str(reg)+"/offset_all_error"][...]
        hh = ofs_avg + sig * ofs_std
        hl = ofs_avg - sig * ofs_std
        print("offset average = ",ofs_avg)
        print("offset error = ",ofs_std)
        print("reduction range = ",sig,"sigma")
        print("data number = ",len(ofs))
        print("after reduction number = ",len(ofs[mask]))
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.plot(ofs,".")
        ax.plot(ofs[mask],".")
        ax.hlines(hh,0,len(ofs),color="red", linestyles='dashed')
        ax.hlines(hl,0,len(ofs),color="red", linestyles='dashed')
        ax.set_title("Offset distribution",fontsize=16)
        ax.set_xlabel("Index",fontsize=16)
        ax.set_ylabel("Offset average[V]",fontsize=16)
        savename = "./graph/offset_index_reg"+str(reg)+".png"
        fig.savefig(savename,dpi=300)
        plt.show()
        
def pha_ofs_lin(fn=h5_fn,bs=1000,secor=False,ran=(1.45,1.4575)):
    with h5py.File(fn) as f:
        ID = f["ID"][:]
        fig = plt.figure(figsize = (16,8))
        gs = GridSpec(1,2,width_ratios=(1,4))
        for reg in range(0,len(ID)-1):
            if secor == True:
                pha = f["analysis/se_cor/pha/"+str(reg)][:]
            else:
                pha = f["analysis/pha"][:]
            ofs = f["data/reg"+str(reg)+"/offset"][:]
            ofs_mask = f["mask/offset/reg"+str(reg)][:]
            ph_mask = (ran[0]<pha) & (pha<ran[1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1],sharey=ax1)
            ax1.hist(pha[ph_mask],orientation="horizontal", bins=bs, histtype="step")
            ax1.invert_xaxis()
            ax1.set_ylabel("Pulse height analysis",fontsize=16)
            ax2 = plt.subplot(gs[1],sharey=ax1)
            ax2.plot(ofs[ofs_mask][ph_mask], pha[ph_mask], ".")
            ax2.set_title("Offset vs PHA",fontsize=16)
            ax2.set_xlabel("Offset[V]",fontsize=16)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.subplots_adjust(wspace=0.1)
            if secor == True:
                fig.savefig("./graph/pha_ofs_lin_secor.png",dpi=300)
            else:
                fig.savefig("./graph/pha_ofs_lin.png",dpi=300)
            plt.show()
            print("reg"+str(reg))
            print(np.max(pha[ph_mask]))

def pha_ofs_lin2(fn=h5_fn,bs=1000,secor=False,ran=(1.45,1.4575)):
    with h5py.File(fn) as f:
        ID = f["ID"][:]
        fig = plt.figure(figsize = (16,8))
        gs = GridSpec(1,2,width_ratios=(1,4))
        pha1 = f["analysis/se_cor/pha/0"][:]*26.3/1.4574566430798386
        pha2 = f["analysis/se_cor/pha/1"][:]*26.3/1.4574161802505683
        pha3 = f["analysis/se_cor/pha/2"][:]*26.3/1.4587
        #ofs = f["data/reg"+str(reg)+"/offset"][:]
        #ofs_mask = f["mask/offset/reg"+str(reg)][:]
        plt.hist(pha1, bins=bs, histtype="step")
        plt.hist(pha2, bins=bs, histtype="step")
        plt.hist(pha3, bins=bs, histtype="step")
        plt.show()
        #print("reg"+str(reg))
        print(np.max(pha[ph_mask]))

def ph_ofs_lin(fn=h5_fn,bs=1000,secor=False):
    with h5py.File(fn) as f:
        ID = f["ID"][:]
        fig = plt.figure(figsize = (16,8))
        gs = GridSpec(1,2,width_ratios=(1,4))
        for reg in range(0,len(ID)-1):
            ph = f["data/reg"+str(reg)+"/ph"][:]
            ofs = f["data/reg"+str(reg)+"/offset"][:]
            ofs_mask = f["mask/offset/reg"+str(reg)][:]
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1],sharey=ax1)
            ax1.hist(ph,orientation="horizontal", bins=bs, histtype="step")
            ax1.invert_xaxis()
            ax1.set_ylabel("Pulse height",fontsize=16)
            ax2 = plt.subplot(gs[1],sharey=ax1)
            ax2.plot(ofs[ofs_mask], ph[ofs_mask], ".")
            #,markersize=4
            ax2.set_title("Offset vs Pulse Height",fontsize=16)
            ax2.set_xlabel("Offset[V]",fontsize=16)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.subplots_adjust(wspace=0.1)
            if secor == True:
                fig.savefig("./graph/pha_ofs_lin_secor.png",dpi=300)
            else:
                fig.savefig("./graph/pha_ofs_lin.png",dpi=300)
            plt.show()

##PULSE ANALYSIS

    
def ph_hist(fn=h5_fn,bs=500,ran=None,vv=None,reg=1):
    with h5py.File(fn) as f:
        ph = f["data/reg"+str(reg)+"/ph"][:]
        omasks = f["mask/reg"+str(reg)+"/omask"][:]
        ph = ph[omasks]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.hist(ph,range=ran,bins=bs)
        if vv is not None:
            ax.vlines(vv,0,30000,color="red", linestyles='dashed')
        ax.set_title("Pulse height distribution",fontsize=16)
        ax.set_xlabel("Pulse height[V]",fontsize=16)
        ax.set_ylabel("Counts",fontsize=16)
        fig.savefig("./graph/ph_hist_reg"+str(reg)+".png",dpi=300)
        plt.show()

def pha_hist(fn=h5_fn,bs=500,ran=None,vv=None,x="ene",se=False,LID=False,reg=1):
    with h5py.File(fn) as f:
        if se == True:
            pha = f["analysis/se/pha/reg"+str(reg)][:]
        else:
            pha = f["analysis/pha/reg"+str(reg)][:]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        if x == "pha":
            ax.hist(pha,range=ran,bins=bs)
            ax.set_xlabel("Pulse height amplitude",fontsize=16)
        else:
            ax.hist(pha*26.3446/1.42446,range=(ran),bins=bs)
            ax.set_xlabel("Energy[keV]",fontsize=16)
        if vv is not None:
            ax.vlines(vv,0,30000,color="red", linestyles='dashed')
        ax.set_title("Pulse height amplitude distribution",fontsize=16)
        ax.set_ylabel("Counts",fontsize=16)
        ax.set_xlim(ran)
        if LID == True:
            ID,E = line_ID()
            ax2 = ax.twiny()
            ax2.set_xticks(E)
            ax2.set_xticklabels(ID,fontsize=8)
            ax2.tick_params(direction="in",which="major",top=True,bottom=False,labeltop=True)
            ax2.set_xlim(ran)
        fig.savefig("./graph/pha_hist.png",dpi=300)
        plt.show()


def tmpl_plot(fn=h5_fn,reg=1):
    with h5py.File(fn) as f:
        t = f["data/time"][:]
        tmpl = f["analysis/tmpl/template/reg"+str(reg)][:]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.plot(t,tmpl)
        ax.set_title("Template",fontsize=16)
        ax.set_xlabel("Time[s]",fontsize=16)
        ax.set_ylabel("Template (arb. unit)",fontsize=16)
        fig.savefig("./graph/template.png",dpi=300)
        plt.show()

def cal_se(fn=h5_fn,pa=[1.5,1.7],all=False,sr=[-0.4,-0.1]):
    with h5py.File(fn) as f:
        ID = f["ID"][:]
        if all == True:
            pa = [0,np.max(pha)+1]
            if "se_all" in f["analysis"].keys():
                del f["analysis/se_all"]
        else:
            if "se" in f["analysis"].keys():
                del f["analysis/se"]
        for reg in range(0,len(ID)-1):
            ofs_mask = f["mask/offset/reg"+str(reg)][:]
            pha = f["analysis/pha/reg"+str(reg)][:][ofs_mask]
            ofs = f["data/reg"+str(reg)+"/offset"][:][ofs_mask]
            alpha_list = np.arange(sr[0],sr[1],0.001)
            bs = int(round(1e+5/np.max(pha),0))
            fs_pha,b_pha = np.histogram(pha,bins=bs)
            H = []
            for alpha in alpha_list:
                Ha = []
                pha_e = pha*(1+alpha*(ofs - np.median(ofs)))
                fs,b = np.histogram(pha_e,bins=b_pha)
                bin_mask = (pa[0] <= b) & (b <= pa[1])
                fs_r = fs[bin_mask[:-1]]
                ad = np.sum(fs_r)
                bn = b[bin_mask]
                for i in range(0,len(bn)-1):
                    if fs_r[i]>0:
                        Ha.append(-fs_r[i]/ad*np.log2(fs_r[i]/ad))
                Hs = np.sum(Ha)
                H.append(Hs)
                print(Hs)
            ma = alpha_list[np.argmin(H)]
            H = np.array(H)
            print(ma,np.min(H))
            pha_e = pha*(1+float(ma)*(ofs - np.median(ofs)))
            fig = plt.figure(figsize=(8.0,6.0))
            ax = fig.add_subplot(111)
            ax.plot(alpha_list,H,".")
            ax.set_title("Spectral Entropy",fontsize=16)
            ax.set_xlabel("α",fontsize=16)
            ax.set_ylabel("H(α)",fontsize=16)
            boxdic = {
            "facecolor" : "white",
            "edgecolor" : "black",
            "boxstyle" : "Round",
            "linewidth" : 1
            }
            fig.text(0.72, 0.84, 'phareg=%.1f to %.1f' %(pa[0],pa[1]), ha='left',fontsize=16,bbox=boxdic)
            fig.text(0.72, 0.78, 'Hmin=%.4f' %(np.min(H)), ha='left',fontsize=16,bbox=boxdic)
            fig.text(0.72, 0.72, 'αmin=%.4f' %(ma), ha='left',fontsize=16,bbox=boxdic)
            f.create_dataset("analysis/se/H/reg"+str(reg),data=H)
            f.create_dataset("analysis/se/ma/reg"+str(reg),data=ma)
            f.create_dataset("analysis/se/pha/reg"+str(reg),data=pha_e)
            f.create_dataset("analysis/se/cal_range/reg"+str(reg),data=pa)
            savenaem="./graph/se_reg"+str(reg)+".png"
            fig.savefig(savenaem,dpi=300)
            plt.show()
            print(ma)
        
        
        
def se_plt(fn=h5_fn):
    with h5py.File(fn) as f:
        H_all = f["analysis/se_all/H"][:]
        H = f["analysis/se/H"][:]
        alpha_list = np.arange(-0.4,-0.1,0.001)
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.plot(alpha_list,H_all,".",label="H all data")
        ax.plot(alpha_list,H,".",label="H")
        ax.set_title("Spectral Entropy",fontsize=16)
        ax.set_xlabel("α",fontsize=16)
        ax.set_ylabel("H(α)",fontsize=16)
        fig.savefig("./graph/se.com.png",dpi=300)
        plt.show()

def pha_secor(fn=h5_fn):
    with h5py.File(fn) as f:
        ID = f["ID"][:]
        if "se_cor" in f["analysis"].keys():
            del f["analysis/se_cor"]
        for reg in range(0,len(ID)-1):        
            pha = f["analysis/pha/reg"+str(reg)][:]
            ofs = f["data/reg"+str(reg)+"/offset"][:]
            ofs_mask = f["mask/offset/reg"+str(reg)][:]
            ma = f["analysis/se/ma/reg"+str(reg)][...]
            pha_secor = pha[ofs_mask]*(1+ma*(ofs[ofs_mask] - np.median(ofs[ofs_mask])))
            f.create_dataset("analysis/se_cor/pha/"+str(reg),data=pha_secor)
            f.create_dataset("analysis/se_cor/ma/"+str(reg),data=ma)
            print(ma)
    
## Option ##

def line_ID():
    f = np.genfromtxt("/Users/tanakakeita/python_script/Th229_lineID.txt",dtype="str")
    E = np.array(f[:,0],dtype="float")
    ID = np.array(f[:,1])
    print(ID,E)
    return ID,E

def data_inf():
    fl=get_fl(s="p")
    for fn in fl:
        t,p=Util.fopen(fn)
        print("---------------------------------------")
        print("File name = ",fn)
        print("Data number = ",len(p))
        print("Time interval = ",t[1]-t[0],"[s]")


# Xspec Analysis

def make_resp(rmfname='test.rmf', bmin=1000., bmax=30000.,binsize=1.):
    bin_min = bmin/1e3
    bin_max = bmax/1e3
    resp_param = 'genrsp',\
     'inrfil=none',\
     'rmffil='+rmfname,\
     'resol_reln=constant',\
     'resol_file=no',\
     'fwhm=0.0001',\
     'disperse=no',\
     'tlscpe=DUMMY',\
     'instrm=DUMMY',\
     'resp_reln=linear',\
     'resp_low='+str(bin_min),\
     'resp_high='+str(bin_max),\
     'resp_number='+str(int((bin_max-bin_min)/binsize)),\
     'chan_reln=linear',\
     'chan_low='+str(bin_min),\
     'chan_high='+str(bin_max),\
     'chan_number='+str(int((bin_max-bin_min)/binsize)),\
     'efffil=none',\
     'detfil=none',\
     'filfil=none',\
     'max_elements=1000000'
    
    resp_param = np.asarray(resp_param)
    subprocess.call(resp_param)

def make_resp_k(rmfname='test.rmf', bmin=0., bmax=30.,binsize=1000.):
    bin_min = bmin
    bin_max = bmax
    resp_param = 'genrsp',\
     'inrfil=none',\
     'rmffil='+rmfname,\
     'resol_reln=constant',\
     'resol_file=no',\
     'fwhm=0.0001',\
     'disperse=no',\
     'tlscpe=DUMMY',\
     'instrm=DUMMY',\
     'resp_reln=linear',\
     'resp_low='+str(bin_min),\
     'resp_high='+str(bin_max),\
     'resp_number='+str(int(binsize)),\
     'chan_reln=linear',\
     'chan_low='+str(bin_min),\
     'chan_high='+str(bin_max),\
     'chan_number='+str(int(binsize)),\
     'efffil=none',\
     'detfil=none',\
     'filfil=none',\
     'max_elements=1000000'
    
    resp_param = np.asarray(resp_param)
    subprocess.call(resp_param)

def histogram(pha, binsize=1.0):
    """
    Create histogram
    
    Parameter:
        pha:        pha data (array-like)
        binsize:    size of bin in eV (Default: 1.0 eV)
    
    Return (n, bins)
        n:      photon count
        bins:   bin edge array
    
    Note:
        - bin size is 1eV/bin.
    """
    
    # Create histogram
    bins = np.arange(np.floor(pha.min()), np.ceil(pha.max())+binsize, binsize)
    n, bins = np.histogram(pha, bins=bins)
    
    return n, bins


def fits2xspec(binsize=1, exptime=1, fwhm=0.0001, gresp=False, garf=False, name='test', arfname='test.arf', TEStype='TMU524', Datatype='PHA', chan='ch65',gen=True,inst=False):
    
    with h5py.File(h5_fn) as f:
        ID = f["ID"][:]
        for reg in range(0,len(ID)-1):
            if gen == True:
                #pha = f["energy/se"][:]*1e+3
                #pha = f["energy/gen"][:]*1e+3
                pha = f["allpha"][:]*26.3*1e+3/1.45
            else:
                pha = f["analysis/se/pha/reg"+str(reg)][:]*26.3*1e+3/1.45
                #pha = f["inst/pha"][:]*1e+4
            # separate bins
            if Datatype=='PHA':
                n, bins = histogram(pha[pha>0], binsize=binsize)
            else:
                n, bins = histogram(pha[pha>0.05], binsize=binsize)
            # py.figure()
            # py.hist(pha, bins=bins, histtype='stepfilled', color='k')
            # py.show()

            # par of fits
            filename = name + "reg" + str(reg) + ".fits"
            rmfname = name + "reg" + str(reg) + ".rmf"
            Exposuretime = exptime
            tc = int(n.sum())
            chn = len(bins)-1
            #x = (bins[:-1]+bins[1:])/2
            x = np.arange(0, (len(bins)-1), 1)
            y = n

            # print('#fits file name : %s' %(filename))
            # print('#info to make responce')
            # print('genrsp')
            # print('none')
            # print('%s' %rmfname)
            # print('constant')
            # print('no')
            # print('%f' %fwhm)
            # print('linear')
            # print( '%.3f' %(bins.min()/1e3))
            # print( '%.3f' %(bins.max()/1e3))
            # print( '%d' %(int((bins.max()-bins.min())/binsize)))
            # print( 'linear')
            # print( '%.3f' %(bins.min()/1e3))
            # print( '%.3f' %(bins.max()/1e3))
            # print( '%d' %(int((bins.max()-bins.min())/binsize)))
            # print( '5000000')
            # print( 'DUMMY')
            # print( 'DUMMY')
            # print( 'none')
            # print( 'none')
            # print( 'none')
            # print( '\n')
            
            resp_param = 'genrsp',\
             'inrfil=none',\
             'rmffil='+rmfname,\
             'resol_reln=constant',\
             'resol_file=no',\
             'fwhm='+str(fwhm),\
             'disperse=no',\
             'tlscpe=DUMMY',\
             'instrm=DUMMY',\
             'resp_reln=linear',\
             'resp_low='+str(bins.min()/1.e3),\
             'resp_high='+str(bins.max()/1.e3),\
             'resp_number='+str(int((bins.max()-bins.min())/binsize)),\
             'chan_reln=linear',\
             'chan_low='+str(bins.min()/1.e3),\
             'chan_high='+str(bins.max()/1.e3),\
             'chan_number='+str(int((bins.max()-bins.min())/binsize)),\
             'efffil=none',\
             'detfil=none',\
             'filfil=none',\
             'max_elements=5000000'
            
            resp_param = np.asarray(resp_param)
            while True:
                try:
                    subprocess.call(resp_param)
                    break
                except OSError:
                    print('Please install HEASOFT')
            
            if gresp==True:
                pass
                # if TEStype == 'Th229':
                #     mrTh._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
                # else:
                #     mr._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
            
            if garf==True:
                arf._make_arf_(pha, binsize=1, aname=arfname, chan=chan)

            # make fits
            col_x = pf.Column(name='CHANNEL', format='J', array=np.asarray(x))
            col_y = pf.Column(name='COUNTS', format='J', unit='count', array=np.asarray(y))
            cols  = pf.ColDefs([col_x, col_y])
            tbhdu = pf.BinTableHDU.from_columns(cols)

            exthdr = tbhdu.header
            exthdr['XTENSION'] = ('BINTABLE', 'binary table extension')
            exthdr['EXTNAME']  = ('SPECTRUM', 'name of this binary table extension')
            exthdr['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
            exthdr['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
            exthdr['HDUVERS1'] = ('1.2.0', 'Obsolete - included for backwards compatibility')
            exthdr['HDUVERS']  = ('1.2.0', 'Version of format (OGIP memo OGIP-92-007)')
            exthdr['HDUCLAS2'] = ('TOTAL', 'Gross PHA Spectrum (source + bkgd)')
            exthdr['HDUCLAS3'] = ('COUNT', 'PHA data stored as Counts (not count/s)')
            exthdr['TLMIN1']   = (0, 'Lowest legal channel number')
            exthdr['TLMAX1']   = (chn-1, 'Highest legal channel number')
            exthdr['TELESCOP'] = ('TES', 'Telescope (mission) name')
            exthdr['INSTRUME'] = (TEStype, 'Instrument name')
            exthdr['FILTER']   = ('NONE', 'no filter in use')
            exthdr['EXPOSURE'] = (exptime, 'Exposure time')
            exthdr['AREASCAL'] = (1.000000E+00, 'area scaling factor') #??
            exthdr['BACKFILE'] = ('none', 'associated background filename')
            exthdr['BACKSCAL'] = (1, 'background file scaling factor')
            exthdr['CORRFILE'] = ('none', 'associated correction filename')
            exthdr['CORRSCAL'] = (1.000000E+00, 'correction file scaling factor')
            exthdr['RESPFILE'] = ('none', 'associated redistrib matrix filename')
            exthdr['ANCRFILE'] = ('none', 'associated ancillary response filename')
            exthdr['PHAVERSN'] = ('1992a', 'obsolete')
            exthdr['DETCHANS'] = (chn, 'total number possible channels')
            exthdr['CHANTYPE'] = ('PI', 'channel type (PHA, PI etc)')
            exthdr['POISSERR'] = (bool(True), 'Poissonian errors to be assumed')
            exthdr['STAT_ERR'] = (0, 'no statistical error specified')
            exthdr['SYS_ERR']  = (0, 'no systematic error specified')
            exthdr['GROUPING'] = (0, 'no grouping of the data has been defined')
            exthdr['QUALITY']  = (0, 'no data quality information specified')
            #HISTORY  FITS SPECTRUM extension written by WTPHA2 1.0.1
            exthdr['DATAMODE'] = ('STANDARD', 'Datamode')
            exthdr['OBJECT']   = ('PERSEUS CLUSTER', 'Name of observed object')
            exthdr['ONTIME']   = (exptime, 'On-source time')
            exthdr['LIVETIME'] = (exptime, 'On-source time')
            exthdr['DATE-OBS'] = ('2006-08-29T18:55:07', 'Start date of observations')
            exthdr['DATE-END'] = ('2006-09-02T01:54:19', 'End date of observations')
            exthdr['TSTART']   = (0, 'start time of experiment in total second')
            exthdr['TSTOP']    = (0, 'end time of experiment in total second')
            exthdr['TELAPSE']  = (exptime, 'elapsed time')
            exthdr['MJD-OBS']  = (exptime, 'MJD of data start time')
            exthdr['MJDREFI']  = (51544, 'MJD reference day')
            exthdr['MJDREFF']  = (7.428703703703700E-04, 'MJD reference (fraction of day)')
            exthdr['TIMEREF']  = ('LOCAL', 'reference time')
            exthdr['TIMESYS']  = ('TT', 'time measured from')
            exthdr['TIMEUNIT'] = ('s', 'unit for time keywords')
            exthdr['EQUINOX']  = (2.000E+03, 'Equinox of celestial coord system')
            exthdr['RADECSYS'] = ('FK5', 'celestial coord system')
            # exthdr['USER']     = ('tasuku', 'User name of creator')
            exthdr['FILIN001'] = ('PerCluster_work1001.xsl', 'Input file name')
            exthdr['FILIN002'] = ('PerCluster_work1002.xsl', 'Input file name')
            exthdr['CREATOR']  = ('extractor v5.23', 'Extractor')
            exthdr['DATE']     = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')
            exthdr['ORIGIN']   = ('NASA/GSFC', 'origin of fits file')
            exthdr['TOTCTS']   = (tc, 'Total counts in spectrum')
            exthdr['RA_PNT']   = (4.993100000000000E+01, 'File average of RA(degrees)')
            exthdr['DEC_PNT']  = (4.152820000000000E+01, 'File average of DEC(degrees)')
            exthdr['SPECDELT'] = (1, 'Binning factor for spectrum')
            exthdr['SPECPIX']  = (0, 'The rebinned channel corresponding to SPECVAL')
            exthdr['SPECVAL']  = (0.000000000000000E+00, 'Original channel value at center of SPECPIX')
            exthdr['DSTYP1']   = ('GRADE', 'Data subspace descriptor: name')
            exthdr['DSVAL1']   = ('0:11', 'Data subspace descriptor: value')
            exthdr['DSTYP2']   = (Datatype, 'Data subspace descriptor: name')
            exthdr['DSVAL2']   = ('0:4095', 'Data subspace descriptor: value')
            exthdr['DSTYP3']   = ('POS(X,Y)', 'Data subspace descriptor: name')
            exthdr['DSREF3']   = (':REG00101', 'Data subspace descriptor: reference')
            exthdr['DSVAL3']   = ('TABLE', 'Data subspace descriptor: value')
            #HISTORY extractor v5.23
            exthdr['CHECKSUM'] = ('54fW64ZV54dV54ZV', 'HDU checksum updated 2014-04-07T08:16:02')
            exthdr['DATASUM']  = ('9584155', 'data unit checksum updated 2014-04-07T08:16:02')

            hdu = pf.PrimaryHDU()
            thdulist = pf.HDUList([hdu, tbhdu])
            thdulist.writeto(filename, clobber=False)

def fits2xspec2(binsize=1, exptime=1, fwhm=0.0001, gresp=False, garf=False, name='test', arfname='test.arf', TEStype='TMU524', Datatype='PHA', chan='ch65',h=True,pha=None): 
    with h5py.File(h5_fn) as f:
        if h == True:
            pha = pha * 26.3*1e+3/1.45

        else :
            pha = f["energy/gen"][:]*1e+3
        # separate bins
        if Datatype=='PHA':
            n, bins = histogram(pha[pha>0], binsize=binsize)
        else:
            n, bins = histogram(pha[pha>0.05], binsize=binsize)
        # py.figure()
        # py.hist(pha, bins=bins, histtype='stepfilled', color='k')
        # py.show()

        # par of fits
        filename = name + ".fits"
        rmfname = name + ".rmf"
        Exposuretime = exptime
        tc = int(n.sum())
        chn = len(bins)-1
        #x = (bins[:-1]+bins[1:])/2
        x = np.arange(0, (len(bins)-1), 1)
        y = n

        # print('#fits file name : %s' %(filename))
        # print('#info to make responce')
        # print('genrsp')
        # print('none')
        # print('%s' %rmfname)
        # print('constant')
        # print('no')
        # print('%f' %fwhm)
        # print('linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( 'linear')
        # print( '%.3f' %(bins.min()/1e3))
        # print( '%.3f' %(bins.max()/1e3))
        # print( '%d' %(int((bins.max()-bins.min())/binsize)))
        # print( '5000000')
        # print( 'DUMMY')
        # print( 'DUMMY')
        # print( 'none')
        # print( 'none')
        # print( 'none')
        # print( '\n')
        
        resp_param = 'genrsp',\
         'inrfil=none',\
         'rmffil='+rmfname,\
         'resol_reln=constant',\
         'resol_file=no',\
         'fwhm='+str(fwhm),\
         'disperse=no',\
         'tlscpe=DUMMY',\
         'instrm=DUMMY',\
         'resp_reln=linear',\
         'resp_low='+str(bins.min()/1.e3),\
         'resp_high='+str(bins.max()/1.e3),\
         'resp_number='+str(int((bins.max()-bins.min())/binsize)),\
         'chan_reln=linear',\
         'chan_low='+str(bins.min()/1.e3),\
         'chan_high='+str(bins.max()/1.e3),\
         'chan_number='+str(int((bins.max()-bins.min())/binsize)),\
         'efffil=none',\
         'detfil=none',\
         'filfil=none',\
         'max_elements=5000000'
        
        resp_param = np.asarray(resp_param)
        while True:
            try:
                subprocess.call(resp_param)
                break
            except OSError:
                print('Please install HEASOFT')
        
        if gresp==True:
            pass
            # if TEStype == 'Th229':
            #     mrTh._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
            # else:
            #     mr._make_resp_(pha, binsize=binsize, elements=1651, rname=rmfname)
        
        if garf==True:
            arf._make_arf_(pha, binsize=1, aname=arfname, chan=chan)

        # make fits
        col_x = pf.Column(name='CHANNEL', format='J', array=np.asarray(x))
        col_y = pf.Column(name='COUNTS', format='J', unit='count', array=np.asarray(y))
        cols  = pf.ColDefs([col_x, col_y])
        tbhdu = pf.BinTableHDU.from_columns(cols)

        exthdr = tbhdu.header
        exthdr['XTENSION'] = ('BINTABLE', 'binary table extension')
        exthdr['EXTNAME']  = ('SPECTRUM', 'name of this binary table extension')
        exthdr['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
        exthdr['HDUCLAS1'] = ('SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)')
        exthdr['HDUVERS1'] = ('1.2.0', 'Obsolete - included for backwards compatibility')
        exthdr['HDUVERS']  = ('1.2.0', 'Version of format (OGIP memo OGIP-92-007)')
        exthdr['HDUCLAS2'] = ('TOTAL', 'Gross PHA Spectrum (source + bkgd)')
        exthdr['HDUCLAS3'] = ('COUNT', 'PHA data stored as Counts (not count/s)')
        exthdr['TLMIN1']   = (0, 'Lowest legal channel number')
        exthdr['TLMAX1']   = (chn-1, 'Highest legal channel number')
        exthdr['TELESCOP'] = ('TES', 'Telescope (mission) name')
        exthdr['INSTRUME'] = (TEStype, 'Instrument name')
        exthdr['FILTER']   = ('NONE', 'no filter in use')
        exthdr['EXPOSURE'] = (exptime, 'Exposure time')
        exthdr['AREASCAL'] = (1.000000E+00, 'area scaling factor') #??
        exthdr['BACKFILE'] = ('none', 'associated background filename')
        exthdr['BACKSCAL'] = (1, 'background file scaling factor')
        exthdr['CORRFILE'] = ('none', 'associated correction filename')
        exthdr['CORRSCAL'] = (1.000000E+00, 'correction file scaling factor')
        exthdr['RESPFILE'] = ('none', 'associated redistrib matrix filename')
        exthdr['ANCRFILE'] = ('none', 'associated ancillary response filename')
        exthdr['PHAVERSN'] = ('1992a', 'obsolete')
        exthdr['DETCHANS'] = (chn, 'total number possible channels')
        exthdr['CHANTYPE'] = ('PI', 'channel type (PHA, PI etc)')
        exthdr['POISSERR'] = (bool(True), 'Poissonian errors to be assumed')
        exthdr['STAT_ERR'] = (0, 'no statistical error specified')
        exthdr['SYS_ERR']  = (0, 'no systematic error specified')
        exthdr['GROUPING'] = (0, 'no grouping of the data has been defined')
        exthdr['QUALITY']  = (0, 'no data quality information specified')
        #HISTORY  FITS SPECTRUM extension written by WTPHA2 1.0.1
        exthdr['DATAMODE'] = ('STANDARD', 'Datamode')
        exthdr['OBJECT']   = ('PERSEUS CLUSTER', 'Name of observed object')
        exthdr['ONTIME']   = (exptime, 'On-source time')
        exthdr['LIVETIME'] = (exptime, 'On-source time')
        exthdr['DATE-OBS'] = ('2006-08-29T18:55:07', 'Start date of observations')
        exthdr['DATE-END'] = ('2006-09-02T01:54:19', 'End date of observations')
        exthdr['TSTART']   = (0, 'start time of experiment in total second')
        exthdr['TSTOP']    = (0, 'end time of experiment in total second')
        exthdr['TELAPSE']  = (exptime, 'elapsed time')
        exthdr['MJD-OBS']  = (exptime, 'MJD of data start time')
        exthdr['MJDREFI']  = (51544, 'MJD reference day')
        exthdr['MJDREFF']  = (7.428703703703700E-04, 'MJD reference (fraction of day)')
        exthdr['TIMEREF']  = ('LOCAL', 'reference time')
        exthdr['TIMESYS']  = ('TT', 'time measured from')
        exthdr['TIMEUNIT'] = ('s', 'unit for time keywords')
        exthdr['EQUINOX']  = (2.000E+03, 'Equinox of celestial coord system')
        exthdr['RADECSYS'] = ('FK5', 'celestial coord system')
        # exthdr['USER']     = ('tasuku', 'User name of creator')
        exthdr['FILIN001'] = ('PerCluster_work1001.xsl', 'Input file name')
        exthdr['FILIN002'] = ('PerCluster_work1002.xsl', 'Input file name')
        exthdr['CREATOR']  = ('extractor v5.23', 'Extractor')
        exthdr['DATE']     = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')
        exthdr['ORIGIN']   = ('NASA/GSFC', 'origin of fits file')
        exthdr['TOTCTS']   = (tc, 'Total counts in spectrum')
        exthdr['RA_PNT']   = (4.993100000000000E+01, 'File average of RA(degrees)')
        exthdr['DEC_PNT']  = (4.152820000000000E+01, 'File average of DEC(degrees)')
        exthdr['SPECDELT'] = (1, 'Binning factor for spectrum')
        exthdr['SPECPIX']  = (0, 'The rebinned channel corresponding to SPECVAL')
        exthdr['SPECVAL']  = (0.000000000000000E+00, 'Original channel value at center of SPECPIX')
        exthdr['DSTYP1']   = ('GRADE', 'Data subspace descriptor: name')
        exthdr['DSVAL1']   = ('0:11', 'Data subspace descriptor: value')
        exthdr['DSTYP2']   = (Datatype, 'Data subspace descriptor: name')
        exthdr['DSVAL2']   = ('0:4095', 'Data subspace descriptor: value')
        exthdr['DSTYP3']   = ('POS(X,Y)', 'Data subspace descriptor: name')
        exthdr['DSREF3']   = (':REG00101', 'Data subspace descriptor: reference')
        exthdr['DSVAL3']   = ('TABLE', 'Data subspace descriptor: value')
        #HISTORY extractor v5.23
        exthdr['CHECKSUM'] = ('54fW64ZV54dV54ZV', 'HDU checksum updated 2014-04-07T08:16:02')
        exthdr['DATASUM']  = ('9584155', 'data unit checksum updated 2014-04-07T08:16:02')

        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu, tbhdu])
        thdulist.writeto(filename, clobber=False)



##XSPEC ANALYSIS BY PYXSPEC

def load_spec(fn="data1_ch4_20_include.pi",ran=[0.0,10.0]):
    AllData.clear()
    s=Spectrum(fn)
    s.ignore("**-"+str(ran[0])+" "+str(ran[1])+"-**")
    Plot.xAxis="keV"
    Plot("data")
    xs=Plot.x(1,1)
    ys=Plot.y(1,1)
    xe=Plot.xErr(1,1)
    ye=Plot.yErr(1,1)
    return s,xs,ys,xe,ye

def set_plot():
    xs=np.array(Plot.x(1,1))
    ys=np.array(Plot.y(1,1))
    xe=np.array(Plot.xErr(1,1))
    ye=np.array(Plot.yErr(1,1))
    return xs,ys,xe,ye

def plot_spec(fn="data1_ch4_20_include.pi",ran=[1.0,10.0]):
    s,xs,ys,xe,ye = load_spec(fn=fn,ran=ran)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=1,color="black",label=fn)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlabel("PHA*1e+4[arb.unit]",fontsize=16)
    ax.set_ylabel("Counts",fontsize=16)
    plt.show()

def fit_spec(fn="data1_ch4_20_include.pi",ran=[1.0,10.0],fm="sl"):
    s,xs,ys,xe,ye = load_spec(fn=fn,ran=ran)
    m=Model("powerlaw+gsmooth*lorentz")
    if fm == "sinlo":
        m.lorentz.LineE.values = [16.025,1e-3,16.0,16.0,16.7,16.7]
        m.lorentz.Width.values = [1.5600e-2,-1,1e-10,1e-10,10,10]
        m.lorentz.norm.values = [1e+3,1,0,0,1e+20,1e+24]
    m.powerlaw.PhoIndex.values = [0.0,-1,-3,-2,9,10]
    m.powerlaw.norm.values = [229.44,1,0,0,1e+20,1e+24]
    m.gsmooth.Sig_6keV.values = [1.26482e-2,1,0,0,1,1]
    m.gsmooth.Index.values = [0.0,-1,-1,-1,1,1]
    Fit.query="yes"
    Plot.add=True
    Fit.perform()
    y=Plot.model()
    Fit.show()
    ys_comps=[]
    comp_N=1
    while(True):
        try:
            ys_tmp = Plot.addComp(comp_N,1,1)
            comp_N += 1
            # execlude components with only 0
            if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                continue
            ys_comps.append(ys_tmp)
        except:
            break
    fig=plt.figure(figsize=(8,6),dpi=300)
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=2,color="black",label="data")
    #ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_xlabel("Energy[keV]",fontsize=16)
    ax.set_ylabel("Normalized counts s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$",fontsize=ls)
    ax.plot(xs,y,"-",label="Fitting All",color="red")
    ax.plot(xs,ys_comps[0],label="CXB")
    ax.plot(xs,ys_comps[1],label="MWH")
    ax.legend(fontsize=14)
    #ax.set_ylim(1e-3,1)
    #ax.set_xlim(0.4,5.0)
    Plot("del")
    xres=Plot.x(1,1)
    yres=Plot.y(1,1)
    xres_e=Plot.xErr(1,1)
    yres_e=Plot.yErr(1,1)
    ax2=fig.add_subplot(gs2[:,:])
    ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=2,color="black",label="data")
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy[keV]",fontsize=ls)
    ax2.set_ylabel("(data-model)/error",fontsize=ls)
    ax2.set_xlim(0.4,5.0)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    print(m.name)


def spec_plotter(xs=None,ys=None,xe=None,ye=None,savename=None,ran=None):
    fig=plt.figure(figsize=(8,6))
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    y=Plot.model()
    ax.errorbar(xs,ys,yerr=ye,xerr=xe,fmt="o",markersize=2,color="black",label="data")
    #ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_xlabel("Energy[keV]",fontsize=16)
    ax.set_ylabel("counts/keV",fontsize=ls)
    ax.plot(xs,y,"-",label="Fitting All",color="red")
    ax.legend(fontsize=14)
    #ax.set_ylim(1e-3,1)
    if "ran" in locals():
        ax.set_xlim(ran[0],ran[1])
    Plot("del")
    xres=Plot.x(1,1)
    yres=Plot.y(1,1)
    xres_e=Plot.xErr(1,1)
    yres_e=Plot.yErr(1,1)
    ax2=fig.add_subplot(gs2[:,:])
    ax2.errorbar(xres,yres,yerr=yres_e,xerr=xres_e,fmt="o",markersize=2,color="black",label="data")
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy[keV]",fontsize=ls)
    ax2.set_ylabel("(data-model)/error",fontsize=ls)
    if "ran" in locals():
        ax2.set_xlim(ran[0],ran[1])
    #ax2.set_xlim(0.4,5.0)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    if not savename == None:
        fig.savefig(savename,dpi=300)
        
### MADE BY MACBOOKPRO IN LAB

def Conv(x,num=10):
    b = np.ones(num)/num
    conv = np.convolve(x,b,mode="same")
    return conv

def ofs_conv(fn=h5_fn):
    with h5py.File(fn) as f:
        ofs = f["data/offset"][:]
        conv = Conv(x=ofs,num=500)
        plt.plot(ofs)
        plt.plot(conv)
        print(len(conv),len(ofs))
    
def max_pha(fn=h5_fn,bs=1000,ran=(1.4,1.5),vv=None,x="ene",ID=True,se=True):
    with h5py.File("CH4_reg1.hdf5") as f:
        if se == True:
            pha = f["analysis/se/pha"][:]
        else:
            pha = f["analysis/pha"][:]
        fs,b = np.histogram(pha,bins=bs,range=ran)
        s = np.argmax(fs)
        cen=(b[s]+b[s+1])/2
        print(cen)
        pha_e = pha*26.3446/cen
        if "inst/pha" in f.keys():
            del f["inst/pha"]
        f.create_dataset("inst/pha",data=pha_e)
        
        plt.hist(pha,bins=bs,range=ran)
        plt.show()    
            
def ma_ll():
    with h5py.File("CH4_reg1-2.hdf5") as fl:
        with h5py.File("CH4_reg1.hdf5") as f:
            a = f["inst/pha"][:]
            with h5py.File("CH4_reg2.hdf5") as ft:
                a = np.append(a,ft["inst/pha"][:])
                
                fl.create_dataset("inst_pha",data=a)

# def _n_ftest_(stad, t, fp=0.9973, plotting=True, savefig=False, region='region1'):
#     '''
#     input:
#     n : noise N-array
    
#     output:
#     p_ : masked data
#     p_mask: data which makes pulse data 
#     '''
#     from scipy.stats import f
    
#     #calculate standard deviation
#     #stad = np.std(n, axis=-1)
#     stad = stad
#     # figure()
#     # hist(stad, bins=1024*40, histtype='step', color='k')
#     # show()
#     # repeat = 1
#     # while repeat == 1:
#     #     mn_l     = float(input('ph_left  : '))
#     #     mn_r     = float(input('ph_righr : '))
#     #     mn  = (mn_l<stad) & (stad<mn_r)
#     m = np.median(stad)
#     print 'standard divation median %f' %m
#     mn = (m*0.9999<stad)&(stad<m*1.0001)
#     ftest = (stad**2)/(stad[mn][0]**2)
#     figure()
#     hist(ftest, bins=1024*40, histtype='step', color='k')
#     #hist(stad**2/stad[mn][1]**2, bins=1024*40, histtype='step', color='k', ls='--')
#     dfn = len(stad)
#     dfd = len(stad)-1
#     axvline(f.ppf(0.6827, dfn, dfd), color='r')
#     axvline(f.ppf(0.9973, dfn, dfd), color='b')
#     axvline(f.ppf(0.9999994, dfn, dfd), color='g')
#     #legend([r'$68.27\%$', r'$99.73\%$', r'$99.99\%$'], fontsize=16)
#     ylabel(r'$\rm Counts$', fontsize=16)
#     xlabel(r'$F-distribution$', fontsize=16)
#     legend([r'$68.27\%$', r'$99.73\%$', r'$99.99\%$', r'$\nu_{2}=3.7635<\sigma^2\times10^6 <3.7636$'], fontsize=16)
#     xlim(0.92, 1.6)
#     tight_layout()

 

#     if plotting:
#         # figure()
#         # t = np.asarray(t)
#         # dt = np.diff(t)[0]
#         # df = (dt * t.shape[-1])**-1
#         # for i in np.array([1.0, 0.6827, 0.9973, 0.9999994]):
#         #     fmask = (stad**2)/(stad[mn][0]**2) < f.ppf(i, dfn, dfd)
#         #     avgns = np.sqrt(Filter.average_noise(n[fmask]) / df)
#         #     plot(np.arange(len(avgns))*df, avgns)
#         #     loglog()
#         #     xlabel(r'$\rm Frequency\ (Hz)$', fontsize=16)
#         #     ylabel(r'$\rm Noise\ (V/\sqrt{\mathrm{Hz}})$', fontsize=16)
#         # legend([r'$\rm All\ Data$', r'$68.27\%$', r'$99.73\%$', r'$99.99\%$'], fontsize=16)
#         # tight_layout()
#         figure()
#         x = np.asarray(range(len(stad)))
#         scatter(x, (stad**2)*1e6, s=0.3)
#         for i in np.array([0.6827, 0.9973, 0.9999994][::-1]):
#             fmask = (stad**2)/(stad[mn][0]**2) < f.ppf(i, dfn, dfd)
#             scatter(x[fmask], (stad[fmask]**2)*1e6, s=0.3)
#         legend([r'$\rm All\ Data$', r'$\rm F-test\ 99.99\%$', r'$\rm F-test\ 99.73\%$', r'$\rm F-test\ 68.27\%$'], fontsize=16, loc=1)    
#         xlabel(r'$\rm time\ series$', fontsize=16)
#         ylabel(r'$\rm \sigma^2\times 10^{-6}$', fontsize=16)
#         tight_layout()
    
#     #3sigma rejection
#     print 'fp = %.4f' %fp
#     fmask = ftest < f.ppf(fp, dfn, dfd)   
#     #n_ = n[fmask]

 

#     if savefig:
#         savefig('noise_level_standard_all_masked_data_%s.png' %region)
        
#     print 'rejection rate %.3f' %(len(stad[fmask])*1e2/float(len(stad)))
    
#     return fmask, ftest
                

def pha_set(fn=h5_fn):
        with h5py.File(fn) as f: 
            reg0 = f["analysis/se/pha/reg0"][:]*26.3639/26.3687
            reg1 = f["analysis/se/pha/reg1"][:]
            reg2 = f["analysis/se/pha/reg2"][:]*26.3639/26.3939

            allpha = reg0
            allpha = np.append(allpha,reg1)
            allpha = np.append(allpha,reg2)

            f.create_dataset("allpha",data=allpha)

def p_fit(fn=h5_fn):
    with h5py.File(fn) as f:
        plen = len(f["analysis/tmpl/pulse/sel_pulse"])
        st = []
        for i in range(0,plen):
            p = f["analysis/tmpl/pulse/sel_pulse"][i]
            n = f["analysis/tmpl/noise/sel_noise"][i]
            ofs = np.median(n)
            sig = np.std(n)
            p -= ofs
            t = f["data/time"][:]
            def pmod(t,tau_r,tau_f,A):
                t0 = t[np.where(-10*sig>p)[0][0] - 1]
                #print(t0)
                return A*(np.exp(-(t-t0)/tau_r)-np.exp(-(t-t0)/tau_f))*(t0 < t)
            popt,pcov = curve_fit(pmod,t,p,p0=(2e-5,3e-4,np.min(p)*(-1)+0.1))
            plt.plot(t,p,".")
            fit = pmod(t,*popt)
            plt.plot(t,fit)
            plt.show()
            print(i,popt[0])
            st.append(popt[0])
        plt.hist(st,bins=100,histtype="step")
        return st

def make_fake(fwhm=15,rmfname="auto.rmf",exptime=2000,filename=None):
    AllData.clear()
    #Plot.device = "/xs"
    #s.ignore("**-"+str(ran[0])+" "+str(ran[1])+"-**")
    #Plot.xAxis="keV"
    #Plot("data")
    sig = Analysis.fwhm2sigma(fwhm)*1e-3
    m=Model("powerlaw+gaussian+gaussian")
    m.powerlaw.PhoIndex.values = [0,-1]
    m.powerlaw.norm.values = [96.7962,-1]
    m.gaussian.LineE.values = [29.18163,-1,0.0,0.0,30.0,30.0]
    m.gaussian.Sigma.values = [sig,-1,1e-10,1e-10,10,10]
    m.gaussian.norm.values = [66.170,-1,0,0,1e+20,1e+24]
    m.gaussian_3.LineE.values = [29.18993,-1,0.0,0.0,30.0,30.0]
    m.gaussian_3.Sigma.values = [sig,-1,1e-10,1e-10,10,10]
    m.gaussian_3.norm.values = [7.8457,-1,0,0,1e+20,1e+24]
    m.show()
    if filename == None:
        filename = "./fake_file/FWHM"+str(fwhm)+"exp_"+str(exptime)+".fak"
    fs = FakeitSettings(response=rmfname,exposure=str(exptime),fileName=filename)
    #fs = FakeitSettings(exposure=str(exptime))
    #rsp = AllData.dummyrsp(lowE=28.0,highE=30.0,nBins=100)
    AllData.fakeit(settings=fs)
    return filename

def fake_fit(fn=None,ran=(28.0,30.0),smethod="chi",tmethod="chi",pow_norm_fit=False,err=True,step=True,fwhm=15):
    AllData.clear()
    exptime = re.findall(r"[-+]?\d*\.\d+|\d+",fn)[1]
    print("exposure = ",exptime)
    sig = Analysis.fwhm2sigma(fwhm)*1e-3
    AllModels.clear()
    s=Spectrum(fn)
    m=Model("powerlaw+gaussian")
    m.powerlaw.PhoIndex.values = [0,-1]
    m.powerlaw.norm.values = [96.7962,-1]
    m.gaussian.LineE.values = [29.182,1e-6,0.0,0.0,30.0,30.0]
    m.gaussian.Sigma.values = [sig,-1,1e-10,1e-10,10,10]
    m.gaussian.norm.values = [100,1e-6,0,0,1e+20,1e+24]
    if pow_norm_fit == True: 
        m.powerlaw.norm.values = [96.7962,1]
    AllData.ignore("**-"+str(ran[0])+" "+str(ran[1])+"-**")
    Fit.query="yes"
    Fit.statMethod = smethod
    Fit.statTest = tmethod
    #Plot("counts")
    Fit.perform()
    Fit.show()
    Plot.xAxis="keV"
    Plot("counts")
    if step == True:
        Fit.steppar("3 29.18 29.187 500")
        dels = np.array(Fit.stepparResults("delstat"))
        stepE = np.array(Fit.stepparResults("3"))
        m.gaussian.LineE.values = [stepE[np.argmin(dels)],1e-6,0.0,0.0,30.0,30.0]
        Fit.perform()
    Test = Fit.testStatistic/Fit.dof
    print(Test)
    if err == True:
        if round(Test,5) < 2.04095 :
            if pow_norm_fit == True:
                Fit.error("1.0 1 3 5")
            else :
                Fit.error("1.0 3 5")
            er_s = True
        else :
            er_s = False
    else:
        er_s = 0
    Plot("counts")
    xs,ys,xe,ye = set_plot()
    #print(xs,ys) 
    spec_plotter(xs=xs,ys=ys,xe=xe,ye=ye,savename="./graph/fake.png",ran=ran)
    all_counts = ys/1e+3
    gaus_counts = m.gaussian.norm*float(exptime)
    print("Exposure Time = ",exptime," [s]")
    print("All counts = ",np.sum(all_counts))  
    print("Gaussian counts = ",gaus_counts) 
    print("Null Hypothesis Probability = ",Fit.nullhyp)
    #print(stepE[np.argmin(dels)])
    return np.sum(all_counts), gaus_counts, Fit.nullhyp, er_s

def make_fakes(fwhm=15,pow_norm_fit=False,err=True,step=True,rmfname="auto.rmf"):
    #num = np.linspace(10,200,10)
    num = [15,30,50,70,90,100,130,150,200]
    for i in num:
        make_fake(fwhm=fwhm,rmfname=rmfname,exptime=i,filename=None)

def fake_fits(ran=(28.0,30.0),smethod="chi",tmethod="chi",pow_norm_fit=False,err=True,step=True,fwhm=15):
    fl = glob.glob("./*.pi")
    print(fl)
    a_counts = []
    counts = []
    nhp = []
    er = []
    for fn in fl :
        all_counts,gaus_counts,nullhyp,er_s = fake_fit(fn=fn,ran=ran,smethod=smethod,tmethod=tmethod,pow_norm_fit=pow_norm_fit,err=err,step=step,fwhm=fwhm)
        a_counts.append(all_counts)
        counts.append(gaus_counts)
        nhp.append(nullhyp)
        er.append(er_s)
    fig=plt.figure(figsize=(8,6))
    plt.plot(counts,nhp,".")
    print(er)
    return counts, nhp

#def ThLine_res():
    # fl = glob.glob("./fake_file/*.pi")
    # fig=plt.figure(figsize=(8,6))
    # for fn in fl :
        
    # #num = [1000,3000,5000,6000,7000,8000,10000,12500,14000,17500,20000,30000]
    # a_counts = []
    # counts = []
    # nhp = []
    # er = []
    # a_counts.append(all_counts)
    # counts.append(gaus_counts)
    # nhp.append(nullhyp)
    # er.append(er_s)
