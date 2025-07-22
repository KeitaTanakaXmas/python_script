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

h5_fn = "data_set_1-CH4.hdf5"


## FUNCTION DEFINE ##

def get_fl(s="p"):
    dir=os.curdir
    fl=[path for path in os.listdir(dir) if path.endswith(s+".fits")]
    return sorted(fl)
    
def noise_def():
    i = 0
    fl = get_fl(s="n")
    for fn in fl:
        i = i + 1
        if i == 1:
            t,n=Util.fopen(fn)
        else:
            t,n_a=Util.fopen(fn)
            n=np.append(n,n_a,axis=0)
    return t,n
    
def pulse_def():
    i = 0
    fl = get_fl(s="p")
    for fn in fl:
        i = i + 1
        if i == 1:
            t,p=Util.fopen(fn)
        else:
            t,p_a=Util.fopen(fn)
            p=np.append(p,p_a,axis=0)
    return t,p
    

def ofs_def(sig=3,tred=1.644e+5):
    i = 0
    nfl = get_fl(s="n")
    for fn in nfl:
        print(fn)
        i = i + 1
        t,n = Util.fopen(fn)
        if i == 1 :
            ofs = np.median(n,axis=1)
            ofs_std = np.std(n,axis=1)
        else :
            ofs = np.append(ofs,np.median(n,axis=1),axis=0)
            ofs_std = np.append(ofs,np.std(n,axis=1),axis=0)
    ofs_all_avg = np.average(ofs)
    ofs_all_std = np.std(ofs)
    mask = (ofs_all_avg-sig*ofs_all_std<ofs) & (ofs_all_avg+sig*ofs_all_std>ofs)
#      & (np.array(range(len(ofs))) < tred)
    return ofs,ofs_std,ofs_all_avg,ofs_all_std,mask
    
                    
    
def ph_def(sig=3):
    i = 0
    pfl = get_fl(s="p")
    for fn in pfl:
        i = i + 1
        t,p = Util.fopen(fn)
        if i == 1 :
            pmin = np.min(p,axis=1)
        else :
            pmin = np.append(pmin,np.min(p,axis=1),axis=0)
    ofs,ofs_std,ofs_all_avg,ofs_all_std,mask = ofs_def(sig=sig)
    ph = ofs - pmin
    return ph
    
    
## Make hdf5 file & data group ##

def make_hd(fn=h5_fn,sig=3):
    with h5py.File(fn) as f:
        p_mask = f["double_mask"][:]
        ofs,ofs_std,ofs_all_avg,ofs_all_std,o_mask = ofs_def(sig=sig)
        print(o_mask)
        print(p_mask)
        mask = (o_mask == True) & (p_mask>0.9)
        ph = ph_def(sig=sig)
        if "data" in f.keys():
            del f["data"]
            del f["mask"]
        f.create_group("data")
        f.create_dataset("data/offset",data=ofs)
        f.create_dataset("data/offset_error",data=ofs_std)
        f.create_dataset("data/offset_all_average",data=ofs_all_avg)
        f.create_dataset("data/offset_all_error",data=ofs_all_std)
        f.create_dataset("data/ph_ofs_mask",data=ph[mask])
        f.create_dataset("data/ph",data=ph)
        f.create_group("mask")
        f.create_dataset("mask/offset",data=mask,dtype=np.bool)


def Am241_mask(fn=h5_fn,rl=1.448,rh=1.46):
    with h5py.File(fn) as f:
        if "Am241" in f["mask"].keys():
            del f["mask/Am241"]
        ph = f["data/ph"][:]
        m = f["mask/offset"][:]
        ph_mask = (rl < ph) & (ph < rh) & (m)
        f.create_dataset("mask/Am241",data=ph_mask,dtype=np.bool)
        print("Number of pulse around Am241 = ",len(ph[ph_mask]))

#tmpl generate
     
def tmpl_psel(fn=h5_fn):
    fl = get_fl(s="p")
    with h5py.File(fn) as f:
        if "analysis" in f.keys():
            del f["analysis"]
        f.create_group("analysis/tmpl")
        i = 0
        for fnl in fl:
            i = i + 1
            t,p=Util.fopen(fnl)
            ph_mask = f["mask/Am241"][0:len(p)]
            ofs_mask = f["mask/offset"][0:len(p)]
            mask = (ph_mask) & (ofs_mask)
            f.create_dataset("analysis/tmpl/pulse/pulse*".replace("*",str(i)),data=p[mask])
        pl = f["analysis/tmpl/pulse"].keys()
        i = 0
        for pi in pl:
            i = i + 1
            if i == 1:
                tmpl_pulse = f["analysis/tmpl/pulse/"+pi][:]
                del f["analysis/tmpl/pulse/"+pi]
            else:
                tmpl_pulse = np.vstack((tmpl_pulse,f["analysis/tmpl/pulse/"+pi][:]))
                del f["analysis/tmpl/pulse/"+pi]
        f.create_dataset("analysis/tmpl/pulse/sel_pulse",data=tmpl_pulse)
        
def tmpl_nsel(fn=h5_fn):
    fl = get_fl(s="n")
    with h5py.File(fn) as f:
        if "noise" in f["analysis/tmpl"].keys():
            del f["analysis/tmpl/noise"]
        i = 0
        for fnl in fl:
            i = i + 1
            t,n=Util.fopen(fnl)
            ph_mask = f["mask/Am241"][0:len(n)]
            ofs_mask = f["mask/offset"][0:len(n)]
            mask = (ph_mask) & (ofs_mask)
            f.create_dataset("analysis/tmpl/noise/noise*".replace("*",str(i)),data=n[mask])
        pl = f["analysis/tmpl/noise"].keys()
        i = 0
        for pi in pl:
            i = i + 1
            if i == 1:
                tmpl_pulse = f["analysis/tmpl/noise/"+pi][:]
                del f["analysis/tmpl/noise/"+pi]
            else:
                tmpl_pulse = np.vstack((tmpl_pulse,f["analysis/tmpl/noise/"+pi][:]))
                del f["analysis/tmpl/noise/"+pi]
        f.create_dataset("analysis/tmpl/noise/sel_noise",data=tmpl_pulse)
        if "time" in f["data"].keys():
            del f["data/time"]
        f.create_dataset("data/time",data=t)
        
def tmpl_sn(ms="Am241",fn=h5_fn):
    with h5py.File(fn) as f:
        if "template" in f["analysis/tmpl"].keys():
            del f["analysis/tmpl/template"]
            del f["analysis/tmpl/sn"]
        n = f["analysis/tmpl/noise/sel_noise"]
        ofs = np.median(n,axis=1)
        for i in range(0,len(n)-1):
            p = f["analysis/tmpl/pulse/sel_pulse"][i] - ofs[i]
        tmpl,sn = Filter.generate_template(p,n,max_shift=10)
        f.create_dataset("analysis/tmpl/template",data=tmpl)
        f.create_dataset("analysis/tmpl/sn",data=sn)
        
def pha_opt(fn=h5_fn):

    fl = get_fl(s="p")
    with h5py.File(fn) as f:
        if "pha" in f["analysis"].keys():
            del f["analysis/pha"]
            del f["analysis/ps"]
        i = 0
        tmpl = f["analysis/tmpl/template"]
        for fnl in fl:
            i = i + 1
            t,p=Util.fopen(fnl)
            pha, ps = Filter.optimal_filter(p, tmpl, max_shift=10)
            f.create_dataset("analysis/pha/pha*".replace("*",str(i)),data=pha)
            f.create_dataset("analysis/ps/ps*".replace("*",str(i)),data=ps)
        pl = f["analysis/pha"].keys()
        i = 0
        for pi in pl:
            i = i + 1
            if i == 1:
                pha = f["analysis/pha/"+pi][:]
                del f["analysis/pha/"+pi]
            else:
                pha = np.hstack((pha,f["analysis/pha/"+pi][:]))
                del f["analysis/pha/"+pi]
        del f["analysis/pha"]
        f.create_dataset("analysis/pha",data=pha)
        
        pl = f["analysis/ps"].keys()
        i = 0
        for pi in pl:
            i = i + 1
            if i == 1:
                ps = f["analysis/ps/"+pi][:]
                del f["analysis/ps/"+pi]
            else:
                ps = np.hstack((ps,f["analysis/ps/"+pi][:]))
                del f["analysis/ps/"+pi]
        del f["analysis/ps"]
        f.create_dataset("analysis/ps",data=ps)


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
        
def ofs_plot(fn=h5_fn,sig=3):
    with h5py.File(fn) as f:
        ofs = f["data/offset"][:]
        mask = f["mask/offset"][:]
        ofs_avg = f["data/offset_all_average"][...]
        ofs_std = f["data/offset_all_error"][...]
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
        ax.hlines(hh,0,len(ofs),color="red", linestyles='dashed')
        ax.hlines(hl,0,len(ofs),color="red", linestyles='dashed')
        ax.set_title("Offset distribution",fontsize=16)
        ax.set_xlabel("Index",fontsize=16)
        ax.set_ylabel("Offset average[V]",fontsize=16)
        fig.savefig("./graph/offset_index.png",dpi=300)
        plt.show()
        
def pha_ofs_lin(fn=h5_fn,bs=1000,secor=False):
    with h5py.File(fn) as f:
        if secor == True:
            pha = f["analysis/se_cor/pha"][:]
        else:
            pha = f["analysis/pha"][:]
        ofs = f["data/offset"][:]
        ofs_mask = f["mask/offset"][:]
        fig = plt.figure(figsize = (16,8))
        gs = GridSpec(1,2,width_ratios=(1,4))
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1],sharey=ax1)
        ax1.hist(pha,orientation="horizontal", bins=bs, histtype="step")
        ax1.invert_xaxis()
        ax1.set_ylabel("Pulse height analysis",fontsize=16)
        ax2 = plt.subplot(gs[1],sharey=ax1)
        ax2.plot(ofs[ofs_mask], pha[ofs_mask], ".")
        ax2.set_title("Offset vs PHA",fontsize=16)
        ax2.set_xlabel("Offset[V]",fontsize=16)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.subplots_adjust(wspace=0.1)
        if secor == True:
            fig.savefig("./graph/pha_ofs_lin_secor.png",dpi=300)
        else:
            fig.savefig("./graph/pha_ofs_lin.png",dpi=300)
        plt.show()

##PULSE ANALYSIS

#def ph_hist(sig=3,ran=None,vline=None):
#    i = 0
#    pfl = get_pfl()
#    for fn in pfl:
#        i = i + 1
#        t,p = Util.fopen(fn)
#        if i == 1 :
#            pmin = np.min(p,axis=1)
#        else :
#            pmin = np.append(pmin,np.min(p,axis=1),axis=0)
#    ofs,ofs_ave,ofs_std,mask = ofs_def(sig=sig)
#    ph = ofs - pmin
#    fig = plt.figure(figsize=(8.0,6.0))
#    ax = fig.add_subplot(111)
#    ax.hist(ph[mask],range=ran,bins=500)
#    if vline is not None:
#        ax.vlines(vline,0,30000,color="red", linestyles='dashed')
#    ax.set_title("Pulse height distribution",fontsize=16)
#    ax.set_xlabel("Pulse height[V]",fontsize=16)
#    ax.set_ylabel("Counts",fontsize=16)
#    fig.savefig("../graph/pulse_height_hist.png",dpi=300)
#    plt.show()
    
def ph_hist(fn=h5_fn,bs=500,ran=None,vv=None):
    with h5py.File(fn) as f:
        ph_ofs = f["data/ph_ofs_mask"][:]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.hist(ph_ofs,range=ran,bins=bs)
        if vv is not None:
            ax.vlines(vv,0,30000,color="red", linestyles='dashed')
        ax.set_title("Pulse height distribution",fontsize=16)
        ax.set_xlabel("Pulse height[V]",fontsize=16)
        ax.set_ylabel("Counts",fontsize=16)
        fig.savefig("./graph/ph_hist.png",dpi=300)
        plt.show()

def pha_hist(fn=h5_fn,bs=500,ran=None,vv=None,x="ene",ID=True,se=False):
    with h5py.File(fn) as f:
        if se == True:
            pha = f["analysis/se/pha"][:]
        else:
            pha = f["analysis/pha"][:]
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
        if ID == True:
            ID,E = line_ID()
            ax2 = ax.twiny()
            ax2.set_xticks(E)
            ax2.set_xticklabels(ID,fontsize=8)
            ax2.tick_params(direction="in",which="major",top=True,bottom=False,labeltop=True)
            ax2.set_xlim(ran)
        fig.savefig("./graph/pha_hist.png",dpi=300)
        plt.show()


def tmpl_plot(fn=h5_fn):
    with h5py.File(fn) as f:
        t = f["data/time"][:]
        tmpl = f["analysis/tmpl/template"][:]
        fig = plt.figure(figsize=(8.0,6.0))
        ax = fig.add_subplot(111)
        ax.plot(t,tmpl)
        ax.set_title("Template",fontsize=16)
        ax.set_xlabel("Time[s]",fontsize=16)
        ax.set_ylabel("Template (arb. unit)",fontsize=16)
        fig.savefig("./graph/template.png",dpi=300)
        plt.show()

def cal_se(fn=h5_fn,pa=[1.5,1.7],all=True):
    with h5py.File(fn) as f:
        ofs_mask = f["mask/offset"][:]
        pha = f["analysis/pha"][:][ofs_mask]
        ofs = f["data/offset"][:][ofs_mask]
        if all == True:
            pa = [0,np.max(pha)+1]
            if "se_all" in f["analysis"].keys():
                del f["analysis/se_all"]
        else:
            if "se" in f["analysis"].keys():
                del f["analysis/se"]
        alpha_list = np.arange(-0.4,-0.1,0.001)
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
        if all == True:
            f.create_dataset("analysis/se_all/H",data=H)
            f.create_dataset("analysis/se_all/ma",data=ma)
            f.create_dataset("analysis/se_all/pha",data=pha_e)
            f.create_dataset("analysis/se_all/reg",data=pa)
            fig.savefig("./graph/se_all.png",dpi=300)
        else:
            f.create_dataset("analysis/se/H",data=H)
            f.create_dataset("analysis/se/ma",data=ma)
            f.create_dataset("analysis/se/pha",data=pha_e)
            f.create_dataset("analysis/se/reg",data=pa)
            fig.savefig("./graph/se.png",dpi=300)
        plt.show()
        
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
        if "se_cor" in f["analysis"].keys():
            del f["analysis/se_cor"]
        pha = f["analysis/pha"][:]
        ofs = f["data/offset"][:]
        ofs_mask = f["mask/offset"][:]
        ma = f["analysis/se/ma"][...]
        pha_secor = pha[ofs_mask]*(1+ma*(ofs[ofs_mask] - np.median(ofs[ofs_mask])))
        f.create_dataset("analysis/se_cor/pha",data=pha_secor)
        f.create_dataset("analysis/se_cor/ma",data=ma)
    
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

def make_resp(rmfname='test.rmf', bmin=1000., bmax=30000.):
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
     'resp_number='+str(int(bin_max-bin_min)),\
     'chan_reln=linear',\
     'chan_low='+str(bin_min),\
     'chan_high='+str(bin_max),\
     'chan_number='+str(int(bin_max-bin_min)),\
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


def fits2xspec(binsize=1, exptime=1, fwhm=0.0001, gresp=False, garf=False, filename='test.fits', rmfname='test.rmf', arfname='test.arf', TEStype='TMU524', Datatype='PHA', chan='ch65',gen=True):
    
    with h5py.File(h5_fn) as f:
        if gen == True:
            pha = f["energy/gen"][:]*1e+3
        else:
            pha = f["analysis/se/pha"][:]*1e+4
        # separate bins
        if Datatype=='PHA':
            n, bins = histogram(pha[pha>0], binsize=binsize)
        else:
            n, bins = histogram(pha[pha>0.05], binsize=binsize)
        # py.figure()
        # py.hist(pha, bins=bins, histtype='stepfilled', color='k')
        # py.show()

        # par of fits
        filename = filename
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


def p_cl(fn=h5_fn,psig=20,nsig=5,tsig=5,order=50,trig=0.1):
    with h5py.File(fn) as f:
        pfl = get_fl(s="p")
        ap_mask = []
        for fl in pfl:
            t,wave_p=Util.fopen(fl)
            length=len(wave_p[0])
            from scipy.signal import argrelextrema
            pulse_cl = []
            p_j      = []
            count_t = int(len(wave_p))
        #wave_p,wave_nとは？
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
                    #print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)
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
        if "double_mask" in f.keys():
            del f["double_mask"]
        f.create_dataset("double_mask",data=ap_mask)
        return ap_mask
        

def p_cl(fn=h5_fn,psig=20,nsig=5,tsig=5,order=50,trig=0.1):
    with h5py.File(fn) as f:
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
                    #print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)
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
        if "double_mask" in f.keys():
            del f["double_mask"]
        f.create_dataset("double_mask",data=ap_mask)
        return ap_mask
        
    # def p_cl(p,psig=20,nsig=5,tsig=5,order=50,trig=0.1):
    #         length = p.shape[1]
    #         pulse_cl = []
    #         p_j      = []
    #         count_t = p.shape[0]
    #         p  = p.reshape((1, len(p)*length))[0]
    #         p -= np.median(p)
    #         p  = p.reshape((int(len(p)/length), length))
    #         p_mask   = []
    #         print('checking double pulse...')
    #         for e, i in enumerate(p):
    #             p_b  = np.correlate(i,[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1], mode='same')
    #             flag = argrelextrema(-p_b, np.greater_equal, order=order)[0]
    #             threshold = -(np.median(p_b[0:90])+(psig*np.std(p_b[0:90])))
    #             check = p_b[flag]<threshold
    #             if len(p_b[flag][check==True]) > 1:
    #                 p_mask.append(False)
    #                 p_j.append(i)
    #             else:
    #                 if length==1000:
    #                     tmin = 80
    #                     tmax = 120
    #                 else:
    #                     tmin = trig*length*0.80
    #                     tmax = trig*length*1.20
    #                 if (tmin<flag[check])&(flag[check]<tmax):
    #                     pulse_cl.append(i)
    #                     p_mask.append(True)
    #                 else:
    #                     p_mask.append(False)
    #                     p_j.append(i)
    #             print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)
    #         count_j = int(len(p_j))
    #         count_cl = int(len(pulse_cl))
    #         #print('\n')
    #         print(f'Number of Junck pulse  {count_j}')
    #         print(f'Number of clean events {count_cl}')
    #         if count_t == 0:
    #             pass
    #         else:
    #             print(f'Pulse removal ratio {count_cl/count_t*100:.2f}%\n')
    #         #print('\n')
    #         pulse_cl = np.asarray(pulse_cl)
    #         p_j = np.asarray(p_j)
    #         p_mask = p_mask
