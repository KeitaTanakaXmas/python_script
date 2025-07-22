import numpy as np
import re
import h5py
from scipy.stats import chi2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fn = "fitting_result.hdf5"
#fnt = "../data_set_1-CH4.hdf5"
fnt = "../data_set_1-CH4_allreg.hdf5"

def linear(x,b,c):
    return b*x + c

def square(x,b,c,d):
    return b*x**2 + c*x + d

def cubic(x,b,c,d,e):
    return b*x**3 + c*x**2 + d*x + e

def multiID(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

def read_line(line="Am241",step=50):
    with open(line+".log") as f:
        l = f.readlines()
    ls = [li.strip() for li in l]
    Csta_ID = [li for li in ls if "#Fit statistic  : C-Statistic" in li ]
    Chi_ID = [li for li in ls if "#Test statistic : Chi-Squared" in li ]
    Model = [li for li in ls if "Model powerlaw<1>" in li ]
    Con = [li for li in ls if "steppar" in li ]
    lid = multiID(ls,Model[-1])[-1]
    cid = multiID(ls,Con[-1])[-1]
    Csta_list = multiID(ls,Csta_ID[-1])[-1]
    Chi_list = multiID(ls,Chi_ID[-1])[-1]
    Csta=np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[Csta_list]),dtype=float)
    Chi=np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[Chi_list]),dtype=float)
    print(Csta)
    print(Chi)
    for i in range(cid,int(cid+1e+5)):
        con = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[i]),dtype=float)
        if len(con) == 6 :
            if not "con_list" in locals() :
                con_list = np.array(con)
            else :
                con_list = np.vstack((con_list,con))
            
                if (int(con[2]) == step & int(con[4]) == step) :
                    break
            
            
    print(con_list)
    print(Model[-1])
    if "pgauss" in str(Model[-1]) :
        pow_Id = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+3]),dtype=float)
        pow_norm = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+4]),dtype=float)
        pgauss_LE =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+5]),dtype=float)
        pgauss_a =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+6]),dtype=float)
        pgauss_b =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+7]),dtype=float)
        pgauss_sig =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+8]),dtype=float)
        pgauss_norm =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+9]),dtype=float)
        for i in range(lid+3,lid+9):
            print(np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[i])))
        with h5py.File(fn) as f:
            if line in f.keys():
                 del f[line]
            f.create_group(str(line))
            f.create_dataset(str(line)+"/pow_Id",data=pow_Id)
            f.create_dataset(str(line)+"/pow_norm",data=pow_norm)
            f.create_dataset(str(line)+"/pgauss_LE",data=pgauss_LE)
            f.create_dataset(str(line)+"/pgauss_a",data=pgauss_a)
            f.create_dataset(str(line)+"/pgauss_b",data=pgauss_b)
            f.create_dataset(str(line)+"/pgauss_sig",data=pgauss_sig)
            f.create_dataset(str(line)+"/pgauss_norm",data=pgauss_norm)
            f.create_dataset(str(line)+"/contour",data=con_list)
            f.create_dataset(str(line)+"/Csta",data=Csta)
            f.create_dataset(str(line)+"/redChi",data=Chi)
            
            
    if "lorentz" in str(Model[-1]) :
        if "-" in str(line) :
            pow_Id = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+3]),dtype=float)
            pow_norm = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+4]),dtype=float)
            gsmooth_sig =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+5]),dtype=float)
            gsmooth_Id=  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+6]),dtype=float)
            plorentz_LE =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+7]),dtype=float)
            plorentz_a =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+8]),dtype=float)
            plorentz_b =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+9]),dtype=float)
            plorentz_sig =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+10]),dtype=float)
            plorentz_norm =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+11]),dtype=float)
            plorentz_LE2 =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+12]),dtype=float)
            plorentz_a2 =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+13]),dtype=float)
            plorentz_b2 =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+14]),dtype=float)
            plorentz_sig2 =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+15]),dtype=float)
            plorentz_norm2 =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+16]),dtype=float)
            for i in range(lid+3,lid+11):
                print(np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[i])))
            with h5py.File(fn) as f:
                if line in f.keys():
                     del f[line]
                f.create_group(str(line))
                f.create_dataset(str(line)+"/pow_Id",data=pow_Id)
                f.create_dataset(str(line)+"/pow_norm",data=pow_norm)
                f.create_dataset(str(line)+"/gsmooth_sig",data=gsmooth_sig)
                f.create_dataset(str(line)+"/gsmooth_Id",data=gsmooth_Id)
                f.create_dataset(str(line)+"/plorentz_LE",data=plorentz_LE)
                f.create_dataset(str(line)+"/plorentz_a",data=plorentz_a)
                f.create_dataset(str(line)+"/plorentz_b",data=plorentz_b)
                f.create_dataset(str(line)+"/plorentz_sig",data=plorentz_sig)
                f.create_dataset(str(line)+"/plorentz_norm",data=plorentz_norm)
                f.create_dataset(str(line)+"/plorentz_LE2",data=plorentz_LE2)
                f.create_dataset(str(line)+"/plorentz_a2",data=plorentz_a2)
                f.create_dataset(str(line)+"/plorentz_b2",data=plorentz_b2)
                f.create_dataset(str(line)+"/plorentz_sig2",data=plorentz_sig2)
                f.create_dataset(str(line)+"/plorentz_norm2",data=plorentz_norm2)
                f.create_dataset(str(line)+"/contour",data=con_list)
                f.create_dataset(str(line)+"/Csta",data=Csta)
                f.create_dataset(str(line)+"/redChi",data=Chi)
        else :
            pow_Id = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+3]),dtype=float)
            pow_norm = np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+4]),dtype=float)
            gsmooth_sig =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+5]),dtype=float)
            gsmooth_Id=  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+6]),dtype=float)
            plorentz_LE =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+7]),dtype=float)
            plorentz_a =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+8]),dtype=float)
            plorentz_b =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+9]),dtype=float)
            plorentz_sig =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+10]),dtype=float)
            plorentz_norm =  np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[lid+11]),dtype=float)
            for i in range(lid+3,lid+11):
                print(np.array(re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",ls[i])))
            with h5py.File(fn) as f:
                if line in f.keys():
                     del f[line]
                f.create_group(str(line))
                f.create_dataset(str(line)+"/pow_Id",data=pow_Id)
                f.create_dataset(str(line)+"/pow_norm",data=pow_norm)
                f.create_dataset(str(line)+"/gsmooth_sig",data=gsmooth_sig)
                f.create_dataset(str(line)+"/gsmooth_Id",data=gsmooth_Id)
                f.create_dataset(str(line)+"/plorentz_LE",data=plorentz_LE)
                f.create_dataset(str(line)+"/plorentz_a",data=plorentz_a)
                f.create_dataset(str(line)+"/plorentz_b",data=plorentz_b)
                f.create_dataset(str(line)+"/plorentz_sig",data=plorentz_sig)
                f.create_dataset(str(line)+"/plorentz_norm",data=plorentz_norm)
                f.create_dataset(str(line)+"/contour",data=con_list)
                f.create_dataset(str(line)+"/Csta",data=Csta)
                f.create_dataset(str(line)+"/redChi",data=Chi)

#   1    1   powerlaw   PhoIndex            0.0          frozen
#   2    1   powerlaw   norm                546.836      +/-  30.4399
#   3    2   pgauss     LineE      keV      26.3446      frozen
#   4    2   pgauss     a                   0.537517     +/-  7.65679E-05
#   5    2   pgauss     b                   7.84080E-02  +/-  4.70458E-04
#   6    2   pgauss     sigma      keV      1.73206E-02  +/-  6.49883E-04
#   7    2   pgauss     norm                266.513      +/-  22.4407

#Model powerlaw<1> + gsmooth<2>*plorentz<3> Source No.: 1   Active/On
#Model Model Component  Parameter  Unit     Value
# par  comp
#   1    1   powerlaw   PhoIndex            0.0          frozen
#   2    1   powerlaw   norm                440.535      +/-  29.4041
#   3    2   gsmooth    Sig_6keV   keV      1.10461E-02  +/-  2.09776E-03
#   4    2   gsmooth    Index               0.0          frozen
#   5    3   plorentz   LineE      keV      30.9730      frozen
#   6    3   plorentz   a                   0.474230     +/-  3.62880E-04
#   7    3   plorentz   b                   1.33649      +/-  2.27472E-03
#   8    3   plorentz   sigma      keV      1.56000E-02  frozen
#   9    3   plorentz   norm                9.86990E-02  +/-  1.25888E-02


def cal_delpha(line="Am241",reli=68):
    with h5py.File(fn) as f:

        if line == "CsKa1" :
            a_fit = f[line+"/plorentz_a"][2]
            b_fit = f[line+"/plorentz_b"][2]
            E = f[line+"/plorentz_LE"][2]
        if "-" in line:
            a_fit = f[line+"/plorentz_a"][2]
            b_fit = f[line+"/plorentz_b"][2]
            E = f[line+"/plorentz_LE"][2]
            E2 = f[line+"/plorentz_LE2"][2]
        if line == "Am241" :
            a_fit = f[line+"/pgauss_a"][2]
            b_fit = f[line+"/pgauss_b"][2]
            E = f[line+"/pgauss_LE"][2]
        
        delc = f[line+"/contour"][:,1]
        a_list = f[line+"/contour"][:,3]
        b_list = f[line+"/contour"][:,5]
        redc = np.abs(delc - chi2.ppf(q=reli/100,df=2))
        a = a_list[np.argmin(redc)]
        b = b_list[np.argmin(redc)]
        print("a = ",a,"b = ",b,"redc = ",np.min(redc))
        PHAfit = a_fit * E + b_fit
        delPHA = np.abs(PHAfit - (a * E + b))
        print("PHA = ",PHAfit," ± ",delPHA)
        result = [PHAfit,delPHA,E]
        if "delpha/"+line in f.keys():
            del f["delpha/"+line]
        if "E2" in locals():
            if "delpha/AgKa2" in f.keys():
                del f["delpha/AgKa2"]
                del f["delpha/AgKa1"]
            PHAfit2 = a_fit * E2 + b_fit
            delPHA2 = np.abs(PHAfit2 - (a * E2 + b))
            print("PHA = ",PHAfit2," ± ",delPHA2)
            result2 = [PHAfit2,delPHA2,E2]
            if line == "AgKa2-1":
                f.create_dataset("delpha/AgKa2",data=result)
                f.create_dataset("delpha/AgKa1",data=result2)
            if line == "CsKa2-1":
                f.create_dataset("delpha/CsKa2",data=result)
                f.create_dataset("delpha/CsKa1",data=result2)
            if line == "CsKb3-1":
                f.create_dataset("delpha/CsKb3",data=result)
                f.create_dataset("delpha/CsKb1",data=result2)
        else :
            f.create_dataset("delpha/"+line,data=result)
     
def PHA_Energy():
    with h5py.File(fn) as f:
        print("PHA change to Energy")
        print("Using following lines")
        print(f["delpha"].keys())
        for line in f["delpha"].keys():
            if not "PHA" in locals():
                PHA = np.array(float(f["delpha/"+line][0]))
            else:
                PHA = np.append(PHA,float(f["delpha/"+line][0]))
            if not "PHA_err" in locals():
                PHA_err = np.array(float(f["delpha/"+line][1]))
            else:
                PHA_err = np.append(PHA_err,float(f["delpha/"+line][1]))
            if not "Energy" in locals():
                Energy = np.array(float(f["delpha/"+line][2]))
            else:
                Energy = np.append(Energy,float(f["delpha/"+line][2]))
        print(PHA)
        print(PHA_err)
        print(Energy)
        return PHA,PHA_err,Energy
     
def PHAtoE(gen=True,func="square",ll=14.0,hh=16.1,pha_h=None):
    with h5py.File(fn) as f:
        print("PHA change to Energy")
        print("Using following lines")
        print(f["delpha"].keys())
        for line in f["delpha"].keys():
            if not "PHA" in locals():
                PHA = np.array(float(f["delpha/"+line][0]))
            else:
                PHA = np.append(PHA,float(f["delpha/"+line][0]))
            if not "PHA_err" in locals():
                PHA_err = np.array(float(f["delpha/"+line][1]))
            else:
                PHA_err = np.append(PHA_err,float(f["delpha/"+line][1]))
            if not "Energy" in locals():
                Energy = np.array(float(f["delpha/"+line][2]))
            else:
                Energy = np.append(Energy,float(f["delpha/"+line][2]))
        print(PHA)
        print(PHA_err)
        print(Energy)

        popt, pcov = curve_fit(linear,Energy,PHA,sigma=PHA_err)
        popt_linear = np.array(popt)
        residuals = np.array(PHA) - linear(Energy,*popt_linear)
        chidl = np.sum(((residuals**2)/PHA_err**2))
        print(f"chi2_1={chidl}")
        print(f"chi2_1/dof={chidl/3}")

        popt, pcov = curve_fit(square,Energy,PHA,sigma=PHA_err)
        popt_square = np.array(popt)
        residuals = np.array(PHA) - square(Energy,*popt_square)
        chids = np.sum(((residuals**2)/PHA_err**2))
        print(f"chi2_2={chids}")
        print(f"chi2_2/dof={chids/2}")

        popt, pcov = curve_fit(cubic,Energy,PHA,sigma=PHA_err)
        popt_cubic = np.array(popt)
        residuals = np.array(PHA) - cubic(Energy,*popt_cubic)
        chidc =np.sum(((residuals**2)/PHA_err**2))
        print(f"chi2_3={chidc}")
        print(f"chi2_3/dof={chidc}")
        print(f"F1to2={(chidl-chids)/(chids/(2))}")
        print(f"F2to3={(chids-chidc)/(chidc/(1))}")

        fig=plt.figure(figsize=(8,6))
        

        
        if gen == True:
            with h5py.File(fnt) as ft:
                popt, pcov = curve_fit(square,PHA,Energy)
                print("a = ",popt[0]," ± ",np.sqrt(pcov[0,0]))
                print("b = ",popt[1]," ± ",np.sqrt(pcov[1,1]))
                print("c = ",popt[2]," ± ",np.sqrt(pcov[2,2]))
                residuals = np.array(Energy) - square(PHA,*popt)
                chids = np.sum(((residuals**2)/PHA_err**2))
                print(f"chi2_2={chids}")
                print(f"chi2_2/dof={chids}")
                #pha_se = ft["analysis/se_cor/pha"][:]*26.3/1.45
                pha_se = pha_h*26.3/1.45
                #*26.3/1.45
                mask = (ll < pha_se) & (pha_se < hh)
                pha_e = pha_se[mask]
                pha_ch = square(pha_e,*popt)
                print(pha_ch)
                plt.hist(pha_e,bins=1000,histtype="step")
                plt.hist(pha_ch,bins=1000,histtype="step")
                if "energy" in ft.keys():
                    del ft["energy/gen"]
                ft.create_dataset("energy/gen",data=pha_ch)
                print(square(PHA,*popt))
                #ft.create_dataset("energy/se",data=pha_se)
                    

def plot_PHAtoE(func="linear"):
    PHA,PHA_err,Energy = PHA_Energy()
    popt, pcov = curve_fit(eval(func),PHA,Energy,sigma=PHA_err)
    popt2 = np.array(popt)
    residuals = np.array(Energy) - eval(func)(PHA,*popt2)
    chidc =np.sum(((residuals**2)/PHA_err**2))
    # print(f"chi2_3={chidc}")
    # print(f"chi2_3/dof={chidc}")
    # print(f"F1to2={(chidl-chids)/(chids/(2))}")
    # print(f"F2to3={(chids-chidc)/(chidc/(1))}")
    fig=plt.figure(figsize=(8,6))
    gs=GridSpec(nrows=2,ncols=1,height_ratios=[2,1])
    gs1=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0,:])
    gs2=GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[1,:])
    ax=fig.add_subplot(gs1[:,:])
    ls=13
    ax.errorbar(Energy,PHA,yerr=PHA_err,fmt="o",markersize=2,color="black",label="data")
    ax.set_title(func+" fitting result ")
    ax.set_ylabel("PHA",fontsize=ls)
    fx = np.linspace(20,35,100)
    ax.plot(eval(func)(fx,*popt2),fx,"-",label="Fitting All",color="red")
    ax.set_xlim(20,35)
    ax.legend(fontsize=14)
    ax2=fig.add_subplot(gs2[:,:])
    # Res = cubic(fx,*popt3) - square(fx,*popt2)
    ax2.errorbar(Energy,residuals,yerr=PHA_err,fmt="o",markersize=2,color="black",label="data")
    #ax2.plot(fx,Res)
    ax2.plot(fx,np.zeros(len(fx)),color="red")
    ax2.set_xlim(20,35)
    ax2.set_xlabel("Energy[keV]",fontsize=ls)
    ax2.set_ylabel("data-model",fontsize=ls)
    fig.subplots_adjust(hspace=.0)
    fig.align_labels()
    plt.show()
    fig.savefig(func+".png",dpi=300)
    print(eval(func)(PHA,*popt2))
    
#def PHAtoE_gen():
    
