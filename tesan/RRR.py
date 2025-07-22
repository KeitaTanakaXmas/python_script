import numpy as np
import matplotlib.pyplot as plt


def data_plot(fn="",sn="test"):
    f = np.genfromtxt(fn)
    time = f[:,0] - f[:,0][0]
    dim = f.shape[1]
    fig = plt.figure()
    for i in range(1,dim):
        data = f[:,int(i)]
        plt.plot(time,data,".",label="ch*".replace("*",str(i)))
        plt.legend()
    plt.xlabel("time[s]",fontsize=16)
    plt.ylabel("Resistance[Ω]",fontsize=16)
    plt.show()
    fig.savefig(sn+".png",dpi=300)
    

def load_data(fn="",ch="1"):
    f = np.genfromtxt(fn)
    time = f[:,0]
    data = f[:,int(ch)]
    return time,data

def fourK_ana(Tn="T_AC370_log_20210610160653_Electroplating_Au_E004-E007.txt",Rn="R_AC372_log_20210610160653_Electroplating_Au_E004-E007.txt"):
    Tf = np.genfromtxt(Tn)
    Rf = np.genfromtxt(Rn)
    
    R_ave = np.average(Rf[:,1])
    R_std = np.std(Rf[:,1])
    T_ave = np.average(Tf[:,4])
    T_std = np.std(Tf[:,4])
    print("Resistance[Ω]= ",R_ave,"±",R_std)
    print("Temperature[K]= ",T_ave,"±",T_std)
    print(len(Rf[:,1]))
    return R_ave,R_std,T_ave,T_std
    
def roomK_ana(Rn="R_AC372_log_20210615195002_210416_RT.txt",s=5):
    Rf = np.genfromtxt(Rn)
    R_ave = np.average(Rf[:,2][::-1][0:s])
    R_std = np.std(Rf[:,2][::-1][0:s])
    print("Resistance[Ω]= ",R_ave,"±",R_std)
    print(len(Rf[:,2][::-1][0:s]))
    return R_ave,R_std
