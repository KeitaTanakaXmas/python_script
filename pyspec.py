from xspec import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

def model_init(self):
    models_list=self.componentNames
    self.components = [v for k,v in self.__dict__.items() if k in models_list]
    for model_tmp in self.components:
        paraName_list = model_tmp.parameterNames
        model_tmp.parameters=[v for k,v in model_tmp.__dict__.items() if k in paraName_list]
        for param in model_tmp.parameters:
            param.model = model_tmp
            
def getpar(self):
    ps=[]
    vs=[]
    fs=[]
    es=[]
    for model_tmp in self.components:
        paras_list=model_tmp.parameters
        vals_list=[s.values[0] for s in paras_list]
        fres_list=[s.frozen for s in paras_list]
        errs_list=[(s.error[0],s.error[1]) for s in paras_list]
        ps.extend(paras_list)
        vs.extend(vals_list)
        fs.extend(fres_list)
        es.extend(errs_list)
    return ps, vs, fs, es
    
def link_sel(self, ind_dict):
    ps, vs, fs, es = self.getpar()
    for k,v in ind_dict.items():
        ps[v-1].link="p" + str(k)
        
def freeze_sel(self, inds_f=[1], inds_t=[]):
    ps, vs, fs, es = self.getpar()
    for p in ps:
        if ps.index(p)+1 in inds_f:
            p.frozen=True
    for p in ps:
        if ps.index(p)+1 in inds_t:
            p.frozen=False

def data_ignore(s, vars, dic_ig = {},dic_no = {}, initial_ignore = False):
    
    for k, v in dic_no.items():
        for num in range(len(s) ):
            if s[num].fileName.startswith(k):
                s[num].notice(v)

    for k, v in dic_ig.items():
        for num in range(len(s) ):
            if s[num].fileName.startswith(k):
                s[num].ignore(v)

    
    if not initial_ignore:
        return s
    
    dict_default = {"pn": "**-0.21 9.1-**",
                   "A+B" : "**-2.1 28.0-**",
                   "xis1" : "**-0.21 8.0-**",
                   "xrt" : "**-0.21 4.2-**"}
    
    for k, v in dict_default.items():
        if k in dic_ig.keys():
            continue
        for num in range(len(s) ):
            if s[num].fileName.startswith(k):
                s[num].ignore(v)
    return s
    
def err_all(models_s):
    try:
        models = models_s[0]
        ps, vs, fs, es = models.getpar()
        ps_num = len(ps)
        mss_num = len(models_s)
        inds = [ ps.index(p)+1 for p in ps]
        inds.extend([1+ s*ps_num for s in range(mss_num) if s > 0])
        Fit.error(", ".join(map(str,inds) ))
    except:
        print("cannot calculate error")
    return models_s
###PLOTER

def obtain_xys(plots=["eeu"]):
    plot_command = " ".join(plots)
    Plot(plot_command)
    xys_s = {}

    for plotWindow_tmp, plot_type in enumerate(plots):
        plotWindow = plotWindow_tmp+1
        xys = {}
        xs, ys, xe, ye, ys_model, ys_comps = [[]]*6
        for plotGroup in range(1, AllData.nGroups+1):
            xs = Plot.x(plotGroup, plotWindow)
            if plot_type in {"eeu", "ratio", "del"}:
                ys = Plot.y(plotGroup, plotWindow)
                xe = Plot.xErr(plotGroup, plotWindow)
                ye = Plot.yErr(plotGroup, plotWindow)

            if plot_type in {"eeu", "eem"}:
                ys_model = Plot.model(plotGroup, plotWindow)

            # obtain comps in models
            ys_comps = []
            comp_N = 1
            while(True):
                try:
                    ys_tmp = Plot.addComp(comp_N, plotGroup, plotWindow)
                    comp_N += 1
                    # execlude components with only 0
                    if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                        continue
                    ys_comps.append(ys_tmp)
                except:
                    break
            xys[plotGroup] = {"xs": xs, "ys": ys, "xe": xe,
                              "ye": ye, "ys_model": ys_model, "ys_comps": ys_comps}
        xys_s[plot_type]=xys

    return xys_s


def plot_xys(plots=["eeu"], xlim=[], ylims={}, colorlist=["r", "g", "b", "c", "m", "y", "k", "w"]):
    plots_diff = set(plots)-{"eeu", "eem", "ratio", "del"}
    if len(plots_diff) >= 1:
        print(f"{', '.join(list(plots_diff))} are not appropriate")

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    xys_s = obtain_xys(plots=plots)
    subplots = []

    # set height ratios for sublots
    gs = gridspec.GridSpec(len(plots), 1, height_ratios=[
                           2 if s in ["eeu", "eem"] else 1 for s in plots])

    for gs_tmp, plot_type in zip(gs, plots):
        xys = xys_s[plot_type]
        # the fisrt subplot
        ax = plt.subplot(gs_tmp)
        subplots.append(ax)

        # set label
        plt.subplots_adjust(hspace=.0)
        if not gs_tmp == gs[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel("Energy (keV)")
        ylabel_dic = {"eeu": r"photons cm$^\mathdefault{-2}$ s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$",
                      "eem": "$EF_\mathdefault{E}$ (keV$^2$)", "ratio": "ratio", "del": "residual"}
        plt.ylabel(ylabel_dic[plot_type])

        fig.align_labels()

        # set range
        plt.xscale("log")

        if not plot_type in ["ratio", "del"]:
            plt.yscale("log")

        # plot eeu
        for plotGroup, xys_tmp in xys.items():
            xs = xys_tmp["xs"]
            if plot_type in {"eeu", "ratio", "del"}:
                ys = xys_tmp["ys"]
                xe = xys_tmp["xe"]
                ye = xys_tmp["ye"]

            # plt.scatter(xs,ys)
            plt.errorbar(xs, ys, yerr=ye, xerr=xe, capsize=0, fmt="o", markersize=5,
                         ecolor=colorlist[plotGroup], markeredgecolor=colorlist[plotGroup], color="none")
            if plot_type in {"eeu", "eem"}:
                ys_model = xys_tmp["ys_model"]
                plt.plot(xs, ys_model, color=colorlist[plotGroup])

            ys_comps = xys_tmp["ys_comps"]
            for ys_comp in ys_comps:
                plt.plot(xs, ys_comp, linestyle="dotted",
                         color=colorlist[plotGroup])
            # plt.plot()
    return fig, subplots, xys_s
