import numpy as np
from pytes import Filter
from pylab import figure,plot,hist,cla,title,tight_layout,xlabel,ylabel,axvline,axhline,axvline,pause,semilogy,loglog,savefig,legend,xlim,subplots

class Thickness:

    def __init__(self, ):
        pass

    def _peaksarch_(self, data, sigma=2):

        a = Filter.median_filter(data,sigma=sigma)
        a_max = data[a].max()
        a_min = data[a].min()
        
        mask = (data > a_max) | (data < a_min)

        b = Filter.median_filter(data[mask],sigma=sigma)

        return a, mask, b

    def _cal_t_(self, data, sigma):
        a, mask, b = self._peaksarch_(data=data, sigma=sigma)

        a_ave = np.average(data[a])        
        a_err = np.std(data[a])

        b_ave = np.average(data[mask][b])        
        b_err = np.std(data[mask][b])
        return a_ave, a_err, b_ave, b_err

    def _showData_(self, data, sigma, d, d_err):
        a, mask, b = self._peaksarch_(data=data, sigma=sigma)
        a_ave, a_err, b_ave, b_err = self._cal_t_(data=data, sigma=sigma)
        if a_ave>b_ave:
            bottom  = b_ave
            bo_err  = b_err
            top     = a_ave
            top_err = a_err
        else:
            bottom  = a_ave
            bo_err  = a_err
            top     = b_ave
            top_err = b_err  

        print(f'Bottom    {bottom:.2f} +- {bo_err:.2f}')
        print(f'Top       {top:.2f} +- {top_err:.2f}')
        print(f'Thickness {d:.2f} +- {d_err:.2f}')


    def thickness(self, data, sigma=2):

        a_ave, a_err, b_ave, b_err = self._cal_t_(data, sigma)
        d = abs(a_ave-b_ave)
        d_err = np.sqrt(a_err**2+b_err**2)
        self._showData_(data,sigma, d, d_err)

        return d, d_err

    def plotThickness(self, data, sigma, hr):
        a, mask, b = self._peaksarch_(data=data, sigma=sigma)
        fig, ax = subplots(1,1)
        x = np.arange(len(data))*hr
        ax.plot(x, data,'k-')
        ax.plot(x[a], data[a],'r.')
        ax.plot(x[mask][b], data[mask][b],'r.')
        ax.set_xlabel(r'$\rm Width\ (\mu m)$', fontsize=16)
        ax.set_ylabel(r'$\rm Thickness\ (nm)$', fontsize=16)
        #ax.tight_layout()
        return fig, ax


