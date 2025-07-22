from pytes import Util, Filter, Analysis
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks


def risefilter(t, peaks, R, L):

    def cutoff(t=t, R=R, L=L):
        return 1-np.exp(-R*t/L)

    filter_cutoff = np.hstack((zeros(int(peaks)),cutoff(t)[:-int(peaks)]))
    return filter_cutoff

pulsepath = 'run011_rowp.fits'
t, p = Util.fopen(pulsepath)

filename = 'Abs5um_1.txt'
data = np.loadtxt(filename)
t_sim = data[:,0] #ms
t_sim = t_sim/1e3 #s
i_sim = data[:,4] #uA

func = interpolate.interp1d(t_sim, i_sim, kind='slinear', fill_value='extrapolate')
peaks, _ = find_peaks(np.diff(func(t)), height=2)
i_resampl = func(t) - func(t)[int(peaks-1)]



