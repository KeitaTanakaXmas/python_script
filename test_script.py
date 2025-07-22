import numpy as np
import random
import scipy
from scipy import signal
import bottleneck as bn
from Basic import Plotter
from pytes import Filter
import matplotlib.pyplot as plt


class TESTPY:
    def __init__(self):
        self.P = Plotter()

    def white_noise(self,sampling_rate):
        self.time = np.arange(0,5e-3,sampling_rate)
        self.noise_amplitude = np.random.normal(0.0,1.0,len(self.time))
        #self.P.plotting(self.time,self.noise_amplitude)

    def LPF_GC(self,x,times,sigma):
        sigma_k = sigma/(times[1]-times[0]) 
        kernel = np.zeros(int(round(3*sigma_k))*2+1)
        for i in range(kernel.shape[0]):
            kernel[i] =  1.0/np.sqrt(2*np.pi)/sigma_k * np.exp((i - round(3*sigma_k))**2/(- 2*sigma_k**2))
            
        kernel = kernel / kernel.sum()
        x_long = np.zeros(x.shape[0] + kernel.shape[0])
        x_long[kernel.shape[0]//2 :-kernel.shape[0]//2] = x
        x_long[:kernel.shape[0]//2 ] = x[0]
        x_long[-kernel.shape[0]//2 :] = x[-1]
            
        x_GC = np.convolve(x_long,kernel,'same')
        
        return x_GC[kernel.shape[0]//2 :-kernel.shape[0]//2]


    def generate_spectrum(self,label=None):
        hres = self.time[1] - self.time[0]
        self.noise_spectrum = Filter.power(self.noise_amplitude)
        self.frq = np.fft.rfftfreq(self.noise_amplitude.shape[-1],hres)
        self.len_m = len(self.noise_spectrum)
        self.P.plotting(self.frq,self.noise_spectrum,xlog=True,ylog=True,label=label)        

    def running_filter(self,n):
        self.noise_amplitude = bn.move_mean(self.noise_amplitude,window=n)[n:]
        self.time            = bn.move_mean(self.time,window=n)[n:]
        self.P.plotting(self.time,self.noise_amplitude)

    def ttt(self):
        n = []
        for i in range(0,100):
            self.white_noise(sampling_rate=1e-7)
            n.append(self.noise_amplitude)
        self.noise_amplitude = np.average(n,axis=0)
        self.P.plotting(self.time,self.noise_amplitude,label='100 average noise (Filter off)')
        self.generate_spectrum(label='Noise supectrum (Filter off)')
        lowcut = 1e+4
        fs = 1e+7
        nyq = fs/2
        low = lowcut/nyq
        b, a = signal.butter(2,low,'low')
        self.noise_amplitude = signal.filtfilt(b, a, self.noise_amplitude)
        self.P.plotting(self.time[1:],self.noise_amplitude[1:],label='100 average noise (Filter on)')
        self.generate_spectrum(label='Noise supectrum (Filter on)')
        plt.show()