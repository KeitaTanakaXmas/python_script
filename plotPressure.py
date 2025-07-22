#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Pressure:
    def __init__(self):
        self.fi = glob.glob('*.CSV')
        if len(self.fi)==0:
            self.fi = glob.glob('*.csv')

    def Dataload(self, filename):
        self.figfile = filename[:-4]+'.pdf'
        self.data = np.loadtxt(filename,skiprows=13,delimiter=',',encoding='shift-jis',converters={0:datestr2num})

    def PlotPressure(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(num2date(self.data[:,0]), self.data[:,2], color='k', label=r'$IG\ Pressure$')
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_xlabel(r'$\rm Time$', fontsize=16)
        ax1.set_ylabel(r'$\rm Pressure\ (Pa)$', fontsize=16)
        ax1.semilogy()
        ax1.legend(loc=2)
        ax1.grid(which='both')
        ax2.plot(num2date(self.data[:,0]), self.data[:,6], label=r'$Ti\ power$')
        ax2.plot(num2date(self.data[:,0]), self.data[:,10], label=r'$Au\ power$')
        ax2.set_ylabel(r'$\rm OUT\ PUT(\%)$', fontsize=16)
        ax2.legend(loc=1)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.tight_layout()
        fig.savefig(self.figfile)

    def AutoPlot(self):
        self.fi = glob.glob('*.CSV')
        if len(self.fi)==0:
            self.fi = glob.glob('*.csv')
        for i in self.fi:
            self.Dataload(filename=i)
            self.PlotPressure()
