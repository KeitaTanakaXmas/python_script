import numpy as np
import glob
import matplotlib.pyplot as plt
from dataclasses import field, dataclass

class Analysis:
    def __init__(self) -> None:
        pass
    
    def load_data(self,filename:str):
        self.f    = np.loadtxt(filename, comments='%')
        self.t    = self.f[:,0]
        self.Tsou = self.f[:,1]
        self.Tabs = self.f[:,2]
        self.Ttes = self.f[:,3]
        self.Tmem = self.f[:,4]
        self.I    = self.f[:,5]
        print(f'load data from {filename}')

    def cal(self):
        print(self.Ttes[10])
        self.Ttes -= self.Ttes[10]
        print(np.max(self.Ttes))

    def plotter(self):
        files = sorted(glob.glob('*.txt'))
        labels = ['50um','100um','120um','150um','200um','400um']
        for file,label in zip(files,labels):
            self.load_data(file)
            self.cal()
            print(self.I[10])
            self.I -= self.I[10]
            plt.plot(self.t,self.I,label=label)
        plt.show()