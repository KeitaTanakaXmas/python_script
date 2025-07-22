import numpy as np
from astropy import constants as const

class LoadLog:

    def __init__(self,filename:str):
        self.filename = filename
        print('-------------------------')
        print(f'Log filename = {self.filename}')

    def keysearch(self,keyword:str):
        """
        Output specific word column from logfile.
        keyword : keyword
        """
        with open(self.filename, mode='r') as f:
            for sline in f:
                if keyword in sline:
                    print("--------------------")
                    print(f'Searching {keyword}')
                    self.target = sline
                    print(self.target)

    def wordcut(self,start:int,end:int):
        print(self.target[start:end])
        self.value = float(self.target[start:end])

    def fileout(self):
        filename = "nh_result.txt"
        np.savetxt(filename,np.array([self.value/1e+22]))

LL = LoadLog('nh.log')
LL.keysearch('average')
LL.wordcut(52,60)
LL.fileout()


