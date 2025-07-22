import numpy as np
import matplotlib.pyplot as plt

class SpectrumSimulation:
    def __init__(self) -> None:
        pass

    def make_energy(self,min,max,dE):
        self.energy = np.arange(min,max+dE,dE)

    def powerlaw(self,energy,index,amplitude):
        return amplitude * energy ** index


    def generate_random_energy(self, num_samples, min_energy, max_energy, index):
        # 逆関数法を用いてパワーロー分布に従う乱数を生成
        r = np.random.uniform(size=num_samples)  # 一様乱数生成
        exponent = index + 1
        random_energy = ( (max_energy**exponent - min_energy**exponent) * r + min_energy**exponent ) ** (1/exponent)
        return random_energy

    def plot_powerlaw(self):
        self.make_energy(1,10,0.01)
        print(self.energy)
        print(self.powerlaw(self.energy,-1.4,10))
        plt.plot(self.energy,self.powerlaw(self.energy,-1.4,10))
        random_energy = self.generate_random_energy(1000,1,10,-1.4)
        plt.hist(random_energy,bins=100,histtype='step')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()