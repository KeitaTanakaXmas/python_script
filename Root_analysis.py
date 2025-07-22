import ROOT
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter



# ROOTファイルからヒストグラムのエントリーを取得するクラス
class RootHistogramLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.histogram_entries = {}

    def load_histogram_entries(self):
        root_file = ROOT.TFile.Open(self.file_path)
        if not root_file:
            raise FileNotFoundError(f"File '{self.file_path}' not found")

        keys = root_file.GetListOfKeys()

        for key in keys:
            obj = key.ReadObj()
            if isinstance(obj, ROOT.TH1):
                hist_name = obj.GetName()
                histogram_entries = []
                bins = obj.GetNbinsX()

                for i in range(1, bins + 1):
                    for _ in range(int(obj.GetBinContent(i))):
                        histogram_entries.append(obj.GetBinLowEdge(i))
                
                self.histogram_entries[hist_name] = {'entries': histogram_entries, 'bins': bins}

        root_file.Close()

    def entries2count(self,num):
        num = num - 1
        for i, (key, value) in enumerate(self.histogram_entries.items()):
            if i == num:
                count, bins = np.array(value['entries']), np.array(value['bins'])
                counter = Counter(count)
                count_list = [counter[num] for num in count]
                Energy = list(counter.keys())
                counts = list(counter.values())
                print(Energy, counts)
        return Energy, counts
    
    def Range_selecter(self,numbers_1,numbers_2,lower_limit,upper_limit):
        # 範囲内の数字のみを取り出す
        selected_numbers_1 = [num for num in numbers_1 if lower_limit <= num <= upper_limit]

        # 対応するインデックスの要素を取り出す
        selected_numbers_2 = [numbers_2[i] for i in range(len(numbers_2)) if lower_limit <= numbers_1[i] <= upper_limit]
        print('----------------------------------------')
        print(f'Selected Range = ({lower_limit}, {upper_limit})')
        print(selected_numbers_1,selected_numbers_2)
        return np.array(selected_numbers_1), np.array(selected_numbers_2)
    
    def Normalization(self,numbers):
        # 合計値を計算
        total = sum(numbers)

        # 合計値で全要素を正規化
        normalized_numbers = [num / total for num in numbers]
        print('----------------------------------------')
        print('Normalized')
        print(normalized_numbers)  # 正規化された配列を出力
        return np.array(normalized_numbers)
    
    def print_row(self,data1,data2):
        print('----------------------------')
        for d1,d2 in zip(data1,data2):
            print(f'{d1}, {d2}')

class Auger_loader:
    def __init__(self,file) -> None:
        self.file = file
    def load_file(self):
        with open(self.file, 'r') as file:
            lines = file.readlines()
        
        # 数値を格納するリスト

        number_list = []
        for line in lines:
            # タブで行を分割して、数字の部分を取り出す
            line_values = line.split('\t')
            for value in line_values:
                # 指数表記の文字列を浮動小数点数に変換してリストに追加
                v = value.split(' ')
                numbers = []
                for e,vv in enumerate(v):
                    if e < 4:
                        print(vv)
                        # if '+' in vv:
                        vv = vv.replace('+','e+')
                        # if '-' in vv:
                        vv = vv.replace('-','e-')
                        number = float(vv.replace('D', 'E'))
                        numbers.append(number)
            number_list.append(numbers)
        
        # 数字の配列を出力
        number_list = np.array(number_list)
        print(number_list)  # 読み取られた数字のリストを出力
        return number_list[:,2], number_list[:,3]


class MatplotlibHistogramPlotter:

    def __init__(self, data):
        self.data = data
        self.t    = ["scattered primary particle: energy spectrum",
                     "scattered primary particle: costheta distribution",
                     "charged secondaries: energy spectrum",
                     "charged secondaries: costheta distribution",
                     "neutral secondaries: energy spectrum",
                     "neutral secondaries: costheta distribution"]

    def plot_histograms(self):
        fig, axs = plt.subplots(2, 3, figsize=(6, 3), dpi=300)  # サイズと解像度の設定

        for i, (key, value) in enumerate(self.data.items()):
            ax = axs.flat[i]
            ax.hist(value['entries'], bins=value['bins'], histtype='step', linewidth=0.5)  # ヒストグラムのスタイルを変更
            ax.set_title(self.t[i], fontsize=5)  # タイトルのフォントサイズを設定
            ax.tick_params(axis='both', which='major', labelsize=4)  # 目盛りのフォントサイズを設定
            ax.set_yscale('log')
            #ax.grid(linestyle='dashed')

        plt.tight_layout()  # レイアウト調整
        plt.savefig('histograms.png', dpi=300, bbox_inches='tight')  # 図をファイルに保存（任意）
        plt.show()

    def plot_histogram_single(self): # サイズと解像度の設定
        fig  = plt.figure(figsize=(8,6))
        ax  = plt.subplot(111)
        for i, (key, value) in enumerate(self.data.items()):
            if i == 4:
                ax.hist(value['entries'], bins=value['bins'], histtype='step', linewidth=1)  # ヒストグラムのスタイルを変更
                ax.set_title(self.t[i], fontsize=15)  # タイトルのフォントサイズを設定
                ax.tick_params(axis='both', which='major', labelsize=15)  # 目盛りのフォントサイズを設定
                ax.set_yscale('log')
                ax.grid(linestyle='dashed')
        plt.xlabel('Energy [keV]',fontsize=15)
        plt.ylabel('Count',fontsize=15)
        plt.tight_layout()  # レイアウト調整
        plt.savefig('histogram.png', bbox_inches='tight')  # 図をファイルに保存（任意）
        plt.show()

# 使用例
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'op':
    # オプション1に対する処理

        root_loader = RootHistogramLoader("photoelec.root")
        root_loader.load_histogram_entries()

        plotter = MatplotlibHistogramPlotter(root_loader.histogram_entries)
        plotter.plot_histograms()
    else:
    # 別の処理
        root_loader = RootHistogramLoader("photoelec.root")
        root_loader.load_histogram_entries()
        E, c = root_loader.entries2count(num=3)
        E, c = root_loader.Range_selecter(E,c,lower_limit=0,upper_limit=7)
        c = root_loader.Normalization(c)
        root_loader.print_row(E,c)
        E_l, c_l = Auger_loader('test.txt').load_file()
        E_l, c_l = root_loader.Range_selecter(E_l,c_l,lower_limit=100,upper_limit=7e+3)
        c_l = root_loader.Normalization(c_l)
        print(c_l,c_l.shape)
        c_l = c_l[np.argsort(E_l)]
        E_l = E_l[np.argsort(E_l)]*1e-3
        res = np.vstack((E,c))
        res_l = np.vstack((E_l,c_l))
        print('-----------------------------------')
        print('G4: Energy[eV], Prob, EADL: Energy[eV], Prob')
        print(res.T)
        print(res_l.T)
        # np.savetxt(X=res.T,fname='G4_res_Si_Auger.txt')
        # np.savetxt(X=res_l.T,fname='EADL_Si_Auger.txt')
        # plt.step(res.T[:,0],res.T[:,1],label='G4 simulation')
        # plt.step(res_l.T[:,0],res_l.T[:,1], label='EADL')
        # plt.yscale('log')
        # plt.xlabel('Energy [keV]')
        # plt.ylabel('Transition probability')
        # plt.grid(linestyle='dashed')
        # #plt.title('Auger electron')
        # plt.legend()
        # plt.show()
        plotter = MatplotlibHistogramPlotter(root_loader.histogram_entries)
        plotter.plot_histogram_single()        



