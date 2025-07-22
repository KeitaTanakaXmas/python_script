import pandas as pd


class Data:

    def __init__(self,filename):

    def load_csv(self,file:str,word:str):
        """_summary_

        Args:
            file (str): csv file includeing fitting result. ex) "test.csv"
            word (str): specific word in csv file. ex) "EM"

        Returns:
            pandas dataframe : df[obsid:str,word:str]
            df is sorted by obsid.
        """

        df_header = pd.read_csv(file, index_col=0, dtype={0: str,1: str})
        df = df_header[["obsid",word]]
        df = df.sort_values("obsid")
        print(df["obsid"])
        return df