import pandas as pd

class DataLoader:

    @staticmethod
    def get_raw_df(path):
        df = pd.read_csv(path)
        return df

