import pandas as pd

class GetData:

    @staticmethod
    def get_raw_df(path):
        df = pd.read_csv(path)
        return df

