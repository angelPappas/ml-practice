import pandas as pd


def load_raw_data() -> pd.DataFrame:
    path = "/home/angel/Code/ml-training/ml-practice/kaggle_IML/data/raw/melb_data.csv"
    return pd.read_csv(path)
