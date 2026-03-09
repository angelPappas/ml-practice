import pandas as pd

SELECTED_FEATURES = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
]

TARGET = "Price"


def preprocess(df: pd.DataFrame):
    # Handle missing values
    df = df.dropna(axis=0)

    # Select features
    X = df[SELECTED_FEATURES]
    y = df[TARGET]

    return X, y
