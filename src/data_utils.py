import pandas as pd
from config import DATA_PATH, TARGET, RANDOM_STATE, TEST_SIZE

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def split_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y

def basic_summary(df):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "failure_count": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
    }
