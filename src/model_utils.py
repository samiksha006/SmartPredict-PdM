"""Reusable feature-engineering and training helpers for SmartPredict-PdM."""
import pandas as pd


def identity_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    air = x["Air temperature [K]"].astype(float)
    process = x["Process temperature [K]"].astype(float)
    speed = x["Rotational speed [rpm]"].astype(float).clip(lower=1e-6)
    torque = x["Torque [Nm]"].astype(float)
    wear = x["Tool wear [min]"].astype(float)
    x["Temperature Difference [K]"] = process - air
    x["Temperature Ratio"] = process / air.clip(lower=1e-6)
    x["Mechanical Power Proxy"] = torque * speed
    x["Torque per Speed"] = torque / speed
    x["Wear per Speed"] = wear / speed
    x["Thermal-Wear Interaction"] = (process - air) * wear
    return x


def lr_schedule(epoch: int) -> float:
    return max(0.01, 0.08 * (0.985 ** epoch))
