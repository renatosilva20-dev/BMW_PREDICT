import pandas as pd
from sklearn.preprocessing import LabelEncoder

def features_target(df):
    df["Market_Trend"] = df["Year"] * df["Month"]

    df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)

    df["Sales_prev_year"] = df["Units_Sold"].shift(12)
    df["Sales_pct_change"] = df["Units_Sold"].pct_change(periods=12)
    df["MA3"] = df["Units_Sold"].rolling(window=3).mean()
    df["Growth"] = df["Units_Sold"].diff(periods=12) / df["Units_Sold"].shift(12)

    df["sales_classification"] = df["Units_Sold"].apply(
        lambda x: "High" if x > df["Units_Sold"].mean() else "Low"
    )

    le = LabelEncoder()
    df["sales_classification"] = le.fit_transform(df["sales_classification"])

    df = df.dropna().reset_index(drop=True)

    X = df.drop(["sales_classification", "Units_Sold", "Revenue_EUR"], axis=1)
    y = df["sales_classification"]

    return X, y, le