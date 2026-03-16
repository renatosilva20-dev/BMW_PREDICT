def features_target(df):

    df["sales_classification"] = df["Units_Sold"].apply(
        lambda x: "High" if x > df["Units_Sold"].median() else "Low"
    )

    X = df.drop(["sales_classification", "Units_Sold", "Revenue_EUR"], axis=1)

    y = df["sales_classification"]

    return X, y