from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def preprocess():

    numeric_features = [
        "Year",
        "Month",
        "Avg_Price_EUR",
        "BEV_Share",
        "Premium_Share",
        "GDP_Growth",
        "Fuel_Price_Index"
    ]

    categorical_features = [
        "Region",
        "Model"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def create_pipeline():

    pipe = Pipeline([
        ("preprocess", preprocess()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    return pipe