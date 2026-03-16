from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess():
    numeric_features = [
        "Year",
        "Month",
        "Avg_Price_EUR",
        "BEV_Share",
        "Premium_Share",
        "GDP_Growth",
        "Fuel_Price_Index",
        "Price_per_GDP",
        "EV_Premium",
        "Fuel_vs_EV",
        "Market_Trend",
        "Sales_prev_year",
        "Sales_pct_change",
        "MA3",
        "Growth"
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

def create_pipeline(model=None, feature_selector=True):
    if model is None:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    steps = [("preprocess", preprocess())]

    if feature_selector:
        steps.append(("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=100))))

    steps.append(("model", model))

    pipe = Pipeline(steps)
    return pipe