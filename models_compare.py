from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def compare_models(preprocess, X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "SVM": SVC(probability=True, kernel='rbf', C=10, gamma=0.1),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {acc:.4f}")
        results.append({"Model": name, "Accuracy": acc})
        trained_models[name] = pipe

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    return results_df, trained_models