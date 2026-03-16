from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_model(pipeline, X_train, y_train, param_grid=None, n_splits=3):
    if param_grid is None:
        param_grid = {
            "model__n_estimators": [200, 300],
            "model__max_depth": [5, 10],
            "model__learning_rate": [0.05, 0.1]
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_

def evaluate_model(modelo, X_test, y_test):
    predictions = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    return accuracy

def feature_importance(modelo):
    try:
        model = modelo.named_steps["model"]
        importances = model.feature_importances_
        feature_names = modelo.named_steps["preprocess"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        print("\nFeature Importance:\n", importance_df.head(15))
        return importance_df
    except AttributeError:
        print("\nFeature importance not available for this model.")
        return None