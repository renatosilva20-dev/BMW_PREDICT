from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    return pipeline


def model(modelo, X_test, y_test):

    predictions = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    return accuracy


def feature_importance(modelo, X):

    model = modelo.named_steps["model"]

    importances = model.feature_importances_

    feature_names = modelo.named_steps["preprocess"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n")
    print(importance_df.head(15))

    return importance_df