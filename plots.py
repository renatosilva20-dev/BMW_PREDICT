import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low","High"],
        yticklabels=["Low","High"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()


def plot_feature_importance(model):

    importances = model.named_steps["model"].feature_importances_

    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))

    plt.barh(df["Feature"][:15], df["Importance"][:15])

    plt.gca().invert_yaxis()

    plt.title("Top 15 Feature Importance")

    plt.show()


def plot_class_distribution(y):

    sns.countplot(x=y)

    plt.title("Sales Classification Distribution")

    plt.show()