from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_random_forest_tree(model, tree_index=0):

    rf_model = model.named_steps["model"]

    tree = rf_model.estimators_[tree_index]

    plt.figure(figsize=(20,10))

    plot_tree(
        tree,
        filled=True,
        max_depth=3,
        fontsize=10
    )

    plt.title(f"Random Forest Tree {tree_index}")

    plt.show()


def plot_model_comparison(results_df):

    plt.figure(figsize=(8,5))

    sns.barplot(
        x="Accuracy",
        y="Model",
        data=results_df
    )

    plt.title("Model Comparison")

    plt.show()


def plot_models_results(trained_models, X_test, y_test):

    for name, model in trained_models.items():

        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix - {name}")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.show()


def plot_feature_importance_models(trained_models):

    for name, model in trained_models.items():

        if hasattr(model.named_steps["model"], "feature_importances_"):

            importances = model.named_steps["model"].feature_importances_

            feature_names = model.named_steps["preprocess"].get_feature_names_out()

            df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(8,5))

            plt.barh(df["Feature"][:15], df["Importance"][:15])

            plt.title(f"Feature Importance - {name}")

            plt.gca().invert_yaxis()

            plt.show()


def plot_tree_models(trained_models):

    for name, model in trained_models.items():

        if hasattr(model.named_steps["model"], "estimators_"):

            tree = model.named_steps["model"].estimators_[0]
            feature_names = model.named_steps["preprocess"].get_feature_names_out()

            plt.figure(figsize=(18,8))

            plot_tree(
                tree,
                filled=True,
                max_depth=3,
                fontsize=9,
                feature_names=feature_names
            )

            plt.title(f"Tree Visualization - {name}")

            plt.show()