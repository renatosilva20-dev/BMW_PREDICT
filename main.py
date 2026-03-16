from load_data import load_data
from preprocess import preprocess
from features_target import features_target
from models_compare import compare_models
from plots import plot_class_distribution
from plots_models import plot_model_comparison, plot_models_results, plot_feature_importance_models, plot_tree_models
from train_test import evaluate_model

import pandas as pd

# Carregar dados
df = load_data("bmw_global_sales_2018_2025.csv")

# Separar features e target
X, y = features_target(df)

# Dividir treino e teste preservando ordem temporal
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Pré-processamento
t_preprocess = preprocess()

# Comparar modelos
results_df, trained_models = compare_models(
    t_preprocess,
    X_train,
    X_test,
    y_train,
    y_test
)
print("\nModel Comparison:")
print(results_df)

# Selecionar melhor modelo pela maior accuracy
best_model_row = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
best_model = trained_models[best_model_name]
print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_model_row["Accuracy"])

# Avaliação final no conjunto de teste
evaluate_model(best_model, X_test, y_test)

# Plots gerais
plot_class_distribution(y)
plot_model_comparison(results_df)
plot_models_results(trained_models, X_test, y_test)
plot_feature_importance_models(trained_models)
plot_tree_models(trained_models)