
from load_data import load_data
from preprocess import *
from features_target import *
from models_compare import compare_models
from plots import *
from plots_models import *
from train_test import evaluate_model
import pandas as pd

# Carregar dados
df = load_data("bmw_global_sales_2018_2025.csv")

# Separar features e target
X, y, le = features_target(df)

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
print("Best Accuracy:{:.2f}".format(best_model_row["Accuracy"]))

# Avaliação final no conjunto de teste
evaluate_model(best_model, X_test, y_test)

# Plots gerais
plot_model_comparison(results_df)
plot_models_results(trained_models, X_test, y_test)
plot_feature_importance_models(trained_models)

#save e load model
from save import save_model, load_model
#save_model(best_model, le) 
load_model(X_test)