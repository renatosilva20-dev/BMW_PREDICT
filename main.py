from load_data import *
from preprocess import *
from sklearn.model_selection import train_test_split
from features_target import *

from plots import *
from models_compare import *
from plots_models import *
from train_test import model

#  Carregar dados


df = load_data("bmw_global_sales_2018_2025.csv")

# Separar features e target

X, y = features_target(df)

# Dividir treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#  Preprocessamento

t_preprocess = preprocess()

#  Comparar modelos

results_df, trained_models = compare_models(
    t_preprocess,
    X_train,
    X_test,
    y_train,
    y_test
)

print("\nModel Comparison:")
print(results_df)

#  Selecionar melhor modelo


best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

#  Avaliação do melhor modelo


model(best_model, X_test, y_test)

#  Plots gerais


plot_class_distribution(y)

plot_model_comparison(results_df)

plot_models_results(trained_models, X_test, y_test)

plot_feature_importance_models(trained_models)

plot_tree_models(trained_models)