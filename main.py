from load_data import *
from preprocess import *
from train_test import *
from sklearn.model_selection import train_test_split
from features_target import *
from plots import *

# carregar dados
df = load_data("bmw_global_sales_2018_2025.csv")

# separar features e target
X, y = features_target(df)

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# criar preprocessamento
t_preprocess = preprocess()

# criar pipeline
t_pipeline = create_pipeline()

# treinar modelo
t_model = train_model(t_pipeline, X_train, y_train)

# avaliar modelo
model(t_model, X_test, y_test)

#Plotar os dados
plot_confusion_matrix(t_model, X_test, y_test)

plot_feature_importance(t_model)

plot_class_distribution(y)