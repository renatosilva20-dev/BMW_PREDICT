
import joblib
from features_target import *
from train_test import *
from preprocess import *
from plots import *
from models_compare import * 
def save_model(model_trained, le):
    resultado = {
        "model": model_trained,
        "label_encoder": le
    }

    joblib.dump(resultado, "models/bmw_model.pkl")
    print("Modelo + encoder salvos com sucesso")
    
def load_model(X_test):
    try:
        resultado_final = joblib.load("models/bmw_model.pkl")

        model = resultado_final["model"]
        le = resultado_final["label_encoder"]

        print("Modelo carregado com sucesso.")

        pred = model.predict(X_test)
        pred_label = le.inverse_transform(pred)
        #print(pred_label)
        print("Predições feitas com o modelo carregado.")
        evaluate_model(model, X_test, pred)
        plot_confusion_matrix(model, X_test, pred)

        return model

    except FileNotFoundError:
        print("Nenhum modelo encontrado.")
        return None