import pandas as pd
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("arquivo carregado com sucesso")
        return df
    except FileNotFoundError:
        error = print("Arquivo nao encontrado")
        return error
    except Exception as e:
        exc = print(f"Erro ao carregar o arquivo: {e}")
        return exc