import pandas as pd
from joblib import dump, load
import os.path


models_path = "trained_models"


def store_model(model, file_path):
    path = os.path.join(models_path, file_path)
    dump(model, path)


def load_model(file_path):
    path = os.path.join(models_path, file_path)
    try:
        model = load(path)
    except FileNotFoundError:
        return None

    return model


def print_all_results():
    results = pd.read_csv('results/results.csv')
    print(results.to_string())


