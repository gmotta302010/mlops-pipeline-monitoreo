import sys
import tarfile
from datetime import datetime
import time
import subprocess
import numpy as np
import pickle
import pandas as pd
import os
import json
import glob

def preprocess_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['int16', 'int32', 'int64']:
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['float16', 'float32', 'float64']:
            # floats con solo 4 decimales
            df[col] = df[col].astype(float).round(4)
    return df

def find_latest_model_folder(base_dir):
    latest_folder = None
    latest_timestamp = None
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                folder_timestamp = datetime.strptime(item, "%Y-%m-%d_%H-%M-%S")
                if latest_timestamp is None or folder_timestamp > latest_timestamp:
                    latest_timestamp = folder_timestamp
                    latest_folder = item_path
            except ValueError:
                pass
    if latest_folder is not None:
        print(f"Modelo: {latest_folder}")
        return latest_folder
    else:
        print(f"Error: No se encontro ningún modelo en {base_dir}")

def load_model_and_metadata(models_dir):
    latest_folder_path = find_latest_model_folder(models_dir)
    model_path = glob.glob(f'{latest_folder_path}/*.pkl')[0]
    metadata_path = glob.glob(f'{latest_folder_path}/*.json')[0]

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return None, None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Archivo de metadatos no encontrado en {metadata_path}")
        return None, None

    return model, metadata

def perform_inference(data_path, model, metadata):
    try:
        df_inference = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

    ml_name = metadata.get("ml_name")
    if ml_name is None:
        print("Error: 'ml_name' no encontrado en la metadata.")
        return None

    df_inference = preprocess_dataframe(df_inference)

    try:
        if ml_name == 'xgb':
            # XGBoost needs DMatrix or numpy array for predict_proba
            predictions = model.predict_proba(df_inference)[:, 1]
        elif ml_name == 'lgbm':
            predictions = model.predict_proba(df_inference)[:, 1]
        elif ml_name == 'catb':
            predictions = model.predict_proba(df_inference)[:, 1]
        else:
            print(f"Error: Algoritmo de ML '{ml_name}' no compatible para la inferencia.")
            return None
    except Exception as e:
        print(f"Error durante la inferencia con el tipo de modelo {ml_name}: {e}")
        return None
    return pd.DataFrame({'predictions': predictions})

def main(models_dir, preprocessed_data_path, output_dir):
    filename = preprocessed_data_path.split('/')[-1] # 'vars_12_extrac.csv'
    parts = filename.replace('.csv', '').split('_') # ['vars', '12', 'extrac']

    if len(parts) < 3:
        print(f"Error: Formato de nombre de archivo inesperado en la ruta de datos: {preprocessed_data_path}")
        return

    partition = parts[1]
    model_name = parts[2]
    model, metadata = load_model_and_metadata(models_dir)

    if model is None or metadata is None:
        return
    predictions_df = perform_inference(preprocessed_data_path, model, metadata)

    if predictions_df is None:
        return

    output_filename = f'inference_{model_name}_{partition}.csv'
    output_path = f'{output_dir}/{output_filename}'

    predictions_df.to_csv(output_path, index=False, header=True)
    print(f"Predictions guardadas en: {output_path}")