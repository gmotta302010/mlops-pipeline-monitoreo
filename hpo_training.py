from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
import time
import json
import sys
import os
import platform
import subprocess
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'catboost==1.2.8'])

import catboost as catb

# Espacio de búsqueda para XGBoost
xgb_param_grid = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.7, 0.3), # (rango de 0.7 a 1.0)
    'colsample_bytree': uniform(0.7, 0.3), # (rango de 0.7 a 1.0)
}

# Espacio de búsqueda para LightGBM
lgbm_param_grid = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 50),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.7, 0.3), # (rango de 0.7 a 1.0)
    'colsample_bytree': uniform(0.7, 0.3), # (rango de 0.7 a 1.0)
}

# Espacio de búsqueda para CatBoost
catb_param_grid = {
    'iterations': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'depth': randint(3, 10),
    'subsample': uniform(0.7, 0.3), # (rango de 0.7 a 1.0)
    'l2_leaf_reg': uniform(1, 10), # Regularización L2
}

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

def save_model(model, ml_name, performance, params, save_dir):
    # Save the model
    model_filename = f'{ml_name}_model.pkl'
    model_path = f'{save_dir}/{model_filename}'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {model_path}")

    # Save metadata
    metadata_filename = f'{ml_name}_metadata.json'
    metadata_path = f'{save_dir}/{metadata_filename}'

    # Get library versions
    library_versions = {
        'xgboost': xgb.__version__ if ml_name == 'xgb' else None,
        'lightgbm': lgb.__version__ if ml_name == 'lgbm' else None,
        'catboost': catb.__version__ if ml_name == 'catb' else None,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': platform.version()}

    metadata = {
        'ml_name': ml_name,
        'performance': performance,
        'hyperparameters': params,
        'library_versions': library_versions,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    # Limpiar parámetros para la serialización JSON (por ejemplo, manejar tipos no serializables)
    def clean_params(p):
        cleaned = {}
        for k, v in p.items():
            try:
                json.dumps(v) # Check if serializable
                cleaned[k] = v
            except (TypeError, OverflowError):
                cleaned[k] = str(v)
        return cleaned

    metadata['hyperparameters'] = clean_params(metadata['hyperparameters'])
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata guardada en: {metadata_path}")

def main(train_path, test_path, model_save_dir):
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = f'{model_save_dir}/{folder_name}'
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Directorio de modelos guardado en: {model_save_dir}")
    # cargar data
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del conjunto de datos. Revisar las rutas: {train_path}, {test_path}")
        return

    # Preprocess data
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # Separar features y target
    X_train = train_df.drop(columns=[train_df.columns[0]])
    y_train = train_df[train_df.columns[0]]
    X_test = test_df.drop(columns=[test_df.columns[0]])
    y_test = test_df[test_df.columns[0]]

    # Asegurarse que las columnas de train y test coincidan
    train_cols = X_train.columns
    test_cols = X_test.columns

    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0

    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0

    X_test = X_test[train_cols]

    # Model training and evaluation
    models = {
        'xgb': {
            'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'param_grid': xgb_param_grid, # <-- MODIFICADO
            'performance': {}
        },
        'lgbm': {
            # Se añade verbose=-1 para silenciar LightGBM durante el HPO
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1), # <-- MODIFICADO
            'param_grid': lgbm_param_grid, # <-- MODIFICADO
            'performance': {}
        },
        'catb': {
            'model': catb.CatBoostClassifier(verbose=0, random_state=42),
            'param_grid': catb_param_grid, # <-- MODIFICADO
            'performance': {}
        }
    }

    best_ml_name = None
    best_auc_test = -np.inf
    best_decay = np.inf

    # --- INICIO: Bucle de Entrenamiento con HPO ---
    for ml_name, model_info in models.items():
        print(f"\n--- Iniciando HPO para: {ml_name.upper()} ---")
        base_model = model_info['model']
        param_grid = model_info['param_grid']

        hpo_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15, 
            cv=3, # Cross-Validation - 3 es rápido.
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1  # Muestra feedback del HPO
        )

        start_time = time.time()

        # Ejecutar la búsqueda de hiperparámetros
        hpo_search.fit(X_train, y_train)

        # model = model_info['model'] # reemplazado por model = hpo_search.best_estimator_

        # if ml_name == 'catb':
             # CatBoost puede manejar diferentes tipos de datos, pero usemos los datos preprocesados
             # model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=0)
        # else:
            # model.fit(X_train, y_train)

        end_time = time.time()

        training_time = end_time - start_time

        # Obtener el *mejor* modelo encontrado por HPO
        model = hpo_search.best_estimator_
        # Obtener los *mejores* parámetros encontrados
        best_params = hpo_search.best_params_

        # --- El resto del código de evaluación sigue igual ---
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        auc_train = roc_auc_score(y_train, y_train_pred_proba)
        auc_test = roc_auc_score(y_test, y_test_pred_proba)
        decay = ((auc_train - auc_test) / auc_train) * 100 if auc_train > 0 else np.inf

        # Guardamos las métricas y los *mejores* parámetros
        models[ml_name]['performance'] = {
            'auc_train': auc_train,
            'auc_test': auc_test,
            'decay_percent': decay,
            'training_time_segs': training_time,
            'best_auc_cv': hpo_search.best_score_ # <-- MÉTRICA ADICIONAL
        }

        models[ml_name]['params'] = best_params

        print(f"Resultados HPO para: {ml_name}")
        print(f"  Mejor AUC en Cross-Validation: {hpo_search.best_score_:.4f}")
        print(f"  Mejores Parámetros: {best_params}")
        print(f"  AUC Train (con mejor modelo): {auc_train:.4f}")
        print(f"  AUC Test (con mejor modelo): {auc_test:.4f}")
        print(f"  Decay (%): {decay:.2f}%")
        print("-" * 20)

        # Check if this model is the champion
        if auc_test > best_auc_test and decay < 10:
            best_auc_test = auc_test
            best_decay = decay
            best_ml_name = ml_name

    if best_ml_name:
        print(f"\nModelo finalista: {best_ml_name}")
        save_model(models[best_ml_name]['model'], 
                   best_ml_name, 
                   models[best_ml_name]['performance'], 
                   models[best_ml_name]['params'], 
                   model_save_dir)
    else:
        print("\nNo se encontró ningún modelo campeón que cumpla con los criterios (Decaimiento < 10%).")
