import os
import glob
import json
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


EPS = 1e-6


# ============================================================
# Utilidades generales
# ============================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _safe_read_csv(path, sep=None):
    """Lee CSV/TXT de forma tolerante. Para .txt intenta separador pipe."""
    if path is None or not os.path.exists(path):
        return pd.DataFrame()

    try:
        if sep is not None:
            return pd.read_csv(path, sep=sep)
        if str(path).lower().endswith(".txt"):
            return pd.read_csv(path, sep="|")
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep="|")
        except Exception as e:
            print(f"No se pudo leer {path}: {e}")
            return pd.DataFrame()


def _safe_read_parquet_many(files):
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"No se pudo leer parquet {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _find_first(patterns):
    for p in patterns:
        files = glob.glob(p)
        if files:
            return sorted(files)[0]
    return None


def _get_params(payload):
    return payload.get("params", {})


def _current_partition(payload):
    return str(_get_params(payload).get("partition", ""))


def _baseline_partition(payload):
    params = _get_params(payload)
    return str(params.get("baseline_partition", params.get("partition_ref", "")))


def _model_name(payload):
    return str(_get_params(payload).get("model_name", "extrac"))


def _monitoring_dir(payload):
    base = payload.get("MONITORING_DIR") or os.path.join(payload.get("DIR_OUTPUT", "."), "monitoring")
    return _ensure_dir(base)


# ============================================================
# Lectura por etapa
# ============================================================

def read_raw_stage(payload, partition):
    raw_dir = payload.get("DIR_RAWDATA", "")
    if not raw_dir:
        return pd.DataFrame()

    # Caso principal del curso: p<period>_extrac.csv
    candidates = [
        os.path.join(raw_dir, f"p{partition}_extrac.csv"),
        os.path.join(raw_dir, f"*{partition}*.csv"),
    ]
    path = _find_first(candidates)
    if path:
        return _safe_read_csv(path)

    # Fallback: si no hay CSV, intenta parquet
    parquet_files = glob.glob(os.path.join(raw_dir, f"*{partition}*.parquet"))
    if not parquet_files:
        parquet_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    df = _safe_read_parquet_many(parquet_files)

    if not df.empty and "partition" in df.columns and partition:
        df_part = df[df["partition"].astype(str) == partition]
        if not df_part.empty:
            return df_part.reset_index(drop=True)
    return df


def read_preprocessed_stage(payload, partition):
    model = _model_name(payload)
    processed = payload.get("DIR_PROCESSED", "")
    candidates = [
        os.path.join(processed, "preprocessed", f"vars_{partition}_{model}.csv"),
        os.path.join(processed, "preprocessed", f"*{partition}*{model}*.csv"),
        os.path.join(processed, "training_data", "preprocessed", f"*vars_*{model}.csv"),
    ]
    return _safe_read_csv(_find_first(candidates))


def read_score_stage(payload, partition):
    model = _model_name(payload)
    score_dir = payload.get("SCORE_DIR", "")
    candidates = [
        os.path.join(score_dir, f"inference_{model}_{partition}.csv"),
        os.path.join(score_dir, f"*{model}*{partition}*.csv"),
    ]
    return _safe_read_csv(_find_first(candidates))


def read_postprocessed_stage(payload, partition):
    model = _model_name(payload)
    out_dir = payload.get("DIR_OUTPUT", "")
    candidates = [
        os.path.join(out_dir, f"scr_{model}_{partition}.txt"),
        os.path.join(out_dir, f"*{model}*{partition}*.txt"),
        os.path.join(out_dir, f"*{model}*{partition}*.csv"),
    ]
    return _safe_read_csv(_find_first(candidates))


# ============================================================
# Data drift: PSI por variable
# ============================================================

def _psi_numeric(ref, act, quantils=10):
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").dropna().values
    act = pd.to_numeric(pd.Series(act), errors="coerce").dropna().values

    if len(ref) == 0 or len(act) == 0:
        return np.nan

    # Si la variable es constante, crea bins artificiales
    if np.nanmin(ref) == np.nanmax(ref):
        breakpoints = np.array([-np.inf, np.nanmax(ref), np.inf])
    else:
        breakpoints = np.nanpercentile(ref, np.linspace(0, 100, quantils + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            breakpoints = np.array([-np.inf, np.nanmedian(ref), np.inf])
        else:
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf

    ref_pct = np.histogram(ref, bins=breakpoints)[0] / max(len(ref), 1)
    act_pct = np.histogram(act, bins=breakpoints)[0] / max(len(act), 1)
    ref_pct = np.where(ref_pct <= 0, EPS, ref_pct)
    act_pct = np.where(act_pct <= 0, EPS, act_pct)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def _psi_categorical(ref, act):
    ref = pd.Series(ref).fillna("__MISSING__").astype(str)
    act = pd.Series(act).fillna("__MISSING__").astype(str)

    if len(ref) == 0 or len(act) == 0:
        return np.nan

    cats = sorted(set(ref.unique()).union(set(act.unique())))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    act_pct = act.value_counts(normalize=True).reindex(cats, fill_value=0).values
    ref_pct = np.where(ref_pct <= 0, EPS, ref_pct)
    act_pct = np.where(act_pct <= 0, EPS, act_pct)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def _drift_one_feature(args):
    feature, ref_values, act_values, quantils = args
    ref_s = pd.Series(ref_values)
    act_s = pd.Series(act_values)

    is_numeric = pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(act_s)
    metric = _psi_numeric(ref_s, act_s, quantils) if is_numeric else _psi_categorical(ref_s, act_s)

    if pd.isna(metric):
        level = "SIN_DATOS"
    elif metric < 0.10:
        level = "SIN_DRIFT"
    elif metric < 0.25:
        level = "DRIFT_MODERADO"
    else:
        level = "DRIFT_ALTO"

    return {
        "feature": feature,
        "psi": metric,
        "drift_level": level,
        "dtype": "numeric" if is_numeric else "categorical",
        "ref_missing_rate": float(ref_s.isna().mean()) if len(ref_s) else np.nan,
        "act_missing_rate": float(act_s.isna().mean()) if len(act_s) else np.nan,
    }


def data_drift(df_ref, df_actual, stage_name, quantils=10, workers=2):
    if df_ref is None or df_actual is None or df_ref.empty or df_actual.empty:
        return pd.DataFrame([{
            "stage": stage_name,
            "feature": "__DATAFRAME__",
            "psi": np.nan,
            "drift_level": "SIN_DATOS",
            "dtype": "NA",
            "ref_missing_rate": np.nan,
            "act_missing_rate": np.nan,
        }])

    common_cols = [c for c in df_actual.columns if c in df_ref.columns]
    tasks = [(c, df_ref[c], df_actual[c], quantils) for c in common_cols]
    rows = []

    if workers and workers >= 2 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_drift_one_feature, t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
    else:
        rows = [_drift_one_feature(t) for t in tasks]

    out = pd.DataFrame(rows)
    out.insert(0, "stage", stage_name)
    return out.sort_values(["drift_level", "psi"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# SHAP / importancia de variables mes a mes
# ============================================================

def _find_latest_model_folder(model_dir):
    if not model_dir or not os.path.exists(model_dir):
        return None

    folders = [f for f in glob.glob(os.path.join(model_dir, "*")) if os.path.isdir(f)]
    if not folders:
        return None

    def _score_folder(f):
        name = os.path.basename(f)
        try:
            return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
        except Exception:
            return datetime.fromtimestamp(os.path.getmtime(f))

    return sorted(folders, key=_score_folder)[-1]


def _load_model(model_dir):
    latest = _find_latest_model_folder(model_dir)
    if latest is None:
        return None, {}, None

    pkl = _find_first([os.path.join(latest, "*.pkl")])
    js = _find_first([os.path.join(latest, "*.json")])

    model = None
    metadata = {}
    if pkl:
        try:
            with open(pkl, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"No se pudo cargar modelo {pkl}: {e}")

    if js:
        try:
            with open(js, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"No se pudo cargar metadata {js}: {e}")

    return model, metadata, latest


def _prepare_features_for_model(df, max_rows=2000, random_state=42):
    if df is None or df.empty:
        return pd.DataFrame()

    X = df.copy()

    # El primer campo puede ser target en training_data; si se llama target, se elimina.
    for col in ["target", "partition", "key_value", "codunicocli"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Solo columnas numéricas/bool para SHAP/modelo.
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)
    X = X.select_dtypes(include=[np.number, "bool"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)

    return X


def _importance_from_shap_or_model(model, X, label):
    if model is None or X.empty:
        return pd.DataFrame(columns=["period", "feature", "importance"])

    # 1) Intento SHAP real
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        vals = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "period": label,
            "feature": X.columns,
            "importance": vals,
            "method": "SHAP_MEAN_ABS"
        }).sort_values("importance", ascending=False)
    except Exception as e:
        print(f"SHAP no disponible, se usará importancia nativa del modelo. Detalle: {e}")

    # 2) Fallback: feature_importances_
    if hasattr(model, "feature_importances_"):
        vals = np.array(model.feature_importances_)
        n = min(len(vals), X.shape[1])
        return pd.DataFrame({
            "period": label,
            "feature": X.columns[:n],
            "importance": vals[:n],
            "method": "MODEL_FEATURE_IMPORTANCE"
        }).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["period", "feature", "importance", "method"])


def shap_monthly_variation(payload, df_ref_pre, df_act_pre):
    model, metadata, model_folder = _load_model(payload.get("MODEL_DIR", ""))
    ref_label = _baseline_partition(payload) or "baseline"
    act_label = _current_partition(payload) or "actual"

    max_rows = int(_get_params(payload).get("shap_sample", 2000))
    X_ref = _prepare_features_for_model(df_ref_pre, max_rows=max_rows)
    X_act = _prepare_features_for_model(df_act_pre, max_rows=max_rows)

    # Alineación de columnas
    common = [c for c in X_act.columns if c in X_ref.columns]
    X_ref = X_ref[common] if common else X_ref
    X_act = X_act[common] if common else X_act

    imp_ref = _importance_from_shap_or_model(model, X_ref, ref_label)
    imp_act = _importance_from_shap_or_model(model, X_act, act_label)

    both = pd.concat([imp_ref, imp_act], ignore_index=True)
    if both.empty:
        return both, pd.DataFrame()

    pivot = both.pivot_table(index="feature", columns="period", values="importance", aggfunc="mean").fillna(0)
    if ref_label not in pivot.columns:
        pivot[ref_label] = 0
    if act_label not in pivot.columns:
        pivot[act_label] = 0

    pivot["abs_change"] = pivot[act_label] - pivot[ref_label]
    pivot["relative_change"] = np.where(
        pivot[ref_label].abs() > EPS,
        pivot["abs_change"] / pivot[ref_label].abs(),
        np.nan
    )
    pivot = pivot.reset_index().sort_values("abs_change", key=lambda s: s.abs(), ascending=False)
    pivot["model_folder"] = model_folder
    pivot["model_type"] = metadata.get("ml_name") if isinstance(metadata, dict) else None

    return both, pivot


# ============================================================
# Dashboard HTML simple
# ============================================================

def _df_to_html_table(df, max_rows=30):
    if df is None or df.empty:
        return "<p>Sin datos disponibles.</p>"
    return df.head(max_rows).to_html(index=False, classes="table", border=0)


def build_dashboard(monitoring_dir, drift_all, shap_variation, payload):
    partition = _current_partition(payload)
    baseline = _baseline_partition(payload)

    summary = (
        drift_all.groupby(["stage", "drift_level"])
        .size()
        .reset_index(name="n_features")
        if not drift_all.empty else pd.DataFrame()
    )

    top_drift = drift_all.sort_values("psi", ascending=False).head(30) if not drift_all.empty else pd.DataFrame()
    top_shap = shap_variation.head(30) if shap_variation is not None and not shap_variation.empty else pd.DataFrame()

    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Monitoreo E2E del Modelo</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 28px; }}
            h1, h2 {{ color: #222; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
            .table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 6px; }}
            .table th {{ background: #f2f2f2; }}
            .note {{ color: #555; }}
        </style>
    </head>
    <body>
        <h1>Tablero de Monitoreo E2E</h1>
        <div class='card'>
            <b>Partición actual:</b> {partition}<br>
            <b>Partición baseline:</b> {baseline}<br>
            <b>Fecha de generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            <p class='note'>Reglas PSI: &lt;0.10 sin drift, 0.10-0.25 drift moderado, &gt;=0.25 drift alto.</p>
        </div>

        <h2>Resumen de Drift por Etapa</h2>
        <div class='card'>{_df_to_html_table(summary, 100)}</div>

        <h2>Top Variables con Mayor Data Drift</h2>
        <div class='card'>{_df_to_html_table(top_drift, 30)}</div>

        <h2>Variación Mes a Mes de Importancia de Variables</h2>
        <div class='card'>{_df_to_html_table(top_shap, 30)}</div>
    </body>
    </html>
    """

    dashboard_path = os.path.join(monitoring_dir, f"dashboard_monitoring_{partition}.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    return dashboard_path


# ============================================================
# Función principal llamada desde Prefect
# ============================================================

def monitor_pipeline(payload, workers=2, quantils=10):
    """
    Monitoreo E2E del pipeline.

    Calcula data drift en 4 etapas:
    1. raw
    2. preprocessed
    3. score_puro
    4. score_posprocesado

    Además calcula variación mensual de importancia de variables usando SHAP
    o feature_importances_ como fallback.
    """
    workers = max(int(workers or 2), 2)
    quantils = int(quantils or 10)

    monitoring_dir = _monitoring_dir(payload)
    partition = _current_partition(payload)
    baseline = _baseline_partition(payload)

    if not partition:
        raise ValueError("payload['params']['partition'] es obligatorio para monitoreo.")
    if not baseline:
        print("No se indicó baseline_partition. Se usará la misma partición como referencia.")
        baseline = partition

    print(f"Monitoreo E2E: actual={partition}, baseline={baseline}, workers={workers}, quantils={quantils}")

    stages = {
        "raw": (read_raw_stage(payload, baseline), read_raw_stage(payload, partition)),
        "preprocessed": (read_preprocessed_stage(payload, baseline), read_preprocessed_stage(payload, partition)),
        "score_puro": (read_score_stage(payload, baseline), read_score_stage(payload, partition)),
        "score_posprocesado": (read_postprocessed_stage(payload, baseline), read_postprocessed_stage(payload, partition)),
    }

    drift_outputs = []
    for stage_name, (df_ref, df_act) in stages.items():
        print(f"Calculando drift etapa: {stage_name} | ref={df_ref.shape} actual={df_act.shape}")
        drift_df = data_drift(df_ref, df_act, stage_name, quantils=quantils, workers=workers)
        drift_outputs.append(drift_df)
        drift_df.to_csv(os.path.join(monitoring_dir, f"drift_{stage_name}_{partition}.csv"), index=False)

    drift_all = pd.concat(drift_outputs, ignore_index=True)
    drift_all_path = os.path.join(monitoring_dir, f"drift_e2e_{partition}.csv")
    drift_all.to_csv(drift_all_path, index=False)

    shap_importance, shap_variation = shap_monthly_variation(
        payload,
        df_ref_pre=stages["preprocessed"][0],
        df_act_pre=stages["preprocessed"][1]
    )

    shap_importance_path = os.path.join(monitoring_dir, f"shap_importance_{partition}.csv")
    shap_variation_path = os.path.join(monitoring_dir, f"shap_variation_{partition}.csv")
    shap_importance.to_csv(shap_importance_path, index=False)
    shap_variation.to_csv(shap_variation_path, index=False)

    dashboard_path = build_dashboard(monitoring_dir, drift_all, shap_variation, payload)

    summary_path = os.path.join(monitoring_dir, f"monitoring_summary_{partition}.json")
    summary = {
        "partition": partition,
        "baseline_partition": baseline,
        "workers": workers,
        "quantils": quantils,
        "outputs": {
            "drift_e2e": drift_all_path,
            "shap_importance": shap_importance_path,
            "shap_variation": shap_variation_path,
            "dashboard": dashboard_path,
            "summary_json": summary_path,
        },
        "drift_counts": drift_all.groupby(["stage", "drift_level"]).size().reset_index(name="n").to_dict("records")
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Monitoreo finalizado.")
    print(f"Dashboard: {dashboard_path}")
    return summary
