"""
training.py
===========
Second step Airflow #2 - Model Training

Responsabilita' di questo step:
    1. Caricare i subset train/val prodotti da preprocessing.py.
    2. Costruire UNA pipeline sklearn unica (missing_handler -> encoding -> GB)
       usando gli iperparametri finali selezionati durante la sperimentazione.
    3. Sul validation set: fare lo sweep delle soglie di decisione e scegliere
       quella finale con una regola esplicita e riproducibile.
    4. Salvare UN SOLO pickle con la pipeline fittata + un JSON di metadata
       con soglia, feature, iperparametri e metriche di supporto.

Iperparametri finali del Gradient Boosting:
    - learning_rate = 0.1
    - max_depth = 5
    - n_estimators = 200
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Import obbligatorio: la classe deve provenire da questo modulo perche'
# joblib la referenzi correttamente in fase di dump/load.
from pipeline.preprocessing import MissingValueHandler


logger = logging.getLogger("training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TARGET_COL = "churn"
RANDOM_STATE = 42

# Iperparametri finali selezionati durante la fase di sperimentazione
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 5
GB_N_ESTIMATORS = 200

# Soglie candidate per lo sweep sul validation set
THRESHOLD_GRID = np.arange(0.10, 0.95, 0.05)

# Tolleranza di default per la selezione della soglia
DEFAULT_THRESHOLD_TOLERANCE = 0.01


def build_pipeline() -> Pipeline:
    """Costruisce la pipeline completa: cleaning -> encoding -> modello."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                make_column_selector(
                    dtype_include=["object", "string"]
                ),
            )
        ],
        remainder="passthrough",
    )

    model = GradientBoostingClassifier(
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        n_estimators=GB_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("missing_handler", MissingValueHandler(threshold=30)),
            ("preprocessor", preprocessor),
            ("gb", model),
        ]
    )


def sweep_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray
) -> pd.DataFrame:
    """Calcola precision/recall/F1 per ogni soglia candidata."""

    rows = []

    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)

        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
        })

    return pd.DataFrame(rows)


def select_threshold(
    thresholds_df: pd.DataFrame,
    tolerance: float = DEFAULT_THRESHOLD_TOLERANCE
) -> dict:
    """
    Sceglie la soglia piu' alta il cui F1 e' entro `tolerance`
    dal massimo F1 osservato.

    Preferisce quindi soglie piu' alte a parita' di F1 quasi-ottimale,
    con l'obiettivo di ridurre i falsi positivi.
    """

    max_f1 = thresholds_df["f1"].max()

    candidates = thresholds_df[
        thresholds_df["f1"] >= max_f1 * (1 - tolerance)
    ]

    best_row = candidates.loc[
        candidates["threshold"].idxmax()
    ]

    max_f1_row = thresholds_df.loc[
        thresholds_df["f1"].idxmax()
    ]

    return {
        "chosen_threshold": float(best_row["threshold"]),
        "chosen_precision": float(best_row["precision"]),
        "chosen_recall": float(best_row["recall"]),
        "chosen_f1": float(best_row["f1"]),
        "max_f1_threshold": float(max_f1_row["threshold"]),
        "max_f1_value": float(max_f1_row["f1"]),
        "tolerance": tolerance,
    }


def train_model(
    data_paths: dict,
    model_output_dir: str,
    threshold_tolerance: float = DEFAULT_THRESHOLD_TOLERANCE
) -> dict:
    """
    Addestra la pipeline completa e salva modello + metadata.

    Parameters
    ----------
    data_paths : dict
        Output di preprocess_data() (task precedente, letto via XCom).

    model_output_dir : str
        Cartella di destinazione per pickle e metadata.

    threshold_tolerance : float
        Tolleranza per select_threshold().

    Returns
    -------
    dict
        Path del modello e dei metadata, da passare al task di valutazione.
    """

    os.makedirs(model_output_dir, exist_ok=True)

    logger.info("Caricamento train/val set...")

    X_train = pd.read_parquet(
        data_paths["train_x"]
    )

    y_train = pd.read_parquet(
        data_paths["train_y"]
    )[TARGET_COL]

    X_val = pd.read_parquet(
        data_paths["val_x"]
    )

    y_val = pd.read_parquet(
        data_paths["val_y"]
    )[TARGET_COL]

    logger.info(
        "Train: %d righe | Val: %d righe",
        len(X_train),
        len(X_val)
    )

    # Costruzione della pipeline con gli iperparametri finali
    pipeline = build_pipeline()

    logger.info(
        "Avvio training Gradient Boosting con parametri finali: "
        "learning_rate=%.2f, max_depth=%d, n_estimators=%d",
        GB_LEARNING_RATE,
        GB_MAX_DEPTH,
        GB_N_ESTIMATORS
    )

    # Training sul train set
    pipeline.fit(X_train, y_train)

    best_model = pipeline

    logger.info("Training completato.")

    # --- Valutazione sul validation set ---
    y_proba_val = best_model.predict_proba(X_val)[:, 1]

    roc_auc_val = roc_auc_score(
        y_val,
        y_proba_val
    )

    logger.info(
        "ROC-AUC validation: %.4f",
        roc_auc_val
    )

    # --- Scelta della soglia sul validation set ---
    threshold_sweep_df = sweep_thresholds(
        y_val,
        y_proba_val
    )

    threshold_info = select_threshold(
        threshold_sweep_df,
        tolerance=threshold_tolerance
    )

    logger.info(
        "Soglia selezionata: %.2f "
        "(F1=%.4f, precision=%.4f, recall=%.4f) "
        "| Soglia F1-max: %.2f (F1=%.4f) "
        "| tolleranza=%.2f%%",
        threshold_info["chosen_threshold"],
        threshold_info["chosen_f1"],
        threshold_info["chosen_precision"],
        threshold_info["chosen_recall"],
        threshold_info["max_f1_threshold"],
        threshold_info["max_f1_value"],
        threshold_info["tolerance"] * 100
    )

    # --- Salvataggio artifact ---
    # UN SOLO pickle con tutta la pipeline:
    # missing handling + preprocessing + modello
    model_path = os.path.join(
        model_output_dir,
        "gb_25features_pipeline.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # Salvataggio risultati threshold sweep
    threshold_sweep_path = os.path.join(
        model_output_dir,
        "threshold_sweep_val.csv"
    )

    threshold_sweep_df.to_csv(
        threshold_sweep_path,
        index=False
    )

    # --- Metadata ---
    metadata = {
        "features": list(X_train.columns),

        "model": {
            "algorithm": "GradientBoostingClassifier",
            "learning_rate": GB_LEARNING_RATE,
            "max_depth": GB_MAX_DEPTH,
            "n_estimators": GB_N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        },

        "val_roc_auc": float(roc_auc_val),

        "threshold": threshold_info["chosen_threshold"],

        "threshold_selection": threshold_info,
    }

    metadata_path = os.path.join(
        model_output_dir,
        "gb_25features_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    logger.info(
        "Modello salvato in: %s",
        model_path
    )

    logger.info(
        "Metadata salvati in: %s",
        metadata_path
    )

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "threshold_sweep_path": threshold_sweep_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training pipeline churn (GB, Top 25 feature)"
    )

    parser.add_argument(
        "--train-x",
        required=True
    )

    parser.add_argument(
        "--train-y",
        required=True
    )

    parser.add_argument(
        "--val-x",
        required=True
    )

    parser.add_argument(
        "--val-y",
        required=True
    )

    parser.add_argument(
        "--model-output-dir",
        required=True
    )

    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=DEFAULT_THRESHOLD_TOLERANCE
    )

    args = parser.parse_args()

    data_paths = {
        "train_x": args.train_x,
        "train_y": args.train_y,
        "val_x": args.val_x,
        "val_y": args.val_y,
    }

    result = train_model(
        data_paths,
        args.model_output_dir,
        args.threshold_tolerance
    )

    print(result)