"""
training.py
===========
Second step Airflow #2 - Model Training

Responsabilita' di questo step:
    1. Caricare i subset train/val prodotti da preprocessing.py.
    2. Costruire UNA pipeline sklearn unica
       (missing_handler -> categorical_caster -> XGBoost) usando gli
       iperparametri finali selezionati durante la sperimentazione.
    3. Sul validation set: calcolare le metriche operative (precision,
       recall, f1, accuracy) alla soglia di decisione fissata, oltre al
       ROC-AUC come metrica di supporto indipendente dalla soglia.
    4. Salvare UN SOLO pickle con la pipeline fittata + un JSON di metadata
       con soglia, feature, iperparametri e metriche di validazione.

Iperparametri finali di XGBoost:
    Selezionati offline (notebook di sperimentazione) tramite GridSearchCV
    (scoring='roc_auc', StratifiedKFold a 5 fold) sulla griglia:
        n_estimators:      [100, 200, 300]
        learning_rate:     [0.01, 0.05, 0.1]
        max_depth:         [3, 5, 7]
        subsample:         [0.8, 1.0]
        colsample_bytree:  [0.8, 1.0]
    Risultato (best_params_):
        n_estimators = 300
        learning_rate = 0.05
        max_depth = 5
        subsample = 0.8
        colsample_bytree = 0.8
    La GridSearchCV NON viene rieseguita ad ogni run del DAG: qui gli
    iperparametri sono congelati come config, esattamente come le
    TOP_25_FEATURES in preprocessing.py.

Gestione delle categoriche:
    Le colonne in CAT_COLS (ethnic, crclscod, asl_flag) vengono castate a
    dtype 'category' da CategoricalCaster e gestite nativamente da XGBoost
    (enable_categorical=True): niente piu' OneHotEncoder.

Soglia di decisione:
    Fissata manualmente a 0.45 in base alla sperimentazione offline (non
    stimata con uno sweep automatico sul validation set).
"""

import json
import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Import obbligatorio: le classi devono provenire da questo modulo perche'
# joblib le referenzi correttamente in fase di dump/load.
from pipeline.preprocessing import CAT_COLS, CategoricalCaster, MissingValueHandler


logger = logging.getLogger("training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TARGET_COL = "churn"
RANDOM_STATE = 42

# Iperparametri finali selezionati durante la fase di sperimentazione
# (GridSearchCV, vedi docstring del modulo)
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Soglia di decisione fissata manualmente (non calcolata via sweep)
DECISION_THRESHOLD = 0.45


def build_pipeline() -> Pipeline:
    """Costruisce la pipeline completa: cleaning -> cast categoriche -> XGBoost."""

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        enable_categorical=True,
        eval_metric="logloss",
    )

    return Pipeline(
        [
            ("missing_handler", MissingValueHandler(threshold=30)),
            ("categorical_caster", CategoricalCaster(cat_cols=CAT_COLS)),
            ("xgb", model),
        ]
    )


def compute_metrics_at_threshold(
    y_true: pd.Series,
    y_proba,
    threshold: float
) -> dict:
    """Calcola precision/recall/f1/accuracy alla soglia data (metriche operative)."""

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_model(
    data_paths: dict,
    model_output_dir: str,
    decision_threshold: float = DECISION_THRESHOLD
) -> dict:
    """
    Addestra la pipeline completa e salva modello + metadata.

    Parameters
    ----------
    data_paths : dict
        Output di preprocess_data() (task precedente, letto via XCom).

    model_output_dir : str
        Cartella di destinazione per pickle e metadata.

    decision_threshold : float
        Soglia di decisione fissata (default: DECISION_THRESHOLD = 0.45),
        usata per calcolare le metriche operative sul validation set e
        salvata nei metadata per l'uso in evaluation.py.

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
        "Avvio training XGBoost con parametri finali: "
        "n_estimators=%d, learning_rate=%.2f, max_depth=%d, "
        "subsample=%.1f, colsample_bytree=%.1f",
        XGB_N_ESTIMATORS,
        XGB_LEARNING_RATE,
        XGB_MAX_DEPTH,
        XGB_SUBSAMPLE,
        XGB_COLSAMPLE_BYTREE
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

    # --- Metriche operative sul validation set, alla soglia fissata ---
    val_metrics = compute_metrics_at_threshold(
        y_val,
        y_proba_val,
        threshold=decision_threshold
    )

    logger.info(
        "Metriche validation @ soglia=%.2f -> "
        "precision=%.4f, recall=%.4f, f1=%.4f, accuracy=%.4f",
        val_metrics["threshold"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["accuracy"]
    )

    # --- Salvataggio artifact ---
    # UN SOLO pickle con tutta la pipeline:
    # missing handling + cast categoriche + modello
    model_path = os.path.join(
        model_output_dir,
        "xgb_25features_pipeline.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # --- Metadata ---
    metadata = {
        "features": list(X_train.columns),

        "cat_cols": CAT_COLS,

        "model": {
            "algorithm": "XGBClassifier",
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": XGB_MAX_DEPTH,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "random_state": RANDOM_STATE,
            "enable_categorical": True,
        },

        "val_roc_auc": float(roc_auc_val),

        # Soglia fissata manualmente (non stimata via sweep), usata anche
        # in evaluation.py per le predizioni sull'holdout.
        "threshold": val_metrics["threshold"],

        "val_metrics_at_threshold": val_metrics,
    }

    metadata_path = os.path.join(
        model_output_dir,
        "xgb_25features_metadata.json"
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
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Training pipeline churn (XGBoost, Top 25 feature)"
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
        "--decision-threshold",
        type=float,
        default=DECISION_THRESHOLD
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
        args.decision_threshold
    )

    print(result)