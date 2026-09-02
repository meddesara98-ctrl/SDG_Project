"""
inference.py
============
Task Airflow #3 - Evaluation 

Responsabilita' di questo step:
    1. Caricare la pipeline addestrata (pickle) e i relativi metadata
       (feature attese, soglia di decisione scelta in training.py).
    2. Caricare il test set (holdout) prodotto da preprocessing.py e MAI
       toccato prima d'ora (ne' in training, ne' in validazione soglia).
    3. Calcolare le metriche di valutazione e loggarle esplicitamente.
    4. Espone anche `predict()` come funzione di scoring generica,
       riutilizzabile per applicare il modello a nuovi dati in un contesto
       di inferenza "vera" (non solo di valutazione offline).

Scelta della metrica
---------------------
Il ROC-AUC e' la metrica primaria: e' indipendente dalla soglia e quindi
la piu' robusta per giudicare la capacita' discriminativa del modello.
Viene affiancata da precision/recall/F1/accuracy ALLA SOGLIA salvata in
training.py, perche' quella e' la metrica operativa: descrive cosa succede
davvero quando il modello viene usato per decidere chi contattare.

La soglia NON viene ricalcolata qui sul test set: e' un parametro fissato
in training.py (0.45) sul validation set. Ricalcolarla sul test
comprometterebbe l'holdout (data leakage nella model/threshold selection).
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

# Import obbligatorio per la deserializzazione del pickle: la pipeline
# contiene gli step MissingValueHandler e CategoricalCaster, le cui classi
# devono essere importabili da questo stesso path (preprocessing.py) al
# momento del load. La cast a dtype 'category' (per XGBoost) e' applicata
# automaticamente da CategoricalCaster come parte della pipeline: non va
# rifatta a mano qui.
from pipeline.preprocessing import CategoricalCaster, MissingValueHandler  # noqa: F401

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_COL = "churn"


def load_artifacts(model_path: str, metadata_path: str) -> tuple:
    """Carica pipeline fittata + metadata (feature attese, soglia)."""
    logger.info("Caricamento modello da: %s", model_path)
    pipeline = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info("Metadata caricati: soglia=%.2f, feature=%d",
                metadata["threshold"], len(metadata["features"]))

    return pipeline, metadata


def predict(pipeline, metadata: dict, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Funzione di scoring generica: applica la pipeline a nuovi dati grezzi
    e restituisce probabilita' + predizione binaria secondo la soglia
    salvata nei metadata.

    Utilizzabile sia per la valutazione offline (evaluate_on_holdout) sia
    per uno scoring batch/real-time su nuovi clienti in produzione.
    """
    missing_cols = [c for c in metadata["features"] if c not in X_raw.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti rispetto a quelle attese dal modello: {missing_cols}")

    X = X_raw[metadata["features"]]
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= metadata["threshold"]).astype(int)

    return pd.DataFrame(
        {"churn_proba": y_proba, "churn_pred": y_pred},
        index=X.index,
    )


def evaluate_on_holdout(model_path: str, metadata_path: str, data_paths: dict,
                         output_dir: str) -> dict:
    """
    Valuta il modello salvato sul test set (holdout) e logga le metriche.
    Salva anche un JSON con le metriche, per tracciabilita' nei run del DAG.

    Parameters
    ----------
    model_path, metadata_path : str
        Output di train_model() (letti via XCom dal task precedente).
    data_paths : dict
        Output di preprocess_data(); qui si usano solo test_x/test_y.
    output_dir : str
        Cartella dove salvare il report delle metriche.

    Returns
    -------
    dict
        Le metriche calcolate (utile anche per eventuali check a valle,
        es. un branching Airflow che blocca il deploy se ROC-AUC < soglia).
    """
    os.makedirs(output_dir, exist_ok=True)

    pipeline, metadata = load_artifacts(model_path, metadata_path)

    logger.info("Caricamento test set (holdout)...")
    X_test = pd.read_parquet(data_paths["test_x"])
    y_test = pd.read_parquet(data_paths["test_y"])[TARGET_COL]
    logger.info("Test set: %d righe", len(X_test))

    predictions = predict(pipeline, metadata, X_test)
    y_proba = predictions["churn_proba"].values
    y_pred = predictions["churn_pred"].values

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # --- Logging esplicito nel log del task Airflow ---
    logger.info("=" * 60)
    logger.info("TEST SET- FINAL EVALUATION")
    logger.info("=" * 60)
    logger.info("Classifcation Threshold: %.2f", metadata["threshold"])
    logger.info("ROC-AUC  : %.4f  (primary metric, independent of the threshold)", roc_auc)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall   : %.4f", recall)
    logger.info("F1       : %.4f", f1)
    logger.info("Accuracy : %.4f", accuracy)

    metrics = {
        "threshold_used": metadata["threshold"],
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "n_test_samples": int(len(X_test)),
    }

    metrics_path = os.path.join(output_dir, "test_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to: %s", metrics_path)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Valutazione modello churn su holdout")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--test-x", required=True)
    parser.add_argument("--test-y", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_paths = {"test_x": args.test_x, "test_y": args.test_y}
    result = evaluate_on_holdout(args.model_path, args.metadata_path, data_paths, args.output_dir)
    print(result)
