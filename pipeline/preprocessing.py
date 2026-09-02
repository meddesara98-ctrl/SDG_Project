"""
preprocessing.py
================
First step Airflow #1 - Data Preprocessing

    1. Load the raw input table.
    2. Isolate the target and select the Top 25 features (frozen list,
       derived from permutation importance analysis performed in the exploratory
       notebook -> here it is NOT recalculated at each run: it's a config).
    3. Split train / val / test.
    4. Save the RAW subsets to disk.

The MissingValueHandler and encoding are NOT applied here: they are
model-facing transformations and live inside the sklearn Pipeline that
is fitted in training.py. 
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger("preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Gestione dei missing values (soglia 30%, come da EDA):
        - Colonne (numeriche o categoriche) con MVs > soglia: rimosse.
        - Numeriche rimanenti: imputate con la mediana (calcolata sul train in fit()).
        - Categoriche rimanenti: imputate con la stringa 'MISSING'.

    NOTA IMPORTANTE per il deploy: questa classe vive qui (preprocessing.py)
    e viene importata sia da training.py (per costruire la Pipeline) sia da
    inference.py (obbligatorio per poter deserializzare il pickle con joblib,
    anche se inference.py non la istanzia mai direttamente).
    """

    def __init__(self, threshold: float = 30, impute_numeric_values: bool = False):
        self.threshold = threshold
        self.impute_numeric_values = impute_numeric_values

    def fit(self, X, y=None):
        X = X.copy()
        missing_pct = (X.isnull().sum() / len(X)) * 100
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "string"]).columns

        num_gt30 = missing_pct[(missing_pct.index.isin(num_cols)) & (missing_pct > self.threshold)]
        cat_gt30 = missing_pct[(missing_pct.index.isin(cat_cols)) & (missing_pct > self.threshold)]

        self.cols_to_drop_ = num_gt30.index.tolist() + cat_gt30.index.tolist()
        if self.cols_to_drop_:
            logger.warning(
                "MissingValueHandler: colonne rimosse per missing > %.0f%%: %s",
                self.threshold, self.cols_to_drop_
            )

        remaining_num_cols = [c for c in num_cols if c not in self.cols_to_drop_]
        if self.impute_numeric_values:
            self.medians_ = {col: X[col].median() for col in remaining_num_cols}
        else:
            self.medians_ = {}
            logger.info(
                "MissingValueHandler: impute_numeric=False -> i NaN sulle colonne "
                "numeriche rimanenti NON vengono imputati (gestiti nativamente da XGBoost)."
            )
        self.remaining_cat_cols_ = [c for c in cat_cols if c not in self.cols_to_drop_]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in self.cols_to_drop_ if c in X.columns])
        for col, median_value in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_value)
        for col in self.remaining_cat_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("MISSING")
        return X


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """
    Converte le colonne categoriche testuali (CAT_COLS) in dtype pandas
    'category', cosi' che XGBoost (enable_categorical=True) le gestisca
    nativamente, senza one-hot encoding.

    Le categorie osservate in fit() (sul train) vengono congelate: in
    transform() un valore non visto in fase di train viene mappato a NaN
    (che XGBoost tratta come missing) invece di generare una categoria
    "fantasma" solo in val/test. Questo evita disallineamenti tra gli split
    e comportamenti non riproducibili in inferenza.

    NOTA IMPORTANTE per il deploy: come MissingValueHandler, questa classe
    vive qui (preprocessing.py) e deve essere importabile da questo stesso
    path sia da training.py (per costruire la Pipeline) sia da evaluation.py
    (obbligatorio per la deserializzazione del pickle via joblib).
    """

    def __init__(self, cat_cols: list):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        self.categories_ = {}
        for col in self.cat_cols:
            if col in X.columns:
                self.categories_[col] = sorted(X[col].dropna().unique().tolist())
            else:
                logger.warning(
                    "CategoricalCaster: colonna categorica attesa '%s' non presente "
                    "nell'input (probabilmente rimossa da MissingValueHandler).", col
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col, cats in self.categories_.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats)
        return X


# ---------------------------------------------------------------------------
# Config congelata: le Top 25 feature emerse dalla permutation importance
# sul Gradient Boosting nel notebook di EDA. Cambiare questa lista implica
# una nuova validazione offline, non va ricalcolata automaticamente qui.
# ---------------------------------------------------------------------------
TOP_25_FEATURES = [
    "eqpdays",
    "months",
    "mou_Mean",
    "change_mou",
    "totmrc_Mean",
    "avgqty",
    "uniqsubs",
    "avgmou",
    "drop_vce_Mean",
    "hnd_price",
    "mou_cvce_Mean",
    "change_rev",
    "avgrev",
    "ethnic",
    "avg3mou",
    "rev_Mean",
    "ovrmou_Mean",
    "mou_peav_Mean",
    "totrev",
    "totcalls",
    "avg6mou",
    "crclscod",
    "ovrrev_Mean",
    "asl_flag",
    "opk_vce_Mean",
]

# Colonne categoriche testuali tra le Top 25: vengono castate a dtype
# 'category' in training.py/evaluation.py (via CategoricalCaster) e gestite
# nativamente da XGBoost (enable_categorical=True), senza one-hot encoding.
CAT_COLS = ["ethnic", "crclscod", "asl_flag"]

TARGET_COL = "churn"
INDEX_COL = "Customer_ID"

# Split: 70% train_full / 30% test(holdout), poi 85/15 su train_full -> train/val
TEST_SIZE = 0.30
VAL_SIZE = 0.15
RANDOM_STATE = 42


def _validate_columns(df: pd.DataFrame) -> None:
    """Fail fast e in modo esplicito se il dataset in input non e' quello atteso."""
    missing_features = [c for c in TOP_25_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Le seguenti feature attese (TOP_25_FEATURES) non sono presenti "
            f"nel dataset di input: {missing_features}. "
            f"Verificare lo schema della sorgente dati."
        )
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonna target '{TARGET_COL}' non trovata nel dataset.")


def _validate_target(y: pd.Series) -> pd.Series:
    """Garantisce che il target sia binario 0/1, senza inferenze silenziose."""
    unique_vals = set(y.dropna().unique().tolist())
    if unique_vals not in ({0, 1}, {0.0, 1.0}):
        raise ValueError(
            f"Il target '{TARGET_COL}' contiene valori inattesi ({unique_vals}); "
            f"atteso un target binario 0/1. Applicare un encoding esplicito a monte."
        )
    return y.astype(int)


def preprocess_data(input_path: str, output_dir: str) -> dict:
    """
    Carica il dataset grezzo, seleziona le Top 25 feature, fa lo split
    train/val/test e salva tutto in parquet.

    Parameters
    ----------
    input_path : str
        Percorso del CSV grezzo (sep=';', decimal=',').
    output_dir : str
        Cartella di destinazione per i parquet prodotti.

    Returns
    -------
    dict
        Mappa nome_artifact -> path, da passare via XCom al task successivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Caricamento dataset grezzo da: %s", input_path)

    df = pd.read_csv(input_path, sep=";", decimal=",").set_index(INDEX_COL)
    logger.info("Dataset caricato: %d righe, %d colonne", df.shape[0], df.shape[1])

    _validate_columns(df)

    X = df[TOP_25_FEATURES]
    y = _validate_target(df[TARGET_COL])

    churn_rate = y.mean()
    logger.info("Distribuzione target: churn=1 -> %.2f%%, churn=0 -> %.2f%%",
                churn_rate * 100, (1 - churn_rate) * 100)

    # Split 1: train_full (70%) / test-holdout (30%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Split 2: train (85% di train_full) / val (15% di train_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
    )

    logger.info(
        "Split completato -> train: %d, val: %d, test (holdout): %d",
        len(X_train), len(X_val), len(X_test)
    )

    paths = {
        "train_x": os.path.join(output_dir, "X_train.parquet"),
        "train_y": os.path.join(output_dir, "y_train.parquet"),
        "val_x": os.path.join(output_dir, "X_val.parquet"),
        "val_y": os.path.join(output_dir, "y_val.parquet"),
        "test_x": os.path.join(output_dir, "X_test.parquet"),
        "test_y": os.path.join(output_dir, "y_test.parquet"),
    }

    X_train.to_parquet(paths["train_x"])
    y_train.to_frame(name=TARGET_COL).to_parquet(paths["train_y"])
    X_val.to_parquet(paths["val_x"])
    y_val.to_frame(name=TARGET_COL).to_parquet(paths["val_y"])
    X_test.to_parquet(paths["test_x"])
    y_test.to_frame(name=TARGET_COL).to_parquet(paths["test_y"])

    logger.info("Artifact di preprocessing salvati in: %s", output_dir)
    for name, path in paths.items():
        logger.info("  - %s: %s", name, path)

    return paths


# ---------------------------------------------------------------------------
# Entry point per esecuzione standalone / debug locale.
# In Airflow questa funzione va richiamata da un PythonOperator/@task,
# leggendo input_path/output_dir da Airflow Variables o da parametri del DAG.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing dati churn")
    parser.add_argument("--input-path", required=True, help="Path al CSV grezzo")
    parser.add_argument("--output-dir", required=True, help="Cartella output parquet")
    args = parser.parse_args()

    result = preprocess_data(args.input_path, args.output_dir)
    print(result)
