"""
Airflow DAG - Churn ML pipeline

Pipeline:
    1. preprocessing -> train/validation/test split + parquet artifacts
    2. training     -> XGBoost (iperparametri fissi) + soglia fissa + model artifacts
    3. evaluation   -> final evaluation on untouched test holdout

The DAG uses XCom only for small dictionaries containing artifact paths.
The actual datasets/models stay on the shared Docker volume.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# These paths are INSIDE the Airflow containers.
# They correspond to the Docker volume mounted in docker-compose.yaml.
RAW_DATA_PATH = os.getenv(
    "CHURN_RAW_DATA_PATH",
    "/opt/airflow/data/dataset.csv",
)
PROCESSED_DIR = os.getenv(
    "CHURN_PROCESSED_DIR",
    "/opt/airflow/data/processed",
)
MODEL_DIR = os.getenv(
    "CHURN_MODEL_DIR",
    "/opt/airflow/data/models",
)
EVALUATION_DIR = os.getenv(
    "CHURN_EVALUATION_DIR",
    "/opt/airflow/data/evaluation",
)

# Soglia di decisione fissata (vedi training.py: DECISION_THRESHOLD = 0.45).
# Resta configurabile via env var per eventuali override senza toccare il codice.
DECISION_THRESHOLD = float(
    os.getenv("CHURN_DECISION_THRESHOLD", "0.50")
)


# ---------------------------------------------------------------------------
# Task 1 - Preprocessing
# ---------------------------------------------------------------------------
def run_preprocessing(**context):
    # Import inside the task so the DAG scheduler does not need to import
    # pandas/sklearn just to parse the DAG.
    from pipeline.preprocessing import preprocess_data

    paths = preprocess_data(
        input_path=RAW_DATA_PATH,
        output_dir=PROCESSED_DIR,
    )

    # Returning a dict automatically pushes it to XCom.
    return paths


# ---------------------------------------------------------------------------
# Task 2 - Training
# ---------------------------------------------------------------------------
def run_training(**context):
    from pipeline.training import train_model

    ti = context["ti"]

    # Read the dictionary returned by preprocessing.
    data_paths = ti.xcom_pull(
        task_ids="preprocess_data"
    )

    if not data_paths:
        raise ValueError("Nessun artifact ricevuto dal task preprocess_data.")

    model_paths = train_model(
        data_paths=data_paths,
        model_output_dir=MODEL_DIR,
        decision_threshold=DECISION_THRESHOLD,
    )

    # Again, return a small dictionary of paths through XCom.
    return model_paths


# ---------------------------------------------------------------------------
# Task 3 - Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(**context):
    from pipeline.evaluation import evaluate_on_holdout

    ti = context["ti"]

    # Paths produced by preprocessing.
    data_paths = ti.xcom_pull(
        task_ids="preprocess_data"
    )

    # Model + metadata paths produced by training.
    model_paths = ti.xcom_pull(
        task_ids="train_model"
    )

    if not data_paths:
        raise ValueError("Nessun artifact ricevuto dal task preprocess_data.")

    if not model_paths:
        raise ValueError("Nessun artifact ricevuto dal task train_model.")

    metrics = evaluate_on_holdout(
        model_path=model_paths["model_path"],
        metadata_path=model_paths["metadata_path"],
        data_paths=data_paths,
        output_dir=EVALUATION_DIR,
    )

    # Returning the metrics also makes them visible in XCom and in the
    # Airflow task result.
    return metrics


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="churn_ml_pipeline",
    description="End-to-end churn preprocessing, training and holdout evaluation",
    start_date=datetime(2026, 1, 1),
    schedule=None,       # Manual trigger for now; change to "@daily" if desired, or "* * * * *" for every minute
    catchup=False,
    max_active_runs=1,
    tags=["ml", "churn"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing,
    )

    training_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    evaluation_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
    )

    # This creates the dependency:
    # preprocessing -> training -> evaluation
    preprocess_task >> training_task >> evaluation_task
