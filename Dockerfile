FROM apache/airflow:2.10.0

# Copia il file dei requisiti nel container
COPY requirements.txt .

# Installa le dipendenze durante la creazione dell'immagine
RUN pip install --no-cache-dir -r requirements.txt