# Dockerfile
FROM python:3.10-slim

# 1) Java + outils
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Variables Java
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# 3) Workdir
WORKDIR /app

# 4) Dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5) Logs Spark (Log4j2) + scripts de démarrage
COPY log4j2.properties /app/log4j2.properties
COPY start.sh /app/start.sh
COPY spark_smoke_test.py /app/spark_smoke_test.py
RUN chmod +x /app/start.sh

# 6) Copier code 
COPY api/ /app/api/
COPY config.yaml /app/config.yaml
COPY config.py /app/config.py
COPY data_loader.py /app/data_loader.py
COPY train.py /app/train.py
COPY evaluate.py /app/evaluate.py

# Embarquer data + artefacts pour test local
COPY data/ /app/data/
COPY models_simple/ /app/models_simple/

# 7) Variables d'env projet
ENV PROJECT_ROOT=/app
ENV DATA_ROOT=/app/data
ENV MODELS_DIR=/app/models_simple
ENV CONFIG_PATH=/app/config.yaml
ENV MOVIELENS_FOLDER=ml-latest-small

# 8) Variables d'env Spark
# Mémoire / threads
ENV SPARK_DRIVER_MEMORY=2g
ENV SPARK_EXECUTOR_MEMORY=2g
ENV SPARK_EXECUTOR_CORES=2
ENV SPARK_LOCAL_CORES=2
ENV SPARK_SQL_SHUFFLE_PARTITIONS=8

# Logs
ENV SPARK_LOG_LEVEL=WARN
ENV SPARK_SUBMIT_OPTS="-Dlog4j2.configurationFile=/app/log4j2.properties"

# Dossiers temporaires Spark
ENV SPARK_LOCAL_DIRS=/tmp/spark
RUN mkdir -p /tmp/spark

# Smoke test - activable en env
# 0 = skip, 1 = run au démarrage
ENV RUN_SPARK_SMOKE_TEST=1
ENV SPARK_SMOKE_ROWS=10000

# 9) API
EXPOSE 8000

# 10) Démarrage 
ENTRYPOINT ["/app/start.sh"]