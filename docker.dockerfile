FROM bitnami/spark:3.5.0

USER root
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# AJOUT IMPORTANT
RUN pip install pyspark

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV SPARK_WORKER_MEMORY=2G
ENV SPARK_DRIVER_MEMORY=2G
ENV SPARK_WORKER_CORES=2
ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python

RUN python - << 'EOF'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("smoke-test").getOrCreate()
df = spark.range(0, 10)
assert df.count() == 10
spark.stop()
print("✔ Smoke test Spark OK")
EOF

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
