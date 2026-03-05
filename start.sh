#!/bin/sh
set -e

echo "[start] RUN_SPARK_SMOKE_TEST=${RUN_SPARK_SMOKE_TEST}"

if [ "${RUN_SPARK_SMOKE_TEST}" = "1" ]; then
  echo "[start] Running Spark smoke test..."
  python /app/spark_smoke_test.py
  echo "[start] Spark smoke test OK ✅"
else
  echo "[start] Spark smoke test skipped."
fi

echo "[start] Starting API..."
exec uvicorn api.app:app --host 0.0.0.0 --port 8000