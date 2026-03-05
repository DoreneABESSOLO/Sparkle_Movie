import os
import sys
from pyspark.sql import SparkSession


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def main() -> int:
    rows = int(_env("SPARK_SMOKE_ROWS", "10000"))
    driver_mem = _env("SPARK_DRIVER_MEMORY", "2g")
    exec_mem = _env("SPARK_EXECUTOR_MEMORY", "2g")
    exec_cores = _env("SPARK_EXECUTOR_CORES", "2")
    local_cores = _env("SPARK_LOCAL_CORES", "2")
    shuffle_parts = _env("SPARK_SQL_SHUFFLE_PARTITIONS", "8")
    log_level = _env("SPARK_LOG_LEVEL", "WARN")

    print("[smoke] Building SparkSession...")

    spark = (
        SparkSession.builder
        .appName("sparkle-smoke-test")
        .master(f"local[{local_cores}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", exec_mem)
        .config("spark.executor.cores", exec_cores)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .getOrCreate()
    )

    try:
        spark.sparkContext.setLogLevel(log_level)
        n = spark.range(rows).count()
        if n != rows:
            raise RuntimeError(f"[smoke] Unexpected count: got {n}, expected {rows}")
        print(f"[smoke] OK ✅ Spark ran a job successfully (count={n}).")
        return 0
    finally:
        spark.stop()
        print("[smoke] Spark stopped.")

if __name__ == "__main__":
    sys.exit(main())