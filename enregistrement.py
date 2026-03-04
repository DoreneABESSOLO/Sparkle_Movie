import os

def save_model(model, path="models/als_model"):
    model.write().overwrite().save(path)


def load_model(spark, path="models/als_model"):
    from pyspark.ml.recommendation import ALSModel
    return ALSModel.load(path)