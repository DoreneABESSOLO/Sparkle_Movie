from pyspark.ml.evaluation import RegressionEvaluator


def evaluate_model(model, test_df):
    predictions = model.transform(test_df)

    rmse_evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    mae_evaluator = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction"
    )

    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    return {
        "RMSE": rmse,
        "MAE": mae
    }
