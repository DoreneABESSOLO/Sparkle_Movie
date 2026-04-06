from pyspark.ml.evaluation import RegressionEvaluator

import numpy as np


def precision_at_k(model, train_matrix, test_matrix, k=10):
    """Calcule Precision@K"""
    precisions = []

    for user in range(test_matrix.shape[0]):
        recommended = model.recommend(user, train_matrix[user], N=k)
        recommended_items = [item for item, _ in recommended]

        true_items = test_matrix[user].indices

        if len(true_items) == 0:
            continue

        hits = len(set(recommended_items) & set(true_items))
        precisions.append(hits / k)

    return np.mean(precisions)


def evaluate_model(model, train_matrix, test_matrix):
    """Évaluation globale"""
    p_at_k = precision_at_k(model, train_matrix, test_matrix)

    print(f"Precision@K: {p_at_k:.4f}")

    return {
        "precision_at_k": p_at_k
    }