import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def create_user_item_matrix(df, user_col, item_col, rating_col):
    """Crée une matrice user-item"""
    user_ids = df[user_col].astype("category").cat.codes
    item_ids = df[item_col].astype("category").cat.codes

    matrix = csr_matrix(
        (df[rating_col], (user_ids, item_ids))
    )

    return matrix


def train_als_model(matrix, factors=50, regularization=0.01, iterations=20):
    """Entraîne le modèle ALS"""
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    model.fit(matrix)
    return model

import pickle

def save_artifacts(model, matrix):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)