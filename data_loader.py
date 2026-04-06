import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """Charge les données depuis un fichier CSV"""
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """Nettoyage et préparation"""
    df = df.dropna()
    return df


def train_test_split_data(df, test_size=0.2, random_state=42):
    """Sépare en train/test"""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test