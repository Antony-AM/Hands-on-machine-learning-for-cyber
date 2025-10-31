import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data(df: pd.DataFrame):
    """
    Sépare, normalise et rééquilibre les données.
    Retourne X_train, X_test, y_train, y_test prêts pour l'entraînement.
    """

    # Séparation X / y
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Normalisation de 'Time' et 'Amount'
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    # Séparation train / test stratifiée (préserve le ratio de fraudes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("\n--- Jeu d'entraînement initial ---")
    print(y_train.value_counts(normalize=True))

    # Rééquilibrage du jeu d'entraînement avec SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n--- Après SMOTE ---")
    print(y_train_res.value_counts(normalize=True))

    return X_train_res, X_test, y_train_res, y_test
