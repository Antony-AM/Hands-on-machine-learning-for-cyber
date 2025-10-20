import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_eda(df: pd.DataFrame):
    print("\n--- Aperçu des données ---")
    print(df.head())

    print("\n--- Informations sur les colonnes ---")
    print(df.info())

    print("\n--- Valeurs manquantes ---")
    print(df.isnull().sum().sum(), "valeurs manquantes au total")

    print("\n--- Statistiques descriptives ---")
    print(df.describe())

    # Visualisation du déséquilibre des classes
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Class', data=df)
    plt.title("Répartition des classes (0 = normal, 1 = fraude)")
    plt.show()

    # Distribution du montant des transactions
    plt.figure(figsize=(7, 4))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title("Distribution des montants de transaction")
    plt.xlabel("Montant (€)")
    plt.ylabel("Fréquence")
    plt.show()

    # Distribution temporelle
    plt.figure(figsize=(7, 4))
    sns.histplot(df['Time'], bins=50, kde=False)
    plt.title("Distribution temporelle des transactions")
    plt.xlabel("Temps écoulé (secondes)")
    plt.ylabel("Nombre de transactions")
    plt.show()
