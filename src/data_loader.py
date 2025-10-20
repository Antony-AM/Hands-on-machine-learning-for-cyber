import pandas as pd
import os

def load_data():
    # Chemin absolu vers le dossier du projet
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    
    df = pd.read_csv(data_path)
    print("[OK] Dataset chargé avec succès")
    print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(df['Class'].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    load_data()
