# src/explain_model.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap

def plot_feature_importance(model, X_train, top_n=10, title="Feature Importance"):
    """
    Affiche les variables les plus importantes selon le modèle.
    """
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def explain_with_shap(model, X_sample):
    """
    Génère les explications SHAP sur un échantillon du jeu de données.
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # Résumé global
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_sample)

    return shap_values
