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
    
    print("\n--- Génération des valeurs SHAP ---")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_sample)
