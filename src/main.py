from data_loader import load_data
from eda import basic_eda
from preprocessing import prepare_data
from model_rf import train_random_forest
from model_xgb import train_xgboost
from explain_model import plot_feature_importance, explain_with_shap
from logistic_regression import train_LogisticRegression

def main():
    df = load_data()
    basic_eda(df)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Entraînement des modèles
    
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
    log_model = train_LogisticRegression(X_train, X_test, y_train, y_test)

    # Importance des features pour Random Forest
    
    plot_feature_importance(rf_model, X_train, top_n=10, title="Top 10 - Random Forest Feature Importance")



    # Explications SHAP sur un échantillon
    sample = X_test.sample(500, random_state=42)
    
    explain_with_shap(rf_model, sample)

if __name__ == "__main__":
    main()
