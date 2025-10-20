# src/model_rf.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n=== Random Forest ===")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Évaluation ---
    print("\n--- Rapport de classification ---")
    print(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC : {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC  : {pr_auc:.4f}")

    # Courbe Precision-Recall
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall - Random Forest')
    plt.legend()
    plt.show()

    return model
