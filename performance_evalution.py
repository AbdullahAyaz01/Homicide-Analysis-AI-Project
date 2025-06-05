import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from classification import classification

def performance_evaluation():
    # Get trained models and data
    X_train, X_test, y_train, y_test, knn, nb, rf, le, _ = classification()
    
    # Predictions
    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    
    # Metrics
    results = pd.DataFrame({
        'Algorithm': ['KNN', 'Naive Bayes', 'Random Forest'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_knn),
            accuracy_score(y_test, y_pred_nb),
            accuracy_score(y_test, y_pred_rf)
        ],
        'F1 Score': [
            f1_score(y_test, y_pred_knn, average='weighted', zero_division=0),
            f1_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        ],
        'Precision': [
            precision_score(y_test, y_pred_knn, average='weighted', zero_division=0),
            precision_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        ],
        'Recall': [
            recall_score(y_test, y_pred_knn, average='weighted', zero_division=0),
            recall_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        ]
    })
    
    # Visualize metrics
    os.makedirs('static', exist_ok=True)
    metrics_melted = results.melt(
        id_vars='Algorithm',
        value_vars=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
        var_name='Metric',
        value_name='Score'
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Algorithm', data=metrics_melted)
    plt.title('Performance Metrics Comparison')
    plt.ylim(0, 1)  # Bound y-axis for clarity
    plt.tight_layout()  # Prevent label overlap
    plt.savefig('static/performance_metrics.png')
    plt.close()
    
    return results

if __name__ == '__main__':
    results = performance_evaluation()
    print("Performance Metrics:")
    print(results)