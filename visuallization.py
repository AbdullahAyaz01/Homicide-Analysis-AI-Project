import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from classification import classification
from performance_evalution import performance_evaluation  # Fixed typo

def visualization():
    # Get trained models and data
    X_train, X_test, y_train, y_test, knn, nb, rf, le, _ = classification()
    
    # Predictions
    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    
    # Create output directory
    os.makedirs('static', exist_ok=True)
    
    # Confusion matrices
    for algo, y_pred in [('KNN', y_pred_knn), ('Naive Bayes', y_pred_nb), ('Random Forest', y_pred_rf)]:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_
        )
        plt.title(f'Confusion Matrix - {algo}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()  # Prevent label overlap
        plt.savefig(f'static/cm_{algo.lower().replace(" ", "_")}.png')
        plt.close()
    
    # ROC curves (generate only if not already present)
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    for algo, model in [('KNN', knn), ('Naive Bayes', nb), ('Random Forest', rf)]:
        roc_file = f'static/roc_{algo.lower().replace(" ", "_")}.png'
        if not os.path.exists(roc_file):
            try:
                y_score = model.predict_proba(X_test)
                plt.figure(figsize=(10, 8))
                for i in range(len(le.classes_)):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {algo}')
                plt.legend(loc='lower right')
                plt.savefig(roc_file)
                plt.close()
            except AttributeError:
                print(f"Warning: {algo} does not support predict_proba. Skipping ROC curve.")
    
    # Metric comparison
    try:
        results = performance_evaluation()
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
    except Exception as e:
        print(f"Warning: Could not generate performance metrics plot due to error: {e}")

if __name__ == '__main__':
    visualization()