import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from classification import classification

# Placeholder for performance_evaluation (since not provided)
try:
    from performance_evalution import performance_evaluation
except ImportError:
    def performance_evaluation():
        # Dummy implementation returning expected structure
        return pd.DataFrame({
            'Algorithm': ['KNN', 'Naive Bayes', 'Random Forest'],
            'Accuracy': [0.0, 0.0, 0.0],
            'F1 Score': [0.0, 0.0, 0.0],
            'Precision': [0.0, 0.0, 0.0],
            'Recall': [0.0, 0.0, 0.0]
        })

def comparison():
    # Get data from classification
    X_train, X_test, y_train, y_test, _, _, _, _, _ = classification()
    
    # Train models with varying training sizes
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {'Train Size': [], 'Algorithm': [], 'Accuracy': []}
    
    for size in train_sizes:
        X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
        
        for algo, model in [('KNN', KNeighborsClassifier(n_neighbors=5)),
                            ('Naive Bayes', GaussianNB()),
                            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))]:
            model.fit(X_subset, y_subset)
            y_pred = model.predict(X_test)
            results['Train Size'].append(size)
            results['Algorithm'].append(algo)
            results['Accuracy'].append(accuracy_score(y_test, y_pred))
    
    results_df = pd.DataFrame(results)
    
    # Visualize accuracy vs training size
    os.makedirs('static', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Train Size', y='Accuracy', hue='Algorithm', marker='o', data=results_df)
    plt.title('Accuracy vs Training Size')
    plt.xlabel('Training Size (Fraction)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Ensure y-axis is bounded for clarity
    plt.savefig('static/accuracy_vs_train_size.png')
    plt.close()
    
    # Metric comparison
    try:
        results_metrics = performance_evaluation()
        # Verify expected columns
        expected_cols = ['Algorithm', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
        if not all(col in results_metrics.columns for col in expected_cols):
            raise ValueError("performance_evaluation must return DataFrame with columns: Algorithm, Accuracy, F1 Score, Precision, Recall")
        
        metrics_melted = results_metrics.melt(id_vars='Algorithm', 
                                             value_vars=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                                             var_name='Metric', value_name='Score')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Metric', y='Score', hue='Algorithm', data=metrics_melted)
        plt.title('Performance Metrics Comparison')
        plt.ylim(0, 1)  # Ensure y-axis is bounded
        plt.savefig('static/performance_metrics.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate performance metrics plot due to error: {e}")
        results_metrics = pd.DataFrame()  # Return empty DataFrame if error occurs
    
    return results_df, results_metrics

if __name__ == '__main__':
    results_df, results_metrics = comparison()
    print("Accuracy vs Training Size:")
    print(results_df)
    print("\nPerformance Metrics:")
    print(results_metrics)