import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from classification import classification

def improvement():
    # Get data and improved models
    X_train, X_test, y_train, y_test, knn, nb, rf, le, features = classification()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print dataset info
    print("Dataset size - X_train:", X_train_scaled.shape, "X_test:", X_test_scaled.shape)
    print("Improved class distribution:", np.bincount(y_train))
    
    # Improved metrics
    improved = {
        'KNN': {'Accuracy': accuracy_score(y_test, knn.predict(X_test_scaled)), 'F1 Score': f1_score(y_test, knn.predict(X_test_scaled), average='weighted')},
        'Naive Bayes': {'Accuracy': accuracy_score(y_test, nb.predict(X_test_scaled)), 'F1 Score': f1_score(y_test, nb.predict(X_test_scaled), average='weighted')},
        'Random Forest': {'Accuracy': accuracy_score(y_test, rf.predict(X_test_scaled)), 'F1 Score': f1_score(y_test, rf.predict(X_test_scaled), average='weighted')}
    }
    for algo, model in [('KNN', knn), ('Naive Bayes', nb), ('Random Forest', rf)]:
        print(f"Improved {algo} Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
    
    # Improvement 1: RandomUnderSampler
    min_class_count = min(np.bincount(y_train))
    if min_class_count < 4:
        print(f"Warning: Minority class has {min_class_count} samples. Skipping undersampling.")
        X_train_bal, y_train_bal = X_train_scaled, y_train
    else:
        try:
            rus = RandomUnderSampler(random_state=42)
            X_train_bal, y_train_bal = rus.fit_resample(X_train_scaled, y_train)
            print("RandomUnderSampler class distribution:", np.bincount(y_train_bal))
        except ValueError as e:
            print(f"Warning: RandomUnderSampler failed due to {e}. Using improved data.")
            X_train_bal, y_train_bal = X_train_scaled, y_train
    
    # Improvement 2: Hyperparameter tuning (using balanced data directly)
    # KNN
    param_dist_knn = {'n_neighbors': np.arange(5, 16, 2), 'weights': ['uniform', 'distance'], 'p': [1, 2]}
    random_knn = RandomizedSearchCV(KNeighborsClassifier(), param_dist_knn, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    random_knn.fit(X_train_bal, y_train_bal)
    best_knn = random_knn.best_estimator_
    
    # Naive Bayes
    param_dist_nb = {'var_smoothing': np.logspace(-9, 0, 10)}
    random_nb = RandomizedSearchCV(GaussianNB(), param_dist_nb, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    random_nb.fit(X_train_bal, y_train_bal)
    best_nb = random_nb.best_estimator_
    
    # Random Forest
    param_dist_rf = {'n_estimators': [50, 100, 150, 200], 'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced', None]}
    random_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist_rf, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    random_rf.fit(X_train_bal, y_train_bal)
    best_rf = random_rf.best_estimator_
    
    # Original metrics
    original = {
        'KNN': {
            'Accuracy': accuracy_score(y_test, best_knn.predict(X_test_scaled)),
            'F1 Score': f1_score(y_test, best_knn.predict(X_test_scaled), average='weighted'),
            'Macro F1': f1_score(y_test, best_knn.predict(X_test_scaled), average='macro'),
            'CV Accuracy': cross_val_score(best_knn, X_train_bal, y_train_bal, cv=5).mean()
        },
        'Naive Bayes': {
            'Accuracy': accuracy_score(y_test, best_nb.predict(X_test_scaled)),
            'F1 Score': f1_score(y_test, best_nb.predict(X_test_scaled), average='weighted'),
            'Macro F1': f1_score(y_test, best_nb.predict(X_test_scaled), average='macro'),
            'CV Accuracy': cross_val_score(best_nb, X_train_bal, y_train_bal, cv=5).mean()
        },
        'Random Forest': {
            'Accuracy': accuracy_score(y_test, best_rf.predict(X_test_scaled)),
            'F1 Score': f1_score(y_test, best_rf.predict(X_test_scaled), average='weighted'),
            'Macro F1': f1_score(y_test, best_rf.predict(X_test_scaled), average='macro'),
            'CV Accuracy': cross_val_score(best_rf, X_train_bal, y_train_bal, cv=5).mean()
        }
    }
    for algo, model in [('KNN', best_knn), ('Naive Bayes', best_nb), ('Random Forest', best_rf)]:
        print(f"Original {algo} Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
    
    # Ensure the 'static' directory exists
    os.makedirs('static', exist_ok=True)
    static_dir = os.path.abspath('static')
    print(f"Saving images to directory: {static_dir}")
    if not os.path.exists(static_dir):
        print(f"Error: Failed to create directory {static_dir}")
    
    # Confusion matrices
    for algo, model in [('KNN', best_knn), ('Naive Bayes', best_nb), ('Random Forest', best_rf)]:
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - Original {algo}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        file_path = os.path.join('static', f'improved_cm_{algo.lower().replace(" ", "_")}.png')
        try:
            plt.savefig(file_path)
            print(f"Saved confusion matrix for {algo} at: {os.path.abspath(file_path)}")
        except Exception as e:
            print(f"Error saving confusion matrix for {algo}: {e}")
        plt.close()
    
    # Metrics comparison
    results = pd.DataFrame({
        'Algorithm': ['KNN', 'KNN', 'Naive Bayes', 'Naive Bayes', 'Random Forest', 'Random Forest'],
        'Type': ['Improved', 'Original', 'Improved', 'Original', 'Improved', 'Original'],
        'Accuracy': [
            improved['KNN']['Accuracy'], original['KNN']['Accuracy'],
            improved['Naive Bayes']['Accuracy'], original['Naive Bayes']['Accuracy'],
            improved['Random Forest']['Accuracy'], original['Random Forest']['Accuracy']
        ],
        'F1 Score': [
            improved['KNN']['F1 Score'], original['KNN']['F1 Score'],
            improved['Naive Bayes']['F1 Score'], original['Naive Bayes']['F1 Score'],
            improved['Random Forest']['F1 Score'], original['Random Forest']['F1 Score']
        ]
    })
    metrics_melted = results.melt(id_vars=['Algorithm', 'Type'], value_vars=['Accuracy', 'F1 Score'], var_name='Metric', value_name='Score')
    g = sns.catplot(x='Metric', y='Score', hue='Type', col='Algorithm', kind='bar', data=metrics_melted, height=5, aspect=1, palette={'Original': 'blue', 'Improved': 'orange'})
    g.set_titles("{col_name}")
    g.fig.suptitle('Improved vs Original Metrics', y=1.05)
    g.set(ylim=(0, 1))
    file_path = os.path.join('static', 'improved_metrics.png')
    try:
        plt.savefig(file_path)
        print(f"Saved metrics comparison at: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"Error saving metrics comparison: {e}")
    plt.close()
    
    # ROC curves
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    for algo, model in [('KNN', best_knn), ('Naive Bayes', best_nb), ('Random Forest', best_rf)]:
        y_score = model.predict_proba(X_test_scaled)
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
        plt.title(f'Original ROC Curves - {algo}')
        plt.legend(loc='lower right')
        file_path = os.path.join('static', f'improved_roc_{algo.lower().replace(" ", "_")}.png')
        try:
            plt.savefig(file_path)
            print(f"Saved ROC curve for {algo} at: {os.path.abspath(file_path)}")
        except Exception as e:
            print(f"Error saving ROC curve for {algo}: {e}")
        plt.close()
    
    # Feature importance for all features using Random Forest
    rf_importance = best_rf.feature_importances_
    top_features = [(features[i], rf_importance[i]) for i in range(len(features))]
    
    # Plot feature importance for all features
    plt.figure(figsize=(10, max(6, len(features) * 0.3)))  # Dynamic height based on number of features
    feature_df = pd.DataFrame({'Feature': features, 'Importance': rf_importance})
    feature_df = feature_df.sort_values(by='Importance', ascending=True)  # Sort for better visualization
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    file_path = os.path.join('static', 'improved_feature_importance.png')
    try:
        plt.savefig(file_path)
        print(f"Saved feature importance plot at: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"Error saving feature importance plot: {e}")
    plt.close()
    
    print("\nTo view the generated images, check the following files in the 'static' directory:")
    print("- improved_cm_knn.png, improved_cm_naive_bayes.png, improved_cm_random_forest.png (Confusion Matrices)")
    print("- improved_metrics.png (Metrics Comparison)")
    print("- improved_roc_knn.png, improved_roc_naive_bayes.png, improved_roc_random_forest.png (ROC Curves)")
    print("- improved_feature_importance.png (Feature Importance)")
    
    return improved, original, random_knn.best_params_, random_nb.best_params_, random_rf.best_params_, len(features), top_features

if __name__ == '__main__':
    improved, original, knn_params, nb_params, rf_params, n_features, top_features = improvement()
    print(f"Improved Metrics: {improved}")
    print(f"Original Metrics: {original}")
    print(f"KNN Best Params: {knn_params}")
    print(f"Naive Bayes Best Params: {nb_params}")
    print(f"Random Forest Best Params: {rf_params}")
    print(f"Number of Features: {n_features}")
    print(f"Top Features: {top_features}")