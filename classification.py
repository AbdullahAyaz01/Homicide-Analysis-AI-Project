import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
from dataset_selection import dataset_selection

def classification():
    # Load and preprocess data
    _, df, features = dataset_selection()
    target = 'Weapon'
    
    # Encode categorical features
    le = LabelEncoder()
    for col in features:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    # Encode target
    y = le.fit_transform(df[target])
    X = df[features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Visualizations
    os.makedirs('static', exist_ok=True)
    
    # Feature importance for Random Forest
    plt.figure(figsize=(12, max(6, len(features) * 0.4)))  
    sns.barplot(x=rf.feature_importances_, y=features)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()
    
    # ROC curves
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    for algo, model in [('KNN', knn), ('Naive Bayes', nb), ('Random Forest', rf)]:
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
        plt.savefig(f'static/roc_{algo.lower().replace(" ", "_")}.png')
        plt.close()
    
    return X_train, X_test, y_train, y_test, knn, nb, rf, le, features

if __name__ == '__main__':
    classification()