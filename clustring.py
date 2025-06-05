import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataset_selection import dataset_selection

def clustering():
    # Load and preprocess data
    _, df, features = dataset_selection()
    
    # Encode categorical features
    for col in features:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    X = df[features]
    
    # Scale features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dictionary to store results
    cluster_results = {}
    os.makedirs('static', exist_ok=True)
    
    # Clustering for different numbers of clusters
    for n_clusters in [2, 3, 4, 5]:
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, clusters)
        cluster_results[n_clusters] = {'clusters': clusters, 'silhouette': silhouette}
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'K-Means Clustering (PCA Reduced) - {n_clusters} Clusters')
        plt.xlabel(f'PCA Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PCA Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.savefig(f'static/clustering_{n_clusters}.png')
        plt.close()
        
        # Silhouette score plot
        plt.figure(figsize=(6, 4))
        sns.barplot(x=[f'Silhouette Score ({n_clusters} Clusters)'], y=[silhouette])
        plt.title(f'Silhouette Score for K-Means ({n_clusters} Clusters)')
        plt.ylim(0, 1)
        plt.savefig(f'static/silhouette_score_{n_clusters}.png')
        plt.close()
    
    # Return results for the last run (5 clusters) or adjust as needed
    return (cluster_results[5]['silhouette'], 
            cluster_results[5]['clusters'], 
            X_pca)

if __name__ == '__main__':
    silhouette, clusters, X_pca = clustering()
    print(f"Silhouette Score: {silhouette}")