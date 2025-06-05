import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns #visuallization
import matplotlib.pyplot as plt
import os

def dataset_selection():
    # Load dataset
    df = pd.read_csv('homicide.csv')
    
    # Select 4 weapon types for 4-class classification
    selected_weapons = ['Firearm', 'Knife', 'Blunt Object', 'Unknown']
    df = df[df['Weapon'].isin(selected_weapons)]
    
    # Select ALL features (excluding the target 'Weapon')
    features = [col for col in df.columns if col != 'Weapon']
    
    # Handle missing values
    df = df.dropna(subset=features + ['Weapon'])
    
    # Summary statistics
    summary = {
        'instances': df.shape[0],
        'features': len(features),
        'classes': df['Weapon'].nunique(),
        'class_distribution': df['Weapon'].value_counts().to_dict()
    }
    
    # Visualizations
    os.makedirs('static', exist_ok=True)
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Weapon', data=df)
    plt.title('Class Distribution (Weapon Types)')
    plt.xlabel('Weapon')
    plt.ylabel('Count')
    plt.savefig('static/class_distribution.png')
    plt.close()
    
    # Correlation heatmap for numeric and encoded categorical features
    df_encoded = df[features].apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('static/correlation_heatmap.png')
    plt.close()
    
    return summary, df, features

if __name__ == '__main__':
    summary, _, _ = dataset_selection()
    print(summary)