# Homicide-Analysis-AI-Project
1. Introduction 
The Homicide Reports Analysis project is a comprehensive machine learning study aimed at analyzing the Homicide Reports dataset (1980-2014) from Kaggle, containing 638,454 records of homicide incidents in the United States. The project focuses on predicting the weapon type used in homicides (Firearm, Knife, Blunt Object, Unknown) through a 4-class classification problem, using algorithms such as K-Nearest Neighbors (KNN), Naive Bayes, and Random Forest. Additionally, K-Means clustering is applied to identify patterns in the data. The project integrates a Flask-based web application to present results through a user-friendly interface with visualizations like ROC curves, confusion matrices, and feature importance plots.
This report provides an overview of the project, its architecture, key achievements, an analysis of strengths and limitations, a detailed output section describing user interactions and their outcomes, and a dedicated section for screenshots to showcase the web interface’s functionality.
2. Project Overview 
2.1 Objective 
The primary objectives of the Homicide Reports Analysis project are to:
•	Classify homicide incidents by weapon type using KNN, Naive Bayes, and Random Forest algorithms.
•	Perform K-Means clustering to uncover patterns in homicide data.
•	Evaluate model performance using metrics such as Accuracy, F1 Score, Precision, and Recall.
•	Enhance model performance through techniques like SMOTE, feature selection, and hyperparameter tuning.
•	Develop a Flask web application to display results, visualizations, and comparisons interactively.
2.2 Technologies Used
•	Backend: Flask (Python web framework), scikit-learn (machine learning), pandas (data processing).
•	Visualization: Matplotlib, Seaborn (plotting ROC curves, confusion matrices, etc.).
•	Frontend: HTML, CSS (Tailwind-inspired custom styles), Jinja2 (templating).
•	Data: Homicide Reports dataset (CSV format, 638,454 records, 24 features).
•	Others: NumPy (numerical operations), SMOTE (class balancing), GridSearchCV (hyperparameter tuning).
2.3 Key Features 
•	Data Preprocessing: Filtering for four weapon types, encoding categorical variables, scaling numerical features.
•	Classification: Training and evaluating KNN, Naive Bayes, and Random Forest models.
•	Clustering: Applying K-Means with 4 clusters and visualizing results using PCA.
•	Performance Evaluation: Comparing models with Accuracy, F1 Score, Precision, and Recall.
•	Model Improvement: Using SMOTE, feature selection, and hyperparameter tuning to enhance performance.
•	Web Interface: A responsive, dark-themed UI with glassmorphism design, displaying metrics and visualizations.
•	Visualizations: ROC curves, confusion matrices, feature importance plots, and clustering scatter plots.
3. System Architecture 
3.1 File Structure 
The project consists of the following key files:
•	app.py: Flask application defining routes for dataset, classification, clustering, performance, visualization, comparison, and improvement pages.
•	dataset_selection.py: Preprocesses the dataset, filters weapon types, and generates summary statistics and visualizations.
•	classification.py: Trains KNN, Naive Bayes, and Random Forest models, producing ROC curves and feature importance plots.
•	clustring.py: Applies K-Means clustering, computes silhouette scores, and visualizes clusters via PCA.
•	performance_evalution.py: Evaluates model performance with metrics and generates comparison plots.
•	visuallization.py: Creates confusion matrices, ROC curves, and performance metrics visualizations.
•	comparison.py: Compares model performance across training sizes and metrics.
•	improvement.py: Enhances KNN and Random Forest models using SMOTE, feature selection, and tuning.
•	Templates (HTML): index.html, dataset.html, classification.html, clustering.html, performance.html, visualization.html, comparison.html, improvement.html for the frontend.
•	Static: Folder for storing generated visualizations (e.g., roc_knn.png, performance_metrics.png).
3.2 Workflow 
1.	Data Preprocessing: Load homicide.csv, filter for Firearm, Knife, Blunt Object, and Unknown weapons, encode categorical features, and scale numerical ones.
2.	Classification: Split data (80% train, 20% test), train KNN, Naive Bayes, and Random Forest models, and generate ROC curves and feature importance plots.
3.	Clustering: Apply K-Means with 4 clusters, reduce dimensions with PCA, and compute silhouette scores.
4.	Performance Evaluation: Calculate Accuracy, F1 Score, Precision, and Recall for all models.
5.	Model Improvement: Balance classes with SMOTE, select top features, and tune hyperparameters using GridSearchCV.
6.	Web Interface: Flask serves HTML templates with dynamic data (metrics, visualizations) via Jinja2.
7.	Visualization: Generate and display plots (e.g., confusion matrices, ROC curves) in the static folder for web rendering.
3.3 Dataset Description
•	Source: Homicide Reports dataset (1980-2014) from Kaggle.
•	Size: 638,454 instances, 24 features.
•	Selected Features: Victim Age, Sex, Race, Perpetrator Age, Sex, Race, Relationship, Crime Type, Agency Type, State.
•	Target: Weapon type (4 classes: Firearm, Knife, Blunt Object, Unknown).
4. Achievements 
4.1 Classification Models 
•	Successfully trained KNN, Naive Bayes, and Random Forest models, with Random Forest achieving the highest performance due to its ability to handle categorical features.
•	Generated feature importance plots, identifying key predictors like Relationship and Crime Type.
4.2 Effective Clustering 
•	Applied K-Means clustering with 4 clusters, visualizing results with PCA scatter plots.
•	Computed silhouette scores to evaluate clustering quality, providing insights into data patterns.
4.3 Comprehensive Performance Evaluation 
•	Evaluated models using Accuracy, F1 Score, Precision, and Recall, presented in tables and bar plots.
•	Compared model performance across training sizes (10%, 30%, 50%, 70%, 90%), highlighting Random Forest’s consistency.

4.4 Model Improvements
•	Improved KNN and Random Forest performance using SMOTE for class balancing, feature selection (top 5 features), and hyperparameter tuning via GridSearchCV.
•	Visualized improvements with comparative metrics plots and updated ROC curves.
4.5 User-Friendly Web Interface 
•	Developed a Flask-based web application with a modern, dark-themed UI featuring glassmorphism and neural network-inspired design.
•	Integrated dynamic visualizations (e.g., ROC curves, confusion matrices) and metrics using Jinja2 templating.
4.6 Extensive Visualizations
•	Produced a wide range of visualizations, including ROC curves, confusion matrices, feature importance plots, class distribution bar plots, correlation heatmaps, PCA scatter plots, and silhouette score plots.
•	Ensured all visualizations are accessible via the web interface with descriptive captions.
5. Analysis 
5.1 Strengths
•	Comprehensive Analysis: Covers classification, clustering, performance evaluation, and model improvement, providing a holistic study of the dataset.
•	High Model Performance: Random Forest consistently outperforms KNN and Naive Bayes, achieving high Accuracy and F1 Scores.
•	Visual Clarity: Extensive visualizations enhance interpretability, making results accessible to technical and non-technical users.
•	Modular Codebase: Separate scripts for each task (e.g., classification.py, improvement.py) ensure maintainability.
•	Professional UI: The web interface’s modern design and responsiveness improve user engagement.
5.2 Limitations
•	Data Preprocessing: Dropping rows with missing values may reduce dataset size and introduce bias.
•	Clustering Assumption: Fixing K-Means to 4 clusters may not be optimal; the elbow method could improve cluster selection.
•	Visualization Redundancy: Multiple pages (e.g., performance.html, visualization.html) display similar plots (e.g., ROC curves), reducing distinctiveness.
•	Limited Interactivity: The web interface is static, lacking interactive elements like plot zooming or model selection.
•	Scalability: Flask’s default server and static image generation may struggle with high traffic or large datasets.
5.3 Potential Improvements 
•	Advanced Preprocessing: Impute missing values (e.g., using mode or median) instead of dropping rows.
•	Dynamic Clustering: Use the elbow method or silhouette analysis to determine the optimal number of clusters.
•	Consolidate Visualizations: Merge redundant pages (e.g., performance.html, visualization.html) or add unique plots (e.g., decision trees).
•	Add Interactivity: Integrate Plotly or Chart.js for interactive visualizations and user controls (e.g., model selection dropdowns).
•	Scalability Enhancements: Use a production server (e.g., Gunicorn) and cache static files for better performance.
•	Accessibility: Improve text-background contrast and add ARIA attributes for web accessibility.

6. Conclusion 
The Homicide Reports Analysis project successfully demonstrates the application of machine learning to classify and cluster homicide data, achieving high model performance with Random Forest and providing valuable insights through K-Means clustering. The Flask web application offers a professional, user-friendly interface to explore results, supported by extensive visualizations. While there are opportunities to address limitations like visualization redundancy and interactivity, the project meets its core objectives and serves as a robust foundation for further research and enhancements.
By implementing suggested improvements, such as interactive visualizations and advanced preprocessing, the project can evolve into a more scalable and engaging platform for data analysis and presentation.

