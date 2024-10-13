import numpy as np
import pandas as pd
from data.data_loader import load_and_preprocess, load_glomeruli_info
from models.random_forest import random_forest_pipeline, get_top_features
from models.logistic_regression import logistic_regression_pipeline
from models.svm import svm_pipeline
from analysis.ablation import ablation_analysis
from analysis.reliability import reliability_analysis
from visualization.plots import (plot_confusion_matrix, plot_virtual_ablation, 
                                 plot_reliability_heatmap, plot_pca, 
                                 plot_feature_importance, save_plot)
from utils.helpers import calculate_accuracy_per_class

# Constants
ODORS = ['Empty', 'Acetophenone', 'Eugenol', '1,4-Cineole', '2-methypyrazine', 
         '2-Ethylphenol', 'Isoeugenol', 'trans-Cinnamaldehyde', 'Allyl Sulfide', 
         'Methyl Salicylate', 'Anisole', 'anisaldehyde']

def main():
    # Load and preprocess data
    X, Y = load_and_preprocess('/path/to/X_data.p', '/path/to/Y_data.p', time_window=10)
    glomeruli_info = load_glomeruli_info('/path/to/glomeruli_info.csv')
    
    # Random Forest analysis
    rf_results = random_forest_pipeline(X, Y)
    
    # Logistic Regression analysis
    lr_results = logistic_regression_pipeline(X, Y)
    
    # SVM analysis
    svm_results = svm_pipeline(X, Y)
    
    # Plot and save confusion matrices
    plot_confusion_matrix(rf_results['confusion_matrix'], 
                          rf_results['accuracy'], 
                          0, 
                          title="Random Forest Confusion Matrix")
    save_plot("rf_confusion_matrix.png")
    
    plot_confusion_matrix(lr_results['confusion_matrix'], 
                          lr_results['accuracy'], 
                          0, 
                          title="Logistic Regression Confusion Matrix")
    save_plot("lr_confusion_matrix.png")
    
    plot_confusion_matrix(svm_results['confusion_matrix'], 
                          svm_results['accuracy'], 
                          0, 
                          title="SVM Confusion Matrix")
    save_plot("svm_confusion_matrix.png")
    
    # Plot and save feature importance (only for Random Forest)
    feature_names = [f"Glomerulus {i+1}" for i in range(X.shape[1])]
    top_features = get_top_features(rf_results['feature_importance'], feature_names)
    plot_feature_importance(rf_results['feature_importance'], feature_names)
    save_plot("feature_importance.png")
    
    # Ablation analysis (using Random Forest feature importance)
    ablation_results = ablation_analysis(X, Y, np.argsort(rf_results['feature_importance'])[::-1])
    
    # Plot and save virtual ablation results
    x = np.linspace(0, 100, len(ablation_results['mean_accuracies']))
    plot_virtual_ablation(x, ablation_results['mean_accuracies'], ablation_results['std_accuracies'])
    save_plot("virtual_ablation.png")
    
    # Reliability analysis
    reliability_results = reliability_analysis(X, Y, ODORS)
    
    # Plot and save reliability heatmap
    plot_reliability_heatmap(reliability_results['reliability_matrix'], ODORS)
    save_plot("reliability_heatmap.png")
    
    # PCA visualization
    plot_pca(X, Y, ODORS)
    save_plot("pca_plot.png")
    
    # Print summary statistics
    print("Random Forest Accuracy:", rf_results['accuracy'])
    print("Logistic Regression Accuracy:", lr_results['accuracy'])
    print("SVM Accuracy:", svm_results['accuracy'])
    print("\nRandom Forest Cross-validation scores:", rf_results['cv_scores'])
    print("Logistic Regression Cross-validation scores:", lr_results['cv_scores'])
    print("SVM Cross-validation scores:", svm_results['cv_scores'])
    print("\nTop 5 important features (Random Forest):")
    for feature, importance in top_features[:5]:
        print(f"{feature}: {importance}")
    
    print("\nReliability Statistics:")
    for stat, value in reliability_results['reliability_stats'].items():
        print(f"{stat}: {value}")
    
    print("\nAccuracy per class (Random Forest):")
    class_accuracies = calculate_accuracy_per_class(rf_results['confusion_matrix'])
    for odor, accuracy in zip(ODORS, class_accuracies):
        print(f"{odor}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()