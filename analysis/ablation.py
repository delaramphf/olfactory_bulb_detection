import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def perform_virtual_ablation(X, Y, imp_features, n_iterations=5, test_size=0.2, random_state=None):
    """
    Perform virtual ablation analysis.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    imp_features (np.array): Array of feature indices sorted by importance.
    n_iterations (int): Number of iterations for each ablation step.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    
    Returns:
    tuple: Arrays of mean accuracies and standard deviations for each ablation step.
    """
    accuracies = []
    
    for i in range(len(imp_features)):
        iter_accuracies = []
        for _ in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=random_state)
            
            X_train_ablated = X_train.copy()
            X_test_ablated = X_test.copy()
            
            # Perform ablation
            X_train_ablated[:, imp_features[:i+1]] = 0
            X_test_ablated[:, imp_features[:i+1]] = 0
            
            # Train and evaluate model
            clf = LogisticRegression(penalty='l2', max_iter=1000000, C=100, solver='lbfgs', multi_class='multinomial')
            clf.fit(X_train_ablated, y_train)
            y_pred = clf.predict(X_test_ablated)
            accuracy = accuracy_score(y_test, y_pred)
            iter_accuracies.append(accuracy)
        
        accuracies.append(iter_accuracies)
    
    accuracies = np.array(accuracies)
    mean_accuracies = np.mean(accuracies, axis=1)
    std_accuracies = np.std(accuracies, axis=1)
    
    return mean_accuracies, std_accuracies

def get_ablation_confusion_matrix(X, Y, imp_features, n_ablated, test_size=0.2, random_state=None):
    """
    Get confusion matrix for a specific ablation step.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    imp_features (np.array): Array of feature indices sorted by importance.
    n_ablated (int): Number of features to ablate.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    
    Returns:
    np.array: Confusion matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=random_state)
    
    X_train_ablated = X_train.copy()
    X_test_ablated = X_test.copy()
    
    # Perform ablation
    X_train_ablated[:, imp_features[:n_ablated]] = 0
    X_test_ablated[:, imp_features[:n_ablated]] = 0
    
    # Train and evaluate model
    clf = LogisticRegression(penalty='l2', max_iter=1000000, C=100, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train_ablated, y_train)
    y_pred = clf.predict(X_test_ablated)
    
    return confusion_matrix(y_test, y_pred)

def ablation_analysis(X, Y, imp_features, n_iterations=5, test_size=0.2, random_state=None, n_steps=10):
    """
    Perform complete ablation analysis.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    imp_features (np.array): Array of feature indices sorted by importance.
    n_iterations (int): Number of iterations for each ablation step.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    n_steps (int): Number of ablation steps to perform.
    
    Returns:
    dict: Dictionary containing all results from the ablation analysis.
    """
    step_size = len(imp_features) // n_steps
    ablation_steps = range(0, len(imp_features), step_size)
    
    mean_accuracies, std_accuracies = perform_virtual_ablation(X, Y, imp_features, n_iterations, test_size, random_state)
    
    confusion_matrices = [get_ablation_confusion_matrix(X, Y, imp_features, n, test_size, random_state) for n in ablation_steps]
    
    return {
        'mean_accuracies': mean_accuracies,
        'std_accuracies': std_accuracies,
        'confusion_matrices': confusion_matrices,
        'ablation_steps': ablation_steps,
        'imp_features': imp_features
    }

