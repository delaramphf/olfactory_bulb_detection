import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

def calculate_accuracy_per_class(conf_matrix):
    accuracies = []
    for i in range(conf_matrix.shape[0]):
        total = np.sum(conf_matrix[i])
        correct = conf_matrix[i, i]
        accuracies.append((correct / total) * 100)
    return accuracies

def sparsity(x):
    """Calculate Hoyer sparseness measure for a vector."""
    n = len(x)
    return (np.sqrt(n) - np.sum(np.abs(x)) / np.sqrt(np.sum(x**2))) / (np.sqrt(n) - 1)

def calculate_sparseness_per_odor(X, Y, ODORS):
    """Calculate sparseness for each odor."""
    S_tot = []
    for k in range(len(ODORS)):
        odor_data = X[Y == k]
        S_tot.append(sparsity(np.mean(odor_data, axis=0)))
    return np.mean(S_tot), np.std(S_tot)

def generate_random_data(n_samples, n_features):
    """Generate random data for testing purposes."""
    return np.random.rand(n_samples, n_features)

def split_data(X, Y, test_size=0.2, random_state=None):
    """Split data into train and test sets."""
    return train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=random_state)

def normalize_data(X):
    """Normalize data to have zero mean and unit variance."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def calculate_reliability(X, Y, ODORS, threshold=1e-9):
    """Calculate reliability for each glomerulus and odor."""
    NUM_G = X.shape[1]
    Reli = []
    norm_fact = {}
    
    for odorant in ODORS:
        MEANs = []
        STDs = []
        for i in range(NUM_G):
            S = X[Y == ODORS.index(odorant), i]
            MEANs.append(np.mean(S))
            STDs.append(np.std(S))
        
        MEANs = np.array(MEANs)
        MEANs[np.abs(MEANs) < threshold] = 0
        reli = 2 * (np.abs(stats.norm.cdf(MEANs / np.array(STDs)) - 0.5))
        reli = reli ** 3
        norm_fact[odorant] = reli.max()
        reli = reli / norm_fact[odorant]
        Reli.append(reli)
    
    return np.vstack(Reli)

def moving_average(data, window_size):
    """Calculate moving average of data."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def calculate_latency(X, threshold=0.5):
    """Calculate latency for each glomerulus."""
    max_response = np.max(X, axis=0)
    latency = []
    for i in range(X.shape[1]):
        response = X[:, i]
        threshold_value = threshold * max_response[i]
        latency.append(np.argmax(response > threshold_value))
    return np.array(latency)

def save_results(filename, data):
    """Save results to a CSV file."""
    pd.DataFrame(data).to_csv(filename, index=False)

def load_results(filename):
    """Load results from a CSV file."""
    return pd.read_csv(filename)

def calculate_auc(y_true, y_scores):
    """Calculate Area Under the Curve (AUC) for binary classification."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)

def bootstrap_confidence_interval(data, num_bootstraps=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval."""
    bootstrap_means = np.zeros(num_bootstraps)
    for i in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    confidence_interval = np.percentile(bootstrap_means, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    return confidence_interval

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std

def perform_ttest(group1, group2):
    """Perform independent t-test between two groups."""
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    return t_statistic, p_value

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient and p-value."""
    return stats.pearsonr(x, y)
