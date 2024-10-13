import numpy as np
from scipy import stats

def calculate_reliability(X, Y, ODORS, threshold=1e-9):
    """
    Calculate reliability for each glomerulus and odor.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    ODORS (list): List of odor names.
    threshold (float): Threshold for considering a response as zero.
    
    Returns:
    np.array: 2D array of reliability scores (odors x glomeruli).
    """
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

def calculate_sparseness(x):
    """
    Calculate Hoyer sparseness measure for a vector.
    
    Args:
    x (np.array): Input vector.
    
    Returns:
    float: Sparseness measure.
    """
    n = len(x)
    return (np.sqrt(n) - np.sum(np.abs(x)) / np.sqrt(np.sum(x**2))) / (np.sqrt(n) - 1)

def calculate_sparseness_per_odor(X, Y, ODORS):
    """
    Calculate sparseness for each odor.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    ODORS (list): List of odor names.
    
    Returns:
    tuple: Arrays of mean and std of sparseness for each odor.
    """
    S_tot = []
    for k in range(len(ODORS)):
        odor_data = X[Y == k]
        S_tot.append(calculate_sparseness(np.mean(odor_data, axis=0)))
    return np.mean(S_tot), np.std(S_tot)

def calculate_reliability_statistics(reliability_matrix):
    """
    Calculate statistics for reliability scores.
    
    Args:
    reliability_matrix (np.array): 2D array of reliability scores.
    
    Returns:
    dict: Dictionary containing various statistics of reliability scores.
    """
    return {
        'mean': np.mean(reliability_matrix),
        'std': np.std(reliability_matrix),
        'median': np.median(reliability_matrix),
        'max': np.max(reliability_matrix),
        'min': np.min(reliability_matrix)
    }

def identify_reliable_glomeruli(reliability_matrix, ODORS, threshold=0.5):
    """
    Identify reliable glomeruli for each odor.
    
    Args:
    reliability_matrix (np.array): 2D array of reliability scores.
    ODORS (list): List of odor names.
    threshold (float): Reliability threshold for considering a glomerulus as reliable.
    
    Returns:
    dict: Dictionary mapping each odor to its reliable glomeruli indices.
    """
    reliable_glomeruli = {}
    for i, odor in enumerate(ODORS):
        reliable_glomeruli[odor] = np.where(reliability_matrix[i] > threshold)[0]
    return reliable_glomeruli

def reliability_analysis(X, Y, ODORS):
    """
    Perform complete reliability analysis.
    
    Args:
    X (np.array): Input features.
    Y (np.array): Target labels.
    ODORS (list): List of odor names.
    
    Returns:
    dict: Dictionary containing all results from the reliability analysis.
    """
    reliability_matrix = calculate_reliability(X, Y, ODORS)
    sparseness_mean, sparseness_std = calculate_sparseness_per_odor(X, Y, ODORS)
    reliability_stats = calculate_reliability_statistics

