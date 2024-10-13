import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path_X, file_path_Y):
    """
    Load X and Y data from pickle files.
    
    Args:
    file_path_X (str): Path to the X data pickle file.
    file_path_Y (str): Path to the Y data pickle file.
    
    Returns:
    tuple: X and Y data as numpy arrays.
    """
    with open(file_path_X, "rb") as f:
        X = pickle.load(f)
    with open(file_path_Y, "rb") as f:
        Y = pickle.load(f)
    return X, Y

def preprocess_data(X, time_window, normalize=True):
    """
    Preprocess the data by applying a time window and optionally normalizing.
    
    Args:
    X (list of np.array): List of 2D arrays, each representing a trial.
    time_window (int): Size of the time window to average over.
    normalize (bool): Whether to normalize the data.
    
    Returns:
    np.array: Processed X data.
    """
    X_temp = []
    for trial in X:
        X_temp.append(np.mean(trial[:, time_window-1:time_window+2], axis=1))
    X_processed = np.stack(X_temp).T
    
    if normalize:
        X_processed = X_processed / np.max(np.abs(X_processed))
    
    return X_processed

def split_data(X, Y, test_size=0.2, random_state=None):
    """
    Split the data into training and testing sets.
    
    Args:
    X (np.array): Processed X data.
    Y (np.array): Y labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=random_state)

def load_additional_data(file_path):
    """
    Load additional data from CSV or Excel files.
    
    Args:
    file_path (str): Path to the data file.
    
    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def create_time_windows(X, window_sizes):
    """
    Create multiple time windows from the original data.
    
    Args:
    X (list of np.array): List of 2D arrays, each representing a trial.
    window_sizes (list of int): List of window sizes to use.
    
    Returns:
    list of np.array: List of processed X data for each window size.
    """
    return [preprocess_data(X, w) for w in window_sizes]

def augment_data(X, Y, augmentation_factor=2):
    """
    Augment the data by adding random noise.
    
    Args:
    X (np.array): Processed X data.
    Y (np.array): Y labels.
    augmentation_factor (int): Factor by which to augment the data.
    
    Returns:
    tuple: Augmented X and Y data.
    """
    X_aug = np.repeat(X, augmentation_factor, axis=0)
    Y_aug = np.repeat(Y, augmentation_factor, axis=0)
    
    noise = np.random.normal(0, 0.1, X_aug.shape)
    X_aug += noise
    
    return X_aug, Y_aug

def load_glomeruli_info(file_path):
    """
    Load glomeruli information from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file containing glomeruli information.
    
    Returns:
    pd.DataFrame: DataFrame containing glomeruli information.
    """
    return pd.read_csv(file_path)

def apply_mask(X, mask):
    """
    Apply a binary mask to the data.
    
    Args:
    X (np.array): Processed X data.
    mask (np.array): Binary mask to apply.
    
    Returns:
    np.array: Masked X data.
    """
    return X * mask

def standardize_data(X):
    """
    Standardize the data to have zero mean and unit variance.
    
    Args:
    X (np.array): Processed X data.
    
    Returns:
    np.array: Standardized X data.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def load_and_preprocess(file_path_X, file_path_Y, time_window, test_size=0.2, random_state=None):
    """
    Load, preprocess, and split the data in one function call.
    
    Args:
    file_path_X (str): Path to the X data pickle file.
    file_path_Y (str): Path to the Y data pickle file.
    time_window (int): Size of the time window to average over.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X, Y = load_data(file_path_X, file_path_Y)
    X_processed = preprocess_data(X, time_window)
    return split_data(X_processed, Y, test_size, random_state)

def save_processed_data(X, Y, file_path):
    """
    Save processed data to a pickle file.
    
    Args:
    X (np.array): Processed X data.
    Y (np.array): Y labels.
    file_path (str): Path to save the processed data.
    """
    with open(file_path, 'wb') as f:
        pickle.dump({'X': X, 'Y': Y}, f)

def load_processed_data(file_path):
    """
    Load processed data from a pickle file.
    
    Args:
    file_path (str): Path to the processed data file.
    
    Returns:
    tuple: X and Y data as numpy arrays.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['Y']

