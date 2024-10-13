import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def set_plot_style(font_scale=2.6):
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams.update({'font.size': 12})
    sns.set(font_scale=font_scale)
    plt.style.use('seaborn-white')

def plot_confusion_matrix(conf_matrix, accuracy, percentage_removed, title=None):
    set_plot_style()
    plt.figure(figsize=(30, 20))
    sns.heatmap(conf_matrix, annot=False, cmap='hot')
    plt.tight_layout()
    plt.ylabel('True label', fontsize=40)
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.2f}; Glomeruli removed (%)={percentage_removed:.2f}', fontsize=40)
    if title:
        plt.title(title, fontsize=50)
    plt.show()

def plot_virtual_ablation(x, y, yerr, label='Random forest'):
    set_plot_style()
    plt.figure(figsize=(30, 20))
    plt.errorbar(x, y, yerr=yerr, marker='s', mfc='red', mec='green', ms=1, mew=1, label=label, color='red')
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('Omitted Glomeruli (%)', fontsize=55)
    plt.ylabel('Test accuracy', fontsize=55)
    plt.legend(fontsize=50)
    plt.show()

def plot_multiple_virtual_ablations(data_dict):
    set_plot_style()
    plt.figure(figsize=(30, 20))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (label, (x, y, yerr)) in enumerate(data_dict.items()):
        plt.errorbar(x, y, yerr=yerr, marker='s', mfc=colors[i], mec=colors[i], 
                     ms=1, mew=1, label=label, color=colors[i])
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('Omitted Glomeruli (%)', fontsize=55)
    plt.ylabel('Test accuracy', fontsize=55)
    plt.legend(fontsize=50)
    plt.show()

def plot_reliability_heatmap(reliability_matrix, odors):
    set_plot_style()
    plt.figure(figsize=(30, 20))
    sns.heatmap(reliability_matrix, xticklabels=range(1, reliability_matrix.shape[1]+1), 
                yticklabels=odors, cmap='viridis', annot=False)
    plt.xlabel('Glomeruli', fontsize=40)
    plt.ylabel('Odors', fontsize=40)
    plt.title('Reliability Heatmap', fontsize=50)
    plt.show()

def plot_reliability_distribution(reliability_vector, odor):
    set_plot_style()
    plt.figure(figsize=(20, 10))
    plt.hist(reliability_vector[reliability_vector != 0], density=True, bins=60)
    plt.title(f'Reliability Distribution for {odor}', fontsize=30)
    plt.xlabel('Reliability', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.show()

def plot_feature_importance(importance, n_top_features=20):
    set_plot_style()
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx[-n_top_features:].shape[0]) + .5
    plt.figure(figsize=(20, 10))
    plt.barh(pos, importance[sorted_idx][-n_top_features:], align='center')
    plt.yticks(pos, [f'Feature {i}' for i in sorted_idx[-n_top_features:]])
    plt.xlabel('Feature Importance', fontsize=20)
    plt.title('Top Features by Importance', fontsize=30)
    plt.show()

def plot_latency(latency):
    set_plot_style()
    plt.figure(figsize=(20, 10))
    plt.plot(latency, marker='o')
    plt.xlabel('Glomerulus Index', fontsize=20)
    plt.ylabel('Latency', fontsize=20)
    plt.title('Latency per Glomerulus', fontsize=30)
    plt.show()

def plot_pca(X, y, odors, n_components=2):
    set_plot_style()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(20, 15))
    for i, odor in enumerate(odors):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=odor, alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=20)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=20)
    plt.title('PCA of Olfactory Data', fontsize=30)
    plt.legend(fontsize=12)
    plt.show()

def plot_tsne(X, y, odors, perplexity=30, n_iter=1000):
    set_plot_style()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(20, 15))
    for i, odor in enumerate(odors):
        mask = y == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=odor, alpha=0.7)
    
    plt.xlabel('t-SNE 1', fontsize=20)
    plt.ylabel('t-SNE 2', fontsize=20)
    plt.title('t-SNE of Olfactory Data', fontsize=30)
    plt.legend(fontsize=12)
    plt.show()

def plot_time_series(X, odors, n_glomeruli=5):
    set_plot_style(font_scale=1.5)
    plt.figure(figsize=(20, 15))
    for i in range(n_glomeruli):
        plt.plot(X[:, i], label=f'Glomerulus {i+1}')
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Response', fontsize=20)
    plt.title(f'Time Series for {n_glomeruli} Glomeruli', fontsize=30)
    plt.legend(fontsize=12)
    plt.show()

def plot_3d_scatter(X, y, odors):
    set_plot_style()
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    for i, odor in enumerate(odors):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=odor, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=15)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=15)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2f})', fontsize=15)
    plt.title('3D PCA of Olfactory Data', fontsize=25)
    plt.legend(fontsize=10)
    plt.show()

def plot_correlation_matrix(X, method='pearson'):
    set_plot_style()
    corr = np.corrcoef(X.T) if method == 'pearson' else np.abs(np.corrcoef(X.T))
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title(f'{method.capitalize()} Correlation Matrix of Glomeruli', fontsize=30)
    plt.show()

def save_plot(filename, dpi=300):
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
