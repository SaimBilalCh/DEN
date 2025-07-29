import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import load_and_explore_data, preprocess_data

def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(n_components)])
    print(f"\nExplained variance ratio by PCA components: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum()}")
    return pca_df, pca

def apply_tsne(df, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_components = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(data=tsne_components, columns=[f'TSNE_{i+1}' for i in range(n_components)])
    return tsne_df

def visualize_2d_data(df, title, x_label, y_label):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

if __name__ == "__main__":
    file_path = "Mall_Customers.csv"

    df = load_and_explore_data(file_path)
    processed_df = preprocess_data(df.drop(columns=['CustomerID']))

    pca_df, pca_model = apply_pca(processed_df)
    visualize_2d_data(pca_df, 'PCA of Mall Customer Data', 'Principal Component 1', 'Principal Component 2')

    tsne_df = apply_tsne(processed_df)
    visualize_2d_data(tsne_df, 't-SNE of Mall Customer Data', 't-SNE Component 1', 't-SNE Component 2')
