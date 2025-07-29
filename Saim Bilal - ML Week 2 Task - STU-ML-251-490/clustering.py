import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import load_and_explore_data, preprocess_data

def apply_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df)
    return clusters, kmeans.inertia_

def apply_hierarchical_clustering(df, n_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(df)
    return clusters

def find_optimal_clusters_elbow(df, max_clusters=10):
    sse = []
    for k in range(1, max_clusters + 1):
        _, inertia = apply_kmeans(df, k)
        sse.append(inertia)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.savefig('Elbow_Method.png')
    plt.show()
    return sse

def find_optimal_clusters_silhouette(df, max_clusters=10):
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        clusters, _ = apply_kmeans(df, k)
        score = silhouette_score(df, clusters)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('Silhouette_Score.png')
    plt.show()
    return silhouette_scores

def visualize_clusters(df, clusters, title, x_label, y_label):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(title='Cluster')
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

def evaluate_clusters(df, clusters):
    silhouette_avg = silhouette_score(df, clusters)
    davies_bouldin = davies_bouldin_score(df, clusters)
    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    return silhouette_avg, davies_bouldin

def interpret_clusters(df, clusters, original_df):
    original_df['Cluster'] = clusters
    print("\nCluster Interpretation (Mean values for each cluster):\n")
    print(original_df.groupby('Cluster').mean(numeric_only=True))
    print("\nCluster Distribution:\n")
    print(original_df['Cluster'].value_counts().sort_index())
    return original_df

if __name__ == "__main__":

    df = load_and_explore_data("Mall_Customers.csv")
    processed_df = preprocess_data(df.drop(columns=['CustomerID']))

    print("\nFinding optimal clusters using Elbow Method...")
    find_optimal_clusters_elbow(processed_df)

    print("\nFinding optimal clusters using Silhouette Score...")
    find_optimal_clusters_silhouette(processed_df)

    optimal_k = 5
    kmeans_clusters, _ = apply_kmeans(processed_df, optimal_k)
    visualize_clusters(processed_df, kmeans_clusters, f'K-Means Clustering (K={optimal_k})', 'Principal Component 1', 'Principal Component 2')

    print("\nEvaluating K-Means Clusters...")
    kmeans_silhouette, kmeans_davies_bouldin = evaluate_clusters(processed_df, kmeans_clusters)
    interpreted_kmeans_df = interpret_clusters(processed_df, kmeans_clusters, df.copy())

    hierarchical_clusters = apply_hierarchical_clustering(processed_df, optimal_k)
    visualize_clusters(processed_df, hierarchical_clusters, f'Hierarchical Clustering (K={optimal_k})', 'Principal Component 1', 'Principal Component 2')

    print("\nEvaluating Hierarchical Clusters...")
    hierarchical_silhouette, hierarchical_davies_bouldin = evaluate_clusters(processed_df, hierarchical_clusters)
    interpreted_hierarchical_df = interpret_clusters(processed_df, hierarchical_clusters, df.copy())

    print("\nComparison of Clustering Algorithms:")
    print(f"K-Means: Silhouette Score = {kmeans_silhouette:.3f}, Davies-Bouldin Index = {kmeans_davies_bouldin:.3f}")
    print(f"Hierarchical: Silhouette Score = {hierarchical_silhouette:.3f}, Davies-Bouldin Index = {hierarchical_davies_bouldin:.3f}")
