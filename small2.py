import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import seaborn as sns

class ClusteringComparison:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        
    def generate_datasets(self):
        """Különböző típusú adathalmazok generálása"""
        np.random.seed(42)
        
        # 1. Jól elkülönülő klaszterek
        X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, 
                                     cluster_std=0.8, random_state=42)
        
        # 2. Hold alakú klaszterek
        X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
        
        # 3. Kör alakú klaszterek
        X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                          factor=0.5, random_state=42)
        
        # 4. Egyenetlen sűrűségű klaszterek
        X_uneven = np.vstack([
            np.random.normal(0, 0.3, (100, 2)),
            np.random.normal(2, 0.8, (200, 2))
        ])
        y_uneven = np.array([0]*100 + [1]*200)
        
        self.datasets = {
            'Jól elkülönülő klaszterek': (X_blobs, y_blobs),
            'Hold alakú klaszterek': (X_moons, y_moons),
            'Kör alakú klaszterek': (X_circles, y_circles),
            'Egyenetlen sűrűségű klaszterek': (X_uneven, y_uneven)
        }
        
    def run_clustering_algorithms(self):
        """Különböző klaszterező algoritmusok futtatása"""
        algorithms = {
            'K-Means': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
            'Gaussian Mixture': GaussianMixture(n_components=3, random_state=42)
        }
        
        results = {}
        
        for dataset_name, (X, y_true) in self.datasets.items():
            print(f"\n=== {dataset_name} ===")
            
            # Adatok normalizálása
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            dataset_results = {}
            
            for algo_name, algorithm in algorithms.items():
                print(f"Futtatás: {algo_name}")
                
                try:
                    if algo_name == 'Gaussian Mixture':
                        # GMM előrejelzéseket ad
                        labels = algorithm.fit_predict(X_scaled)
                    else:
                        # K-Means és DBSCAN
                        labels = algorithm.fit_predict(X_scaled)
                    
                    # Metrikák számítása
                    if len(np.unique(labels)) > 1:  # Legalább 2 klaszter
                        silhouette = silhouette_score(X_scaled, labels)
                    else:
                        silhouette = -1
                    
                    # Adjusted Rand Index (csak ha van valós címke)
                    ari = adjusted_rand_score(y_true, labels)
                    
                    dataset_results[algo_name] = {
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'adjusted_rand_score': ari,
                        'n_clusters': len(np.unique(labels[labels >= 0]))  # DBSCAN esetén a zajt nem számoljuk
                    }
                    
                    print(f"  - Klaszterek száma: {dataset_results[algo_name]['n_clusters']}")
                    print(f"  - Silhouette score: {silhouette:.3f}")
                    print(f"  - Adjusted Rand Index: {ari:.3f}")
                    
                except Exception as e:
                    print(f"  Hiba: {e}")
                    dataset_results[algo_name] = None
            
            results[dataset_name] = dataset_results
        
        self.results = results
        return results
    
    def plot_results(self):
        """Eredmények vizualizálása"""
        fig, axes = plt.subplots(len(self.datasets), 4, 
                               figsize=(20, 5*len(self.datasets)))
        
        if len(self.datasets) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (dataset_name, (X, y_true)) in enumerate(self.datasets.items()):
            # Eredeti adatok
            axes[idx, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50)
            axes[idx, 0].set_title(f'{dataset_name}\n(Eredeti)')
            axes[idx, 0].set_xlabel('X1')
            axes[idx, 0].set_ylabel('X2')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Algoritmus eredmények
            for algo_idx, algo_name in enumerate(['K-Means', 'DBSCAN', 'Gaussian Mixture']):
                if (self.results[dataset_name][algo_name] is not None and 
                    algo_name in self.results[dataset_name]):
                    
                    labels = self.results[dataset_name][algo_name]['labels']
                    silhouette = self.results[dataset_name][algo_name]['silhouette_score']
                    ari = self.results[dataset_name][algo_name]['adjusted_rand_score']
                    
                    scatter = axes[idx, algo_idx + 1].scatter(
                        X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50
                    )
                    
                    title = f'{algo_name}\nSilhouette: {silhouette:.3f}\nARI: {ari:.3f}'
                    axes[idx, algo_idx + 1].set_title(title)
                    axes[idx, algo_idx + 1].set_xlabel('X1')
                    axes[idx, algo_idx + 1].set_ylabel('X2')
                    axes[idx, algo_idx + 1].grid(True, alpha=0.3)
                
                else:
                    axes[idx, algo_idx + 1].text(0.5, 0.5, 'Nem elérhető', 
                                               ha='center', va='center', 
                                               transform=axes[idx, algo_idx + 1].transAxes)
                    axes[idx, algo_idx + 1].set_title(f'{algo_name}\n(Nem futott)')
        
        plt.tight_layout()
        plt.show()
    
    def performance_comparison(self):
        """Teljesítmény összehasonlítás"""
        metrics_data = []
        
        for dataset_name in self.datasets.keys():
            for algo_name in ['K-Means', 'DBSCAN', 'Gaussian Mixture']:
                if (self.results[dataset_name][algo_name] is not None and 
                    algo_name in self.results[dataset_name]):
                    
                    result = self.results[dataset_name][algo_name]
                    metrics_data.append({
                        'Dataset': dataset_name,
                        'Algorithm': algo_name,
                        'Silhouette_Score': result['silhouette_score'],
                        'Adjusted_Rand_Index': result['adjusted_rand_score'],
                        'N_Clusters': result['n_clusters']
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Metrikák ábrázolása
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Silhouette Score összehasonlítás
        silhouette_pivot = metrics_df.pivot_table(
            values='Silhouette_Score', 
            index='Dataset', 
            columns='Algorithm'
        )
        silhouette_pivot.plot(kind='bar', ax=axes[0, 0], colormap='Set3')
        axes[0, 0].set_title('Silhouette Score Összehasonlítás')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Adjusted Rand Index összehasonlítás
        ari_pivot = metrics_df.pivot_table(
            values='Adjusted_Rand_Index', 
            index='Dataset', 
            columns='Algorithm'
        )
        ari_pivot.plot(kind='bar', ax=axes[0, 1], colormap='Set2')
        axes[0, 1].set_title('Adjusted Rand Index Összehasonlítás')
        axes[0, 1].set_ylabel('Adjusted Rand Index')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Klaszterek száma
        clusters_pivot = metrics_df.pivot_table(
            values='N_Clusters', 
            index='Dataset', 
            columns='Algorithm'
        )
        clusters_pivot.plot(kind='bar', ax=axes[1, 0], colormap='Pastel1')
        axes[1, 0].set_title('Klaszterek Száma')
        axes[1, 0].set_ylabel('Klaszterek száma')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Heatmap teljesítményről
        performance_data = metrics_df.pivot_table(
            values='Silhouette_Score', 
            index='Dataset', 
            columns='Algorithm'
        )
        sns.heatmap(performance_data, annot=True, cmap='YlOrRd', 
                   ax=axes[1, 1], cbar_kws={'label': 'Silhouette Score'})
        axes[1, 1].set_title('Teljesítmény Heatmap')
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df

def main():
    """Fő program"""
    print("=== Scikit-learn Klaszterezési Algoritmusok Összehasonlítása ===\n")
    
    # Összehasonlító objektum létrehozása
    comparator = ClusteringComparison()
    
    # Adathalmazok generálása
    print("1. Adathalmazok generálása...")
    comparator.generate_datasets()
    
    # Algoritmusok futtatása
    print("2. Klaszterezési algoritmusok futtatása...")
    comparator.run_clustering_algorithms()
    
    # Eredmények ábrázolása
    print("3. Eredmények vizualizálása...")
    comparator.plot_results()
    
    # Teljesítmény összehasonlítás
    print("4. Teljesítmény összehasonlítás...")
    metrics_df = comparator.performance_comparison()
    
    # Összefoglaló
    print("\n=== ÖSSZEFOGLALÓ ===")
    print("K-Means: Jól működik gömb alakú, egyenletes sűrűségű klaszterekkel")
    print("DBSCAN: Képes felfedezni tetszőleges alakú klasztereket és zajt kezelni")
    print("Gaussian Mixture: Valószínűségi megközelítés, lágy klaszterezés")
    
    return comparator, metrics_df

if __name__ == "__main__":
    comparator, results = main()