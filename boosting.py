import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class BoostComparison:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        
    def generate_datasets(self):
        """Különböző nehézségű besorolási problémák generálása"""
        np.random.seed(42)
        
        # 1. Könnyű, lineárisan szeparálható probléma
        X_easy, y_easy = make_classification(
            n_samples=500, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )
        
        # 2. Közepesen nehéz probléma
        X_medium, y_medium = make_classification(
            n_samples=500, n_features=10, n_redundant=2, n_informative=8,
            n_clusters_per_class=2, flip_y=0.1, random_state=123
        )
        
        # 3. Nehéz, zajos probléma
        X_hard, y_hard = make_classification(
            n_samples=500, n_features=20, n_redundant=5, n_informative=15,
            flip_y=0.2, random_state=456
        )
        
        self.datasets = {
            'Könnyű probléma (2 feature)': (X_easy, y_easy),
            'Közepes probléma (10 feature)': (X_medium, y_medium),
            'Nehéz probléma (20 feature)': (X_hard, y_hard)
        }
    
    def initialize_models(self):
        """Boost modellek inicializálása"""
        self.models = {
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
            'GradientBoost': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'HistGradientBoost': HistGradientBoostingClassifier(max_iter=50, random_state=42)
        }
    
    def run_comparison(self):
        """Boost algoritmusok összehasonlítása"""
        self.initialize_models()
        results = {}
        
        for dataset_name, (X, y) in self.datasets.items():
            print(f"\n{'='*50}")
            print(f"{dataset_name}")
            print(f"{'='*50}")
            
            # Adatok felosztása
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            dataset_results = {}
            
            for model_name, model in self.models.items():
                print(f"\n  {model_name}:")
                
                try:
                    # Model betanítása
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Metrikák számítása
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    dataset_results[model_name] = {
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'model': model
                    }
                    
                    print(f"    - Pontosság: {accuracy:.4f}")
                    print(f"    - Keresztvalidáció: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    print(f"    - Betanítás idő: {getattr(model, 'fit_time_', 'N/A')}")
                    
                except Exception as e:
                    print(f"    Hiba: {e}")
                    dataset_results[model_name] = None
            
            results[dataset_name] = dataset_results
        
        self.results = results
        return results
    
    def plot_performance_comparison(self):
        """Teljesítmény összehasonlítás ábrázolása"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Adat előkészítése
        comparison_data = []
        for dataset_name in self.datasets.keys():
            for model_name in self.models.keys():
                if (self.results[dataset_name][model_name] is not None and 
                    model_name in self.results[dataset_name]):
                    
                    result = self.results[dataset_name][model_name]
                    comparison_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Accuracy': result['accuracy'],
                        'CV_Mean': result['cv_mean']
                    })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # 1. Pontosság összehasonlítás
        accuracy_pivot = comp_df.pivot_table(
            values='Accuracy', index='Dataset', columns='Model'
        )
        accuracy_pivot.plot(kind='bar', ax=axes[0, 0], color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0, 0].set_title('Pontosság Összehasonlítás')
        axes[0, 0].set_ylabel('Pontosság')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Értékek hozzáadása az oszlopokhoz
        for i, (idx, row) in enumerate(accuracy_pivot.iterrows()):
            for j, val in enumerate(row):
                axes[0, 0].text(i + j*0.25 - 0.25, val + 0.01, f'{val:.3f}', 
                              ha='center', va='bottom', fontsize=9)
        
        # 2. Keresztvalidáció összehasonlítás
        cv_pivot = comp_df.pivot_table(
            values='CV_Mean', index='Dataset', columns='Model'
        )
        cv_pivot.plot(kind='bar', ax=axes[0, 1], color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0, 1].set_title('Keresztvalidáció Pontosság')
        axes[0, 1].set_ylabel('CV Pontosság')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Értékek hozzáadása az oszlopokhoz
        for i, (idx, row) in enumerate(cv_pivot.iterrows()):
            for j, val in enumerate(row):
                axes[0, 1].text(i + j*0.25 - 0.25, val + 0.01, f'{val:.3f}', 
                              ha='center', va='bottom', fontsize=9)
        
        # 3. Heatmap - pontosság
        heatmap_data = comp_df.pivot_table(
            values='Accuracy', index='Dataset', columns='Model'
        )
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=axes[1, 0],
                   cbar_kws={'label': 'Pontosság'}, fmt='.3f')
        axes[1, 0].set_title('Pontosság Heatmap')
        
        # 4. Heatmap - keresztvalidáció
        cv_heatmap_data = comp_df.pivot_table(
            values='CV_Mean', index='Dataset', columns='Model'
        )
        sns.heatmap(cv_heatmap_data, annot=True, cmap='viridis', ax=axes[1, 1],
                   cbar_kws={'label': 'CV Pontosság'}, fmt='.3f')
        axes[1, 1].set_title('Keresztvalidáció Heatmap')
        
        plt.tight_layout()
        plt.show()
        
        return comp_df
    
    def plot_decision_boundaries(self):
        """Döntési határok vizualizálása (csak 2D adathalmazra)"""
        dataset_name = 'Könnyű probléma (2 feature)'
        X, y = self.datasets[dataset_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Eredeti adatok
        scatter = axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        axes[0, 0].set_title('Eredeti adatok\n(Valós osztályok)')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Modellek predikciói
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        plot_positions = [(0, 1), (1, 0), (1, 1)]
        for (model_name, (i, j)) in zip(self.models.keys(), plot_positions):
            if (self.results[dataset_name][model_name] is not None and 
                model_name in self.results[dataset_name]):
                
                result = self.results[dataset_name][model_name]
                accuracy = result['accuracy']
                
                # Predikciók ábrázolása
                scatter = axes[i, j].scatter(X_test[:, 0], X_test[:, 1], 
                                           c=result['predictions'], cmap='coolwarm', alpha=0.6)
                axes[i, j].set_title(f'{model_name}\nPontosság: {accuracy:.3f}')
                axes[i, j].set_xlabel('Feature 1')
                axes[i, j].set_ylabel('Feature 2')
                axes[i, j].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[i, j])
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_report(self):
        """Részletes jelentés nyomtatása"""
        print("\n" + "="*70)
        print("RÉSZLETES ÖSSZEFOGLALÓ")
        print("="*70)
        
        summary_data = []
        
        for dataset_name in self.datasets.keys():
            print(f"\n{dataset_name}:")
            print("-" * len(dataset_name))
            
            for model_name in self.models.keys():
                if (self.results[dataset_name][model_name] is not None and 
                    model_name in self.results[dataset_name]):
                    
                    result = self.results[dataset_name][model_name]
                    
                    print(f"  {model_name}:")
                    print(f"    • Pontosság: {result['accuracy']:.4f}")
                    print(f"    • Keresztvalidáció: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
                    
                    summary_data.append({
                        'Adathalmaz': dataset_name,
                        'Modell': model_name,
                        'Pontosság': result['accuracy'],
                        'Keresztvalidáció': result['cv_mean']
                    })
        
        # Összegző táblázat
        summary_df = pd.DataFrame(summary_data)
        print("\n" + "="*70)
        print("ÖSSZEGZŐ TÁBLÁZAT")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """Fő program"""
    print("=== Boost Algoritmusok Összehasonlítása ===")
    print("AdaBoost vs GradientBoost vs HistGradientBoost\n")
    
    # Összehasonlító objektum létrehozása
    comparator = BoostComparison()
    
    # Adathalmazok generálása
    print("1. Adathalmazok generálása...")
    comparator.generate_datasets()
    
    # Modellek összehasonlítása
    print("2. Boost algoritmusok összehasonlítása...")
    comparator.run_comparison()
    
    # Teljesítmény összehasonlítás
    print("3. Teljesítmény összehasonlítás...")
    results_df = comparator.plot_performance_comparison()
    
    # Döntési határok
    print("4. Döntési határok vizualizálása...")
    comparator.plot_decision_boundaries()
    
    # Részletes jelentés
    print("5. Részletes eredmények...")
    summary_df = comparator.print_detailed_report()
    
    # Összefoglaló
    print("\n" + "="*70)
    print("ALGORITMUS ISMERTETŐ")
    print("="*70)
    print("AdaBoost:")
    print("  - Szekvenciálisan javítja a gyenge tanulókat")
    print("  - Minden mintára fókuszál, ami nehéz volt az előző iterációban")
    print("  - Érzékeny a zajos adatokra")
    
    print("\nGradientBoost:")
    print("  - Gradiens süllyedés alapú boosting")
    print("  - Az előző iteráció hibájára fókuszál")
    print("  - Lassabb, de általában pontosabb mint az AdaBoost")
    
    print("\nHistGradientBoost:")
    print("  - Optimalizált változat, gyorsabb a nagy adathalmazokon")
    print("  - Histogram alapú, kevesebb memóriát használ")
    print("  - Jól skálázódik nagyméretű adatokra")
    
    return comparator, summary_df

if __name__ == "__main__":
    comparator, results = main()