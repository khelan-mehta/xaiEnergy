"""
Building Clustering Module
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def aggregate_building_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to building level.
    
    Args:
        df: DataFrame with building data
    
    Returns:
        DataFrame with building-level statistics
    """
    print("\n" + "=" * 60)
    print("AGGREGATING BUILDING STATISTICS")
    print("=" * 60)
    
    building_stats = df.groupby('building_id').agg({
        'meter_reading': ['mean', 'std', 'max', 'min'],
        'performance_gap': ['mean', 'std'],
        'performance_gap_pct': 'mean',
        'air_temperature': 'mean',
        'square_feet': 'first',
        'primary_use': 'first',
        'year_built': 'first',
        'site_id': 'first'
    }).reset_index()
    
    # Flatten column names
    building_stats.columns = [
        'building_id', 'mean_energy', 'std_energy', 'max_energy', 'min_energy',
        'mean_gap', 'std_gap', 'mean_gap_pct', 'mean_temp',
        'square_feet', 'primary_use', 'year_built', 'site_id'
    ]
    
    # Derived features
    building_stats['energy_intensity'] = (
        building_stats['mean_energy'] / (building_stats['square_feet'] + 1)
    )
    building_stats['gap_intensity'] = (
        building_stats['mean_gap'] / (building_stats['square_feet'] + 1)
    )
    building_stats['cv_energy'] = (
        building_stats['std_energy'] / (building_stats['mean_energy'] + 1)
    )
    
    print(f"Aggregated {len(building_stats)} buildings")
    
    return building_stats


def perform_clustering(
    building_stats: pd.DataFrame,
    cluster_features: List[str] = None,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Perform K-Means clustering on buildings.
    
    Args:
        building_stats: Building-level DataFrame
        cluster_features: Features to use for clustering
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Tuple of (updated DataFrame, kmeans model, scaler)
    """
    print("\n" + "=" * 60)
    print("PERFORMING CLUSTERING")
    print("=" * 60)
    
    if cluster_features is None:
        cluster_features = ['mean_gap_pct', 'energy_intensity', 'std_gap', 'mean_temp']
    
    print(f"Clustering features: {cluster_features}")
    
    # Prepare data
    X_cluster = building_stats[cluster_features].dropna()
    valid_idx = X_cluster.index
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add to dataframe
    building_stats.loc[valid_idx, 'cluster'] = clusters
    
    print(f"\nCluster distribution:")
    print(building_stats['cluster'].value_counts().sort_index())
    
    return building_stats, kmeans, scaler


def find_optimal_k(
    building_stats: pd.DataFrame,
    cluster_features: List[str] = None,
    k_range: range = range(2, 8),
    save_path: Optional[Path] = None
) -> int:
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        building_stats: Building-level DataFrame
        cluster_features: Features for clustering
        k_range: Range of K values to try
        save_path: Path to save elbow plot
    
    Returns:
        Suggested optimal K
    """
    if cluster_features is None:
        cluster_features = ['mean_gap_pct', 'energy_intensity', 'std_gap', 'mean_temp']
    
    X_cluster = building_stats[cluster_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return 4  # Default recommendation


def characterize_clusters(building_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cluster characterization summary.
    
    Args:
        building_stats: DataFrame with cluster assignments
    
    Returns:
        DataFrame with cluster characteristics
    """
    print("\n" + "=" * 60)
    print("CLUSTER CHARACTERIZATION")
    print("=" * 60)
    
    cluster_summary = building_stats.groupby('cluster').agg({
        'building_id': 'count',
        'mean_gap_pct': ['mean', 'std'],
        'mean_energy': 'mean',
        'energy_intensity': 'mean',
        'square_feet': 'mean',
        'year_built': 'mean'
    }).round(2)
    
    cluster_summary.columns = [
        'n_buildings', 'mean_gap_%', 'std_gap_%',
        'mean_energy', 'energy_intensity',
        'mean_sqft', 'mean_year_built'
    ]
    
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Dominant building types per cluster
    print("\nDominant Building Types per Cluster:")
    for cluster in sorted(building_stats['cluster'].dropna().unique()):
        cluster_buildings = building_stats[building_stats['cluster'] == cluster]
        top_use = cluster_buildings['primary_use'].value_counts().head(3)
        print(f"\nCluster {int(cluster)}:")
        for use, count in top_use.items():
            print(f"  - {use}: {count} buildings")
    
    return cluster_summary


def plot_clusters(
    building_stats: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Create cluster visualization plots.
    
    Args:
        building_stats: DataFrame with cluster assignments
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Gap vs Intensity scatter
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        building_stats['energy_intensity'],
        building_stats['mean_gap_pct'],
        c=building_stats['cluster'],
        cmap='viridis', alpha=0.6, s=50
    )
    ax1.set_xlabel('Energy Intensity (kWh/sqft)')
    ax1.set_ylabel('Mean Performance Gap (%)')
    ax1.set_title('Building Clusters: Gap vs Intensity')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. Gap boxplot by cluster
    ax2 = axes[0, 1]
    building_stats.boxplot(column='mean_gap_pct', by='cluster', ax=ax2)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Mean Performance Gap (%)')
    ax2.set_title('Performance Gap Distribution by Cluster')
    plt.suptitle('')
    
    # 3. Energy boxplot by cluster
    ax3 = axes[1, 0]
    building_stats.boxplot(column='mean_energy', by='cluster', ax=ax3)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Mean Energy (kWh)')
    ax3.set_title('Energy Consumption by Cluster')
    plt.suptitle('')
    
    # 4. Building type distribution
    ax4 = axes[1, 1]
    cluster_use = pd.crosstab(building_stats['cluster'], building_stats['primary_use'])
    # Select top 5 building types
    top_types = building_stats['primary_use'].value_counts().head(5).index
    cluster_use_top = cluster_use[top_types]
    cluster_use_top.plot(kind='bar', stacked=True, ax=ax4, colormap='tab10')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Buildings')
    ax4.set_title('Building Types by Cluster (Top 5)')
    ax4.legend(title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def run_clustering_analysis(
    df: pd.DataFrame,
    n_clusters: int = 4,
    figures_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete clustering analysis.
    
    Args:
        df: Full DataFrame with all data
        n_clusters: Number of clusters
        figures_dir: Directory to save figures
    
    Returns:
        Tuple of (building_stats, cluster_summary)
    """
    # Aggregate to building level
    building_stats = aggregate_building_stats(df)
    
    # Find optimal K (optional)
    if figures_dir:
        find_optimal_k(
            building_stats,
            save_path=figures_dir / 'elbow_plot.png'
        )
    
    # Perform clustering
    building_stats, kmeans, scaler = perform_clustering(
        building_stats, n_clusters=n_clusters
    )
    
    # Characterize clusters
    cluster_summary = characterize_clusters(building_stats)
    
    # Plot clusters
    plot_clusters(
        building_stats,
        save_path=figures_dir / 'clusters.png' if figures_dir else None
    )
    
    return building_stats, cluster_summary


if __name__ == "__main__":
    print("Clustering module loaded successfully")
