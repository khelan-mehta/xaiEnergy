"""
Explainability Module (SHAP Analysis)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    sample_size: int = 10000
) -> Tuple[np.ndarray, shap.TreeExplainer, np.ndarray]:
    """
    Compute SHAP values for a tree-based model.
    
    Args:
        model: Trained LightGBM model
        X: Feature matrix
        feature_names: List of feature names
        sample_size: Number of samples for SHAP computation
    
    Returns:
        Tuple of (shap_values, explainer, X_sample)
    """
    print("\n" + "=" * 60)
    print("COMPUTING SHAP VALUES")
    print("=" * 60)
    
    # Sample data
    if sample_size and sample_size < len(X):
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
        print(f"Using sample of {sample_size:,} observations")
    else:
        X_sample = X
        sample_idx = np.arange(len(X))
    
    # Create explainer
    print("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    print("Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values, explainer, X_sample, sample_idx


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "SHAP Summary Plot"
):
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_shap_importance(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "SHAP Feature Importance"
):
    """
    Create SHAP feature importance bar plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                      plot_type='bar', show=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    expected_value: float,
    idx: int,
    save_path: Optional[Path] = None,
    title: str = "SHAP Waterfall",
    max_display: int = 15
):
    """
    Create SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        expected_value: Base value from explainer
        idx: Index of sample to explain
        save_path: Path to save figure
        title: Plot title
        max_display: Maximum features to display
    """
    plt.figure(figsize=(12, 8))
    
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X[idx],
        feature_names=feature_names
    )
    
    shap.waterfall_plot(explanation, show=False, max_display=max_display)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 4,
    save_path: Optional[Path] = None
):
    """
    Create SHAP dependence plots for top features.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_path: Path to save figure
    """
    # Get top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:][::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (ax, feat_idx) in enumerate(zip(axes.flat, top_indices)):
        shap.dependence_plot(
            feat_idx, shap_values, X,
            feature_names=feature_names, ax=ax, show=False
        )
        ax.set_title(f'SHAP Dependence: {feature_names[feat_idx]}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_shap_heatmap_by_hour(
    shap_values: np.ndarray,
    hours: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[Path] = None
):
    """
    Create SHAP importance heatmap by hour of day.
    
    Args:
        shap_values: SHAP values array
        hours: Array of hour values
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Get mean importance by hour
    shap_by_hour = pd.DataFrame()
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            shap_by_hour[h] = pd.Series(
                np.abs(shap_values[mask]).mean(axis=0),
                index=feature_names
            )
    
    # Select top features
    mean_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    heatmap_data = shap_by_hour.loc[top_features]
    
    plt.figure(figsize=(14, 8))
    import seaborn as sns
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False,
                xticklabels=[f'{h:02d}:00' for h in range(24)])
    plt.xlabel('Hour of Day')
    plt.ylabel('Feature')
    plt.title('SHAP Feature Importance by Hour of Day', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def run_shap_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    hours: Optional[np.ndarray] = None,
    sample_size: int = 10000,
    figures_dir: Optional[Path] = None
) -> Dict:
    """
    Run complete SHAP analysis.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values (for finding extreme cases)
        feature_names: List of feature names
        hours: Array of hour values (for heatmap)
        sample_size: Number of samples for SHAP
        figures_dir: Directory to save figures
    
    Returns:
        Dictionary with SHAP results
    """
    print("\n" + "=" * 60)
    print("RUNNING SHAP ANALYSIS")
    print("=" * 60)
    
    # Compute SHAP values
    shap_values, explainer, X_sample, sample_idx = compute_shap_values(
        model, X, feature_names, sample_size
    )
    
    # Get feature importance
    importance_df = get_feature_importance(shap_values, feature_names)
    print("\nTop 10 Features by SHAP Importance:")
    print(importance_df.head(10).to_string(index=False))
    
    # Create plots
    print("\nGenerating SHAP plots...")
    
    # Summary plot
    plot_shap_summary(
        shap_values, X_sample, feature_names,
        save_path=figures_dir / 'shap_summary.png' if figures_dir else None,
        title='SHAP Summary - Performance Gap Prediction'
    )
    
    # Importance bar plot
    plot_shap_importance(
        shap_values, X_sample, feature_names,
        save_path=figures_dir / 'shap_importance.png' if figures_dir else None,
        title='SHAP Feature Importance'
    )
    
    # Waterfall plots for extreme cases
    y_sample = y[sample_idx]
    high_gap_idx = np.argmax(y_sample)
    low_gap_idx = np.argmin(y_sample)
    
    plot_shap_waterfall(
        shap_values, X_sample, feature_names, explainer.expected_value,
        high_gap_idx,
        save_path=figures_dir / 'shap_waterfall_high.png' if figures_dir else None,
        title=f'High Gap Day (Gap = {y_sample[high_gap_idx]:.0f} kWh)'
    )
    
    plot_shap_waterfall(
        shap_values, X_sample, feature_names, explainer.expected_value,
        low_gap_idx,
        save_path=figures_dir / 'shap_waterfall_low.png' if figures_dir else None,
        title=f'Low Gap Day (Gap = {y_sample[low_gap_idx]:.0f} kWh)'
    )
    
    # Dependence plots
    plot_shap_dependence(
        shap_values, X_sample, feature_names,
        save_path=figures_dir / 'shap_dependence.png' if figures_dir else None
    )
    
    # Heatmap by hour
    if hours is not None:
        hours_sample = hours[sample_idx]
        plot_shap_heatmap_by_hour(
            shap_values, hours_sample, feature_names,
            save_path=figures_dir / 'shap_heatmap.png' if figures_dir else None
        )
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_sample': X_sample,
        'sample_idx': sample_idx,
        'importance': importance_df
    }


if __name__ == "__main__":
    print("Explainability module loaded successfully")
