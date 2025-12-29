"""
Visualization Module
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def plot_data_exploration(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Create data exploration plots.
    
    Args:
        df: DataFrame with energy data
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Meter reading distribution
    ax1 = axes[0, 0]
    df['meter_reading'].hist(bins=100, ax=ax1, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Meter Reading (kWh)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Electricity Consumption')
    
    # 2. Log-transformed distribution
    ax2 = axes[0, 1]
    np.log1p(df['meter_reading']).hist(bins=100, ax=ax2, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Log(Meter Reading + 1)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Log-Transformed Distribution')
    
    # 3. Consumption by building type
    ax3 = axes[1, 0]
    df.groupby('primary_use')['meter_reading'].mean().sort_values().plot(
        kind='barh', ax=ax3, color='steelblue'
    )
    ax3.set_xlabel('Mean Meter Reading (kWh)')
    ax3.set_title('Average Consumption by Building Type')
    
    # 4. Hourly pattern
    ax4 = axes[1, 1]
    hourly_mean = df.groupby('hour')['meter_reading'].mean()
    ax4.plot(hourly_mean.index, hourly_mean.values, 'o-', linewidth=2, markersize=6)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Mean Meter Reading (kWh)')
    ax4.set_title('Hourly Consumption Pattern')
    ax4.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_performance_gap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Create performance gap visualization plots.
    
    Args:
        df: DataFrame with gap data
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Baseline by hour
    ax1 = axes[0, 0]
    hourly_actual = df.groupby('hour')['meter_reading'].mean()
    hourly_baseline = df.groupby('hour')['baseline_median'].mean()
    ax1.plot(hourly_actual.index, hourly_actual.values, 'b-', label='Actual', linewidth=2)
    ax1.plot(hourly_baseline.index, hourly_baseline.values, 'r--', label='Baseline', linewidth=2)
    ax1.fill_between(hourly_actual.index, hourly_baseline.values, hourly_actual.values,
                     alpha=0.3, color='orange', label='Gap')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('Actual vs Designed Energy by Hour')
    ax1.legend()
    
    # 2. Gap distribution
    ax2 = axes[0, 1]
    df['performance_gap_pct'].hist(bins=100, ax=ax2, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Gap')
    ax2.set_xlabel('Performance Gap (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Performance Gap')
    ax2.legend()
    
    # 3. Gap by building type
    ax3 = axes[1, 0]
    gap_by_use = df.groupby('primary_use')['performance_gap_pct'].mean().sort_values()
    colors = ['green' if x < 0 else 'red' for x in gap_by_use.values]
    gap_by_use.plot(kind='barh', ax=ax3, color=colors)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Mean Performance Gap (%)')
    ax3.set_title('Performance Gap by Building Type')
    
    # 4. Gap by day of week
    ax4 = axes[1, 1]
    gap_by_day = df.groupby('dayofweek')['performance_gap_pct'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax4.bar(days, gap_by_day.values, color='steelblue', edgecolor='black')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Mean Performance Gap (%)')
    ax4.set_title('Performance Gap by Day of Week')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_model_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Create model comparison plots.
    
    Args:
        results: Dictionary with model results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Linear Reg', 'LightGBM', 'LSTM']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Get metrics
    lr_metrics = results['lr']['metrics']
    lgb_metrics = results['lgb_energy']['metrics']
    lstm_metrics = results['lstm']['metrics']
    
    # RMSE
    rmse_values = [lr_metrics['RMSE'], lgb_metrics['RMSE'], lstm_metrics['RMSE']]
    axes[0].bar(models, rmse_values, color=colors, edgecolor='black')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 1, f'{v:.1f}', ha='center')
    
    # R²
    r2_values = [lr_metrics['R2'], lgb_metrics['R2'], lstm_metrics['R2']]
    axes[1].bar(models, r2_values, color=colors, edgecolor='black')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² Comparison')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Actual vs Predicted (LightGBM)
    y_test = results['y_test_meter']
    lgb_pred = results['lgb_energy']['predictions']
    sample = min(5000, len(y_test))
    axes[2].scatter(y_test[:sample], lgb_pred[:sample], alpha=0.3, s=10)
    max_val = max(y_test[:sample].max(), lgb_pred[:sample].max())
    axes[2].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].set_title(f'LightGBM: Actual vs Predicted (R²={lgb_metrics["R2"]:.3f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_lstm_training(
    history: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot LSTM training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('LSTM Training - Loss')
    axes[0].legend()
    
    # MAE
    axes[1].plot(history['mae'], label='Train MAE')
    axes[1].plot(history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('LSTM Training - MAE')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_summary_dashboard(
    df: pd.DataFrame,
    results: Dict,
    shap_importance: pd.DataFrame,
    building_stats: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Create summary results dashboard.
    
    Args:
        df: Full DataFrame
        results: Model results dictionary
        shap_importance: SHAP feature importance DataFrame
        building_stats: Building statistics DataFrame
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Get metrics
    lr_metrics = results['lr']['metrics']
    lgb_metrics = results['lgb_energy']['metrics']
    lstm_metrics = results['lstm']['metrics']
    
    # 1. Gap Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    gap_categories = pd.cut(df['performance_gap_pct'],
                           bins=[-100, -20, 0, 20, 50, 100, 500],
                           labels=['<-20%', '-20-0%', '0-20%', '20-50%', '50-100%', '>100%'])
    gap_categories.value_counts().sort_index().plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Gap Range')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Performance Gap Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Model Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['LR', 'LGB', 'LSTM']
    r2_values = [lr_metrics['R2'], lgb_metrics['R2'], lstm_metrics['R2']]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    ax2.bar(models, r2_values, color=colors, edgecolor='black')
    ax2.set_ylabel('R²')
    ax2.set_title('Model R² Comparison')
    ax2.set_ylim(0, 1)
    
    # 3. Top SHAP Features
    ax3 = fig.add_subplot(gs[0, 2])
    top_10 = shap_importance.head(10)
    ax3.barh(range(10), top_10['importance'].values[::-1], color='coral', edgecolor='black')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top_10['feature'].values[::-1])
    ax3.set_xlabel('Mean |SHAP|')
    ax3.set_title('Top 10 Features')
    
    # 4. Actual vs Predicted
    ax4 = fig.add_subplot(gs[1, 0])
    y_test = results['y_test_meter']
    lgb_pred = results['lgb_energy']['predictions']
    sample = min(5000, len(y_test))
    ax4.scatter(y_test[:sample], lgb_pred[:sample], alpha=0.3, s=10, c='steelblue')
    max_val = max(y_test[:sample].max(), lgb_pred[:sample].max())
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax4.set_xlabel('Actual')
    ax4.set_ylabel('Predicted')
    ax4.set_title(f'LightGBM (R²={lgb_metrics["R2"]:.3f})')
    
    # 5. Gap by Hour
    ax5 = fig.add_subplot(gs[1, 1])
    hourly_gap = df.groupby('hour')['performance_gap_pct'].mean()
    ax5.bar(hourly_gap.index, hourly_gap.values, color='teal', edgecolor='black')
    ax5.axhline(y=0, color='red', linestyle='--')
    ax5.set_xlabel('Hour')
    ax5.set_ylabel('Mean Gap (%)')
    ax5.set_title('Gap by Hour')
    
    # 6. Gap by Building Type
    ax6 = fig.add_subplot(gs[1, 2])
    gap_by_type = df.groupby('primary_use')['performance_gap_pct'].mean().sort_values()
    top_types = gap_by_type.tail(8)
    colors = ['green' if x < 0 else 'red' for x in top_types.values]
    top_types.plot(kind='barh', ax=ax6, color=colors)
    ax6.axvline(x=0, color='black')
    ax6.set_xlabel('Mean Gap (%)')
    ax6.set_title('Gap by Building Type')
    
    # 7. Cluster Summary (bottom row)
    ax7 = fig.add_subplot(gs[2, :])
    cluster_gap = building_stats.groupby('cluster')['mean_gap_pct'].mean()
    cluster_counts = building_stats['cluster'].value_counts().sort_index()
    
    x = np.arange(len(cluster_gap))
    width = 0.35
    
    ax7_twin = ax7.twinx()
    bars1 = ax7.bar(x - width/2, cluster_gap.values, width, label='Mean Gap %', color='coral', edgecolor='black')
    bars2 = ax7_twin.bar(x + width/2, cluster_counts.values, width, label='N Buildings', color='steelblue', edgecolor='black')
    
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Mean Gap (%)', color='coral')
    ax7_twin.set_ylabel('N Buildings', color='steelblue')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'Cluster {int(c)}' for c in cluster_gap.index])
    ax7.set_title('Building Archetypes')
    ax7.axhline(y=0, color='black', linestyle='--')
    ax7.legend(loc='upper left')
    ax7_twin.legend(loc='upper right')
    
    plt.suptitle('Explainable AI Analysis of Building Energy Performance Gap',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully")
