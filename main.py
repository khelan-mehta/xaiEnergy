#!/usr/bin/env python3
"""
ASHRAE XAI Performance Gap Analysis
====================================
Main execution script for the complete analysis pipeline.

Usage:
    python main.py
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import project modules
from config import (
    TRAIN_FILE, BUILDING_FILE, WEATHER_FILE,
    OUTPUT_DIR, FIGURES_DIR, MODELS_DIR,
    METER_TYPE, OUTLIER_QUANTILE, ZERO_READING_THRESHOLD,
    TRAIN_RATIO, BUSINESS_HOUR_START, BUSINESS_HOUR_END,
    LAG_HOURS, ROLLING_WINDOW, SAMPLE_SIZE, SHAP_SAMPLE_SIZE,
    LGB_PARAMS, LGB_NUM_ROUNDS, LGB_EARLY_STOPPING,
    LSTM_SEQUENCE_LENGTH, LSTM_SAMPLE_SIZE, LSTM_EPOCHS,
    LSTM_BATCH_SIZE, LSTM_PATIENCE, N_CLUSTERS, RANDOM_SEED
)

from data_loader import load_and_preprocess
from feature_engineering import engineer_all_features, get_feature_lists
from models import temporal_train_test_split, train_all_models
from explainability import run_shap_analysis
from clustering import run_clustering_analysis
from visualization import (
    plot_data_exploration, plot_performance_gap,
    plot_model_comparison, plot_lstm_training,
    plot_summary_dashboard
)


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("ASHRAE XAI PERFORMANCE GAP ANALYSIS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # =========================================================================
    # STEP 1: Load and Preprocess Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 70)
    
    df = load_and_preprocess(
        TRAIN_FILE, BUILDING_FILE, WEATHER_FILE,
        meter_type=METER_TYPE,
        outlier_quantile=OUTLIER_QUANTILE,
        zero_threshold=ZERO_READING_THRESHOLD
    )
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    df, label_encoder, ALL_FEATURES = engineer_all_features(
        df,
        business_hour_start=BUSINESS_HOUR_START,
        business_hour_end=BUSINESS_HOUR_END,
        lag_hours=LAG_HOURS,
        rolling_window=ROLLING_WINDOW
    )
    
    # =========================================================================
    # STEP 3: Visualize Data and Performance Gap
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: DATA VISUALIZATION")
    print("=" * 70)
    
    print("\nCreating data exploration plots...")
    plot_data_exploration(df, save_path=FIGURES_DIR / 'data_exploration.png')
    
    print("\nCreating performance gap plots...")
    plot_performance_gap(df, save_path=FIGURES_DIR / 'performance_gap.png')
    
    # =========================================================================
    # STEP 4: Train-Test Split
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    train_df, test_df, split_date = temporal_train_test_split(df, TRAIN_RATIO)
    
    # =========================================================================
    # STEP 5: Train Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING")
    print("=" * 70)
    
    results = train_all_models(
        train_df, test_df, ALL_FEATURES,
        lgb_params=LGB_PARAMS,
        sample_size=SAMPLE_SIZE,
        lstm_sample_size=LSTM_SAMPLE_SIZE
    )
    
    # Plot LSTM training history
    print("\nPlotting LSTM training history...")
    plot_lstm_training(
        results['lstm']['history'],
        save_path=FIGURES_DIR / 'lstm_training.png'
    )
    
    # Plot model comparison
    print("\nPlotting model comparison...")
    plot_model_comparison(results, save_path=FIGURES_DIR / 'model_comparison.png')
    
    # =========================================================================
    # STEP 6: SHAP Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    
    # Get test data for SHAP
    X_test = test_df[ALL_FEATURES].values
    y_test_gap = test_df['performance_gap'].values
    hours_test = test_df['hour'].values
    
    shap_results = run_shap_analysis(
        results['lgb_gap']['model'],
        X_test, y_test_gap, ALL_FEATURES,
        hours=hours_test,
        sample_size=SHAP_SAMPLE_SIZE,
        figures_dir=FIGURES_DIR
    )
    
    # =========================================================================
    # STEP 7: Building Clustering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: BUILDING CLUSTERING")
    print("=" * 70)
    
    building_stats, cluster_summary = run_clustering_analysis(
        df, n_clusters=N_CLUSTERS, figures_dir=FIGURES_DIR
    )
    
    # =========================================================================
    # STEP 8: Summary Dashboard
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: SUMMARY DASHBOARD")
    print("=" * 70)
    
    plot_summary_dashboard(
        df, results, shap_results['importance'], building_stats,
        save_path=FIGURES_DIR / 'summary_dashboard.png'
    )
    
    # =========================================================================
    # STEP 9: Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: SAVING RESULTS")
    print("=" * 70)
    
    # Model metrics summary
    metrics_summary = {
        'Linear Regression': results['lr']['metrics'],
        'LightGBM (Energy)': results['lgb_energy']['metrics'],
        'LightGBM (Gap)': results['lgb_gap']['metrics'],
        'LSTM': results['lstm']['metrics']
    }
    
    # Performance gap statistics
    gap_stats = {
        'mean_gap_kwh': float(df['performance_gap'].mean()),
        'std_gap_kwh': float(df['performance_gap'].std()),
        'mean_gap_pct': float(df['performance_gap_pct'].mean()),
        'std_gap_pct': float(df['performance_gap_pct'].std()),
        'pct_above_baseline': float((df['performance_gap'] > 0).mean() * 100)
    }
    
    # Top SHAP features
    top_shap_features = shap_results['importance'].head(10)['feature'].tolist()
    
    # Full results summary
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_info': {
            'n_rows': len(df),
            'n_buildings': int(df['building_id'].nunique()),
            'n_sites': int(df['site_id'].nunique()),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'split_date': split_date
        },
        'performance_gap': gap_stats,
        'model_metrics': {k: {m: float(v) for m, v in metrics.items()} 
                         for k, metrics in metrics_summary.items()},
        'top_shap_features': top_shap_features,
        'n_clusters': N_CLUSTERS,
        'cluster_summary': cluster_summary.to_dict()
    }
    
    # Save JSON
    results_path = OUTPUT_DIR / 'results_summary.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"Saved: {results_path}")
    
    # Save building clusters
    clusters_path = OUTPUT_DIR / 'building_clusters.csv'
    building_stats.to_csv(clusters_path, index=False)
    print(f"Saved: {clusters_path}")
    
    # Save SHAP importance
    shap_path = OUTPUT_DIR / 'shap_importance.csv'
    shap_results['importance'].to_csv(shap_path, index=False)
    print(f"Saved: {shap_path}")
    
    # Save models
    results['lgb_energy']['model'].save_model(str(MODELS_DIR / 'lgb_energy_model.txt'))
    results['lgb_gap']['model'].save_model(str(MODELS_DIR / 'lgb_gap_model.txt'))
    results['lstm']['model'].save(str(MODELS_DIR / 'lstm_model.h5'))
    print(f"Saved models to: {MODELS_DIR}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\n### Key Results ###\n")
    
    print("Performance Gap:")
    print(f"  - Mean gap: {gap_stats['mean_gap_kwh']:.2f} kWh ({gap_stats['mean_gap_pct']:.1f}%)")
    print(f"  - % above baseline: {gap_stats['pct_above_baseline']:.1f}%")
    
    print("\nBest Model (LightGBM - Energy):")
    lgb_metrics = results['lgb_energy']['metrics']
    print(f"  - R²: {lgb_metrics['R2']:.4f}")
    print(f"  - RMSE: {lgb_metrics['RMSE']:.2f} kWh")
    
    print("\nTop 5 SHAP Features:")
    for i, feat in enumerate(top_shap_features[:5], 1):
        print(f"  {i}. {feat}")
    
    print("\nBuilding Clusters:")
    for cluster in sorted(building_stats['cluster'].dropna().unique()):
        n = len(building_stats[building_stats['cluster'] == cluster])
        gap = building_stats[building_stats['cluster'] == cluster]['mean_gap_pct'].mean()
        print(f"  Cluster {int(cluster)}: {n} buildings, avg gap = {gap:.1f}%")
    
    print("\n### Output Files ###\n")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Results: {OUTPUT_DIR}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results_summary


if __name__ == "__main__":
    results = main()
