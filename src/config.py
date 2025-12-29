"""
Configuration settings for ASHRAE XAI Performance Gap Analysis
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
BUILDING_FILE = DATA_DIR / "building_metadata.csv"
WEATHER_FILE = DATA_DIR / "weather_train.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Meter type to analyze (0=electricity, 1=chilledwater, 2=steam, 3=hotwater)
METER_TYPE = 0
METER_NAMES = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

# Data cleaning thresholds
OUTLIER_QUANTILE = 0.999  # Remove top 0.1%
ZERO_READING_THRESHOLD = 0.5  # Remove buildings with >50% zeros

# Train-test split ratio
TRAIN_RATIO = 0.7

# =============================================================================
# FEATURE SETTINGS
# =============================================================================

# Business hours definition (for baseline calculation)
BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18

# Lag features to create
LAG_HOURS = [1, 2, 3, 24]

# Rolling window size
ROLLING_WINDOW = 24

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Sampling for faster training (set to None for full data)
SAMPLE_SIZE = 500000
SHAP_SAMPLE_SIZE = 10000

# LightGBM parameters
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1
}

LGB_NUM_ROUNDS = 500
LGB_EARLY_STOPPING = 50

# LSTM parameters
LSTM_SEQUENCE_LENGTH = 24
LSTM_SAMPLE_SIZE = 200000
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 256
LSTM_PATIENCE = 10

# Clustering
N_CLUSTERS = 4

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
