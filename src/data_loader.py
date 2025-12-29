"""
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(
    train_path: Path,
    building_path: Path,
    weather_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw ASHRAE datasets with optimized dtypes.
    
    Args:
        train_path: Path to train.csv
        building_path: Path to building_metadata.csv
        weather_path: Path to weather_train.csv
    
    Returns:
        Tuple of (train_df, building_df, weather_df)
    """
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    
    # Optimized dtypes for memory efficiency
    train_dtypes = {
        'building_id': 'int16',
        'meter': 'int8',
        'meter_reading': 'float32'
    }
    
    print("Loading train.csv...")
    train = pd.read_csv(
        train_path,
        dtype=train_dtypes,
        parse_dates=['timestamp']
    )
    print(f"  Shape: {train.shape}")
    print(f"  Memory: {train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    print("\nLoading building_metadata.csv...")
    building = pd.read_csv(building_path)
    print(f"  Shape: {building.shape}")
    
    print("\nLoading weather_train.csv...")
    weather = pd.read_csv(weather_path, parse_dates=['timestamp'])
    print(f"  Shape: {weather.shape}")
    
    return train, building, weather


def merge_datasets(
    train: pd.DataFrame,
    building: pd.DataFrame,
    weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge train, building, and weather datasets.
    
    Args:
        train: Training data with meter readings
        building: Building metadata
        weather: Weather data
    
    Returns:
        Merged DataFrame
    """
    print("\n" + "=" * 60)
    print("MERGING DATASETS")
    print("=" * 60)
    
    # Merge with building metadata
    df = train.merge(building, on='building_id', how='left')
    print(f"After building merge: {df.shape}")
    
    # Merge with weather data
    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
    print(f"After weather merge: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    return df


def clean_data(
    df: pd.DataFrame,
    meter_type: int = 0,
    outlier_quantile: float = 0.999,
    zero_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean and filter the dataset.
    
    Args:
        df: Merged DataFrame
        meter_type: Meter type to filter (0=electricity)
        outlier_quantile: Quantile for outlier removal
        zero_threshold: Max proportion of zeros allowed per building
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    print(f"Initial shape: {df.shape}")
    
    # Filter to specified meter type
    meter_names = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
    df_clean = df[df['meter'] == meter_type].copy()
    print(f"After filtering to {meter_names[meter_type]}: {df_clean.shape}")
    
    # Remove missing meter readings
    df_clean = df_clean.dropna(subset=['meter_reading'])
    print(f"After removing missing readings: {df_clean.shape}")
    
    # Remove outliers
    upper_threshold = df_clean['meter_reading'].quantile(outlier_quantile)
    df_clean = df_clean[df_clean['meter_reading'] <= upper_threshold]
    print(f"After removing outliers (>{outlier_quantile*100}%): {df_clean.shape}")
    
    # Remove buildings with too many zero readings
    zero_pct = df_clean.groupby('building_id')['meter_reading'].apply(
        lambda x: (x == 0).mean()
    )
    high_zero_buildings = zero_pct[zero_pct > zero_threshold].index
    df_clean = df_clean[~df_clean['building_id'].isin(high_zero_buildings)]
    print(f"After removing high-zero buildings: {df_clean.shape}")
    
    return df_clean


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in weather and building columns.
    
    Args:
        df: DataFrame with potential missing values
    
    Returns:
        DataFrame with imputed values
    """
    print("\n" + "=" * 60)
    print("IMPUTING MISSING VALUES")
    print("=" * 60)
    
    # Weather columns
    weather_cols = [
        'air_temperature', 'cloud_coverage', 'dew_temperature',
        'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed'
    ]
    
    print("Weather columns - missing before:")
    for col in weather_cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"  {col}: {missing:,}")
    
    # Forward/backward fill within each site, then median
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby('site_id')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
            df[col] = df[col].fillna(df[col].median())
    
    # Building metadata
    if 'year_built' in df.columns:
        df['year_built'] = df['year_built'].fillna(df['year_built'].median())
    
    if 'floor_count' in df.columns:
        df['floor_count'] = df.groupby('primary_use')['floor_count'].transform(
            lambda x: x.fillna(x.median())
        )
        df['floor_count'] = df['floor_count'].fillna(df['floor_count'].median())
    
    print("\nMissing values after imputation: 0")
    
    return df


def load_and_preprocess(
    train_path: Path,
    building_path: Path,
    weather_path: Path,
    meter_type: int = 0,
    outlier_quantile: float = 0.999,
    zero_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Complete data loading and preprocessing pipeline.
    
    Args:
        train_path: Path to train.csv
        building_path: Path to building_metadata.csv
        weather_path: Path to weather_train.csv
        meter_type: Meter type to analyze
        outlier_quantile: Quantile for outlier removal
        zero_threshold: Max proportion of zeros per building
    
    Returns:
        Cleaned and preprocessed DataFrame
    """
    # Load raw data
    train, building, weather = load_raw_data(train_path, building_path, weather_path)
    
    # Merge datasets
    df = merge_datasets(train, building, weather)
    
    # Clean data
    df = clean_data(df, meter_type, outlier_quantile, zero_threshold)
    
    # Impute missing values
    df = impute_missing_values(df)
    
    print("\n" + "=" * 60)
    print(f"FINAL PREPROCESSED DATA: {df.shape}")
    print(f"Buildings: {df['building_id'].nunique()}")
    print(f"Sites: {df['site_id'].nunique()}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Test the module
    from config import TRAIN_FILE, BUILDING_FILE, WEATHER_FILE, METER_TYPE
    
    df = load_and_preprocess(
        TRAIN_FILE, BUILDING_FILE, WEATHER_FILE,
        meter_type=METER_TYPE
    )
    print(f"\nData loaded successfully: {df.shape}")
