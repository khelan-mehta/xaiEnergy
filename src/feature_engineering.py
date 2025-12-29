"""
Feature Engineering Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from timestamp.
    
    Args:
        df: DataFrame with 'timestamp' column
    
    Returns:
        DataFrame with temporal features added
    """
    print("Creating temporal features...")
    
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    # Binary flags
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    print(f"  Created 12 temporal features")
    
    return df


def create_baseline_features(
    df: pd.DataFrame,
    business_hour_start: int = 9,
    business_hour_end: int = 18
) -> pd.DataFrame:
    """
    Create design baseline and performance gap features.
    
    This is the KEY INNOVATION of the research.
    
    Args:
        df: DataFrame with meter readings and temporal features
        business_hour_start: Start of business hours
        business_hour_end: End of business hours
    
    Returns:
        DataFrame with baseline and gap features
    """
    print("Creating baseline and performance gap features...")
    
    # Define business hours
    df['is_business_hours'] = (
        (df['hour'] >= business_hour_start) & 
        (df['hour'] <= business_hour_end) & 
        (df['is_weekend'] == 0)
    ).astype(int)
    
    # Calculate baseline: median during business hours by building and month
    baseline_stats = df[df['is_business_hours'] == 1].groupby(
        ['building_id', 'month']
    )['meter_reading'].agg(['median', 'std']).reset_index()
    baseline_stats.columns = ['building_id', 'month', 'baseline_median', 'baseline_std']
    
    # Merge baseline back
    df = df.merge(baseline_stats, on=['building_id', 'month'], how='left')
    
    # Fill missing baselines with overall building median
    overall_baseline = df.groupby('building_id')['meter_reading'].transform('median')
    df['baseline_median'] = df['baseline_median'].fillna(overall_baseline)
    df['baseline_std'] = df['baseline_std'].fillna(0)
    
    # Calculate performance gap
    df['performance_gap'] = df['meter_reading'] - df['baseline_median']
    df['performance_gap_pct'] = (
        df['performance_gap'] / (df['baseline_median'] + 1)
    ) * 100
    df['performance_gap_pct'] = df['performance_gap_pct'].clip(-100, 500)
    
    print(f"  Baseline and gap features created")
    print(f"  Mean gap: {df['performance_gap'].mean():.2f} kWh")
    print(f"  Mean gap %: {df['performance_gap_pct'].mean():.1f}%")
    
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived weather features.
    
    Args:
        df: DataFrame with weather columns
    
    Returns:
        DataFrame with weather features added
    """
    print("Creating weather features...")
    
    # Temperature deviation from site monthly median
    df['temp_median'] = df.groupby(['site_id', 'month'])['air_temperature'].transform('median')
    df['temp_deviation'] = df['air_temperature'] - df['temp_median']
    
    # Comfort index (distance from 22°C)
    df['comfort_index'] = np.abs(df['air_temperature'] - 22)
    
    # Humidity proxy (dew point depression)
    if 'dew_temperature' in df.columns and 'air_temperature' in df.columns:
        df['dew_point_depression'] = df['air_temperature'] - df['dew_temperature']
    
    print(f"  Created 4 weather features")
    
    return df


def create_building_features(df: pd.DataFrame, reference_year: int = 2017) -> pd.DataFrame:
    """
    Create building-related features.
    
    Args:
        df: DataFrame with building metadata
        reference_year: Year for age calculation
    
    Returns:
        DataFrame with building features added
    """
    print("Creating building features...")
    
    # Building age
    df['building_age'] = reference_year - df['year_built']
    df['building_age'] = df['building_age'].clip(0, 200)  # Sanity check
    
    # Energy intensity
    df['energy_intensity'] = df['meter_reading'] / (df['square_feet'] + 1)
    
    # Encode primary use
    le = LabelEncoder()
    df['primary_use_encoded'] = le.fit_transform(df['primary_use'].astype(str))
    
    print(f"  Created 3 building features")
    
    return df, le


def create_lag_features(
    df: pd.DataFrame,
    lag_hours: List[int] = [1, 2, 3, 24],
    rolling_window: int = 24
) -> pd.DataFrame:
    """
    Create lag and rolling features for time series.
    
    Args:
        df: DataFrame sorted by building and timestamp
        lag_hours: List of lag periods in hours
        rolling_window: Window size for rolling statistics
    
    Returns:
        DataFrame with lag features added
    """
    print("Creating lag features...")
    
    # Sort by building and time
    df = df.sort_values(['building_id', 'timestamp'])
    
    # Lag features
    for lag in lag_hours:
        df[f'meter_lag_{lag}h'] = df.groupby('building_id')['meter_reading'].shift(lag)
    
    # Rolling statistics
    df['meter_rolling_mean_24h'] = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    df['meter_rolling_std_24h'] = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.shift(1).rolling(rolling_window, min_periods=1).std()
    )
    
    # Drop rows with NaN lag features
    initial_len = len(df)
    df = df.dropna(subset=[f'meter_lag_{lag_hours[0]}h', f'meter_lag_{lag_hours[-1]}h'])
    print(f"  Created {len(lag_hours) + 2} lag features")
    print(f"  Dropped {initial_len - len(df):,} rows with NaN lags")
    
    return df


def get_feature_lists() -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Get lists of feature names by category.
    
    Returns:
        Tuple of (temporal, weather, building, lag, all) feature lists
    """
    TEMPORAL_FEATURES = [
        'hour', 'dayofweek', 'month', 'is_weekend', 'is_business_hours',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    
    WEATHER_FEATURES = [
        'air_temperature', 'dew_temperature', 'cloud_coverage',
        'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed',
        'temp_deviation', 'comfort_index'
    ]
    
    BUILDING_FEATURES = [
        'square_feet', 'year_built', 'floor_count', 'building_age',
        'primary_use_encoded', 'site_id'
    ]
    
    LAG_FEATURES = [
        'meter_lag_1h', 'meter_lag_2h', 'meter_lag_3h', 'meter_lag_24h',
        'meter_rolling_mean_24h', 'meter_rolling_std_24h'
    ]
    
    ALL_FEATURES = TEMPORAL_FEATURES + WEATHER_FEATURES + BUILDING_FEATURES + LAG_FEATURES
    
    return TEMPORAL_FEATURES, WEATHER_FEATURES, BUILDING_FEATURES, LAG_FEATURES, ALL_FEATURES


def engineer_all_features(
    df: pd.DataFrame,
    business_hour_start: int = 9,
    business_hour_end: int = 18,
    lag_hours: List[int] = [1, 2, 3, 24],
    rolling_window: int = 24
) -> Tuple[pd.DataFrame, LabelEncoder, List[str]]:
    """
    Run complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        business_hour_start: Start of business hours
        business_hour_end: End of business hours
        lag_hours: Lag periods for lag features
        rolling_window: Window for rolling statistics
    
    Returns:
        Tuple of (featured DataFrame, label encoder, feature list)
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Create all features
    df = create_temporal_features(df)
    df = create_baseline_features(df, business_hour_start, business_hour_end)
    df = create_weather_features(df)
    df, label_encoder = create_building_features(df)
    df = create_lag_features(df, lag_hours, rolling_window)
    
    # Get feature list
    _, _, _, _, ALL_FEATURES = get_feature_lists()
    
    print(f"\nTotal features: {len(ALL_FEATURES)}")
    print(f"Final shape: {df.shape}")
    
    return df, label_encoder, ALL_FEATURES


if __name__ == "__main__":
    # Test with sample data
    print("Feature engineering module loaded successfully")
    _, _, _, _, features = get_feature_lists()
    print(f"Feature count: {len(features)}")
