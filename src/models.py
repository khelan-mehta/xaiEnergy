"""
Machine Learning Models Module
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split data temporally (NOT randomly).
    
    Args:
        df: DataFrame with 'timestamp' column
        train_ratio: Proportion for training
    
    Returns:
        Tuple of (train_df, test_df, split_date)
    """
    print("\n" + "=" * 60)
    print("TEMPORAL TRAIN-TEST SPLIT")
    print("=" * 60)
    
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * train_ratio)
    split_date = df.iloc[split_idx]['timestamp']
    
    train_df = df[df['timestamp'] < split_date].copy()
    test_df = df[df['timestamp'] >= split_date].copy()
    
    print(f"Split date: {split_date}")
    print(f"Train: {len(train_df):,} rows ({train_ratio*100:.0f}%)")
    print(f"Test: {len(test_df):,} rows ({(1-train_ratio)*100:.0f}%)")
    
    return train_df, test_df, str(split_date)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare feature matrices and targets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
    
    Returns:
        Tuple of (X_train, X_test, y_train_meter, y_test_meter, y_train_gap, y_test_gap, scaler)
    """
    print("\nPreparing feature matrices...")
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    y_train_meter = train_df['meter_reading'].values
    y_test_meter = test_df['meter_reading'].values
    
    y_train_gap = train_df['performance_gap'].values
    y_test_gap = test_df['performance_gap'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, \
           y_train_meter, y_test_meter, y_train_gap, y_test_gap, scaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    # Handle edge cases
    y_pred = np.clip(y_pred, 0, None)  # Energy can't be negative
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[LinearRegression, np.ndarray, Dict[str, float]]:
    """
    Train Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        Tuple of (model, predictions, metrics)
    """
    print("\n" + "-" * 40)
    print("Training Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    params: Optional[Dict] = None,
    num_rounds: int = 500,
    early_stopping: int = 50,
    sample_size: Optional[int] = None
) -> Tuple[lgb.Booster, np.ndarray, Dict[str, float]]:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        params: LightGBM parameters
        num_rounds: Number of boosting rounds
        early_stopping: Early stopping rounds
        sample_size: Sample size for training (None for full data)
    
    Returns:
        Tuple of (model, predictions, metrics)
    """
    print("\n" + "-" * 40)
    print("Training LightGBM...")
    
    # Default parameters
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
    
    # Sample if specified
    if sample_size and sample_size < len(X_train):
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[idx]
        y_train_sample = y_train[idx]
        print(f"  Training on sample: {sample_size:,} rows")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Create datasets
    train_data = lgb.Dataset(X_train_sample, label=y_train_sample, feature_name=feature_names)
    valid_size = min(100000, len(X_test))
    valid_data = lgb.Dataset(X_test[:valid_size], label=y_test[:valid_size], feature_name=feature_names)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predict
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    print(f"\n  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input.
    
    Args:
        X: Feature matrix
        y: Target array
        seq_length: Sequence length
    
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(seq_length: int, n_features: int) -> Sequential:
    """
    Build LSTM model architecture.
    
    Args:
        seq_length: Input sequence length
        n_features: Number of features
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    seq_length: int = 24,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
    sample_size: Optional[int] = None
) -> Tuple[Sequential, np.ndarray, Dict[str, float], dict]:
    """
    Train LSTM model.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target
        X_test_scaled: Scaled test features
        y_test: Test target
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs
        batch_size: Batch size
        patience: Early stopping patience
        sample_size: Sample size for training
    
    Returns:
        Tuple of (model, predictions, metrics, history)
    """
    print("\n" + "-" * 40)
    print("Training LSTM...")
    
    # Sample if specified
    if sample_size and sample_size < len(X_train_scaled) - seq_length:
        train_size = sample_size
    else:
        train_size = len(X_train_scaled) - seq_length
    
    # Create sequences
    print(f"  Creating training sequences (size: {train_size:,})...")
    X_lstm_train, y_lstm_train = create_sequences(
        X_train_scaled[:train_size + seq_length],
        y_train[:train_size + seq_length],
        seq_length
    )
    
    test_size = min(50000, len(X_test_scaled) - seq_length)
    print(f"  Creating test sequences (size: {test_size:,})...")
    X_lstm_test, y_lstm_test = create_sequences(
        X_test_scaled[:test_size + seq_length],
        y_test[:test_size + seq_length],
        seq_length
    )
    
    print(f"  Train shape: {X_lstm_train.shape}")
    print(f"  Test shape: {X_lstm_test.shape}")
    
    # Build model
    n_features = X_train_scaled.shape[1]
    model = build_lstm_model(seq_length, n_features)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    print("  Training...")
    history = model.fit(
        X_lstm_train, y_lstm_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Predict
    predictions = model.predict(X_lstm_test).flatten()
    metrics = calculate_metrics(y_lstm_test, predictions)
    
    print(f"\n  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return model, predictions, metrics, history.history, y_lstm_test


def train_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    lgb_params: Optional[Dict] = None,
    sample_size: int = 500000,
    lstm_sample_size: int = 200000
) -> Dict:
    """
    Train all models and return results.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        lgb_params: LightGBM parameters
        sample_size: Sample size for gradient boosting
        lstm_sample_size: Sample size for LSTM
    
    Returns:
        Dictionary with all models, predictions, and metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    # Prepare features
    X_train, X_test, X_train_scaled, X_test_scaled, \
    y_train_meter, y_test_meter, y_train_gap, y_test_gap, scaler = \
        prepare_features(train_df, test_df, feature_cols)
    
    results = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'y_test_meter': y_test_meter,
        'y_test_gap': y_test_gap
    }
    
    # Sample for faster training
    if sample_size and sample_size < len(X_train):
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[sample_idx]
        y_train_meter_sample = y_train_meter[sample_idx]
        y_train_gap_sample = y_train_gap[sample_idx]
    else:
        X_train_sample = X_train
        y_train_meter_sample = y_train_meter
        y_train_gap_sample = y_train_gap
    
    # Linear Regression
    lr_model, lr_pred, lr_metrics = train_linear_regression(
        X_train_sample, y_train_meter_sample, X_test, y_test_meter
    )
    results['lr'] = {'model': lr_model, 'predictions': lr_pred, 'metrics': lr_metrics}
    
    # LightGBM - Energy
    lgb_model, lgb_pred, lgb_metrics = train_lightgbm(
        X_train_sample, y_train_meter_sample, X_test, y_test_meter,
        feature_cols, lgb_params, sample_size=None
    )
    results['lgb_energy'] = {'model': lgb_model, 'predictions': lgb_pred, 'metrics': lgb_metrics}
    
    # LightGBM - Gap
    lgb_gap_model, lgb_gap_pred, lgb_gap_metrics = train_lightgbm(
        X_train_sample, y_train_gap_sample, X_test, y_test_gap,
        feature_cols, lgb_params, sample_size=None
    )
    results['lgb_gap'] = {'model': lgb_gap_model, 'predictions': lgb_gap_pred, 'metrics': lgb_gap_metrics}
    
    # LSTM
    lstm_model, lstm_pred, lstm_metrics, lstm_history, y_lstm_test = train_lstm(
        X_train_scaled, y_train_meter,
        X_test_scaled, y_test_meter,
        sample_size=lstm_sample_size
    )
    results['lstm'] = {
        'model': lstm_model, 
        'predictions': lstm_pred, 
        'metrics': lstm_metrics,
        'history': lstm_history,
        'y_test': y_lstm_test
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 56)
    print(f"{'Linear Regression':<20} {lr_metrics['RMSE']:<12.2f} {lr_metrics['MAE']:<12.2f} {lr_metrics['R2']:<12.4f}")
    print(f"{'LightGBM (Energy)':<20} {lgb_metrics['RMSE']:<12.2f} {lgb_metrics['MAE']:<12.2f} {lgb_metrics['R2']:<12.4f}")
    print(f"{'LightGBM (Gap)':<20} {lgb_gap_metrics['RMSE']:<12.2f} {lgb_gap_metrics['MAE']:<12.2f} {lgb_gap_metrics['R2']:<12.4f}")
    print(f"{'LSTM':<20} {lstm_metrics['RMSE']:<12.2f} {lstm_metrics['MAE']:<12.2f} {lstm_metrics['R2']:<12.4f}")
    
    return results


if __name__ == "__main__":
    print("Models module loaded successfully")
