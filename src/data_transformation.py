import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from config import Config

class DataTransformer:
    """Handles data transformation and feature engineering for energy price forecasting"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and outliers
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Starting data cleaning process")
            df_clean = df.copy()
            
            # Ensure timestamp is datetime and sorted
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Handle outliers in price data
            df_clean = self._handle_price_outliers(df_clean)
            
            # Ensure non-negative values for generation forecasts
            forecast_columns = ['wind_forecast', 'solar_forecast', 'total_load_forecast']
            for col in forecast_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].clip(lower=0)
            
            self.logger.info(f"Data cleaning completed. {len(df_clean)} records processed")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""

        # Set timestamp as index for time-based interpolation if not already
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # For price data: use time-based interpolation
        if 'price' in df.columns:
            df['price'] = df['price'].interpolate(method='time', limit=6)
            df['price'] = df['price'].ffill()
            df['price'] = df['price'].bfill()

        # For forecast data: forward-fill then backward-fill
        forecast_columns = ['wind_forecast', 'solar_forecast', 'total_load_forecast']
        for col in forecast_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)

        # Reset index to keep 'timestamp' as a column
        df = df.reset_index()

        return df
    
    def _handle_price_outliers(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Handle outliers in price data using IQR method"""
        
        if price_col not in df.columns:
            return df
        
        # Calculate IQR
        Q1 = df[price_col].quantile(0.25)
        Q3 = df[price_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (using 3*IQR for energy markets which can be volatile)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Cap outliers instead of removing them
        outlier_count = ((df[price_col] < lower_bound) | (df[price_col] > upper_bound)).sum()
        
        if outlier_count > 0:
            self.logger.info(f"Capping {outlier_count} price outliers")
            df[price_col] = df[price_col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for machine learning
        
        Args:
            df: Cleaned data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info("Starting feature engineering")
            df_features = df.copy()
            
            # Ensure timestamp is in the correct timezone
            if df_features['timestamp'].dt.tz is None:
                df_features['timestamp'] = df_features['timestamp'].dt.tz_localize(Config.TIMEZONE)
            else:
                df_features['timestamp'] = df_features['timestamp'].dt.tz_convert(Config.TIMEZONE)
            
            # Create time-based features
            df_features = self._create_time_features(df_features)
            
            # Create lagged features
            df_features = self._create_lagged_features(df_features)
            
            # Create rolling features
            df_features = self._create_rolling_features(df_features)
            
            # Create ratio features
            df_features = self._create_ratio_features(df_features)
            
            # Drop rows with NaN values created by lagging/rolling
            initial_rows = len(df_features)
            df_features = df_features.dropna()
            dropped_rows = initial_rows - len(df_features)
            
            if dropped_rows > 0:
                self.logger.info(f"Dropped {dropped_rows} rows due to NaN values in engineered features")
            
            self.logger.info(f"Feature engineering completed. {len(df_features)} records with {len(df_features.columns)} features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            raise
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Basic time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Peak hours (typically 7-22 for electricity markets)
        df['is_peak_hour'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 22)).astype(int)
        
        # Season indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Cyclical encoding for hour and month (preserves continuity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series modeling"""
        
        # Price lags (24h, 48h, 7 days)
        if 'price' in df.columns:
            df['price_lag_24h'] = df['price'].shift(24)  # 24 hours ago
            df['price_lag_48h'] = df['price'].shift(48)  # 48 hours ago
            df['price_lag_7d'] = df['price'].shift(24 * 7)  # 7 days ago (same hour)
        
        # Generation forecast lags
        for col in ['wind_forecast', 'solar_forecast']:
            if col in df.columns:
                df[f'{col}_lag_24h'] = df[col].shift(24)
        
        # Load forecast lag
        if 'total_load_forecast' in df.columns:
            df['load_lag_24h'] = df['total_load_forecast'].shift(24)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features"""
        
        # Price rolling features
        if 'price' in df.columns:
            # 7-day rolling statistics
            df['price_rolling_7d_mean'] = df['price'].rolling(window=24*7, min_periods=24).mean()
            df['price_rolling_7d_std'] = df['price'].rolling(window=24*7, min_periods=24).std()
            df['price_rolling_7d_min'] = df['price'].rolling(window=24*7, min_periods=24).min()
            df['price_rolling_7d_max'] = df['price'].rolling(window=24*7, min_periods=24).max()
            
            # 24-hour rolling statistics
            df['price_rolling_24h_mean'] = df['price'].rolling(window=24, min_periods=12).mean()
            df['price_rolling_24h_std'] = df['price'].rolling(window=24, min_periods=12).std()
            
            # Price position within recent range
            df['price_pct_of_7d_range'] = ((df['price'] - df['price_rolling_7d_min']) / 
                                         (df['price_rolling_7d_max'] - df['price_rolling_7d_min']))
        
        # Generation rolling features
        for col in ['wind_forecast', 'solar_forecast']:
            if col in df.columns:
                df[f'{col}_rolling_24h_mean'] = df[col].rolling(window=24, min_periods=12).mean()
                df[f'{col}_rolling_7d_mean'] = df[col].rolling(window=24*7, min_periods=24).mean()
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and interaction features"""
        
        # Renewable penetration ratios
        if all(col in df.columns for col in ['wind_forecast', 'solar_forecast', 'total_load_forecast']):
            # Total renewable generation
            df['total_renewable_forecast'] = df['wind_forecast'] + df['solar_forecast']
            
            # Renewable penetration rate
            df['renewable_penetration'] = (df['total_renewable_forecast'] / 
                                         (df['total_load_forecast'] + 1e-6))  # Add small value to avoid division by zero
            
            # Individual renewable shares
            df['wind_share'] = df['wind_forecast'] / (df['total_load_forecast'] + 1e-6)
            df['solar_share'] = df['solar_forecast'] / (df['total_load_forecast'] + 1e-6)
        
        # Price deviation from recent average
        if all(col in df.columns for col in ['price', 'price_rolling_7d_mean']):
            df['price_deviation_7d'] = df['price'] - df['price_rolling_7d_mean']
            df['price_deviation_7d_pct'] = (df['price_deviation_7d'] / 
                                          (df['price_rolling_7d_mean'] + 1e-6))
        
        return df
    
    def prepare_model_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for machine learning model training and inference
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary containing train/validation splits and feature information
        """
        try:
            self.logger.info("Preparing data for machine learning")
            
            # Define feature columns (exclude timestamp and target)
            feature_columns = [col for col in df.columns 
                             if col not in ['timestamp', 'price'] and not col.startswith('price_lag')]
            
            # Add lagged price features to features (these are inputs, not targets)
            lagged_price_features = [col for col in df.columns if col.startswith('price_lag')]
            feature_columns.extend(lagged_price_features)
            
            # Remove any remaining NaN rows
            df_clean = df.dropna()
            
            # Sort by timestamp
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
            
            # Create train/validation split (80/20, but maintain time order)
            split_idx = int(len(df_clean) * 0.8)
            
            train_data = df_clean.iloc[:split_idx].copy()
            val_data = df_clean.iloc[split_idx:].copy()
            
            # Prepare feature matrices and target vectors
            X_train = train_data[feature_columns]
            y_train = train_data['price']
            X_val = val_data[feature_columns]
            y_val = val_data['price']
            
            # Feature importance tracking
            feature_info = {
                'feature_names': feature_columns,
                'n_features': len(feature_columns),
                'time_features': [col for col in feature_columns if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'peak'])],
                'lag_features': [col for col in feature_columns if 'lag' in col],
                'rolling_features': [col for col in feature_columns if 'rolling' in col],
                'ratio_features': [col for col in feature_columns if any(x in col for x in ['share', 'penetration', 'deviation', 'pct'])]
            }
            
            result = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'train_data': train_data,
                'val_data': val_data,
                'feature_info': feature_info
            }
            
            self.logger.info(f"Data preparation completed:")
            self.logger.info(f"  Training samples: {len(X_train)}")
            self.logger.info(f"  Validation samples: {len(X_val)}")
            self.logger.info(f"  Features: {len(feature_columns)}")
            self.logger.info(f"  Time range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {e}")
            raise
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a summary of all features"""
        
        summary_data = []
        
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            col_data = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_pct': df[col].isnull().mean() * 100,
                'mean': df[col].mean() if df[col].dtype in ['float64', 'int64'] else None,
                'std': df[col].std() if df[col].dtype in ['float64', 'int64'] else None,
                'min': df[col].min() if df[col].dtype in ['float64', 'int64'] else None,
                'max': df[col].max() if df[col].dtype in ['float64', 'int64'] else None,
            }
            summary_data.append(col_data)
        
        return pd.DataFrame(summary_data)

# Utility function for full data processing pipeline
def process_raw_data(raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Complete data processing pipeline from raw data to ML-ready data
    
    Args:
        raw_df: Raw data from ENTSO-E API
        
    Returns:
        Dictionary with processed data and model-ready splits
    """
    transformer = DataTransformer()
    
    # Step 1: Clean data
    clean_df = transformer.clean_data(raw_df)
    
    # Step 2: Engineer features
    features_df = transformer.engineer_features(clean_df)
    
    # Step 3: Prepare for ML
    ml_data = transformer.prepare_model_data(features_df)
    
    return ml_data