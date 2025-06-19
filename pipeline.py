import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
import argparse

# Add src directory to path
sys.path.append('src')

from config import Config
from database import DatabaseManager
from data_extraction import EntsoeDataExtractor, get_date_ranges_for_extraction
from data_transformation import DataTransformer, process_raw_data

class EnergyForecastPipeline:
    """Main pipeline class for energy price forecasting"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_mlflow()
        
        # Initialize components
        self.db_manager = DatabaseManager(Config.DATABASE_URL)
        self.extractor = EntsoeDataExtractor(Config.ENTSOE_API_KEY, verify_ssl=False)
        self.transformer = DataTransformer()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking"""
        mlflow.set_experiment("german_energy_price_forecasting")
        self.logger.info("MLflow experiment setup completed")
    
    def run_data_extraction(self, start_date: datetime = None, end_date: datetime = None,
                          force_refresh: bool = True) -> bool:
        """
        Run the data extraction process
        
        Args:
            start_date: Start date for extraction (optional)
            end_date: End date for extraction (optional)
            force_refresh: If True, re-extract all data regardless of what's in DB
            
        Returns:
            True if extraction was successful, False otherwise
        """
        try:
            self.logger.info("Starting data extraction process")
            
            # Determine date range for extraction
            if start_date is None or end_date is None:
                if force_refresh:
                    latest_timestamp = None
                else:
                    latest_timestamp = self.db_manager.get_latest_timestamp()
                
                start_date, end_date = get_date_ranges_for_extraction(
                    latest_timestamp, lookback_days=30
                )

                if start_date >= end_date:
                    self.logger.warning(f"Start date {start_date} is not before end date {end_date}. No extraction needed.")
                    return False
            
            self.logger.info(f"Extracting data from {start_date} to {end_date}")
            
            # Extract data from ENTSO-E
            raw_data = self.extractor.extract_all_data(start_date, end_date)
            
            if raw_data.empty:
                self.logger.warning("No new data extracted")
                return False
            
            # Clean and transform data
            processed_data = process_raw_data(raw_data)
            
            # Prepare data for database storage
            # db_data = processed_data['train_data'].append(processed_data['val_data'], ignore_index=True)
            db_data = pd.concat([processed_data['train_data'], processed_data['val_data']], ignore_index=True)

            # Select only columns that exist in our database schema
            db_columns = [
                'timestamp', 'price', 'wind_forecast', 'solar_forecast', 'total_load_forecast',
                'hour_of_day', 'day_of_week', 'month', 'is_weekend',
                'price_lag_24h', 'price_lag_48h', 'price_lag_7d',
                'price_rolling_7d_mean', 'price_rolling_7d_std'
            ]
            
            # Only keep columns that exist in both the data and our schema
            available_columns = [col for col in db_columns if col in db_data.columns]
            db_data_filtered = db_data[available_columns]
            
            # Store in database
            self.db_manager.insert_or_update_data(db_data_filtered)
            
            self.logger.info(f"Successfully extracted and stored {len(db_data_filtered)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data extraction: {e}")
            return False
    
    # def run_model_training(self, retrain: bool = False) -> dict:
    #     """
    #     Run the model training process
        
    #     Args:
    #         retrain: If True, force retraining even if recent model exists
            
    #     Returns:
    #         Dictionary with training results and metrics
    #     """
    #     try:
    #         self.logger.info("Starting model training process")
            
    #         # Get data from database
    #         data = self.db_manager.get_data()
            
    #         if len(data) < 24 * 7:  # Need at least a week of data
    #             raise ValueError("Insufficient data for training. Need at least 7 days of hourly data.")
            
    #         # Process data for ML
    #         processed_data = process_raw_data(data)
            
    #         # Start MLflow run
    #         with mlflow.start_run():
    #             # Log data information
    #             mlflow.log_param("training_samples", len(processed_data['X_train']))
    #             mlflow.log_param("validation_samples", len(processed_data['X_val']))
    #             mlflow.log_param("n_features", len(processed_data['feature_info']['feature_names']))
    #             mlflow.log_param("data_start", processed_data['train_data']['timestamp'].min())
    #             mlflow.log_param("data_end", processed_data['val_data']['timestamp'].max())
                
    #             # Train LightGBM model
    #             model_results = self._train_lightgbm_model(processed_data)
                
    #             # Log model metrics
    #             for metric_name, metric_value in model_results['metrics'].items():
    #                 mlflow.log_metric(metric_name, metric_value)
                
    #             # Log model parameters
    #             for param_name, param_value in model_results['params'].items():
    #                 mlflow.log_param(param_name, param_value)

    #              # Log feature importance as artifact
    #             feature_importance_df = model_results['feature_importance']
            
                
    #             # Log and save model
    #             mlflow.lightgbm.log_model(
    #                 model_results['model'], 
    #                 "model",
    #                 input_example=processed_data['X_train'].head(5)
    #             )
                
    #             # Save model locally as well
    #             import joblib
    #             joblib.dump(model_results['model'], 'models/latest_model.pkl')
    #             joblib.dump(processed_data['feature_info'], 'models/feature_info.pkl')
                
    #             self.logger.info("Model training completed successfully")
    #             return model_results
                
    #     except Exception as e:
    #         self.logger.error(f"Error in model training: {e}")
    #         raise

    def run_model_training(self, retrain: bool = False) -> dict:
        """
        Run the model training process
        
        Args:
            retrain: If True, force retraining even if recent model exists
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.logger.info("Starting model training process")
            
            # Get data from database
            data = self.db_manager.get_data()
            
            if len(data) < 24 * 7:  # Need at least a week of data
                raise ValueError("Insufficient data for training. Need at least 7 days of hourly data.")
            
            # Process data for ML
            processed_data = process_raw_data(data)
            
            # Start MLflow run
            with mlflow.start_run():
                # Log data information
                mlflow.log_param("training_samples", len(processed_data['X_train']))
                mlflow.log_param("validation_samples", len(processed_data['X_val']))
                mlflow.log_param("n_features", len(processed_data['feature_info']['feature_names']))
                mlflow.log_param("data_start", processed_data['train_data']['timestamp'].min())
                mlflow.log_param("data_end", processed_data['val_data']['timestamp'].max())
                
                # Train LightGBM model
                model_results = self._train_lightgbm_model(processed_data)
                
                # Log model metrics
                for metric_name, metric_value in model_results['metrics'].items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model parameters
                for param_name, param_value in model_results['params'].items():
                    mlflow.log_param(param_name, param_value)
                
                # Log feature importance as artifact
                feature_importance_df = model_results['feature_importance']
                
                # Save feature importance plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                top_features = feature_importance_df.head(20)  # Top 20 features
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                # Save plot
                plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
                mlflow.log_artifact('feature_importance.png')
                plt.close()
                
                # Log feature importance as CSV
                feature_importance_df.to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
                
                # Log additional metrics for the top features
                top_5_features = feature_importance_df.head(5)
                for i, row in top_5_features.iterrows():
                    mlflow.log_metric(f"feature_importance_rank_{i+1}", row['importance'])
                    mlflow.log_param(f"top_feature_{i+1}", row['feature'])
                
                # Create and log model predictions plot
                train_data = processed_data['train_data'] 
                val_data = processed_data['val_data']
                
                plt.figure(figsize=(15, 8))
                
                # Plot actual vs predicted for validation set (last 7 days)
                recent_val = val_data.tail(24*7)  # Last 7 days
                val_predictions = model_results['predictions']['val'][-len(recent_val):]
                
                plt.subplot(2, 1, 1)
                plt.plot(recent_val['timestamp'], recent_val['price'], label='Actual', alpha=0.8)
                plt.plot(recent_val['timestamp'], val_predictions, label='Predicted', alpha=0.8)
                plt.title('Model Performance: Actual vs Predicted (Last 7 Days)')
                plt.xlabel('Time')
                plt.ylabel('Price (EUR/MWh)')
                plt.legend()
                plt.xticks(rotation=45)
                
                # Plot residuals
                plt.subplot(2, 1, 2)
                residuals = recent_val['price'].values - val_predictions
                plt.plot(recent_val['timestamp'], residuals, alpha=0.7)
                plt.title('Prediction Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual (EUR/MWh)')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
                mlflow.log_artifact('model_performance.png')
                plt.close()
                
                # Log and save model
                mlflow.lightgbm.log_model(
                    model_results['model'], 
                    "model",
                    input_example=processed_data['X_train'].head(5)
                )
                
                # Save model locally as well
                import joblib
                joblib.dump(model_results['model'], 'models/latest_model.pkl')
                joblib.dump(processed_data['feature_info'], 'models/feature_info.pkl')
                
                self.logger.info("Model training completed successfully")
                return model_results
                
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def _train_lightgbm_model(self, processed_data: dict) -> dict:
        """Train LightGBM model with the processed data"""
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.logger.info("Training LightGBM model...")
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'validation'],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mape': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_mape': np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100,
        }
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Model training metrics:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'params': params,
            'feature_importance': feature_importance,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            }
        }
    
    def generate_forecast(self, forecast_hours: int = 24) -> pd.DataFrame:
        """
        Generate price forecast for the next N hours
        
        Args:
            forecast_hours: Number of hours to forecast
            
        Returns:
            DataFrame with forecasted prices
        """
        try:
            self.logger.info(f"Generating {forecast_hours}-hour forecast")
            
            # Load trained model
            import joblib
            model = joblib.load('models/latest_model.pkl')
            feature_info = joblib.load('models/feature_info.pkl')
            
            # Get latest data for feature engineering
            latest_data = self.db_manager.get_data()
            
            if latest_data.empty:
                raise ValueError("No data available for forecasting")
            
            # Process data to get features
            processed_data = process_raw_data(latest_data)
            
            # Get the latest feature row as starting point
            latest_features = processed_data['X_train'].iloc[-1:].copy()
            
            # Generate forecast timestamps
            last_timestamp = latest_data['timestamp'].max()
            forecast_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=forecast_hours,
                freq='h',
                tz=Config.TIMEZONE
            )
            
            forecasts = []
            
            for i, forecast_time in enumerate(forecast_timestamps):
                # Update time-based features for this forecast hour
                features = latest_features.copy()
                
                # Update time features
                features['hour_of_day'] = forecast_time.hour
                features['day_of_week'] = forecast_time.dayofweek
                features['month'] = forecast_time.month
                features['is_weekend'] = int(forecast_time.dayofweek >= 5)
                
                # Update cyclical features
                features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
                features['month_sin'] = np.sin(2 * np.pi * forecast_time.month / 12)
                features['month_cos'] = np.cos(2 * np.pi * forecast_time.month / 12)
                
                # Make prediction
                prediction = model.predict(features, num_iteration=model.best_iteration)[0]
                
                forecasts.append({
                    'timestamp': forecast_time,
                    'forecast_price': prediction,
                    'forecast_horizon_hours': i + 1
                })
            
            forecast_df = pd.DataFrame(forecasts)
            
            self.logger.info(f"Generated forecasts for {len(forecast_df)} hours")
            return forecast_df
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            raise
    
    def run_full_pipeline(self, retrain_model: bool = False, forecast_hours: int = 24):
        """
        Run the complete pipeline: extraction -> training -> forecasting
        
        Args:
            retrain_model: Whether to retrain the model
            forecast_hours: Number of hours to forecast
        """
        try:
            self.logger.info("Starting full pipeline execution")
            
            # Step 1: Data extraction
            extraction_success = self.run_data_extraction()
            
            if not extraction_success:
                self.logger.error("Data extraction failed. Pipeline stopped.")
                return
            
            # Step 2: Model training (if needed)
            if retrain_model or not os.path.exists('models/latest_model.pkl'):
                self.run_model_training()
            else:
                self.logger.info("Using existing model (retrain_model=False)")
            
            # Step 3: Generate forecast
            forecast = self.generate_forecast(forecast_hours)
            
            # Save forecast
            forecast.to_csv(f'forecasts/forecast_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
            
            self.logger.info("Full pipeline execution completed successfully")
            self.logger.info(f"Next 24h average forecasted price: {forecast['forecast_price'].mean():.2f} EUR/MWh")
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline execution: {e}")
            raise

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='German Energy Price Forecasting Pipeline')
    parser.add_argument('--extract-only', action='store_true', help='Only run data extraction')
    parser.add_argument('--train-only', action='store_true', help='Only run model training')
    parser.add_argument('--forecast-only', action='store_true', help='Only generate forecast')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--forecast-hours', type=int, default=24, help='Hours to forecast')
    
    args = parser.parse_args()
    
    # Validate configuration
    Config.validate_config()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('forecasts', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Initialize pipeline
    pipeline = EnergyForecastPipeline()
    
    try:
        if args.extract_only:
            pipeline.run_data_extraction()
        elif args.train_only:
            pipeline.run_model_training(retrain=args.retrain)
        elif args.forecast_only:
            forecast = pipeline.generate_forecast(args.forecast_hours)
            print(forecast)
        else:
            # Run full pipeline
            pipeline.run_full_pipeline(
                retrain_model=args.retrain,
                forecast_hours=args.forecast_hours
            )
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()