# src/data_extraction.py
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Union
import time
import warnings
from config import Config


class EntsoeDataExtractor:
    """
    Extracts data from ENTSO-E Transparency Platform with robust error handling
    
    Based on entsoe-py library patterns:
    - query_day_ahead_prices() returns pandas Series
    - query_wind_and_solar_forecast() returns pandas DataFrame  
    - query_load_forecast() returns pandas DataFrame
    """
    
    def __init__(self, api_key: str, verify_ssl: bool = True):
        """
        Initialize the ENTSO-E data extractor
        
        Args:
            api_key: ENTSO-E API key
            verify_ssl: Whether to verify SSL certificates (default: True for security)
        """
        self.client = EntsoePandasClient(api_key=api_key)
        self.verify_ssl = verify_ssl
        
        if not verify_ssl:
            self._configure_ssl_bypass()
        
        self.setup_logging()
    
    def _configure_ssl_bypass(self):
        """Configure SSL bypass for development environments only"""
        try:
            import urllib3
            import requests
            
            warnings.warn(
                "SSL verification disabled. This should only be used in development!",
                UserWarning
            )
            
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            session = requests.Session()
            session.verify = False
            self.client.session = session
            
        except Exception as e:
            self.logger.warning(f"Could not configure SSL bypass: {e}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_timezone_aware(self, dt: datetime, tz) -> pd.Timestamp:
        """Ensure datetime has proper timezone handling"""
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts.tz_localize(tz)
        else:
            return ts.tz_convert(tz)
    
    def _handle_dst_transitions(self, start_date: datetime, end_date: datetime, country_code: str, 
                               fetch_function) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Handle DST transitions by splitting requests around problematic periods
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            country_code: ENTSO-E domain code
            fetch_function: Function to fetch data
            
        Returns:
            Combined data from multiple requests
        """
        try:
            # First try the full range
            return fetch_function(country_code, start_date, end_date)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a DST-related error
            if any(dst_indicator in error_str for dst_indicator in 
                   ["length mismatch", "daylight saving", "23 elements", "25 elements"]):
                
                self.logger.warning(f"DST transition detected in range {start_date} to {end_date}, splitting request")
                
                # Split the request into daily chunks to isolate DST issues
                current_date = start_date
                all_data = []
                
                while current_date < end_date:
                    next_date = min(current_date + timedelta(days=1), end_date)
                    
                    try:
                        daily_data = fetch_function(country_code, current_date, next_date)
                        if daily_data is not None and not daily_data.empty:
                            all_data.append(daily_data)
                    except Exception as daily_e:
                        self.logger.warning(f"Skipping {current_date.date()} due to error: {daily_e}")
                    
                    current_date = next_date
                
                # Combine all daily data
                if all_data:
                    if isinstance(all_data[0], pd.Series):
                        return pd.concat(all_data)
                    else:  # DataFrame
                        return pd.concat(all_data, ignore_index=True)
                else:
                    self.logger.error("No data retrieved after splitting by days")
                    return None
            else:
                # Re-raise non-DST errors
                raise e
    
    def get_day_ahead_prices(self, start_date: datetime, end_date: datetime, 
                            country_code: str = None) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices from ENTSO-E
        Returns: pandas Series with datetime index and price values
        """
        if country_code is None:
            country_code = Config.COUNTRY_DOMAIN
            
        try:
            self.logger.info(f"Fetching day-ahead prices for {country_code} from {start_date} to {end_date}")
            
            start_ts = self._ensure_timezone_aware(start_date, Config.TIMEZONE)
            end_ts = self._ensure_timezone_aware(end_date, Config.TIMEZONE)
            
            def price_fetch_function(cc, start, end):
                return self._fetch_with_retry(
                    lambda: self.client.query_day_ahead_prices(cc, start=start, end=end)
                )
            
            prices = self._handle_dst_transitions(start_ts, end_ts, country_code, price_fetch_function)
            
            if prices is None or prices.empty:
                self.logger.warning("No price data received")
                return pd.DataFrame()
            
            # Convert Series to DataFrame
            df = prices.to_frame(name='price').reset_index()
            df.columns = ['timestamp', 'price']
            
            # Ensure timezone consistency
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(Config.TIMEZONE)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(Config.TIMEZONE)
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} price records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching day-ahead prices: {e}")
            return pd.DataFrame()
    
    def get_wind_solar_forecasts(self, start_date: datetime, end_date: datetime,
                               country_code: str = None) -> pd.DataFrame:
        """
        Fetch wind and solar generation forecasts from ENTSO-E
        Returns: pandas DataFrame with columns for different generation types
        """
        if country_code is None:
            country_code = Config.COUNTRY_DOMAIN
            
        try:
            self.logger.info(f"Fetching generation forecasts for {country_code} from {start_date} to {end_date}")
            
            start_ts = self._ensure_timezone_aware(start_date, Config.TIMEZONE)
            end_ts = self._ensure_timezone_aware(end_date, Config.TIMEZONE)
            
            def forecast_fetch_function(cc, start, end):
                return self._fetch_with_retry(
                    lambda: self.client.query_wind_and_solar_forecast(cc, start=start, end=end)
                )
            
            forecasts = self._handle_dst_transitions(start_ts, end_ts, country_code, forecast_fetch_function)
            
            if forecasts is None or forecasts.empty:
                self.logger.warning("No generation forecast data received")
                return pd.DataFrame()
            
            # Handle DataFrame result
            df = forecasts.reset_index()
            
            # Ensure we have a timestamp column
            if 'index' in df.columns:
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            elif df.index.name is not None:
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            
            # Ensure timezone consistency  
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(Config.TIMEZONE)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(Config.TIMEZONE)
            
            # Standardize column names based on entsoe-py documentation patterns
            column_mapping = {
                'Solar': 'solar_forecast',
                'Wind Onshore': 'wind_onshore_forecast',
                'Wind Offshore': 'wind_offshore_forecast', 
                'Wind': 'wind_forecast'  # Some countries have combined wind data
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Combine wind forecasts if separate onshore/offshore data exists
            if 'wind_onshore_forecast' in df.columns and 'wind_offshore_forecast' in df.columns:
                df['wind_forecast'] = (
                    df['wind_onshore_forecast'].fillna(0) + 
                    df['wind_offshore_forecast'].fillna(0)
                )
            elif 'wind_onshore_forecast' in df.columns and 'wind_forecast' not in df.columns:
                df['wind_forecast'] = df['wind_onshore_forecast']
            elif 'wind_offshore_forecast' in df.columns and 'wind_forecast' not in df.columns:
                df['wind_forecast'] = df['wind_offshore_forecast']
            
            # Ensure we have the expected columns, fill with zeros if missing
            expected_columns = ['wind_forecast', 'solar_forecast']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    self.logger.warning(f"Column {col} not found in forecast data, filled with zeros")
            
            # Select only relevant columns
            result_columns = ['timestamp'] + expected_columns
            available_columns = [col for col in result_columns if col in df.columns]
            df = df[available_columns]
            
            # Fill any remaining NaN values
            for col in expected_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} forecast records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching generation forecasts: {e}")
            return pd.DataFrame()
    
    def get_load_forecast(self, start_date: datetime, end_date: datetime,
                         country_code: str = None) -> pd.DataFrame:
        """
        Fetch total load forecast from ENTSO-E
        Returns: pandas DataFrame with timestamp and load forecast columns
        """
        if country_code is None:
            country_code = Config.COUNTRY_DOMAIN
            
        try:
            self.logger.info(f"Fetching load forecast for {country_code} from {start_date} to {end_date}")
            
            start_ts = self._ensure_timezone_aware(start_date, Config.TIMEZONE)
            end_ts = self._ensure_timezone_aware(end_date, Config.TIMEZONE)
            
            def load_fetch_function(cc, start, end):
                return self._fetch_with_retry(
                    lambda: self.client.query_load_forecast(cc, start=start, end=end)
                )
            
            load_forecast = self._handle_dst_transitions(start_ts, end_ts, country_code, load_fetch_function)
            
            if load_forecast is None or load_forecast.empty:
                self.logger.warning("No load forecast data received")
                return pd.DataFrame()
            
            # Handle DataFrame result (load_forecast returns DataFrame)
            df = load_forecast.reset_index()
            
            # Ensure we have a timestamp column
            if 'index' in df.columns:
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            elif df.index.name is not None:
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            
            # Ensure timezone consistency
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(Config.TIMEZONE)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(Config.TIMEZONE)
            
            # Rename load column to standardized name
            load_columns = [col for col in df.columns if col != 'timestamp']
            if load_columns:
                # Take the first non-timestamp column as the load forecast
                df.rename(columns={load_columns[0]: 'total_load_forecast'}, inplace=True)
            
            # Fill any NaN values
            if 'total_load_forecast' in df.columns:
                df['total_load_forecast'] = df['total_load_forecast'].ffill().fillna(0)
            
            # Select relevant columns
            result_columns = ['timestamp', 'total_load_forecast']
            available_columns = [col for col in result_columns if col in df.columns]
            df = df[available_columns]
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} load forecast records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching load forecast: {e}")
            return pd.DataFrame()
    
    def _fetch_with_retry(self, fetch_function, max_retries: int = 3, 
                         base_delay: int = 5) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Fetch data with exponential backoff retry logic
        """
        for attempt in range(max_retries):
            try:
                data = fetch_function()
                return data
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle specific ENTSO-E errors
                if any(auth_error in error_str for auth_error in ["401", "unauthorized", "forbidden"]):
                    self.logger.error(
                        "Authentication failed. Please check your ENTSO-E API key. "
                        "Note: New API keys require email confirmation before activation."
                    )
                    return None  # Don't retry auth errors
                
                elif any(rate_error in error_str for rate_error in ["429", "too many requests"]):
                    rate_limit_delay = 60 + (attempt * 30)
                    self.logger.warning(f"Rate limit hit. Waiting {rate_limit_delay} seconds...")
                    time.sleep(rate_limit_delay)
                    continue
                
                elif "400" in error_str:
                    self.logger.error(f"Bad request - check parameters: {e}")
                    return None
                
                # Let DST errors bubble up to be handled by _handle_dst_transitions
                elif any(dst_error in error_str for dst_error in 
                        ["length mismatch", "daylight saving", "23 elements", "25 elements"]):
                    raise e
                
                else:
                    self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {max_retries} attempts failed for API call")
                    return None
        
        return None
    
    def extract_all_data(self, start_date: datetime, end_date: datetime,
                        country_code: str = None) -> pd.DataFrame:
        """
        Extract all required data (prices, forecasts) and merge into single DataFrame
        """
        if country_code is None:
            country_code = Config.COUNTRY_DOMAIN
            
        try:
            self.logger.info(f"Starting full data extraction for {country_code} from {start_date} to {end_date}")
            
            # Extract all data sources
            prices_df = self.get_day_ahead_prices(start_date, end_date, country_code)
            forecasts_df = self.get_wind_solar_forecasts(start_date, end_date, country_code)
            load_df = self.get_load_forecast(start_date, end_date, country_code)
            
            # Validate that we have price data (minimum requirement)
            if prices_df.empty:
                self.logger.error("No price data available - cannot proceed with merge")
                return pd.DataFrame()
            
            # Start with prices as the base
            merged_df = prices_df.copy()
            
            # Merge forecasts data
            if not forecasts_df.empty:
                merged_df = pd.merge(merged_df, forecasts_df, on='timestamp', how='left')
                self.logger.info("Successfully merged generation forecasts")
            else:
                # Add empty forecast columns if no data available
                merged_df['wind_forecast'] = 0.0
                merged_df['solar_forecast'] = 0.0
                self.logger.warning("No generation forecast data - using zero values")
            
            # Merge load forecast
            if not load_df.empty:
                merged_df = pd.merge(merged_df, load_df, on='timestamp', how='left')
                self.logger.info("Successfully merged load forecasts")
            else:
                merged_df['total_load_forecast'] = 0.0
                self.logger.warning("No load forecast data - using zero values")
            
            # Final data cleaning
            forecast_columns = ['wind_forecast', 'solar_forecast', 'total_load_forecast']
            for col in forecast_columns:
                if col in merged_df.columns:
                    # Fill any remaining NaN values with forward fill, then zero
                    merged_df[col] = merged_df[col].ffill().fillna(0)
            
            # Validate final dataset
            if merged_df.isnull().any().any():
                null_cols = merged_df.columns[merged_df.isnull().any()].tolist()
                self.logger.warning(f"Final dataset still contains NaN values in columns: {null_cols}")
            
            # Set timestamp as index for time series operations (required by transformation step)
            # merged_df.set_index('timestamp', inplace=True)
            # merged_df.sort_index(inplace=True)
            
            self.logger.info(f"Successfully merged all data: {len(merged_df)} records")
            self.logger.info(f"Columns: {list(merged_df.columns)}")
            self.logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error in full data extraction: {e}")
            return pd.DataFrame()
        
def get_date_ranges_for_extraction(latest_db_timestamp: Optional[datetime] = None,
                                 lookback_days: int = 30) -> Tuple[datetime, datetime]:
    """
    Determine the date range for data extraction, avoiding DST transition dates
    """
    timezone = Config.TIMEZONE

    # End date: tomorrow at midnight (to capture today's full data)
    dt_end = (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if dt_end.tzinfo is None:
        end_date = timezone.localize(dt_end)
    else:
        end_date = dt_end.astimezone(timezone)

    if latest_db_timestamp is None:
        # Initial load: start from lookback_days ago  
        dt_start = (datetime.now() - timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        if dt_start.tzinfo is None:
            start_date = timezone.localize(dt_start)
        else:
            start_date = dt_start.astimezone(timezone)
        logging.info(f"Initial data load: extracting {lookback_days} days of historical data")
    else:
        # Incremental load: start from the day after latest data
        if latest_db_timestamp.tzinfo is None:
            latest_db_timestamp = timezone.localize(latest_db_timestamp)
        dt_start = (latest_db_timestamp + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if dt_start.tzinfo is None:
            start_date = timezone.localize(dt_start)
        else:
            start_date = dt_start.astimezone(timezone)
        logging.info(f"Incremental data load: extracting from {start_date}")

    return start_date, end_date


# def get_date_ranges_for_extraction(latest_db_timestamp: Optional[datetime] = None,
#                                  lookback_days: int = 30) -> Tuple[datetime, datetime]:
#     """
#     Determine the date range for data extraction, avoiding DST transition dates
#     """
#     timezone = Config.TIMEZONE
    
#     # End date: tomorrow at midnight (to capture today's full data)
#     end_date = timezone.localize(
#         (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
#     )
    
#     if latest_db_timestamp is None:
#         # Initial load: start from lookback_days ago  
#         start_date = timezone.localize(
#             (datetime.now() - timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
#         )
#         logging.info(f"Initial data load: extracting {lookback_days} days of historical data")
#     else:
#         # Incremental load: start from the day after latest data
#         if latest_db_timestamp.tzinfo is None:
#             latest_db_timestamp = timezone.localize(latest_db_timestamp)
        
#         start_date = timezone.localize(
#             (latest_db_timestamp + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
#         )
#         logging.info(f"Incremental data load: extracting from {start_date}")
    
#     return start_date, end_date

# Example usage and testing
if __name__ == "__main__":
    from config import Config
    
    # Initialize extractor 
    extractor = EntsoeDataExtractor(
        api_key=Config.ENTSOE_API_KEY,
        verify_ssl=False  
    )
    
    # Test with a shorter, safer date range (avoid DST transitions)
    timezone = Config.TIMEZONE
    end_date = timezone.localize(datetime(2025, 6, 17, 23, 0, 0))  # Yesterday
    start_date = timezone.localize(datetime(2025, 6, 15, 0, 0, 0))   # 2 days ago
    
    print(f"Testing data extraction from {start_date} to {end_date}")
    print("Note: Using recent dates to avoid DST transition issues")
    
    # Test full extraction
    data = extractor.extract_all_data(start_date, end_date)
    
    if not data.empty:
        print(f"✅ Successfully extracted {len(data)} records")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        # print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
        print("\nSample data:")
        print(data.head())
        print(f"\nData types:\n{data.dtypes}")
        print(f"\nMissing values:\n{data.isnull().sum()}")
    else:
        print("❌ No data extracted")