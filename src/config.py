import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz

load_dotenv()

class Config:
    """Configuration class for the energy forecasting project"""
    
    # ENTSO-E API Configuration
    ENTSOE_API_KEY = os.getenv('ENTSOE_API_KEY')
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'energy_forecast')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    # Database URL
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Timezone Configuration (Critical for energy data!)
    TIMEZONE = pytz.timezone('Europe/Brussels')  # Belgium timezone
    
    # ENTSO-E Domain Configuration for Belgium (reliable data source)
    COUNTRY_DOMAIN = '10YBE----------2'  # Belgium control area
    # GERMANY_DOMAIN = '10Y1001A1001A83F'  # Germany control area (backup)
    
    # Data Configuration
    DATA_START_DATE = datetime(2020, 1, 1, tzinfo=TIMEZONE)
    LOOKBACK_DAYS = 7  # For rolling features
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_fields = ['ENTSOE_API_KEY', 'DB_USER', 'DB_PASSWORD']
        missing_fields = [field for field in required_fields if not getattr(cls, field)]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        return True