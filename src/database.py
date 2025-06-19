from sqlalchemy import create_engine, Column, DateTime, Float, Integer, String, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime
import logging

Base = declarative_base()

class EnergyData(Base):
    """Database model for energy market data"""
    __tablename__ = 'energy_data'
    
    # Primary key - timestamp in Europe/Berlin timezone
    timestamp = Column(DateTime, primary_key=True)
    
    # Price data (EUR/MWh)
    price = Column(Float)
    
    # Generation forecasts (MW)
    wind_forecast = Column(Float)
    solar_forecast = Column(Float)
    total_load_forecast = Column(Float)
    
    # Engineered features
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    month = Column(Integer)
    is_weekend = Column(Integer)
    
    # Lagged features
    price_lag_24h = Column(Float)
    price_lag_48h = Column(Float)
    price_lag_7d = Column(Float)
    
    # Rolling features
    price_rolling_7d_mean = Column(Float)
    price_rolling_7d_std = Column(Float)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_hour_day', 'hour_of_day'),
        Index('idx_date', 'timestamp'),
    )

class DatabaseManager:
    """Handles database operations for the energy forecasting project"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def insert_or_update_data(self, df: pd.DataFrame):
        """Insert or update data in the energy_data table"""
        try:
            # Convert DataFrame to dict records
            records = df.to_dict('records')
            
            with self.engine.begin() as conn:
                # Use PostgreSQL's ON CONFLICT DO UPDATE for upsert
                stmt = insert(EnergyData).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['timestamp'],
                    set_={
                        'price': stmt.excluded.price,
                        'wind_forecast': stmt.excluded.wind_forecast,
                        'solar_forecast': stmt.excluded.solar_forecast,
                        'total_load_forecast': stmt.excluded.total_load_forecast,
                        'hour_of_day': stmt.excluded.hour_of_day,
                        'day_of_week': stmt.excluded.day_of_week,
                        'month': stmt.excluded.month,
                        'is_weekend': stmt.excluded.is_weekend,
                        'price_lag_24h': stmt.excluded.price_lag_24h,
                        'price_lag_48h': stmt.excluded.price_lag_48h,
                        'price_lag_7d': stmt.excluded.price_lag_7d,
                        'price_rolling_7d_mean': stmt.excluded.price_rolling_7d_mean,
                        'price_rolling_7d_std': stmt.excluded.price_rolling_7d_std,
                    }
                )
                conn.execute(stmt)
            
            self.logger.info(f"Successfully inserted/updated {len(records)} records")
            
        except Exception as e:
            self.logger.error(f"Error inserting data: {e}")
            raise
    
    def get_data(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Retrieve data from the database"""
        try:
            query = "SELECT * FROM energy_data"
            conditions = []
            
            if start_date:
                conditions.append(f"timestamp >= '{start_date}'")
            if end_date:
                conditions.append(f"timestamp <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql(query, self.engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Retrieved {len(df)} records from database")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            raise
    
    def get_latest_timestamp(self) -> datetime:
        """Get the latest timestamp in the database"""
        try:
            query = "SELECT MAX(timestamp) as max_timestamp FROM energy_data"
            result = pd.read_sql(query, self.engine)
            latest = result['max_timestamp'].iloc[0]
            
            if pd.isna(latest):
                return None
            
            return pd.to_datetime(latest)
            
        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {e}")
            return None

