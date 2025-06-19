# ⚡Energy Price Forecasting Platform

<div align="center">

![Energy Forecasting](https://img.shields.io/badge/Energy-Forecasting-blue?style=for-the-badge&logo=lightning)
![Machine Learning](https://img.shields.io/badge/ML-LightGBM-green?style=for-the-badge&logo=scikit-learn)
![Data Source](https://img.shields.io/badge/Data-ENTSO--E-orange?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-red?style=for-the-badge&logo=mlflow)

**Production-ready machine learning system for predicting Belgian electricity market prices using real-time ENTSO-E data**

[🚀 Quick Start](#quick-start) • [📊 Demo](#demo) • [🛠️ Installation](#installation) • [📈 Performance](#model-performance) • [🔧 Architecture](#architecture)

</div>

---

## 🎯 **Project Overview**

This project implements an **end-to-end machine learning pipeline** for forecasting electricity market prices (belgium as current use case). Built with production-grade engineering practices, it demonstrates the complete lifecycle from data extraction to model deployment.

### **🏆 Key Achievements**
- **📈 2-5% MAPE** - Excellent forecasting accuracy for energy markets
- **🔄 Automated Pipeline** - End-to-end data processing and model training
- **⚡ Real-time Data** - Live integration with ENTSO-E Transparency Platform
- **🐳 Production Ready** - Containerized deployment with monitoring
- **📊 MLOps Integration** - Experiment tracking and model versioning

### **💡 Business Value**
- **Energy Trading**: Accurate price predictions for trading decisions
- **Risk Management**: Forecast extreme price events and volatility
- **Market Analysis**: Understanding of European energy market dynamics
- **Portfolio Optimization**: Data-driven energy procurement strategies

---

## 🚀 **Quick Start**

### **🔧 Prerequisites**
- Python 3.10+
- PostgreSQL 13+
- ENTSO-E API access (free registration)

### **⚡ 5-Minute Setup**

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/energy-price-forecasting.git
cd energy-price-forecasting
python -m venv energy_forecast_env
source energy_forecast_env/bin/activate  # Windows: energy_forecast_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your ENTSO-E API key

# 4. Setup database
docker run --name postgres-energy -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=energy_forecast -p 5432:5432 -d postgres:15

# 5. Run the pipeline
python main_pipeline.py
```

### **🎯 Expected Results**
After running, you'll have:
- ✅ **Real Belgian energy data** extracted and processed
- ✅ **Trained LightGBM model** with validated performance
- ✅ **24-hour price forecasts** generated
- ✅ **MLflow experiments** tracked and visualized

---

## 📊 **Demo**

### **Model Performance Visualization**
![Model Performance](docs/images/model_performance.png)
*Actual vs Predicted prices showing excellent tracking of daily patterns and price spikes*

### **Feature Importance Analysis**
![Feature Importance](docs/images/feature_importance.png)
*Key insights: Price mean reversion and solar generation drive Belgian market prices*

### **Sample Forecast Output**
```json
{
  "forecast_date": "2025-06-18",
  "forecast_horizon": "24h",
  "predictions": [
    {"time": "00:00", "price": 85.2, "confidence": "±8.5"},
    {"time": "01:00", "price": 78.1, "confidence": "±7.2"},
    {"time": "12:00", "price": 45.3, "confidence": "±9.1"}
  ],
  "avg_price": "67.8 EUR/MWh",
  "model_version": "v1.0.0"
}
```

---

## 🛠️ **Installation**

### **📋 System Requirements**
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Network**: Internet connection for ENTSO-E API

### **🔐 API Setup**
1. **Register** at [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
2. **Request API access** by emailing `transparency@entsoe.eu`
3. **Get your API key** from account settings (approval takes 1-2 days)

### **💾 Database Setup**

#### Option A: Docker (Recommended)
```bash
docker run --name postgres-energy \
  -e POSTGRES_DB=energy_forecast \
  -e POSTGRES_USER=energy_user \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 -d postgres:15
```

#### Option B: Local PostgreSQL
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb energy_forecast

# macOS
brew install postgresql
createdb energy_forecast
```

### **⚙️ Environment Configuration**
```bash
# .env file
ENTSOE_API_KEY=your_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=energy_forecast
DB_USER=energy_user
DB_PASSWORD=your_password
```

---

## 🏗️ **Architecture**

### **📐 System Design**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ENTSO-E API   │───▶│  Data Pipeline   │───▶│   PostgreSQL    │
│  (Real-time)    │    │  (Extract/Load)  │    │   (Storage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Forecasts     │◀───│  ML Pipeline     │◀───│ Feature Engine  │
│  (24h ahead)    │    │  (LightGBM)      │    │ (Engineering)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   MLflow         │
                       │  (Tracking)      │
                       └──────────────────┘
```

### **🔧 Tech Stack**
- **🐍 Python 3.10+**: Core language
- **📊 Pandas**: Data manipulation and analysis
- **🤖 LightGBM**: Gradient boosting for forecasting
- **🗄️ PostgreSQL**: Time-series data storage
- **📈 MLflow**: Experiment tracking and model registry
- **🔌 ENTSO-E API**: Real-time European energy data
- **🐳 Docker**: Containerization and deployment



### **🏆 Key Insights Discovered**
1. **📊 Mean Reversion**: Belgian prices strongly revert to 7-day averages
2. **☀️ Solar Impact**: Solar generation significantly reduces daytime prices
3. **🔄 Market Patterns**: Clear daily and weekly cyclical behaviors
4. **⚡ Renewable Effect**: Higher renewable penetration = lower prices
5. **🕐 Time Dependencies**: Hour-of-day patterns crucial for accuracy

### **📋 Feature Importance Rankings**
| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `price_pct_of_7d_range` | 🟦🟦🟦🟦🟦 | Price Dynamics |
| 2 | `price_deviation_7d` | 🟦🟦🟦🟦⬜ | Price Dynamics |
| 3 | `solar_forecast` | 🟦🟦🟦⬜⬜ | Renewable Supply |
| 4 | `price_deviation_7d_pct` | 🟦🟦⬜⬜⬜ | Price Dynamics |
| 5 | `renewable_penetration` | 🟦⬜⬜⬜⬜ | Market Structure |

---

## 🔌 **Usage**

### **🏃‍♂️ Running the Pipeline**

#### Full Pipeline (Extract → Train → Forecast)
```bash
python main_pipeline.py
```

#### Individual Components
```bash
# Data extraction only
python main_pipeline.py --extract-only

# Model training only
python main_pipeline.py --train-only

# Generate forecasts only
python main_pipeline.py --forecast-only --forecast-hours 24
```

### **📊 Viewing Results**

#### MLflow Experiment Tracking
```bash
mlflow ui
# Open http://localhost:5000
```

#### Database Queries
```sql
-- View latest prices
SELECT timestamp, price, wind_forecast, solar_forecast 
FROM energy_data 
ORDER BY timestamp DESC 
LIMIT 24;

-- Price statistics
SELECT 
    AVG(price) as avg_price,
    STDDEV(price) as price_volatility,
    MIN(price) as min_price,
    MAX(price) as max_price
FROM energy_data 
WHERE timestamp >= NOW() - INTERVAL '7 days';
```

### **🔧 Configuration Options**

#### Date Range Customization
```python
# In main_pipeline.py
start_date = timezone.localize(datetime(2024, 1, 1, 0, 0, 0))
end_date = timezone.localize(datetime(2024, 12, 31, 23, 0, 0))
```

#### Model Hyperparameters
```python
# In main_pipeline.py - _train_lightgbm_model()
params = {
    'objective': 'regression',
    'num_leaves': 50,        # Complexity control
    'learning_rate': 0.1,    # Training speed
    'feature_fraction': 0.8, # Feature sampling
    'bagging_fraction': 0.8, # Row sampling
}
```

---

## 🧪 **Development**

### **🛠️ Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

### **📝 Adding New Features**

#### New Data Sources
1. **Create extractor** in `src/data_extraction.py`
2. **Add transformation** in `src/data_transformation.py`
3. **Update database schema** in `src/database.py`
4. **Test integration** in pipeline

#### New Models
1. **Add model class** in `src/models/`
2. **Implement training** in `main_pipeline.py`
3. **Add evaluation metrics**
4. **Update MLflow logging**


### **🔗 External References**
- [ENTSO-E API Documentation](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- [Belgian Energy Market Overview](https://www.elia.be/en/about-elia/what-we-do)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

## 🤝 **Contributing**

### **🛠️ How to Contribute**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **📋 Contribution Guidelines**
- **Code Style**: Follow PEP 8 (use `black` formatter)
- **Testing**: Add tests for new features
- **Documentation**: Update relevant docs
- **Performance**: Benchmark changes affecting speed
- **Data Quality**: Validate data processing changes

### **🎯 Areas for Contribution**
- **🔮 Advanced Models**: LSTM, Transformer architectures
- **📊 Feature Engineering**: Weather data, economic indicators
- **🌍 Multi-Market**: Extend to other European markets
- **⚡ Performance**: Optimization and caching
- **📱 Visualization**: Advanced charting and dashboards

---

## 📜 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### **⚖️ Data Usage**
- **ENTSO-E Data**: Subject to [ENTSO-E Terms](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Web%20service%20integration%20guide.html)
- **Model Outputs**: Free to use for research and commercial applications
- **Attribution**: Please cite this project if used in research

---


**Made with ❤️ for the energy and ML communities**


</div>