# well-dash

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

An interactive web application for visualizing and predicting groundwater levels in Northern California using machine learning and time series analysis.

[Well Dash App](https://38afdf4f-b1d0-4a37-b350-2902a21bd553.plotly.app/)

## 🌊 Overview

This repository contains a Dash-based web application that enables users to explore continuous groundwater level measurements from monitoring stations across Northern California. The app combines historical data visualization with machine learning predictions to estimate water levels at both monitored and unmonitored locations.

### Key Features

- **📍 Interactive Map Interface**: Explore 100+ groundwater monitoring stations across Northern California with real-time filtering by data frequency (daily/monthly)
- **📈 Time Series Visualization**: View historical water level measurements with quality control flags and metadata
- **🤖 Machine Learning Predictions**: Generate water level forecasts using XGBoost regression models trained on temporal, spatial, and environmental features
- **📊 Statistical Decomposition**: Analyze trend, seasonal, and residual components using STL (Seasonal-Trend decomposition using LOESS)
- **🌍 Spatial Interpolation**: Predict groundwater levels at any location by clicking on the map - the model uses nearest basin information and 3DEP elevation data
- **🔍 Nearby Station Analysis**: Compare predictions with actual measurements from nearby monitoring stations

## 🎯 Use Cases

- **Water Resource Planning**: Forecast seasonal water availability for agricultural and municipal planning
- **Research & Analysis**: Study long-term groundwater trends and seasonal patterns
- **Site Assessment**: Estimate water table depth at proposed well locations before drilling
- **Environmental Monitoring**: Track groundwater response to precipitation and climate patterns

## 🛠️ Technology Stack

### Core Framework
- **Dash/Plotly**: Interactive web application and visualizations
- **Python 3.12+**: Backend processing and data analysis

### Machine Learning & Statistics
- **XGBoost**: Gradient boosting regression for water level prediction
- **Statsmodels**: Time series decomposition (STL, Unobserved Components)
- **scikit-learn**: Data preprocessing and model evaluation
- **Gaussian Process Regression**: Uncertainty quantification for predictions

### Data Processing
- **pandas**: Data manipulation and time series handling
- **NumPy**: Numerical computations
- **py3dep**: USGS 3D Elevation Program (3DEP) integration for terrain data

### Data Source
- **Kaggle API**: Continuous Groundwater Level Measurements dataset (2023)
  - Daily measurements from select stations
  - Monthly measurements from 100+ stations
  - Station metadata (coordinates, elevation, well depth, basin)

## 📊 Model Details

### Features Used for Prediction

**Temporal Features:**
- Days since reference date
- Day, month, year
- Cyclical encodings (sin/cos of day-of-year and month)

**Spatial Features:**
- Latitude, longitude
- Ground surface elevation (from 3DEP)
- Well depth
- Groundwater basin classification

**Categorical Features:**
- Station ID (label encoded)
- Basin name (label encoded)

### Model Performance
- Trained on monthly and daily measurements (1970-2022)
- Cross-validated using time series split
- Handles missing data and irregular sampling

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.12
pip
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/well-dash.git
cd well-dash
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the data:**
The app automatically downloads data from Kaggle on first run. Ensure you have:
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`)
- Or the data will be fetched automatically via `kagglehub`

4. **Run the application:**
```bash
python run.py
```

5. **Open your browser:**
Navigate to `http://localhost:8050`

### Project Structure
```
well-dash/
├── app.py                          # Main application entry point
├── wellapp/
│   ├── __init__.py 
│   ├── callbacks.py                # Dash callback functions
│   ├── utils.py                    # Utility functions (data loading, plotting)
│   └── layout.py                   # Layout of web page                
├── models/
│   ├── wl_xgb_model_basin.bin     # Trained XGBoost model
│   ├── station_encoder.joblib     # Station ID encoder
|   ├── predictions_cache.pkl      # Pre-calculated Predictions
│   └── basin_encoder.joblib       # Basin name encoder
├── training/
│   ├── gw_levels.ipynb             # Notebook for ML training 
│   ├── precompute_predictions.py   # Script to generate predictions_cache.pkl 
│   └── utils.py                    # Utility functions for notebook
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
├── pyproject.toml                  # Set up file
├── plotly-cloud.toml               # Plotly set up file
├── .gitignore                      # git ignore file  
└── README.md                       # This file
```

## 📖 Usage Guide

### 1. Station Analysis Tab

**Explore Existing Monitoring Stations:**
1. Use the frequency dropdown to filter stations by data availability
2. Click on any blue station marker to view its data
3. The main plot shows:
   - Historical measurements (blue line with markers)
   - XGBoost predictions (red dashed line)
   - Gaussian Process predictions with uncertainty (green line + shaded area)
4. STL decomposition plots below show:
   - Long-term trend
   - Seasonal patterns
   - Residual variations

### 2. Spatial Prediction Tab

**Predict Water Levels at New Locations:**
1. Click anywhere on the map to select a location
2. The app automatically:
   - Queries ground elevation from USGS 3DEP
   - Identifies the nearest groundwater basin
   - Displays nearby monitoring stations for context
3. Enter estimated well depth (or use default 100 ft)
4. View predicted water level time series (2000-present)
5. Compare with nearby station measurements

## 🧪 Model Training

The XGBoost model was trained using:
- **Dataset**: 100,000+ monthly and daily measurements from 100+ stations
- **Time period**: 1970-2023
- **Features**: 14 input features (temporal + spatial + categorical)
- **Target**: Water Surface Elevation (WSE) in feet above sea level
- **Validation**: Time series cross-validation with forward chaining

### To Train a Model:
```python
# Key steps:
# 1. Load and preprocess data
# 2. Engineer features (temporal, spatial, cyclical)
# 3. Train XGBoost with hyperparameter tuning
# 4. Validate on held-out time periods
# 5. Save model and encoders
```

## 🌐 Deployment

### Deploy to Plotly Cloud (Free)

1. **Create `Procfile`:**
```
web: gunicorn app:server
```

2. **Ensure `requirements.txt` includes:**
```
gunicorn==23.0.0
```

3. **Push to GitHub and connect to Render**

4. **Set start command:** `gunicorn app:server`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions for various platforms.

## 📊 Data Sources

- **Groundwater Level Data**: [Kaggle - Continuous Groundwater Level Measurements (2023)](https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023)
- **Elevation Data**: USGS 3D Elevation Program (3DEP) via `py3dep` package

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Data Provider**: Ali Farahmandfar for the Kaggle groundwater dataset
- **USGS**: For 3DEP elevation data access
- **California DWR**: For groundwater data
- **Plotly/Dash**: For the excellent visualization framework

## 📧 Contact

Tomas Snyder - tomassnyder45@gmail.com

Project Link: [https://github.com/quasiStellar45/well-dash](https://github.com/quasiStellar45/well-dash)

---

## 🔮 Future Enhancements

- [ ] Add precipitation data integration
- [ ] Implement ensemble models
- [ ] Historical drought impact analysis
- [ ] Real-time data integration from USGS/DWR APIs

## ⚠️ Disclaimer

This application is for **informational and educational purposes only**. 

- Predictions are based on historical data and statistical models
- **NOT** a substitute for professional hydrogeological assessment
- **NOT** suitable for regulatory compliance or permitting decisions
- Water well drilling should only be done by licensed professionals
- Consult with qualified experts before making water resource decisions

The authors and contributors are not liable for any decisions made based on this tool.
---

⭐ **Star this repo** if you find it useful!
