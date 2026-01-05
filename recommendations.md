# Recommendations for Code Structure Improvements

## Executive Summary
Your codebase is well-organized, but there are opportunities to improve maintainability, testability, and separation of concerns. This document provides specific recommendations for refactoring the interface between `utils.py` and `callbacks.py`.

---

## 1. Data Loading Strategy

### Current Issue
Data is loaded once when `register_callbacks()` is called, which is good for performance but limits flexibility.

### Recommendation: Create a Data Manager Class

```python
# wellapp/data_manager.py
"""
Centralized data management for the groundwater monitoring application.
"""

class DataManager:
    """
    Singleton class to manage data loading and caching.
    
    Ensures data is loaded once and provides clean access to all datasets.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize and load all data on first instantiation."""
        if not DataManager._initialized:
            self._load_data()
            self._load_models()
            DataManager._initialized = True
    
    def _load_data(self):
        """Load all datasets from Kaggle."""
        (self.df_daily, 
         self.df_monthly, 
         self.quality_codes, 
         self.stations_df) = utils.load_data()
    
    def _load_models(self):
        """Load ML models and encoders."""
        self.model = utils.load_ml_model()
        self.station_encoder = utils.load_encoder("station_encoder.joblib")
        self.basin_encoder = utils.load_encoder("basin_encoder.joblib")
    
    def get_data_by_freq(self, freq):
        """
        Get the appropriate dataframe for the selected frequency.
        
        Parameters
        ----------
        freq : str
            Either 'daily' or 'monthly'
        
        Returns
        -------
        pd.DataFrame
            Corresponding dataframe
        """
        return self.df_daily if freq == 'daily' else self.df_monthly
    
    def get_station_data(self, station_id, freq='monthly'):
        """
        Get data for a specific station.
        
        Parameters
        ----------
        station_id : str
            Station identifier
        freq : str, optional
            Data frequency, by default 'monthly'
        
        Returns
        -------
        pd.DataFrame
            Station data
        """
        df = self.get_data_by_freq(freq)
        return df[df.STATION == station_id].copy()

# Usage in callbacks.py:
def register_callbacks(app):
    data_mgr = DataManager()  # Get singleton instance
    
    @app.callback(...)
    def update_plot(station_id, freq):
        station_data = data_mgr.get_station_data(station_id, freq)
        # ...
```

**Benefits:**
- Single source of truth for data
- Easy to add caching or refresh logic
- Testable without loading real data
- Clear interface for data access

---

## 2. Prediction Pipeline Refactoring

### Current Issue
`generate_ml_predictions()` has too many optional parameters and handles multiple use cases (station vs. spatial predictions).

### Recommendation: Separate Prediction Classes

```python
# wellapp/predictors.py
"""
Prediction classes for groundwater level modeling.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BasePrediction(ABC):
    """
    Abstract base class for prediction pipelines.
    """
    
    def __init__(self, model, data_manager):
        """
        Initialize predictor.
        
        Parameters
        ----------
        model : xgb.XGBRegressor
            Trained XGBoost model
        data_manager : DataManager
            Data manager instance
        """
        self.model = model
        self.data_mgr = data_manager
    
    @abstractmethod
    def prepare_features(self, dates):
        """
        Prepare feature matrix for prediction.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates for prediction
        
        Returns
        -------
        np.ndarray
            Feature matrix
        """
        pass
    
    def predict(self, start_date, end_date=None):
        """
        Generate predictions for date range.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start date for predictions
        end_date : pd.Timestamp, optional
            End date, defaults to current date
        
        Returns
        -------
        predictions : np.ndarray
            Predicted water levels
        dates : pd.DatetimeIndex
            Corresponding dates
        """
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        X = self.prepare_features(dates)
        predictions = self.model.predict(X)
        
        return predictions, dates


class StationPrediction(BasePrediction):
    """
    Predictions for existing monitoring stations.
    """
    
    def __init__(self, model, data_manager, station_id):
        """
        Initialize station predictor.
        
        Parameters
        ----------
        model : xgb.XGBRegressor
            Trained model
        data_manager : DataManager
            Data manager
        station_id : str
            Station identifier
        """
        super().__init__(model, data_manager)
        self.station_id = station_id
        self._load_station_info()
    
    def _load_station_info(self):
        """Load station metadata."""
        station_info = self.data_mgr.stations_df[
            self.data_mgr.stations_df.STATION == self.station_id
        ].iloc[0]
        
        self.lat = station_info['LATITUDE']
        self.lon = station_info['LONGITUDE']
        self.elevation = station_info['ELEV']
        self.well_depth = station_info['WELL_DEPTH']
        self.basin = station_info['BASIN_NAME']
        
        # Encode categorical features
        self.station_encoded = utils.encode_station(
            self.station_id, 
            self.data_mgr.station_encoder
        )
        self.basin_encoded = utils.encode_station(
            self.basin, 
            self.data_mgr.basin_encoder
        )
    
    def prepare_features(self, dates):
        """
        Prepare features for station prediction.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Prediction dates
        
        Returns
        -------
        np.ndarray
            Feature matrix [n_dates, n_features]
        """
        n = len(dates)
        ref_date = self.data_mgr.df_monthly["MSMT_DATE"].min()
        
        # Temporal features
        days_since_ref = (dates - ref_date).days.values
        day_of_year = dates.dayofyear.values
        month = dates.month.values
        year = dates.year.values
        day = dates.day.values
        
        # Cyclical encodings
        sin_doy = np.sin(2 * np.pi * day_of_year / 365)
        cos_doy = np.cos(2 * np.pi * day_of_year / 365)
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        
        # Assemble feature matrix
        X = np.column_stack([
            np.full(n, self.station_encoded),
            np.full(n, self.basin_encoded),
            day,
            month,
            year,
            days_since_ref,
            sin_doy,
            cos_doy,
            sin_month,
            cos_month,
            np.full(n, self.elevation),
            np.full(n, self.lat),
            np.full(n, self.lon),
            np.full(n, self.well_depth),
        ])
        
        return X


class SpatialPrediction(BasePrediction):
    """
    Predictions for arbitrary spatial locations.
    """
    
    def __init__(self, model, data_manager, lat, lon, elevation, well_depth):
        """
        Initialize spatial predictor.
        
        Parameters
        ----------
        model : xgb.XGBRegressor
            Trained model
        data_manager : DataManager
            Data manager
        lat : float
            Latitude
        lon : float
            Longitude
        elevation : float
            Ground surface elevation
        well_depth : float
            Well depth
        """
        super().__init__(model, data_manager)
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.well_depth = well_depth
        
        # Find nearest station's basin
        self._determine_basin()
        
        # Use default encoding for unknown station
        self.station_encoded = 0
    
    def _determine_basin(self):
        """Determine basin from nearest station."""
        basin_name, _, _ = utils.nearest_station_basin(
            {"lat": self.lat, "lon": self.lon},
            self.data_mgr.stations_df
        )
        self.basin = basin_name
        self.basin_encoded = utils.encode_station(
            basin_name,
            self.data_mgr.basin_encoder
        )
    
    def prepare_features(self, dates):
        """Prepare features for spatial prediction (similar to StationPrediction)."""
        # Implementation similar to StationPrediction.prepare_features()
        # ... (code omitted for brevity, would be same as above)
        pass

# Usage in callbacks:
@app.callback(...)
def update_station_plot(station_id):
    predictor = StationPrediction(
        model=data_mgr.model,
        data_manager=data_mgr,
        station_id=station_id
    )
    
    station_data = data_mgr.get_station_data(station_id)
    start_date = station_data['MSMT_DATE'].min()
    
    predictions, dates = predictor.predict(start_date)
    # ...
```

**Benefits:**
- Clear separation of concerns
- Each prediction type has its own class
- Easier to test individual prediction pipelines
- Can add new prediction types (e.g., ensemble) without modifying existing code
- Feature engineering logic is encapsulated and reusable

---

## 3. Plotting Functions Refactoring

### Current Issue
Many plotting functions in `utils.py` are tightly coupled to specific data formats and callback logic.

### Recommendation: Create Plot Builder Classes

```python
# wellapp/plotters.py
"""
Reusable plot builders for water level visualization.
"""

import plotly.graph_objects as go
from typing import Optional, List, Tuple

class WaterLevelPlot:
    """
    Builder for water level time series plots.
    """
    
    def __init__(self, title: str = "Water Level"):
        """
        Initialize plot builder.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        self.fig = go.Figure()
        self.title = title
        self._configure_layout()
    
    def _configure_layout(self):
        """Set default layout."""
        self.fig.update_layout(
            template="plotly_white",
            title=self.title,
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title='Date'
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title="Water Surface Elevation (ft asl)"
            ),
            showlegend=True,
            hovermode='x unified'
        )
    
    def add_observed_data(
        self, 
        dates, 
        values, 
        qc_descriptions: Optional[List[str]] = None,
        name: str = "Observed"
    ):
        """
        Add observed water level data.
        
        Parameters
        ----------
        dates : array-like
            Measurement dates
        values : array-like
            Water level values
        qc_descriptions : list of str, optional
            Quality control descriptions
        name : str
            Trace name
        
        Returns
        -------
        self
            For method chaining
        """
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=name,
            line=dict(color='blue'),
            customdata=qc_descriptions if qc_descriptions else None,
            hovertemplate=(
                "Water Level: %{y:.2f} ft<br>" +
                ("QC Flag: %{customdata[0]}<br>" if qc_descriptions else "") +
                "<extra></extra>"
            )
        ))
        return self
    
    def add_prediction(
        self,
        dates,
        predictions,
        name: str = "Prediction",
        color: str = "red",
        dash: str = "dash"
    ):
        """
        Add prediction trace.
        
        Parameters
        ----------
        dates : array-like
            Prediction dates
        predictions : array-like
            Predicted values
        name : str
            Trace name
        color : str
            Line color
        dash : str
            Line dash style
        
        Returns
        -------
        self
            For method chaining
        """
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name=name,
            line=dict(color=color, dash=dash, width=2),
            hovertemplate="%{y:.2f} ft"
        ))
        return self
    
    def add_confidence_band(
        self,
        dates,
        mean,
        std,
        n_std: float = 1.96,
        name: str = "95% confidence",
        color: str = "rgba(235, 216, 190, 0.3)"
    ):
        """
        Add confidence interval band.
        
        Parameters
        ----------
        dates : array-like
            Dates
        mean : array-like
            Mean predictions
        std : array-like
            Standard deviations
        n_std : float
            Number of standard deviations (1.96 ≈ 95%)
        name : str
            Legend name
        color : str
            Fill color (RGBA)
        
        Returns
        -------
        self
            For method chaining
        """
        # Upper bound
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=mean + n_std * std,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Lower bound (fills to previous)
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=mean - n_std * std,
            mode="lines",
            fill="tonexty",
            fillcolor=color,
            line=dict(width=0),
            name=name,
            hoverinfo="skip"
        ))
        return self
    
    def build(self) -> go.Figure:
        """
        Return the completed figure.
        
        Returns
        -------
        plotly.graph_objects.Figure
            The constructed plot
        """
        return self.fig


# Usage in callbacks:
@app.callback(...)
def update_plot(station_id):
    # Get data
    observed_data = data_mgr.get_station_data(station_id)
    predictions, dates = predictor.predict(...)
    
    # Build plot using fluent interface
    plot = (WaterLevelPlot(title=f"Water Level for {station_id}")
            .add_observed_data(
                observed_data['MSMT_DATE'],
                observed_data['WSE'],
                qc_descriptions=observed_data['QC_DESC']
            )
            .add_prediction(dates, predictions, name="XGBoost")
            .add_confidence_band(dates, gsp_mean, gsp_std))
    
    return plot.build()
```

**Benefits:**
- Fluent/chainable interface for building complex plots
- Reusable across different callbacks
- Easier to test plot construction
- Clear separation between data preparation and visualization

---

## 4. Configuration Management

### Current Issue
Magic numbers and configuration scattered throughout code (e.g., `RADIUS_MILES = 20`, `MAX_STATIONS = 5`, model paths).

### Recommendation: Centralized Configuration

```python
# wellapp/config.py
"""
Application configuration and constants.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """Data loading configuration."""
    kaggle_handle: str = "alifarahmandfar/continuous-groundwater-level-measurements-2023"
    daily_file: str = "gwl-daily.csv"
    monthly_file: str = "gwl-monthly.csv"
    quality_codes_file: str = "gwl-quality_codes.csv"
    stations_file: str = "gwl-stations.csv"


@dataclass
class ModelConfig:
    """Model paths and configuration."""
    xgboost_path: str = "wl_xgb_model_basin.bin"
    station_encoder_path: str = "station_encoder.joblib"
    basin_encoder_path: str = "basin_encoder.joblib"


@dataclass
class SpatialConfig:
    """Spatial analysis configuration."""
    nearby_radius_miles: float = 20.0
    max_nearby_stations: int = 5
    grid_resolution: int = 50
    map_padding_factor: float = 0.1
    min_padding: float = 0.5


@dataclass
class PlotConfig:
    """Plotting configuration."""
    map_zoom: int = 5
    map_height: int = 600
    plot_height: int = 500
    station_colors: List[str] = None
    
    def __post_init__(self):
        if self.station_colors is None:
            self.station_colors = [
                '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea'
            ]


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig = DataConfig()
    models: ModelConfig = ModelConfig()
    spatial: SpatialConfig = SpatialConfig()
    plots: PlotConfig = PlotConfig()


# Singleton instance
config = AppConfig()

# Usage:
from wellapp.config import config

RADIUS_MILES = config.spatial.nearby_radius_miles
MAX_STATIONS = config.spatial.max_nearby_stations
```

**Benefits:**
- Single source of truth for configuration
- Easy to modify behavior without code changes
- Type-safe with dataclasses
- Can load from environment variables or config files

---

## 5. Error Handling and Validation

### Current Issue
Limited error handling; exceptions may propagate to user interface.

### Recommendation: Add Validation Layer

```python
# wellapp/validators.py
"""
Input validation and error handling.
"""

from typing import Optional, Tuple
import pandas as pd

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_station_id(station_id: str, stations_df: pd.DataFrame) -> None:
    """
    Validate that station ID exists.
    
    Parameters
    ----------
    station_id : str
        Station identifier to validate
    stations_df : pd.DataFrame
        DataFrame containing valid stations
    
    Raises
    ------
    ValidationError
        If station ID is invalid
    """
    if station_id not in stations_df['STATION'].values:
        raise ValidationError(f"Invalid station ID: {station_id}")


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validate latitude and longitude.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    
    Raises
    ------
    ValidationError
        If coordinates are out of valid range
    """
    if not (-90 <= lat <= 90):
        raise ValidationError(f"Invalid latitude: {lat}. Must be in [-90, 90]")
    if not (-180 <= lon <= 180):
        raise ValidationError(f"Invalid longitude: {lon}. Must be in [-180, 180]")


def validate_well_depth(depth: Optional[float]) -> float:
    """
    Validate and return well depth.
    
    Parameters
    ----------
    depth : float or None
        Well depth in feet
    
    Returns
    -------
    float
        Validated depth, or default value if None
    
    Raises
    ------
    ValidationError
        If depth is negative
    """
    if depth is None:
        return 100.0  # Default value
    
    if depth < 0:
        raise ValidationError(f"Well depth must be positive, got {depth}")
    
    return depth


# Usage in callbacks with graceful error handling:
@app.callback(...)
def update_plot(station_id):
    try:
        validate_station_id(station_id, data_mgr.stations_df)
        # ... rest of logic
    except ValidationError as e:
        return create_error_plot(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return create_error_plot("An unexpected error occurred")
```

**Benefits:**
- Prevent invalid data from causing crashes
- Provide meaningful error messages to users
- Centralized validation logic
- Easier debugging with clear error messages

---

## 6. Testing Strategy

### Recommendation: Add Unit Tests

```python
# tests/test_predictors.py
"""
Unit tests for prediction classes.
"""

import pytest
import pandas as pd
import numpy as np
from wellapp.predictors import StationPrediction, SpatialPrediction
from wellapp.data_manager import DataManager

@pytest.fixture
def mock_data_manager(mocker):
    """Create a mock data manager for testing."""
    mgr = mocker.Mock(spec=DataManager)
    
    # Mock dataframes
    mgr.df_monthly = pd.DataFrame({
        'MSMT_DATE': pd.date_range('2000-01-01', periods=100, freq='ME'),
        'STATION': ['STATION_1'] * 100,
        'WSE': np.random.randn(100) + 1000
    })
    
    mgr.stations_df = pd.DataFrame({
        'STATION': ['STATION_1'],
        'LATITUDE': [40.0],
        'LONGITUDE': [-105.0],
        'ELEV': [5000.0],
        'WELL_DEPTH': [100.0],
        'BASIN_NAME': ['Test Basin']
    })
    
    return mgr


@pytest.fixture
def mock_model(mocker):
    """Create a mock XGBoost model."""
    model = mocker.Mock()
    model.predict.return_value = np.array([1000.0, 1001.0, 1002.0])
    return model


def test_station_prediction_initialization(mock_data_manager, mock_model):
    """Test that StationPrediction initializes correctly."""
    predictor = StationPrediction(
        model=mock_model,
        data_manager=mock_data_manager,
        station_id='STATION_1'
    )
    
    assert predictor.station_id == 'STATION_1'
    assert predictor.lat == 40.0
    assert predictor.lon == -105.0


def test_station_prediction_features(mock_data_manager, mock_model):
    """Test feature matrix creation."""
    predictor = StationPrediction(
        model=mock_model,
        data_manager=mock_data_manager,
        station_id='STATION_1'
    )
    
    dates = pd.date_range('2020-01-01', periods=3, freq='ME')
    X = predictor.prepare_features(dates)
    
    # Check dimensions
    assert X.shape == (3, 14)  # 3 dates, 14 features
    
    # Check feature types
    assert np.all(np.isfinite(X))  # No NaN or inf


def test_spatial_prediction_basin_determination(mock_data_manager, mock_model):
    """Test that spatial predictions correctly identify nearest basin."""
    predictor = SpatialPrediction(
        model=mock_model,
        data_manager=mock_data_manager,
        lat=40.1,
        lon=-105.1,
        elevation=5100.0,
        well_depth=150.0
    )
    
    assert predictor.basin == 'Test Basin'

# Run with: pytest tests/
```

---

## 7. Documentation Structure

### Recommendation: Add Comprehensive Documentation

```
docs/
├── architecture.md          # System architecture overview
├── data_pipeline.md         # Data flow and processing
├── prediction_models.md     # ML model documentation
├── api/                     # API documentation
│   ├── data_manager.md
│   ├── predictors.md
│   └── plotters.md
└── user_guide.md           # End-user documentation
```

---

## Priority Recommendations

### High Priority (Implement First)
1. **DataManager class** - Centralizes data access and improves maintainability
2. **Configuration management** - Easy wins for maintainability
3. **Error handling** - Improves user experience and debugging

### Medium Priority
4. **Prediction classes** - Improves code organization and testability
5. **Validation layer** - Prevents bugs and improves robustness

### Low Priority (Nice to Have)
6. **Plot builders** - Reduces code duplication but not critical
7. **Comprehensive testing** - Important for long-term maintenance
8. **Documentation** - Helps onboarding and maintenance

---

## Migration Strategy

1. **Phase 1**: Add new classes alongside existing code
   - Create `data_manager.py`, `config.py`
   - Don't break existing callbacks yet

2. **Phase 2**: Gradually migrate callbacks
   - Start with simplest callbacks
   - Test thoroughly after each migration
   - Keep old functions until migration complete

3. **Phase 3**: Clean up and optimize
   - Remove old functions
   - Add comprehensive tests
   - Update documentation

This approach allows you to improve the code incrementally without breaking the existing application.