# wellapp/predictors.py
"""
Prediction classes for groundwater level modeling.

This module provides different prediction classes for various use cases:
- StationPrediction: predictions for existing monitoring stations
- SpatialPrediction: predictions for arbitrary geographic locations
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple

import wellapp.utils as utils
from wellapp.config import config
from wellapp.validators import (
    validate_station_id,
    validate_coordinates,
    validate_well_depth,
    validate_elevation,
    validate_date_range
)


class BasePrediction(ABC):
    """
    Abstract base class for prediction pipelines.
    
    This class defines the common interface and shared functionality for all
    prediction types. Subclasses must implement the prepare_features method.
    
    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained XGBoost model
    data_manager : DataManager
        Data manager instance for accessing datasets
    
    Attributes
    ----------
    model : xgb.XGBRegressor
        The prediction model
    data_mgr : DataManager
        Access to all datasets
    """
    
    def __init__(self, model: xgb.XGBRegressor, data_manager):
        """
        Initialize base predictor.
        
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
    def prepare_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Prepare feature matrix for prediction.
        
        This method must be implemented by subclasses to create the
        appropriate feature matrix for their specific use case.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates for which to generate predictions
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape [n_dates, n_features]
        """
        pass
    
    def _compute_temporal_features(
        self, 
        dates: pd.DatetimeIndex
    ) -> dict:
        """
        Compute temporal features from dates.
        
        Generates both linear and cyclical time features used by the model.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates for feature computation
        
        Returns
        -------
        dict
            Dictionary containing all temporal features:
            - days_since_ref: days since reference date
            - day, month, year: date components
            - sin_doy, cos_doy: cyclical day of year
            - sin_month, cos_month: cyclical month
        """
        ref_date = self.data_mgr.get_reference_date()
        
        # Linear time features
        days_since_ref = (dates - ref_date).days.values
        day_of_year = dates.dayofyear.values
        month = dates.month.values
        year = dates.year.values
        day = dates.day.values
        
        # Cyclical encodings to capture periodicity
        sin_doy = np.sin(2 * np.pi * day_of_year / 365)
        cos_doy = np.cos(2 * np.pi * day_of_year / 365)
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        
        return {
            'days_since_ref': days_since_ref,
            'day': day,
            'month': month,
            'year': year,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'sin_month': sin_month,
            'cos_month': cos_month
        }
    
    def predict(
        self, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp = None
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Generate predictions for a date range.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start date for predictions
        end_date : pd.Timestamp, optional
            End date for predictions, defaults to current date
        
        Returns
        -------
        predictions : np.ndarray
            Predicted water surface elevations
        dates : pd.DatetimeIndex
            Corresponding dates for each prediction
        
        Raises
        ------
        ValidationError
            If date range is invalid
        
        Examples
        --------
        >>> predictor = StationPrediction(model, data_mgr, 'STATION_001')
        >>> predictions, dates = predictor.predict(
        ...     pd.Timestamp('2020-01-01'),
        ...     pd.Timestamp('2021-01-01')
        ... )
        """
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        validate_date_range(start_date, end_date)
        
        # Generate date range at monthly frequency
        dates = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=config.prediction.prediction_frequency
        )
        
        # Prepare features and generate predictions
        X = self.prepare_features(dates)
        predictions = self.model.predict(X)
        
        return predictions, dates


class StationPrediction(BasePrediction):
    """
    Predictions for existing monitoring stations.
    
    This class handles predictions for stations with known metadata,
    using the station's actual coordinates, elevation, and well depth.
    
    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model
    data_manager : DataManager
        Data manager
    station_id : str
        Station identifier
    
    Attributes
    ----------
    station_id : str
        Station identifier
    lat, lon : float
        Station coordinates
    elevation : float
        Ground surface elevation
    well_depth : float
        Well depth
    basin : str
        Basin name
    station_encoded : int
        Encoded station ID for model
    basin_encoded : int
        Encoded basin name for model
    
    Examples
    --------
    >>> from wellapp.data_manager import DataManager
    >>> data_mgr = DataManager()
    >>> predictor = StationPrediction(data_mgr.model, data_mgr, 'STATION_001')
    >>> predictions, dates = predictor.predict(pd.Timestamp('2020-01-01'))
    """
    
    def __init__(self, model: xgb.XGBRegressor, data_manager, station_id: str):
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
        
        Raises
        ------
        ValidationError
            If station_id is invalid
        """
        super().__init__(model, data_manager)
        
        # Validate station exists
        validate_station_id(station_id, data_manager.stations_df)
        
        self.station_id = station_id
        self._load_station_info()
    
    def _load_station_info(self):
        """
        Load station metadata and encode categorical features.
        
        Retrieves station information from the data manager and encodes
        the station ID and basin name for use in the model.
        """
        station_info = self.data_mgr.get_station_info(self.station_id)
        
        # Extract metadata
        self.lat = station_info['LATITUDE']
        self.lon = station_info['LONGITUDE']
        self.elevation = station_info['ELEV']
        self.well_depth = station_info['WELL_DEPTH']
        self.basin = station_info['BASIN_NAME']
        
        # Validate extracted values
        validate_coordinates(self.lat, self.lon)
        validate_elevation(self.elevation)
        validate_well_depth(self.well_depth)
        
        # Encode categorical features
        self.station_encoded = utils.encode_station(
            self.station_id, 
            self.data_mgr.station_encoder
        )
        self.basin_encoded = utils.encode_station(
            self.basin, 
            self.data_mgr.basin_encoder
        )
    
    def prepare_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Prepare feature matrix for station prediction.
        
        Creates a feature matrix with station metadata and temporal features
        for each date in the provided date range.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Prediction dates
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape [n_dates, 14]
            Features: [station_encoded, basin_encoded, day, month, year,
                      days_since_ref, sin_doy, cos_doy, sin_month, cos_month,
                      elevation, lat, lon, well_depth]
        """
        n = len(dates)
        
        # Get temporal features
        temporal = self._compute_temporal_features(dates)
        
        # Assemble feature matrix (must match training feature order)
        X = np.column_stack([
            np.full(n, self.station_encoded),      # Station ID (encoded)
            np.full(n, self.basin_encoded),        # Basin (encoded)
            temporal['day'],                        # Day of month
            temporal['month'],                      # Month
            temporal['year'],                       # Year
            temporal['days_since_ref'],            # Days since reference
            temporal['sin_doy'],                   # Cyclical day of year
            temporal['cos_doy'],
            temporal['sin_month'],                 # Cyclical month
            temporal['cos_month'],
            np.full(n, self.elevation),            # Ground elevation
            np.full(n, self.lat),                  # Latitude
            np.full(n, self.lon),                  # Longitude
            np.full(n, self.well_depth),           # Well depth
        ])
        
        return X
    
    def get_start_date(self) -> pd.Timestamp:
        """
        Get the earliest measurement date for this station.
        
        Returns
        -------
        pd.Timestamp
            Earliest date with data for this station
        
        Examples
        --------
        >>> predictor = StationPrediction(model, data_mgr, 'STATION_001')
        >>> start = predictor.get_start_date()
        >>> predictions, dates = predictor.predict(start)
        """
        station_data = self.data_mgr.get_station_data(
            self.station_id, 
            freq='monthly'
        )
        
        if len(station_data) == 0:
            # If no monthly data, use a default start date
            return pd.Timestamp(f"{config.prediction.default_start_year}-01-01")
        
        return station_data['MSMT_DATE'].min()


class SpatialPrediction(BasePrediction):
    """
    Predictions for arbitrary spatial locations.
    
    This class handles predictions for any geographic location, using
    provided coordinates and elevation. The basin is inferred from the
    nearest monitoring station.
    
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
        Ground surface elevation (feet)
    well_depth : float, optional
        Well depth (feet), uses default if not provided
    
    Attributes
    ----------
    lat, lon : float
        Location coordinates
    elevation : float
        Ground elevation
    well_depth : float
        Well depth
    basin : str
        Nearest station's basin
    basin_encoded : int
        Encoded basin for model
    station_encoded : int
        Default encoding (0) for unknown location
    
    Examples
    --------
    >>> predictor = SpatialPrediction(
    ...     model=data_mgr.model,
    ...     data_manager=data_mgr,
    ...     lat=40.0,
    ...     lon=-105.0,
    ...     elevation=5000.0,
    ...     well_depth=150.0
    ... )
    >>> predictions, dates = predictor.predict(pd.Timestamp('2000-01-01'))
    """
    
    def __init__(
        self, 
        model: xgb.XGBRegressor, 
        data_manager, 
        lat: float, 
        lon: float, 
        elevation: float,
        well_depth: float = None
    ):
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
        well_depth : float, optional
            Well depth, uses default if not provided
        
        Raises
        ------
        ValidationError
            If coordinates, elevation, or well_depth are invalid
        """
        super().__init__(model, data_manager)
        
        # Validate inputs
        validate_coordinates(lat, lon)
        validate_elevation(elevation)
        
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.well_depth = validate_well_depth(
            well_depth, 
            config.prediction.default_well_depth
        )
        
        # Find nearest station's basin and encode
        self._determine_basin()
        
        # Use default encoding for unknown station
        self.station_encoded = config.prediction.unknown_station_encoding
    
    def _determine_basin(self):
        """
        Determine basin from nearest monitoring station.
        
        Finds the geographically nearest station and uses its basin
        classification for the spatial prediction.
        """
        basin_name, nearest_station, distance = utils.nearest_station_basin(
            {"lat": self.lat, "lon": self.lon},
            self.data_mgr.stations_df
        )
        
        self.basin = basin_name
        self.nearest_station = nearest_station
        self.distance_to_nearest = distance
        
        # Encode basin for model
        self.basin_encoded = utils.encode_station(
            basin_name,
            self.data_mgr.basin_encoder
        )
    
    def prepare_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Prepare feature matrix for spatial prediction.
        
        Creates a feature matrix using the provided location information
        and temporal features.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Prediction dates
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape [n_dates, 14]
            Features: [station_encoded, basin_encoded, day, month, year,
                      days_since_ref, sin_doy, cos_doy, sin_month, cos_month,
                      elevation, lat, lon, well_depth]
        """
        n = len(dates)
        
        # Get temporal features
        temporal = self._compute_temporal_features(dates)
        
        # Assemble feature matrix (same structure as StationPrediction)
        X = np.column_stack([
            np.full(n, self.station_encoded),      # Unknown station = 0
            np.full(n, self.basin_encoded),        # Nearest station's basin
            temporal['day'],
            temporal['month'],
            temporal['year'],
            temporal['days_since_ref'],
            temporal['sin_doy'],
            temporal['cos_doy'],
            temporal['sin_month'],
            temporal['cos_month'],
            np.full(n, self.elevation),
            np.full(n, self.lat),
            np.full(n, self.lon),
            np.full(n, self.well_depth),
        ])
        
        return X
    
    def get_start_date(self) -> pd.Timestamp:
        """
        Get the default start date for spatial predictions.
        
        Returns
        -------
        pd.Timestamp
            Default start date from configuration
        
        Examples
        --------
        >>> predictor = SpatialPrediction(model, data_mgr, 40.0, -105.0, 5000.0)
        >>> start = predictor.get_start_date()
        >>> print(start)
        2000-01-01 00:00:00
        """
        return pd.Timestamp(f"{config.prediction.default_start_year}-01-01")