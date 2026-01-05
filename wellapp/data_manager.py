# wellapp/data_manager.py
"""
Centralized data management for the groundwater monitoring application.

This module provides a singleton DataManager class that handles all data
loading, caching, and access. This ensures data is loaded once and provides
a clean interface for all components of the application.
"""

from typing import Optional
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import wellapp.utils as utils
from wellapp.config import config
from wellapp.validators import (
    validate_station_id,
    validate_frequency,
    validate_dataframe_not_empty,
    validate_required_columns
)


class DataManager:
    """
    Singleton class to manage data loading and caching.
    
    This class ensures that all datasets and models are loaded exactly once
    and provides clean, validated access to data throughout the application.
    
    Attributes
    ----------
    df_daily : pd.DataFrame
        Daily groundwater level measurements
    df_monthly : pd.DataFrame
        Monthly groundwater level measurements
    quality_codes : pd.DataFrame
        Quality code descriptions
    stations_df : pd.DataFrame
        Station metadata (location, elevation, etc.)
    model : xgb.XGBRegressor
        Trained XGBoost model for predictions
    station_encoder : LabelEncoder
        Encoder for station IDs
    basin_encoder : LabelEncoder
        Encoder for basin names
    
    Examples
    --------
    >>> data_mgr = DataManager()  # First call loads data
    >>> data_mgr2 = DataManager()  # Returns same instance
    >>> assert data_mgr is data_mgr2  # True - singleton
    >>> station_data = data_mgr.get_station_data('STATION_001', 'monthly')
    """
    
    _instance: Optional['DataManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """
        Ensure only one instance of DataManager exists (singleton pattern).
        
        Returns
        -------
        DataManager
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the DataManager and load all data on first instantiation.
        
        Subsequent calls to __init__ do nothing, maintaining the singleton.
        """
        if not DataManager._initialized:
            print("Loading data and models...")
            self._load_data()
            self._load_models()
            self._validate_data()
            DataManager._initialized = True
            print("Data loading complete!")
    
    def _load_data(self):
        """
        Load all datasets from Kaggle.
        
        Loads daily measurements, monthly measurements, quality codes,
        and station metadata into instance attributes.
        """
        (self.df_daily, 
         self.df_monthly, 
         self.quality_codes, 
         self.stations_df) = utils.load_data()
    
    def _load_models(self):
        """
        Load machine learning models and encoders.
        
        Loads the XGBoost model and label encoders for stations and basins.
        """
        self.model = utils.load_ml_model(config.models.xgboost_path)
        self.station_encoder = utils.load_encoder(config.models.station_encoder_path)
        self.basin_encoder = utils.load_encoder(config.models.basin_encoder_path)
    
    def _validate_data(self):
        """
        Validate that all required data is present and well-formed.
        
        Raises
        ------
        ValidationError
            If any dataset is missing or malformed
        """
        # Validate dataframes are not empty
        validate_dataframe_not_empty(self.df_daily, "Daily data")
        validate_dataframe_not_empty(self.df_monthly, "Monthly data")
        validate_dataframe_not_empty(self.quality_codes, "Quality codes")
        validate_dataframe_not_empty(self.stations_df, "Stations data")
        
        # Validate required columns
        validate_required_columns(
            self.df_daily,
            ['STATION', 'MSMT_DATE', 'WSE', 'WSE_QC'],
            "Daily data"
        )
        validate_required_columns(
            self.df_monthly,
            ['STATION', 'MSMT_DATE', 'WSE'],
            "Monthly data"
        )
        validate_required_columns(
            self.stations_df,
            ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEV', 'WELL_DEPTH', 'BASIN_NAME'],
            "Stations data"
        )
    
    def get_data_by_freq(self, freq: str) -> pd.DataFrame:
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
        
        Raises
        ------
        ValidationError
            If frequency is invalid
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> daily_df = data_mgr.get_data_by_freq('daily')
        >>> monthly_df = data_mgr.get_data_by_freq('monthly')
        """
        validate_frequency(freq)
        return self.df_daily if freq == 'daily' else self.df_monthly
    
    def get_station_data(
        self, 
        station_id: str, 
        freq: str = 'monthly'
    ) -> pd.DataFrame:
        """
        Get data for a specific station.
        
        Parameters
        ----------
        station_id : str
            Station identifier
        freq : str, optional
            Data frequency ('daily' or 'monthly'), by default 'monthly'
        
        Returns
        -------
        pd.DataFrame
            Station data sorted by date, or empty dataframe if station
            has no data at this frequency
        
        Raises
        ------
        ValidationError
            If station_id or frequency is invalid
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> station_data = data_mgr.get_station_data('STATION_001', 'monthly')
        >>> print(station_data.columns)
        Index(['STATION', 'MSMT_DATE', 'WSE', ...])
        """
        validate_station_id(station_id, self.stations_df)
        validate_frequency(freq)
        
        df = self.get_data_by_freq(freq)
        station_data = df[df.STATION == station_id].copy()
        
        if len(station_data) > 0:
            station_data = station_data.sort_values('MSMT_DATE')
        
        return station_data
    
    def get_station_info(self, station_id: str) -> pd.Series:
        """
        Get metadata for a specific station.
        
        Parameters
        ----------
        station_id : str
            Station identifier
        
        Returns
        -------
        pd.Series
            Station metadata including coordinates, elevation, well depth, etc.
        
        Raises
        ------
        ValidationError
            If station_id is invalid
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> info = data_mgr.get_station_info('STATION_001')
        >>> print(f"Latitude: {info['LATITUDE']}, Elevation: {info['ELEV']}")
        """
        validate_station_id(station_id, self.stations_df)
        return self.stations_df[self.stations_df.STATION == station_id].iloc[0]
    
    def get_available_stations(self, freq: Optional[str] = None) -> list:
        """
        Get list of available stations, optionally filtered by data frequency.
        
        Parameters
        ----------
        freq : str, optional
            If provided, only return stations with data at this frequency
        
        Returns
        -------
        list
            List of station IDs
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> all_stations = data_mgr.get_available_stations()
        >>> monthly_stations = data_mgr.get_available_stations('monthly')
        """
        if freq is None:
            return self.stations_df['STATION'].unique().tolist()
        
        validate_frequency(freq)
        df = self.get_data_by_freq(freq)
        return df['STATION'].unique().tolist()
    
    def station_has_data(self, station_id: str, freq: str) -> bool:
        """
        Check if a station has data at the specified frequency.
        
        Parameters
        ----------
        station_id : str
            Station identifier
        freq : str
            Data frequency ('daily' or 'monthly')
        
        Returns
        -------
        bool
            True if station has data at this frequency
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> has_daily = data_mgr.station_has_data('STATION_001', 'daily')
        """
        try:
            validate_station_id(station_id, self.stations_df)
            validate_frequency(freq)
        except:
            return False
        
        available_stations = self.get_available_stations(freq)
        return station_id in available_stations
    
    def get_reference_date(self) -> pd.Timestamp:
        """
        Get the earliest date in the monthly dataset (reference date).
        
        This is used for calculating 'days since reference' features
        in the prediction models.
        
        Returns
        -------
        pd.Timestamp
            Earliest measurement date in monthly data
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> ref_date = data_mgr.get_reference_date()
        >>> print(ref_date)
        """
        return self.df_monthly["MSMT_DATE"].min()
    
    def get_quality_code_description(self, qc_code: str) -> str:
        """
        Get human-readable description for a quality code.
        
        Parameters
        ----------
        qc_code : str
            Quality control code
        
        Returns
        -------
        str
            Description of the quality code, or 'Unknown' if not found
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> desc = data_mgr.get_quality_code_description('A')
        >>> print(desc)
        """
        qc_map = dict(
            zip(self.quality_codes["QUALITY_CODE"], 
                self.quality_codes["DESCRIPTION"])
        )
        return qc_map.get(qc_code, "Unknown")
    
    def reset(self):
        """
        Reset the singleton instance (primarily for testing).
        
        This method allows the DataManager to be reinitialized with fresh
        data. Should be used with caution in production.
        
        Examples
        --------
        >>> data_mgr = DataManager()
        >>> data_mgr.reset()
        >>> data_mgr2 = DataManager()  # Will reload data
        """
        DataManager._instance = None
        DataManager._initialized = False