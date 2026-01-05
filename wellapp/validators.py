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