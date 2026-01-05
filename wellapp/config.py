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