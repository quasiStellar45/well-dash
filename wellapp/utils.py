"""
Any utility functions such as loading data
or editing strings, etc...
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.express as px
import pandas as pd
import py3dep

def load_kaggle_data(file_name, data_handle):
    """
    Loads Kaggle data into a Pandas dataframe.

    Args:
    - file_name: file with .csv extenstion
    - data_handle: handle found on Kaggle site

    Returns:
    - df: Pandas dataframe with data loaded from Kaggle
    """
    # Load the latest version
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        data_handle,
        file_name
    )

    return df

def load_data():
    """Loads the data from Kaggle."""
    # Load all csvs from the data handle
    data_handle = "alifarahmandfar/continuous-groundwater-level-measurements-2023"
    df_daily = load_kaggle_data('gwl-daily.csv',data_handle)
    df_monthly = load_kaggle_data('gwl-monthly.csv',data_handle)
    quality_codes = load_kaggle_data('gwl-quality_codes.csv',data_handle)
    stations_df = load_kaggle_data('gwl-stations.csv',data_handle)

    # Change the date column to datetime
    df_daily['MSMT_DATE'] = pd.to_datetime(df_daily['MSMT_DATE'])
    df_monthly['MSMT_DATE'] = pd.to_datetime(df_monthly['MSMT_DATE'])

    return df_daily, df_monthly, quality_codes, stations_df

def get_columns(df):
    cols = df.columns
    return cols


def create_map(df: pd.DataFrame):
    """Function to create the map plot."""

    # color map for plotting
    color_map = {
            "selected": "red",
            "included": "blue",
            "excluded": "lightgray",
        }
    # figure for map plot
    fig = px.scatter_mapbox(
        df,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="STATION",
        hover_data={
            "ELEV": True,
            "WELL_DEPTH": True,
            "LATITUDE": False,  # Hide lat/lon if not useful
            "LONGITUDE": False,
        },
        mapbox_style="carto-positron",
        zoom=5,
        height=600,
        custom_data=["STATION"],
        color="highlight",
        color_discrete_map=color_map,
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def plot_station_data(df: pd.DataFrame, station_id: str):
    """Plots the data for the selected station."""
    # Locate the data for the station
    df = df.copy().loc[df.STATION == station_id]
    fig = px.scatter(
        df,
        x='MSMT_DATE',
        y='WSE',
        hover_data={
            'WSE_QC':True
        }
        )
    
    # Update figure options
    fig.update_layout(
        template="plotly_white",
        title=f"Water Level for Station {station_id}",
        xaxis=dict(showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        xaxis_title="Date",
        yaxis_title="Water Surface Elevation (ft asl)"
    )
    
    fig.update_traces(mode="lines+markers")
    return fig

def determine_elevation_from_raster(long: float, lat: float):
    """Determines the elevation from 3DEP data at a lat and long provided."""
    surface_elevation = py3dep.elevation_bycoords(
        [(long, lat)],
        crs=4326
    )

    return surface_elevation