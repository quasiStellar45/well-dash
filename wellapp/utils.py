"""
Any utility functions such as loading data
or editing strings, etc...
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.express as px
import pandas as pd
import os

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
    data_handle = "alifarahmandfar/continuous-groundwater-level-measurements-2023"
    df_daily = load_kaggle_data('gwl-daily.csv',data_handle)
    df_monthly = load_kaggle_data('gwl-monthly.csv',data_handle)
    quality_codes = load_kaggle_data('gwl-quality_codes.csv',data_handle)
    stations_df = load_kaggle_data('gwl-stations.csv',data_handle)

    return df_daily, df_monthly, quality_codes, stations_df

def get_columns():
    df = load_data()
    return df.columns


def create_map(df: pd.DataFrame):
    """Function to create the map plot."""
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
        zoom=1,
        height=600
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig