"""
Any utility functions such as loading data
or editing strings, etc...
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.express as px
import pandas as pd
import py3dep
import xgboost as xgb
import joblib
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.seasonal import STL
import datetime 

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
            "LATITUDE": True,  # Hide lat/lon if not useful
            "LONGITUDE": True,
        },
        mapbox_style="carto-positron",
        zoom=5,
        height=600,
        custom_data=["STATION"],
        color="highlight",
        color_discrete_map=color_map,
    )

    # Edit hover box style
    for trace in fig.data:
        trace.hoverlabel = dict(
            bgcolor="white",
            bordercolor=trace.marker.color,
            font=dict(color="black")
        )

    # Update hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Latitude: %{lat:.4f}<br>" +
            "Longitude: %{lon:.4f}<br>" +
            "Elevation: %{customdata[1]:.0f} ft<br>" +
            "Well Depth: %{customdata[2]:.0f} ft<br>" +
            "<extra></extra>"
            )
    )

    # Update layout to move legend to bottom
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
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
            'WSE_QC': True,
            'MSMT_DATE': False,  # Hide to avoid duplication
            'WSE': False  # Hide to avoid duplication
        },
    )

    # Change trace type and name for legend
    fig.update_traces(
        mode="lines+markers", 
        name="Water Level Data",
        showlegend=True,
        hovertemplate=(
            "Water Level: %{y:.2f} ft<br>" +
            "QC Flag: %{customdata[0]}<br>" +
            "<extra></extra>"
        )
    )

    # Update figure options
    fig.update_layout(
        template="plotly_white",
        title=f"Water Level for Station {station_id}",
        xaxis=dict(showline=True, linewidth=1, linecolor='black', title='Date'),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', title="Water Surface Elevation (ft asl)"),
        showlegend=True,
        hovermode='x unified'
    )

    return fig

def determine_elevation_from_raster(long: float, lat: float):
    """Determines the elevation from 3DEP data at a lat and long provided."""
    m_to_ft = 3.28
    surface_elevation = py3dep.elevation_bycoords(
        [(long, lat)],
        crs=4326
    )

    return round(surface_elevation * m_to_ft, 2)

def load_ml_model(model_name = "wl_xgb_model_2.json"):
    """Load a XGBoost ml model."""
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model(model_name)
    return loaded_model

def load_encoder(encoder_path = "encoder.joblib"):
    """Loads the encoder for station labels."""
    le = joblib.load(encoder_path)
    return le

def encode_station(station_id, encoder):
    """
    Encode a station ID using the fitted LabelEncoder.
    Returns 0 if station is not in the encoder's classes.
    """
    try:
        return encoder.transform([station_id])[0]
    except ValueError:
        # Station not seen during training
        return encoder.transform([encoder.classes_[0]])[0] 
    
def create_stl_plot(station_df):
    """
    Creates a plot of seasonal variation for the data and ml model
    
    :param station_df: df for the station
    """
    # Ensure station_df has a DatetimeIndex
    df_test = station_df.copy().sort_values('MSMT_DATE')
    df_test['MSMT_DATE'] = pd.to_datetime(df_test['MSMT_DATE'])
    df_test = df_test.set_index('MSMT_DATE')
    y = df_test['WSE']

    # Fit UnobservedComponents model
    mod = UnobservedComponents(y, level='local level', seasonal=13)
    res_ucm = mod.fit()

    # Get smoothed level as a Series with correct index
    filled = pd.Series(res_ucm.smoothed_state[0], index=y.index)

    # STL decomposition — monthly example: period=13
    stl = STL(filled, period=13)
    res_stl = stl.fit()

    # Observed + trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filled.index,
        y=y,
        mode='lines+markers',
        name='Observed',
        line=dict(color='blue')
    ))
    fig_trend.add_trace(go.Scatter(
        x=filled.index,
        y=filled,
        mode='lines',
        name='Unobserved Estimation',
        line=dict(color='green', dash='dash')
    ))
    fig_trend.add_trace(go.Scatter(
        x=filled.index,
        y=res_stl.trend,
        mode='lines',
        name='Trend',
        line=dict(color='orange', dash='dash')
    ))
    fig_trend.update_layout(
        title='Trend Component',
        xaxis_title='Date',
        yaxis_title="Water Surface Elevation (ft asl)",
        template='plotly_white',
        hovermode='x unified'
    )

    # Seasonal component
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(
        x=filled.index,
        y=res_stl.seasonal,
        mode='lines',
        name='Seasonal',
        line=dict(color='magenta'),
        hovertemplate=(
            "%{y:.2f}"
        )
    ))
    fig_seasonal.update_layout(
        title='Seasonal Component',
        xaxis_title='Date',
        yaxis_title="Seasonal Variation (ft)",
        template='plotly_white',
        hovermode='x unified'
    )

    # Residual component
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=filled.index,
        y=res_stl.resid,
        mode='lines',
        name='Residual',
        line=dict(color='brown'),
        hovertemplate=(
            "%{y:.2f}"
        )
    ))
    fig_resid.update_layout(
        title='Residual Component',
        xaxis_title='Date',
        yaxis_title="Residual (ft)",
        template='plotly_white',
        hovermode='x unified'
    )

    return fig_trend, fig_seasonal, fig_resid

def create_empty_fig(title, yaxis_title, xaxis_title="Date"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis=dict(showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig

def generate_ml_predictions(station_id, station_df, stations_df, monthly_df, model):
    """Generate ML predictions for a station."""
    le = load_encoder()
    
    # Get station info
    station_info = stations_df.loc[stations_df.STATION == station_id].iloc[0]
    lat = station_info['LATITUDE']
    lon = station_info['LONGITUDE']
    elevation = station_info['ELEV']
    well_depth = station_info['WELL_DEPTH']
    station_encoded = encode_station(station_id, le)
    
    # Create date range
    start_date = station_df['MSMT_DATE'].min()
    end_date = pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    predictions = []
    dates = []
    ref_date = monthly_df["MSMT_DATE"].min()
    
    for date in date_range:
        days_since_ref = (date - ref_date).days
        day_of_year = date.dayofyear
        
        X = np.array([[
            station_encoded,
            date.day,
            date.month,
            date.year,
            days_since_ref,
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            np.sin(2 * np.pi * date.month / 12),
            np.cos(2 * np.pi * date.month / 12),
            elevation,
            lat,
            lon,
            well_depth
        ]])
        
        pred = model.predict(X)[0]
        predictions.append(pred)
        dates.append(date)
    
    return predictions, dates

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula.
    Returns distance in miles.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    km_to_miles = 1.609344
    return c * r / km_to_miles

def add_map_marker(fig, click_data):
    fig.add_trace(go.Scattermapbox(
        lat=[click_data["lat"]],
        lon=[click_data["lon"]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='circle'),
        name='Selected Location',
        showlegend=True,
        hovertemplate=(
            f"<b>Selected Location</b><br>"
            f"Latitude: {click_data['lat']:.4f}<br>"
            f"Longitude: {click_data['lon']:.4f}<br>"
            f"Elevation: {click_data.get('elevation', 'N/A')} ft<br>"
            "<extra></extra>"
        )
    ))

def generate_ml_prediction(station_encoded, date: datetime.date, elevation: float, lat: float, lon: float, 
                           well_depth: float, start_date: datetime.date, ref_date: datetime.date, model):
    """
    Docstring for generate_ml_prediction
    
    :param station_encoded: Encoded station name.
    :param date: Date for measurement
    :type date: datetime.date
    :param elevation: Surface elevation of well.
    :type elevation: float
    :param lat: Latitude of the well.
    :type lat: float
    :param lon: Longitude of the well.
    :type lon: float
    :param well_depth: Depth of the well.
    :type well_depth: float
    :param start_date: Start date for prediction
    :type start_date: datetime.date
    :param ref_date: Reference date for ml model
    :type ref_date: datetime.date
    :param model: ml model.
    """
    end_date = pd.Timestamp.now()
    date_range_full = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    predictions_full = []
    dates_full = []
    
    for date in date_range_full:
        days_since_ref = (date - ref_date).days
        day_of_year = date.dayofyear
        
        # Create feature vector matching your training columns
        X = np.array([[
            station_encoded,
            date.day,
            date.month,
            date.year,
            days_since_ref,
            np.sin(2 * np.pi * day_of_year / 365.25),  # day_sin
            np.cos(2 * np.pi * day_of_year / 365.25),  # day_cos
            np.sin(2 * np.pi * date.month / 12),        # month_sin
            np.cos(2 * np.pi * date.month / 12),        # month_cos
            elevation,
            lat,
            lon,
            well_depth
        ]])
        
        pred = model.predict(X)[0]
        predictions_full.append(pred)
        dates_full.append(date)

        return predictions_full, dates_full
