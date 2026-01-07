"""
Any utility functions such as loading data
or editing strings, etc...
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.express as px
import pandas as pd
import py3dep
from xgboost import XGBRegressor
import joblib
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.seasonal import STL
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
import os

def load_kaggle_data(file_name, data_handle):
    """
    Load Kaggle dataset into a Pandas DataFrame.

    Parameters
    ----------
    file_name : str
        File with .csv extension to load from the dataset
    data_handle : str
        Dataset handle found on Kaggle site (format: 'username/dataset-name')

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with data loaded from Kaggle
    """
    # Load the latest version
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        data_handle,
        file_name
    )

    return df

def load_data():
    """
    Load all groundwater level datasets from Kaggle.
    
    Loads daily measurements, monthly measurements, quality codes, and station
    metadata. Automatically converts date columns to datetime format.

    Returns
    -------
    df_daily : pd.DataFrame
        Daily groundwater level measurements with datetime index
    df_monthly : pd.DataFrame
        Monthly groundwater level measurements with datetime index
    quality_codes : pd.DataFrame
        Quality code descriptions and definitions
    stations_df : pd.DataFrame
        Station metadata including location, elevation, and well depth
    """
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

def create_map(df: pd.DataFrame):
    """
    Create an interactive map plot of groundwater monitoring stations.
    
    Displays stations on a map with color-coding for selection status
    (selected, included, or excluded). Hover information includes station
    details such as elevation, well depth, and coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing station information with columns:
        - LATITUDE: station latitude
        - LONGITUDE: station longitude
        - STATION: station identifier
        - ELEV: elevation in feet
        - WELL_DEPTH: well depth in feet
        - highlight: station status ('selected', 'included', or 'excluded')

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map visualization with station markers and hover information
    """

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
        ),
        legend_title_text="Station Selection"
    )
    
    return fig

def plot_station_data(df: pd.DataFrame, station_id: str, quality_codes: pd.DataFrame):
    """
    Create a time series plot of water level data for a specific station.
    
    Plots water surface elevation over time with quality control flag information
    in hover tooltips. Data points are connected with lines and markers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing water level measurements with columns:
        - STATION: station identifier
        - MSMT_DATE: measurement date
        - WSE: water surface elevation in feet above sea level
        - WSE_QC: quality control code
    station_id : str
        Identifier for the station to plot
    quality_codes : pd.DataFrame
        DataFrame mapping quality codes to descriptions with columns:
        - QUALITY_CODE: code identifier
        - DESCRIPTION: human-readable description

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive time series plot with hover information
    """
    # Locate the data for the station
    df = df.copy().loc[df.STATION == station_id]

    # Map QC codes to descriptions
    qc_map = dict(
        zip(quality_codes["QUALITY_CODE"], quality_codes["DESCRIPTION"])
    )
    df["QC_DESC"] = df["WSE_QC"].map(qc_map)

    fig = px.scatter(
        df,
        x='MSMT_DATE',
        y='WSE',
        hover_data={
            "QC_DESC": True,
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
    """
    Query ground surface elevation from 3DEP raster data at a coordinate.
    
    Uses the USGS 3DEP (3D Elevation Program) dataset to retrieve elevation
    data at the specified latitude and longitude. Converts elevation from
    meters to feet.

    Parameters
    ----------
    long : float
        Longitude in decimal degrees (WGS84)
    lat : float
        Latitude in decimal degrees (WGS84)

    Returns
    -------
    float
        Ground surface elevation in feet, rounded to 2 decimal places
    """
    m_to_ft = 3.28
    surface_elevation = py3dep.elevation_bycoords(
        [(long, lat)],
        crs=4326
    )

    return round(surface_elevation * m_to_ft, 2)

def load_ml_model(model_name = "wl_xgb_model_basin.json"):
    """
    Load a trained XGBoost model from disk.

    Parameters
    ----------
    model_name : str, optional
        Path to the saved XGBoost model file, by default "wl_xgb_model_basin.bin"

    Returns
    -------
    xgb.XGBRegressor
        Loaded XGBoost regression model ready for predictions
    """
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "models", model_name)

    loaded_model = XGBRegressor()
    loaded_model._estimator_type = "regressor" # set estimator type to avoid error on deployment
    loaded_model.load_model(model_path)

    return loaded_model

def load_encoder(encoder_name = "station_encoder.joblib"):
    """
    Load a fitted LabelEncoder from disk.

    Parameters
    ----------
    encoder_name : str, optional
        Name of the saved encoder file, by default "station_encoder.joblib"

    Returns
    -------
    sklearn.preprocessing.LabelEncoder
        Fitted label encoder for transforming categorical labels
    """
    base_dir = os.getcwd()
    encoder_path = os.path.join(base_dir, "models", encoder_name)
    le = joblib.load(encoder_path)
    return le

def encode_station(station_id, encoder):
    """
    Encode a station ID using a fitted LabelEncoder.
    
    If the station ID is not found in the encoder's classes (i.e., not seen
    during training), returns the encoding of the first class as a fallback.

    Parameters
    ----------
    station_id : str
        Station identifier to encode
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder containing known station IDs

    Returns
    -------
    int
        Encoded station ID as an integer. Returns encoding of first class
        if station_id is unknown to the encoder.
    """
    try:
        return encoder.transform([station_id])[0]
    except ValueError:
        # Station not seen during training
        return encoder.transform([encoder.classes_[0]])[0] 
    
def create_stl_plot(station_df):
    """
    Perform Seasonal-Trend decomposition using LOESS (STL) and create plots.
    
    First applies an Unobserved Components model to smooth the water level data,
    then performs STL decomposition to separate trend, seasonal, and residual
    components. Creates three separate plots for visualization.

    Parameters
    ----------
    station_df : pd.DataFrame
        DataFrame containing water level measurements with columns:
        - MSMT_DATE: measurement date
        - WSE: water surface elevation in feet

    Returns
    -------
    fig_trend : plotly.graph_objects.Figure
        Plot showing observed data, unobserved estimation, and trend component
    fig_seasonal : plotly.graph_objects.Figure
        Plot showing the seasonal variation component
    fig_resid : plotly.graph_objects.Figure
        Plot showing the residual component after removing trend and seasonality
    """
    # Ensure station_df has a DatetimeIndex
    df_test = station_df.copy().sort_values('MSMT_DATE')
    df_test['MSMT_DATE'] = pd.to_datetime(df_test['MSMT_DATE'])
    df_test = df_test.set_index('MSMT_DATE')
    y = df_test['WSE']

    # Fit UnobservedComponents model
    mod = UnobservedComponents(y, level='local level', seasonal=12)
    res_ucm = mod.fit()

    # Get smoothed level as a Series with correct index
    filled = pd.Series(res_ucm.smoothed_state[0], index=y.index)

    # STL decomposition — monthly example: period=12
    stl = STL(filled, period=12)
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
        line=dict(color='cyan'),
        visible='legendonly'
    ))
    fig_trend.add_trace(go.Scatter(
        x=filled.index,
        y=res_stl.trend,
        mode='lines',
        name='Unobserved Estimation Trend',
        line=dict(color='cyan', dash='dash')
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
        name='Unobserved Seasonal',
        line=dict(color='cyan'),
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
        name='Unobserved Residual',
        line=dict(color='cyan'),
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
    """
    Create an empty Plotly figure with formatted axes and title.
    
    Useful for initializing plots before data is available or for
    placeholder visualizations.

    Parameters
    ----------
    title : str
        Figure title
    yaxis_title : str
        Label for the y-axis
    xaxis_title : str, optional
        Label for the x-axis, by default "Date"

    Returns
    -------
    plotly.graph_objects.Figure
        Empty figure with specified formatting
    """
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

def generate_ml_predictions(station_id=None, station_df=None, stations_df=None, monthly_df=None, model=None, 
                            click_data=None, well_depth=None):
    """
    Generate machine learning predictions for groundwater levels.
    
    Can generate predictions either for an existing station (using station_id)
    or for an arbitrary location (using click_data from map). Creates feature
    matrix including temporal, spatial, and station metadata features.

    Parameters
    ----------
    station_id : str, optional
        Station identifier for known monitoring locations, by default None
    station_df : pd.DataFrame, optional
        DataFrame containing station water level history, by default None
    stations_df : pd.DataFrame, optional
        DataFrame containing all station metadata, by default None
    monthly_df : pd.DataFrame, optional
        DataFrame containing monthly measurements for reference date, by default None
    model : xgb.XGBRegressor, optional
        Trained XGBoost model for predictions, by default None
    click_data : dict, optional
        Dictionary containing 'lat', 'lon', and optionally 'elevation' for
        arbitrary locations, by default None
    well_depth : float, optional
        Well depth in feet (used for click_data locations), by default None

    Returns
    -------
    predictions : np.ndarray
        Predicted water surface elevations in feet
    dates : pd.DatetimeIndex
        Dates corresponding to each prediction

    Notes
    -----
    Features used in prediction include:
    - Station and basin encodings
    - Temporal features (day, month, year, days since reference)
    - Cyclical time features (sin/cos of day of year and month)
    - Spatial features (elevation, latitude, longitude, well depth)
    """
    le_basin = load_encoder("basin_encoder.joblib")
    if click_data:
        # Load data from map click
        lat = click_data["lat"]
        lon = click_data["lon"]
        elevation = click_data.get("elevation", 0)

        # Start date of 2000 for spatial clicks
        start_date = pd.Timestamp("2000-01-01")
        
        station_encoded = 0  # or use mean encoding for unknown stations

        # Determine nearest station
        basin_name = nearest_station_basin(click_data, stations_df)
        basin_encoded = encode_station(basin_name, le_basin)
    else:
        # Load label encoder for station names
        le = load_encoder()
    
        # Get station info
        station_info = stations_df.loc[stations_df.STATION == station_id].iloc[0]
        lat = station_info['LATITUDE']
        lon = station_info['LONGITUDE']
        elevation = station_info['ELEV']
        well_depth = station_info['WELL_DEPTH']
        station_encoded = encode_station(station_id, le)
        start_date = station_df['MSMT_DATE'].min()
        basin_name = station_info['BASIN_NAME']
        basin_encoded = encode_station(basin_name, le_basin)
    
    # Create date range
    end_date = pd.Timestamp.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    ref_date = monthly_df["MSMT_DATE"].min()

    days_since_ref = (dates - ref_date).days.values
    day_of_year = dates.dayofyear.values
    month = dates.month.values
    year = dates.year.values
    day = dates.day.values

    sin_doy = np.sin(2 * np.pi * day_of_year / 365)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    n = len(dates)

    # --- Feature matrix ---
    X = np.column_stack([
        np.full(n, station_encoded),
        np.full(n, basin_encoded),
        day,
        month,
        year,
        days_since_ref,
        sin_doy,
        cos_doy,
        sin_month,
        cos_month,
        np.full(n, elevation),
        np.full(n, lat),
        np.full(n, lon),
        np.full(n, well_depth),
    ])

    predictions = model.predict(X)
    
    return predictions, dates

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Computes the shortest distance over the Earth's surface between two
    geographic coordinates.

    Parameters
    ----------
    lat1 : float
        Latitude of first point in decimal degrees
    lon1 : float
        Longitude of first point in decimal degrees
    lat2 : float
        Latitude of second point in decimal degrees
    lon2 : float
        Longitude of second point in decimal degrees

    Returns
    -------
    float
        Distance between the two points in miles
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
    """
    Add a red marker to a map figure at a clicked location.
    
    Adds a prominent marker with hover information to indicate a selected
    location on the map.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing map figure to add marker to
    click_data : dict
        Dictionary containing location information with keys:
        - 'lat': latitude in decimal degrees
        - 'lon': longitude in decimal degrees
        - 'elevation': elevation in feet (optional)

    Returns
    -------
    None
        Modifies the figure in-place
    """
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

def generate_gsp(data : pd.DataFrame, date_range):
    """
    Generate Gaussian Process predictions for water level time series.
    
    Fits a Gaussian Process Regressor with seasonal and trend kernels to
    model water level behavior. Uses a combination of exponential sine
    squared (for seasonality), RBF (for trends), and white noise kernels.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing water level measurements with columns:
        - MSMT_DATE: measurement dates (datetime)
        - WSE: water surface elevation in feet
    date_range : pd.DatetimeIndex
        Dates for which to generate predictions

    Returns
    -------
    gsp_pred : np.ndarray
        Mean predictions of water surface elevation
    std_pred : np.ndarray
        Standard deviation (uncertainty) of predictions

    Notes
    -----
    The model is trained on all available data and uses three kernel components:
    - ExpSineSquared: captures annual seasonality (365-day period)
    - RBF: captures long-term trends
    - WhiteKernel: accounts for measurement noise
    """
    # Clean data
    data = data.copy()[['MSMT_DATE', 'WSE']].dropna()
    # Generate the train set
    t0 = data["MSMT_DATE"].min() # Need to enter into training as timedelta
    X = ((data["MSMT_DATE"] - t0) / np.timedelta64(1, "D")).to_numpy().reshape(-1, 1)
    y = data['WSE'].to_numpy()

    n_split = int(1*len(X)) # Split so the data is trained on the first 80% of waterlevels
    x_train = X[:n_split]
    y_train = y[:n_split]

    # Create Gaussian Process model
    kernel = (
        1.0 * ExpSineSquared(periodicity=365.0, length_scale=30) +
        1.0 * RBF(length_scale=100) +
        WhiteKernel(noise_level=0.01)
    )
    gsp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    model = gsp.fit(x_train, y_train)

    # Prediction for the entire dataset
    X_pred = ((date_range - t0) / np.timedelta64(1, "D")).to_numpy().reshape(-1, 1)
    gsp_pred, std_pred = model.predict(X_pred, return_std=True)

    return gsp_pred, std_pred

def add_gsp_plot(fig, dates_full, mean_pred, std_pred):
    """
    Add Gaussian Process predictions with confidence intervals to a plot.
    
    Adds the mean prediction line and a shaded region representing the 95%
    confidence interval to an existing figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing figure to add GSP predictions to
    dates_full : pd.DatetimeIndex or array-like
        Date values for x-axis
    mean_pred : np.ndarray
        Mean predicted water surface elevations
    std_pred : np.ndarray
        Standard deviation of predictions for uncertainty bounds

    Returns
    -------
    None
        Modifies the figure in-place

    Notes
    -----
    The confidence interval is calculated as mean ± 1.96 * std, representing
    approximately 95% confidence bounds assuming normality.
    """
    # Add mean prediction
    fig.add_trace(go.Scatter(
        x=dates_full,
        y=mean_pred,
        mode='lines',
        name='GSP Mean',
        line=dict(color='green', dash='dash', width=2),
        hovertemplate=(
            "%{y:.2f} ft"
        )
    ))

    # Upper bound
    fig.add_trace(go.Scatter(
        x=dates_full,
        y=mean_pred + 1.96 * std_pred,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Lower bound (fills to previous trace)
    fig.add_trace(go.Scatter(
        x=dates_full,
        y=mean_pred - 1.96 * std_pred,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(235, 216, 190, 0.3)", 
        line=dict(width=0),
        name="95% confidence interval",
        hoverinfo="skip"
    ))

def nearest_station_basin(click_data, df_stations):
    """
    Find the nearest monitoring station to a clicked point and return its basin.
    
    Uses the Haversine formula to calculate great circle distances from the
    clicked location to all stations, then identifies the nearest one.

    Parameters
    ----------
    click_data : dict
        Dictionary containing clicked location with keys:
        - 'lat': latitude in decimal degrees
        - 'lon': longitude in decimal degrees
    df_stations : pd.DataFrame
        DataFrame containing station information with columns:
        - STATION: station identifier
        - LATITUDE: station latitude
        - LONGITUDE: station longitude
        - BASIN_NAME: name of the basin/watershed

    Returns
    -------
    basin_name : str
        BASIN_NAME of the nearest station
    station_name : str
        Name/identifier of the nearest station
    distance_m : float
        Distance to the nearest station in meters
    """
    # Earth radius in meters
    R = 6371000  

    # Extract lat and lon
    lat=click_data["lat"]
    lon=click_data["lon"]

    # Convert degrees to radians
    lat_click_rad = np.radians(lat)
    lon_click_rad = np.radians(lon)
    lat_stations_rad = np.radians(df_stations['LATITUDE'].values)
    lon_stations_rad = np.radians(df_stations['LONGITUDE'].values)

    # Haversine formula
    dlat = lat_stations_rad - lat_click_rad
    dlon = lon_stations_rad - lon_click_rad
    a = np.sin(dlat/2)**2 + np.cos(lat_click_rad) * np.cos(lat_stations_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c  # meters

    idx = np.argmin(distances)
    nearest_station = df_stations.iloc[idx]

    return nearest_station['BASIN_NAME'], nearest_station['STATION'], distances[idx]

def add_to_stl(fig_trend, fig_seasonal, fig_resid, dates_trend, predictions_trend, name='XGB', color='red'):
    """
    Perform STL decomposition on predictions and add components to plots.
    
    This function takes time series predictions, performs Seasonal-Trend 
    decomposition using LOESS (STL), and adds the decomposed components 
    (trend, seasonal, residual) to existing Plotly figures.
    
    Parameters
    ----------
    fig_trend : plotly.graph_objects.Figure
        Figure for trend component. Will have both the raw predictions 
        and extracted trend added as traces.
    fig_seasonal : plotly.graph_objects.Figure
        Figure for seasonal component. Will have the seasonal pattern added.
    fig_resid : plotly.graph_objects.Figure
        Figure for residual component. Will have the residuals added.
    dates_trend : pd.DatetimeIndex or array-like
        Dates corresponding to the predictions
    predictions_trend : np.ndarray or array-like
        Time series predictions to decompose
    name : str, optional
        Name prefix for traces in legend (e.g., 'XGB', 'GPR'), by default 'XGB'
    color : str, optional
        Color for all traces from this decomposition, by default 'red'
    
    Returns
    -------
    None
        Modifies the figures in-place by adding traces
    
    Notes
    -----
    - The raw predictions are added to the trend plot but hidden by default 
      (visible='legendonly'). Users can click the legend to show them.
    - STL decomposition uses a period of 12 (monthly seasonality)
    - The trend component uses a dashed line to distinguish from raw predictions
    - All figures are modified in-place; no return value needed
    
    Examples
    --------
    >>> # After creating base STL plots for observed data
    >>> fig_trend, fig_seasonal, fig_resid = utils.create_stl_plot(station_data)
    >>> 
    >>> # Add XGBoost predictions to the plots
    >>> add_to_stl(
    ...     fig_trend, fig_seasonal, fig_resid,
    ...     dates=ml_dates,
    ...     predictions_trend=ml_predictions,
    ...     name='XGB',
    ...     color='red'
    ... )
    >>> 
    >>> # Add Gaussian Process predictions to the same plots
    >>> add_to_stl(
    ...     fig_trend, fig_seasonal, fig_resid,
    ...     dates=gp_dates,
    ...     predictions_trend=gp_predictions,
    ...     name='GPR',
    ...     color='green'
    ... )
    
    See Also
    --------
    statsmodels.tsa.seasonal.STL : The STL decomposition implementation
    utils.create_stl_plot : Creates the base STL plots for observed data
    """
    
    # ---- Perform STL decomposition on ML predictions ----
    stl_ml = STL(predictions_trend, period=12)
    res_stl_ml = stl_ml.fit()

    # Add data to trend plot
    fig_trend.add_trace(go.Scatter(
        x=dates_trend,
        y=predictions_trend,
        mode='lines',
        name=f'{name} Prediction',
        line=dict(color=color, width=2),
        hovertemplate="%{y:.2f}",
        visible='legendonly'
    ))

    # Add trend component
    fig_trend.add_trace(go.Scatter(
        x=dates_trend,
        y=res_stl_ml.trend,
        mode='lines',
        name=f'{name} Trend',
        line=dict(color=color, dash='dash')
    ))

    # Add seasonal component
    fig_seasonal.add_trace(go.Scatter(
        x=dates_trend,
        y=res_stl_ml.seasonal,
        mode='lines',
        name=f'{name} Seasonal',
        line=dict(color=color),
        hovertemplate="%{y:.2f}"
    ))

    # Add residual component
    fig_resid.add_trace(go.Scatter(
        x=dates_trend,
        y=res_stl_ml.resid,
        mode='lines',
        name=f'{name} Residual',
        line=dict(color=color),
        hovertemplate="%{y:.2f}"
    ))