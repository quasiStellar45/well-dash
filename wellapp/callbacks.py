# wellapp/callbacks.py
"""
Dash callback functions for the groundwater monitoring application.

This module registers all interactive callbacks that handle user interactions
with the dashboard, including map clicks, station selection, data visualization,
and spatial predictions.
"""

from dash import Input, Output, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import wellapp.utils as utils

# =========================================================================
# DATA LOADING (executed once)
# =========================================================================
df_daily, df_monthly, quality_codes, stations_df = utils.load_data()
model = utils.load_ml_model()


def register_callbacks(app):
    """
    Register all Dash callbacks for the application.
    
    This function sets up the interactive components of the dashboard by
    connecting user inputs (clicks, selections, etc.) to outputs (plots,
    maps, information displays).
    
    Parameters
    ----------
    app : dash.Dash
        The Dash application instance to register callbacks with
    
    Notes
    -----
    Callbacks are organized by functionality:
    1. Map visualization and station selection
    2. Water level plotting with ML predictions
    3. Spatial prediction interface
    4. Nearby stations analysis
    """

    # =========================================================================
    # CALLBACK 1: Update main station map
    # =========================================================================
    @app.callback(
        Output("map-plot", "figure"),
        Input("freq-dropdown", "value"),
        Input("selected-station", "data"),
        Input("map-plot", "relayoutData")   
    )
    def update_map(freq, selected_station, relayout):
        """
        Update the station map based on data frequency and selection.
        
        Highlights stations based on:
        - Selected station (red)
        - Stations with data at chosen frequency (blue)
        - Stations without data at chosen frequency (gray)
        
        Preserves map zoom and center when user pans/zooms.
        
        Parameters
        ----------
        freq : str
            Data frequency selected ('daily' or 'monthly')
        selected_station : str or None
            Currently selected station ID
        relayout : dict or None
            Map layout data containing zoom/center if user has interacted
        
        Returns
        -------
        plotly.graph_objects.Figure
            Updated map figure with station markers
        """
        # Filter stations based on data availability at selected frequency
        if freq == "daily":
            allowed = set(df_daily.STATION.unique())
        elif freq == "monthly":
            allowed = set(df_monthly.STATION.unique())

        # Assign highlight status to each station
        stations_df["highlight"] = stations_df["STATION"].apply(
            lambda s: "selected" if s == selected_station
            else ("included" if s in allowed else "excluded")
        )

        # Create the base map
        fig = utils.create_map(stations_df)

        # Preserve user's zoom and center position if they've interacted with the map
        if relayout is not None:
            if "mapbox.zoom" in relayout:
                fig.update_layout(mapbox={"zoom": relayout["mapbox.zoom"]})
            if "mapbox.center" in relayout:
                fig.update_layout(mapbox={"center": relayout["mapbox.center"]})

        return fig
    
    # =========================================================================
    # CALLBACK 2: Store selected station from map click
    # =========================================================================
    @app.callback(
        Output("selected-station", "data"),
        Input("map-plot", "clickData")
    )
    def store_selected_station(clickData):
        """
        Store the station ID when user clicks on a station marker.
        
        This callback extracts the station ID from map click events and
        stores it in a dcc.Store component for use by other callbacks.
        
        Parameters
        ----------
        clickData : dict or None
            Click event data from the map, contains point information
        
        Returns
        -------
        str or None
            Station ID if a station marker was clicked, None otherwise
        """
        if not clickData:
            return None

        # Extract station ID from custom data embedded in the marker
        point = clickData["points"][0]
        if "customdata" in point and point["customdata"]:
            return point["customdata"][0]

        # If click was on empty map (not a marker), don't change selection
        return None
    
    # =========================================================================
    # CALLBACK 3: Update water level plot and STL decomposition
    # =========================================================================
    @app.callback(
        Output("wl-plot", "figure"),
        Output("stl-trend", "figure"),
        Output("stl-seasonal", "figure"),
        Output("stl-resid", "figure"),
        Input("selected-station", "data"),
        Input("freq-dropdown", "value"),
        prevent_initial_call=False
    )
    def update_wl_plot_with_ml(station_id, freq):
        """
        Update water level plots with observed data, ML predictions, and STL decomposition.
        
        This callback generates four plots:
        1. Main water level plot with observed data, XGBoost predictions, and Gaussian Process
        2. Trend component from STL decomposition
        3. Seasonal component from STL decomposition
        4. Residual component from STL decomposition
        
        Parameters
        ----------
        station_id : str or None
            Selected station identifier
        freq : str
            Data frequency ('daily' or 'monthly')
        
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Main water level plot with predictions
        fig_trend : plotly.graph_objects.Figure
            Trend component plot
        fig_seasonal : plotly.graph_objects.Figure
            Seasonal component plot
        fig_resid : plotly.graph_objects.Figure
            Residual component plot
        
        Notes
        -----
        - XGBoost predictions extend to the current date
        - Gaussian Process predictions include uncertainty bounds
        - STL decomposition uses monthly data only
        - Returns empty plots if no station is selected or data is unavailable
        """
        # Return empty plots if no station is selected
        if not station_id:
            fig = utils.create_empty_fig(
                "Select a station to plot waterlevel...", 
                "Water Surface Elevation (ft asl)"
            )
            fig_trend = utils.create_empty_fig(
                "Trend Component", 
                "Water Surface Elevation (ft asl)"
            )
            fig_seasonal = utils.create_empty_fig(
                "Seasonal Component", 
                "Seasonal Variation (ft)"
            )
            fig_resid = utils.create_empty_fig(
                "Residual Component", 
                "Residual (ft)"
            )
            return fig, fig_trend, fig_seasonal, fig_resid
                
        # Select appropriate dataframe based on frequency
        if freq == "daily":
            df = df_daily
        elif freq == "monthly":
            df = df_monthly
        
        # Get data for selected station
        station_df = df.loc[df.STATION == station_id]
        
        if not station_df.empty:
            # ---- Plot observed data ----
            fig = utils.plot_station_data(df, station_id, quality_codes)
            
            # We use monthly data for computation for speed
            station_df_monthly = df_monthly.loc[
                df_monthly.STATION == station_id, 
                ['MSMT_DATE', 'WSE']
            ]

            # ---- Generate XGBoost predictions (extends to present) ----
            predictions_full, dates_full = utils.generate_ml_predictions(
                station_id, station_df, stations_df, df_monthly, model
            )
            
            # ---- Add XGBoost prediction to main plot ----
            fig.add_trace(go.Scatter(
                x=dates_full,
                y=predictions_full,
                mode='lines',
                name='XGB Prediction',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate="%{y:.2f} ft"
            ))

            # ---- Add Gaussian Process prediction with uncertainty ----
            try:
                mean_pred, std_pred = utils.generate_gsp(station_df_monthly, dates_full)
                utils.add_gsp_plot(fig, dates_full, mean_pred, std_pred)
            except ValueError:
                # GSP may fail if insufficient data
                pass

            # ---- Create STL decomposition plots ----
            try:
                fig_trend, fig_seasonal, fig_resid = utils.create_stl_plot(
                    station_df_monthly
                )
            except ValueError:
                # Return empty plots if STL decomposition fails
                fig = utils.create_empty_fig(
                    f"No data for {station_id} for this data frequency.", 
                    "Water Surface Elevation (ft asl)"
                )
                fig_trend = utils.create_empty_fig(
                    "Trend Component", 
                    "Water Surface Elevation (ft asl)"
                )
                fig_seasonal = utils.create_empty_fig(
                    "Seasonal Component", 
                    "Seasonal Variation (ft)"
                )
                fig_resid = utils.create_empty_fig(
                    "Residual Component", 
                    "Residual (ft)"
                )
                return fig, fig_trend, fig_seasonal, fig_resid
            
            # ---- Generate predictions for trend plot (only to last data point) ----
            last_data_date = station_df_monthly['MSMT_DATE'].max()
            date_range_trend = pd.date_range(
                start=dates_full[0], 
                end=last_data_date, 
                freq='ME'
            )
                
            # Truncate predictions to match trend date range
            predictions_trend = predictions_full[0:len(date_range_trend)]
            dates_trend = dates_full[0:len(date_range_trend)]
            
            # ---- Add XGBoost prediction to trend plot ----
            utils.add_to_stl(fig_trend, fig_seasonal, fig_resid, dates_trend, predictions_trend)

            # Truncate GSP prediction to data limits
            gsp_trend = mean_pred[0:len(date_range_trend)]

            # ---- Add GSP prediction to trend plot ----
            utils.add_to_stl(fig_trend, fig_seasonal, fig_resid, dates_trend, gsp_trend, 'GSP', 'green')
            
            return fig, fig_trend, fig_seasonal, fig_resid
            
        # Handle case where station has no data
        else:
            fig = utils.create_empty_fig(
                f"No data for {station_id} for this data frequency.", 
                "Water Surface Elevation (ft asl)"
            )
            fig_trend = utils.create_empty_fig(
                "Trend Component", 
                "Water Surface Elevation (ft asl)"
            )
            fig_seasonal = utils.create_empty_fig(
                "Seasonal Component", 
                "Seasonal Variation (ft)"
            )
            fig_resid = utils.create_empty_fig(
                "Residual Component", 
                "Residual (ft)"
            )
            return fig, fig_trend, fig_seasonal, fig_resid
    
    # =========================================================================
    # CALLBACK 4: Update spatial prediction map
    # =========================================================================
    @app.callback(
        Output("spatial-map", "figure"),
        Input("spatial-click-location", "data"),
        prevent_initial_call=False
    )
    def update_spatial_map(click_data):
        """
        Create an interactive map for spatial predictions with clickable grid.
        
        Displays:
        - Reference station markers (gray, background)
        - Invisible clickable grid covering the map area
        - Red marker at clicked location (if any)
        
        Parameters
        ----------
        click_data : dict or None
            Dictionary containing clicked location info:
            - 'lat': latitude
            - 'lon': longitude
            - 'elevation': ground surface elevation
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive map with clickable grid and markers
        """
        fig = go.Figure()
        
        # ---- Add reference station markers (background layer) ----
        fig.add_trace(go.Scattermapbox(
            lat=stations_df["LATITUDE"],
            lon=stations_df["LONGITUDE"],
            mode='markers',
            marker=dict(size=6, color='lightgray', opacity=0.4),
            name='Reference Stations',
            showlegend=True,
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Latitude: %{lat:.4f}<br>"
                "Longitude: %{lon:.4f}<br>"
                "<extra></extra>"
            ),
            hovertext=stations_df["STATION"]
        ))
        
        # ---- Create invisible clickable grid ----
        # Calculate map bounds with padding
        lat_min, lat_max = stations_df["LATITUDE"].min(), stations_df["LATITUDE"].max()
        lon_min, lon_max = stations_df["LONGITUDE"].min(), stations_df["LONGITUDE"].max()
        
        pad_lat = max(0.5, (lat_max - lat_min) * 0.1)
        pad_lon = max(0.5, (lon_max - lon_min) * 0.1)
        
        # Create grid of invisible but clickable points
        lats = np.linspace(lat_min - pad_lat, lat_max + pad_lat, 50)
        lons = np.linspace(lon_min - pad_lon, lon_max + pad_lon, 50)
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        
        fig.add_trace(go.Scattermapbox(
            lat=lat_grid.flatten(),
            lon=lon_grid.flatten(),
            mode='markers',
            marker=dict(size=15, opacity=0),  # Invisible but clickable
            showlegend=False,
            hoverinfo='none',
            name='clickable_background'
        ))
        
        # ---- Add marker for clicked location (foreground layer) ----
        if click_data:
            utils.add_map_marker(fig, click_data)
        
        # ---- Configure map layout ----
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5
            ),
            margin={"r":0, "t":0, "l":0, "b":0},
            showlegend=True,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    # =========================================================================
    # CALLBACK 5: Capture spatial map clicks and get elevation
    # =========================================================================
    @app.callback(
        Output("spatial-click-location", "data"),
        Input("spatial-map", "clickData"),
        prevent_initial_call=True
    )
    def capture_spatial_click(click_data):
        """
        Process map clicks to extract location and query elevation data.
        
        When user clicks on the spatial prediction map, this callback:
        1. Extracts latitude and longitude from the click
        2. Queries 3DEP for ground surface elevation
        3. Stores the location data for use by other callbacks
        
        Parameters
        ----------
        click_data : dict or None
            Click event data from the map
        
        Returns
        -------
        dict or None
            Location data containing:
            - 'lat': latitude in decimal degrees
            - 'lon': longitude in decimal degrees
            - 'elevation': ground surface elevation in feet
            Returns None if no click occurred
        """
        if not click_data:
            return None
        
        # Extract coordinates from click event
        point = click_data["points"][0]
        lat = point["lat"]
        lon = point["lon"]
        
        # Query elevation from 3DEP raster data
        elevation = utils.determine_elevation_from_raster(lon, lat)
        
        return {
            "lat": lat,
            "lon": lon,
            "elevation": elevation
        }
    
    # =========================================================================
    # CALLBACK 6: Display clicked location information
    # =========================================================================
    @app.callback(
        Output("click-info", "children"),
        Input("spatial-click-location", "data"),
        prevent_initial_call=False
    )
    def display_click_info(click_data):
        """
        Display formatted information about the clicked location.
        
        Shows either:
        - Prompt to click on map (if no location selected)
        - Location details with coordinates and elevation (if location selected)
        
        Parameters
        ----------
        click_data : dict or None
            Stored location data from map click
        
        Returns
        -------
        dash.html.Div
            Formatted HTML div with location information or prompt
        """
        # Show prompt if no location is selected
        if not click_data:
            return html.Div(
                "👆 Click on the map to select a location",
                style={
                    "padding": "12px 16px",
                    "backgroundColor": "#f3f4f6",
                    "borderRadius": "8px",
                    "fontSize": "14px",
                    "color": "#6b7280",
                    "border": "1px dashed #d1d5db"
                }
            )
        
        # Show location details if selected
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "✓ ", 
                            style={"color": "#10b981", "fontWeight": "700"}
                        ),
                        html.Span(
                            "Location Selected", 
                            style={"fontWeight": "600", "color": "#111827"}
                        )
                    ],
                    style={"marginBottom": "8px"}
                ),
                html.Div(
                    [
                        html.Div(
                            f"Latitude: {click_data['lat']:.4f}°", 
                            style={"fontSize": "13px", "color": "#4b5563"}
                        ),
                        html.Div(
                            f"Longitude: {click_data['lon']:.4f}°", 
                            style={"fontSize": "13px", "color": "#4b5563"}
                        ),
                        html.Div(
                            f"Elevation: {click_data.get('elevation', 'N/A')} ft", 
                            style={"fontSize": "13px", "color": "#4b5563"}
                        )
                    ]
                )
            ],
            style={
                "padding": "12px 16px",
                "backgroundColor": "#ecfdf5",
                "borderRadius": "8px",
                "border": "1px solid #10b981"
            }
        )
    
    # =========================================================================
    # CALLBACK 7: Generate spatial prediction for clicked location
    # =========================================================================
    @app.callback(
        Output("spatial-prediction-plot", "figure"),
        Input("spatial-click-location", "data"),
        Input("well-depth-input", "value"),
        prevent_initial_call=False
    )
    def update_spatial_prediction(click_data, well_depth):
        """
        Generate XGBoost predictions for an arbitrary spatial location.
        
        Creates a time series prediction of water levels from 2000 to present
        for any location on the map, using:
        - Location coordinates and elevation
        - User-specified well depth
        - Basin information from nearest station
        
        Parameters
        ----------
        click_data : dict or None
            Clicked location data with lat, lon, and elevation
        well_depth : float or None
            User-specified well depth in feet
        
        Returns
        -------
        plotly.graph_objects.Figure
            Time series plot of predicted water levels or empty plot if
            no location is selected
        """
        # Return empty plot if no location is selected
        if not click_data:
            return utils.create_empty_fig(
                "Click on the map to generate predictions",
                "Water Surface Elevation (ft asl)"
            )
        
        # Extract location data
        lat = click_data["lat"]
        lon = click_data["lon"]
        
        # Generate ML predictions for the clicked location
        predictions, dates = utils.generate_ml_predictions(
            monthly_df=df_monthly, 
            model=model, 
            click_data=click_data, 
            well_depth=well_depth, 
            stations_df=stations_df
        )
        
        # Create prediction plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name='XGB Prediction',
            line=dict(color='#dc2626', width=2)
        ))
        
        fig.update_layout(
            template="plotly_white",
            title=f"Predicted Water Level at ({lat:.4f}°, {lon:.4f}°)",
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title="Date"
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
        
        return fig
    
    # =========================================================================
    # CALLBACK 8: Plot water levels from nearby stations
    # =========================================================================
    @app.callback(
        Output("nearby-stations-plot", "figure"),
        Input("spatial-click-location", "data")
    )
    def update_nearby_stations_plot(click_location):
        """
        Plot water level time series from stations near the clicked location.
        
        Finds stations within a specified radius of the clicked point and
        displays their historical water level data for context and comparison
        with spatial predictions.
        
        Parameters
        ----------
        click_location : dict or None
            Clicked location with lat, lon, and elevation
        
        Returns
        -------
        plotly.graph_objects.Figure
            Multi-line time series plot showing water levels from up to 5
            nearest stations within 20 miles, or empty plot if no click
        
        Notes
        -----
        - Maximum of 5 stations displayed for readability
        - Stations sorted by distance (nearest first)
        - Different colors assigned to each station
        - Hover shows station details including distance and well info
        """
        # Return empty plot with prompt if no location clicked
        if not click_location:
            fig = go.Figure()
            fig.update_layout(
                title="Click on the map to see nearby station water levels",
                template="plotly_white",
                height=500,
                xaxis_title="Date",
                yaxis_title="Water Level (ft)",
                font=dict(family="Arial, sans-serif", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="white"
            )
            return fig
        
        # Extract clicked coordinates
        clicked_lat = click_location['lat']
        clicked_lon = click_location['lon']
        
        # Select monthly only
        df = df_monthly
        
        # Get unique stations with their coordinates
        stations = stations_df[['STATION', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
        
        # Calculate distance from clicked location to each station
        stations['distance_miles'] = stations.apply(
            lambda row: utils.calculate_distance(
                clicked_lat, clicked_lon, 
                row['LATITUDE'], row['LONGITUDE']
            ),
            axis=1
        )
        
        # Filter stations within search radius
        RADIUS_MILES = 20
        nearby_stations = stations[
            stations['distance_miles'] <= RADIUS_MILES
        ].sort_values('distance_miles')
        
        # Limit to top N nearest stations for plot readability
        MAX_STATIONS = 5
        nearby_stations = nearby_stations.head(MAX_STATIONS)
        
        # Create figure
        fig = go.Figure()
        
        # Handle case where no stations are nearby
        if len(nearby_stations) == 0:
            fig.update_layout(
                title=f"No stations found within {RADIUS_MILES} miles of selected location",
                template="plotly_white",
                height=500,
                xaxis_title="Date",
                yaxis_title="Water Level (ft)",
                font=dict(family="Arial, sans-serif", size=12)
            )
            return fig
        
        # Color palette for different stations
        colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea']
        
        # Plot time series for each nearby station
        for idx, station_row in nearby_stations.iterrows():
            site_code = station_row['STATION']
            distance = station_row['distance_miles']
            
            # Get time series data for this station
            station_data = df[df['STATION'] == site_code].copy()
            station_data = station_data.sort_values('MSMT_DATE')
            
            if len(station_data) > 0:
                # Assign color from palette
                color_idx = list(nearby_stations.index).index(idx) % len(colors)
                
                # Get station metadata for hover info
                station_info = stations_df.loc[stations_df.STATION == site_code].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=station_data['MSMT_DATE'],
                    y=station_data['WSE'],
                    mode='lines',
                    name=f"{site_code} ({distance:.1f} miles)",
                    line=dict(color=colors[color_idx], width=2),
                    hovertemplate=(
                        f"<b>{site_code}</b><br>" +
                        f"Distance: {distance:.1f} miles<br>" +
                        "Water Level: %{y:.2f} ft<br>" +
                        f"Elevation: {station_info['ELEV']:.0f} ft<br>" +
                        f"Well Depth: {station_info['WELL_DEPTH']:.0f} ft<br>" +
                        "<extra></extra>"
                    )
                ))
        
        # Configure plot layout
        fig.update_layout(
            title=dict(
                text=f"Water Levels at {len(nearby_stations)} Stations within {RADIUS_MILES} miles",
                font=dict(size=16, color="#111827", family="Arial, sans-serif")
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False
            ),
            yaxis=dict(
                title="Water Level (ft above sea level)",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=True,
                zerolinecolor="rgba(0,0,0,0.2)",
                zerolinewidth=1
            ),
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
            margin=dict(l=60, r=200, t=50, b=50)
        )
        
        return fig