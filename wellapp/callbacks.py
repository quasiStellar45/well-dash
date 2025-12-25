# wellapp/callbacks.py
from dash import Input, Output
import plotly.graph_objects as go
import wellapp.utils as utils
import pandas as pd
import numpy as np
from dash import dcc, html, State
import plotly.express as px

def register_callbacks(app):
    # Load the data
    df_daily, df_monthly, quality_codes, stations_df = utils.load_data()

    # Load the ml model
    model = utils.load_ml_model()

    # Create the map
    @app.callback(
        Output("map-plot", "figure"),
        Input("freq-dropdown", "value"),
        Input("selected-station", "data"),
        Input("map-plot", "relayoutData")   
    )
    def update_map(freq, selected_station, relayout):

        # Determine which stations have daily/monthly data   
        if freq == "daily":
            allowed = set(df_daily.STATION.unique())
        elif freq == "monthly":
            allowed = set(df_monthly.STATION.unique())

        stations_df["highlight"] = stations_df["STATION"].apply(
            lambda s: "selected" if s == selected_station
            else ("included" if s in allowed else "excluded")
        )

        fig = utils.create_map(stations_df)

        # If relayoutData has zoom/center, preserve them
        if relayout is not None:
            if "mapbox.zoom" in relayout:
                fig.update_layout(mapbox={"zoom": relayout["mapbox.zoom"]})
            if "mapbox.center" in relayout:
                fig.update_layout(mapbox={"center": relayout["mapbox.center"]})

        return fig
    
    # Store click data from map
    @app.callback(
        Output("selected-station", "data"),
        Input("map-plot", "clickData")
    )
    def store_selected_station(clickData):
        if not clickData:
            return None

        # Check for customdata in the click
        point = clickData["points"][0]
        if "customdata" in point and point["customdata"]:
            return point["customdata"][0]

        # Otherwise, user clicked on empty map, do not change station selection
        return None
    
    # Update the waterlevel plot with selection
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
        if not station_id:
            fig = utils.create_empty_fig("Select a station to plot waterlevel...", "Water Surface Elevation (ft asl)")
            fig_trend = utils.create_empty_fig("Trend Component", "Water Surface Elevation (ft asl)")
            fig_seasonal = utils.create_empty_fig("Seasonal Component", "Seasonal Variation (ft)")
            fig_resid = utils.create_empty_fig("Residual Component", "Residual (ft)")
            return fig, fig_trend, fig_seasonal, fig_resid
                
        # Determine which dataframe to use
        if freq == "daily":
            df = df_daily
        elif freq == "monthly":
            df = df_monthly
        
        # If a real station was selected, add real data AND get its info for ML
        station_df = df.loc[df.STATION == station_id]
        if not station_df.empty:
            # Plot real station data
            fig = utils.plot_station_data(df, station_id, quality_codes)
        
            # Generate ML prediction for main plot (extends to now)
            predictions_full, dates_full = utils.generate_ml_predictions(station_id, station_df, stations_df, df_monthly, model)
            
            # Generate ML prediction for trend plot (only up to last data point)
            station_df_monthly = df_monthly.loc[df_monthly.STATION == station_id, ['MSMT_DATE','WSE']]
            last_data_date = station_df_monthly['MSMT_DATE'].max()
            date_range_trend = pd.date_range(start=dates_full[0], end=last_data_date, freq='ME')
                
            predictions_trend = predictions_full[0:len(date_range_trend)]
            dates_trend = dates_full[0:len(date_range_trend)]
            
            # Add prediction trace to main figure (full range)
            fig.add_trace(go.Scatter(
                x=dates_full,
                y=predictions_full,
                mode='lines',
                name='ML Prediction',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate=(
                    "%{y:.2f} ft"
                )
            ))
            
            # Update title to show it includes predictions
            current_title = fig.layout.title.text if fig.layout.title else ""
            fig.update_layout(
                title=f'{current_title} (with ML Prediction)',
                showlegend=True
            )

            # Create STL plots
            try:
                fig_trend, fig_seasonal, fig_resid = utils.create_stl_plot(station_df_monthly)
            except ValueError:
                fig = utils.create_empty_fig(f"No data for {station_id} for this data frequency.", "Water Surface Elevation (ft asl)")
                fig_trend = utils.create_empty_fig("Trend Component", "Water Surface Elevation (ft asl)")
                fig_seasonal = utils.create_empty_fig("Seasonal Component", "Seasonal Variation (ft)")
                fig_resid = utils.create_empty_fig("Residual Component", "Residual (ft)")
            
            # Add ML prediction to trend plot (only up to last data point)
            fig_trend.add_trace(go.Scatter(
                x=dates_trend,
                y=predictions_trend,
                mode='lines',
                name='ML Prediction',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate=(
                    "%{y:.2f}"
                )
            ))
            
            fig_trend.update_layout(showlegend=True)
        try:
            return fig, fig_trend, fig_seasonal, fig_resid
        except UnboundLocalError:
            fig = utils.create_empty_fig(f"No data for {station_id} for this data frequency.", "Water Surface Elevation (ft asl)")
            fig_trend = utils.create_empty_fig("Trend Component", "Water Surface Elevation (ft asl)")
            fig_seasonal = utils.create_empty_fig("Seasonal Component", "Seasonal Variation (ft)")
            fig_resid = utils.create_empty_fig("Residual Component", "Residual (ft)")
            return fig, fig_trend, fig_seasonal, fig_resid
    
    # Create spatial prediction map with station markers as reference
    @app.callback(
        Output("spatial-map", "figure"),
        Input("spatial-click-location", "data"),
        prevent_initial_call=False
    )
    def update_spatial_map(click_data):
        # Start with empty figure
        fig = go.Figure()
        
        # Add station markers as reference (grey, small, behind everything)
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
        
        # Create invisible clickable grid covering the map
        lat_min, lat_max = stations_df["LATITUDE"].min(), stations_df["LATITUDE"].max()
        lon_min, lon_max = stations_df["LONGITUDE"].min(), stations_df["LONGITUDE"].max()
        
        pad_lat = max(0.5, (lat_max - lat_min) * 0.1)
        pad_lon = max(0.5, (lon_max - lon_min) * 0.1)
        
        # Create a grid of invisible points
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
        
        # If a location has been clicked, add a marker for it (on top)
        if click_data:
            utils.add_map_marker(fig, click_data)
        
        # Set map center and zoom
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
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
    
    # Handle clicks on spatial map
    @app.callback(
        Output("spatial-click-location", "data"),
        Input("spatial-map", "clickData"),
        prevent_initial_call=True
    )
    def capture_spatial_click(click_data):
        if not click_data:
            return None
        
        # Extract lat/lon from click
        point = click_data["points"][0]
        lat = point["lat"]
        lon = point["lon"]
        
        # Get elevation from raster
        elevation = utils.determine_elevation_from_raster(lon, lat)
        
        return {
            "lat": lat,
            "lon": lon,
            "elevation": elevation
        }
    
    # Display click information
    @app.callback(
        Output("click-info", "children"),
        Input("spatial-click-location", "data"),
        prevent_initial_call=False
    )
    def display_click_info(click_data):
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
        
        return html.Div(
            [
                html.Div(
                    [
                        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
                        html.Span("Location Selected", style={"fontWeight": "600", "color": "#111827"})
                    ],
                    style={"marginBottom": "8px"}
                ),
                html.Div(
                    [
                        html.Div(f"Latitude: {click_data['lat']:.4f}°", style={"fontSize": "13px", "color": "#4b5563"}),
                        html.Div(f"Longitude: {click_data['lon']:.4f}°", style={"fontSize": "13px", "color": "#4b5563"}),
                        html.Div(f"Elevation: {click_data.get('elevation', 'N/A')} ft", style={"fontSize": "13px", "color": "#4b5563"})
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
    
    # Generate spatial prediction
    @app.callback(
        Output("spatial-prediction-plot", "figure"),
        Input("spatial-click-location", "data"),
        Input("well-depth-input", "value"),
        prevent_initial_call=False
    )
    def update_spatial_prediction(click_data, well_depth):
        if not click_data:
            return utils.create_empty_fig(
                "Click on the map to generate predictions",
                "Water Surface Elevation (ft asl)"
            )
        
        # Load data from map click
        lat = click_data["lat"]
        lon = click_data["lon"]
        
        # Generate ml predictions for click location
        predictions, dates = utils.generate_ml_predictions(monthly_df=df_monthly, model=model, click_data=click_data, well_depth=well_depth)
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name='ML Prediction',
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
    
    @app.callback(
        Output("nearby-stations-plot", "figure"),
        Input("spatial-click-location", "data"),
        State("freq-dropdown", "value")
    )
    def update_nearby_stations_plot(click_location, freq):
        """
        Plot water levels from stations within radius of clicked location.
        """
        # Default empty figure
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
        
        # Get clicked coordinates
        clicked_lat = click_location['lat']
        clicked_lon = click_location['lon']
        
        # Select appropriate dataset
        df = df_monthly if freq == 'monthly' else df_daily
        
        # Get unique stations with their coordinates
        stations = stations_df[['STATION', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
        
        # Calculate distance to each station
        stations['distance_miles'] = stations.apply(
            lambda row: utils.calculate_distance(
                clicked_lat, clicked_lon, 
                row['LATITUDE'], row['LONGITUDE']
            ),
            axis=1
        )
        
        # Filter stations within radius
        RADIUS_MILES = 20
        nearby_stations = stations[stations['distance_miles'] <= RADIUS_MILES].sort_values('distance_miles')
        
        # Limit to top 5 nearest stations for readability
        MAX_STATIONS = 5
        nearby_stations = nearby_stations.head(MAX_STATIONS)
        
        # Create figure
        fig = go.Figure()
        
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
        colors = [
            '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea',
        ]
        
        # Plot each nearby station
        for idx, station_row in nearby_stations.iterrows():
            site_code = station_row['STATION']
            distance = station_row['distance_miles']
            
            # Get time series data for this station
            station_data = df[df['STATION'] == site_code].copy()
            station_data = station_data.sort_values('MSMT_DATE')
            
            if len(station_data) > 0:
                color_idx = list(nearby_stations.index).index(idx) % len(colors)
                
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
                        f"Elevation: {stations_df.loc[stations_df.STATION == site_code, 'ELEV'].iloc[0].item()} ft<br>" +
                        f"Well Depth: {stations_df.loc[stations_df.STATION == site_code, 'WELL_DEPTH'].iloc[0].item()} ft<br>" +
                        "<extra></extra>"
                    )
                ))
        
        # Update layout
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