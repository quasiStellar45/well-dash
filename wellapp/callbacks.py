# wellapp/callbacks.py
from dash import Input, Output
import plotly.graph_objects as go
import wellapp.utils as utils
import pandas as pd
import numpy as np

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
        else:
            allowed = set(stations_df.STATION.unique())

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
    
    # Determine click coordinates
    @app.callback(
        Output("click-coords", "children"),
        Input("map-plot", "clickData")
    )
    def display_click_coords(clickData):
        if clickData:
            lat = clickData["points"][0]["lat"]
            lon = clickData["points"][0]["lon"]
            return f"Clicked at LAT: {lat:.5f}, LON: {lon:.5f}"
        return "Click the map to get coordinates."
    
    # Store click coords in a store component
    @app.callback(
        Output("click-location", "data"),
        Input("map-plot", "clickData")
    )
    def store_click_coords(clickData):
        if not clickData:
            return None
        point = clickData["points"][0]
        return {"lat": point["lat"], "lon": point["lon"]}
    
    # Update the waterlevel plot with selection
    @app.callback(
        Output("wl-plot", "figure"),
        Input("selected-station", "data"),
        Input("click-location", "data"),
        Input("freq-dropdown", "value"),
        prevent_initial_call=False
    )
    def update_wl_plot_with_ml(station_id, click_loc, freq):
        if not station_id and not click_loc:
            fig = utils.create_empty_fig("Select a station to plot waterlevel...", "Water Surface Elevation (ft asl)")
                
        # Determine which dataframe to use
        if freq == "daily":
            df = df_daily
        elif freq == "monthly":
            df = df_monthly
        else:
            df = pd.concat([df_daily, df_monthly], ignore_index=True)
        
        # Variables to store location info for ML prediction
        lat, lon, elevation, well_depth = None, None, None, None
        le = utils.load_encoder()
        # If a real station was selected, add real data AND get its info for ML
        if station_id:
            station_df = df.loc[df.STATION == station_id]
            
            if not station_df.empty:
                # Plot real station data
                fig = utils.plot_station_data(df, station_id)
                
                # Get station info for ML prediction
                station_info = stations_df.loc[stations_df.STATION == station_id].iloc[0]
                lat = station_info['LATITUDE']
                lon = station_info['LONGITUDE']
                elevation = station_info['ELEV']
                well_depth = station_info['WELL_DEPTH']
                station_encoded = utils.encode_station(station_id, le)
                start_date = station_df['MSMT_DATE'].min()
        
        # If clicked location provided (but no station), use click coordinates
        elif click_loc:
            lat = click_loc["lat"]
            lon = click_loc["lon"]
            elevation = utils.determine_elevation_from_raster(lon, lat)
            well_depth = 100  # Unknown for arbitrary location
            start_date = None
            station_id = 'test'
            station_encoded = utils.encode_station(station_id, le)

            # Create empty figure
            fig = utils.create_empty_fig(f"Water Level for Station {station_id}", "Water Surface Elevation (ft asl)")
        
        # Generate ML prediction if we have a location (either from station or click)
        if lat is not None and lon is not None:
            # Create time range for prediction
            end_date = pd.Timestamp.now()
            if not start_date:
                start_date = pd.Timestamp('2000-01-01')
            date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
            
            predictions = []
            dates = []
            
            ref_date = pd.Timestamp('1800-01-01')  # Adjust to model
            
            for date in date_range:
                days_since_ref = (date - ref_date).days
                day_of_year = date.dayofyear
                
                # Create feature vector matching your training columns
                X = np.array([[
                    station_encoded,  # STATION_encoded (use 0 or mean for unknown location)
                    date.day,
                    date.month,
                    date.year,
                    days_since_ref,
                    np.sin(2 * np.pi * day_of_year / 365),  # day_sin
                    np.cos(2 * np.pi * day_of_year / 365),  # day_cos
                    np.sin(2 * np.pi * date.month / 12),        # month_sin
                    np.cos(2 * np.pi * date.month / 12),        # month_cos
                    elevation,
                    lat,
                    lon,
                    well_depth
                ]])
                
                pred = model.predict(X)[0]
                predictions.append(pred)
                dates.append(date)
            
            # Add prediction trace to figure
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines',
                name='ML Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Update title to show it includes predictions
            if station_id != 'test':
                current_title = fig.layout.title.text if fig.layout.title else ""
                fig.update_layout(
                    title=f'{current_title} (with ML Prediction)'
                )
            else:
                fig.update_layout(
                    title=f'ML Prediction for LAT: {lat:.4f}, LON: {lon:.4f}',
                    xaxis_title='Date',
                    yaxis_title='Water Level',
                    showlegend=True
                )

            fig.update_layout(showlegend=True)
        
        return fig

    @app.callback(
        Output("stl-trend", "figure"),
        Output("stl-seasonal", "figure"),
        Output("stl-resid", "figure"),
        Input("selected-station", "data"),
        prevent_initial_call=False
    )
    def plot_seasonal_variation(station_id):
        if not station_id:
            fig_trend = utils.create_empty_fig("Trend Component", "Water Surface Elevation (ft asl)")
            fig_seasonal = utils.create_empty_fig("Seasonal Component", "Seasonal Variation (ft)")
            fig_resid = utils.create_empty_fig("Residual Component", "Residual (ft)")
        else:
            station_df = df_monthly.loc[df_monthly.STATION == station_id, ['MSMT_DATE','WSE']]
            fig_trend, fig_seasonal, fig_resid = utils.create_stl_plot(station_df)

        return fig_trend, fig_seasonal, fig_resid