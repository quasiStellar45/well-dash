# wellapp/callbacks.py
from dash import Input, Output
import plotly.graph_objects as go
import wellapp.utils as utils
import pandas as pd
import numpy as np

def register_callbacks(app):
    # Load the data
    df_daily, df_monthly, quality_codes, stations_df = utils.load_data()

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
        if "customdata" not in point or point["customdata"] is None:
            return None
        # Retrieve station id from click data
        station_id = clickData["points"][0]["customdata"][0]

        return station_id
    
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
        Input("freq-dropdown", "value")
    )
    def update_wl_plot(station_id, freq):
        if station_id is None:
            return go.Figure()

        if freq == "daily":
            df = df_daily
        elif freq == "monthly":
            df = df_monthly
        else:
            df = pd.concat([df_daily, df_monthly], ignore_index=True)

        station_df = df.loc[df.STATION == station_id]

        if station_df.empty:
            return go.Figure()
        
        return utils.plot_station_data(df, station_id)
    
    @app.callback(
        Output("wl-plot", "figure"),
        Input("selected-station", "data"),    # Select station mode
        Input("click-location", "data")       # Model prediction mode
    )
    def add_ml_plot(station_id, click_loc):
        fig = go.Figure()

        # If clicked location provided, call ML
        if click_loc:
            lat = click_loc["lat"]
            lon = click_loc["lon"]
            elevation = utils.determine_elevation_from_raster(lon, lat)
            X = np.array()
            #pred = model.predict(...)
            #fig.add_trace(pred)
        
        # If a real station was selected, add real data
        #if station_id:
            # fig.add_trace(real_station_timeseries)

        return fig
