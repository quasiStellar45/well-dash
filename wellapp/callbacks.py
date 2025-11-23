# app/callbacks.py
from dash import Input, Output
import plotly.graph_objects as go
import wellapp.utils as utils

def register_callbacks(app):
    df_daily, df_monthly, quality_codes, stations_df = utils.load_data()

    @app.callback(
        Output("map-plot", "figure"),
        Input("column-dropdown", "value")
    )
    def update_map(selected_col):
        return utils.create_map(stations_df)
