"""
Configure the app layout.
"""
from dash import dcc, html
from wellapp.utils import get_columns

# Define app layout with improved UI
layout = html.Div(
    [
        # Header with title and status
        html.Div(
            [
                html.H2("Northern California DWR Well Data", style={"flex": "1", "marginBottom": "0"}),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "10px",
                "borderBottom": "1px solid #ddd",
                "paddingBottom": "10px",
            },
        ),
        # Dashboard content 
        html.Div(
            id="dashboard-content",
            children=[
                html.Div(
                    [
                        # Map header
                        html.Div(
                            [
                                html.H4("Air Pollution Map", style={"marginBottom": "5px"}),
                            ],
                            style={"marginBottom": "5px", "textAlign": "center"},
                        ),
                        html.Div(
                            [
                                html.Label("Select Parameter to Plot:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='column-dropdown',
                                    options=get_columns(),
                                    multi=True,
                                    style={"width": "100%"},
                                    value=["AQI Value"],
                                    clearable=True,
                                    placeholder="Select parameter...",
                                ),
                            ],
                            style={
                                "marginTop": "15px",
                                "padding": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                            },
                        ),
                        # Map plot
                        dcc.Graph(
                            id="map-plot",
                            style={
                                "height": "500px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                            },
                            config={
                                "scrollZoom": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "10px",
                        "backgroundColor": "#f9f9f9",
                        "borderRadius": "10px",
                    },
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        ),
    ],
    style={"padding": "20px"},
)