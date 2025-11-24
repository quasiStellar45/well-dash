"""
Configure the app layout.
"""
from dash import dcc, html

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
                                html.H4("Well Map", style={"marginBottom": "5px"}),
                            ],
                            style={"marginBottom": "5px", "textAlign": "center"},
                        ),
                        # Plots
                        html.Div(
                            [
                                # Dropdown to select what data to display
                                dcc.Dropdown(
                                id="freq-dropdown",
                                options=[
                                    {"label": "Daily Only", "value": "daily"},
                                    {"label": "Monthly Only", "value": "monthly"},
                                    {"label": "All Data", "value": "all"},
                                ],
                                value="daily",
                                clearable=False,
                                style={"width": "200px"}
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
                            
                            # Water level plot
                            dcc.Graph(
                                id="wl-plot",
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
                            style={"marginBottom": "5px", "textAlign": "center"},
                        )
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

        # Store components
        dcc.Store(id="selected-station")
    ],
    style={"padding": "20px"},
    
)