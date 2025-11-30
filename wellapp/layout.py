"""
Configure the app layout.
"""
from dash import dcc, html

# Define app layout with improved UI
layout = html.Div(
    [
        # Header with gradient background
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "Northern California DWR Well Data",
                            style={
                                "color": "white",
                                "margin": "0",
                                "fontSize": "32px",
                                "fontWeight": "600",
                                "letterSpacing": "-0.5px"
                            }
                        ),
                        html.P(
                            "Interactive groundwater monitoring and analysis",
                            style={
                                "color": "rgba(255, 255, 255, 0.9)",
                                "margin": "5px 0 0 0",
                                "fontSize": "16px"
                            }
                        )
                    ],
                    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 20px"}
                )
            ],
            style={
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "padding": "30px 0",
                "marginBottom": "30px",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
            }
        ),
        
        # Main content container
        html.Div(
            [
                # Control panel card
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Data Frequency",
                                    style={
                                        "fontWeight": "600",
                                        "fontSize": "14px",
                                        "color": "#374151",
                                        "marginBottom": "8px",
                                        "display": "block"
                                    }
                                ),
                                dcc.Dropdown(
                                    id="freq-dropdown",
                                    options=[
                                        {"label": "Daily Only", "value": "daily"},
                                        {"label": "Monthly Only", "value": "monthly"},
                                        {"label": "All Data", "value": "all"},
                                    ],
                                    value="daily",
                                    clearable=False,
                                    style={
                                        "width": "100%",
                                        "fontSize": "14px"
                                    }
                                ),
                            ]
                        )
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.08)",
                        "marginBottom": "20px"
                    }
                ),
                
                # Side-by-side map and plot container
                html.Div(
                    [
                        # Map card (left side)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Well Location Map",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "fontSize": "20px",
                                                "fontWeight": "600",
                                                "color": "#1f2937"
                                            }
                                        ),
                                        html.P(
                                            "Click on a well marker to view its data",
                                            style={
                                                "margin": "0 0 15px 0",
                                                "fontSize": "14px",
                                                "color": "#6b7280"
                                            }
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="map-plot",
                                    style={
                                        "height": "600px",
                                        "borderRadius": "8px",
                                        "overflow": "hidden"
                                    },
                                    config={
                                        "scrollZoom": True,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                        "displaylogo": False
                                    }
                                )
                            ],
                            style={
                                "backgroundColor": "white",
                                "padding": "25px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.08)",
                                "width": "48%"
                            }
                        ),
                        
                        # Water level plot card (right side)
                        html.Div(
                            [
                                html.Div(id='click-coords'),
                                html.Div(
                                    [
                                        html.H3(
                                            "Water Level Time Series",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "fontSize": "20px",
                                                "fontWeight": "600",
                                                "color": "#1f2937"
                                            }
                                        ),
                                        html.P(
                                            "Historical groundwater elevation measurements",
                                            style={
                                                "margin": "0 0 15px 0",
                                                "fontSize": "14px",
                                                "color": "#6b7280"
                                            }
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="wl-plot",
                                    style={
                                        "height": "600px",
                                        "borderRadius": "8px",
                                        "overflow": "hidden"
                                    },
                                    config={
                                        "scrollZoom": True,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                        "displaylogo": False
                                    }
                                )
                            ],
                            style={
                                "backgroundColor": "white",
                                "padding": "25px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.08)",
                                "width": "48%"
                            }
                        )
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "gap": "20px",
                        "marginBottom": "20px"
                    }
                )
            ],
            id="dashboard-content",
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "padding": "0 20px"
            }
        ),
        
        # Footer
        html.Div(
            [
                html.P(
                    "Data source: California Department of Water Resources",
                    style={
                        "textAlign": "center",
                        "color": "#9ca3af",
                        "fontSize": "13px",
                        "margin": "0"
                    }
                )
            ],
            style={
                "padding": "30px 0",
                "marginTop": "40px",
                "borderTop": "1px solid #e5e7eb"
            }
        ),

        # Store components
        dcc.Store(id="selected-station"),
        dcc.Store(id="click-location")
    ],
    style={
        "backgroundColor": "#f3f4f6",
        "minHeight": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    }
)