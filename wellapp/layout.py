"""
Configure the app layout.
"""
from dash import dcc, html

# Define app layout 
layout = html.Div(
    [
        # Header with gradient background
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "Northern California Groundwater Monitoring",
                            style={
                                "color": "white",
                                "margin": "0",
                                "fontSize": "36px",
                                "fontWeight": "700",
                                "letterSpacing": "-0.5px"
                            }
                        ),
                        html.P(
                            "Interactive groundwater level analysis and ML predictions",
                            style={
                                "color": "rgba(255, 255, 255, 0.95)",
                                "margin": "8px 0 0 0",
                                "fontSize": "18px",
                                "fontWeight": "300"
                            }
                        )
                    ],
                    style={"maxWidth": "1600px", "margin": "0 auto", "padding": "0 30px"}
                )
            ],
            style={
                "background": "linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%)",
                "padding": "40px 0",
                "marginBottom": "30px",
                "boxShadow": "0 10px 25px rgba(0, 0, 0, 0.15)"
            }
        ),
        
        # Main content container
        html.Div(
            [
                # Tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="station-analysis",
                    children=[
                        # Station Analysis Tab
                        dcc.Tab(
                            label="📊 Station Analysis",
                            value="station-analysis",
                            style={
                                "padding": "12px 24px",
                                "fontWeight": "600",
                                "fontSize": "15px"
                            },
                            selected_style={
                                "padding": "12px 24px",
                                "fontWeight": "600",
                                "fontSize": "15px",
                                "borderTop": "3px solid #2563eb",
                                "backgroundColor": "white"
                            },
                            children=[
                                html.Div(
                                    [
                                        # Control panel card
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Span(
                                                                    "⚙️",
                                                                    style={"fontSize": "20px", "marginRight": "8px"}
                                                                ),
                                                                html.Span(
                                                                    "Data Frequency",
                                                                    style={
                                                                        "fontWeight": "600",
                                                                        "fontSize": "15px",
                                                                        "color": "#111827"
                                                                    }
                                                                )
                                                            ],
                                                            style={"marginBottom": "10px"}
                                                        ),
                                                        dcc.Dropdown(
                                                            id="freq-dropdown",
                                                            options=[
                                                                {"label": "📅 Monthly Data", "value": "monthly"},
                                                                {"label": "📊 Daily Data", "value": "daily"},
                                                            ],
                                                            value="monthly",
                                                            clearable=False,
                                                            style={
                                                                "width": "250px",
                                                                "fontSize": "14px"
                                                            }
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column"}
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "white",
                                                "padding": "20px 25px",
                                                "borderRadius": "16px",
                                                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                "marginBottom": "25px",
                                                "border": "1px solid #e5e7eb"
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
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            "📍",
                                                                            style={"fontSize": "24px", "marginRight": "10px"}
                                                                        ),
                                                                        html.H3(
                                                                            "Well Location Map",
                                                                            style={
                                                                                "margin": "0",
                                                                                "fontSize": "22px",
                                                                                "fontWeight": "700",
                                                                                "color": "#111827",
                                                                                "display": "inline"
                                                                            }
                                                                        )
                                                                    ],
                                                                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                                ),
                                                                html.P(
                                                                    "Click on a well marker to view historical data and predictions",
                                                                    style={
                                                                        "margin": "0 0 20px 0",
                                                                        "fontSize": "14px",
                                                                        "color": "#6b7280",
                                                                        "lineHeight": "1.5"
                                                                    }
                                                                )
                                                            ]
                                                        ),
                                                        dcc.Graph(
                                                            id="map-plot",
                                                            style={
                                                                "height": "650px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb"
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
                                                        "padding": "30px",
                                                        "borderRadius": "16px",
                                                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                        "width": "48%",
                                                        "border": "1px solid #e5e7eb"
                                                    }
                                                ),
                                                
                                                # Water level plot card (right side)
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            "📈",
                                                                            style={"fontSize": "24px", "marginRight": "10px"}
                                                                        ),
                                                                        html.H3(
                                                                            "Water Level Time Series",
                                                                            style={
                                                                                "margin": "0",
                                                                                "fontSize": "22px",
                                                                                "fontWeight": "700",
                                                                                "color": "#111827",
                                                                                "display": "inline"
                                                                            }
                                                                        )
                                                                    ],
                                                                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                                ),
                                                                html.P(
                                                                    [
                                                                        html.Span("Historical measurements ", style={"color": "#6b7280"}),
                                                                        html.Span("(blue)", style={"color": "#2563eb", "fontWeight": "600"}),
                                                                        html.Span(" with ML predictions ", style={"color": "#6b7280"}),
                                                                        html.Span("(red)", style={"color": "#dc2626", "fontWeight": "600"})
                                                                    ],
                                                                    style={
                                                                        "margin": "0 0 20px 0",
                                                                        "fontSize": "14px",
                                                                        "lineHeight": "1.5"
                                                                    }
                                                                )
                                                            ]
                                                        ),
                                                        dcc.Graph(
                                                            id="wl-plot",
                                                            style={
                                                                "height": "650px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb"
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
                                                        "padding": "30px",
                                                        "borderRadius": "16px",
                                                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                        "width": "48%",
                                                        "border": "1px solid #e5e7eb"
                                                    }
                                                )
                                            ],
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "gap": "25px",
                                                "marginBottom": "30px"
                                            }
                                        ),

                                        # STL Decomposition Section
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Span(
                                                                    "🔬",
                                                                    style={"fontSize": "24px", "marginRight": "10px"}
                                                                ),
                                                                html.H3(
                                                                    "Time Series Decomposition (STL)",
                                                                    style={
                                                                        "margin": "0",
                                                                        "fontSize": "22px",
                                                                        "fontWeight": "700",
                                                                        "color": "#111827",
                                                                        "display": "inline"
                                                                    }
                                                                )
                                                            ],
                                                            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                        ),
                                                        html.P(
                                                            "Seasonal-Trend decomposition showing long-term trends, seasonal patterns, and residual variations",
                                                            style={
                                                                "margin": "0 0 25px 0",
                                                                "fontSize": "14px",
                                                                "color": "#6b7280",
                                                                "lineHeight": "1.5"
                                                            }
                                                        )
                                                    ]
                                                ),
                                                
                                                # Three STL plots stacked vertically
                                                html.Div(
                                                    [
                                                        dcc.Graph(
                                                            id='stl-trend',
                                                            style={
                                                                "height": "600px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb",
                                                                "marginBottom": "15px"
                                                            },
                                                            config={
                                                                "scrollZoom": True,
                                                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                                                "displaylogo": False
                                                            }
                                                        ),
                                                        dcc.Graph(
                                                            id='stl-seasonal',
                                                            style={
                                                                "height": "300px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb",
                                                                "marginBottom": "15px"
                                                            },
                                                            config={
                                                                "scrollZoom": True,
                                                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                                                "displaylogo": False
                                                            }
                                                        ),
                                                        dcc.Graph(
                                                            id='stl-resid',
                                                            style={
                                                                "height": "300px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb"
                                                            },
                                                            config={
                                                                "scrollZoom": True,
                                                                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                                                "displaylogo": False
                                                            }
                                                        )
                                                    ]
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "white",
                                                "padding": "30px",
                                                "borderRadius": "16px",
                                                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                "marginBottom": "30px",
                                                "border": "1px solid #e5e7eb"
                                            }
                                        )
                                    ],
                                    style={"padding": "25px 0"}
                                )
                            ]
                        ),
                        
                        # Spatial Prediction Tab
                        dcc.Tab(
                            label="🗺️ Spatial Prediction",
                            value="spatial-prediction",
                            style={
                                "padding": "12px 24px",
                                "fontWeight": "600",
                                "fontSize": "15px"
                            },
                            selected_style={
                                "padding": "12px 24px",
                                "fontWeight": "600",
                                "fontSize": "15px",
                                "borderTop": "3px solid #2563eb",
                                "backgroundColor": "white"
                            },
                            children=[
                                html.Div(
                                    [
                                        # Instructions and controls
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "🎯",
                                                            style={"fontSize": "24px", "marginRight": "10px"}
                                                        ),
                                                        html.H3(
                                                            "Spatial ML Prediction",
                                                            style={
                                                                "margin": "0",
                                                                "fontSize": "22px",
                                                                "fontWeight": "700",
                                                                "color": "#111827",
                                                                "display": "inline"
                                                            }
                                                        )
                                                    ],
                                                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                ),
                                                html.P(
                                                    "Click anywhere on the map to generate groundwater level predictions for that location",
                                                    style={
                                                        "margin": "0 0 20px 0",
                                                        "fontSize": "14px",
                                                        "color": "#6b7280",
                                                        "lineHeight": "1.5"
                                                    }
                                                ),
                                                
                                                # Well depth input
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Well Depth (ft)",
                                                            style={
                                                                "fontWeight": "600",
                                                                "fontSize": "14px",
                                                                "color": "#374151",
                                                                "marginBottom": "8px",
                                                                "display": "block"
                                                            }
                                                        ),
                                                        dcc.Input(
                                                            id="well-depth-input",
                                                            type="number",
                                                            value=100,
                                                            min=0,
                                                            step=10,
                                                            style={
                                                                "width": "200px",
                                                                "padding": "8px 12px",
                                                                "fontSize": "14px",
                                                                "border": "1px solid #d1d5db",
                                                                "borderRadius": "8px"
                                                            }
                                                        ),
                                                        html.P(
                                                            "Enter the estimated well depth for the prediction",
                                                            style={
                                                                "margin": "8px 0 0 0",
                                                                "fontSize": "12px",
                                                                "color": "#9ca3af",
                                                                "fontStyle": "italic"
                                                            }
                                                        )
                                                    ],
                                                    style={"marginTop": "15px"}
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "white",
                                                "padding": "25px 30px",
                                                "borderRadius": "16px",
                                                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                "marginBottom": "25px",
                                                "border": "1px solid #e5e7eb"
                                            }
                                        ),
                                        
                                        # Side-by-side map and plot
                                        html.Div(
                                            [
                                                # Prediction map
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            "📍",
                                                                            style={"fontSize": "24px", "marginRight": "10px"}
                                                                        ),
                                                                        html.H3(
                                                                            "Select Location",
                                                                            style={
                                                                                "margin": "0",
                                                                                "fontSize": "22px",
                                                                                "fontWeight": "700",
                                                                                "color": "#111827",
                                                                                "display": "inline"
                                                                            }
                                                                        )
                                                                    ],
                                                                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                                ),
                                                                html.P(
                                                                    "Click anywhere to predict groundwater levels",
                                                                    style={
                                                                        "margin": "0 0 15px 0",
                                                                        "fontSize": "14px",
                                                                        "color": "#6b7280",
                                                                        "lineHeight": "1.5"
                                                                    }
                                                                ),
                                                                html.Div(id="click-info", style={"marginBottom": "15px"})
                                                            ]
                                                        ),
                                                        dcc.Graph(
                                                            id="spatial-map",
                                                            style={
                                                                "height": "650px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb"
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
                                                        "padding": "30px",
                                                        "borderRadius": "16px",
                                                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                        "width": "48%",
                                                        "border": "1px solid #e5e7eb"
                                                    }
                                                ),
                                                
                                                # Prediction plot
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            "🤖",
                                                                            style={"fontSize": "24px", "marginRight": "10px"}
                                                                        ),
                                                                        html.H3(
                                                                            "ML Prediction",
                                                                            style={
                                                                                "margin": "0",
                                                                                "fontSize": "22px",
                                                                                "fontWeight": "700",
                                                                                "color": "#111827",
                                                                                "display": "inline"
                                                                            }
                                                                        )
                                                                    ],
                                                                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                                ),
                                                                html.P(
                                                                    "Predicted groundwater levels over time",
                                                                    style={
                                                                        "margin": "0 0 20px 0",
                                                                        "fontSize": "14px",
                                                                        "color": "#6b7280",
                                                                        "lineHeight": "1.5"
                                                                    }
                                                                )
                                                            ]
                                                        ),
                                                        dcc.Graph(
                                                            id="spatial-prediction-plot",
                                                            style={
                                                                "height": "650px",
                                                                "borderRadius": "12px",
                                                                "overflow": "hidden",
                                                                "border": "1px solid #e5e7eb"
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
                                                        "padding": "30px",
                                                        "borderRadius": "16px",
                                                        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                        "width": "48%",
                                                        "border": "1px solid #e5e7eb"
                                                    }
                                                )
                                            ],
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "gap": "25px",
                                                "marginbottom": "25px"
                                            }
                                        ),
                                        # Nearby stations plot
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Span(
                                                                    "📊",
                                                                    style={"fontSize": "24px", "marginRight": "10px"}
                                                                ),
                                                                html.H3(
                                                                    "Nearby Station Water Levels",
                                                                    style={
                                                                        "margin": "0",
                                                                        "fontSize": "22px",
                                                                        "fontWeight": "700",
                                                                        "color": "#111827",
                                                                        "display": "inline"
                                                                    }
                                                                )
                                                            ],
                                                            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}
                                                        ),
                                                        html.P(
                                                            "Historical water levels from monitoring stations near the selected location",
                                                            style={
                                                                "margin": "0 0 20px 0",
                                                                "fontSize": "14px",
                                                                "color": "#6b7280",
                                                                "lineHeight": "1.5"
                                                            }
                                                        )
                                                    ]
                                                ),
                                                dcc.Graph(
                                                    id="nearby-stations-plot",
                                                    style={
                                                        "height": "500px",
                                                        "borderRadius": "12px",
                                                        "overflow": "hidden",
                                                        "border": "1px solid #e5e7eb"
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
                                                "padding": "30px",
                                                "borderRadius": "16px",
                                                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                                                "border": "1px solid #e5e7eb"
                                            }
                                        )
                                    ],
                                    style={"padding": "25px 0"}
                                )
                            ]
                        )
                    ],
                    style={
                        "marginBottom": "0"
                    }
                )
            ],
            id="dashboard-content",
            style={
                "maxWidth": "1600px",
                "margin": "0 auto",
                "padding": "0 30px"
            }
        ),
        
        # Footer
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                html.Span("Data source: ", style={"color": "#9ca3af"}),
                                html.A(
                                    "California Department of Water Resources",
                                    href="https://water.ca.gov/",
                                    target="_blank",
                                    style={
                                        "color": "#2563eb",
                                        "textDecoration": "none",
                                        "fontWeight": "500"
                                    }
                                )
                            ],
                            style={
                                "textAlign": "center",
                                "fontSize": "14px",
                                "margin": "0"
                            }
                        )
                    ],
                    style={"maxWidth": "1600px", "margin": "0 auto", "padding": "0 30px"}
                )
            ],
            style={
                "padding": "30px 0",
                "marginTop": "50px",
                "borderTop": "1px solid #e5e7eb",
                "backgroundColor": "#f9fafb"
            }
        ),

        # Store components
        dcc.Store(id="selected-station"),
        dcc.Store(id="spatial-click-location")
    ],
    style={
        "backgroundColor": "#f3f4f6",
        "minHeight": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    }
)