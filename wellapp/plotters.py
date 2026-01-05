# wellapp/plotters.py
"""
Reusable plot builders for water level visualization.

This module provides classes that use a fluent interface (method chaining)
to build complex plots incrementally. This reduces code duplication and
provides a clean separation between data preparation and visualization.
"""

import plotly.graph_objects as go
from typing import Optional, List
import numpy as np

from wellapp.config import config


class WaterLevelPlot:
    """
    Builder for water level time series plots.
    
    Provides a fluent interface for constructing plots with observed data,
    predictions, confidence intervals, and other components.
    
    Parameters
    ----------
    title : str, optional
        Plot title, by default "Water Level"
    
    Attributes
    ----------
    fig : plotly.graph_objects.Figure
        The figure being constructed
    title : str
        Current plot title
    
    Examples
    --------
    >>> plot = (WaterLevelPlot(title="Station 001")
    ...         .add_observed_data(dates, values)
    ...         .add_prediction(pred_dates, predictions)
    ...         .build())
    """
    
    def __init__(self, title: str = "Water Level"):
        """
        Initialize plot builder.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        self.fig = go.Figure()
        self.title = title
        self._configure_layout()
    
    def _configure_layout(self):
        """Set default layout with standard styling."""
        self.fig.update_layout(
            template=config.plots.plot_template,
            title=self.title,
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title='Date'
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title="Water Surface Elevation (ft asl)"
            ),
            showlegend=True,
            hovermode='x unified',
            height=config.plots.plot_height
        )
    
    def add_observed_data(
        self, 
        dates, 
        values, 
        qc_descriptions: Optional[List[str]] = None,
        name: str = "Observed",
        color: str = 'blue'
    ):
        """
        Add observed water level data to the plot.
        
        Parameters
        ----------
        dates : array-like
            Measurement dates
        values : array-like
            Water level values
        qc_descriptions : list of str, optional
            Quality control descriptions for hover info
        name : str, optional
            Trace name for legend
        color : str, optional
            Line color
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Examples
        --------
        >>> plot = WaterLevelPlot().add_observed_data(dates, values)
        """
        hover_template = "Water Level: %{y:.2f} ft<br>"
        if qc_descriptions is not None:
            hover_template += "QC Flag: %{customdata[0]}<br>"
        hover_template += "<extra></extra>"
        
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=name,
            line=dict(color=color),
            marker=dict(size=4),
            customdata=[[desc] for desc in qc_descriptions] if qc_descriptions else None,
            hovertemplate=hover_template
        ))
        return self
    
    def add_prediction(
        self,
        dates,
        predictions,
        name: str = "Prediction",
        color: str = "red",
        dash: str = "dash",
        width: int = 2
    ):
        """
        Add a prediction trace to the plot.
        
        Parameters
        ----------
        dates : array-like
            Prediction dates
        predictions : array-like
            Predicted values
        name : str, optional
            Trace name for legend
        color : str, optional
            Line color
        dash : str, optional
            Line dash style ('solid', 'dash', 'dot', etc.)
        width : int, optional
            Line width
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Examples
        --------
        >>> plot = (WaterLevelPlot()
        ...         .add_prediction(dates, preds, name="XGBoost", color="red"))
        """
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name=name,
            line=dict(color=color, dash=dash, width=width),
            hovertemplate="%{y:.2f} ft<extra></extra>"
        ))
        return self
    
    def add_confidence_band(
        self,
        dates,
        mean,
        std,
        n_std: float = None,
        name: str = "95% confidence",
        color: str = "rgba(235, 216, 190, 0.3)"
    ):
        """
        Add confidence interval band to the plot.
        
        Creates a shaded region representing the uncertainty in predictions.
        
        Parameters
        ----------
        dates : array-like
            Dates corresponding to predictions
        mean : array-like
            Mean predicted values
        std : array-like
            Standard deviations
        n_std : float, optional
            Number of standard deviations for interval (default from config: 1.96 ≈ 95%)
        name : str, optional
            Legend name for the interval
        color : str, optional
            Fill color (RGBA format recommended)
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Examples
        --------
        >>> plot = WaterLevelPlot().add_confidence_band(dates, mean, std)
        """
        if n_std is None:
            n_std = config.gp.confidence_interval
        
        # Upper bound (invisible line)
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=mean + n_std * std,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Lower bound (fills to previous trace)
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=mean - n_std * std,
            mode="lines",
            fill="tonexty",
            fillcolor=color,
            line=dict(width=0),
            name=name,
            hoverinfo="skip"
        ))
        return self
    
    def update_title(self, title: str):
        """
        Update the plot title.
        
        Parameters
        ----------
        title : str
            New title
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self.title = title
        self.fig.update_layout(title=title)
        return self
    
    def build(self) -> go.Figure:
        """
        Return the completed figure.
        
        Returns
        -------
        plotly.graph_objects.Figure
            The constructed plot ready for display
        
        Examples
        --------
        >>> fig = (WaterLevelPlot()
        ...        .add_observed_data(dates, values)
        ...        .build())
        >>> fig.show()
        """
        return self.fig


class STLPlot:
    """
    Builder for STL (Seasonal-Trend decomposition using LOESS) component plots.
    
    Creates standardized plots for trend, seasonal, and residual components.
    
    Parameters
    ----------
    component_type : str
        Type of component ('trend', 'seasonal', or 'residual')
    
    Examples
    --------
    >>> plot = (STLPlot('seasonal')
    ...         .add_component(dates, seasonal_values, name='Observed')
    ...         .add_component(dates, ml_seasonal, name='ML', color='red')
    ...         .build())
    """
    
    def __init__(self, component_type: str):
        """
        Initialize STL component plot builder.
        
        Parameters
        ----------
        component_type : str
            Component type: 'trend', 'seasonal', or 'residual'
        """
        self.fig = go.Figure()
        self.component_type = component_type
        self._configure_layout()
    
    def _configure_layout(self):
        """Configure layout based on component type."""
        titles = {
            'trend': 'Trend Component',
            'seasonal': 'Seasonal Component',
            'residual': 'Residual Component'
        }
        
        yaxis_titles = {
            'trend': 'Water Surface Elevation (ft asl)',
            'seasonal': 'Seasonal Variation (ft)',
            'residual': 'Residual (ft)'
        }
        
        self.fig.update_layout(
            template=config.plots.plot_template,
            title=titles.get(self.component_type, 'Component'),
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title='Date'
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                title=yaxis_titles.get(self.component_type, 'Value')
            ),
            showlegend=True,
            hovermode='x unified',
            height=config.plots.plot_height
        )
    
    def add_component(
        self,
        dates,
        values,
        name: str = "Component",
        color: str = "cyan",
        dash: str = "solid"
    ):
        """
        Add a component trace to the plot.
        
        Parameters
        ----------
        dates : array-like
            Dates
        values : array-like
            Component values
        name : str, optional
            Trace name
        color : str, optional
            Line color
        dash : str, optional
            Line style
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self.fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=color, dash=dash),
            hovertemplate="%{y:.2f}<extra></extra>"
        ))
        return self
    
    def build(self) -> go.Figure:
        """
        Return the completed figure.
        
        Returns
        -------
        plotly.graph_objects.Figure
            The constructed plot
        """
        return self.fig


def create_empty_plot(
    title: str,
    yaxis_title: str,
    xaxis_title: str = "Date"
) -> go.Figure:
    """
    Create an empty placeholder plot.
    
    Useful for displaying when no data is available or before user interaction.
    
    Parameters
    ----------
    title : str
        Plot title
    yaxis_title : str
        Y-axis label
    xaxis_title : str, optional
        X-axis label
    
    Returns
    -------
    plotly.graph_objects.Figure
        Empty figure with formatting
    
    Examples
    --------
    >>> fig = create_empty_plot(
    ...     "Select a station",
    ...     "Water Surface Elevation (ft asl)"
    ... )
    """
    fig = go.Figure()
    fig.update_layout(
        template=config.plots.plot_template,
        title=title,
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='black',
            title=xaxis_title
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='black',
            title=yaxis_title
        ),
        height=config.plots.plot_height
    )
    return fig


def create_error_plot(error_message: str) -> go.Figure:
    """
    Create a plot displaying an error message.
    
    Used to show user-friendly error messages when plot generation fails.
    
    Parameters
    ----------
    error_message : str
        Error message to display
    
    Returns
    -------
    plotly.graph_objects.Figure
        Figure with error message
    
    Examples
    --------
    >>> fig = create_error_plot("Station data not found")
    """
    fig = go.Figure()
    
    # Add text annotation with error message
    fig.add_annotation(
        text=f"⚠️ {error_message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="red"),
        xanchor="center",
        yanchor="middle"
    )
    
    fig.update_layout(
        template=config.plots.plot_template,
        title="Error",
        xaxis=dict(showticklabels=False, showline=False),
        yaxis=dict(showticklabels=False, showline=False),
        height=config.plots.plot_height
    )
    
    return fig