from dash import Dash
from wellapp.layout import layout  
from wellapp.callbacks import register_callbacks  

app = Dash(__name__)
app.layout = layout
register_callbacks(app)
