from dash import Dash
from wellapp.layout import layout  
from wellapp.callbacks import register_callbacks  

# Define dash app
app = Dash(__name__)
# Ensure server is exposed for gunicorn
server = app.server

app.layout = layout
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
