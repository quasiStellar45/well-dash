from wellapp import app

# Ensure server is exposed for gunicorn
server = app.server

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
