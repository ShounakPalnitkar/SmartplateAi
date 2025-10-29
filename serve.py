from flask import Flask, send_file
import os

app = Flask(__name__)

# Serve your single HTML file for all frontend routes
@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path=None):
    return send_file('index.html')

# Import your backend API routes
from backend import *

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
