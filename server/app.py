#!/usr/bin/env python
import json

from server import Server
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

server = Server()

@app.route("/", methods=['GET', 'POST'])
def index():
    message = 'nu är det ändrat igen'
    if request.method == 'POST':
        message = "\nFederation ID: " + request.form['fed_id'] \
            + "\nDevices started: " + request.form['n_devices']
    return render_template('index.html', message=message)

@app.route("/create-federation", methods=['GET'])
def create_federation():
    server.create_federation()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')