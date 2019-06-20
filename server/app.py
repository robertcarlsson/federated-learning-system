#!/usr/bin/env python

from flask import Flask, render_template, request, jsonify
from server import Server
import json

app = Flask(__name__)

server = Server()

@app.route("/", methods=['GET', 'POST'])
def index():
    message = ''
    if request.method == 'POST':
        message = "\nFederation ID: " + request.form['fed_id'] \
            + "\nDevices started: " + request.form['n_devices']
    return render_template('index.html', message=message)

@app.route("/create-federation", methods=['GET'])
def create_federation():
    server.create_federation()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')