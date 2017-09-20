#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

from flask import Flask, request as req, make_response, jsonify, abort
from flask_cors import CORS
import argparse

import optimiz as optimiz

app = Flask(__name__)

CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False)
parser.add_argument('--port', default=5000)

args = vars(parser.parse_args())

# With debug True, slower computation due to multiple evalutations to plot a figure
debug = args['debug']

@app.route('/')
def index():
    return "LeChalutier is running"

@app.route('/optimise', methods=['POST'])
def optimise():
    if not req.json or not 'currencies' in req.json:
        return make_response(jsonify({'error': 'Wrong usage'}), 400)
    return make_response(jsonify(optimiz.optimiz(req.json['currencies'], debug)), 200)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=debug, host= '0.0.0.0', port=args['port'])
