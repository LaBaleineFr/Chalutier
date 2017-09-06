#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import json

from optimiz.optimiz import optimiz

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False)
parser.add_argument('--currencies', nargs='+')

args = vars(parser.parse_args())

debug = args['debug']

if args['currencies'] is not None:
  print(json.dumps(optimiz(args['currencies'], debug), indent=2))
