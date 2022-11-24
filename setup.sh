#!/bin/bash

virtualenv --python=python3.8 venv

source venv/bin/activate

python -m pip install -r requirements.txt