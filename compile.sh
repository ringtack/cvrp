#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# If activate script doesn't exist, create new venv and install requirements
if [ ! -f cvrp/bin/activate ]; then
  python3 -m venv cvrp
  pip3 install -r requirements.txt
fi