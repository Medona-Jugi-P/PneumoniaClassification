#!/bin/bash

set -o errexit

# Upgrade pip to the latest version
pip install --upgrade pip

# Install Python dependencies from requirements.txt
pip install -r requirement.txt
