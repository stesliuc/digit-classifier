#!/bin/bash

# This will cause bash to stop executing the script if there's an error
set -e

# Run scraper
python scripts/run_classifier.py --data_folder "data/"

