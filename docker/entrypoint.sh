#!/bin/bash

pixplot --images "/home/images/*.jpg" --metadata "/home/metadata.csv"
python -m http.server 5000