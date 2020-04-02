#!/bin/bash

python pixplot.py --images "/home/images/*.jpg" --metadata "/home/metadata.csv" --image_size 299 299 --model inception
echo "Start server"
python -m http.server 5000