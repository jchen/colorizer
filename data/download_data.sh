#!/bin/bash
# Run this from the /data directory to download the datasets (MIT Places)
wget http://data.csail.mit.edu/places/places365/test_256.tar
mkdir train
tar -xvf test_256.tar -C train/.
wget http://data.csail.mit.edu/places/places365/val_256.tar
mkdir test
tar -xvf val_256.tar -C test/.