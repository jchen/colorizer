#!/bin/bash
# Run this from the /data directory. 
wget http://data.csail.mit.edu/places/places365/test_256.tar
tar -xvf test_256.tar -C train/.
wget http://data.csail.mit.edu/places/places365/val_256.tar
tar -xvf val_256.tar -C test/.