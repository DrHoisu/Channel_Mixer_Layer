#!/usr/bin/env bash

# run this script on the root
cd data

# Download iono_electron dataset in tensorflow format
wget https://zenodo.org/record/13165939/files/iono_electron.zip
unzip iono_electron.zip
rm iono_electron.zip

echo "finished"


# Download and arrange them in the following structure:
# Channel_Mixer_Layer
# └── data
#     ├── iono_electron
#     │   ├── test
#     │   ├── train
