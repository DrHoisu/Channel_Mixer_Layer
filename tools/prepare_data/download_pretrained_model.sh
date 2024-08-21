#!/usr/bin/env bash

# run this script on the root

# Download iono_electron dataset in tensorflow format
wget https://zenodo.org/record/13349678/files/work_dirs.zip
unzip work_dirs.zip
rm work_dirs.zip

echo "finished"


# Download and arrange them in the following structure:
# Channel_Mixer_Layer
# └── work_dirs
#     ├── iono
#     │   ├── convlstm
#     │   ├── e3dlstm
