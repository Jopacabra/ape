#!/bin/bash
# For a local install, run this script first. Run this script with elevated permissions
# (e.g. sudo bash ubuntu_prereqs.sh)

# Update apt-get stuff
sudo apt-get -y update

# Install dependencies -- mostly for Duke event generator
sudo apt-get -y install build-essential libtool autoconf unzip wget
sudo apt-get -y install cmake
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install libhdf5-serial-dev