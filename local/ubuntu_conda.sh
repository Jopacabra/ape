#!/bin/bash
# You should have already run "ubuntu_prereqs.sh" before running this script.
# You should have already run "conda init" and restarted terminal before running this script.
# Make sure to run this script without elevated permissions
# (e.g. "bash ubuntu_conda.sh", NOT "sudo bash ubuntu_conda.sh")

# create conda virtual environment with dependencies
conda config --add channels conda-forge
conda create --yes --prefix ~/anaconda3/envs/ape python=3.11.9 numpy scipy cython h5py pandas xarray pyyaml fastparquet pythia8 lhapdf jupyter matplotlib

# Activate conda environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/ape

# Go to home directory
cd ~

# Install fragmentation functions
lhapdf install "JAM20-SIDIS_FF_hadron_nlo"
git clone -n --depth=1 --filter=tree:0 https://github.com/QCDHUB/JAM22.git
cd JAM22
git sparse-checkout set --no-cone /JAM22-FF_hadron_nlo
git checkout
mv ~/JAM22/JAM22-FF_hadron_nlo ~/anaconda3/envs/ape/share/LHAPDF/
rm -rf ~/JAM22