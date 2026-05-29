#!/bin/bash
# You should have already run "ubuntu_prereqs.sh" before running this script.
# You should have already run "conda init" and restarted terminal before running this script.
# Make sure to run this script without elevated permissions
# (e.g. "bash ubuntu_conda.sh", NOT "sudo bash ubuntu_conda.sh")

# create conda virtual environment with dependencies
export VIRTUAL_ENV='ape'
conda config --add channels conda-forge
conda create --yes --prefix ~/anaconda3/envs/$VIRTUAL_ENV numpy scipy cython h5py pandas pyyaml pyarrow conda-forge::pythia8 conda-forge::lhapdf jupyter matplotlib conda-forge::fastjet conda-forge::pyhepmc pytorch::pytorch

# Activate conda environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/$VIRTUAL_ENV

# Go to home directory
cd ~

# Install fragmentation functions
lhapdf install "JAM20-SIDIS_FF_hadron_nlo"
git clone -n --depth=1 --filter=tree:0 https://github.com/QCDHUB/JAM22.git
cd JAM22
git sparse-checkout set --no-cone /JAM22-FF_hadron_nlo
git checkout
mv ~/JAM22/JAM22-FF_hadron_nlo ~/anaconda3/envs/$VIRTUAL_ENV/share/LHAPDF/
rm -rf ~/JAM22