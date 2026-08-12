#!/bin/bash
# You should have already run "ubuntu_prereqs.sh" before running this script.
# You should have already run "conda init" and restarted terminal before running this script.
# Make sure to run this script without elevated permissions
# (e.g. "bash ubuntu_conda.sh", NOT "sudo bash ubuntu_conda.sh")
# Due to the nature of conda's environment solver, this script is likely to take over an hour sitting on
# "Solving environment". Be patient and budget a lot of time for it.

# Accept terms of service by default
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create conda virtual environment with dependencies
export VIRTUAL_ENV='ape'
conda config --add channels conda-forge
conda config --add channels pytorch
conda create --yes --name $VIRTUAL_ENV python=3.12.13 numpy scipy cython h5py \
      pandas pyyaml pyarrow matplotlib jupyter \
      conda-forge::pythia8 conda-forge::lhapdf conda-forge::fastjet conda-forge::pyhepmc \
      pytorch::pytorch

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $VIRTUAL_ENV

# Install fragmentation functions
cd ~ || exit
lhapdf install "JAM20-SIDIS_FF_hadron_nlo"
git clone -n --depth=1 --filter=tree:0 https://github.com/QCDHUB/JAM22.git
cd JAM22 || exit
git sparse-checkout set --no-cone /JAM22-FF_hadron_nlo
git checkout
mv ~/JAM22/JAM22-FF_hadron_nlo $(conda info --base)/envs/$VIRTUAL_ENV/share/LHAPDF/
rm -rf ~/JAM22