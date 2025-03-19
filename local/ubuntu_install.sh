#!/bin/bash
# You should have already installed the prerequisites, e.g. with ubuntu_prereqs.sh
# You should have already made a conda environment called "ape", e.g. by running
# the ubuntu_conda.sh script.
# Ensure that you activate your environment with e.g. conda activate ape

# Get directory of script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Activate conda
source ~/anaconda3/bin/activate ~/anaconda3/envs/ape
#conda activate ape

# Debug print of working directory
# Go to ape directory
cd $SCRIPT_DIR/..
pwd

# Install hic - required for soft particle v2 analysis
# subshell allows temporary environment modification
cd hic
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  pip install .
) || exit 1
cd ..

# Install freestream - required before installing osu-hydro
# subshell allows temporary environment modification
cd freestream
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  pip install .
) || exit 1
cd ..

# Install frzout - required before installing osu-hydro
cd frzout
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  pip install .
) || exit 1
cd ..

# Build trento
cd trento
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
fi
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
# Note that we install as root but run in the read-only file system without root.
# We install into /usr so we can access the binaries
# We select to set native architecture optimization off.
# This causes problems with many osg job sites.
# We only call trento once, and it accounts for insignificant portion of compute time.
# Better to be safe than save a few seconds.
cmake -DCMAKE_INSTALL_PREFIX=~ -DNATIVE=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=True ..
# Install the module
make install
cd ..
cd ..

# Build osu-hydro
cd osu-hydro
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
fi
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
# Note that we install as root but run in the read-only file system without root.
# We install into /usr so we can access the binaries
# We select to set native architecture optimization off.
# This causes problems with many osg job sites.
cmake -DCMAKE_INSTALL_PREFIX=~ -DNATIVE=OFF ..
# Install the module
make install
cd ..
cd ..

# Build UrQMD
cd urqmd-afterburner
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
fi
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
# Note that we install as root but run in the read-only file system without root.
# We install into /usr so we can access the binaries
# We select to set native architecture optimization off.
# This causes problems with many osg job sites.
cmake -DCMAKE_INSTALL_PREFIX=~ -DNATIVE=OFF -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch=True ..
# Install the module
make install
cd ..
cd ..