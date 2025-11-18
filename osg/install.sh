#!/bin/bash
# This file is meant to be run while building a container for deployment on the OSG.

# Install fragmentation functions
lhapdf install "JAM20-SIDIS_FF_hadron_nlo"
cd /usr
git clone -n --depth=1 --filter=tree:0 https://github.com/QCDHUB/JAM22.git
cd JAM22
git sparse-checkout set --no-cone /JAM22-FF_hadron_nlo
git checkout
mv /usr/JAM22/JAM22-FF_hadron_nlo /usr/conda/ape/share/LHAPDF/

# Go to the project root
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
pwd

# Install hic - required for soft particle v2 analysis
# subshell allows temporary environment modification
cd hic
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python3 setup.py install
) || exit 1
cd ..

# Install freestream - required before installing osu-hydro
# subshell allows temporary environment modification
cd freestream
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python3 setup.py install
) || exit 1
cd ..

# Install frzout - required before installing osu-hydro
cd frzout
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python3 setup.py install
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
cmake -DCMAKE_INSTALL_PREFIX=/usr -DNATIVE=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=True ..
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
cmake -DCMAKE_INSTALL_PREFIX=/usr -DNATIVE=OFF ..
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
cmake -DCMAKE_INSTALL_PREFIX=/usr -DNATIVE=OFF ..
# Install the module
make install
cd ..
cd ..

