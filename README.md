# Anisotropic Parton Evolution (APE)
Anisotropic parton evolution in hydrodynamic QGP grids generated with the DukeQCD hic-eventgen package (https://github.com/Duke-QCD/hic-eventgen).

### Dependencies
#### External Python modules required:

* scipy
* numpy
* cython
* h5py
* pandas
* xarray
* py-yaml
* hic
* tkinter (For plasma inspector only)
* matplotlib (For plotting only)
* jupyter (For sample scripts only)

#### Trento & OSU-Hydro dependencies

* Python 3.5+ with numpy, scipy, cython, and h5py
* C, C++, and Fortran compilers
* CMake 3.4+
* Boost and HDF5 C++ libraries

#### Requires LHAPDF 

* Installed separately, e.g. as in the included install.sh script
* Requires Python interface

#### Requires Pythia

* Installed separately, e.g. as in the included install.sh script
* Requires Python interface

### Usage

APE was built specifically for deployment on the Open Science Grid (OSG) via container distribution. Included is a container build script, an event running script, and a submission script in the "osg" folder. The workflow goes as follows:

1. Build the container with apptainer:
    e.g.: apptainer build container.sif image.def
2. Update the submission script with the location of the container image on the OSPool.
3. Submit script. Make sure to edit config.yml, rename to user_config.yml, and pass as input file with submission script. No default configuration is provided.
4. Collect result "*.pickle" files containing parton results

### Data & Analysis
...
