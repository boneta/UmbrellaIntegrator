# Umbrella Integrator

![python](https://img.shields.io/badge/python-3.8+-red.svg)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


*Umbrella Integration of PMF calculations - 1D & 2D*


## Usage
```
  umbrellaint --dim <#> [options] [-h]
```

## Installation
```bash
  pip install git+https://github.com/boneta/UmbrellaIntegrator.git
```
### Requirements
*Python 3.8 - 3.11*  
*Python 3.12+* is not supported for direct installation due to the deprecation of the `distutils` module of `numpy`, but it works fine if installed manually.

A Fortran compiler (tested with *gfortran*)

Packages:
  - NumPy
  - SciPy (only needed for 2D)
  - Meson and Ninja (only needed for Python 3.12+)

### Manual Installation
It can also be installed by cloning/downloading the source code from the GitHub repository.  
Then, to take advantage of extremely fast functions written in Fortran, a f2py module must be compiled.  
After that, ensure that the UmbrellaIntegrator directory can be found in the `PYTHONPATH`.
```bash
  git clone https://github.com/boneta/UmbrellaIntegrator.git
  pip install -r UmbrellaIntegrator/requirements.txt
  make -C UmbrellaIntegrator
```

## Input format

Currently supported fDynamo style files. Examples are provided.

First line: Force Constant and Reference Distance  
Rest of lines: Distances Sampled  
Default file naming: 1D: dat_1.# || 2D: dat_1.#.# dat_2.#.#


## Running options
Built-in help (-h).  
Only mandatory parameter is the dimension of the PMF: 1 or 2  
Temperature (in Kelvin) and output units (kj/mol or kcal/mol) can be chosen. Default: 298.0K and kj/mol  
Relative location of the input files with '--path'

#### 1 Dimension
The coordinate is split into equally spaced bins, local derivatives are calculated on each according to umbrella integration equations and the whole trajectory is integrated. The number of bins to use is set with '--bins'.

#### 2 Dimensions
Two working modes available:

 - *Irregular/Incomplete Grid* : Default mode. Surfaces of any shape, irregularly filled and/or missing points are welcomed. Local derivatives are calculated into an equally spaced grid with '--dist' distance between points. The surface is only generated in areas containing input values. Recommended method if not restricted to a perfect rectangular result.
 
 - *Regular/Rectangular Grid* : Activated with '--regular'. The PMF points are placed in a matrix according to their file name. The output is a grid with the same limits and a density controlled by '--gridf' parameter and based on the initial files. Example: 60x30 with grid=1.2 -> 72x36. Missing input files will lead to errors.

#### Examples
Download the example files from [here](https://github.com/boneta/UmbrellaIntegrator/releases/latest)  
`umbrellaint --dim 1 --out pmf_1d.dat`  
`umbrellaint --dim 1 --out pmf_1d.dat --path examples/1D --temp 298.15 --units kcal --bins 5000`

`umbrellaint --dim 2 --out pmf_2d.dat`  
`umbrellaint --dim 2 --temp 328. --dist 0.04`
`umbrellaint --dim 2 --out pmf_2d.dat --path ../examples/2D/ --temp 313 --regular --gridf 1.5`  


## How to cite
  > Boneta, S., _Umbrella Integrator_, 2021, https://github.com/boneta/UmbrellaIntegrator


## References
Based on the _Umbrella Integration_ method developed by Johannes Kästner and Walter Thiel

  > Kästner, J. & Thiel, W., _J Chem Phys._, 2005, 123(14), 144104, https://doi.org/10.1063/1.2052648  
  > Kästner, J., _J Chem Phys._, 2009, 131(3), 034109, https://doi.org/10.1063/1.3175798
