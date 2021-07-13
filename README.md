# Umbrella Integrator

![GitHub version](https://img.shields.io/badge/version-0.5.5-brightgreen.svg)
![Last Uptade](https://img.shields.io/badge/%F0%9F%93%85%20last%20update%20-%2006--11--2020-green.svg)
![python](https://img.shields.io/badge/python-3.7-red.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


*Umbrella Integration of PMF calculations - 1D & 2D*


## Usage
```
  umbrellaint.py --dim <#> [options] [-h]
```

## Requirements
*Python 3.7* \
Python 2 compatible (no warranty)

Packages:
  - NumPy
  - SciPy (only needed for 2D)

Extremely fast functions written in Fortran are optionally available. To take advantage of them, a f2py module must be compiled once. Tested with gfortran.

`python3 -m numpy.f2py -c umbrellaint_fortran.f90 -m umbrellaint_fortran`


## Input format

Currently supported fDynamo style files. Examples are provided.

First line: Force Constant and Reference Distance \
Rest of lines: Distances Sampled \
Default file naming: 1D: dat_1.# || 2D: dat_1.#.# dat_2.#.#


## Running options
Built-in help (-h).\
Only mandatory parameter is the dimension of the PMF: 1 or 2\
Temperature (in Kelvin) and output units (kj/mol or kcal/mol) can be chosen. Default: 298.0K and kj/mol\
Relative location of the input files with '--path'

#### 1 Dimension
The coordinate is split into equally spaced bins, local derivatives are calculated on each according to umbrella integration equations and the whole trajectory is integrated. The number of bins to use is set with '--bins'.

#### 2 Dimensions
Two working modes available:

 - *Rectangular grid* : Default mode. The PMF points are placed in a matrix according to their file name. The output is a grid with the same limits and a density controlled by '--grid' parameter and based on the initial files. Example: 60x30 with grid=1.2 -> 72x36. Missing input files will lead to errors.

 - *Incomplete grid* : Activated with '--igrid'. Surfaces of any shape, irregularly filled and/or missing points are welcomed. Local derivatives are calculated into an equally spaced grid with '--idist' distance between points. The surface is only generated in areas containing input values. Recommended method if not restricted to a perfect rectangular result.

#### Examples
`umbrellaint.py --dim 1 --out pmf_1d.dat`\
`umbrellaint.py --dim 1 --out pmf_1d.dat --path examples/1D --temp 298.15 --units kcal --bins 5000`

`umbrellaint.py --dim 2 --out pmf_2d.dat`\
`umbrellaint.py --dim 2 --out pmf_2d.dat --path ../examples/2D/ --temp 313 --grid 1.5`\
`umbrellaint.py --dim 2 --temp 328. --igrid --idist 0.04`


## How to cite
  > Boneta, S., _Umbrella Integrator_, 2020, https://github.com/boneta/UmbrellaIntegrator

## References
Based on the _Umbrella Integration_ method developed by Johannes Kästner and Walter Thiel

  > Kästner, J. & Thiel, W., _J Chem Phys._, 2005, 123(14), 144104, https://doi.org/10.1063/1.2052648 \
  > Kästner, J., _J Chem Phys._, 2009, 131(3), 034109, https://doi.org/10.1063/1.3175798
