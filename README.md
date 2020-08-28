# Umbrella Integrator
![GitHub version](https://img.shields.io/badge/version-0.4.0-brightgreen.svg)
![Last Uptade](https://img.shields.io/badge/%F0%9F%93%85%20last%20update%20-%2028--08--2020-green.svg)
![python](https://img.shields.io/badge/python-3.7-red.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

\
*Umbrella Integration of PMF calculations - 1D & 2D*


## Usage
```
  umbrellaint.py --dim <#> [options] [-h]
```

## Requirements
*Python 3.7* \
Compatible with Python 2 (no warranty provided)

Packages:
  - NumPy
  - SciPy (optional for 1D)

Extremely fast functions written in Fortran can be used for incomplete grid ('igrid') method in 2D. To take advantage of them, a f2py module must be compiled once: \
`python3 -m numpy.f2py -c umbrellaint_fortran.f90 -m umbrellaint_fortran`


## Input format

Currently supported fDynamo style files.

First line: Force Constant and Reference Distance \
Rest of lines: Distances Sampled \
File naming: 1D: dat_x.# || 2D: dat_x.#.# dat_y.#.#


## Running options
Built-in help argument (-h).\
Only mandatory parameter is the dimension of the PMF: 1 or 2\
Temperature (in Kelvin) and output units (kj/mol or kcal/mol) can be chosen. Default: 298.0K and kj/mol\
Relative location of the input files is set with '--path'

###### 1 Dimension
The coordinate is split into equally spaced bins, local gradients are calculated on each according to umbrella integration equations and the whole trajectory is integrated. The number of bins to use is set with '--bins'. Default: 2000

###### 2 Dimension
Two working modes available:

 - *Rectangular grid* : Default mode. The PMF points are forced into a matrix according to their file name. The output is a grid with the same limits and a density controlled by '--grid' parameter. Example: 60x30 with grid=1.2 -> 72x36. Missing input files will lead to errors.

 - *Incomplete grid* : Activated with '--igrid'. Surfaces of any shape, irregularly filled and/or missing points are welcomed. Local gradients are interpolated into a equally spaced grid with '--idist' distance between points. The surface is only integrated in areas with input values. It is the recommended method if not forced to a perfect rectangular result. It is faster, as it can take advantage of subroutines written in Fortran.

###### Examples
`umbrellaint.py --dim 1 --out pmf_1d.dat`\
`umbrellaint.py --dim 1 --out pmf_1d.dat --path ../pmf_1d --temp 313.5 --units kcal --bins 5000`

`umbrellaint.py --dim 2 --out pmf_2d.dat`\
`umbrellaint.py --dim 2 --out pmf_2d.dat --path pmf_2d --temp 306 --grid 1.5`\
`umbrellaint.py --dim 2 --temp 328. --igrid --idist 0.04`


## How to cite
> S. Boneta, _Umbrella Integrator_, 2020, https://github.com/boneta/UmbrellaIntegrator

## References
Based on the Umbrella Integration method developed by Johannes Kästner and Walter Thiel

  > Kästner, J. & Thiel, W. _J Chem Phys._ 2005, 123(14), 144104, https://doi.org/10.1063/1.2052648 \
  > Kästner, J. _J Chem Phys._ 2009, 131(3), 034109, https://doi.org/10.1063/1.3175798
