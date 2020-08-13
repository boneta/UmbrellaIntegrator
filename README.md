# Umbrella Integrator
*Umbrella Integration of PMF calculations - 1D & 2D*

#### Version:  0.3 - 13/08/2020

### Usage
```
  umbrellaint.py --dim <XD> [options]
```

### References
Based on the Umbrella Integration method developed by Johannes Kästner and Walter Thiel

    Kästner, J., & Thiel, W. J Chem Phys. 2005, 123(14), 144104
    Kästner, J. J Chem Phys. 2009, 131(3), 034109

### Requeriments
Python 3 strongly recommended \
Supported: Python 2.7 and Python 3.7

Packages:
  - NumPy
  - SciPy (only for 2D)

Extremely fast functions written in Fortran can be used for irregular grid (igrid) method in 2D. To use them, a f2py module must be compiled once: \
`python3 -m numpy.f2py -c umbrellaint_fortran.f90 -m umbrellaint_fortran --opt='-Ofast'`

### Input format

Currently supported fDynamo style files.

First line: Force Constant and Reference Distance \
Rest of lines: Distances Sampled \
File naming: 1D: dat_x.# || 2D: dat_x.#.# dat_y.#.#
