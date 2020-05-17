# Umbrella Integrator
*Umbrella Integration of PMF calculations - 1D & 2D*

#### Version:  0.2 - 18/05/2020

### Usage
```
  umbrellaint.py --dim <XD> [options]
```

### References
Based on the Umbrella Integration method developed by Johannes Kästner and Walter Thiel

    Kästner, J., & Thiel, W. J Chem Phys. 2005, 123(14), 144104
    Kästner, J. J Chem Phys. 2009, 131(3), 034109

### Requeriments
Python3 recommended \
Supported: Python 2.7 and Python 3.7

Packages:
  - NumPy
  - SciPy - 2D Real Space Grid integration

### Input format

Currently supported fDynamo style files.

First line: Force Constant and Reference Distance \
Rest of lines: Distances Sampled \
File naming: 1D: dat_x.# | 2D: dat_x.#.# dat_y.#.#

### Direct Download
`wget https://raw.githubusercontent.com/boneta/UmbrellaIntegrator/master/umbrellaint.py`
