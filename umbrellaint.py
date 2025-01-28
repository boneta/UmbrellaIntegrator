#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''

              ,·'´
         __. /.__
     .-"'..':`..`"-.          _   _ __  __ ___ ___ ___ _    _      _
   .´ .' .  :  . `. `.       | | | |  \/  | _ ) _ \ __| |  | |    /_\
 /_ _._ _.._:_.._ _._ _\     | |_| | |\/| | _ \   / _|| |__| |__ / _ \
   '   '    |    '   '        \___/|_|  |_|___/_|_\___|____|____/_/ \_\
            |           ___ _  _ _____ ___ ___ ___    _ _____ ___  ___
            |          |_ _| \| |_   _| __/ __| _ \  /_\_   _/ _ \| _ \
            |           | || .` | | | | _| (_ |   / / _ \| || (_) |   /
            |          |___|_|\_| |_| |___\___|_|_\/_/ \_\_| \___/|_|_\
           /
        .-'                                   method by Kästner & Thiel
                                           implemented by Sergio Boneta


Functions
---------

    read_dynamo_1D
    read_dynamo_2D_rgrid
    read_dynamo_2D_igrid
    umbrella_integration_1D
    umbrella_integration_2D_rgrid
    umbrella_integration_2D_igrid
    igrid_gen
    igrid_topol
    igrid_grad
    integration_2D_rgrid
    integration_2D_igrid
    write_1D
    write_2D_rgrid
    write_2D_igrid

'''

__version__ = '0.6.1'
__author__ = 'Sergio Boneta'

#######################################################################
##                                                                   ##
##                        Umbrella Integrator                        ##
##                                                                   ##
#######################################################################
#
# Copyright (C) 2021, Sergio Boneta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/
#
#
# Based on the method developed by Johannes Kästner and Walter Thiel
# Kästner, J., & Thiel, W. J Chem Phys. 2005, 123(14), 144104
# Kästner, J., & Thiel, W. J Chem Phys. 2006, 124(23), 234106
# Kästner, J. J Chem Phys. 2009, 131(3), 034109
#
# Check for updates:  https://github.com/boneta/UmbrellaIntegrator


#######################################################################
##  DEPENDENCIES                                                     ##
#######################################################################
## Support: Python 3.8+

import argparse
import os
import sys
from glob import glob

import numpy as np
from numpy import ndarray

# fortranized key functions
try:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import umbrellaint_fortran
except:
    sys.stdout.write("WARNING: Umbrella Integrator's fortran subroutines could not be imported\n" )
    umbrellaint_fortran = None

# scipy
try:
    from scipy import optimize as scipy_optimize
except ImportError:
    scipy_optimize = None


#######################################################################
##  CONSTANTS                                                        ##
#######################################################################

## Predefined variables
equilibration = 0           # initial equilibration frames to skip
impossible = 0.123456789    # unlikely value used to denote 'False' in float format


## Thermodynamical constants
# http://physics.nist.gov/cuu/Constants/index.html
_c       = 299792458.             # m * s-1
_h       = 6.62607015e-34         # J * s
_kB      = 1.380649e-23           # J * K-1
_NA      = 6.02214076e23          # mol-1
_R       = _kB * _NA              # J * K-1 * mol-1

# conversions
_tau     = 2.*np.pi               # 2 * pi
_cal2J   = 4.1868                 # cal -> J
_J2cal   = 1./_cal2J              # J -> cal

# redefinition of kB (kJ and per mol)
kB = _R * 0.001                   # kJ * K-1 * mol-1


#######################################################################
##  FUNCTIONS                                                        ##
#######################################################################

def read_dynamo_1D(
        directory: str,
        name: str = 'dat_1',
        equilibration: int = 0,
        minsteps: int = 0,
        printread: bool = True
        ) -> tuple:
    '''
        Read 1D data from fDynamo files into arrays

        Parameters
        ----------
        directory : str
            path to the files
        name : str, optional
            prefix of files (def: 'dat_1')
        equilibration : int, optional
            number of steps considered equilibration and excluded (def: 0)
        minsteps : int, optional
            minimum number of steps to be taken in consideration (def: 0)
        printread : bool, optional
            print file names while reading (def: True)

        Returns
        -------
        n_i : int
            number of windows simulated
        a_fc : ndarray(n_i)
            array of force constants
        a_rc0 : ndarray(n_i)
            array of reference coordinates
        a_mean : ndarray(n_i)
            array of mean coordinates
        a_std : ndarray(n_i)
            array of standard deviation of coordinates
        a_N : ndarray(n_i)
            array of number of samples for each window
        limits : ndarray(2)
            array of minimum and maximum coordinates
    '''

    sys.stdout.write("# Reading input files\n")

    # get 'dat_*' file lists
    coor1 = glob(os.path.join(directory, f"{name}.*"))
    if len(coor1) == 0:
        raise NameError(f"No {name}.* files found on the specified path")
    coor1.sort()

    # number of windows
    n_i = int(len(coor1))

    # initialize 1D matrices
    a_rc0   = np.zeros((n_i), dtype=float)                             # initial distance set
    a_mean  = np.zeros_like(a_rc0)                                     # mean of distance
    a_fc    = np.zeros_like(a_rc0)                                     # force constants
    a_std   = np.zeros_like(a_rc0)                                     # standard deviation of distance
    a_N     = np.zeros((n_i), dtype=int)                               # number of samples
    removef = []                                                       # files to remove

    # read 'dat_*' files
    for i, fx in enumerate(coor1):
        # check if empty file
        if os.stat(fx).st_size == 0:
            removef.append(i)
            continue
        # open/read file
        datx = open(fx).readlines()
        # force and initial distance
        a_fc[i], a_rc0[i] = datx.pop(0).split()
        # convert to numpy, ignoring equilibration part
        datx = np.asarray(datx[equilibration:], dtype=float)
        # mean, standard deviation and number of samples
        a_mean[i] = np.mean(datx)
        a_std[i]  = np.std(datx)
        a_N[i] = len(datx)
        # remove if not minimum steps
        if a_N[i] < minsteps:
            removef.append(i)
            continue
        # print progress
        if printread:
            sys.stdout.write(f"{fx:s}  -  {a_N[i]:d}\n")

    # delete empty files
    if removef:
        n_i   -= len(removef)
        a_rc0  = np.delete(a_rc0, removef)
        a_mean = np.delete(a_mean, removef)
        a_fc   = np.delete(a_fc, removef)
        a_std  = np.delete(a_std, removef)
        a_N    = np.delete(a_N, removef)

    # check if any suitable data left
    if n_i == 0:
        raise ValueError('No suitable files found')

    # build data matrix and get limits
    limits = np.zeros((2), dtype=float)
    limits = np.min(a_mean), np.max(a_mean)

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write(f"\n# No Windows:  {n_i:d}\n")
    sys.stdout.write(f"# No Samples (Avg):  {np.mean(a_N):<10.2f}\n")
    sys.stdout.write(f"# x_min: {limits[0]:>7.3f}    x_max: {limits[1]:>7.3f} \n\n")
    sys.stdout.flush()

    # return results
    return n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits

def read_dynamo_2D_rgrid(
        directory: str,
        name1: str = 'dat_1',
        name2: str = 'dat_2',
        equilibration: int = 0,
        printread: bool = True
        ) -> tuple:
    '''
        Read 2D data from fDynamo files into matrices from a regular/rectangular grid

        Parameters
        ----------
        directory : str
            path to the files
        name1 : str, optional
            prefix of x files (def: 'dat_1')
        name2 : str, optional
            prefix of y files (def: 'dat_2')
        equilibration : int, optional
            number of steps considered equilibration and excluded (def: 0)
        printread : bool, optional
            print file names while reading (def: True)

        Returns
        -------
        n_i : int
            number of i windows simulated
        n_j : int
            number of j windows simulated
        m_fc : ndarray(n_j,n_i,2,2)
            matrix of force constants matrices
        m_rc0 : ndarray(n_j,n_i,2)
            matrix of reference coordinates
        m_mean : ndarray(n_j,n_i,2)
            matrix of mean coordinates
        m_covar : ndarray(n_j,n_i,2,2)
            matrix of covariance matrices of coordinates
        m_N : ndarray(n_j,n_i)
            matrix of number of samples for each window
        limits : ndarray(2,2)
            matrix of minimum and maximum coordinates
    '''

    sys.stdout.write("# Reading input files\n")

    # get 'dat_*' file lists
    coor1 = glob(os.path.join(directory, f"{name1}.*"))
    coor2 = glob(os.path.join(directory, f"{name2}.*"))
    if len(coor1) == 0 or len(coor2) == 0:
        raise NameError(f"No {name1}.*/{name2}.* files found on the specified path")
    if len(coor1) != len(coor2):
        raise NameError(f"Different number of {name1}.* and {name2}.* files")
    coor1.sort()
    coor2.sort()

    # number of windows on every axis
    name, n_i, n_j = os.path.basename(coor1[-1]).split('.')
    # name, n_i, n_j = coor1[-1].split('.')
    n_i = int(n_i) + 1
    n_j = int(n_j) + 1

    # initialize matrices 3D: rectangular jxi
    m_rc0   = np.zeros((n_j, n_i, 2), dtype=float)                     # initial distance set (depth 2)
    m_mean  = np.zeros_like(m_rc0)                                     # mean of distance (depth 2)
    m_fc    = np.zeros((n_j, n_i, 2, 2), dtype=float)                  # force constants matrices (2,2)
    m_covar = np.zeros_like(m_fc)                                      # covariance of distance matrices (2,2)
    m_N     = np.zeros((n_j, n_i), dtype=int)                          # number of samples (2D matrix)

    # read 'dat_*' files
    line0 = np.zeros((2,2), dtype=float)
    for fx, fy in zip(coor1, coor2):
        # check if empty file
        if os.stat(fx).st_size == 0 or os.stat(fy).st_size == 0:
            raise ValueError(f"Empty file found: {fx} {fy}")
        # set i,j and open/read files
        name, i, j = os.path.basename(fx).split('.')
        i, j = int(i), int(j)
        datx = open(fx).readlines()
        daty = open(fy).readlines()
        # force matrix and initial distance from line0
        line0[0], line0[1] = datx.pop(0).split(), daty.pop(0).split()
        m_fc[j,i] = np.diagflat(line0[:,0])
        m_rc0[j,i] = line0[:,1]
        # convert to numpy, ignoring equilibration part and truncating to the shorter
        shorter = min(len(datx), len(daty))
        datx = np.asarray(datx[equilibration:shorter], dtype=float)
        daty = np.asarray(daty[equilibration:shorter], dtype=float)
        # mean, covariance and number of samples
        m_mean[j,i,:] = np.mean(datx), np.mean(daty)
        m_covar[j,i] = np.cov(datx, daty)
        m_N[j,i] = len(datx)
        # print progress
        if printread:
            sys.stdout.write(f"{fx:s}  {fy:s}  -  {m_N[j,i]:d}\n")

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(m_mean[:,:,0]), np.max(m_mean[:,:,0])
    limits[1] = np.min(m_mean[:,:,1]), np.max(m_mean[:,:,1])

    # correct force constant
    # m_fc = m_fc * 0.5

    # print information
    sys.stdout.write(f"\n# No Windows:  {n_i*n_j:d}\n")
    sys.stdout.write(f"# Surface dimensions:  {n_i:d} x {n_j:d}\n")
    sys.stdout.write(f"# No Samples (Avg):  {np.mean(m_N):<10.2f}\n")
    sys.stdout.write(f"# x_min: {limits[0,0]:>7.3f}    x_max: {limits[0,1]:>7.3f} \n"
                     f"# y_min: {limits[1,0]:>7.3f}    y_max: {limits[1,1]:>7.3f} \n\n")
    sys.stdout.flush()

    # return results
    return n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits

def read_dynamo_2D_igrid(
        directory: str,
        name1: str = 'dat_1',
        name2: str = 'dat_2',
        equilibration: int = 0,
        minsteps: int = 0,
        printread: bool = True
        ) -> tuple:
    '''
        Read 2D data from fDynamo files into arrays from an irregular/incomplete grid

        Parameters
        ----------
        directory : str
            path to the files
        name1 : str, optional
            prefix of x files (def: 'dat_1')
        name2 : str, optional
            prefix of y files (def: 'dat_2')
        equilibration : int, optional
            number of steps considered equilibration and excluded (def: 0)
        minsteps : int, optional
            minimum number of steps to be taken in consideration (def: 0)
        printread : bool, optional
            print file names while reading (def: True)

        Returns
        -------
        a_fc : ndarray(n_i,2,2)
            array of force constants matrices
        a_rc0 : ndarray(n_i,2)
            array of reference coordinates
        a_mean : ndarray(n_i,2)
            array of mean coordinates
        a_covar : ndarray(n_i,2,2)
            array of covariance matrices of coordinates
        a_N : ndarray(n_i,2)
            array of number of samples for each window
        limits : ndarray(2,2)
            array of minimum and maximum coordinates
    '''

    sys.stdout.write("# Reading input files\n")

    # get 'dat_*' file lists
    coor1 = glob(os.path.join(directory, f"{name1}.*"))
    coor2 = glob(os.path.join(directory, f"{name2}.*"))
    if len(coor1) == 0 or len(coor2) == 0:
        raise NameError(f"No {name1}.*/{name2}.* files found on the specified path")
    if len(coor1) != len(coor2):
        raise NameError(f"Different number of {name1}.* and {name2}.* files")
    coor1.sort()
    coor2.sort()

    # number of total windows
    n_i = len(coor1)

    # initialize arrays
    a_rc0   = np.zeros((n_i, 2), dtype=float)                          # initial distance set
    a_mean  = np.zeros_like(a_rc0)                                     # mean of distance
    a_fc    = np.zeros((n_i, 2, 2), dtype=float)                       # force constants matrices (2,2)
    a_covar = np.zeros_like(a_fc)                                      # covariance of distance matrices (2,2)
    a_N     = np.zeros((n_i), dtype=int)                               # number of samples
    removef = []                                                       # files to remove

    # read 'dat_*' files
    line0 = np.zeros((2,2), dtype=float)
    for i, (fx, fy) in enumerate(zip(coor1, coor2)):
        # check if empty file
        if os.stat(fx).st_size == 0 or os.stat(fy).st_size == 0:
            removef.append(i)
            continue
        # open/read files
        datx = open(fx).readlines()
        daty = open(fy).readlines()
        # force matrix and initial distance from line0
        line0[0], line0[1] = datx.pop(0).split(), daty.pop(0).split()
        a_fc[i] = np.diagflat(line0[:,0])
        a_rc0[i] = line0[:,1]
        # convert to numpy, ignoring equilibration part and truncating to the shorter
        shorter = min(len(datx), len(daty))
        datx = np.asarray(datx[equilibration:shorter], dtype=float)
        daty = np.asarray(daty[equilibration:shorter], dtype=float)
        # mean, covariance and number of samples
        a_mean[i,:] = np.mean(datx), np.mean(daty)
        a_covar[i] = np.cov(datx, daty)
        a_N[i] = len(datx)
        # remove if not minimum steps
        if a_N[i] < minsteps:
            removef.append(i)
            continue
        # print progress
        if printread:
            sys.stdout.write(f"{fx:s}  {fy:s}  -  {a_N[i]:d}\n")

    # delete empty files
    if removef:
        n_i   -= len(removef)
        a_rc0   = np.delete(a_rc0, removef, 0)
        a_mean  = np.delete(a_mean, removef, 0)
        a_fc    = np.delete(a_fc, removef, 0)
        a_covar = np.delete(a_covar, removef, 0)
        a_N     = np.delete(a_N, removef, 0)

    # check if any suitable data left
    if n_i == 0:
        raise ValueError("No suitable files found")

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(a_mean[:,0]), np.max(a_mean[:,0])
    limits[1] = np.min(a_mean[:,1]), np.max(a_mean[:,1])

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write(f"\n# No Windows:  {n_i:d}\n")
    sys.stdout.write(f"# No Samples (Avg):  {np.mean(a_N):<10.2f}\n")
    sys.stdout.write(f"# x_min: {limits[0,0]:>7.3f}    x_max: {limits[0,1]:>7.3f} \n"
                     f"# y_min: {limits[1,0]:>7.3f}    y_max: {limits[1,1]:>7.3f} \n\n")
    sys.stdout.flush()

    # return results
    return a_fc, a_rc0, a_mean, a_covar, a_N, limits

def umbrella_integration_1D(
        n_bins: int,
        n_i: int,
        a_fc: ndarray,
        a_rc0: ndarray,
        a_mean: ndarray,
        a_std: ndarray,
        a_N: ndarray,
        limits: ndarray,
        temp: float = 298.,
        integrator: str = 'simpson',
        fortranization: bool = True
        ) -> tuple:
    '''
        Umbrella Integration algorithm for 1D

        Parameters
        ----------
        n_bins : int
            number of bins for UI
        n_i : int
            number of windows simulated
        a_fc : ndarray(n_i)
            array of force constants
        a_rc0 : ndarray(n_i)
            array of reference coordinates
        a_mean : ndarray(n_i)
            array of mean coordinates
        a_std : ndarray(n_i)
            array of standard deviation of coordinates
        a_N : ndarray(n_i)
            array of number of samples for each window
        limits : ndarray(2)
            array of minimum and maximum coordinates
        temp : float, optional
            temperature (K) (def: 298.)
        integrator : {trapz, simpson}, optional
            integration algorithm (def: 'simpson')
        fortranization : bool, optional
            use faster functions writen in Fortran (def: True)

        Returns
        -------
        bins : ndarray(n_bins)
            array of coordinates of bins
        dA_bins : ndarray(n_bins)
            array of free energy derivatives
        A_bins : ndarray(n_bins)
            array of integrated free energy
    '''

    # check integrator specification
    if integrator not in {'trapz', 'simpson'}:
        raise NameError(f"Integrator '{integrator}' not recognized")

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau_sqrt = np.sqrt(_tau)
    a_var = a_std ** 2
    bins = np.linspace(limits[0],limits[1],n_bins)
    db = abs(bins[0] - bins[1])                         # space between bins
    dA_bins = np.zeros_like(bins)
    A_bins = np.zeros_like(bins)

    ## Derivatives ----------------------------------------------------

    if fortranization:
        sys.stdout.write(f"# Calculating derivatives - {n_bins} bins - Fortranized\n")
        dA_bins = umbrellaint_fortran.ui_derivate_1d(bins, a_fc, a_rc0, a_mean, a_std, a_N, beta)
    else:
        sys.stdout.write(f"# Calculating derivatives - {n_bins} bins \n")
        # normal probability [Kästner 2005 - Eq.5]
        def probability(rc, i):
            return np.exp(-0.5 * ((rc - a_mean[i])/a_std[i])**2) / (a_std[i] * tau_sqrt)
        # local derivative of free energy [Kästner 2005 - Eq.6]
        def dA(rc, i):
            return (rc - a_mean[i]) / (beta * a_var[i])  -  a_fc[i] * (rc - a_rc0[i])
        # normalization total [Kästner 2005 - Eq.8]
        def normal_tot(rc):
            return np.sum([a_N[i]*probability(rc,i) for i in range(n_i)])
        # calculate derivatives of free energy over the bins [Kästner 2005 - Eq.7]
        for ib in range(n_bins):
            rc = bins[ib]
            normal = normal_tot(rc)
            dA_bins[ib] = np.sum([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ])

    ## Integration ----------------------------------------------------

    ## composite trapezoidal rule integration [scipy.integrate.cumtrapz]
    if integrator == 'trapz':
        sys.stdout.write("# Integrating             - Trapezoidal \n\n")
        stp = db/2.
        for b in range(1, n_bins):
            A_bins[b] = A_bins[b-1] + stp * (dA_bins[b-1] + dA_bins[b])

    ## Simpson's rule integration
    elif integrator == 'simpson':
        sys.stdout.write("# Integrating             - Simpson's rule \n\n")
        stp = db/6.
        for b in range(1,n_bins-1):
            A_bins[b] = A_bins[b-1] + stp * (dA_bins[b-1] + 4*dA_bins[b] + dA_bins[b+1])
        A_bins[-1] = A_bins[-2] + db/2. * (dA_bins[-2] + dA_bins[-1])   # last point trapezoidal

    # set minimum to zero
    A_bins = A_bins - np.min(A_bins)

    # return results
    return bins, dA_bins, A_bins

def umbrella_integration_2D_rgrid(
        grid_f: float,
        n_i: int,
        n_j: int,
        m_fc: ndarray,
        m_rc0: ndarray,
        m_mean: ndarray,
        m_covar: ndarray,
        m_N: ndarray,
        limits: ndarray,
        temp: float = 298.,
        integrator: str = 'simpson+mini',
        integrate: bool = True,
        fortranization: bool = True
        ) -> tuple:
    '''
        Umbrella Integration algorithm for 2D from matrices to a regular/rectangular grid

        Parameters
        ----------
        grid_f : float
            multiplicative factor applied to the number
            of windows to calculate the grid dimensions
        n_i : int
            number of i windows simulated
        n_j : int
            number of j windows simulated
        m_fc : ndarray(n_j,n_i,2,2)
            matrix of force constants matrices
        m_rc0 : ndarray(n_j,n_i,2)
            matrix of reference coordinates
        m_mean : ndarray(n_j,n_i,2)
            matrix of mean coordinates
        m_covar : ndarray(n_j,n_i,2,2)
            matrix of covariance matrices of coordinates
        m_N : ndarray(n_j,n_i)
            matrix of number of samples for each window
        limits : ndarray(2,2)
            matrix of minimum and maximum coordinates
        temp : float, optional
            temperature (K) (def: 298.)
        integrator : {trapz, simpson, trapz+mini, simpson+mini, fourier}, optional
            integration algorithm (default: 'simpson+mini')
        integrate : bool, optional
            perform the surface integration (def: True)
            if False, only the free energy derivatives are
            calculated and A_grid returns as matrix of zeros
        fortranization : bool, optional
            use faster functions writen in Fortran (def: True)

        Returns
        -------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivatives
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy
    '''

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    m_prec = np.linalg.inv(m_covar[:,:])                  # precision matrices (inverse of covar)
    m_det_sqrt = np.sqrt(np.linalg.det(m_covar[:,:]))     # sqrt of determinants of covar matrix

    ## Grid -----------------------------------------------------------
    # grid dimensions
    n_ig = int(np.ceil(grid_f*n_i))
    n_jg = int(np.ceil(grid_f*n_j))
    # sides
    grid_x, dx = np.linspace(limits[0,0],limits[0,1],n_ig, retstep=True)
    grid_y, dy = np.linspace(limits[1,0],limits[1,1],n_jg, retstep=True)
    # grid_x, dx = np.linspace(m_mean[0,0,0], m_mean[-1,-1,0], n_ig, retstep=True)
    # grid_y, dy = np.linspace(m_mean[0,0,1], m_mean[-1,-1,1], n_jg, retstep=True)
    # grid building
    grid = np.zeros((n_jg,n_ig,2))
    for j in range(n_jg):
        for i in range(n_ig):
            grid[j,i] = grid_x[i], grid_y[j]
    # initialize gradient matrix
    dA_grid = np.zeros_like(grid)

    ## Derivatives ----------------------------------------------------

    if fortranization:
        sys.stdout.write(f"# Calculating derivatives - Regular Grid {n_ig:d} x {n_jg:d} - Fortranized\n")
        dA_grid = umbrellaint_fortran.ui_derivate_2d_rgrid(grid, m_fc, m_rc0, m_mean, m_prec, m_det_sqrt, m_N, beta)
    else:
        sys.stdout.write(f"# Calculating derivatives - Regular Grid {n_ig} x {n_jg}\n")
        # normal probability [Kästner 2009 - Eq.9]
        def probability(rc, j, i):
            diff = rc - m_mean[j,i]
            # return np.exp(-0.5 * np.matmul( diff, np.matmul(m_prec[j,i], diff) )) / (m_det_sqrt[j,i] * _tau)
            return np.exp( -0.5 * diff.dot(m_prec[j,i].dot(diff)) ) / (m_det_sqrt[j,i] * _tau)   # dot faster than matmul for small matrices
        # local derivative of free energy [Kästner 2009 - Eq.10]
        def dA(rc, j, i):
            # return  np.matmul((rc - m_mean[j,i])/beta, m_prec[j,i])  -  np.matmul((rc - m_rc0[j,i]), m_fc[j,i])
            return ((rc - m_mean[j,i])/beta).dot(m_prec[j,i])  -  (rc - m_rc0[j,i]).dot(m_fc[j,i])
        # normalization total [Kästner 2009 - Eq.11]
        def normal_tot(rc):
            return np.sum([m_N[j,i]*probability(rc,j,i) for i in range(n_i) for j in range(n_j)])
        # calculate normalization denominator for all the grid coordinates in advance --> same speed as on the fly calc, not worthy
        # def normal_denominator(): return np.asarray([ normal_tot(grid[j,i]) for j in range(n_jg) for i in range(n_ig)]).reshape(n_jg,n_ig)
        # calculate gradient field of free energy over grid [Kästner 2009 - Eq.11]
        for jg in range(n_jg):
            for ig in range(n_ig):
                rc = grid[jg,ig]
                normal = normal_tot(rc)
                dA_grid[jg,ig] = np.sum(np.transpose([ m_N[j,i]*probability(rc,j,i)/normal * dA(rc,j,i) for i in range(n_i) for j in range(n_j) ]), axis=1)
            # progress bar
            sys.stdout.write("\r# [{:21s}] - {:>6.2f}%".format( "■"*int(21.*(jg+1)/n_jg), (jg+1)/n_jg*100)); sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    ## Integration ----------------------------------------------------
    if integrate:
        A_grid = integration_2D_rgrid(grid, dA_grid, integrator)
    else:
        sys.stdout.write("# Integrating             - Skipping \n\n")
        A_grid = np.zeros((n_jg,n_ig))

    # return results
    return grid, dA_grid, A_grid

def igrid_gen(
        grid_d: float,
        a_coord: ndarray,
        thr_close: float = 1e-1,
        fortranization: bool = True
        ) -> ndarray:
    '''
        Irregular/incomplete grid generator in 2D

        Parameters
        ----------
        grid_d : float
            distance between grid points
        a_coord : ndarray(n_i,2)
            array of coordinates
        limits : ndarray(2,2)
            array of minimum and maximum coordinates
        thr_close : float, optional
            distance threshold to include grid points
            if are close to a data point (def: 1e-1)
        fortranization : bool, optional
            use faster functions writen in Fortran (def: True)

        Returns
        -------
        grid : ndarray(n_ig,2)
            array of grid coordinates
    '''

    # check if grid point is close to a data point
    if fortranization:
        a_coord = np.asfortranarray(a_coord)
        def point_in(rc):
            return bool(umbrellaint_fortran.point_in(rc,a_coord,thr_close))
    else:
        def point_in(rc):
            thr = thr_close**2        # threshold to consider close points
            coord_dist = (a_coord[:,0]-rc[0])**2 + (a_coord[:,1]-rc[1])**2
            return bool(min(coord_dist) < thr)

    # get limits
    limits_x = (np.min(a_coord[:,0]), np.max(a_coord[:,0]))
    limits_y = (np.min(a_coord[:,1]), np.max(a_coord[:,1]))

    # grid axes
    grid_x = np.arange(limits_x[0],limits_x[1]+grid_d/2.,grid_d)
    grid_y = np.arange(limits_y[0],limits_y[1]+grid_d/2.,grid_d)
    n_x = grid_x.shape[0]
    n_y = grid_y.shape[0]


    # grid building
    grid = [[grid_x[i], grid_y[j]] for j in range(n_y) for i in range(n_x) if point_in([grid_x[i], grid_y[j]])]
    grid = np.asarray(grid)

    # return final grid
    return grid

def igrid_topol(
        grid_d: float,
        grid: ndarray
        ) -> tuple:
    '''
        Irregular/incomplete grid topology guessing

        Parameters
        ----------
        grid_d : float
            distance between grid points
        grid : ndarray(n_ig,2)
            array of grid coordinates

        Returns
        -------
        grid_topol : ndarray(n_ig, 4)
            array of index of the points around
            down, up, left, right [[↓, ↑, ←, →]]
            if there is no points in a given
            direction self-index is placed
        grid_topol_d : ndarray(n_ig, 2)
            array of distance to the points around
            in horizonal and vertical direction
                2 points: grid_d * 2
                1 points: grid_d
                0 points: inf
    '''

    n_ig = grid.shape[0]
    # grid_d = round(min({abs(grid[i0,0] - grid[i1,0]) for i0 in range(n_ig) for i1 in range(n_ig)} - {0.}), 6)

    # grid_topol:  [[↓, ↑, ←, →]]
    grid_topol = np.zeros((n_ig, 4), dtype=int)
    # grid_topol_d:  [[horizonal_d, vertical_d]]
    grid_topol_d = np.zeros((n_ig, 2), dtype=float)
    # relative position reference
    position_ref = [[0., -grid_d],   # below
                    [0.,  grid_d],   # above
                    [-grid_d, 0.],   # left
                    [ grid_d, 0.]]   # right
    # grid topology guessing
    for ig in range(n_ig):
        rc = grid[ig]
        grid_rel = np.round(grid[:] - rc, 6)
        for n in range(4):
            grid_topol[ig,n] = ig
            for ig2 in range(n_ig):
                if grid_rel[ig2,0] == position_ref[n][0] and grid_rel[ig2,1] == position_ref[n][1]:
                    grid_topol[ig,n] = ig2
                    break
        # horizonal gradient
        if grid_topol[ig,2] != ig and grid_topol[ig,3] != ig: grid_topol_d[ig,0] = grid_d * 2.
        elif grid_topol[ig,2] != ig or grid_topol[ig,3] != ig: grid_topol_d[ig,0] = grid_d
        else: grid_topol_d[ig,0] = np.inf
        # vertical gradient
        if grid_topol[ig,0] != ig and grid_topol[ig,1] != ig: grid_topol_d[ig,1] = grid_d * 2.
        elif grid_topol[ig,0] != ig or grid_topol[ig,1] != ig: grid_topol_d[ig,1] = grid_d
        else: grid_topol_d[ig,1] = np.inf

    # grid_topol_d = 1 / grid_topol_d

    # return results
    return grid_topol, grid_topol_d

def igrid_grad(
        A_grid: ndarray,
        grid_topol: ndarray,
        grid_topol_d: ndarray
        ) -> ndarray:
    '''
        Irregular/incomplete grid topology gradient calculation

        Parameters
        ----------
        A_grid : ndarray(n_ig)
            array of grid values
        grid_topol : ndarray(n_ig, 4)
            array of index of the points around
        grid_topol_d : ndarray(n_ig, 2)
            array of distance to the points around
            in horizonal and vertical direction

        Returns
        -------
        dgrid : ndarray(n_ig,2)
            array of grid gradients,
            horizonal and vertical
    '''

    dgrid = np.zeros((A_grid.shape[0],2))
    # horizonal gradient
    dgrid[:,0] = A_grid[grid_topol[:,3]] - A_grid[grid_topol[:,2]]
    # vertical gradient
    dgrid[:,1] = A_grid[grid_topol[:,1]] - A_grid[grid_topol[:,0]]
    # distance division
    dgrid[:] = dgrid[:] / grid_topol_d[:]

    # return gradient
    return dgrid

def umbrella_integration_2D_igrid(
        grid_d: float,
        a_fc: ndarray,
        a_rc0: ndarray,
        a_mean: ndarray,
        a_covar: ndarray,
        a_N: ndarray,
        limits: ndarray,
        temp: float = 298.,
        thr_close: float = 1e-1,
        integrate: bool = True,
        fortranization: bool = True
        ) -> tuple:
    '''
        Umbrella Integration algorithm for 2D from arrays to a irregular/incomplete grid

        Parameters
        ----------
        grid_d : float
            distance between grid points
        a_fc : ndarray(n_i,2,2)
            array of force constants matrices
        a_rc0 : ndarray(n_i,2)
            array of reference coordinates
        a_mean : ndarray(n_i,2)
            array of mean coordinates
        a_covar : ndarray(n_i,2,2)
            array of covariance matrices of coordinates
        a_N : ndarray(n_i,2)
            array of number of samples for each window
        limits : ndarray(2,2)
            array of minimum and maximum coordinates
        temp : float, optional
            temperature (K) (def: 298.)
        thr_close : float, optional
            distance threshold to include grid points
            if are close to a data point (def: 1e-1)
        integrate : bool, optional
            perform the surface integration (def: True)
            if False, only the free energy derivatives are
            calculated and A_grid returns as array of zeros
        fortranization : bool, optional
            use faster functions writen in Fortran (def: True)

        Returns
        -------
        grid : ndarray(n_ig,2)
            array of grid coordinates
        dA_grid : ndarray(n_ig,2)
            array of free energy derivatives
        A_grid : ndarray(n_ig)
            array of integrated free energy
    '''

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    n_i = a_mean.shape[0]                                 # number of total windows
    a_prec = np.linalg.inv(a_covar[:])                    # precision matrices (inverse of covar)
    a_det_sqrt = np.sqrt(np.linalg.det(a_covar[:]))       # sqrt of determinants of covar matrix

    ## Grid -----------------------------------------------------------
    grid = igrid_gen(grid_d, a_mean, thr_close, fortranization)
    n_ig = grid.shape[0]
    # initialize gradient matrix
    dA_grid = np.zeros_like(grid)

    ## Derivatives ------------------------------------------------------

    if fortranization:
        sys.stdout.write(f"# Calculating derivatives - Irregular/Incomplete Grid ({n_ig:d}) - Fortranized\n")
        dA_grid = umbrellaint_fortran.ui_derivate_2d_igrid(grid, a_fc, a_rc0, a_mean, a_prec, a_det_sqrt, a_N, beta, impossible)
    else:
        sys.stdout.write(f"# Calculating derivatives - Irregular/Incomplete Grid ({n_ig:d})\n")
        # normal probability [Kästner 2009 - Eq.9]
        def probability(rc, i):
            diff = rc - a_mean[i]
            # return np.exp(-0.5 * np.matmul( diff, np.matmul(a_prec[i], diff) )) / (a_det_sqrt[i] * _tau)
            return np.exp( -0.5 * diff.dot(a_prec[i].dot(diff)) ) / (a_det_sqrt[i] * _tau)   # dot faster than matmul for small matrices
        # local derivative of free energy [Kästner 2009 - Eq.10]
        def dA(rc, i):
            # return  np.matmul((rc - a_mean[i])/beta, a_prec[i])  -  np.matmul((rc - a_rc0[i]), a_fc[i])
            return  ((rc - a_mean[i])/beta).dot(a_prec[i])  -  (rc - a_rc0[i]).dot(a_fc[i])
        # normalization total [Kästner 2009 - Eq.11]
        def normal_tot(rc):
            return np.sum([a_N[i]*probability(rc,i) for i in range(n_i)])
        # calculate gradient field of free energy over array grid [Kästner 2009 - Eq.11]
        for ig in range(n_ig):
            rc = grid[ig]
            normal = normal_tot(rc)
            if normal > 1e-9:
                dA_grid[ig] = np.sum(np.transpose([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ]), axis=1)
            else:
                dA_grid[ig] = [impossible,impossible]
            # progress bar
            sys.stdout.write("\r# [{:21s}] - {:>6.2f}%".format( "■"*int(21.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    # remove 'impossible' values
    if impossible in dA_grid:
        removef = np.unique(np.where(dA_grid == impossible)[0])
        grid = np.delete(grid, removef, 0)
        dA_grid = np.delete(dA_grid, removef, 0)

    ## Integration ----------------------------------------------------
    if integrate:
        A_grid = integration_2D_igrid(grid_d, grid, dA_grid)
    else:
        sys.stdout.write("# Integrating             - Skipping \n\n")
        A_grid = np.zeros((n_ig))

    # return results
    return grid, dA_grid, A_grid

def integration_2D_rgrid(
        grid: ndarray,
        dA_grid: ndarray,
        integrator: str = 'simpson+mini'
        ) -> ndarray:
    '''
        Integration of a 2D regular/rectangular surface from its gradient

        Parameters
        ----------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivatives
        integrator : {trapz, simpson, trapz+mini, simpson+mini, fourier}, optional
            integration algorithm (default: 'simpson+mini')

        Returns
        -------
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy,
            minimum value set to zero
    '''

    # check integrator
    if integrator not in {'trapz', 'simpson', 'trapz+mini', 'simpson+mini', 'fourier'}:
        raise ValueError(f"Integrator '{integrator}' not recognized")

    ## grid related definitions
    n_ig = grid.shape[1]
    n_jg = grid.shape[0]
    n_grid = n_jg * n_ig
    dx, dy = abs(grid[0,0,0] - grid[0,1,0]), abs(grid[0,0,1] - grid[1,0,1])    # space between points
    # initialize integrated surface matrix
    A_grid = np.zeros((n_jg,n_ig))

    # difference of gradients per grid point [Kästner 2009 - Eq.14] (optimization format)
    def D_tot(F):
        F = F.reshape(n_jg,n_ig)
        dFy, dFx = np.gradient(F,dy,dx)
        dF = np.stack((dFx,dFy), axis=-1)
        return np.sum((dA_grid - dF)**2) / n_grid

    ## composite trapezoidal rule integration
    if 'trapz' in integrator:
        sys.stdout.write("# Integrating             - Trapezoidal ")
        for i in range(n_ig):
            for j in range(n_jg):
                if i == 0 and j == 0:
                    A_grid[j, i] = 0  # corner point to zero
                elif i == 0:
                    A_grid[j, i] = A_grid[j-1, i] + (dA_grid[j-1, i, 1] + dA_grid[j, i, 1]) * dy / 2
                elif j == 0:
                    A_grid[j, i] = A_grid[j, i-1] + (dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 2
                else:
                    A_grid[j, i] = A_grid[j, i-1] \
                                   + (dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 2 \
                                   + (dA_grid[j-1, i, 1] + dA_grid[j, i, 1]) * dy / 2

    ## Simpson's rule integration
    elif 'simpson' in integrator:
        sys.stdout.write("# Integrating             - Simpson's rule ")
        for j in range(n_jg):
            for i in range(n_ig):
                if i == 0 and j == 0:
                    A_grid[j, i] = 0  # corner point to zero
                elif i == 0:
                    A_grid[j, i] = A_grid[j-1, i] + (dA_grid[j-1, i, 1] + dA_grid[j, i, 1]) * dy / 2
                elif j == 0:
                    A_grid[j, i] = A_grid[j, i-1] + (dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 2
                else:
                    A_grid[j, i] = A_grid[j-1, i-1] \
                                   + (dA_grid[j-1, i-1, 0] + dA_grid[j-1, i, 0] + dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 6 \
                                   + (dA_grid[j-1, i-1, 1] + dA_grid[j-1, i, 1] + dA_grid[j, i-1, 1] + dA_grid[j, i, 1]) * dy / 6

    ## real-space grid minimization
    # TODO: Global optimization methods -> Differential Evolution
    # FIXME: Now minimization of the squared difference of gradients
    #        per grid point instead of the derivative of difference
    #        of gradients (it matters?)
    if 'mini' in integrator:
        sys.stdout.write("+ Real Space Grid Mini ")
        sys.stdout.flush()
        # L-BFGS-B minimization of sumation of square of gradient differences
        mini_result = scipy_optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        if not mini_result.success:
            sys.stdout.write("\nWARNING: Minimization could not converge")
        A_grid = mini_result.x.reshape(n_jg,n_ig)

    ## expansion of the gradient in a Fourier series
    # FIXME: make f** Fourier work properly
    elif integrator == 'fourier':
        sys.stdout.write("# Integrating             - Fourier Series\n")

        raise NotImplementedError("Fourier series integration not implemented yet")

        fourier_f = 2.0

        # Fourier coefficient matrix (n_jg, n_ig, 2)|(n_jg, n_ig) / (M,N,2)|(M,N)
        a = np.zeros_like(A_grid)
        b = np.zeros_like(A_grid)
        c = np.zeros_like(grid)
        d = np.zeros_like(grid)

        ## linear transformation to [0, 2pi]
        # original lenght and origin
        x_lt_len = abs(grid[0,0,0] - grid[0,-1,0])
        x_lt_0 = grid[0,0,0]
        y_lt_len = abs(grid[0,0,1] - grid[-1,0,1])
        y_lt_0 = grid[0,0,1]
        # transform
        grid_lt = np.zeros_like(grid)
        dA_grid_lt = np.zeros_like(dA_grid)
        grid_lt[:,:,0] = ( grid[:,:,0] - x_lt_0 ) * _tau / x_lt_len
        grid_lt[:,:,1] = ( grid[:,:,1] - y_lt_0 ) * _tau / y_lt_len
        dA_grid_lt[:,:,0] = dA_grid[:,:,0] * _tau / x_lt_len
        dA_grid_lt[:,:,1] = dA_grid[:,:,1] * _tau / y_lt_len

        ## calculation of Fourier coefficients
        # gradient dependent [Kästner 2009 - Eq.A3,A4]
        for ig in range(n_ig):
            for jg in range(n_jg):
                c[jg,ig] = np.sum(np.transpose([ dA_grid_lt[k2,k1] * np.cos(_tau*(ig*k1/n_ig + jg*k2/n_jg))
                                    for k1 in range(n_ig) for k2 in range(n_jg) ]), axis=1) / n_grid
                d[jg,ig] = np.sum(np.transpose([ dA_grid_lt[k2,k1] * np.sin(_tau*(ig*k1/n_ig + jg*k2/n_jg))
                                    for k1 in range(n_ig) for k2 in range(n_jg) ]), axis=1) / n_grid
            # progress bar
            sys.stdout.write("\r# [{:21s}] - {:>6.2f}%".format( "■"*int(21.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n")

        # integrated [Kästner 2009 - Eq.A5-A10]
        # general (averaged)
        for ig in range(1,n_ig):
            for jg in range(1,n_jg):
                a[jg,ig] = ( (-d[jg,ig,0]+d[jg,0,0])/ig + (-d[jg,ig,1]+d[0,ig,1])/jg )/2
                b[jg,ig] = ( ( c[jg,ig,0]-c[jg,0,0])/ig + ( c[jg,ig,1]-c[0,ig,1])/jg )/2
            # progress bar
            sys.stdout.write("\r# [{:21s}] - {:>6.2f}%".format( "■"*int(21.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n")

        # zeros
        b[0,:] = (c[0,:,0] - c[0,0,0])/n_ig
        a[0,:] = -d[0,:,0]/n_ig
        b[:,0] = (c[:,0,1] - c[0,0,1])/n_jg
        a[:,0] = -d[:,0,1]/n_jg

        dxy = min(dx, dy)/fourier_f  # space between points as minimum
        # resize sides according
        grid_x = np.arange(limits[0,0],limits[0,1],dxy)
        grid_y = np.arange(limits[1,0],limits[1,1],dxy)
        n_ig = grid_x.size
        n_jg = grid_y.size
        if n_ig%2 != 0:
            n_ig -= 1
            grid_x = grid_x[:-1]
        if n_jg%2 != 0:
            n_jg -= 1
            grid_y = grid_y[:-1]

        # grid building
        grid = np.zeros((n_jg,n_ig,2))
        for j in range(n_jg):
            for i in range(n_ig):
                grid[j,i] = grid_x[i], grid_y[j]

        # integrate in real space
        n_ig2, n_jg2 = int(n_ig/2), int(n_jg/2)
        for ix in range(n_ig):
            for jy in range(n_jg):
                x, y = grid_lt[jy,ix]
                A_grid[jy,ix] = np.sum([ a[jg,ig]*np.cos(ig*x+jg*y) + b[jg,ig]*np.sin(ig*x+jg*y) for ig in range(-n_ig2,n_ig2-1) for jg in range(-n_jg2,n_jg2-1) ])
                # A_grid[jy,ix] = np.sum([ a[jg,ig]*np.cos(ig*x+jg*y) + b[jg,ig]*np.sin(ig*x+jg*y) for ig in range(n_ig) for jg in range(n_jg) ])
            # progress bar
            sys.stdout.write("\r# [{:21s}] - {:>6.2f}%".format( "■"*int(21.*(ix+1)/n_ig), (ix+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n\n")

        # A_grid = A_grid * x_lt_len / _tau

    # integration error
    sys.stdout.write(f"\n# Integration error:        {D_tot(A_grid.ravel()):.2f}\n\n")

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid

def integration_2D_igrid(
        grid_d: float,
        grid: ndarray,
        dA_grid: ndarray
        ) -> ndarray:
    '''
        Integration of a irregular/incomplete 2D surface from its gradient

        Parameters
        ----------
        grid_d : float
            distance between grid points
        grid : ndarray(n_ig,2)
            array of grid coordinates
        dA_grid : ndarray(n_ig,2)
            array of free energy derivatives

        Returns
        -------
        A_grid : ndarray(n_ig)
            array of integrated free energy,
            minimum value set to zero
    '''

    ## grid related definitions
    n_ig = grid.shape[0]
    # grid_d = round(min({abs(grid[i0,0] - grid[i1,0]) for i0 in range(n_ig) for i1 in range(n_ig)} - {0.}), 6)
    # initialize integrated surface matrix
    A_grid = np.zeros((n_ig))

    # grid topology [[↓, ↑, ←, →]] [[horizonal_d, vertical_d]]
    grid_topol, grid_topol_d = igrid_topol(grid_d, grid)
    grid_topol_d = 1 / grid_topol_d

    # difference of gradients per grid point [Kästner 2009 - Eq.14] (optimization format)
    dF = np.zeros_like(dA_grid)
    def D_tot(F):
        # horizonal gradient
        dF[:,0] = F[grid_topol[:,3]] - F[grid_topol[:,2]]
        # vertical gradient
        dF[:,1] = F[grid_topol[:,1]] - F[grid_topol[:,0]]
        # distance division
        dF[:] = dF[:] * grid_topol_d[:]
        return np.sum((dA_grid - dF)**2) / n_ig

    ## real-space grid minimization
    sys.stdout.write("# Integrating             - Real Space Grid Mini ")
    sys.stdout.flush()
    # L-BFGS-B minimization of sumation of square of gradient differences
    mini_result = scipy_optimize.minimize(D_tot, A_grid, method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
    if not mini_result.success:
        sys.stdout.write("\nWARNING: Minimization could not converge")
    A_grid = mini_result.x

    # integration error
    sys.stdout.write(f"\n# Integration error:        {D_tot(A_grid):.2f}\n\n")

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid

def write_1D(
        file: str,
        bins: ndarray,
        A_bins: ndarray,
        temp: str = 'UNKNOWN',
        integrator: str = 'UNKNOWN',
        samples: str = 'UNKNOWN',
        units: str = 'UNKNOWN'
        ) -> None:
    '''
        Write 1D result into file

        Parameters
        ----------
        file : str
            name of the file to write
        bins : ndarray(n_bins)
            array of coordinates of bins
        A_bins : ndarray(n_bins)
            array of integrated free energy
        temp : str, optional
            temperature (K) (def: 'UNKNOWN')
        integrator : str, optional
            integrator method (def: 'UNKNOWN')
        samples : str, optional
            average number of samples (def: 'UNKNOWN')
        units : str, optional
            results' units (def: 'UNKNOWN')
    '''

    with open(file, 'w') as f:
        # general info
        f.write("## UMBRELLA INTEGRATED\n")
        f.write(f"# No bins: {len(bins)}\n")
        f.write(f"# Integrator: {integrator}\n")
        f.write(f"# Temperature (K): {temp}\n")
        f.write(f"# No Samples (Avg): {samples}\n")
        f.write(f"# Units: {units}/mol\n#\n")
        f.write("#  ReactionCoord               PMF\n")
        # data
        for rc, A in zip(bins, A_bins):
            f.write("{:16.9f}  {:16.9f}\n".format(rc, A))

def write_2D_rgrid(
        file: str,
        grid: ndarray,
        A_grid: ndarray,
        temp: str = 'UNKNOWN',
        integrator: str = 'UNKNOWN',
        samples: str = 'UNKNOWN',
        units: str = 'UNKNOWN'
        ) -> None:
    '''
        Write 2D result into file from a regular/rectangular grid

        Parameters
        ----------
        file : str
            name of the file to write
        grid : ndarray(n_j,n_i,2)
            matrix of grid coordinates
        A_grid : ndarray(n_j,n_i)
            matrix of integrated free energy
        temp : str, optional
            temperature (K) (def: 'UNKNOWN')
        integrator : str, optional
            integrator method (def: 'UNKNOWN')
        samples : str, optional
            average number of samples (def: 'UNKNOWN')
        units : str, optional
            results' units (def: 'UNKNOWN')
    '''

    n_j, n_i = np.shape(A_grid)
    with open(file, 'w') as f:
        # general info
        f.write("## UMBRELLA INTEGRATED\n")
        f.write(f"# Grid: {n_i} x {n_j}\n")
        f.write(f"# Integrator: {integrator}\n")
        f.write(f"# Temperature (K): {temp}\n")
        f.write(f"# No Samples (Avg): {samples}\n")
        f.write(f"# Units: {units}/mol\n#\n")
        f.write("#              x                 y               PMF\n")
        # data
        for i in range(n_i):
            for j in range(n_j):
                f.write("{:16.9f}  {:16.9f}  {:16.9f}\n".format(grid[j,i,0], grid[j,i,1], A_grid[j,i]))
            f.write("\n")

def write_2D_igrid(
        file: str,
        grid: ndarray,
        A_grid: ndarray,
        temp: str = 'UNKNOWN',
        integrator: str = 'UNKNOWN',
        samples: str = 'UNKNOWN',
        units: str = 'UNKNOWN'
        ) -> None:
    '''
        Write 2D result into file from a irregular/incomplete grid

        Parameters
        ----------
        file : str
            name of the file to write
        grid : ndarray(n_i,2)
            array of grid coordinates
        A_grid : ndarray(n_i)
            array of integrated free energy
        temp : str, optional
            temperature (K) (def: 'UNKNOWN')
        integrator : str, optional
            integrator method (def: 'UNKNOWN')
        samples : str, optional
            average number of samples (def: 'UNKNOWN')
        units : str, optional
            results' units (def: 'UNKNOWN')
    '''

    # combine data into matrix and sort based on i,j
    data = np.stack((grid[:,0],grid[:,1],A_grid[:]),axis=1)
    data = data[np.lexsort((data[:,1],data[:,0]))]

    with open(file, 'w') as f:
        # general info
        f.write("## UMBRELLA INTEGRATED\n")
        f.write("# Grid: Incomplete Grid\n")
        f.write(f"# Integrator: {integrator}\n")
        f.write(f"# Temperature (K): {temp}\n")
        f.write(f"# No Samples (Avg): {samples}\n")
        f.write(f"# Units: {units}/mol\n#\n")
        f.write("#              x                 y               PMF\n")
        # data
        for i in range(data.shape[0]):
            f.write("{:16.9f}  {:16.9f}  {:16.9f}\n".format(data[i,0], data[i,1], data[i,2]))
            # new line between different x coord
            try:
                if data[i,0] != data[i+1,0]: f.write("\n")
            except: pass


#######################################################################
##  MAIN                                                             ##
#######################################################################
def main():

    ##  PARSER  #######################################################
    parser = argparse.ArgumentParser(description=' -- Umbrella Integration of PMF calculations - 1D & 2D --\n',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version',
                        version='Umbrella Integrator v{} / GPLv3'.format(__version__))
    parser.add_argument('-d', '--dim', metavar='X', type=int,
                        choices=[1,2], required=True,
                        help='dimension of PMF [1 / 2]')
    parser.add_argument('-o', '--out', type=str,
                        help='file name for output (def: PMF-UI.dat)',
                        default='PMF-UI.dat')
    parser.add_argument('-p', '--path', type=str,
                        help='files path (def: current dir)\n'+
                             ' 1D   dat_1.#\n'+
                             ' 2D   dat_1.#.#  dat_2.#.#',
                        default='.')
    parser.add_argument('-t', '--temp', type=float,
                        help='temperature in Kelvin (def: 298.)',
                        default=298.)
    parser.add_argument('-u', '--units', metavar='U', type=str,
                        choices=['kj','kcal'],
                        help='output units per mol [kj / kcal] (def: kj)',
                        default='kj')
    parser.add_argument('-m', '--minsteps', metavar='#', type=int,
                        help='minimum number of steps to take a file (def: 0)',
                        default=0)
    parser.add_argument('-i', '--int', metavar='INT', type=str,
                        choices=['trapz','simpson','trapz+mini', 'simpson+mini', 'mini'],
                        help='integration method for the derivatives\n'+
                             ' 1D\n'+
                             "  'trapz'        - composite trapezoidal rule\n"+
                             "  'simpson'      - Simpson's rule (def)\n"+
                             ' 2D (regular/rectangular grid)\n'+
                             "  'trapz'        - composite trapezoidal rule\n"+
                             "  'simpson'      - Simpson's rule\n"+
                             "  'trapz+mini'   - trapezoidal integration as initial guess \n"+
                             "                   for real space grid minimization\n"+
                             "  'simpson+mini' - Simpson's rule as initial guess \n"+
                             "                   for real space grid minimization (def)\n"+
                             "  'fourier'      - expansion in a Fourier series (not implemented)\n"+
                             ' 2D (irregular/incomplete grid)\n'+
                             "  'mini'         - real space grid minimization (def)\n")
    parser.add_argument('-b', '--bins', type=int,
                        help='number of bins to re-sample in 1D (def: 2000)',
                        default=2000)
    parser.add_argument('--dist', metavar='D', type=float,
                        help='distance between re-sampled points for irregular/incomplete\n'+
                             'grids in 2D (def: 0.05)\n',
                        default=0.05)
    parser.add_argument('--gridf', type=float,
                        help='multiplier factor to apply the number of windows for\n'+
                             're-sampling regular/rectangular grids in 2D (def: 1.5)',
                        default=1.5)
    parser.add_argument('--regular', action='store_true',
                        help='force the use of a regular/rectangular grid instead\n'+
                             'of an irregular/incomplete grid method in 2D\n'),
    parser.add_argument('--names', type=str, nargs=2, metavar=('dat_1','dat_2'),
                        help='basename of the input files for x and y\n'+
                             'coordinates (def: dat_1 dat_2)\n',
                        default=['dat_1', 'dat_2'])
    parser.add_argument('--verbose', action='store_true',
                        help='print additional information (e.g. file reading progress)\n')
    parser.add_argument('--nofortran', action='store_false', help=argparse.SUPPRESS)

    # assignation of input variables
    args = parser.parse_args()
    dimension    = args.dim
    outfile      = args.out
    directory    = args.path
    temperature  = args.temp
    units        = args.units
    minsteps     = args.minsteps
    n_bins       = args.bins
    grid_d       = args.dist
    regular      = args.regular
    grid_f       = args.gridf
    name1, name2 = args.names
    verbose      = args.verbose
    fortranization = args.nofortran if umbrellaint_fortran else False
    if args.int is not None:
        integrator = args.int
    elif dimension == 1:
        integrator = 'simpson'
    elif dimension == 2 and regular:
        integrator = 'simpson+mini'
    else:
        integrator = 'mini'

    ##  GENERAL INFO  #################################################
    sys.stdout.write(f"## UMBRELLA INTEGRATOR v{__version__} ##\n")
    sys.stdout.write(f"# Name: {outfile}\n")
    sys.stdout.write(f"# Path: {directory}\n")
    sys.stdout.write(f"# Temperature (K): {temperature:.2f}\n")
    sys.stdout.write(f"# Units: {units}/mol\n")
    sys.stdout.write(f"# Dimension: {dimension}\n\n")

    ##  1D  ###########################################################
    if dimension == 1:
        ## read input
        n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits = read_dynamo_1D(directory, name1, equilibration, minsteps, verbose)
        ## umbrella integration
        bins, dG, G = umbrella_integration_1D(n_bins, n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits, temp=temperature, integrator=integrator, fortranization=fortranization)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_1D(outfile, bins, G, temp=temperature, integrator=integrator, samples=np.mean(a_N), units=units)

    ##  2D - Rectangular Grid #########################################
    elif dimension == 2 and regular:
        ## check scipy
        if not scipy_optimize:
            raise ImportError("SciPy could not be imported")
        ## read input
        n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits = read_dynamo_2D_rgrid(directory, name1, name2, equilibration, verbose)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D_rgrid(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=temperature, integrator=integrator, fortranization=fortranization)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_2D_rgrid(outfile, grid, G, temp=temperature, integrator=integrator, samples=np.mean(m_N), units=units)

    ##  2D - Incomplete Grid ##########################################
    elif dimension == 2:
        ## check scipy
        if not scipy_optimize:
            raise ImportError("SciPy could not be imported")
        ## read input
        a_fc, a_rc0, a_mean, a_covar, a_N, limits = read_dynamo_2D_igrid(directory, name1, name2, equilibration, minsteps, verbose)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D_igrid(grid_d, a_fc, a_rc0, a_mean, a_covar, a_N, limits, temp=temperature, fortranization=fortranization)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_2D_igrid(outfile, grid, G, temp=temperature, integrator='mini', samples=np.mean(a_N), units=units)

    sys.stdout.write("# Integration finished! \n")
    sys.exit(0)

if __name__ == '__main__':
    main()
