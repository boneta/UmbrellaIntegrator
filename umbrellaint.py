#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: umbrellaint.py
# Description : Umbrella integration of PMF calculations - 1D & 2D
# Version : 0.5.2
# Last update : 29-08-2020
# Author : Sergio Boneta

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

'''
#######################################################################
##                                                                   ##
##                        Umbrella Integrator                        ##
##                                                                   ##
#######################################################################
#
# Copyright (C) 2020, Sergio Boneta
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
## Support: Python >2.7 and >3.7

import os
import sys
import argparse
import numpy as np

# fortranized key functions
try:
    import umbrellaint_fortran
    fortranization = True
except:
    sys.stdout.write("WARNING: Umbrella Integrator's fortran subroutines could not be imported\n" )
    fortranization = False

# scipy
try:
    import scipy.optimize as optimize
    _scipy = True
except ImportError:
    _scipy = False


#######################################################################
##  CONSTANTS                                                        ##
#######################################################################

## Predefined variables
name1 = 'dat_x'
name2 = 'dat_y'
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
##  PARSER                                                           ##
#######################################################################

def __parserbuilder():
    '''Build parser for CLI'''

    parser = argparse.ArgumentParser(prog='umbrellaint.py',
                                    description=' -- Umbrella Integration of PMF calculations - 1D & 2D --\n',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version='Umbrella Integrator  v0.5.2 - 29082020\nby Sergio Boneta / GPL')
    parser.add_argument('-d',
                        '--dim',
                        metavar='X',
                        required=True,
                        type=int,
                        choices=[1,2],
                        help='dimension of PMF [1 / 2]')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        help='file name for output (def: PMF-UI.dat)',
                        default='PMF-UI.dat')
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        help='dat_* files path (def: current dir)\n'+
                             ' 1D   dat_x.#\n'+
                             ' 2D   dat_x.#.#  dat_y.#.#',
                        default='.')
    parser.add_argument('-t',
                        '--temp',
                        type=float,
                        help='temperature in Kelvin (def: 298.)',
                        default=298.)
    parser.add_argument('-u',
                        '--units',
                        metavar='UNITS',
                        type=str,
                        choices=['kj','kcal'],
                        help='output units per mol [kj / kcal] (def: kj)',
                        default='kj')
    parser.add_argument('-i',
                        '--int',
                        metavar='INT',
                        type=str,
                        choices=['trapz','simpson','mini','trapz+mini'],
                        help='integration method for the derivates\n'+
                             ' 1D\n'+
                             "   'trapz'      - composite trapezoidal rule (def)\n"+
                             "   'simpson'    - Simpson's rule\n"+
                             ' 2D\n'+
                             "   'mini'        - real space grid minimization\n"+
                             "   'trapz+mini'  - trapezoidal integration as initial guess \n"+
                             "                   for real space grid minimization (def)\n"+
                             "   'fourier'     - expansion in a Fourier series (not working)")
    parser.add_argument('-b',
                        '--bins',
                        type=int,
                        help='number of bins in 1D (def: 2000)',
                        default=2000)
    parser.add_argument('-g',
                        '--grid',
                        type=float,
                        help='multiplicative factor for grid building based\n'+
                             'on the number of windows in 2D (def: 1.5)',
                        default=1.5)
    parser.add_argument('-ig',
                        '--igrid',
                        action='store_true',
                        help='use incomplete grid instead of a\n'+
                             'rectangular grid based method in 2D\n'+
                             "(integrator: 'mini', space between points: #idist)")
    parser.add_argument('-id',
                        '--idist',
                        type=float,
                        help='distance between grid points for incomplete\n'+
                             'grid based method in 2D (def: 0.05)\n',
                        default=0.05)

    return parser

#######################################################################
##  FUNCTIONS                                                        ##
#######################################################################

##  Read 1D Data  #####################################################
def read_dynamo_1D(directory, name='dat_x', equilibration=0, printread=True):
    '''
        Read 1D data from fDynamo files into arrays

        Parameters
        ----------
        directory : str
            path to the files
        name : str
            prefix of files (def: 'dat_x')
        equilibration : int
            number of steps considered equilibration and excluded (def: 0)
        printread : bool
            print file names while reading

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
    files = os.listdir(directory)
    coor1 = [ i for i in files if name in i ]
    if len(coor1) == 0: raise NameError('No {}.* files found on the specified path'.format(name))
    coor1.sort()

    # number of windows
    n_i = int(len(coor1))

    # initialize 1D matrices
    a_fc    = np.zeros((n_i), dtype=float)                             # force constants
    a_rc0   = np.zeros_like(a_fc)                                      # initial distance set
    a_mean  = np.zeros_like(a_fc)                                      # mean of distance
    a_std   = np.zeros_like(a_fc)                                      # standard deviation of distance

    a_N     = np.zeros((n_i), dtype=int)                               # number of samples

    # read 'dat_*' files
    for fx, i in zip(coor1, range(n_i)):
        # open/read file
        datx = open(os.path.join(directory, fx)).readlines()
        # force and initial distance
        a_fc[i], a_rc0[i] = datx.pop(0).split()
        # convert to numpy, ignoring equilibration part
        datx = np.asarray(datx[equilibration:], dtype=float)
        # mean, standard deviation and number of samples
        a_mean[i] = np.mean(datx)
        a_std[i]  = np.std(datx)
        a_N[i] = len(datx)
        # print progress
        if printread: sys.stdout.write("{:s}  -  {:d}\n".format(fx, a_N[i]))

    # build data matrix and get limits
    limits = np.zeros((2), dtype=float)
    limits = np.min(a_mean), np.max(a_mean)

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i))
    sys.stdout.write("# No Samples (Avg):  {:<10.2f}\n".format(np.mean(a_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n\n'.format(limits[0],limits[1]))
    sys.stdout.flush()

    # return results
    return n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits

##  Read 2D Data  #####################################################
def read_dynamo_2D(directory, name1='dat_x', name2='dat_y',
                    equilibration=0, printread=True):
    '''
        Read 2D data from fDynamo files into matrices

        Parameters
        ----------
        directory : str
            path to the files
        name1 : str
            prefix of x files (def: 'dat_x')
        name2 : str
            prefix of y files (def: 'dat_y')
        equilibration : int
            number of steps considered equilibration and excluded (def: 0)
        printread : bool
            print file names while reading

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
    files = os.listdir(directory)
    coor1 = [ i for i in files if name1 in i ]
    coor2 = [ i for i in files if name2 in i ]
    if len(coor1) == 0 or len(coor2) == 0: raise NameError('No {}.*/{}.* files found on the specified path'.format(name1,name2))
    if len(coor1) != len(coor2): raise NameError('Different number of {}.* and {}.* files'.format(name1,name2))
    coor1.sort()
    coor2.sort()

    # number of windows on every axis
    name, n_i, n_j = coor1[-1].split('.')
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
        # set i,j and open/read files
        name, i, j = fx.split('.')
        i, j = int(i), int(j)
        datx = open(os.path.join(directory, fx)).readlines()
        daty = open(os.path.join(directory, fy)).readlines()
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
        if printread: sys.stdout.write("{:s}  {:s}  -  {:d}\n".format(fx, fy, m_N[j,i]))

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(m_mean[:,:,0]), np.max(m_mean[:,:,0])
    limits[1] = np.min(m_mean[:,:,1]), np.max(m_mean[:,:,1])

    # correct force constant
    # m_fc = m_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i*n_j))
    sys.stdout.write("# Surface dimensions:  {:d} x {:d}\n".format(n_i, n_j))
    sys.stdout.write("# No Samples (Avg):  {:<10.2f}\n".format(np.mean(m_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n'
                        '# y_min: {:>7.3f}    y_max: {:>7.3f} \n\n'.format(limits[0,0],limits[0,1],limits[1,0],limits[1,1]))
    sys.stdout.flush()

    # return results
    return n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits

##  Read 2D Data - Incomplete Grid  ###################################
def read_dynamo_2D_igrid(directory, name1='dat_x', name2='dat_y',
                          equilibration=0, printread=True):
    '''
        Read 2D data from fDynamo files into arrays

        Parameters
        ----------
        directory : str
            path to the files
        name1 : str
            prefix of x files (def: 'dat_x')
        name2 : str
            prefix of y files (def: 'dat_y')
        equilibration : int
            number of steps considered equilibration and excluded (def: 0)
        printread : bool
            print file names while reading

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
    files = os.listdir(directory)
    coor1 = [ i for i in files if name1 in i ]
    coor2 = [ i for i in files if name2 in i ]
    if len(coor1) == 0 or len(coor2) == 0: raise NameError('No {}.*/{}.* files found on the specified path'.format(name1,name2))
    if len(coor1) != len(coor2): raise NameError('Different number of {}.* and {}.* files'.format(name1,name2))
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

    # read 'dat_*' files
    line0 = np.zeros((2,2), dtype=float)
    for fx, fy, i in zip(coor1, coor2, range(n_i)):
        # open/read files
        datx = open(os.path.join(directory, fx)).readlines()
        daty = open(os.path.join(directory, fy)).readlines()
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
        # print progress
        if printread: sys.stdout.write("{:s}  {:s}  -  {:d}\n".format(fx, fy, a_N[i]))

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(a_mean[:,0]), np.max(a_mean[:,0])
    limits[1] = np.min(a_mean[:,1]), np.max(a_mean[:,1])

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i))
    sys.stdout.write("# No Samples (Avg):  {:<10.2f}\n".format(np.mean(a_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n'
                        '# y_min: {:>7.3f}    y_max: {:>7.3f} \n\n'.format(limits[0,0],limits[0,1],limits[1,0],limits[1,1]))
    sys.stdout.flush()

    # return results
    return a_fc, a_rc0, a_mean, a_covar, a_N, limits

##  Umbrella Integration 1D  ##########################################
def umbrella_integration_1D(n_bins, n_i, a_fc, a_rc0, a_mean, a_std,
                             a_N, limits, temp=298., integrator='trapz'):
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
        temp : int, optional
            temperature (K) (def: 298.)
        integrator : {'trapz', 'simpson'}, optional
            integration algorithm (def: 'trapz')

        Returns
        -------
        bins : ndarray(n_bins)
            array of coordinates of bins
        dA_bins : ndarray(n_bins)
            array of free energy derivates
        A_bins : ndarray(n_bins)
            array of integrated free energy
    '''

    # check integrator specification
    if integrator not in {'trapz', 'simpson'}:
        raise NameError('Integrator not recognized')

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau_sqrt = np.sqrt(_tau)
    a_var = a_std ** 2
    bins = np.linspace(limits[0],limits[1],n_bins)
    db = abs(bins[0] - bins[1])                         # space between bins
    dA_bins = np.zeros_like(bins)
    A_bins = np.zeros_like(bins)

    ## Derivates ------------------------------------------------------

    def derivate():
        # normal probability [Kästner 2005 - Eq.5]
        def probability(rc, i):
            return np.exp(-0.5 * ((rc - a_mean[i])/a_std[i])**2) / (a_std[i] * tau_sqrt)

        # local derivate of free energy [Kästner 2005 - Eq.6]
        def dA(rc, i):
            return (rc - a_mean[i]) / (beta * a_var[i])  -  a_fc[i] * (rc - a_rc0[i])

        # normalization total [Kästner 2005 - Eq.8]
        def normal_tot(rc):
            return np.sum([a_N[i]*probability(rc,i) for i in range(n_i)])

        # calculate derivates of free energy over the bins [Kästner 2005 - Eq.7]
        sys.stdout.write("# Calculating derivates - {} bins \n".format(n_bins))
        for ib in range(n_bins):
            rc = bins[ib]
            normal = normal_tot(rc)
            dA_bins[ib] = np.sum([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ])

        return dA_bins

    if fortranization:
        sys.stdout.write("# Calculating derivates - {} bins - Fortranized\n".format(n_bins))
        dA_bins = umbrellaint_fortran.ui_derivate_1d(bins, a_fc, a_rc0, a_mean, a_std, a_N, beta)
    else:
        dA_bins = derivate()

    ## Integration ----------------------------------------------------

    ## composite trapezoidal rule integration [scipy.integrate.cumtrapz]
    if integrator == 'trapz':
        sys.stdout.write("# Integrating           - Trapezoidal \n\n")
        stp = db/2.
        for b in range(1, n_bins):
            A_bins[b] = A_bins[b-1] + stp * (dA_bins[b-1] + dA_bins[b])

    ## Simpson's rule integration
    elif integrator == 'simpson':
        sys.stdout.write("# Integrating           - Simpson's rule \n\n")
        stp = db/6.
        for b in range(1,n_bins-1):
            A_bins[b] = A_bins[b-1] + stp * (dA_bins[b-1] + 4*dA_bins[b] + dA_bins[b+1])
        A_bins[-1] = A_bins[-2] + db/2. * (dA_bins[-2] + dA_bins[-1])   # last point trapezoidal


    # set minimum to zero
    A_bins = A_bins - np.min(A_bins)

    # return results
    return bins, dA_bins, A_bins

##  Umbrella Integration 2D  ##########################################
def umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean,
                             m_covar, m_N, limits, temp=298.,
                             integrator='trapz+mini', integrate=True):
    '''
        Umbrella Integration algorithm for 2D from matrices

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
        integrator : {'mini','trapz+mini', 'fourier'}, optional
            integration algorithm (def: 'trapz+mini')
        integrate : bool, optional
            perform the surface integration (def: True)
            if False, only the free energy derivates are
            calculated and A_grid returns as matrix of zeros

        Returns
        -------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivates
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy
    '''

    # check integrator specification
    if integrator not in {'mini', 'trapz+mini', 'fourier'}:
        raise NameError('Integrator not recognized')

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
    ## if Fourier: regular and even grid
    if integrator=='fourier':
        # FIXME: better difinition of even grid
        dxy = min(dx, dy)  # space between points as minimum
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
    # initialize gradient matrix
    dA_grid = np.zeros_like(grid)


    ## Derivates ------------------------------------------------------

    def derivate():
        # normal probability [Kästner 2009 - Eq.9]
        def probability(rc, j, i):
            diff = rc - m_mean[j,i]
            # return np.exp(-0.5 * np.matmul( diff, np.matmul(m_prec[j,i], diff) )) / (m_det_sqrt[j,i] * _tau)
            return np.exp( -0.5 * diff.dot(m_prec[j,i].dot(diff)) ) / (m_det_sqrt[j,i] * _tau)   # dot faster than matmul for small matrices

        # local derivate of free energy [Kästner 2009 - Eq.10]
        def dA(rc, j, i):
            # return  np.matmul((rc - m_mean[j,i])/beta, m_prec[j,i])  -  np.matmul((rc - m_rc0[j,i]), m_fc[j,i])
            return ((rc - m_mean[j,i])/beta).dot(m_prec[j,i])  -  (rc - m_rc0[j,i]).dot(m_fc[j,i])

        # normalization total [Kästner 2009 - Eq.11]
        def normal_tot(rc):
            return np.sum([m_N[j,i]*probability(rc,j,i) for i in range(n_i) for j in range(n_j)])

        # # calculate normalization denominator for all the grid coordinates in advance --> same speed as on the fly calc, not worthy
        # def normal_denominator(): return np.asarray([ normal_tot(grid[j,i]) for j in range(n_jg) for i in range(n_ig)]).reshape(n_jg,n_ig)

        # calculate gradient field of free energy over grid [Kästner 2009 - Eq.11]
        sys.stdout.write("# Calculating derivates - Grid {} x {}\n".format(n_ig,n_jg))
        for jg in range(n_jg):
            for ig in range(n_ig):
                rc = grid[jg,ig]
                normal = normal_tot(rc)
                dA_grid[jg,ig] = np.sum(np.transpose([ m_N[j,i]*probability(rc,j,i)/normal * dA(rc,j,i) for i in range(n_i) for j in range(n_j) ]), axis=1)
            # progress bar
            sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(jg+1)/n_jg), (jg+1)/n_jg*100)); sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        return dA_grid

    if fortranization:
        sys.stdout.write("# Calculating derivates - Grid {:d} x {:d} - Fortranized\n".format(n_ig,n_jg))
        dA_grid = umbrellaint_fortran.ui_derivate_2d_rgrid(grid, m_fc, m_rc0, m_mean, m_prec, m_det_sqrt, m_N, beta)
    else:
        dA_grid = derivate()


    ## Integration ----------------------------------------------------
    if integrate:
        A_grid = integration_2D(grid, dA_grid, integrator)
    else:
        sys.stdout.write("# Integrating           - Skipping \n\n")
        A_grid = np.zeros((n_jg,n_ig))

    # return results
    return grid, dA_grid, A_grid

##  Incomplete Grid Generator 2D  #####################################
def igrid_gen(grid_d, a_coord, thr_close=1e-1):
    '''
        Incomplete grid generator in 2D

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
            if min(coord_dist) < thr: return True
            else: return False

    # get limits
    limits_x = (np.min(a_coord[:,0]), np.max(a_coord[:,0]))
    limits_y = (np.min(a_coord[:,1]), np.max(a_coord[:,1]))

    # grid axes
    grid_x = np.arange(limits_x[0],limits_x[1]+grid_d/2.,grid_d)
    grid_y = np.arange(limits_y[0],limits_y[1]+grid_d/2.,grid_d)
    n_x = grid_x.shape[0]
    n_y = grid_y.shape[0]

    # grid building
    grid = []
    for j in range(n_y):
        for i in range(n_x):
            rc = grid_x[i], grid_y[j]
            if point_in(rc): grid.append(rc)
    grid = np.asarray(grid)

    # return final grid
    return grid

##  Incomplete Grid Topology 2D  ######################################
def igrid_topol(grid_d, grid):
    '''
        Incomplete grid topology guessing

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

##  Incomplete Grid Gradient 2D  ######################################
def igrid_grad(A_grid, grid_topol, grid_topol_d):
    '''
        Incomplete grid topology gradient calculation

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

##  Umbrella Integration 2D - Incomplete Grid #########################
def umbrella_integration_2D_igrid(grid_d, a_fc, a_rc0, a_mean, a_covar,
                                   a_N, limits, temp=298.,
                                   thr_close=1e-1, integrate=True):
    '''
        Umbrella Integration algorithm for 2D from arrays to a incomplete grid

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
            if False, only the free energy derivates are
            calculated and A_grid returns as array of zeros

        Returns
        -------
        grid : ndarray(n_ig,2)
            array of grid coordinates
        dA_grid : ndarray(n_ig,2)
            array of free energy derivates
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
    grid = igrid_gen(grid_d, a_mean, thr_close)
    n_ig = grid.shape[0]
    # initialize gradient matrix
    dA_grid = np.zeros_like(grid)


    ## Derivates ------------------------------------------------------

    def derivate_igrid():
        # normal probability [Kästner 2009 - Eq.9]
        def probability(rc, i):
            diff = rc - a_mean[i]
            # return np.exp(-0.5 * np.matmul( diff, np.matmul(a_prec[i], diff) )) / (a_det_sqrt[i] * _tau)
            return np.exp( -0.5 * diff.dot(a_prec[i].dot(diff)) ) / (a_det_sqrt[i] * _tau)   # dot faster than matmul for small matrices

        # local derivate of free energy [Kästner 2009 - Eq.10]
        def dA(rc, i):
            # return  np.matmul((rc - a_mean[i])/beta, a_prec[i])  -  np.matmul((rc - a_rc0[i]), a_fc[i])
            return  ((rc - a_mean[i])/beta).dot(a_prec[i])  -  (rc - a_rc0[i]).dot(a_fc[i])

        # normalization total [Kästner 2009 - Eq.11]
        def normal_tot(rc):
            return np.sum([a_N[i]*probability(rc,i) for i in range(n_i)])

        # calculate gradient field of free energy over array grid [Kästner 2009 - Eq.11]
        sys.stdout.write("# Calculating derivates - Incomplete Grid ({:d})\n".format(n_ig))
        for ig in range(n_ig):
            rc = grid[ig]
            normal = normal_tot(rc)
            if normal > 1e-9:
                dA_grid[ig] = np.sum(np.transpose([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ]), axis=1)
            else:
                dA_grid[ig] = [impossible,impossible]
            # progress bar
            sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        return dA_grid

    if fortranization:
        sys.stdout.write("# Calculating derivates - Incomplete Grid ({:d}) - Fortranized\n".format(n_ig))
        dA_grid = umbrellaint_fortran.ui_derivate_2d_igrid(grid, a_fc, a_rc0, a_mean, a_prec, a_det_sqrt, a_N, beta, impossible)
    else:
        dA_grid = derivate_igrid()

    # check for 'impossible' values
    # TODO: just remove 'impossible' values and continue
    if impossible in dA_grid: raise ValueError('A grid point is too far from data')

    ## Integration ----------------------------------------------------
    if integrate:
        A_grid = integration_2D_igrid(grid_d, grid, dA_grid)
    else:
        sys.stdout.write("# Integrating           - Skipping \n\n")
        A_grid = np.zeros((n_ig))

    # return results
    return grid, dA_grid, A_grid

##  Integrate 2D surface  #############################################
def integration_2D(grid, dA_grid, integrator, fourier_f=2.0):
    '''
        Integration of a 2D surface from its gradient

        Parameters
        ----------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivates
        integrator : {'mini','trapz+mini','fourier'}
            integration algorithm

        Returns
        -------
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy,
            minimum value set to zero
    '''

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

    ## real-space grid minimization
    # TODO: Global optimization methods -> Differential Evolution
    # FIXME: Now minimization of the squared difference of gradients
    #        per grid point instead of the derivate of difference
    #        of gradients (it matters?)
    if integrator == 'mini':
        sys.stdout.write("# Integrating           - Real Space Grid Mini \n\n")

        # L-BFGS-B minimization of sumation of square of gradient differences
        mini_result = optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        if not mini_result.success:
            sys.stdout.write("WARNING: Minimization could not converge\n\n")
        A_grid = mini_result.x.reshape(n_jg,n_ig)

    ## double composite trapezoidal rule integration + real-space grid minimization
    elif integrator == 'trapz+mini':
        sys.stdout.write("# Integrating           - Trapezoidal + Real Space Grid Mini\n\n")

        # double trapezoidal from a corner
        def trapz_corner(init, dir0):
            A = np.zeros((n_jg,n_ig))
            # parameter specification
            if init == 'DownLeft':
                i_range = list(range(1, n_ig))
                j_range = list(range(1, n_jg))
                fi, fj = -1, -1
            elif init == 'UpLeft':
                i_range = list(range(1, n_ig))
                j_range = list(reversed(range(0, n_jg-1)))
                fi, fj = -1, 1
            elif init == 'DownRight':
                i_range = list(reversed(range(0, n_ig-1)))
                j_range = list(range(1, n_jg))
                fi, fj = 1, -1
            elif init == 'UpRight':
                i_range = list(reversed(range(0, n_ig-1)))
                j_range = list(reversed(range(0, n_jg-1)))
                fi, fj = 1, 1
            stp_x, stp_y = -1*fi*dx/2., -1*fj*dy/2.
            # integration
            if dir0 == 'x':
                for i in i_range:
                    A[0,i] = A[0,i+fi] + stp_x * (dA_grid[0,i+fi,0] + dA_grid[0,i,0])
                for j in j_range:
                    A[j,:] = A[j+fj,:] + stp_y * (dA_grid[j+fj,:,1] + dA_grid[j,:,1])
            elif dir0 == 'y':
                for j in j_range:
                    A[j,0] = A[j+fj,0] + stp_y * (dA_grid[j+fj,0,1] + dA_grid[j,0,1])
                for i in i_range:
                    A[:,i] = A[:,i+fi] + stp_x * (dA_grid[:,i+fi,0] + dA_grid[:,i,0])
            # relativized result
            return A - np.min(A)

        # calculate trapezoidal integrals from every corner and average -> inital guess
        A_grid = ( trapz_corner('DownLeft', 'x')  + trapz_corner('DownLeft', 'y') +
                   trapz_corner('UpLeft',   'x')  + trapz_corner('UpLeft',   'y') +
                   trapz_corner('DownRight','x')  + trapz_corner('DownRight','y') +
                   trapz_corner('UpRight',  'x')  + trapz_corner('UpRight',  'y') ) / 8

        # L-BFGS-B minimization of sumation of square of gradient differences
        mini_result = optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        if not mini_result.success:
            sys.stdout.write("WARNING: Minimization could not converge\n\n")
        A_grid = mini_result.x.reshape(n_jg,n_ig)

    ## expansion of the gradient in a Fourier series
    # FIXME: make f** Fourier work properly
    elif integrator == 'fourier':
        sys.stdout.write("# Integrating           - Fourier Series\n")

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
            sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n")

        # integrated [Kästner 2009 - Eq.A5-A10]
        # general (averaged)
        for ig in range(1,n_ig):
            for jg in range(1,n_jg):
                a[jg,ig] = ( (-d[jg,ig,0]+d[jg,0,0])/ig + (-d[jg,ig,1]+d[0,ig,1])/jg )/2
                b[jg,ig] = ( ( c[jg,ig,0]-c[jg,0,0])/ig + ( c[jg,ig,1]-c[0,ig,1])/jg )/2
            # progress bar
            sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(ig+1)/n_ig), (ig+1)/n_ig*100)); sys.stdout.flush()
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
            sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(ix+1)/n_ig), (ix+1)/n_ig*100)); sys.stdout.flush()
        sys.stdout.write("\n\n")

        # A_grid = A_grid * x_lt_len / _tau

    ## unknown integrator
    else:
        raise NameError('Unknown integrator')

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid

##  Integrate 2D surface - Incomplete Grid ############################
def integration_2D_igrid(grid_d, grid, dA_grid):
    '''
        Integration of a incomplete 2D surface from its gradient

        Parameters
        ----------
        grid_d : float
            distance between grid points
        grid : ndarray(n_ig,2)
            array of grid coordinates
        dA_grid : ndarray(n_ig,2)
            array of free energy derivates

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
    sys.stdout.write("# Integrating           - Real Space Grid Mini \n\n")
    # L-BFGS-B minimization of sumation of square of gradient differences
    mini_result = optimize.minimize(D_tot, A_grid, method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
    if not mini_result.success:
        sys.stdout.write("WARNING: Minimization could not converge\n\n")
    A_grid = mini_result.x

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid

##  Write 1D Results  #################################################
def write_1D(file, bins, A_bins, temp='UNKNOWN', integrator='UNKNOWN',
              samples='UNKNOWN', units='UNKNOWN'):
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
        f.write("# No bins: {}\n".format(len(bins)))
        f.write("# Integrator: {}\n".format(integrator))
        f.write("# Temperature (K): {}\n".format(temp))
        f.write("# No Samples (Avg): {}\n".format(samples))
        f.write("# Units: {}/mol\n#\n".format(units))
        f.write("#  ReactionCoord               PMF\n")
        # data
        for rc, A in zip(bins, A_bins):
            f.write("{:16.9f}  {:16.9f}\n".format(rc, A))

##  Write 2D Results  #################################################
def write_2D(file, grid, A_grid, temp='UNKNOWN', integrator='UNKNOWN',
              samples='UNKNOWN', units='UNKNOWN'):
    '''
        Write 2D result into file

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
        f.write("# Grid: {} x {}\n".format(n_i,n_j))
        f.write("# Integrator: {}\n".format(integrator))
        f.write("# Temperature (K): {}\n".format(temp))
        f.write("# No Samples (Avg): {}\n".format(samples))
        f.write("# Units: {}/mol\n#\n".format(units))
        f.write("#              x                 y               PMF\n")
        # data
        for i in range(n_i):
            for j in range(n_j):
                f.write("{:16.9f}  {:16.9f}  {:16.9f}\n".format(grid[j,i,0], grid[j,i,1], A_grid[j,i]))
            f.write("\n")

##  Write 2D Results - Incomplete Grid  ###############################
def write_2D_igrid(file, grid, A_grid, temp='UNKNOWN', integrator='UNKNOWN',
                    samples='UNKNOWN', units='UNKNOWN'):
    '''
        Write 2D result into file from a incomplete grid

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
        f.write("# Integrator: {}\n".format(integrator))
        f.write("# Temperature (K): {}\n".format(temp))
        f.write("# No Samples (Avg): {}\n".format(samples))
        f.write("# Units: {}/mol\n#\n".format(units))
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
if __name__ == '__main__':

    # parser
    parser = __parserbuilder()
    args = parser.parse_args()

    # assignation of input variables
    dimension    = args.dim
    outfile      = args.out
    directory    = args.path
    temperature  = args.temp
    units        = args.units
    n_bins       = args.bins
    grid_f       = args.grid
    igrid        = args.igrid
    grid_d       = args.idist
    if args.int is None:
        if dimension == 1: integrator = 'trapz'
        elif dimension == 2: integrator = 'trapz+mini'
    else:
        integrator = args.int

    ##  GENERAL INFO  #################################################
    sys.stdout.write("## UMBRELLA INTEGRATION ##\n")
    sys.stdout.write("# Name: {}\n".format(outfile))
    sys.stdout.write("# Path: {}\n".format(directory))
    sys.stdout.write("# Temperature (K): {:.2f}\n".format(temperature))
    sys.stdout.write("# Dimension: {}\n\n".format(dimension))

    ##  1D  ###########################################################
    if dimension == 1:
        ## read input
        n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits = read_dynamo_1D(directory, name1, equilibration)
        ## umbrella integration
        bins, dG, G = umbrella_integration_1D(n_bins, n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits, temp=temperature, integrator=integrator)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_1D(outfile, bins, G, temp=temperature, integrator=integrator, samples=np.mean(a_N), units=units)

    ##  2D  ###########################################################
    if dimension == 2 and not igrid:
        ## check scipy
        if not _scipy:
            raise ImportError('SciPy could not be imported')
        ## read input
        n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits = read_dynamo_2D(directory, name1, name2, equilibration)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=temperature, integrator=integrator)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_2D(outfile, grid, G, temp=temperature, integrator=integrator, samples=np.mean(m_N), units=units)

    ##  2D - Incomplete Grid ##########################################
    elif dimension == 2 and igrid:
        ## check scipy
        if not _scipy:
            raise ImportError('SciPy could not be imported')
        ## read input
        a_fc, a_rc0, a_mean, a_covar, a_N, limits = read_dynamo_2D_igrid(directory, name1, name2, equilibration)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D_igrid(grid_d, a_fc, a_rc0, a_mean, a_covar, a_N, limits, temp=temperature)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        sys.stdout.write("# Writing output file\n\n")
        write_2D_igrid(outfile, grid, G, temp=temperature, integrator='mini', samples=np.mean(a_N), units=units)

    sys.stdout.write("# Integration finished! \n")
