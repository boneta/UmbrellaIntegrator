#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
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
##  Dependences                                                      ##
#######################################################################
## Support: Python >2.7 and >3.7

import os
import sys
import argparse
import numpy as np

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


## Thermodynamical constants
# http://physics.nist.gov/cuu/Constants/index.html
_c       = 299792458.             # m * s-1
_h       = 6.62607015e-34         # J * s
_kB      = 1.380649e-23           # J * K-1
_NA      = 6.02214076e23          # mol-1
_R       = _kB * _NA              # J * K-1 * mol-1

# conversions
_cal2J   = 4.1868                 # cal -> J
_J2cal   = 1./_cal2J              # J -> cal

# redefinition of kB (kJ and per mol)
kB = _R * 0.001                   # kJ * K-1 * mol-1


#######################################################################
##  Parser                                                           ##
#######################################################################

def __parserbuilder():
    '''Build parser for CLI'''

    global parser
    parser = argparse.ArgumentParser(prog='umbrellaint.py',
                                    description=' -- Umbrella Integration of PMF calculations - 1D & 2D --\n',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version='Umbrella Integrator  0.2 - 18052020\nby Sergio Boneta / GPL')
    parser.add_argument('-d',
                        '--dim',
                        metavar='XD',
                        required=True,
                        type=str,
                        choices=['1D','2D'],
                        help='dimension of PMF [1D / 2D]')
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
                        help='temperature (def: 298.)',
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
                             "   'trapz+mini'  - trapezoidal integration as initial guess \n"
                             "                  for real space grid minimization (def)")
    parser.add_argument('-b',
                        '--bins',
                        type=int,
                        help='number of bins in 1D (def: 2000)',
                        default=2000)
    parser.add_argument('-g',
                        '--grid',
                        type=float,
                        help='multiplicative factor for grid building based\n'+
                             'on the number of windows in 2D (def: 2.)',
                        default=2.)


#######################################################################
##  FUNCTIONS                                                        ##
#######################################################################

##  Read 1D Data  #####################################################
def read_dynamo_1D(directory, name='dat_x', equilibration=0):
    '''Read 1D data from fDynamo files into matrices
    
        Parameters
        ----------
        directory : str
            path to the files
        name : str
            prefix of files (def: 'dat_x')
        equilibration : int
            number of steps considered equilibration and excluded (def: 0)
        
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
    for i in range(n_i):
        # set file and open/read
        fx = coor1[i]
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
        sys.stdout.write("{:s}  -  {:d}\n".format(fx, a_N[i]))

    # build data matrix and get limits
    limits = np.zeros((2), dtype=float)
    limits = np.min(a_mean), np.max(a_mean)

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i))
    sys.stdout.write("# Average samples:  {:<10.2f}\n".format(np.mean(a_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n\n'.format(limits[0],limits[1]))
    sys.stdout.flush()

    # return results
    return n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits

##  Read 2D Data  #####################################################
def read_dynamo_2D(directory, name1='dat_x', name2='dat_y', equilibration=0):
    '''Read 2D data from fDynamo files into matrices
    
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
        m_N : ndarray(n_j,n_i,2)
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
        sys.stdout.write("{:s}  {:s}  -  {:d}\n".format(fx, fy, m_N[j,i]))

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(m_mean[:,:,0]), np.max(m_mean[:,:,0])
    limits[1] = np.min(m_mean[:,:,1]), np.max(m_mean[:,:,1])

    # correct force constant
    # m_fc = m_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i*n_j))
    sys.stdout.write("# Surface dimensions:  {:d} x {:d}\n".format(n_i, n_j))
    sys.stdout.write("# Average samples:  {:<10.2f}\n".format(np.mean(m_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n'
                        '# y_min: {:>7.3f}    y_max: {:>7.3f} \n\n'.format(limits[0,0],limits[0,1],limits[1,0],limits[1,1]))
    sys.stdout.flush()

    # return results
    return n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits

##  Read 2D Data - Line ###############################################
def read_dynamo_2D_line(directory, name1='dat_x', name2='dat_y', equilibration=0):
    '''Read 2D data from fDynamo files into arrays
    
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
        
        Returns
        -------
        n_i : int
            number of total windows simulated
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
        sys.stdout.write("{:s}  {:s}  -  {:d}\n".format(fx, fy, a_N[i]))

    # get limits
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(a_mean[:,0]), np.max(a_mean[:,0])
    limits[1] = np.min(a_mean[:,1]), np.max(a_mean[:,1])

    # correct force constant
    # a_fc = a_fc * 0.5

    # print information
    sys.stdout.write("\n# No Windows:  {:d}\n".format(n_i))
    sys.stdout.write("# Average samples:  {:<10.2f}\n".format(np.mean(a_N)))
    sys.stdout.write('# x_min: {:>7.3f}    x_max: {:>7.3f} \n'
                        '# y_min: {:>7.3f}    y_max: {:>7.3f} \n\n'.format(limits[0,0],limits[0,1],limits[1,0],limits[1,1]))
    sys.stdout.flush()

    # return results
    return n_i, a_fc, a_rc0, a_mean, a_covar, a_N, limits

##  Umbrella Integration 1D  ##########################################
def umbrella_integration_1D(n_bins, n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits, temp=298., integrator='trapz'):
    '''Umbrella Integration algorithm for 1D
    
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
        raise ValueError('Integrator not recognized')

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau_sqrt = np.sqrt(2.*np.pi)
    m_var = a_std ** 2
    bins = np.linspace(limits[0],limits[1],n_bins)
    db = abs(bins[0] - bins[1])                         # space between bins
    dA_bins = np.zeros_like(bins)
    A_bins = np.zeros_like(bins)

    ## Derivates ------------------------------------------------------
 
    # normal probability [Kästner 2005 - Eq.5]
    def probability(rc, i):
        return np.exp(-0.5 * ((rc - a_mean[i])/a_std[i])**2) / (a_std[i] * tau_sqrt)

    # local derivate of free energy [Kästner 2005 - Eq.6]
    def dA(rc, i):
        return (rc - a_mean[i]) / (beta * m_var[i])  -  a_fc[i] * (rc - a_rc0[i]) 

    # normalization total [Kästner 2005 - Eq.8]
    def normal_tot(rc):
        return np.sum([a_N[i]*probability(rc,i) for i in range(n_i)])

    # calculate derivates of free energy over the bins [Kästner 2005 - Eq.7]
    sys.stdout.write("# Calculating derivates - {} bins \n".format(n_bins))
    for b in range(n_bins):
        rc = bins[b]
        normal = normal_tot(rc)
        dA_bins[b] = np.sum([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ])


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
def umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=298., integrator='trapz+mini'):
    '''Umbrella Integration algorithm for 2D
    
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
        m_N : ndarray(n_j,n_i,2)
            matrix of number of samples for each window
        limits : ndarray(2,2)
            matrix of minimum and maximum coordinates
        temp : int, optional
            temperature (K) (def: 298.)
        integrator : {'mini','trapz+mini'}, optional
            integration algorithm (def: 'trapz+mini')

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
    if integrator not in {'mini', 'trapz+mini'}: 
        raise ValueError('Integrator not recognized')

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau = 2.*np.pi
    m_prec = np.linalg.inv(m_covar[:,:])                  # precision matrices (inverse of covar)
    m_det_sqrt = np.sqrt(np.linalg.det(m_covar[:,:]))     # sqrt of determinants of covar matrix

    ## Grid -----------------------------------------------------------

    # grid dimensions
    n_ig = int(np.ceil(grid_f*n_i))
    n_jg = int(np.ceil(grid_f*n_j))
    n_grid = n_jg * n_ig
    # sides
    grid_x = np.linspace(limits[0,0],limits[0,1],n_ig)
    grid_y = np.linspace(limits[1,0],limits[1,1],n_jg)
    dx, dy = abs(grid_x[0] - grid_x[1]), abs(grid_y[0] - grid_y[1])    # space between points
    # grid building
    grid = np.zeros((n_jg,n_ig,2))
    for j in range(n_jg):
        for i in range(n_ig):
            grid[j,i] = grid_x[i], grid_y[j]
    # initialize results matrix
    dA_grid = np.zeros_like(grid)
    A_grid  = np.zeros((n_jg,n_ig))
    # A_grid = np.random.rand(n_jg,n_ig)*10

    ## Derivates ------------------------------------------------------

    # normal probability [Kästner 2009 - Eq.9]
    def probability(rc, j, i):
        diff = rc - m_mean[j,i]
        return np.exp(-0.5 * np.matmul( diff, np.matmul(m_prec[j,i], diff) )) / (m_det_sqrt[j,i] * tau)

    # local derivate of free energy [Kästner 2009 - Eq.10]
    def dA(rc, j, i):
        return  np.matmul((rc - m_mean[j,i])/beta, m_prec[j,i])  -  np.matmul((rc - m_rc0[j,i]), m_fc[j,i])

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


    ## Integration ----------------------------------------------------
    # TODO: Integration by expansion in a Fourier series
 
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
        sys.stdout.write("# Integrating           - Trapezoidal + Real Space Grid Mini\n")

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

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return results
    return grid, dA_grid, A_grid

##  Write 1D Results  #################################################
def write_1D(file, bins, A_bins, temp='UNKNOWN', integrator='UNKNOWN', samples='UNKNOWN', units='UNKNOWN'):
    '''Write 1D result into file
            
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

    sys.stdout.write("# Writing output file\n\n")

    with open(file, 'w') as f:
        # general info
        f.write("## UMBRELLA INTEGRATED\n")
        f.write("# No bins: {}\n".format(len(bins)))
        f.write("# Integrator: {}\n".format(integrator))
        f.write("# Temperature (K): {}\n".format(temp))
        f.write("# Avg samples: {}\n".format(samples))
        f.write("# Units: {}/mol\n#\n".format(units))
        f.write("#  ReactionCoord               PMF\n")
        # data
        for rc, A in zip(bins, A_bins):
            f.write("{:16.9f}  {:16.9f}\n".format(rc, A))

##  Write 1D Results  #################################################
def write_2D(file, grid, A_grid, temp='UNKNOWN', integrator='UNKNOWN', samples='UNKNOWN', units='UNKNOWN'):
    '''Write 2D result into file
        
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

    sys.stdout.write("# Writing output file\n\n")

    n_j, n_i = np.shape(A_grid)
    with open(file, 'w') as f:
        # general info
        f.write("## UMBRELLA INTEGRATED\n")
        f.write("# Grid: {} x {}\n".format(n_i,n_j))
        f.write("# Integrator: {}\n".format(integrator))
        f.write("# Temperature (K): {}\n".format(temp))
        f.write("# Avg samples: {}\n".format(samples))
        f.write("# Units: {}/mol\n#\n".format(units))
        f.write("#              x                 y               PMF\n")
        # data
        for i in range(n_i):
            for j in range(n_j):
                f.write("{:16.9f}  {:16.9f}  {:16.9f}\n".format(grid[j,i,0], grid[j,i,1], A_grid[j,i]))
            f.write("\n")


#######################################################################
##  MAIN                                                             ##
#######################################################################
if __name__ == '__main__':

    # parser
    __parserbuilder()
    args = parser.parse_args()

    # assignation of input variables
    dimension    = args.dim
    outfile      = args.out
    directory    = args.path
    temperature  = args.temp
    units        = args.units
    n_bins       = args.bins
    grid_f       = args.grid
    if args.int is None:
        if dimension == '1D': integrator = 'trapz'
        elif dimension == '2D': integrator = 'trapz+mini'
    else:
        integrator = args.int

    ##  GENERAL INFO  #################################################
    sys.stdout.write("## UMBRELLA INTEGRATION ##\n")
    sys.stdout.write("# Name: {}\n".format(outfile))
    sys.stdout.write("# Path: {}\n".format(directory))
    sys.stdout.write("# Temperature (K): {:.2f}\n".format(temperature))
    sys.stdout.write("# Dimension: {}\n\n".format(dimension))

    ##  1D  ###########################################################
    if dimension == '1D':
        ## read input
        n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits = read_dynamo_1D(directory, name1, equilibration)
        ## umbrella integration
        bins, dG, G = umbrella_integration_1D(n_bins, n_i, a_fc, a_rc0, a_mean, a_std, a_N, limits, temp=temperature, integrator=integrator)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        write_1D(outfile, bins, G, temp=temperature, integrator=integrator, samples=np.mean(a_N), units=units)
    
    ##  2D  ###########################################################
    if dimension == '2D':
        ## check scipy
        if not _scipy:
            raise ImportError('SciPy could not be imported')
            # sys.stdout.write("WARNING: SciPy could not be imported. Integrator changed to 'trapz'.\n")
        ## read input
        n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits = read_dynamo_2D(directory, name1, name2, equilibration)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=temperature, integrator=integrator)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        write_2D(outfile, grid, G, temp=temperature, integrator=integrator, samples=np.mean(m_N), units=units)

    sys.stdout.write("# Integration finished! \n")