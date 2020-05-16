#!/usr/bin/python
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
    from scipy.integrate import simps
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
                        version='Umbrella Integrator  0.1 - 16052020\nby Sergio Boneta / GPL')
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
                        choices=['trapz','simpson','real'],
                        help='integration method for the derivates\n'+
                             ' 1D\n'+
                             "   'trapz'   - composite trapezoidal rule\n"+
                             "   'simpson' - Simpson's rule (def)\n"+
                             ' 2D\n'+
                             "   'real'    - real space grid (def)")
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
    dat_all = []
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
        # accumulate data
        dat_all.extend(datx)
        # print progress
        sys.stdout.write("{:s}  -  {:d}\n".format(fx, a_N[i]))

    # build data matrix and get limits
    dat_all = np.asarray(dat_all)
    limits = np.zeros((2), dtype=float)
    limits = np.min(dat_all), np.max(dat_all)

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
    all_x, all_y = [], []
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
        # accumulate data
        all_x.extend(datx)
        all_y.extend(daty)
        # print progress
        sys.stdout.write("{:s}  {:s}  -  {:d}\n".format(fx, fy, m_N[j,i]))

    # build data matrix and get limits
    dat_all = np.stack((all_x, all_y))
    del all_x, all_y
    limits = np.zeros((2,2), dtype=float)
    limits[0] = np.min(dat_all[0]), np.max(dat_all[0])
    limits[1] = np.min(dat_all[1]), np.max(dat_all[1])

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

    # correct number of bins before integral triming
    n_bins_o = n_bins
    if integrator == 'trapz': n_bins = n_bins + 1
    elif integrator == 'simpson': n_bins = n_bins + 2
    else: raise ValueError('Integrator not recognized')
    
    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau_sqrt = np.sqrt(2.*np.pi)
    m_var = a_std ** 2
    bins = np.linspace(limits[0],limits[1],n_bins)
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

    sys.stdout.write("# Calculating derivates - {} bins \n".format(n_bins_o))

    # calculate derivates of free energy over the bins [Kästner 2005 - Eq.7]
    for j in range(n_bins):
        rc = bins[j]
        normal = normal_tot(rc)
        dA_bins[j] = np.sum([ a_N[i]*probability(rc,i)/normal * dA(rc,i) for i in range(n_i) ])


    ## Integration ----------------------------------------------------
    # TODO: Non acumulative 1D integrals -> More efficiency

    # composite trapezoidal rule integration
    if integrator == 'trapz':
        sys.stdout.write("# Integrating           - Trapezoidal \n\n")
        for j in range(n_bins-1):
            A_bins[j] = np.trapz(dA_bins[:j+1], bins[:j+1])
        bins, A_bins = bins[:-1], A_bins[:-1]
    
    # Simpson's rule integration
    elif integrator == 'simpson':
        sys.stdout.write("# Integrating           - Simpson's rule \n\n")
        for j in range(n_bins-2):
            A_bins[j] = simps(dA_bins[:j+2], bins[:j+2], even='avg')
        bins, A_bins = bins[:-2], A_bins[:-2]


    # set minimum to zero
    A_bins = A_bins - np.min(A_bins)

    # return results
    return bins, dA_bins, A_bins

##  Umbrella Integration 2D  ##########################################
def umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=298., integrator='real'):
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
        integrator : {'real'}, optional
            integration algorithm (def: 'real')

        Returns
        -------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivates
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy
    '''

    # initial definitions
    sys.stdout.write("## Umbrella Integration\n")
    beta = 1./(kB*temp)
    tau = 2.*np.pi
    m_prec = np.linalg.inv(m_covar[:,:])                  # precision matrices (inverse of covar)
    m_det_sqrt = np.sqrt(np.linalg.det(m_covar[:,:]))     # sqrt of determinants of covar matrix

    ## Grid -----------------------------------------------------------
    # grid dimensions
    n_i_g = int(np.ceil(grid_f*n_i))
    n_j_g = int(np.ceil(grid_f*n_j))
    n_grid = n_j_g * n_i_g
    # sides
    grid_x = np.linspace(limits[0,0],limits[0,1],n_i_g)
    grid_y = np.linspace(limits[1,0],limits[1,1],n_j_g)
    dx, dy = abs(grid_x[0] - grid_x[1]), abs(grid_y[0] - grid_y[1])    # space between points
    # grid building
    grid = np.zeros((n_j_g,n_i_g,2))
    for j in range(n_j_g):
        for i in range(n_i_g):
            grid[j,i] = grid_x[i], grid_y[j]
    # initialize results matrix
    dA_grid = np.zeros_like(grid)
    A_grid = np.random.rand(n_j_g,n_i_g)*10

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
    # def normal_denominator(): return np.asarray([ normal_tot(grid[j,i]) for j in range(n_j_g) for i in range(n_i_g)]).reshape(n_j_g,n_i_g)

    # calculate gradient field of free energy over grid [Kästner 2009 - Eq.11]
    sys.stdout.write("# Calculating derivates - Grid {} x {}\n".format(n_i_g,n_j_g))
    for jg in range(n_j_g):
        for ig in range(n_i_g):
            rc = grid[jg,ig]
            normal = normal_tot(rc)
            dA_grid[jg,ig] = np.sum(np.transpose([ m_N[j,i]*probability(rc,j,i)/normal * dA(rc,j,i) for i in range(n_i) for j in range(n_j) ]), axis=1)
        # progress bar
        sys.stdout.write("\r# [{:19s}] - {:>6.2f}%".format( "■"*int(19.*(jg+1)/n_j_g), (jg+1)/n_j_g*100)); sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


    ## Integration ----------------------------------------------------
    # TODO: Integration by expansion in a Fourier series
    # TODO: Global optimization methods -> Differential Evolution
    # FIXME: Now minimization of the squared difference of gradients
    #        per grid point instead of the derivate of difference
    #        of gradients (it matters?)
 
    # real-space grid integration
    if integrator == 'real':
        sys.stdout.write("# Integrating           - Real Space Grid \n\n")

        # derivate of gradient differences [Kästner 2009 - Eq.16]
        # FIXME: Elements of two boundary levels are excluded
        def dD(F,j,i):
            if (j and i > 1) and (j < n_j_g-1 and i < n_i_g-1):
                return (dA_grid[j,i-1,0] - dA_grid[j,i+1,0] + (-F[j,i-2]+2*F[j,i]-F[j,i-2])/(2*dx))/dx \
                     + (dA_grid[j-1,i,1] - dA_grid[j+1,i,1] + (-F[j-2,i]+2*F[j,i]-F[j-2,i])/(2*dy))/dy
            else:
                return 0

        # sumation of derivates of gradient differences (optimization format)
        def dD_tot(F):
            F = F.reshape(n_j_g,n_i_g)
            return np.sum([ dD(F,j,i)**2 for i in range(2, n_i_g-2) for j in range(2, n_j_g-2) ])

        # difference of gradients per grid point [Kästner 2009 - Eq.14] (optimization format)
        # boundary elements are excluded. loop format --> gradient on the fly --> F*****g slow 
        # def D_tot_core_loop(F):
        #     F = F.reshape(n_j_g,n_i_g)
        #     core = np.sum([ (dA_grid[j,i,0] - (F[j,i+1]-F[j,i-1])/(2*dx))**2 + (dA_grid[j,i,1] - (F[j+1,i]-F[j-1,i])/(2*dy))**2 for i in range(1, n_i_g-1) for j in range(1, n_j_g-1) ])
        #     return core

        # difference of gradients per grid point [Kästner 2009 - Eq.14] (optimization format)
        # vectorized --> F*****g fast
        def D_tot(F):
            F = F.reshape(n_j_g,n_i_g)
            dFy, dFx = np.gradient(F,dy,dx)
            dF = np.stack((dFx,dFy), axis=-1)
            return np.sum((dA_grid - dF)**2) / n_grid
        
        # difference of gradients per grid point (optimization format)
        # boundary elements are excluded. vectorized
        # def D_tot_core(F):
        #     F = F.reshape(n_j_g,n_i_g)
        #     dFy, dFx = np.gradient(F,dy,dx)
        #     dF = np.stack((dFx,dFy), axis=-1)
        #     return np.sum((dA_grid[1:-1,1:-1] - dF[1:-1,1:-1])**2) / n_grid

        # L-BFGS-B minimization of sumation of square of gradient differences
        mini_result = optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        if not mini_result.success:
            sys.stdout.write("WARNING: Minimization could not converge\n\n")
        A_grid = mini_result.x.reshape(n_j_g,n_i_g)

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
        if dimension == '1D': integrator = 'simpson'
        elif dimension == '2D': integrator = 'real'

    ##  GENERAL INFO  #################################################
    sys.stdout.write("## UMBRELLA INTEGRATION ##\n")
    sys.stdout.write("# Name: {}\n".format(outfile))
    sys.stdout.write("# Path: {}\n".format(directory))
    sys.stdout.write("# Temperature (K): {:.2f}\n".format(temperature))
    sys.stdout.write("# Dimension: {}\n\n".format(dimension))

    ##  1D  ###########################################################
    if dimension == '1D':
        ## check integrator
        if integrator == 'simpson' and not _scipy: 
            sys.stdout.write("WARNING: SciPy could not be imported. Integrator changed to 'trapz'.\n")
            integrator = 'trapz'
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
        ## read input
        n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits = read_dynamo_2D(directory, name1, name2, equilibration)
        ## umbrella integration
        grid, dG, G = umbrella_integration_2D(grid_f, n_i, n_j, m_fc, m_rc0, m_mean, m_covar, m_N, limits, temp=temperature, integrator=integrator)
        ## write output
        if units.lower() == 'kcal': G = G * _J2cal
        write_2D(outfile, grid, G, temp=temperature, integrator=integrator, samples=np.mean(m_N), units=units)

    sys.stdout.write("# Integration finished! \n")