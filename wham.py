#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r'''

WHAM (Wheighted Histogram Analysis Method)
==========================================

For reference and testing purposes.
Adapted from legacy code.

'''


import argparse
import os
import sys
from glob import glob

import numpy as np


#######################################################################
##  CONSTANTS                                                        ##
#######################################################################

## Thermodynamical constants
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
##  WHAM 1D                                                          ##
#######################################################################

class WHAM_1D:
    '''Weighted Histogram Analysis Method (WHAM) for 2D PMF calculations'''

    def __init__(
            self,
            temp: float = 298.,
            n_bins: int = -1
            ) -> None:
        '''
            Initialize WHAM 1D calculator

            Parameters
            ----------
            temp : float, optional
                Temperature in Kelvin (def: 298.)
            n_bins : int, optional
                Number of bins for histogram construction, -1 for n_windows*10 (def: -1)
        '''
        # Constants
        self.temp = temp
        self.RT = temp * kB

        # Sampling data
        self.n_windows = 0
        self.n_bins = n_bins

        # Arrays for data storage
        self.fc = np.array([], dtype=np.float64)                       # Force constants
        self.ref = np.array([], dtype=np.float64)                      # Reference positions
        self.frq = np.array([], dtype=np.float64)                      # Frequencies
        self.indx = np.array([], dtype=np.int64)                       # Indices

        # Arrays for WHAM calculation
        self.crd = np.array([], dtype=np.float64)                      # Coordinates
        self.crdn = np.array([], dtype=np.float64)                     # Coordinate counts
        self.umb = None                                                # Umbrella potential energy matrix
        self.umbe = None                                               # Exp(-U/RT) matrix
        self.f = None                                                  # Free energies
        self.rho = None                                                # Densities
        self.pmf = None                                                # Potential of mean force

    def load_data(
            self,
            directory: str = '.',
            name: str = 'dat_1',
            dsp: float = 1.e-6,
            printread: bool = False
            ) -> None:
        '''
            Load and process umbrella sampling data from files

            Parameters
            ----------
            directory : str, optional
                Path to data files (def: current directory)
            name : str, optional
                Basename for coordinate files (def: 'dat_1')
            dsp : float, optional
                Displacement for coordinate range (def: 1.e-6)
            printread : bool, optional
                Print file names and information while reading (def: False)
        '''
        sys.stdout.write("# Reading input files\n")
        # get 'dat_*' file lists
        coor1 = glob(os.path.join(directory, f"{name}.*"))
        if len(coor1) == 0:
            raise NameError(f"No {name}.* files found on the specified path")
        coor1.sort()
        if self.n_bins < 0:
            self.n_bins = len(coor1) * 10

        all_data = []  # Store all coordinate data

        # Process each file
        for fn in coor1:
            with open(fn, 'rt') as f:
                # Read force constant and reference position
                fc, ref = map(float, f.readline().split())
                self.fc = np.append(self.fc, fc)
                self.ref = np.append(self.ref, ref)
                # Read coordinates
                coords = np.array([float(line) for line in f])
                all_data.append(coords)
                self.frq = np.append(self.frq, len(coords))
                self.indx = np.append(self.indx, self.n_windows)
                self.n_windows += 1
            if printread:
                sys.stdout.write(f"{fn}  -  {len(coords)}\n")

        # Sort windows by reference position
        sort_idx = np.argsort(self.ref)
        self.ref = self.ref[sort_idx]
        self.fc = self.fc[sort_idx]
        self.frq = self.frq[sort_idx]
        self.indx = np.arange(self.n_windows)

        # Determine coordinate range and create bins
        all_coords = np.concatenate(all_data)
        vmin = np.min(all_coords) - dsp
        vmax = np.max(all_coords) + dsp

        if self.n_bins is None:
            self.n_bins = 10 * self.n_windows

        # Create histogram bins and compute counts
        bins = np.linspace(vmin, vmax, self.n_bins + 1)
        self.crd = (bins[1:] + bins[:-1]) / 2  # bin centers
        self.crdn, _ = np.histogram(all_coords, bins=bins)

        # print information
        sys.stdout.write(f"\n# No Windows: {self.n_windows}\n")
        sys.stdout.write(f"# x_min: {vmin:>7.3f}    x_max: {vmax:>7.3f}\n\n")
        sys.stdout.flush()

    def _calculate_umbrella_matrices(
            self
            ) -> None:
        '''Calculate umbrella potential energy matrices'''
        # Create coordinate and reference position meshgrids
        ref_mesh, crd_mesh = np.meshgrid(self.ref, self.crd)
        fc_mesh = np.broadcast_to(self.fc, (self.n_bins, self.n_windows))

        # Calculate umbrella potential energy matrix
        self.umb = 0.5 * fc_mesh * (crd_mesh - ref_mesh)**2

        # Calculate Boltzmann factors
        self.umbe = np.exp(-self.umb / self.RT)

    def solve(
            self,
            outfile: str ,
            conv_thr: float = 0.001,
            max_iter: int = 100000
            ) -> None:
        '''
            Solve WHAM equations

            Parameters
            ----------
            outfile : str
                Output file name
            conv_thr : float, optional
                Convergence threshold (def: 0.001)
            maxit : int, optional
                Maximum number of iterations (def: 100000)
        '''
        sys.stdout.write("## WHAM Calculation\n")
        sys.stdout.write(f"# No bins: {self.n_bins}\n")
        # Initialize free energies and calculate matrices
        self.f = np.zeros(self.n_windows)
        self._calculate_umbrella_matrices()
        self.rho = np.zeros(self.n_bins)

        # WHAM iteration
        for iteration in range(max_iter):
            f_old = self.f.copy()

            # Calculate density
            boltz_bias = np.exp(-(self.umb - self.f) / self.RT)  # shape: (nb, nw)
            denominator = np.sum(self.frq * boltz_bias, axis=1)
            self.rho = self.crdn / denominator

            # Update free energies
            self.f = -self.RT * np.log(np.sum(self.umbe * self.rho[:, np.newaxis], axis=0))

            # Check convergence
            max_diff = np.max(np.abs(self.f - f_old))
            converged = max_diff < conv_thr
            if iteration % 100 == 0:
                sys.stderr.write(f'# Iter: {iteration:>6d}   MaxDiff: {max_diff:.10f}  [{conv_thr}]\n')
            if converged:
                sys.stderr.write(f"\n# Converged! Iteration: {iteration}   MaxDiff: {max_diff:.6f}\n")
                break
        else:
            sys.stderr.write(f'\n# Convergency falied! :(\n')
            return

        # Write results
        sys.stderr.write(f'\n# Writing results to {outfile}\n')
        with open(outfile, 'wt') as out:
            out.write('# Temperature: {:.2f}, Number of Iterations: {}, MaxDiff: {:.10f}, Converged: {}\n'.format(self.temp, iteration, max_diff, converged))

            if converged:
                # Write window information
                out.write('#\n#       Points           Reference      Force Constant         Free Energy\n')
                for i in range(self.n_windows):
                    out.write(f'#    {self.frq[i]:9.0f}    {self.ref[i]:16.10f}    {self.fc[i]:16.10f}    {self.f[i]:16.10f}\n')

                # Calculate and normalize PMF
                self.rho /= np.sum(self.rho)
                self.pmf = np.where(self.rho > 0, -self.RT * np.log(self.rho), 0)
                self.pmf -= np.min(self.pmf)  # Shift to zero

                # Write PMF data
                out.write('#\n#       Points           Reference             Density                 PMF\n')
                for i in range(self.n_bins):
                    out.write(f'    {self.crdn[i]:10.0f}    {self.crd[i]:16.10f}    {self.rho[i]:16.10g}    {self.pmf[i]:16.10f}\n')


#######################################################################
##  WHAM 2D                                                          ##
#######################################################################

class WHAM_2D:
    '''Weighted Histogram Analysis Method (WHAM) for 2D PMF calculations'''

    def __init__(
            self,
            temp: float = 298.,
            grid_f: float = 2
            ) -> None:
        '''
            Initialize WHAM 2D calculator

            Parameters
            ----------
            temp : float, optional
                Temperature in Kelvin (def: 298.)
            grid_f : float, optional
                Grid factor for histogram construction (def: 2)
        '''
        # Constants
        self.temp = temp
        self.RT = temp * kB
        self.beta = 1.0 / self.RT
        self.inf = 1.e30

        # Data storage
        self.grid_f = grid_f
        self.n_i = 0
        self.n_j = 0
        self.windows = []
        self.axes = None
        self.rho = None
        self.pmf = None

        # Coordinate bounds
        self.min1 = self.max1 = self.min2 = self.max2 = None

    def load_data(
            self,
            directory: str = '.',
            name1: str = 'dat_1',
            name2: str = 'dat_2',
            equilibration: int = 0,
            printread: bool = False
            ) -> None:
        '''
            Load and process umbrella sampling data from files

            Parameters
            ----------
            directory : str, optional
                Path to data files (def: current directory)
            name1 : str, optional
                Basename for x-coordinate files (def: 'dat_1')
            name2 : str, optional
                Basename for y-coordinate files (def: 'dat_2')
            equilibration : int, optional
                Number of equilibration points to skip (def: 0)
            printread : bool, optional
                Print file names and information while reading (def: False)
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
        __name, n_i, n_j = os.path.basename(coor1[-1]).split('.')
        self.n_i = int(n_i) + 1
        self.n_j = int(n_j) + 1

        # Read coordinate files
        self.windows = []
        for fx, fy in zip(coor1, coor2):
            rc0, fc, rc = [], [], []

            for fname in (fx, fy):
                with open(os.path.join(directory, fname)) as f:
                    data = f.readlines()

                # Parse force constant and reaction coordinate
                fcs, rc0s = data[0].split()
                fc.append(float(fcs))
                rc0.append(float(rc0s))

                # Get coordinates and convert to float
                coords = []
                for line in data[equilibration+1:]:
                    try:
                        coords.append(float(line.strip()))
                    except ValueError as e:
                        sys.stderr.write(f"\nError converting line to float: {line.strip()}")
                        raise e

                rc.append(coords)

            if printread:
                sys.stdout.write(f"{fx}  {fy}  -  {len(rc[0])}\n")

            try:
                # Store window data
                window_data = [
                    np.array(rc0),
                    np.array(fc) * 0.5,
                    np.array(rc, dtype=float).T  # Transpose after converting to array
                ]
                self.windows.append(window_data)
            except Exception as e:
                sys.stderr.write(f"\nError processing window data for files {fx} and {fy}")
                raise e

        self._get_limits()

        # print information
        sys.stdout.write(f"\n# No Windows: {len(self.windows)}\n")
        sys.stdout.write(f"# x_min: {self.min1:>7.3f}    x_max: {self.max1:>7.3f}\n")
        sys.stdout.write(f"# y_min: {self.min2:>7.3f}    y_max: {self.max2:>7.3f}\n\n")
        sys.stdout.flush()

    def solve(
            self,
            outfile: str,
            conv_thr: float = 0.001,
            max_iter: int = 100000
            ) -> tuple:
        '''
            Solve WHAM equations

            Parameters
            ----------
            outfile : str
                Output file name
            conv_thr : float, optional
                Convergence threshold (def: 0.001)
            max_iter : int, optional
                Maximum number of iterations (def: 100000)

            Returns
            -------
            tuple
                Tuple containing axes, density and PMF arrays
        '''
        sys.stdout.write("## WHAM Calculation\n")
        # grid dimensions
        n_ig = int(np.ceil(self.grid_f*self.n_i))
        n_jg = int(np.ceil(self.grid_f*self.n_j))
        bins = (
            (self.min1, self.max1, n_ig),
            (self.min2, self.max2, n_jg)
        )
        sys.stdout.write(f"# Grid: {n_ig} x {n_jg}\n")

        # Generate coordinate axes
        self.axes = [np.linspace(min_val, max_val, n) for min_val, max_val, n in bins]

        # Build histograms and potentials
        hist = np.array([self._histogram(rc, bins) for rc0, fc, rc in self.windows])
        pot = np.array([self._bias_potential(rc0, fc) for rc0, fc, _ in self.windows])

        npoints = self._integrate(hist)
        nwin = len(self.windows)
        dim = len(bins)

        # Initialize WHAM iteration
        f = np.zeros(nwin)
        num = np.sum(hist, axis=0).astype(float)

        # Main WHAM loop
        for iteration in range(max_iter):
            # Expand f array for broadcasting
            f_expanded = f[(slice(None),) + (np.newaxis,) * dim]

            # Calculate exponential terms safely
            expnt = self.beta * (f_expanded - pot)
            expnt = np.where(expnt > -200, expnt, -200)

            # Calculate denominator
            den = np.sum(npoints[(slice(None),) + (np.newaxis,) * dim] * np.exp(expnt), axis=0)

            # Update density and free energy
            good = den != 0
            self.rho = np.where(good, num/den, 0)

            expnt = -self.beta * pot
            expnt = np.where(expnt > -200, expnt, -200)

            f_new = -np.log(self._integrate(self.rho * np.exp(expnt))) / self.beta

            # Check convergence
            max_diff = np.max(np.abs(f - f_new))
            if iteration % 100 == 0:
                sys.stderr.write(f'# Iter: {iteration:>6d}   MaxDiff: {max_diff:.10f}  [{conv_thr}]\n')
            if max_diff < conv_thr:
                sys.stderr.write(f"\n# Converged! Iteration: {iteration}   MaxDiff: {max_diff:.6f}\n")
                break
            f = f_new

        else:
            sys.stderr.write(f"\n# Convergence failed! :(\n")

        # Calculate PMF
        mask = self.rho == 0
        self.pmf = -np.log(self.rho + mask) / self.beta + self.inf * mask
        self.pmf -= np.min(self.pmf)

        # Post-process PMF
        big = np.equal(self.pmf, self.inf)
        pmf_min = np.minimum.reduce(np.ravel(self.pmf))
        # self.pmf = np.choose(big, (self.pmf - pmf_min, 0))
        self.pmf = np.choose(big, (self.pmf - pmf_min, np.nan))
        pmf_max = np.maximum.reduce(np.ravel(self.pmf))

        # Write formatted data for gnuplot
        sys.stderr.write(f'\n# Writing results to {outfile}\n')
        with open(f'{outfile}', 'w') as f:
            for i in range(self.pmf.shape[0]):
                for j in range(self.pmf.shape[1]):
                    if np.isnan(self.pmf[i, j]):
                        continue
                    f.write(f"{self.axes[0][i]}\t{self.axes[1][j]}\t{self.pmf[i,j]}\n")
                f.write("\n")

        return self.axes, self.rho, self.pmf

    def _get_limits(self) -> None:
        '''Calculate coordinate bounds from all windows'''
        try:
            points = np.vstack([window[2] for window in self.windows])
            self.min1, self.min2 = np.min(points, axis=0)
            self.max1, self.max2 = np.max(points, axis=0)
        except Exception as e:
            sys.stderr.write("\n# Error in _get_limits. Aborting.")
            raise e

    def _histogram(self, data: np.ndarray, bins: tuple) -> np.ndarray:
        '''Calculate histogram of data points'''
        mins = np.array([b[0] for b in bins])
        maxs = np.array([b[1] for b in bins])
        ns = np.array([b[2] for b in bins])
        widths = (maxs - mins) / (ns - 1)

        # Calculate bin indices
        indices = ((data - mins + widths/2) / widths).astype(int)

        # Create histogram
        hist = np.zeros(ns, dtype=int)
        mask = np.all((indices >= 0) & (indices < ns), axis=1)
        np.add.at(hist, tuple(indices[mask].T), 1)

        return hist

    def _bias_potential(self, rc0: np.ndarray, fc: np.ndarray) -> np.ndarray:
        '''Calculate biasing potential'''
        p = np.zeros_like(self.axes[0][:, np.newaxis])
        for i, (axis, rc, f) in enumerate(zip(self.axes, rc0, fc)):
            p = p + f * (axis.reshape([-1, 1] if i == 0 else [1, -1]) - rc) ** 2
        return p

    def _integrate(self, arr: np.ndarray) -> np.ndarray:
        '''Integrate array over all dimensions'''
        return np.sum(arr, axis=tuple(range(-(arr.ndim-1), 0)))

##  MAIN  #############################################################

def main():

    # parser
    parser = argparse.ArgumentParser(description=' -- WHAM of PMF calculations - 1D & 2D --\n        For reference and testing!\n',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dim', metavar='X', type=int,
                        choices=[1,2], required=True,
                        help='dimension of PMF [1 / 2]')
    parser.add_argument('-o', '--out', type=str,
                        help='file name for output (def: PMF-WHAM.dat)',
                        default='PMF-WHAM.dat')
    parser.add_argument('-p', '--path', type=str,
                        help='files path (def: current dir)\n'+
                             ' 1D   dat_1.#\n'+
                             ' 2D   dat_1.#.#  dat_2.#.#',
                        default='.')
    parser.add_argument('-t', '--temp', type=float,
                        help='temperature in Kelvin (def: 298.)',
                        default=298.)
    parser.add_argument('-c', '--conv', metavar='#', type=float,
                        help='convergence threshold (def: 0.001)',
                        default=0.001)
    parser.add_argument('--bins', type=int,
                        help='number of bins in 1D (def: #windows*10)',
                        default=-1)
    parser.add_argument('--gridf', type=float,
                        help='multiplier factor to apply to the number of windows'
                             'to get the histograms in 2D (def: 1.5)',
                        default=1.5)
    parser.add_argument('--names', type=str, nargs=2, metavar=('dat_1','dat_2'),
                        help='basename of the input files for x and y\n'+
                             'coordinates (def: dat_1 dat_2)\n',
                        default=['dat_1', 'dat_2'])

    # assignation of input variables
    args = parser.parse_args()
    dimension    = args.dim
    outfile      = args.out
    directory    = args.path
    temperature  = args.temp
    conv         = args.conv
    n_bins       = args.bins
    grid_f       = args.gridf
    name1, name2 = args.names

    # general info
    sys.stdout.write("## WHAM ##\n")
    sys.stdout.write(f"# Name: {outfile}\n")
    sys.stdout.write(f"# Path: {directory}\n")
    sys.stdout.write(f"# Temperature (K): {temperature:.2f}\n")
    sys.stdout.write(f"# Dimension: {dimension}\n\n")

    # WHAM calculation
    if dimension == 1:
        wham_obj = WHAM_1D(temperature, n_bins)
        wham_obj.load_data(directory, name1, printread=False)
        wham_obj.solve(outfile, conv)

    elif dimension == 2:
        wham_obj = WHAM_2D(temperature, grid_f)
        wham_obj.load_data(directory, name1, name2, printread=False)
        wham_obj.solve(outfile, conv)

    sys.stdout.write("\n# WHAMization finished! \n")
    sys.exit(0)

if __name__ == '__main__':
    main()