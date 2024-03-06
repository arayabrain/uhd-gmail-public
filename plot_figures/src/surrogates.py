#! /usr/bin/env python3
"""
iaaft - Iterative amplitude adjusted Fourier transform surrogates

        This module implements the IAAFT method [1] to generate time series
        surrogates (i.e. randomized copies of the original time series) which
        ensures that each randomised copy preserves the power spectrum of the
        original time series.

[1] Venema, V., Ament, F. & Simmer, C. A stochastic iterative amplitude
    adjusted Fourier Transform algorithm with improved accuracy (2006), Nonlin.
    Proc. Geophys. 13, pp. 321--328
    https://doi.org/10.5194/npg-13-321-2006

"""
# Created: Tue Jun 22, 2021  09:44am
# Last modified: Tue Jun 22, 2021  12:39pm
#
# Copyright (C) 2021  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de> This
# program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import numpy as np
import six
from tqdm import tqdm


def iaaft(x, ns, tol_pc=5.0, verbose=True, maxiter=1e6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc, disable=not verbose):
        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.0

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.0) / nx

            # 4) repeat until number of unequal entries between r_curr and
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


def aaft(xV):
    """Amplitude Adjusted Fourier Transform Surrogates

    This code is based on the following MATLAB code

    <AAFTsur.m>, v 1.0 2010/02/11 22:09:14  Kugiumtzis & Tsimpiris
    This is part of the MATS-Toolkit http://eeganalysis.web.auth.gr/

    Copyright (C) 2010 by Dimitris Kugiumtzis and Alkiviadis Tsimpiris
                           <dkugiu@gen.auth.gr>

    Reference : D. Kugiumtzis and A. Tsimpiris, "Measures of Analysis of Time
                Series (MATS): A Matlab  Toolkit for Computation of Multiple
                Measures on Time Series Data Bases", Journal of Statistical
                Software, 2010
    """
    n = len(xV)
    zM = np.empty(n)
    T = np.argsort(xV)
    oxV = np.sort(xV)
    ixV = np.argsort(T)

    # Rank order a white noise time series 'wV' to match the ranks of 'xV'
    wV = np.random.randn(n) * np.std(xV, ddof=1)  # match Matlab std
    owV = np.sort(wV)
    yV = owV[ixV].copy()

    # Fourier transform, phase randomization, inverse Fourier transform
    n2 = n // 2
    tmpV = np.fft.fft(yV, 2 * n2)
    magnV = np.abs(tmpV)
    fiV = np.angle(tmpV)
    rfiV = np.random.rand(n2 - 1) * 2 * np.pi
    nfiV = np.append([0], rfiV)
    nfiV = np.append(nfiV, fiV[n2 + 1])
    nfiV = np.append(nfiV, -rfiV[::-1])
    tmpV = np.append(magnV[: n2 + 1], magnV[n2 - 1 : 0 : -1])
    tmpV = tmpV * np.exp(nfiV * 1j)
    yftV = np.real(np.fft.ifft(tmpV, n))  # Transform back to time domain

    # Rank order the 'xV' to match the ranks of the phase randomized
    # time series
    T2 = np.argsort(yftV)
    iyftV = np.argsort(T2)
    zM = oxV[iyftV]  # the AAFT surrogate of xV
    return zM


def ft(x):
    # x is a 2D array with shape (n_sample, n_times)
    n_sample, n_times = x.shape
    y = np.fft.fft(x)
    if n_times % 2 == 0:
        l = int(n_times / 2 - 1)
        r = np.exp(2j * np.pi * np.random.rand(l * n_sample)).reshape(n_sample, -1)
        v = np.ones((n_sample, 1))
        v = np.concatenate((v, r), axis=1)
        v = np.concatenate((v, np.ones((n_sample, 1))), axis=1)
        v = np.concatenate((v, np.conjugate(r)[:, ::-1]), axis=1)
    else:
        l = int((n_times - 1) / 2)
        r = np.exp(2j * np.pi * np.random.rand(l * n_sample)).reshape(n_sample, -1)
        v = np.ones((n_sample, 1))
        v = np.concatenate((v, r), axis=1)
        v = np.concatenate((v, np.conjugate(r)[:, ::-1]), axis=1)
    z = np.fft.ifft(y * v)
    return np.real(z)
