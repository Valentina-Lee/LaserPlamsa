#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:09:53 2025

@author: valentinalee
"""
import cupy as cp
from cupyx.scipy import special as csp
import math
import numpy as np

class plasmaIonization():
    
    def nplasma_1(ne, lam):
        '''
        Calculate n_plasma-1
        ne: electron density in 10^17(cm^3)
        lam: wavelength in m
        '''
        return -ne*(lam*1e6)**2* 4.47869e-5

    def ngas_1(gas):
        '''
        Calculate n_gas-1
        '''
        return gas.alpha* 6.283e-07

    def adk_rate_linear(Eavg, gas):
        w = plasmaIonization.adk_rate_static(gas.EI, Eavg, gas.Z, gas.l, gas.m)
        w *= 0.305282*cp.sqrt(Eavg/gas.EI**1.5)
        return w
        
    def adk_rate_static(EI, E, Z, l, m):
        """
        ADK tunneling ionization rate (static field).
    
        Parameters
        ----------
        EI : float or cupy.ndarray
            Ionization energy of the electron in eV.
        E : float or cupy.ndarray
            Electric field strength in GV/m.
        Z : int
            Atomic residue (1 = first electron, 2 = second, etc.).
        l : int
            Orbital quantum number of the electron being ionized.
        m : int
            Magnetic quantum number of the electron being ionized.
    
        Returns
        -------
        w : float or cupy.ndarray
            Ionization rate in 1/fs.
        """
    
        # Convert EI, E to cupy arrays for GPU if needed
        EI = cp.asarray(EI)
        E  = cp.asarray(E)
    
        # Effective quantum number n*
        n = 3.68859 * Z / cp.sqrt(EI)
        E0 = EI ** 1.5
#        print(E0)
    
        # ADK constant Cn²
        # 4**n / (n * Gamma(2n))
        n = cp.asarray(n, dtype=cp.float64)  # ADK uses non-integer n; use float64 for accuracy
#        logGamma_2n = csp.gammaln(2.0 * n)   # log Gamma(2n)
#        Cn2 = cp.power(4.0, n) / (n * cp.exp(logGamma_2n))
        Cn2 = cp.power(4.0, n) / (n * csp.gamma(2*n))
#        print('gamma:', (n * csp.gamma(2*n)))
    
        # Angular factor N
        # N = 1.51927 * (2*l+1) * (l+|m|)! / (2^|m| * |m|! * (l - |m|)!)
        abs_m = abs(m)
        num = math.factorial(l + abs_m)
        den = (2**abs_m) * math.factorial(abs_m) * math.factorial(l - abs_m)
        
        N = 1.51927 * (2*l + 1) * num / den
#        print('N', N)
        # Initialize output
        w = cp.zeros_like(E)


        factor = (20.4927 * E0 / E)
        exponent = 2*n - abs_m - 1
#        print(E[256:768, 512])
        # Only compute where E > 0 to avoid division by zero
#        print('power', cp.power(factor, exponent)[256:768, 512])
#        print('exp', cp.exp(-6.83089 * E0 / E)[256:768, 512])
        mask = (E > 0)
        if cp.any(mask):
            Em = E[mask]
            factor = (20.4927 * E0 / Em)
            exponent = 2*n - abs_m - 1
            w_mask = (N * Cn2 * EI *
                      cp.power(factor, exponent) *
                      cp.exp(-6.83089 * E0 / Em))
            w[mask] = w_mask
    
        return w

    def energy_loss(EI, ne, dz, dt, E):
        a = 1.207e-2 * EI * ne * (dz*1e6) / ((dt*1e15) * cp.abs(E)**2)
        return cp.where(a < 1, cp.sqrt(1 - a), 0.0)
    
