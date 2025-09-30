#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:41:43 2025

@author: valentinalee
"""

import cupy as cp
from cupy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy import constants
import matplotlib.pyplot as plt
import numpy as np

class propagation_GPU():

    def nonlinear_propagate(pulse, material, thickness, z_steps,
                                   GVD=True, Diffraction=True, SPM_Kerr=True, \
                            save= False, noteName='', absStart= 0):
        """
        GPU-accelerated propagation using CuPy.
        """

        # Move pulse to GPU

        E_field = cp.array(pulse.e)
        lambda0 = pulse.lam*1e-6 #convert to m
        x= pulse.x*1e-6 #convert to m
        y= pulse.y*1e-6 #convert to m
        t= pulse.t*1e-6 #convert to m
        k0 = 2 * cp.pi * material.n0 / lambda0
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = t[1] - t[0]
        
        z= np.linspace(0, thickness, z_steps)
        Nx, Ny, Nt = pulse.Nx, pulse.Ny, pulse.Nt
        
        omega0= 2*cp.pi* constants.c/ lambda0
        omega = 2 * cp.pi * (fftfreq(Nt, dt))
        domega= omega- omega0
        kx = 2 * cp.pi * (fftfreq(Nx, dx))
        ky = 2 * cp.pi * (fftfreq(Ny, dy))
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k_perp2 = KX**2 + KY**2
        dz = float(thickness)/int(z_steps)
        kz = cp.sqrt((k0*k0 - k_perp2).astype(cp.complex128))
        diffraction_phase = cp.exp(1j * kz * dz)   
        gvd_phase = cp.exp(-0.5j * material.beta2 * domega**2 * dz)
        I_to_phase = k0 * material.n2 * dz

        for s in range(z_steps):
            if Diffraction:
                E_field = fft2(E_field, axes=(1, 2))
                E_field *= diffraction_phase[None, :, :]
                E_field = ifft2(E_field, axes=(1, 2))

            if GVD:
                E_field = fft(E_field, axis=0)
                E_field *= gvd_phase[:, None, None]
                E_field = ifft(E_field, axis=0)

            if SPM_Kerr:
                intensity = 0.5 * constants.c * constants.epsilon_0 * (cp.abs(E_field)*1e9)**2
                E_field *= cp.exp(1j * I_to_phase * intensity)
            
            pulse.e = cp.asnumpy(E_field)
            if save:
                pulse.save_field(pulse.e, z[s]+absStart, noteName= noteName)
                pass
        del E_field

#        pulse.e = cp.asnumpy(E_field)  # Convert back to NumPy