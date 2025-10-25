#!/usr/bin/env python3plasmaIonization
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
from cupyx.scipy.ndimage import gaussian_filter
import os
import sys
import glob
for f in glob.glob(os.path.expanduser('../../LaserPlasma'), recursive=True):
    sys.path.insert(0, f)
from beam.plasmaIonization import plasmaIonization
cp.fft.config.set_plan_cache_size(0)


class propagation_GPU():

    def nonlinear_propagate(pulse, material, thickness, z_steps,
                        GVD=True, Diffraction=True, SPM_Kerr=True, \
                        Chromatic= False, save= False, noteName='', absStart= 0, Return= False):
        """
        GPU-accelerated propagation using CuPy.
        """

        # Move pulse to GPU
        
        E_field = cp.array(pulse.e)
        lambda0 = pulse.lam*1e-6 #convert to m
        x= pulse.x*1e-6 #convert to m
        y= pulse.y*1e-6 #convert to m
        t= pulse.t*1e-15 #convert to s
        k0 = 2 * cp.pi * material.n0 / lambda0
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        f_abs= cp.array(pulse.f_abs)
        f_amp= cp.array(pulse.f_amp)
        
        z= np.linspace(0, thickness, z_steps)
#        pulse.z= z
        Nx, Ny, Nt = pulse.Nx, pulse.Ny, pulse.Nt
        
        kx = 2 * cp.pi * (fftfreq(Nx, dx))
        ky = 2 * cp.pi * (fftfreq(Ny, dy))
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k_perp2 = KX**2 + KY**2
        dz = float(thickness)/int(z_steps)
        kz = cp.sqrt((k0*k0 - k_perp2).astype(cp.complex64))
        if Chromatic:
            k0_w= 2* cp.pi* f_abs* material.n0 /constants.c
            kz = cp.sqrt((k0_w[:, None, None]**2 - k_perp2[None, :, :]).astype(cp.complex64))

        diffraction_phase = cp.exp(1j * kz * dz)   
        
        I_to_phase = k0 * material.n2 * dz
        dt = t[1] - t[0]
        omega0= 2*cp.pi* constants.c/ lambda0
        omega = 2 * cp.pi * (fftfreq(Nt, dt))
        domega= omega- omega0
        gvd_phase = cp.exp(-0.5j * material.beta2 * domega**2 * dz)
            
        xz_intensity_slice= np.zeros((pulse.e.shape[1], z_steps))
        yz_intensity_slice= np.zeros((pulse.e.shape[2], z_steps))
        xz_intensity_integrated= np.zeros((pulse.e.shape[1], z_steps))
        yz_intensity_integrated= np.zeros((pulse.e.shape[2], z_steps))
        for s in range(z_steps):
#            E_field = E_field * cp.sqrt(cp.maximum(\
#                gaussian_filter(cp.abs(E_field)**2, sigma=0.1, mode='wrap'), 0.0) / \
#                        (cp.abs(E_field)**2 + cp.finfo(cp.float64).eps))
            if Diffraction:
                if not Chromatic:
                    E_field = fft2(E_field, axes=(1, 2))
                    E_field *= diffraction_phase[None, :, :]
                    E_field = ifft2(E_field, axes=(1, 2))
                else:
                    E_field = fft(E_field, axis=0)
                    E_field = fft2(E_field, axes=(1, 2))
                    #TODO
                    #Basically I didn't match the angular freq because I'm an idiot
                    #fft in time gives you freq in time 
                    #fftfreq goes like this [ 0., 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
                    #Now, diffraction_phase = cp.exp(1j * kz * dz) is freq dependent because of kz
                    #kz through k0_w = 2 * cp.pi * material.n0 / lambda0
                    #if you convert this to freq k0_w= 2 pi f_abs n0/c
                    #the fftfreq 0 frequency is the central frequency (c/lam0)
                    #So you need to get the f_abs by doing f_abs= f0+f_fftfreq
                    #and then you can calculate the relative intensity 
                    #and you can do the following line
                    E_field= E_field* diffraction_phase #*f_amp[:, None, None]
                    E_field = ifft2(E_field, axes=(1, 2))
                    E_field = ifft(E_field, axis=0)
#                    temp_eField_sum= cp.zeros(E_field.shape)
#                    for count in range(len(Chromatic[0])):
#                        temp_eField= temp_eField+ \
#                                     E_field* Chromatic[1][count]
#                        temp_eField= temp_eField+ \
#                                     E_field* diffraction_phase[count, :, :]*Chromatic[1][count]
#                        temp_eField= E_field* diffraction_phase[count, :, :]*Chromatic[1][count]
#                        temp_eField = ifft2(temp_eField, axes=(1, 2))
#                        temp_eField_sum= temp_eField_sum+ \
#                        
#                    print(cp.sum(cp.sum(abs(E_field))))

            if GVD:
                E_field = fft(E_field, axis=0)
                E_field *= gvd_phase[:, None, None]
                E_field = ifft(E_field, axis=0)

            if SPM_Kerr:
                #Intensity = (0.5 * constants.c * constants.epsilon_0 * (cp.abs(E_field)*1e9)**2) put it directly into phase to save GPU memory
                E_field *= cp.exp(1j * I_to_phase* \
                                  (0.5 * constants.c * constants.epsilon_0 * (cp.abs(E_field)*1e9)**2))

            xz_intensity_slice[:, s]= pulse.intensity_from_field\
                                      (cp.asnumpy(E_field[pulse.Nt//2, :, pulse.Ny//2]))
            yz_intensity_slice[:, s]= pulse.intensity_from_field\
                                      (cp.asnumpy(E_field[pulse.Nt//2, pulse.Nx//2, :]))
            xz_intensity_integrated[:, s]= np.sum(pulse.intensity_from_field\
                                      (cp.asnumpy(E_field[:, :, pulse.Ny//2])), axis= 0)/Nt
            yz_intensity_integrated[:, s]= np.sum(pulse.intensity_from_field\
                                      (cp.asnumpy(E_field[:, pulse.Nx//2, :])), axis= 0)/Nt

            if save:
                pulse.e = cp.asnumpy(E_field)
                pulse.save_field(pulse.e, z[s]+absStart, noteName= noteName)
                
        pulse.e = cp.asnumpy(E_field)
        del E_field
        cp._default_memory_pool.free_all_blocks()
        cp.fft.config.clear_plan_cache()
        if Return== True:
            return xz_intensity_slice, yz_intensity_slice, \
            xz_intensity_integrated, yz_intensity_integrated

#        pulse.e = cp.asnumpy(E_field)  # Convert back to NumPy

    def plasma_ionization(pulse, plasma, temp=0.0, save= False, Return= False):
        assert pulse.Nx == plasma.Nx and pulse.Ny == plasma.Ny, \
        'Nx and/or Ny in pulse and plasma do not match'

        E_field = cp.array(pulse.e)
        lam = cp.array(pulse.lam*1e-6) #convert to m
        x= cp.array(pulse.x*1e-6) #convert to m
        y= cp.array(pulse.y*1e-6) #convert to m
        t= cp.array(pulse.t*1e-15) #convert to s
        z= cp.array(plasma.z*1e-6) #convert to m
        k0 = 2 * cp.pi * 1 / lam
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = t[1] - t[0]
        dz = z[1]- z[0]
        
        #n and ne are in 1e17
        z_steps= plasma.Nz
        kx = 2 * cp.pi * (fftfreq(pulse.Nx, dx))
        ky = 2 * cp.pi * (fftfreq(pulse.Ny, dy))
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k_perp2 = KX**2 + KY**2
        kz = cp.sqrt((k0*k0 - k_perp2).astype(cp.complex64))

        # n2 is measured at atmospheric pressure, calculate it per 1e17cm^-3
        dn2 = plasma.gas.n2 * 3.99361e-3 
        kerr_phase_coeff = 1j*k0 * dn2 * dz
        plasma_phase_coeff= 1j*k0*dz*\
        (plasmaIonization.nplasma_1(1, lam)- plasmaIonization.ngas_1(plasma.gas))
        diffraction_phase = cp.exp(1j * kz * dz)
        
        xz_plasma_density= np.zeros((plasma.ne.shape[0], z_steps))
        yz_plasma_density= np.zeros((plasma.ne.shape[1], z_steps))

        for s in range(z_steps):
            print('step:', s)
            n_total= cp.array(plasma.n[:, :, s])
            ne= cp.array(plasma.ne[:, :, s])
            for j in range(pulse.Nt):
                e0= E_field[j, :, :]
                e0= ifft2(diffraction_phase[:, :]*fft2(e0))
                I_j = 0.5* constants.c* constants.epsilon_0* (cp.abs(e0)* 1e9)**2
                ng= n_total- ne #n_total is not changing with j, ne is
                phase = plasma_phase_coeff* ne + 1j* kerr_phase_coeff* ng* I_j
                e0 = e0 * cp.exp(phase)
                #ionize the gas 
                Eavg= 0.5* (cp.abs(cp.array(E_field[j, :, :]))+ cp.abs(e0))
                rate = plasmaIonization.adk_rate_linear(Eavg, plasma.gas)
                #then check rate
                ne_new= n_total-ng*cp.exp(-rate*dt*1e15)                 
                #Energy loss
                dE= plasmaIonization.energy_loss(plasma.gas.EI+temp, ne_new-ne, dz, dt, e0)
                ne= ne_new
                e0= e0*dE
                E_field[j, :, :]= e0
                
            plasma.ne[:, :, s]= cp.asnumpy(ne)
            xz_plasma_density[:, s]= plasma.ne[:, plasma.Ny//2, s]
            yz_plasma_density[:, s]= plasma.ne[plasma.Nx//2, :, s]
            
            if save:
                plasma.save_plasma_density(plasma.ne[:, :, s], s)
                
        pulse.e = cp.asnumpy(E_field)
        del E_field
        cp._default_memory_pool.free_all_blocks()
        cp.fft.config.clear_plan_cache()
        if Return== True:
            return xz_plasma_density, yz_plasma_density

                



