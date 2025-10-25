#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:15:01 2017

@author: robert, valentina
"""

import pyfftw
import numpy as np
from beam import beam
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fftshift, fft2
from optics import gaussian
from scipy import constants
import re
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
import time
import gc

class Pulse(beam.Beam):
    """ A laser pulse class that stores the field for each transverse slice.
    
    This class stores a three dimensional grid, two transverse spatial
    directions and a temporal direction. The temporal coordinate is measured so
    that t=0 is at the center of the grid, Nt/2. The temporal component is
    stored in complex notation, it can be used as either an envelope or
    explicit field. For this class, z tracks the location of t=0.
    
    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction, a power of 2 is recommended.
    Ny : int
        Number of grid points in the y direction, a power of 2 is recommended.
    Nt : int
        Number of grid points in the x direction, a power of 2 is recommended.
    X : double
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
    Y : double
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
    T : double
        Length of the grid in the temporal direction.
    lam : double
        The vacuum wavelength of the laser radiation.
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    load : bool
        Boolean specifying if we are loading an existing object.
    cyl : bool
        Whether the beam is cylindrically symmetric or not. Controls whether
        the entire transverse field is saved or only a 1D slice.
    """
    keys = ['Nx',
            'Ny',
            'Nt',
            'X',
            'Y',
            'T',
            'lam',
            'tau',
            'path',
            'name',
            'load',
            'cyl']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        self.k = 2*np.pi / self.params['lam']
        # Create internal variables
        if self.load is False:
            self.create_grid()
            self.initialize_field()
            self.save_initial()
    
    def create_grid(self):
        """ Create an x-y rectangular grid and temporal grid. """
        T = self.T
        X = self.X
        Y = self.Y
        self.t = np.linspace(-T/2, T/2, self.Nt, False, dtype='double')
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        self.z = []
    
    def initialize_field(self, e=None):
        """ Create the array to store the electric field values in.
        
        Parameters
        ----------
        e : array-like, optional
            The array of field values to initialize the field to.
        """
        if e is None:
            self.e = np.zeros((self.Nt, self.Nx, self.Ny,), dtype='complex64')
        else:
            self.e = np.array(e, dtype='complex64')
        self.saveInd = 0
        self.z = []
        self.save_field(self.e, 0.0)
        
    def initialize_field_TwoToThreeD(self, beam):
        """ Create the array to store the electric field values in. 
        When given a laser profile and pulse tau, create a pulse.
        
        Parameters
        ----------
        e : array-like, optional
            The array of field values to initialize the field to.
        """

        self.e = np.array(beam.e[None, :, :]*gaussian.temporal_gaussian_envelope(self.t, self.tau).astype(np.float32)[:, None, None], dtype='complex64')
        self.saveInd = 0
        self.z = []
        self.save_field(self.e, 0.0)

    def load_beam(self):
        """ Load the beam, specifically load the z grid and saveInd. """
        self.create_grid()
        self.z = np.load(self.filePre + '_z.npy', allow_pickle=True).tolist()
        self.saveInd = len(self.z)
        e, temp = self.load_field(self.saveInd - 1)
        if not self.cyl:
            self.e = e
        else:
            self.e = np.zeros((self.Nt, self.Nx, self.Ny,), dtype='complex64')
            x = self.x
            y = self.y
            for i in range(self.Nt):
                self.e[i, :, :] = self.reconstruct_from_cyl(x, e[i, :], x, y)

    def normalize_beam(self, J, n0=1):
        """normalize beam with pulse energy
        J: pulse energy in J
        """
        dx = (self.x[1] - self.x[0])
        dy = (self.y[1] - self.y[0])
        try:
            dt = (self.t[1] - self.t[0])
        except:
            dt=1
        norm = np.sum(np.abs(self.e)**2) * dx * dy * dt
        prefactor = (n0 * constants.epsilon_0 * constants.c / 2)
        E0 = np.sqrt(J / (prefactor * norm))
        self.e= (E0* self.e *10**(4.5)).astype(np.complex64)

    def resize_beam_old(self, new_X, new_Y, new_Nx, new_Ny):
        e_list= []
        self.X= new_X
        self.Y= new_Y
        self.Nx= new_Nx
        self.Ny= new_Ny
        old_x= self.x
        old_y= self.y
        self.create_grid()
        for timeSlice in range(0, self.Nt):
            f_real = RectBivariateSpline(old_x, old_y, self.e[0, :, :].real)
            f_imag = RectBivariateSpline(old_x, old_y, self.e[0, :, :].imag)
            self.e= self.e[1:, :, :]
            e_list.append((f_real(self.x, self.y).astype(np.float32)\
                          + 1j* f_imag(self.x, self.y).astype(np.float32))\
                          .astype(np.complex64))
        self.e= np.array(e_list)
        
    def resize_beam(self, new_X, new_Y, new_Nx, new_Ny):
        e_list= []
        self.X= new_X
        self.Y= new_Y
        self.Nx= new_Nx
        self.Ny= new_Ny
        old_x= self.x
        old_y= self.y
        self.create_grid()
        for timeSlice in range(0, self.Nt):
            f_amp = RectBivariateSpline(old_x, old_y, abs(self.e[0, :, :]), kx=1, ky=1, s=0.0)
            f_phase = RectBivariateSpline(old_x, old_y, \
                                    np.unwrap(np.unwrap(np.angle(self.e[0, :, :]), axis=0), axis=1))
            self.e= self.e[1:, :, :]
            e_list.append((f_amp(self.x, self.y).astype(np.float32)*\
                           np.exp(1j* f_phase(self.x, self.y).astype(np.float32)))
                          .astype(np.complex64))
        self.e= np.array(e_list)

    def resize_beam_test(self, new_X, new_Y, new_Nx, new_Ny):
        assert new_X/self.X<1 and new_Y/self.Y < 1, 'New beam needs to be smaller to use this function'

        crop_Nx= self.Nx*new_X//self.X+1
        crop_Nx += crop_Nx % 2
        crop_Ny= self.Nx*new_Y//self.Y+1
        crop_Ny += crop_Nx % 2
        ix0 = int((self.Nx - crop_Nx) // 2)
        iy0 = int((self.Ny - crop_Ny) // 2)
        ix1 = int(ix0 + crop_Nx)
        iy1 = int(iy0 + crop_Ny)

        old_x = self.x[ix0:ix1]
        old_y = self.y[iy0:iy1]
        copy  = np.ascontiguousarray(self.e[:, ix0:ix1, iy0:iy1])  
        del self.e
        time.sleep(3)
        
        self.X= new_X
        self.Y= new_Y
        self.Nx= new_Nx
        self.Ny= new_Ny
        self.create_grid()
        amp= abs(copy).astype(np.float32)
        phase= np.unwrap(np.unwrap(np.unwrap(np.angle(copy), \
                                             axis=0), axis=1), axis=2).astype(np.float32)
        del copy
        time.sleep(3)
        interp_func_amp = RegularGridInterpolator(\
            (self.t, old_x, old_y), amp, bounds_error=False, fill_value=0.0)
        interp_func_phase = RegularGridInterpolator(\
            (self.t, old_x, old_y), phase, \
            bounds_error=False, fill_value=0.0)
        Tq, Xq, Yq = np.meshgrid(self.t, self.x, self.y, indexing='ij') 
        points = np.stack([Tq.ravel(), Xq.ravel(), Yq.ravel()], axis=-1).astype(np.float32) 
        del Tq, Xq, Yq
        self.e = (interp_func_amp(points).astype(np.float32) *\
                 np.exp(1j * interp_func_phase(points).astype(np.float32))\
                  ).reshape(len(self.t), new_Nx, new_Ny).astype(np.complex64)
        

#    def reshape_temporal(self):
#        temporal_profile= np.sum(np.sum(abs(PSPulse.e)**2, axis= 1), axis= 1)
    # Getters and setters
    #--------------------------------------------------------------------------
        
    def set_field(self, e):
        """ Set the value of the electric field. """
        self.e = np.array(e, dtype='complex64')
        self.save_field(self.e, self.z[-1])
        
    def get_dx(self):
        """ Get the grid spacing. """
        x = self.x
        return x[1] - x[0]
    
    def get_dy(self):
        """ Get the grid spacing. """
        y = self.y
        return y[1] - y[0]
    
    def get_dt(self):
        """ Get the grid spacing. """
        t = self.t
        return t[1] - t[0]
        
    def get_f(self):
        """ Get the spatial frequencies of the fft of e. """
        dx = self.get_dx()
        dy = self.get_dy()
        fx = fftfreq(self.Nx, dx)
        fy = fftfreq(self.Ny, dy)
        return fx, fy

    def get_chromatic_components(self, tau, num_comp):
        dfreq= 0.44/tau/np.sqrt(2)
        dlam= (self.lam*1e-6)**2/constants.c*dfreq
        lam_min= self.lam*1e-6- 0.4*dlam
        lam_max= self.lam*1e-6+ 0.4*dlam
        print(lam_min, lam_max)
        lam= np.linspace(lam_min, lam_max, num_comp)
        amp = np.exp(-4*np.log(2) * ((lam - self.lam*1e-6)**2) / (dlam**2))
        amp= amp/np.sum(amp)
        return lam, amp

    def get_chromoatic_freq(self):
        dt= (self.t*1e-15)[1]-(self.t*1e-15)[0]
        f_rev= fftfreq(self.Nt, dt)
        f0= constants.c/(self.lam*1e-6)
        f_abs= f0+f_rev
        dfreq= 0.44/(self.tau*1e-15)
        amp= np.exp(-2*np.log(2)*((f_abs - f0)**2) / (dfreq**2))
        amp= amp/np.sum(amp)
        self.f_abs= f_abs
        self.f_amp= amp
        
    # File managment
    #--------------------------------------------------------------------------
        
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
    
    def save_field(self, e, z, save= True, noteName= ''):
        """ Save the current electric field to file and adavnce z. """
        if self.cyl:
            e = e[:, :, int(self.Ny/2)]
        if save== True:
            np.save(self.filePre + noteName+'_field_' + str(self.saveInd) + '.npy', e)
        self.saveInd += 1
        self.z.append(z)
        self.save_z()
        
    def save_z(self):
        """ Save the z array. """
        np.save(self.filePre + '_z.npy', self.z)
        
    def load_field(self, ind):
        """ Load the electric field at the specified index. 
        
        Parameters
        ----------
        ind : int
            The save index to load the field at.
        
        Returns
        -------
        e : array-like
            The electric field at the specified index.
        z : double
            The z coordinate of the field.
        """
        e = np.load(self.filePre + '_field_' + str(ind) + '.npy')
        z = self.z[ind]
        return e, z
        
    # Physics functions
    #--------------------------------------------------------------------------
        
    def pulse_energy(self):
        """ Calculate the energy in the pulse in joules. """
        I = self.intensity_from_field(self.e)
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        try:
            dt = self.t[1] - self.t[0]
        except:
            dt=1
        return np.sum(I)*dt*dx*dy*1e-9
        
    
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def plot_current_tran_intensity(self, lim=None):
        """ Plots the current intensity at the center of the pulse. """
        im = self.plot_tran_intensity(self.e[int(self.Nt/2), :, :], self.z[-1],
                                     lim)
        plt.show(im)
        
    def plot_tran_intensity_at(self, ind):
        """ Plots the intensity at a particular z distance.
        
        Parameters
        ----------
        ind : int
            The save index to plot the field at, see the _z file to find z.
        """
        e, z = self.load_field(ind)
        if not self.cyl:
            data = e[int(self.Nt/2), :, :]
        else:
            x = self.x
            y = self.y
            data = self.reconstruct_from_cyl(x, e[int(self.Nt/2), :], x, y)
        im = self.plot_tran_intensity(data, z)
        plt.show(im)
    
    def plot_tran_intensity(self, e, z, lim=None):
        """ Create a transverse intensity plot. """
        X = self.X
        Y = self.Y
        
        I = self.intensity_from_field(e)
        I = self.prep_data(I)
        im = plt.imshow(I, aspect='auto', extent=[-X/2, X/2, -Y/2, Y/2])
        cb = plt.colorbar()
        cb.set_label(r'Intensity ($\mathrm{10^{14}W/cm^2}$)')
        plt.set_cmap('viridis')
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        if lim is not None:
            plt.xlim(lim)
            plt.ylim(lim)
        plt.title('Transverse intensity at z='+str(z))
        return im
    
    def plot_current_long_intensity(self):
        """ Plots the current intensity at the in the x-t plane. """
        e = np.array(self.e[:, :, int(self.Ny/2)])
        im = self.plot_long_intensity(e, self.z[-1])
        plt.show(im)
        
    def plot_long_intensity_at(self, ind):
        """ Plots the intensity in x-t at a particular z distance.
        
        Parameters
        ----------
        ind : int
            The save index to plot the field at, see the _z file to find z.
        """
        e, z = self.load_field(ind)
        if not self.cyl:
            e = e[:, :, int(self.Ny/2)]
        im = self.plot_long_intensity(e, z)
        plt.show(im)
    
    def plot_long_intensity(self, e, z):
        """ Create an longitudinal intensity plot. """
        T = self.T
        X = self.X
        
        I = self.intensity_from_field(e)
        I = self.prep_data(I)
        im = plt.imshow(I, aspect='auto', extent=[-T/2, T/2, -X/2, X/2])
        cb = plt.colorbar()
        cb.set_label(r'Intensity ($\mathrm{10^{14}W/cm^2}$)')
        plt.set_cmap('viridis')
        plt.xlabel(r't')
        plt.ylabel(r'x')
        plt.title('Longitudinal intensity at z='+str(z))
        return im
    
    def plot_current_field(self, xlim=None, flim=None, log=False, wrap_order=0):
        beam = self
        # XXX Not sure why I have to copy, I suspect the fft
        e = np.copy(beam.e[int(self.Nt/2), :, :])
        I = beam.intensity_from_field(e)
        If = abs(fftshift(fft2(e)))**2
        fx, fy = beam.get_f()
        fx = fftshift(fx)
        fy = fftshift(fy)
        phase = np.angle(e)

        # Images
        X = beam.X
        Y = beam.Y
        ext = [-X/2, X/2, -Y/2, Y/2]
        extf = [fx[0], fx[-1], fy[0], fy[-1]]
        plt.figure(figsize=(16, 4), dpi=150)
        plt.subplot(131)
        plt.imshow(beam.prep_data(I), aspect='auto', extent=ext, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Intensity ($10^{14}$ W/cm^2)')
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'$y$ (um)')
        if xlim != None:
            plt.xlim(xlim)
            plt.ylim(xlim)
        
        if wrap_order == 0:
            axis0 = 0
            axis1 = 1
        elif wrap_order == 1:
            axis0 = 1
            axis1 = 0
        plt.subplot(132)
        plt.imshow(np.unwrap(np.unwrap(beam.prep_data(phase), axis=axis0), axis=axis1), aspect='auto', extent=ext, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Phase (rad)')
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'$y$ (um)')
        if xlim != None:
            plt.xlim(xlim)
            plt.ylim(xlim)

        plt.subplot(133)
        plt.imshow(beam.prep_data(If), aspect='auto', extent=extf, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Intensity (arb unit)')
        plt.xlabel(r'$f_x$ (um$^{-1}$)')
        plt.ylabel(r'$f_y$ (um$^{-1}$)')
        if flim != None:
            plt.xlim(flim)
            plt.ylim(flim)

        plt.tight_layout()
        plt.show()
        # Lineouts
        # We've already taken the transpose so y is the first index
        indy = int(beam.Ny/2)
        indx = int(beam.Nx/2)
        x = beam.x
        y = beam.y
        plt.figure(figsize=(16, 4), dpi=150)
        plt.subplot(131)
        plt.plot(x, I[:, indy], label='y')
        plt.plot(y, I[indx, :], 'm--', label='x')
        plt.legend()
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'Intensity ($10^{14}$ W/cm^2)')
        if xlim != None:
            plt.xlim(xlim)

        plt.subplot(132)
        plt.plot(x, np.unwrap(phase[:, indy]), label='x')
        plt.plot(y, np.unwrap(phase[indx, :]), 'm--', label='y')
        plt.legend()
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'Phase (rad)')
        if xlim != None:
            plt.xlim(xlim)

        plt.subplot(133)
        plt.plot(fx, If[:, indy], label='x')
        plt.plot(fy, If[indx, :], 'm--', label='y')
        plt.legend()
        plt.xlabel(r'$f_x$ (um$^{-1}$)')
        plt.ylabel(r'Intensity (arb unit)')
        if flim != None:
            plt.xlim(flim)

        plt.tight_layout()
        plt.show()
        
        if log == True:
            # Lineouts
            plt.figure(figsize=(16, 4), dpi=150)
            plt.subplot(131)
            plt.plot(x, I[:, indy], label='x')
            plt.plot(y, I[indx, :], 'm--', label='y')
            plt.legend()
            plt.xlabel(r'$x$ (um)')
            plt.ylabel(r'Intensity ($10^{14}$ W/cm^2)')
            plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)

            plt.subplot(132)
            plt.plot(x, np.unwrap(phase[:, indy]), label='x')
            plt.plot(y, np.unwrap(phase[indx, :]), 'm--', label='y')
            plt.legend()
            plt.xlabel(r'$x$ (um)')
            plt.ylabel(r'Phase (rad)')
            plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)

            plt.subplot(133)
            plt.plot(fx, If[:, indy], label='x')
            plt.plot(fy, If[indx, :], 'm--', label='y')
            plt.legend()
            plt.xlabel(r'$f_x$ (um$^{-1}$)')
            plt.ylabel(r'Intensity (arb unit)')
            plt.yscale('log')
            if flim != None:
                plt.xlim(flim)

            plt.tight_layout()
            plt.show()

    def plot_propagation(self, axis= 'x', timeSlice= None, noteName='', xylim=None):
        def extract_number(filename):
            # Find the last group of digits in the filename
            match = re.findall(r'\d+', filename)
            return int(match[-1]) if match else -1          
        print('Starting...')
        if timeSlice== None:
            timeSlice= self.Nt//2
        load_path= self.path+ 'beams/beam_'+ self.name + '/'
        starter= self.name+ noteName
        files = []
        for filename in os.listdir(load_path):
            if filename.startswith(starter):
                files.append(filename)

        files_sorted = sorted(files, key=extract_number)
        start_z= int(re.findall(r'\d+', files_sorted[0])[0])
        end_z= int(re.findall(r'\d+', files_sorted[-1])[0])
        z_grid= np.array(self.z[start_z:end_z+1])

        if axis== 'x':
            propagation_result= np.zeros((self.Nx, len(files_sorted)))
            count= 0
            for file in files_sorted:
                print(file)
                e= np.load(load_path+ file)
                I = self.intensity_from_field(e)
                propagation_result[:, count]= I[timeSlice, :, self.Ny//2]
                count= count+1
        if axis== 'y':
            propagation_result= np.zeros((self.Ny, len(files_sorted)))
            count= 0
            for file in files_sorted:
                print(file)
                e= np.load(load_path+ file)
                I = self.intensity_from_field(e)
                propagation_result[:, count]= I[timeSlice, self.Nx//2, :]
                count= count+1       
        
        start_z= int(re.findall(r'\d+', files_sorted[0])[0])
        end_z= int(re.findall(r'\d+', files_sorted[-1])[0])
        z_grid= np.array(self.z[start_z:end_z+1])
        z_grid= np.array(self.z)
        
        plt.figure(figsize=(8, 2.5))
        plt.pcolormesh(z_grid, getattr(self, axis), propagation_result)
        plt.ylim(xylim)
        plt.xlabel('z (m)')
        plt.ylabel(axis+ ' (um)')
        plt.colorbar(label= 'Intensity ($10^{14}$W/cm$^2$)')
        plt.twinx()
        plt.plot(z_grid, propagation_result[int(propagation_result.shape[0]//2), :], \
                'w--')
        return z_grid, getattr(self, axis), propagation_result

    def plot_propagation_array(self, propagation_result, axis= 'x', xylim= None):
        z_grid= np.array(self.z)
        
        plt.figure(figsize=(8, 2.5))
        plt.pcolormesh(z_grid, getattr(self, axis), propagation_result)
        plt.ylim(xylim)
        plt.xlabel('z (m)')
        plt.ylabel(axis+ ' (um)')
        plt.colorbar(label= 'Intensity ($10^{14}$W/cm$^2$)')
        plt.twinx()
        plt.plot(z_grid, propagation_result[int(propagation_result.shape[0]//2), :], \
                'w--')
        
        

class GaussianPulse(Pulse):
    """ A laser pulse class that creates a Gaussian electric field. 
    
    The pulse is Gaussian in both space and time.
    
    Parameters
    ----------
    E0 : double
        The peak value of the electric field at the Gaussian waist in GV/m. 
    waist : double
        The spot size of the Gaussian waist.
    z : double
        The position relative to the waist to start the beam at. +z is after
        the waist, -z is before the waist.
    tau : double
        The RMS duration of the pulse.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['E0',
                 'waist',
                 'z0',
                 'tau'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        k = self.k
        w0 = self.waist
        z0 = self.z0
        E0 = self.E0
        t2 = np.reshape(self.t, (self.Nt, 1, 1))**2
        x2 = np.reshape(self.x, (1, self.Nx, 1))**2
        y2 = np.reshape(self.y, (1, 1, self.Ny))**2
        # Calculate all the parameters for the Gaussian beam
        r2 = x2 + y2
        zr = np.pi*w0**2 / self.lam
        if z0 != 0:
            wz = w0 * np.sqrt(1+(z0/zr)**2)
            psi = np.arctan(z0/zr)
            Rz = z0 * (1 + (zr/z0)**2)
            # Create the Gaussian field
            e = E0 * w0 / wz * np.exp(-r2/wz**2-t2*np.pi/(2*self.tau**2)) \
                 * np.exp(1j*(k*z0 + k*r2/(2*Rz) - psi))
        else:
            e = E0 * np.exp(-r2/w0**2-t2*np.pi/(2*self.tau**2))
        super().initialize_field(e)


class RadialPulse(Pulse):
    """ A laser pulse with a radially dependent field and periodic phi phase.
    
    The pulse is Gaussian in time and has a radially dependent intensity and
    phase. It can have a periodic phase in phi of order n.
    
    Parameters
    ----------
    tau : double
        The RMS duration of the pulse.
    order : int
        The number of periods of the phase in phi.
    r : array-like
        An array of radial coordinates the electric field is specified at.
    E : array-like
        The radial electric field specified at each element in r.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['tau',
                 'order',
                 'r',
                 'E'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        order = self.order
        tau = self.tau
        x = self.x
        y = self.y
        t = self.t
        Nx = self.Nx
        Ny = self.Ny
        
        e = self.reconstruct_from_cyl(self.r, self.E, x, y)
        e = e[None, :, :] * np.exp(-t[:, None, None]**2 * np.pi/(2*tau**2))
        # Add the phi dependent phase
        phi = np.zeros((Nx, Ny), dtype='complex64') 
        # Handle when x/y -> ininity
        phi[int(Nx/2), int(Ny/2):] = np.pi/2
        phi[int(Nx/2), :int(Ny/2)] = -np.pi/2
        # Handle the positive x half plane
        sel = np.array(x > 0)
        xp = x[sel]
        xp = np.reshape(xp, (np.size(xp), 1))
        phi[int(Nx/2+1):, :] = np.arctan(y/xp)
        # Handle the negative x half plane
        sel = np.array(x < 0)
        xn = x[sel]
        xn = np.reshape(xn, (np.size(xn), 1))
        phi[:int(Nx/2), :] = np.arctan(y/xn) + np.pi
        e = e * np.exp(1j*phi*order)
        super().initialize_field(e)
        
