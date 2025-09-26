#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:20:56 2025

@author: valentinalee
This code is based on Robert Ariniello's plasma-source code
"""
import os
import glob
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RectBivariateSpline

class Beam:
    """ The base class for beams.
    
    Implements base methods to check class construction.
    """
    keys = ['name',
            'path',
            'load']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        self.params = params
        self.check_params(params)
        self.params_to_attrs(params)
        # Create a folder to store the beam data in
        self.dirName = dirName = self.path + 'beams/beam_' + self.name + '/'
        self.filePre = dirName + self.name
        if self.load is True:
            self.load_params()
            self.load_beam()
        elif self.load is False:
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            self.clear_dir()
    
    def check_params(self, params):
        """ Check to ensure all required keys are in the params dictionary. """
        for key in self.keys:
            if key not in params:
                raise ValueError('The params object has no key: %s' % key)
    
    def params_to_attrs(self, params):
        """ Add all params as attributes of the class. """
        for key in params:
            setattr(self, key, params[key])
    
    def load_beam(self):
        """ Prototype for child specific loading setup. """
    
    # File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object. """
        np.save(self.filePre + '_params.npy', self.params)    
    
    def clear_dir(self):
        """ Clear all .npy files from the beam directory. """ 
        filelist = glob.glob(self.dirName + '*.npy')
        for f in filelist:
            os.remove(f)
            
    def load_params(self):
        """ Load the params from a saved file"""
        self.params = np.load(self.filePre + '_params.npy', allow_pickle=True).item()
        self.check_params(self.params)
        self.params_to_attrs(self.params)
        self.load = True
    
    # Physics functions
    #--------------------------------------------------------------------------
    
    def intensity_from_field(self, e, n=1):
        """ Calculates the time averaged intensity from the complex field. 
        
        This function goes from GV/m -> 10^14W/cm^2
        """
        return 1.32721e-3 * n * abs(e)**2
    
    def total_cyl_power(self, r, I):
        """ Calculates the total power in the beam.
        
        The functions takes an array of intensity values in 10^14W/cm^2 and an
        array of radii in um and returns power in TW.
        """
        r = r*1e-4 # Convert to cm
        return 2*np.pi*simps(r*I, r)*100
    
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def prep_data(self, data):
        """ Restructures data so that imshow displays it properly. 
        
        If data starts in (x, y) format, this will display it on the correct 
        axis.
        """
        return np.flipud(np.transpose(data))
    
    def reconstruct_from_cyl(self, r, data, x, y):
        """ Create a 2D field from a radial slice of a cylindircal field. """
        dataOfR = interp1d(r, data, bounds_error=False, fill_value=0.0, kind='cubic')
        return dataOfR(np.sqrt(x[:, None]**2 + y[None, :]**2))
    
    def reconstruct_from_cyl_beam(self, beam):
        """ Create a 2D field from a cyclindircally symmetric input beam. """
        r = beam.x
        if len(np.shape(beam.e)) == 2:
            data = np.array(beam.e[:, int(beam.Ny/2)])
        if len(np.shape(beam.e)) == 3:
            data = np.array(beam.e[int(beam.Nt/2),: , int(beam.Ny/2)])
        x = self.x
        y = self.y
        return self.reconstruct_from_cyl(r, data, x, y)
    
    def rescale_field(self, beam1, beam2):
        """
        Rescale complex field from beam1's (x1,y1) grid onto beam2's (x2,y2) grid.
        Uses cubic splines (RectBivariateSpline) and fill_value=0 outside beam1 bounds.
        Assumes beam1.e is 2D. Units of x/y must match between beams.
        """
        x1 = np.asarray(beam1.x)
        y1 = np.asarray(beam1.y)
        E1 = np.asarray(beam1.e)
    
        # RectBivariateSpline expects Z shape == (len(x), len(y)).
        # If your field is stored as (len(y), len(x)), transpose it.
        if E1.shape == (len(y1), len(x1)):
            E1 = E1.T
        elif E1.shape != (len(x1), len(y1)):
            raise ValueError(f"beam1.e shape {E1.shape} doesn't match grids (len(x1), len(y1))")
    
        # Ensure monotonic increasing grids
        if not (np.all(np.diff(x1) > 0) and np.all(np.diff(y1) > 0)):
            # sort and reorder field accordingly
            sx = np.argsort(x1)
            sy = np.argsort(y1)
            x1 = x1[sx]
            y1 = y1[sy]
            E1 = E1[np.ix_(sx, sy)]
    
        # Build cubic spline for real/imag parts
        spl_r = RectBivariateSpline(x1, y1, E1.real, kx=3, ky=3)
        spl_i = RectBivariateSpline(x1, y1, E1.imag, kx=3, ky=3)
    
        x2 = np.asarray(beam2.x)
        y2 = np.asarray(beam2.y)
    
        # Evaluate on target axes; returns shape (len(x2), len(y2))
        Er = spl_r(x2, y2)
        Ei = spl_i(x2, y2)
        E  = Er + 1j*Ei
    
        # Zero outside the source domain (RectBivariateSpline extrapolates by default)
        X2, Y2 = np.meshgrid(x2, y2, indexing='ij')  # match (len(x2), len(y2))
        mask = (
            (X2 < x1.min()) | (X2 > x1.max()) |
            (Y2 < y1.min()) | (Y2 > y1.max())
        )
        E[mask] = 0.0
    
        return E