#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 19:07:01 2025

@author: valentinalee
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

class hipace():
    def create_hipace_plasma(path, header, xPath, yPath, zPath, savePath):
        fileList= hipace.get_sorted_file_list(path, header)
        x_grid= np.load(xPath)
        y_grid= np.load(yPath)
        z_grid= np.load(zPath)*1e-6
        output_file = os.path.join(savePath, "hipaceDensity.txt")
        with open(output_file, "w") as f:
            f.write("# HiPACE++ density table\n")
            f.write("# Each line: z_position [m]   superGaussian(x,y)\n\n")
    
            # --- loop through files ---
            for count, file in enumerate(fileList):
                profile = np.load(file)
                func_str, params= hipace.fit_super_gaussian_2d(x_grid, y_grid, profile)
                z_position = z_grid[count]
    
                # Write one line: z_position followed by fitted function
                f.write(f"{z_position:.6e}   {func_str}\n")
    
        print(f"Saved HiPACE density table to {output_file}")
#            x, y = np.meshgrid(x_grid, y_grid)
#            x0, y0, wx, wy, m= params
#            Z_fit= hipace.super_gaussian_2d(x, y, np.amax(profile), x0, y0, wx, wy, m)
           
            

    def get_sorted_file_list(path, header):
        """
        Return a sorted list of full file paths in `path` that start with `header`,
        sorted numerically by the integer following the last underscore in the filename.
    
        Example:
            Directory: CHER_data_1.h5, CHER_data_2.h5, CHER_data_10.h5
            header='CHER_data_'
            --> returns ['.../CHER_data_1.h5', '.../CHER_data_2.h5', '.../CHER_data_10.h5']
        """
        # List only files starting with header
        files = [f for f in os.listdir(path)
                 if f.startswith(header) and os.path.isfile(os.path.join(path, f))]
    
        # Extract the number following the last underscore
        def extract_number(filename):
            match = re.search(r'_(\d+)(?=\.[^.]+$)', filename)  # match digits before file extension
            return int(match.group(1)) if match else -1
    
        # Sort files numerically
        files.sort(key=extract_number)
    
        # Return full paths
        return [os.path.join(path, f) for f in files]


    def super_gaussian_2d(x, y, A, x0, y0, wx, wy, m):
        """
        Unrotated 2D super-Gaussian:
          f(x,y) = A * exp( - ( ( ((x-x0)/wx)**2 + ((y-y0)/wy)**2 ) )**(m/2) )
        """
        rx2 = ((x - x0) / (wx + 1e-30))**2
        ry2 = ((y - y0) / (wy + 1e-30))**2
        return A * np.exp(- (rx2 + ry2)**(m/2.0))

        
    def fit_super_gaussian_2d(x_grid, y_grid, z, max_order=20):
        """
        Fit a 2D super-Gaussian to (x,y)->z data.
    
        Parameters
        ----------
        x_grid, y_grid : 2D arrays
            Meshgrid-style coordinate grids matching z2d's shape.
        z2d : 2D array
            Measured profile.
        tie_xy : bool, optional
            If True, enforce wx == wy during the fit.
        max_order : float, optional
            Upper bound on the super-Gaussian order 'm'.
    
        Returns
        -------
        func_str : str
            A string of the fitted function f(x,y) you can eval/use later.
        params : dict
            Fitted parameters for convenience.
        """

   
        # Initial guesses (robust-ish)
        A0 = np.amax(z)
        if A0== 0:
            return 0, None
    
        wgt = np.clip(z - z.min(), 0, None) + 1e-30
        x0 = float(np.sum(wgt * x_grid) / np.sum(wgt))
        y0 = float(np.sum(wgt * y_grid) / np.sum(wgt))

        var_x = float(np.sum(wgt * (x_grid - x0)**2) / np.sum(wgt))
        var_y = float(np.sum(wgt * (y_grid - y0)**2) / np.sum(wgt))
        sx = np.sqrt(max(var_x, 1e-30))
        sy = np.sqrt(max(var_y, 1e-30))
        wx0 = 1.25 * sx
        wy0 = 1.25 * sy

        m0 = 4.0

        x, y = np.meshgrid(x_grid, y_grid)
        data_flat = z.ravel()
        x_flat = x.ravel()
        y_flat = y.ravel()

        initial_guesses = np.array([x0, y0, wx0, wy0, m0])
        def residual(params, x, y, data):
            x0, y0, sigma_x, sigma_y, p = params
            return data - hipace.super_gaussian_2d(x, y, A0, x0, y0, sigma_x, sigma_y, p)
        
        fitted_params, success = leastsq(residual, initial_guesses, args=(x_flat, y_flat, data_flat))
        x0, y0, wx, wy, m= fitted_params
        params = dict(A=A0, x0=x0, y0=y0, wx=wx, wy=wy, m=m)
        func_str = (
                f"({A0:.8e}) * 1e23* exp(-(( ((x-({x0:.8e}))/{wx:.8e})**2 + "
                f"((y-({y0:.8e}))/{wy:.8e})**2 )**({m:.8e}/2)))"
            )
    
        return func_str, fitted_params
