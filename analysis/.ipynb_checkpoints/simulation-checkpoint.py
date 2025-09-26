# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 23:45:23 2022

@author: Robert
"""

import numpy as np
#import importlib
#an = importlib.import_module("A-analysis")

from beam import laserbeam

def create_beam_from_image(image, size, lam, path, threads, Nx, X):
    image_in = image.data[int(0.5*image.height-size):int(0.5*image.height+size), int(0.5*image.width-size):int(0.5*image.width+size)]
    x_in = image.x[int(0.5*image.width-size):int(0.5*image.width+size)]
    y_in = image.y[int(0.5*image.height-size):int(0.5*image.height+size)]
    dx = x_in[1]-x_in[0]
    # Initial beam at camera resolution
    XD = (x_in[-1]-x_in[0]+dx)*1e3
    NxD = len(x_in)
    beamParams = {'Nx' : NxD,
                  'Ny' : NxD,
                  'X' : XD,
                  'Y' : XD,
                  'lam' : lam,
                  'path' : path,
                  'name' : 'CamBeam',
                  'threads' : threads,
                  'cyl' : False,
                  'load' : False}
    beamD = laserbeam.Laser(beamParams)
    e = np.fliplr(np.transpose(np.sqrt(image_in)))
    beamD.initialize_field(e)
    beamParams = {'Nx' : Nx,
                  'Ny' : Nx,
                  'X' : X,
                  'Y' : X,
                  'lam' : lam,
                  'path' : path,
                  'name' : 'CamBeam2',
                  'threads' : threads,
                  'cyl' : False,
                  'load' : False}
    beam = laserbeam.Laser(beamParams)
    e = beam.rescale_field(beamD, beam)
    beam.initialize_field(e)
    return beam

