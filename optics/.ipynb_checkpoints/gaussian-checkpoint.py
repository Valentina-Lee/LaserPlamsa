# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:51:10 2022

@author: Robert
"""

import numpy as np
from scipy.special import gamma
import scipy.constants
c = scipy.constants.c
eps_0 = scipy.constants.epsilon_0

def E_from_energy(energy, tau, w_0, m, n=1.0):
    """ Calculate the peak electric field strength of a super-Gaussian pulse
    
    Parameters
    ----------
    energy : float
        Energy of the pulse, J.
    tau : float
        FWHM pulse length of the pulse, s.
    w_0 : float
        Spot size of the pulse, m.
    m : int
        Super-Gaussian order of the transverse pulse shape. If using m from a fit, enter 2*m.
    n : float, optional
        Index of refraction of the medium the wave is in, defaults to 1.0.
    
    Returns
    -------
    E_0 : float
        The peak electric field strength of the pulse.
    """
    return np.sqrt(energy*2*np.sqrt(np.log(2))*m/(c*n*eps_0*np.pi**1.5*4**(-1/m)*gamma(2/m)*tau*w_0**2))

def fluence_from_intensity(I_0, tau):
    """ Calculate the laser fluence of a Gaussian pulse from the peak intensity.
    
    Parameters
    ----------
    I_0 : float
        Peak intensity of the laser pulse, W/m^2.
    tau : float
        FWHM pulse length of the pulse, s.
    
    Returns
    -------
    fluence : float
        Laser fluence of the pulse, J/m^2.
    """
    return I_0*tau*np.sqrt(np.pi/(4*np.log(2)))

def tran_super_gaussian(x, y, E_0, w_0, m, phi_0=None):
    """ Calculate the field of a super-Gaussian pulse on a transverse grid.
    
    Parameters
    ----------
    x : array of floats
        Location of each x grid point.
    y : array of floats
        Location of each y grid point.
    E_0 : float
        Peak electric field amplitude.
    w_0 : float
        Spot size of the pulse, m.
    m : int
        Super-Gaussian order of the transverse pulse shape, must be even.
    phi_0 : array of floats, optional
        Initial phase of the pulse, shape (Nx, Ny), defaults to 0, rad.
        
    Returns
    -------
    
    """
    r = np.sqrt(x[:, None]**2 + y[None, :]**2)
    if phi_0 is None:
        return E_0*np.exp(-(r/w_0)**m)
    else:
        return E_0*np.exp(-(r/w_0)**m)*np.exp(1j*phi_0)
    
def tran_super_gaussian_fit(x, y, E_0, w_0, m, phi_0=None):
    """ Calculate the field of a super-Gaussian pulse on a transverse grid.
    m can be any positive value, designed for when fitting a super-Gaussian
    
    Parameters
    ----------
    x : array of floats
        Location of each x grid point.
    y : array of floats
        Location of each y grid point.
    E_0 : float
        Peak electric field amplitude.
    w_0 : float
        Spot size of the pulse, m.
    m : int
        Super-Gaussian order of the transverse pulse shape (half of the normal definition).
    phi_0 : array of floats, optional
        Initial phase of the pulse, shape (Nx, Ny), defaults to 0, rad.
        
    Returns
    -------
    E : array of complex
        Electric field on the x-y grid.
    """
    r = np.sqrt(x[:, None]**2 + y[None, :]**2)
    if phi_0 is None:
        return E_0*np.exp(-((r/w_0)**2)**m)
    else:
        return E_0*np.exp(-((r/w_0)**2)**m)*np.exp(1j*phi_0)
    
def temporal_gaussian(t, tau, omega_0, b=0):
    """ Calculate the field of a Gaussian pulse on a temporal grid.

    Parameters
    ----------
    t : array of floats
        Time of each t grid point.
    tau : float
        FWHM intensity pulse length.
    omega_0 : float
        The pulses center angular frequency.
    b : float, optional
        Chirp parameter, defaults to 0.

    Returns
    -------
    E : array of complex
        Electric field on the t grid.
    """
    return np.exp(-2*np.log(2)*(t)**2/(tau**2))*np.exp(1j*(-b*t**2-omega_0*t))

def temporal_gaussian_envelope(t, tau):
    """ Calculate the field amplitude of a Gaussian pulse on a temporal grid.

    Parameters
    ----------
    t : array of floats
        Time of each t grid point.
    tau : float
        FWHM intensity pulse length.

    Returns
    -------
    E : array of float
        Electric field amplitude on the t grid.
    """
    return np.exp(-2*np.log(2)*(t)**2/(tau**2))