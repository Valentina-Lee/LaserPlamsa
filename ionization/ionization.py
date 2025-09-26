#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:39:10 2017

@author: rariniello
"""

import numpy as np
from ionization import adk


# Define some common gas species
# These are mol, short for molecule, dictionaries
"""
Parameters
----------
EI : double
    Ionization energy of the electron in eV.
Z : int
    Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
l : int, optional
    Orbital quantum number of the electron being ionized.
m : int, optional
    Magnetic quantum number of the electron being ionized.
"""
H = {'EI': 13.5984,
     'Z': 1,
     'l': 0,
     'm': 0,
     'alpha': 0.667}
H2 = {'EI': 15.426,
      'Z': 1,
      'l': 0,
      'm': 0,
      'alpha': 0.787}
H2p = {'EI': 29.99,
       'Z': 2,
       'l': 0,
       'm': 0}
He = {'EI': 24.5874,
      'Z': 1,
      'l': 0,
      'm': 0,
      'alpha':0.208}
Hep = {'EI': 54.4178,
       'Z': 2,
       'l': 0,
       'm': 0}
Ar = {'EI': 15.7596,
      'Z': 1,
      'l': 1,
      'm': 0,
      'alpha': 1.664}
Arp = {'EI': 27.62965,
       'Z': 2,
       'l': 1,
       'm': 0}
Li = {'EI': 5.3917,
      'Z': 1,
      'l': 0,
      'm': 0,
      'alpha': 24.33}


def keldysh(EI, I, wavelength):
    """ Calculates the Keldysh parameter.

        Calculates the Keldysh parameter from the ionization energy and laser
        intensity assuming a linearly polarized electric field. Tunneling
        ionization if gamma << 1, multi-photon ionization if gamma > 1.

        Parameters
        ----------
        EI : double
            Ionization energy of the electron in eV.
        I : array-like
            Intensity in 10^14 W/cm^2.
        wavelength
            Laser wavelength in um.

        Returns
        -------
        gamma : array-like
            Keldysh parameter.
    """
    gamma = np.sqrt(EI / (18.6*I*np.power(wavelength, 2)))
    return gamma


def field_from_intensity(I, n=1.0):
    """ Calculates the electric field from the intensity.

        Calculates the peak electric field from the intensity assuming a
        monochromatic propogating wave.

        Parameters
        ----------
        I : array-like
            Intensity in 10^14 W/cm^2.
        n : double
            Index of refraction of the medium the wave propogates through.

        Returns
        -------
        E : array-like
            Electric field in GV/m.
    """
    E = 27.4492 * np.sqrt(I/n)
    return E


def intensity_from_field(E, n=1.0):
    """ Calculates the electric field from the intensity.

    Calculates the peak electric field from the intensity assuming a
    monochromatic propogating wave.

    Parameters
    ----------
    E : array-like
        Electric field in GV/m.
    n : double
        Index of refraction of the medium the wave propogates through.

    Returns
    -------
    I : array-like
        Intensity in 10^14 W/cm^2.
    """
    I = n * (0.03643 * abs(E))**2
    return I


def gaussian_envelope(I, t, tau, chirp=0):
    """ Returns the envelope of the electric field of a Gaussian pulse.

    Parameters
    ----------
    I : double
        Peak intensity of the pulse.
    t : array-like
        Array of time points to return the electric field at.
    tau : double
        RMS length of the pulse.
    chirp : double
        Frequency chirp of the pulse.

    Returns
    -------
    E : array-like
        Array of electric field values at times given by array t.
    """
    a = np.pi / (2*tau**2)
    Gamma = a + chirp*1j
    E0 = field_from_intensity(I)
    E = E0 * np.exp(-Gamma * t**2)
    return E


def gaussian_field(I, t, f, tau, chirp=0):
    """ Returns the electric field of a Gaussian pulse.

    Parameters
    ----------
    I : double
        Peak intensity of the pulse in 10^14 W/cm^2.
    t : array-like
        Array of time points to return the electric field at.
    f : double
        Carrier wave frequency.
    tau : double
        RMS length of the pulse.
    chirp : double
        frequency chirp of the pulse.

    Returns
    -------
    E : array-like
        Array of electric field vlues at times given by array t.
    """
    w0 = 2*np.pi * f
    E = gaussian_envelope(I, t, tau, chirp) * np.exp(-1j*w0 * t)
    return E


def intensity_from_density(ion, frac):
    """ Retruns the intensity necessary to create a given density of plasma.

    Uses the ADK model to calculate the intensity needed to ionize a gas for a
    specific pulse shape.

    Parameters
    ----------
    ion : dictionary
        atom : dictionary
            See the description in ionization.ionization for details.
        tau : double
            Temporal length of the pulse in fs.
        type : string
            The type of pulse, i.e. gaussian, flattop, etc.
    frac : array-like
        Array of desired ionization fraction.

    Returns
    -------
    I : array-like
        Array of intensity for each ionization fraction.
    """
    tau = ion['tau']
    atom = ion['atom']
    # Depending on the type, define the envelope function
    if ion['type'] == 'gaussian':
        def f(I, t):
            return gaussian_envelope(I, t, tau).real
        N = 5000
        # All species we consider are fully ionized by 20X10^14 W/cm^2
        I = np.linspace(0, 30, N)
        t = np.linspace(-3*tau, 3*tau, N)
    # Calculate the interpolating function
    g = adk.intensity_func(atom['EI'], I, t, f,
                           atom['Z'], atom['l'], atom['m'], True)
    I = g(frac)
    return I
