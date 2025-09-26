#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:36:12 2025

@author: valentinalee
"""

class materials:
    """
    Material class with built-in database of common materials.

    Usage:
    -------
    mat = Material("CaF2")  # loads CaF2 parameters
    print(mat.n0, mat.beta2, mat.n2)
    """

    # Internal database: add materials here
    _database = {
        "CaF2": {
            "n0": 1.43,             # linear refractive index at 800 nm
            "beta2": 41.283e-33,        # GVD (s^2/m) at 800nm
            "n2": 1.71e-20           # nonlinear refractive index (m^2/W)
        },
        "FusedSilica": {
            "n0": 1.45,
            "beta2": 35e-33,        # GVD (s^2/m)
            "n2": 2.19e-20
        },
        "Ti:Sapphire": {
            "n0": 1.76,
            "beta2": 58.115e-33,        # GVD (s^2/m)
            "n2": 3.2e-20
        },
        "Vacuum": {
            "n0": 1,
            "beta2": 0,        # GVD (s^2/m)
            "n2": 0
        }
    }

    def __init__(self, name: str):
        if name not in self._database:
            raise ValueError(f"Material '{name}' not found. Available: {list(self._database.keys())}")

        params = self._database[name]
        self.name = name
        self.n0 = params["n0"]
        self.beta2 = params["beta2"]
        self.n2 = params["n2"]

    def __repr__(self):
        return f"<Material {self.name}: n0={self.n0}, beta2={self.beta2:.2e} s^2/m, n2={self.n2:.2e} m^2/W>"
