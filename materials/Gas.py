#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:56:59 2025

@author: valentinalee
"""

class gas:
    """
    Gas class with built-in database of common atomic and molecular species
    for ionization calculations.

    Parameters stored:
    ------------------
    EI : float
        Ionization energy [eV].
    Z : int
        Atomic residue (which electron is being ionized: 1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    alpha : float, optional
        Static polarizability [Å³], if available.
    n2 : float, optional
        Nonlinear refractive index [m²/W], if available.

    Usage:
    -------
    >>> g = gas("H2")
    >>> print(g.EI, g.Z, g.l, g.m, g.alpha, g.n2)
    15.426 1 0 0 0.787 0.0
    """

    # Internal database
    _database = {
        "H":   {'EI': 13.5984,  'Z': 1, 'l': 0, 'm': 0, 'alpha': 0.667},
        "H2":  {'EI': 15.426,   'Z': 1, 'l': 0, 'm': 0, 'alpha': 0.787, 'n2': 0.0},
        "H2p": {'EI': 29.99,    'Z': 2, 'l': 0, 'm': 0},
        "He":  {'EI': 24.5874,  'Z': 1, 'l': 0, 'm': 0, 'alpha': 0.208},
        "Hep": {'EI': 54.4178,  'Z': 2, 'l': 0, 'm': 0},
        "Ar":  {'EI': 15.7596,  'Z': 1, 'l': 1, 'm': 0, 'alpha': 1.664},
        "Arp": {'EI': 27.62965, 'Z': 2, 'l': 1, 'm': 0},
        "Li":  {'EI': 5.3917,   'Z': 1, 'l': 0, 'm': 0, 'alpha': 24.33},
    }

    def __init__(self, name: str):
        if name not in self._database:
            raise ValueError(
                f"Gas '{name}' not found. Available: {list(self._database.keys())}"
            )

        params = self._database[name]
        self.name = name
        self.EI = params["EI"]
        self.Z = params["Z"]
        self.l = params["l"]
        self.m = params["m"]
        self.alpha = params.get("alpha", None)
        self.n2 = params.get("n2", None)

    def __repr__(self):
        alpha_str = f", alpha={self.alpha:.3f} Å³" if self.alpha is not None else ""
        n2_str = f", n2={self.n2:.2e} m²/W" if self.n2 is not None else ""
        return f"<Gas {self.name}: EI={self.EI:.4f} eV, Z={self.Z}, l={self.l}, m={self.m}{alpha_str}{n2_str}>"
