# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:10:28 2024

@author: sdd380
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

def normalize(vek, interval):
    
    # Syntax: y=normalize(VEK, interv) - normalisiert den Vektor VEK in Schritte der L?nge 'interv' ausgehend von seiner urspr?nglichen
    # "Zeitachse" auf 100# in #-Schritten, wie durch "interv" angegeben. "VEK"
    # sind die zu normalisierenden Daten, "interv" gibt die gew?nschte
    # Schrittweite der Normalisierung an.
    t_mod = np.linspace(0, 100, len(vek))

    # Define interval and create t_nor array
    
    t_nor = np.arange(0, 100 + interval, interval)
    
    # Interpolate using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    pchip_interpolator = PchipInterpolator(t_mod, vek)
    y = pchip_interpolator(t_nor)
    
    return y