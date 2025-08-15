# ALTAIR_calibration_utils

The goal of this code is to serve as a companion to JG_Streaktools to allow for quick and easy magnitude zero-point calibration of astronomical images of streaked sources. 

The repository has multiple libraries which serve to:
- determine the flux and magnitude of a known satellite or weather balloon source given its colour and energy output with uncertainty handling.
- detemine an effective psf from an image.
- determine the geometry of an observer and the observed beacon with uncertainty handling.
- set up a JG_streaktools interface for photometry.
- estimate the magnitude zeropoint of an image from a plate solution.
- simulate preprocessed images of stars fields with noise.
- get plate solutions from astrometry.net
