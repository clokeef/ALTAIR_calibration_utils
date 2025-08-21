import numpy as np
from geopy import distance
from scipy.signal import convolve2d, fftconvolve

from JG_Streaktools import *

###check_err needs to be modified as right now it only handles floats, tuples, and arrays
def check_err(var, type1, type2):
    """
    checks if a given variable has an associated uncertainty, if not an uncertainty of 0 is assigned
    Args:
        var: variable to check
        type1: variable type for no associated uncertainty
        type2: variable type for associated uncertainty

    Returns: variable value and uncertainty
    """
    if isinstance(var, type1):
        if type1 == float:

            return var, 0
        if type1 == tuple:

            return var, (0,0)

    elif isinstance(var, type2):

        return var[0], var[1]

    else:
        print(f'variable must be either {type1} or {type2}')


def coord_conv(coords):
    """
    converts a 2-tuple of coordinates (lat, lon) from deg,arcmin,arcsec to decimal degrees
    Args:
        coords: 2-tuple of lists: (latitude [deg, arcmin, arcsec], longitude [deg, arcmin, arcsec])

    Returns: 2-tuple of latitude and longitude in decimal degrees,
            if the input (lat, lon) is already in decimal degrees returns a message indicating no conversion occurred
    """
    if not isinstance(coords, tuple):
        print('Please enter your coordinates as a tuple (N,W) of floats, or a tuple of lists of floats')

    if isinstance(coords[0], float) == True and isinstance(coords[1], float) == True:
        print('no conversion required')
        return coords

    else:
        coords_n = (coords[0][0]) + (coords[0][1] / 60) + (coords[0][2] / 3600)
        coords_w = (coords[1][0]) + (coords[1][1] / 60) + (coords[1][2] / 3600)
        return tuple((coords_n, coords_w))


def get_geod_err(obs_coords_val, sour_coords_val, obs_coords_err, sat_coords_err):
    """
    forces through uncertainty in the geodesic distance determination given the uncertainty in the coordinates
    Args:
        obs_coords_val: 2-tuple of observer coordinates in decimal degrees
        sour_coords_val: 2-tuple of source coordinates in decimal degrees
        obs_coords_err: 2-tuple of observer coordinate uncertainties in decimal degrees
        sat_coords_err: 2-tuple of source coordinate uncertainties in decimal degrees

    Returns: max and min geodesic distance
    """
    if obs_coords_val[0] >= sour_coords_val[0]:
        if obs_coords_val[1] >= sour_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            max_sat_coords = (sour_coords_val[0] - sat_coords_err[0], sour_coords_val[1] - sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            min_sat_coords = (sour_coords_val[0] + sat_coords_err[0], sour_coords_val[1] + sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

        elif obs_coords_val[1] < sour_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            max_sat_coords = (sour_coords_val[0] - sat_coords_err[0], sour_coords_val[1] + sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            min_sat_coords = (sour_coords_val[0] + sat_coords_err[0], sour_coords_val[1] - sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

    if obs_coords_val[0] < sour_coords_val[0]:
        if obs_coords_val[1] >= sour_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            max_sat_coords = (sour_coords_val[0] + sat_coords_err[0], sour_coords_val[1] - sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            min_sat_coords = (sour_coords_val[0] - sat_coords_err[0], sour_coords_val[1] + sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

        elif obs_coords_val[1] < sour_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            max_sat_coords = (sour_coords_val[0] + sat_coords_err[0], sour_coords_val[1] + sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            min_sat_coords = (sour_coords_val[0] - sat_coords_err[0], sour_coords_val[1] - sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

    return max_geod, min_geod

def get_radec_err(x_pix, y_pix, wcs_obj):
    """
    get the uncertainty in ra and dec from a coordinate transformation from pixel values to ra, dec given some uncertatiny in pixel values
    Args:
        x_pix: x pixel value of the object
        y_pix: y pixel value of the object
        wcs_obj: astropy.wcs WCS object for the image

    Returns: ra uncertainty and dec uncertainty in degrees
    """
    x_pix_val, x_pix_err = check_err(x_pix, float, tuple)
    y_pix_val, y_pix_err = check_err(y_pix, float, tuple)

    #we want to define the ra,dec which define the corners of the pixel uncertainty
    A = (x_pix_val-x_pix_err, y_pix_val+y_pix_err)
    B = (x_pix_val+x_pix_err, y_pix_val+y_pix_err)
    C = (x_pix_val-x_pix_err, y_pix_val-y_pix_err)
    D = (x_pix_val+x_pix_err, y_pix_val-y_pix_err)
    coords = np.array([A,B,C,D])
    corners = wcs_obj.all_pix2world(coords, 0)

    ra_max = np.max(corners[:,0])
    ra_min = np.min(corners[:,0])
    dec_max = np.max(corners[:,1])
    dec_min = np.min(corners[:,1])

    ra_err = (ra_max-ra_min)/2
    dec_err = (dec_max-dec_min)/2

    return ra_err, dec_err


def downsample(img, factor=2, normalization='sum'):
    """
    This is an adaptation of downsample form JG_streaktools,
    it will take an image and downsample by a given factor.
    Args:
        img: image to downsample
        factor: downsample factor, default is 2
        normalization: normalization mode, default is 'sum'

    Returns:
        downsampled image
    """

    if factor is None or factor < 1:
        return img

    if not isinstance(factor, int):
        raise TypeError('Input "factor" must be a scalar integer. ')

    k = np.ones((factor, factor), dtype=img.dtype)
    if normalization == "mean":
        k = k / np.sum(k)
    elif normalization != "sum":
        raise KeyError('Input "normalization" must be "mean" or "sum". 'f'Got "{normalization}" instead. ')

    im_conv = convolve2d(img, k, mode="same")

    return im_conv[factor - 1:: factor, factor - 1:: factor]
