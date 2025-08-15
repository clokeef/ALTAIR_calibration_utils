import numpy as np
from geopy import distance
from scipy.signal import convolve2d, fftconvolve

from JG_Streaktools import *

def check_err(var, type1, type2):
    """
    checks if a given variable has an associated uncertainty, if the initially passed variable has no associated
    uncertainty, it is returned with an uncertainty of 0
    :param var: variable to check
    :param type1: class type for no associated uncertainty
    :param type2: class type for a variable with associated uncertainty
    :return: variable value, and uncertainty
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
    converts a given 2-tuple of coordinates (lat, lon) from deg, arcmin, arcsec, to decimal degrees
    :param coords: 2-tuple of lists: (latitude [deg, arcmin, arcsec], longitude [deg ,arcmin, arcsec])
    :return: 2, tuple of the latitude and longitude in decimal degrees
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


def get_geod_err(obs_coords_val, sat_coords_val, obs_coords_err, sat_coords_err):
    """
    forces through uncertainty in the geodesic calculation given the uncertainty on the coordinates
    :param obs_coords_val: observer coordinates in decimal degrees
    :param sat_coords_val: light source coordinates in decimal degrees
    :param obs_coords_err: uncertainty for observer coordinates in decimal degrees
    :param sat_coords_err: uncertainty for light source coordinates in decimal degrees
    :return: max and min geod values given the uncertainty
    """
    if obs_coords_val[0] >= sat_coords_val[0]:
        if obs_coords_val[1] >= sat_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            max_sat_coords = (sat_coords_val[0] - sat_coords_err[0], sat_coords_val[1] - sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            min_sat_coords = (sat_coords_val[0] + sat_coords_err[0], sat_coords_val[1] + sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

        elif obs_coords_val[1] < sat_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            max_sat_coords = (sat_coords_val[0] - sat_coords_err[0], sat_coords_val[1] + sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            min_sat_coords = (sat_coords_val[0] + sat_coords_err[0], sat_coords_val[1] - sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

    if obs_coords_val[0] < sat_coords_val[0]:
        if obs_coords_val[1] >= sat_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            max_sat_coords = (sat_coords_val[0] + sat_coords_err[0], sat_coords_val[1] - sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            min_sat_coords = (sat_coords_val[0] - sat_coords_err[0], sat_coords_val[1] + sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

        elif obs_coords_val[1] < sat_coords_val[1]:
            max_obs_coords = (obs_coords_val[0] - obs_coords_err[0], obs_coords_val[1] - obs_coords_err[1])
            max_sat_coords = (sat_coords_val[0] + sat_coords_err[0], sat_coords_val[1] + sat_coords_err[1])

            max_geod = distance.distance(max_obs_coords, max_sat_coords).km

            min_obs_coords = (obs_coords_val[0] + obs_coords_err[0], obs_coords_val[1] + obs_coords_err[1])
            min_sat_coords = (sat_coords_val[0] - sat_coords_err[0], sat_coords_val[1] - sat_coords_err[1])

            min_geod = distance.distance(min_obs_coords, min_sat_coords).km

    return max_geod, min_geod

def get_radec_err(x_pix, y_pix, wcs_obj):
    """
    get the uncertainty in ra and dec from a coordinate transformation from pixel values to ra and dec
    :param x_pix_val: centre of the streak on the x-axis in the image
    :param x_pix_err: uncertainty in the x-axis pixel centre
    :param y_pix_val: centre of the streak on the y-axis in the image
    :param y_pix_err: uncertainty in the y-axis pixel centre
    :param wcs_obj: wcs astropy object
    :return: returns the uncertainty in ra and dec
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


def downsample(im, factor=2, normalization='sum'):
    """
    This is an adaptation of downsample form JG_streaktools,
    it will take an image and downsample by a given factor.
    :param im: input image data to be downsampled
    :param factor: sampling factor, the image will be reduced in size by this factor
    :param normalization: mode for whether the downsample takes the sum of pixels in the kernel or the median
    :return: returns a downsampled image
    """

    if factor is None or factor < 1:
        return im

    if not isinstance(factor, int):
        raise TypeError('Input "factor" must be a scalar integer. ')

    k = np.ones((factor, factor), dtype=im.dtype)
    if normalization == "mean":
        k = k / np.sum(k)
    elif normalization != "sum":
        raise KeyError('Input "normalization" must be "mean" or "sum". 'f'Got "{normalization}" instead. ')

    im_conv = convolve2d(im, k, mode="same")

    return im_conv[factor - 1:: factor, factor - 1:: factor]