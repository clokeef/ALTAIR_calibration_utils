#import necessary packages
import numpy as np

#handling geodesic distances over the surface of the earth
from geopy import distance

#import astronomical coordinate handling
from astropy.time import Time
from astropy.wcs import WCS
import astropy.wcs.utils as wcsutils
from astropy import units
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun

from utils import *

#global variables
R = 6371 #radius of the Earth in km

def get_geod(obs_coords, sat_coords, err=True):
    """
    determine the geodesic distance between two coordinate points
    :param obs_coords: array of the observer coordinates and uncertainty for lat and lon
    :param sat_coords: array of the light source coordinates and uncertainty for lat and lon
    :param err: boolean which determines if the error is determined
    :return: the geodesic distance between these two points
    """
    obs_coords_val, obs_coords_err = check_err(obs_coords, tuple, np.ndarray)

    sat_coords_val, sat_coords_err = check_err(sat_coords, tuple, np.ndarray)


    geod = distance.distance(sat_coords_val, obs_coords_val).km

    if err == True:

        max_geod, min_geod = get_geod_err(obs_coords_val, sat_coords_val, obs_coords_err, sat_coords_err)

        geod_minus = (geod-min_geod)
        geod_plus = (geod-max_geod)
        geod_err = round(np.mean([np.abs(geod_minus), np.abs(geod_plus)]), 3)

        return (geod, geod_err)

    if err == False:

        return geod

def get_alpha(x_pix, y_pix, wcs, obs_coords, obs_time, utc = True, time_zone = 0, err = True):
    """
    converts the pixel location of the light source into the local altitude angle, alpha
    :param x_pix: centre location of light source in x
    :param y_pix: centre location of light source in y
    :param wcs: wcs file from astrometry.net plat solution
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param obs_time: time of observation
    :param utc: boolean to indicate is the observation time is given in UTC, default is True
    :param time_zone: int of the time zone offset, default is 0
    :param err: boolean to indicate if the error is returned
    :return: local altitude angle in degrees
    """
    x_pix_val, x_pix_err = check_err(x_pix, float, tuple)

    y_pix_val, y_pix_err = check_err(y_pix, float, tuple)

    obs_coords_val, obs_coords_err = check_err(obs_coords, tuple, np.ndarray)

    obs_lat = obs_coords_val[0]
    obs_lon = obs_coords_val[1]

    lat_err = obs_coords_err[0]
    lon_err = obs_coords_err[1]

    #initialize a wcs instance in astropy
    wcs_obj = WCS(wcs)

    #set the location of the streak on the celestial sphere
    streak = wcsutils.pixel_to_skycoord(x_pix_val, y_pix_val, wcs_obj)

    location = EarthLocation(lat=obs_lat * units.deg, lon=-obs_lon * units.deg)

    if utc == False:
        utcoffset = time_zone * units.hour  # timezone offset

        time = Time(obs_time, location=location) - utcoffset

    else:
        time = Time(obs_time, location=location)

    streakaltaz = streak.transform_to(AltAz(obstime=time, location=location))
    alpha = np.deg2rad(streakaltaz.alt)

    lst = time.sidereal_time('apparent')
    lst_rad = lst.to_value(units.rad)

    ra = streak.ra.to_value('rad')
    dec = streak.dec.to_value('rad')

    if err == True:
        ra_err, dec_err = get_radec_err(x_pix, y_pix, wcs_obj)

        #we are now going to get the error in the altitude angle
        phi = obs_lat
        arg = np.sqrt(1-(np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.cos(lst_rad-ra))**2)
        d_dphi = np.cos(phi)*np.sin(dec) - np.sin(phi)*np.cos(dec)*np.cos(lst_rad-ra)/arg
        d_ddec = np.sin(phi)*np.cos(dec) - np.cos(phi)*np.sin(dec)*np.cos(lst_rad-ra)/arg
        d_dra = np.cos(phi)*np.sin(dec)*np.sin(lst_rad-ra)/arg

        alpha_err = np.abs(d_dphi)*lat_err + np.abs(d_ddec)*dec_err + np.abs(d_dra)*ra_err

        return (alpha.value, alpha_err)

    if err ==False:

        return alpha.value

def get_alt(obs_coords, sat_coords, alpha, err = True):
    """
    get the altitude in km of the light source
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param sat_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the light source
    :param alpha: local horizontal altitude angle in degrees
    :param err: boolean to indicate if error is returned
    :return: altitude in km of the light source
    """

    b = get_geod(obs_coords, sat_coords, err = err)

    b_val, b_err = check_err(b, float, tuple)
    alpha_val, alpha_err = check_err(alpha, float, tuple)

    beta = b_val/R

    a = R*((np.cos(alpha_val)*(1-np.cos(beta)) + np.sin(alpha_val)*np.sin(beta))/(np.cos(alpha_val)*np.cos(beta) - np.sin(alpha_val)*np.sin(beta)))

    if err == True:
        d_dalpha = R * ((np.sin(beta))/((np.cos(alpha_val)*np.cos(beta) - np.sin(alpha_val)*np.sin(beta))**2))
        d_db = (np.cos(alpha_val)*(np.sin(alpha_val)*np.cos(beta) + np.cos(alpha_val)*np.sin(beta)))/((np.cos(alpha_val)*np.cos(beta) - np.sin(alpha_val)*np.sin(beta))**2)

        a_err = d_dalpha*alpha_err + d_db*b_err

        return (a, a_err)

    if err == False:
        return a

def get_sepdist(obs_coords, sat_coords, alpha, err = True):
    """
    get the separation distance between the observer and light source
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param sat_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the light source
    :param alpha: local horizontal altitude angle in degrees
    :param err: boolean to indicate if the error is returned
    :return: separation distance between the observer and light source
    """
    alt = get_alt(obs_coords, sat_coords, alpha, err = err)

    alpha_val, alpha_err = check_err(alpha, float, tuple)
    alt_val, alt_err = check_err(alt, float, tuple)

    arg = (((R + alt_val) / R) ** 2) - (np.cos(alpha_val) ** 2)
    d = R * (np.sqrt(arg) - np.sin(alpha_val))

    if err == True:
        d_dalpha = R*(((np.sin(alpha_val)*np.cos(alpha_val))/(np.sqrt(arg))) - np.cos(alpha_val))
        d_dalt = ((R+alt_val)/R)/(np.sqrt(arg))

        d_err = d_dalpha*alpha_err + d_dalt*alt_err

        return (d, d_err)

    if err == False:

        return d



def get_nadir(obs_coords, sat_coords, alpha, err = True):
    """
    get the angle between the light source nadir and the line of sight to the observer
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param sat_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the light source
    :param alpha: local horizontal altitude angle in degrees
    :param err: boolean to indicate if the error is returned
    :return: angle between the light source nadir and the line of sight
    """
    d = get_sepdist(obs_coords, sat_coords, alpha, err = err)

    alpha_val, alpha_err = check_err(alpha, float, tuple)
    d_val, d_err = check_err(d, float, tuple)

    r_star = np.sqrt((R / d_val) ** 2 + 2 * (R / d_val) * np.sin(alpha_val) + 1)
    nadir = np.pi / 2 - alpha_val - np.arcsin(np.cos(alpha_val) / r_star)

    if err == True:
        d_dalpha = 1-((1/np.sqrt((r_star**2)-(np.cos(alpha_val)**2)))*(np.sin(alpha_val)+((R/d_val)*(np.cos(alpha_val)**2)/(r_star**2))))
        d_ddist = (1/np.sqrt((r_star**2)-(np.cos(alpha_val)**2)))*((R/d_val)*np.cos(alpha_val)*((np.sin(alpha_val)/d_val)-(R/(d_val**2))))/(r_star**2)

        nadir_err = d_dalpha*alpha_err + d_ddist*d_err

        return (nadir, nadir_err)

    if err == False:

        return nadir


def get_gamma(obs_coords, sat_coords, err = True):
    """
    get the rotation angle, gamma, of the line of sight from the light source x-axis
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param sat_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the light source
    :param err: boolean to indicate if the error is returned
    :return: rotation angle, gamma, of the line of sight
    """
    obs_coords_val, obs_coords_err = check_err(obs_coords, tuple, np.ndarray)
    sat_coords_val, sat_coords_err = check_err(sat_coords, tuple, np.ndarray)

    lambda_1 = obs_coords_val[0]
    lambda_1_err = obs_coords_err[0]
    phi_1 = obs_coords_val[1]
    phi_1_err = obs_coords_err[1]

    lambda_2 = sat_coords_val[0]
    lambda_2_err = sat_coords_err[0]
    phi_2 = sat_coords_val[1]
    phi_2_err = sat_coords_err[1]

    del_lambda = lambda_2 - lambda_1
    del_phi = phi_2 - phi_1

    x = distance.distance((lambda_1, phi_1), (lambda_2, phi_1)).km
    y = distance.distance((lambda_1, phi_1), (lambda_1, phi_2)).km

    if del_phi < 0:
        if del_lambda < 0:
            gamma = np.arctan(y/x)
        elif del_lambda > 0:
            gamma = -np.arctan(y/x)
        elif del_lambda == 0:
            gamma = 0

    elif del_phi > 0:
        if del_lambda < 0:
            gamma = np.pi - np.arctan(y/x)
        elif del_lambda > 0:
            gamma = np.pi + np.arctan(y/x)
        elif del_lambda == 0:
            gamma = np.pi

    elif del_phi == 0:
        if del_lambda < 0:
            gamma = np.pi/2
        elif del_lambda > 0:
            gamma = -np.pi/2
        elif del_lambda == 0:
            gamma = 0

    else:
        return 'Could not determine the rotation angle, gamma'

    if err == True:
        gamma_err = (1/((del_phi**2)+(del_lambda**2)))*np.abs(del_phi*(lambda_1_err+lambda_2_err) + del_lambda*(phi_1_err+phi_2_err))

        return (gamma, gamma_err)

    if err == False:

        return gamma


def get_sepangle(obs_coords, sat_coords, alpha, pitch, roll, err = True):
    """
    get the angle between the pointing of the light source and the line of sight to the observer
    :param obs_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the observer
    :param sat_coords: coordinates in [(lat,lon),(lat_err,lon_err)] of the light source
    :param alpha: local horizontal altitude angle in degrees
    :param pitch: pitch angle of the light source in degrees
    :param roll: roll angle of the light source in degrees
    :param err: boolean to indicate if the error is returned
    :return: separation angle
    """
    nadir = get_nadir(obs_coords, sat_coords, alpha, err = err)
    gamma = get_gamma(obs_coords, sat_coords, err = err)

    nadir_val, nadir_err = check_err(nadir, float, tuple)
    gamma_val, gamma_err = check_err(gamma, float, tuple)
    roll_val, roll_err = check_err(roll, float, tuple)
    pitch_val, pitch_err = check_err(pitch, float, tuple)

    i = np.sin(nadir_val)*np.cos(gamma_val)
    j = np.sin(nadir_val)*np.sin(gamma_val)
    k = -np.cos(nadir_val)

    f = np.cos(roll_val)*np.sin(pitch_val)
    g = np.sin(roll_val)
    h = -np.cos(roll_val)*np.cos(pitch_val)

    cos_ang = i*f + j*g + k*h
    theta = np.arccos(cos_ang)

    if err == True:
        acos_d = -1/np.sqrt(1-(cos_ang**2))

        d_dnadir = np.abs(acos_d * (np.cos(nadir_val) * (np.cos(gamma_val)*np.cos(roll_val)*np.sin(pitch_val) + np.sin(gamma_val)*np.sin(roll_val)) - np.sin(nadir_val)*np.cos(roll_val)*np.cos(pitch_val)))
        d_dgamma = np.abs(acos_d * (np.sin(nadir_val) * (np.cos(gamma_val)*np.sin(roll_val) - np.sin(gamma_val)*np.cos(roll_val)*np.sin(pitch_val))))
        d_droll = np.abs(acos_d * (np.sin(nadir_val) * (np.sin(gamma_val)*np.cos(roll_val) - np.cos(gamma_val)*np.sin(roll_val)*np.sin(pitch_val)) - np.cos(nadir_val)*np.sin(roll_val)*np.cos(pitch_val)))
        d_dpitch = np.abs(acos_d * (np.sin(nadir_val) * np.cos(gamma_val) * np.cos(roll_val) * np.cos(pitch_val) - np.cos(nadir_val)  * np.cos(roll_val) * np.cos(pitch_val)))

        theta_err = d_dnadir*nadir_err + d_dgamma*gamma_err + d_droll*roll_err + d_dpitch*pitch_err

        return (theta, theta_err)

    if err == False:

        return theta
