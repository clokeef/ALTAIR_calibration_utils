#import necessary packages
import numpy as np
from colour import sd_mesopic_luminous_efficiency_function

def get_flux(lums, peaks, bandpass, err=True):
    """
    determine the peak radiant intensity of the light source
    :param lums: array of the luminous radiance and error of the components of the light source
    :param peak: list of the peak wavelengths of the components of the light source
    :param bandpass: array of the bandpass in each filter at each peak wavelength of the components of the light source
    :param err: boolean to indicate if the error is returned
    :return: peak radiant intensity of the light source
    """
    lum, lum_err = check_err(lums, tuple, np.ndarray)
    peak, peak_err = check_err(peaks, tuple, np.ndarray)

    flux = 0
    flux_err = 0

    for i in range(len(lum)):
        distr = sd_mesopic_luminous_efficiency_function(lum[i])
        peak_ind = distr.domain == peak[i] * (1E9)
        conv = distr.values[peak_ind][0]

        flux += (1 / (np.pi * 683.002)) * ((lum[i] * (np.sum(bandpass[i]))) / conv)
        if err == True:
            flux_err += (1 / (np.pi * 683.002)) * ((lum_err[i] * (np.sum(bandpass[i]))) / conv)

    if err == True:
            flux_err += (1/(np.pi*683.002))*((lum_err[i]*(np.sum(bandpass[i])))/conv)

    if err == True:
        return (flux, flux_err)

    if err == False:
        return flux


def get_F0(band_pass, peaks, err = True):
    """
    get the flux zero-point in the AB magnitude system for the light source array
    :param band_pass: array of the bandpass at each peak wavelength of the components of the light source
    :param peaks: array of the peak wavelengths and error of the components of the light source
    :param err: boolean to indicate if the error is returned
    :return: flux zero-point in the AB magnitude system
    """
    peak, peak_err = check_err(peaks, tuple, np.ndarray)

    F0 = 0
    F0_err = 0

    for i in range(len(peak)):
        F0 += 3631E-26*3E8*((np.sum(band_pass[i]))/peak[i])
        if err == True:
            F0_err += 3631E-26*3E8*((np.sum(band_pass[i]))*peak_err[i]/(peak[i]**2))

    if err == True:
        return (F0, F0_err)

    if err == False:
        return F0

def get_Fmeas(peak_rad, theta, dist, delta_t, exp_time, err = True):
    """
    calculates the measured flux at the observer
    :param peak_rad: peak radiant intensity of the light source
    :param theta: separation angle
    :param dist: distance to the light source
    :param delta_t: time length of the pulse which produces the measured streak
    :param exp_time: exposure time of the image
    :param err: boolean to indicate if error is returned
    :return: measured flux at observer
    """
    peak_rad_val,peak_rad_err = check_err(peak_rad, float, tuple)
    theta_val, theta_err = check_err(theta, float, tuple)
    dist_val, dist_err = check_err(dist, float, tuple)
    delta_t_val, delta_t_err = check_err(delta_t, float, tuple)
    exp_time_val, exp_time_err = check_err(exp_time, float, tuple)

    Fmeas = peak_rad_val*np.cos(theta_val)*delta_t_val/(dist_val**2)/exp_time_val

    if err == True:
        d_dE = np.cos(theta_val)*delta_t_val/(dist_val**2)/exp_time_val
        d_dtheta = peak_rad_val*np.sin(theta_val)*delta_t_val/(dist_val**2)/exp_time_val
        d_ddist = 2*peak_rad_val*np.cos(theta_val)*delta_t_val/(dist_val**3)/exp_time_val
        d_dt = peak_rad_val*np.cos(theta_val)/(dist_val**2)/exp_time_val
        d_dT = peak_rad_val*np.cos(theta_val)*delta_t_val/(dist_val**2)/(exp_time_val**2)

        Fmeas_err = d_dE*peak_rad_err + d_dtheta*theta_err  + d_ddist*dist_err + d_dt*delta_t_err + d_dT*exp_time_err

        return (Fmeas, Fmeas_err)

    if err == False:
        return Fmeas

def get_magzero(F_meas, F0, count, err=True):
    """
    calculates the magnitude zero-point given a known magnitude and recorded counts for image callibration
    :param F_meas: measured flux at the observer
    :param F0: flux zero-point in the AB magnitude system
    :param count: recorded counts from photometry
    :param err: boolean to indicate if error is returned
    :return: calibrated magnitude zero-point
    """
    F_meas_val, F_meas_err = check_err(F_meas, float, tuple)
    F0_val, F0_err = check_err(F0, float, tuple)
    count_val, count_err = check_err(count, float, tuple)

    mag_zero = -2.5*np.log10(F_meas_val/F0_val) + 2.5*np.log10(count_val)

    if err == True:

        mag_zero_err = 2.5/(np.log(10))*((F_meas_err/F_meas_val)+(F0_err/F0_val)+(count_err/count_val))

        return (mag_zero, mag_zero_err)

    if err == False:

        return mag_zero
