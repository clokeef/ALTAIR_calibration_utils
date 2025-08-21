#import necessary packages
import numpy as np
import numpy.random
from astropy.table import Table
from JG_Streaktools import *

rng = np.random.default_rng()

def read_noise(img, read_std, gain):
    """
    simulate a plate of gaussian read noise with a given sigma which is the size of a given image
    Args:
        img: image for which read noise is being produced
        read_std: standard deviation in electrons of the gaussian distribution the pixel values are being drawn from
        gain: gain of the simulated image in electron/ADU

    Returns: plate of read noise
    """
    size = np.shape(img)
    noise = np.random.normal(0, read_std / gain, size=size)

    return noise


def signal_noise(img, just_noise=True):
    """
    produces poissonian noise for an image
    Args:
        img: image which produces the noise
        just_noise: boolean to indicate if only the noise image is returned

    Returns: image of the noise or the base image with the noise added
    """
    size = np.shape(img)

    noise_img = np.copy(img)
    for i in range(size[0]):
        for j in range(size[1]):
            mean = img[i][j]

            noise = np.random.poisson(mean)

            if just_noise == True:
                noise_img[i][j] = noise - mean

            else:
                noise_img[i][j] = noise

    return noise_img


def bias_noise(img, read_std, gain, bias_frame_num):
    """
    produces the noise associate with subtraction of the master bias frame during preprocessing
    Args:
        img: image which this noise will be added to
        read_std: standard deviation in electrons of the gaussian distribution the pixel values are being drawn from
        gain: gain of the simulated image in electron/ADU
        bias_frame_num: number of bias frames which are median combined to produce the master bias frame

    Returns: noise associated with the master bias frame
    """
    imgs = []
    for i in range(bias_frame_num):
        img_noise = read_noise(img, read_std, gain)
        imgs.append(img_noise)

    r_noise = np.median(imgs, axis=0)

    return r_noise


def dark_noise(img, read_std, gain, dark_current, exp_time, bias_frame_num, dark_frame_num):
    """
    produce the nosie associated with the subtraction of a master dark frame during preprocessing
    Args:
        img: image which this noise will be added to
        read_std: standard deviation in electrons of the gaussian distribution the pixel values are being drawn from
        gain: gain of the simulated image in electron/ADU
        dark_current: dark current of the simulated image in electron/second
        exp_time: exposure time of the simulated image
        bias_frame_num: number of bias frames which are median combined to produce the master bias frame
        dark_frame_num: number of dark frames which are median combined to produce the master dark frame

    Returns:
        noise associated with the master dark frame
    """
    #produce the noise of the master bias frame
    b_noise = bias_noise(img, read_std, gain, bias_frame_num)

    d_img = np.ones(shape=np.shape(img)) * dark_current * exp_time / gain

    d_imgs = []
    for i in range(dark_frame_num):
        # produce the readout noise for the darks
        r_img_noise = read_noise(img, read_std, gain)
        #produce the dark current noise
        d_img_noise = signal_noise(d_img)
        #comine the readout and dark noise with the noise associated to subtracting the master bias frame
        img_noise = r_img_noise + d_img_noise + b_noise
        d_imgs.append(img_noise)
    d_noise = np.median(d_imgs, axis=0)

    return d_noise


def img_noise(img, read_std, gain, dark_current, exp_time, bias_frame_num, dark_frame_num, just_noise=True):
    """
    adds noise to the image associated with preprocessing assuming enough flat frames were taken the flat field adds negligible error
    Args:
        img: image to add noise to
        read_std: standard deviation in electrons of the gaussian distribution the pixel values are being drawn from
        gain: gain of the simulated image in electron/ADU
        dark_current: dark current of the simulated image in electron/second
        exp_time: exposure time of the simulated image
        bias_frame_num: number of bias frames which are median combined to produce the master bias frame
        dark_frame_num: number of dark frames which are median combined to produce the master dark frame
        just_noise: boolean to indicate if only the noise is returned

    Returns: the noise or the image with the noise added
    """
    img_sig_noise = signal_noise(img)

    d_img = np.ones(shape=np.shape(img)) * dark_current *exp_time /gain

    r_img_noise = read_noise(img, read_std, gain)
    d_img_noise = signal_noise(d_img)
    d_master_noise = dark_noise(img, read_std, gain, dark_current, exp_time, bias_frame_num, dark_frame_num)
    b_noise = bias_noise(img, read_std, gain, bias_frame_num)

    tot_noise = img_sig_noise + r_img_noise + b_noise + d_img_noise + d_master_noise

    if just_noise == True:

        return tot_noise

    else:
        return img + tot_noise


def gen_star_field(img, num, psf, mag_zero, low_count, high_count):
    """
    randomly generates point sources on an image
    Args:
        img: image to populate with sources
        num: number of sources to produce
        psf: psf of the sources
        mag_zero: magnitude zero-point ot set for the image
        low_count: lower bound of source counts
        high_count: higher bound of source counts

    Returns: image populated with sources, correlation table of the location, counts, and magnitudes of the generated sources
    """
    ny, nx = np.shape(img)

    x = []
    y = []
    flux = []
    mags = []
    for i in range(num):
        xi = rng.integers(low=0, high=nx)
        x.append(xi)
        yi = rng.integers(low=0, high=ny)
        y.append(yi)
        flux_i = rng.uniform(low=low_count, high=high_count)
        flux.append(flux_i)
        psf_i = (psf / np.sum(psf)) * flux_i
        mags.append(-2.5*np.log10(flux_i)+mag_zero)
        img, removable = padmatch2(img, psf_i, yi, xi)

    corr = Table([x,y,flux,mags], names = ('field_x', 'field_y', 'flux', 'MAG'))

    return img,  corr

def write_out_sim(img, corr, file_path, file_name, psf_name, read_std, gain, dark_current, exp_time, bias_frame_num, dark_frame_num, star_num, mag_zero, low_count, high_count, bkg_comment):
    """
    write out a simulated image of sources which has had simulated noise added
    Args:
        img: image to write
        corr: correlation table which tracks the location and counts of generated
        file_path: path to directory
        file_name: file name to write to
        psf_name: psf file name used to produce the image
        read_std: standard deviation in electrons of the gaussian distribution the pixel values are being drawn from
        gain: gain of the simulated image in electron/ADU
        dark_current: dark current of the simulated image in electron/second
        exp_time: exposure time of the simulated image
        bias_frame_num: number of bias frames which are median combined to produce the master bias frame
        dark_frame_num: number of dark frames which are median combined to produce the master dark frame
        star_num: number of stars added to the image
        mag_zero: magnitude zero-point ot set for the image
        low_count: lower bound of source counts
        high_count: higher bound of source counts
        bkg_comment: comment describing the kind of background added to the image

    Returns: writes the image to a fits file and puts all relevant production parameters in header
            prints a comment indicating the file was written to drive
    """
    hdu = fits.PrimaryHDU(data=img)

    hdr = hdu.header
    hdr['GAIN'] = gain
    hdr['EXPTIME'] = exp_time
    hdr['BIASSTD'] = read_std
    hdr['BIASNUM'] = bias_frame_num
    hdr['DARKCURR'] = dark_current
    hdr['DARKNUM'] = dark_frame_num
    hdr['PSF'] = file_path + psf_name
    hdr['STARNUM'] = star_num
    hdr['hierarch COUNTLOW'] = low_count,
    hdr['hierarch COUNTHIGH'] = high_count
    hdr['MAGZERO'] = mag_zero
    hdr['BKG'] = bkg_comment

    table_hdu = fits.BinTableHDU(data=corr)

    hdu_list = fits.HDUList([hdu, table_hdu])

    hdu_list.writeto(file_path + file_name, overwrite=True)

    return 'File written to drive'
