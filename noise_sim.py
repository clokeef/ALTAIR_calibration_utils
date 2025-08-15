from astropy.table import Table
from JG_Streaktools import *

#setup random number generator
rng = np.random.default_rng()


def read_noise(read_std, gain, img):
    """

    Args:
        read_std:
        gain:
        img:

    Returns:

    """
    size = np.shape(img)
    noise = np.random.normal(0, read_std / gain, size=size)

    return noise


def signal_noise(img, just_noise=True):
    """
    image should be in units of ADU or counts
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
    size = np.shape(img)
    imgs = []
    for i in range(bias_frame_num):
        img_noise = read_noise(read_std, gain, img)
        imgs.append(img_noise)

    r_noise = np.median(imgs, axis=0)

    return r_noise


def dark_noise(img, read_std, dark_current, gain, exp_time, dark_frame_num, bias_frame_num):
    #produce the noise of the master bias frame
    b_noise = bias_noise(img, read_std, gain, bias_frame_num)

    d_img = np.ones(shape=np.shape(img)) * dark_current * exp_time / gain

    d_imgs = []
    for i in range(dark_frame_num):
        # produce the readout noise for the darks
        r_img_noise = read_noise(read_std, gain, img)
        #produce the dark current noise
        d_img_noise = signal_noise(d_img)
        #comine the readout and dark noise with the noise associated to subtracting the master bias frame
        img_noise = r_img_noise + d_img_noise + b_noise
        d_imgs.append(img_noise)
    d_noise = np.median(d_imgs, axis=0)

    return d_noise


def img_noise(img, read_std, dark_current, gain, exp_time, dark_frame_num, bias_frame_num, just_noise=True):
    bkg_noise = signal_noise(img)

    d_img = np.ones(shape=np.shape(img)) * dark_current *exp_time /gain

    r_img_noise = read_noise(read_std, gain, img)
    d_img_noise = signal_noise(d_img)
    d_master_noise = dark_noise(img, read_std, dark_current, gain, exp_time, dark_frame_num, bias_frame_num)
    b_noise = bias_noise(img, read_std, gain, bias_frame_num)

    tot_bkg = bkg_noise + r_img_noise + b_noise + d_img_noise + d_master_noise

    if just_noise == True:

        return tot_bkg

    else:
        return img + tot_bkg


def gen_star_field(img, num, psf, mag0, low_count, high_count):
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
        mags.append(-2.5*np.log10(flux_i)+mag0)
        img, removable = padmatch2(img, psf_i, yi, xi)

    corr = Table([x,y,flux,mags], names = ('field_x', 'field_y', 'flux', 'MAG'))

    return img,  corr

def write_out_sim(img, corr, file_path, file_name, psf_name, gain, exp_time, read_std, bias_frame_num, dark_current, dark_frame_num, star_num, low_count, high_count, mag_zero, bkg_comment):
    """

    Args:
        img:
        corr:
        file_path:
        file_name:
        psf_name:
        gain:
        read_std:
        bias_frame_num:
        dark_current:
        dark_frame_num:
        star_num:
        low_count:
        high_count:
        mag_zero:
        bkg_comment:

    Returns:

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