#import photutils for background estimation and psf finding
from photutils.background import Background2D, MedianBackground

#import streak_tools
from JG_Streaktools import *


def background(data, bkg_scale, bkg_sub, verbose=True):
    """
    determine the background map, the median background for a fits image and the background noise map and median noise
    the returned image can be either background subtracted or not and the returned background values are assocaited with the returned image
    Args:
        data: data array of an image
        bkg_scale: scale of the background map
        bkg_sub: boolean to determine if the background map is subtracted off the image data
        verbose: boolean to determine if the background median and noise is printed out

    Returns: image array, median background count, background noise map, median background noise
    """
    # estimate the background map
    bkg_estimator = MedianBackground()
    sigma_clip = stats.SigmaClip(sigma=3.0)

    bkg = Background2D(data, (bkg_scale, bkg_scale), filter_size=(3, 3), sigma_clip=sigma_clip,
                       bkg_estimator=bkg_estimator)
    if bkg_sub == True:
        # subtract it off
        data_bkg = data - bkg.background

        # we also want to check that the median background level is now 0
        bkg2 = Background2D(data_bkg, (bkg_scale, bkg_scale), filter_size=(3, 3), sigma_clip=sigma_clip,
                        bkg_estimator=bkg_estimator)
        if verbose == True:
            print(f'The median background count of the initial image is {bkg.background_median}')
            print(f'The median background count of the subtracted image is {bkg2.background_median}')

            print(f'The rms noise of the initial image is {bkg.background_rms_median}')
            print(f'The rms noise of the subtracted image is {bkg2.background_rms_median}')

        return data_bkg, bkg2.background_median, bkg2.background_rms, bkg2.background_rms_median

    else:
        return data, bkg.background_median, bkg.background_rms, bkg.background_rms_median


def local_bkg(data, x, y, L, theta, bkg_scale):
    """
    determine the background in a small section around the relevant streak
    used for determining a background estimate for streak fitting
    Args:
        data: data array of an image
        x: start of the streak
        y: start of the streak
        L: length of the streak
        theta: angle of the streak clockwise from horizontal
        bkg_scale: scale of the background map

    Returns: the median background level in the region near the streak
    """
    # find x2, y2
    # handle the horizontal and vertical cases
    if round(np.sin(theta)) == 0:
        x2 = x + 20
        y2 = y + L*np.cos(theta)

    elif round(np.cos(theta)) == 0:
        x2 = x + L*np.sin(theta)
        y2 = y + 20

    else:
        x2 = x + L*np.sin(theta)
        y2 = y + L*np.cos(theta)

    x_scale = np.abs(x2-x)
    y_scale = np.abs(y2-y)

    x_low = int(x-2*x_scale)
    x_high = int(x2+2*x_scale)

    y_low = int(y-2*y_scale)
    y_high = int(y2+2*y_scale)

    # we need to ensure we don't try clip outside the data range
    if x_low < 0:
        x_low = 0

    if y_low < 0:
        y_low = 0

    if x_high > data.shape[0]:
        x_high = data.shape[0]
    if y_high > data.shape[1]:
        y_high = data.shape[1]

    data_clip = data[x_low:x_high, y_low:y_high]
    clip_bkg, clip_bkgmed, clip_bkgmap, clip_bkgnoise = background(data_clip, bkg_scale, bkg_sub=False)

    return clip_bkgmed, clip_bkgnoise

def sim_setup(file_path, file_name, bkg_scale, corr = True, bkg_sub=True, verbose=True):
    """
    initializes a streak_interface instance
    Args:
        file_path: path to the file directory
        file_name: file name
        bkg_scale: scale used for background estimation
        table: boolean to indicate if the fits file has an associated correlation table
        bkg_sub: boolean to indicate if the background is subtracted off during setup
        verbose: boolean to indicate if print statements are printed

    Returns:
        streak_interface instance with updated im_noise, VM properties, image header, median background, and the determined noise map
        if corr = True, the correlation table is also returned
    """
    #read in our image file
    f = fits.open(file_path + file_name)
    hd = f[0].header
    data = f[0].data

    if bkg_sub == True:
        data_bkg, bkg_med, noise_map, im_noise = background(data, bkg_scale, bkg_sub=bkg_sub, verbose=verbose)

        sim1 = streak_interface(data_bkg)
        sim1.VM = np.max(data_bkg)
        sim1.im_noise = im_noise

    else:
        data, bkg_med, noise_map, im_noise = background(data, bkg_scale, bkg_sub=bkg_sub, verbose=verbose)

        sim1 = streak_interface(data)
        sim1.VM = np.max(data)
        sim1.im_noise = im_noise

    if corr == True:
        corr = f

        return sim1, hd, corr, bkg_med, noise_map


    else:
        return sim1, hd, bkg_med, noise_map
