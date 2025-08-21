#import necessary package
import numpy as np
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

#import photutils for background estimation and psf finding
from photutils.utils import calc_total_error
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceCatalog
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder

from trippy import psf

from calibration_utils.utils import downsample

def photutils_cat(sim, bkg_noise_map, gain, threshold_factor, conv_fwhm, conv_size, source_size):
    """
    determine a catalog of sources in an image using the photutils package
    Args:
        sim: streak interface instance of an image
        bkg_noise_map: noise map associated with the image
        gain: gain of the image
        threshold_factor: snr threshold above which stars will be exptracted
        conv_fwhm: fwhm of the gaussian filter
        conv_size: size of the gaussian filter
        source_size: size of the extracted source

    Returns: catalog of extracted sources in the image
    """
    data = np.array(sim.current_image, dtype='float') #the passed data should be background subtracted already
    error_map = calc_total_error(data, bkg_noise_map, effective_gain = gain)

    #set the threshold map
    threshold = threshold_factor*bkg_noise_map

    kernel = make_2dgaussian_kernel(conv_fwhm, conv_size)
    convolved_data = convolve(data, kernel)

    segment_map = detect_sources(convolved_data, threshold, npixels=source_size)
    segm_deblend = deblend_sources(convolved_data, segment_map, npixels = source_size,
                                   nlevels=32, contrast=0.001, progress_bar=False)

    #this will produce a catalog of sources, we are interested in their locations and flux, flux_err
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data, error = error_map)

    return cat

def catalog_trim(catalog, sim, source_size, snr_thresh, initalpha, initbeta,
                 bgRadius = 20.0, repFact = 3, ftol= 1.49012e-8,
                 crowd_trim = True, snr_trim = True, shape_trim = True):
    """
    trim the catalog based on set parameters:
    crowd_trim: trim out extracted sources which have sources in the extracted region
    snr_trim: trim out extracted stars which are below an snr threshold
    shape_trim: trim out sources whose shape is not consistent with the median source shape
    based on the catalog trimming from TRIPPy: Python based Trailed Source Photometry
    uses psf profiles and fitting from TRIPPy
    Args:
        catalog: catalog of extracted stars produced by photutils_psf
        sim: streak interface instance of the image
        source_size: size of the extracted sources
        snr_thresh: snr threshold for trimming
        initalpha: initial alpha guess for fitting the sources to moffat profile
        initbeta: initial beta guess for fitting the sources to moffat profiles
        bgRadius: radius from source centre at which the background level is measured
        repFact: subsampling factor for the moffat profile fitting
        ftol: fitting tolerance, increasing this number will lead to faster less accurate fitting
        crowd_trim: boolean which indicates if the ctatlog is trimmed for crowding
        snr_trim: boolean which indicates if the catalog is trimmed for snr
        shape_trim: boolean which indicates if the catalog is trimmed for shape

    Returns: the median fwhm of the trimmed catalog, boolean array which indicates good sources,
            array of parameters of the extracted sources
    """
    #make a list to keep track of which are the good stars
    goodStars = []
    for i in range(len(catalog)):
        goodStars.append(True)

    goodStars = np.array(goodStars)

    #trim for sources too close to the edge of the frame
    hsize = (source_size - 1) / 2
    for i in range(len(goodStars)):
        x = catalog.xcentroid[i]
        y = catalog.ycentroid[i]
        data = sim.current_image
        if x < hsize or x > data.shape[1] - hsize - 1 or y < hsize or y > data.shape[0] - hsize - 1:
            goodStars[i] = False

    #trim for snr
    if snr_trim == True:
        for i in range(len(goodStars)):
            if goodStars[i] == False:
                continue
            snr = catalog.segment_flux[i]/catalog.segment_fluxerr[i]
            if snr < snr_thresh:
                goodStars[i] = False

    #trim for crowding in the catalog
    if crowd_trim == True:
        for i in range(len(goodStars)):
            #if already ruled out don't check again
            if goodStars[i] == False:
                continue

            dist = ((catalog.xcentroid - catalog.xcentroid[i]) ** 2 + (
                    catalog.ycentroid - catalog.ycentroid[i]) ** 2) ** 0.5
            args = np.argsort(dist)
            dist = dist[args]
            if dist[1] < (2 * (source_size ** 2)) ** 0.5:
                goodStars[i] = False

    #fit the sources to a moffat profile
    points = []

    for i in range(len(goodStars)):
        if goodStars[i] == False:
            continue

        mpsf = psf.modelPSF(np.arange(source_size), np.arange(source_size), alpha=initalpha, beta=initbeta,
                            repFact=repFact)
        mpsf.fitMoffat(data, catalog.xcentroid[i], catalog.ycentroid[i], boxSize=hsize,
                       verbose=False, bgRadius=bgRadius, ftol=ftol, quickFit=True)
        fwhm = mpsf.FWHM(fromMoffatProfile=True)

        points.append([fwhm, mpsf.chi, mpsf.alpha, mpsf.beta, catalog.xcentroid[i], catalog.ycentroid[i], mpsf.bg])

    points = np.array(points, dtype='float')

    fwhm = sigma_clipped_stats(points[:,0])[0]

    #trim for similar shape
    if shape_trim == True:
        #to do trimming by the shape, we need to fit our sources to moffat profiles
        #we'll them compare how similar their fit parameters are

        mean_a, median_a, std_a = sigma_clipped_stats(points[:, 2])
        mean_b, median_b, std_b = sigma_clipped_stats(points[:, 3])
        w = np.where( (np.abs(points[:,2]-mean_a)>2.0*std_a) | (np.abs(points[:,3]-mean_b)>2.0*std_b) )
        goodStars[w] = False

    return fwhm, goodStars, points

def extract_star_frames(catalog, good_ind, sim, source_size):
    """
    extracts the star frames of some size associated with a catalog and boolean array which indicates which sources are extracted
    Args:
        catalog: catalog of sources in the image
        good_ind: boolean array of good sources produced by catalog trim
        sim: streak interface instance from which the sources were identified
        source_size: size of the extracted frame around the source

    Returns: returns a 'catalog', which is a photutils EPSFStars instance, of the extracted source frames
    """
    good_source = catalog[good_ind]

    x = good_source.xcentroid
    y = good_source.ycentroid
    stars_tbl = Table()
    stars_tbl['x'] = x
    stars_tbl['y'] = y


    nddata = NDData(data=sim.current_image)
    stars = extract_stars(nddata, stars_tbl, size=source_size)

    return stars

def eff_psf(stars, oversampfac):
    """
    from a catalog of extracted source frames, construct an effect psf
    Args:
        stars: EPSFStars catalog of source frames
        oversampfac: oversampling factor

    Returns: a normalized median epsf for the catalog
    """
    epsf_builder = EPSFBuilder(oversampling=oversampfac, maxiters=3, progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)

    psf = downsample(epsf.data, factor=oversampfac, normalization='mean')
    psf = psf / np.sum(psf)

    return psf

def epsf_write_out(epsf, fwhm, sim_hd, file_path, psf_name):
    """
    write out the effective psf, saving the median fwhm in the header
    Args:
        epsf: 2d array of the normalized effective psf
        fwhm: full width half maximum of the profile
        sim_hd: header of the file from which the epsf was rpoduced
        file_path: file path to the directory to write to
        psf_name: file name to write the psf to

    Returns: prints a message to state the file hs been written out
    """
    sim_hd.append(('FWHM', fwhm))
    hdu = fits.PrimaryHDU(epsf, sim_hd)
    hdu.writeto(file_path + psf_name, overwrite=True)

    return 'File written to drive'

def epsf_read_in(sim, file_path, psf_name):
    """
    reads in the psf and assigns the fwhm and psf to an active streak interface instance
    Args:
        sim: streak interface instance
        file_path: path to the directory which contains the psf
        psf_name: psf file name

    Returns: assigns the fwhm and psf to the streak interface,
            prints a message to indicate successful read in
    """
    psf_file = fits.open(file_path + psf_name)

    psf_hd = psf_file[0].header
    psf_data = psf_file[0].data

    sim.psf = psf_data
    sim.fwhm = psf_hd['FWHM']

    return 'PSF written in'
