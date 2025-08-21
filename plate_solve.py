#file handling and astropy helpers
import astropy.io.fits as fits

#import astrometry api handler
from astrometry_net_client import Session, FileUpload

def plate_solve(api_key, file_path, file_name, wcs_name, corr_name, cal_name, read_in = False):
    """
    uploads a local file to astrometry.net to get a plate solution
    Args:
        api_key: string of the api key associated with your astrometry.net account
        file_path: path to the file directory
        file_name: file name of the image to be solved
        wcs_name: name of the wcs file
        corr_name: name of the correlation file
        cal_name: name of the calibration file
        read_in: boolean to indicate if the above files should be read in, no plate solve necessary

    Returns: if plate solution is successful, returns the solved job instance and writes out relevant files
            if plate solution is unsuccessful, prints a message to indicate solution failure
    """
    if read_in == True:
        wcs = read_wcs(file_path, wcs_name)
        corr = read_corr(file_path, corr_name)
        cal = read_calibration(file_path, cal_name)

        return wcs, corr, cal

    else:
        s = Session(api_key = api_key)
        upl = FileUpload(file_path+file_name, session=s)
        submission = upl.submit()
        submission.until_done()

        job = submission.jobs[0]
        job.until_done()

        if job.success():

            wcs = job.wcs_file()
            corr = job.corr_file()

            #this writes out the files to the working directory under the provided file names
            write_wcs(wcs, file_path, wcs_name)
            write_corr(corr, file_path, corr_name)
            write_calibration(job, file_path, cal_name)

            return job

        else:
            print('plate solve failed')

def write_wcs(wcs, file_path, file_out):
    """
    writes out the wcs file as a fits file
    Args:
        wcs: wcs instance from plate solution
        file_path: path to directory to be written to
        file_out: file name of the output fits file

    Returns: prints a message indicating the file has been written to drive
    """
    wcs_hdu = fits.PrimaryHDU(header = wcs)
    wcs_hdu.writeto(file_path + file_out, overwrite = True)
    print('WCS file written to drive')

def read_wcs(file_path, file_name):
    """
    read in a wcs fits file, where the wcs data is contained in the header
    Args:
        file_path: path to the file directory
        file_name: file name to be read in

    Returns: wcs data
    """
    wcs_file = fits.open(file_path + file_name)
    wcs = wcs_file[0].header

    return wcs

def write_corr(corr, file_path, file_out):
    """
    write out the correlation file as a fits table
    Args:
        corr: correlation file
        file_path: path to directory to be written to
        file_out: file name of the output fits file

    Returns: prints a message indicating the file has been written to drive
    """
    corr_hdu = fits.PrimaryHDU(header = corr[0].header)
    table_hdu = fits.BinTableHDU(data = corr[1].data, header = corr[1].header)
    hdul = fits.HDUList([corr_hdu, table_hdu])
    hdul.writeto(file_path + file_out, overwrite = True)
    print('Correlation file written to drive')

def read_corr(file_path, file_name):
    """
    read in a correlation fits file
    Args:
        file_path: path to the file directory
        file_name: file name to be read in

    Returns: corr data
    """
    corr = fits.open(file_path + file_name)

    return corr

def write_calibration(job, file_path, file_out):
    """
    write out the calibration data to a fits header
    Args:
        job: plate solution job instance
        file_path: path to the file directory
        file_out: file name to write to

    Returns: prints a message indicating the file has been written to drive
    """
    cal_hdu = fits.PrimaryHDU()

    cal_hdu.header['ra'] = job.info()['calibration']['ra']
    cal_hdu.header['dec'] = job.info()['calibration']['dec']
    cal_hdu.header['radius'] = job.info()['calibration']['radius']
    cal_hdu.header['pixscale'] = job.info()['calibration']['pixscale']
    cal_hdu.header['hierarch orientation'] = job.info()['calibration']['orientation']
    cal_hdu.header['parity'] = job.info()['calibration']['parity']

    cal_hdu.writeto(file_path+file_out, overwrite=True)
    print('Calibration file written to drive')

def read_calibration(file_path, file_out):
    """
    read in a calibration fits file
    Args:
        file_path: path to file direcotry
        file_out:  file name to read in

    Returns: header which contains the calibration data
    """
    cal = fits.open(file_path + file_out)
    return cal[0].header
