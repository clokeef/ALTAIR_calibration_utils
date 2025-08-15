#import the necessary packages

#file handling and astropy helpers
import astropy.io.fits as fits

#import astrometry api handler
from astrometry_net_client import Session, FileUpload


def plate_solve(api_key, file_path, file_name, wcs_name, corr_name, cal_name, read_in = False):
    """
    upload a local file to astrometry.net to be plate solved through api
    :param api_key: float of the api key associated with your astrometry.net account
    :param file_path: float of the path to your file directory
    :param file_name: float of the file name you want uploaded
    :return: if plate solution is successful, returns the solved job,
    if unsuccessful, prints a message indicating solution failure
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

            write_wcs(wcs, file_path, wcs_name)
            write_corr(corr, file_path, corr_name)
            write_calibration(job, file_path, cal_name)

            return job

        else:
            print('plate solve failed')




def write_wcs(wcs, file_path, file_out):
    wcs_hdu = fits.PrimaryHDU(header = wcs)
    wcs_hdu.writeto(file_path + file_out, overwrite = True)
    print('WCS file written to drive')

def read_wcs(file_path, file_name):
    wcs_file = fits.open(file_path + file_name)
    wcs = wcs_file[0].header

    return wcs

def write_corr(corr, file_path, file_out):
    corr_hdu = fits.PrimaryHDU(header = corr[0].header)
    table_hdu = fits.BinTableHDU(data = corr[1].data, header = corr[1].header)
    hdul = fits.HDUList([corr_hdu, table_hdu])
    hdul.writeto(file_path + file_out, overwrite = True)
    print('Correlation file written to drive')

def read_corr(file_path, file_name):
    corr = fits.open(file_path + file_name)

    return corr


def write_calibration(job, file_path, file_out):
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
    cal = fits.open(file_path + file_out)
    return cal[0].header