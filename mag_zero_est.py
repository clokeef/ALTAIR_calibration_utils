#import astropy table
from astropy.table import Table

from JG_Streaktools import *


def build_corr_table(corr):
    """
    builds an Astropy table of the astrometry.net correlation file data
    Args:
        corr: correlation file from astrometry.net plate solution

    Returns: Astropy table of the correlation data
    """
    hd = corr[1].header
    data = corr[1].data

    #we populate an astropy table with our data
    data_rows = np.zeros(shape=(hd['NAXIS2'], hd['TFIELDS']))
    for i in range(len(data_rows)):
        for j in range(len(data_rows[0])):
            data_rows[i][j] = data[i][j]

    names = []
    for i in range(hd['TFIELDS']):
        names.append(hd['TTYPE' + f'{i + 1}'])

    table = Table(rows=data_rows, names=names)
    return table

def get_magzero_est(corr, sim, file_path, file_name, read_in = False):
    """
    estimates the magnitude zero-point for an image
    Args:
        corr: table or correlation file from astrometry.net
        sim: background subtracted streak interface instance of the associated iamge
        file_path: path to the working directory
        file_name: file name to write the output table
        read_in: boolean which indicates if the magzero estimate table should be read in and not recalculated

    Returns: correlation table with an added magzero column and the median magnitude zero-point
            this correlation table is then written out as a fits file with the median magnitude zero-point in the header, to be read in if needed
    """
    if read_in == True:
        table_fits  = fits.open(file_path + file_name)
        table = Table(table_fits[1].data)
        mag_zero = table['M0']
        mag_zero_est = table_fits[1].header['Mag 0']

    else:
        if isinstance(corr, Table):
            table = corr
        else:
            table = build_corr_table(corr)

        mag_zero = []
        for i in range(len(table)):
            # initialize a streak and set its centre to be the location from the table
            r1 = real_streak(sim)
            r1.y1 = table['field_x'][i]
            r1.x1 = table['field_y'][i]
            # length of 1 so the program does not crash
            r1.L = 1

            # tell streaktools the magnitude of the reference star
            r1.totalmag = table['MAG'][i]

            sim.simpill(r1.x1, r1.y1, 1.5 * sim.fwhm, r1.L, r1.theta, r1, visout=False)

            mag_zero.append(r1.pill_magzero[0])

        mag_zero_est = np.nanmedian(mag_zero)

        table.add_column(mag_zero, name = 'M0', index=-1)

        #we are going to write the table out as a new table with the M0 est in the header
        hdu = fits.PrimaryHDU()
        hdu.header = corr[0].header
        table_hdu = fits.BinTableHDU(data = table, header = corr[1].header)
        table_hdu.header['hierarch Mag 0'] = mag_zero_est

        hdulist = fits.HDUList([hdu, table_hdu])
        hdulist.writeto(file_path + file_name, overwrite=True)

    return mag_zero_est, mag_zero
