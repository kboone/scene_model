#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
3D PSF-based point-source extractor. There are several PSF choices. The default
PSF is a exponential profile in Fourier space convolved with an instrumental
PSF and a model of the tracking.
"""

__author__ = "K. Boone, Y. Copin, C. Buton, E. Pecontal"
__version__ = '$Id: extract_star2.py,v 1.0 2018/07/23 13:17:00 kboone Exp $'

import os

import pyfits as F
import numpy as np

import pySNIFS
import libExtractStar as libES
import ToolBox.Atmosphere as TA
from ToolBox.Arrays import metaslice
from ToolBox.Misc import warning2stdout

import scene_model
from scene_model import snifs as snifs_scene

import warnings
warnings.showwarning = warning2stdout   # Redirect warnings to stdout
warnings.filterwarnings("ignore", "Overwriting existing file")

# Numpy setup
np.set_printoptions(linewidth=999)       # X-wide lines

reference_wavelength = scene_model.config.reference_wavelength

# MLA tilt: MLA vertical is rotated by ~5° wrt. north: theta_MLA =
# theta_DTCS + 5°
DTHETA = {'B': 2.6, 'R': 5.0}             # [deg]

MAX_POSITION_PRIOR_OFFSET = 1    # Max position offset wrt. prior [spx]
MAX_SEEING_PRIOR_OFFSET = 40     # Max seeing offset wrt. prior [%]
MAX_AIRMASS_PRIOR_OFFSET = 20    # Max airmass offset wrt. prior [%]
MAX_PARANG_PRIOR_OFFSET = 20     # Max parangle offset wrt. prior [deg]
MAX_POSITION = 6                 # Max position wrt. FoV center [spx]

MIN_SEEING = 0.3                 # Min reasonable seeing ['']
MAX_SEEING = 4.0                 # Max reasonable seeing ['']
MAX_AIRMASS = 4.                 # Max reasonable airmass
MIN_ELLIPTICITY = 0.2            # Min reasonable ellipticity
MAX_ELLIPTICITY = 5.0            # Max reasonable ellipticity


# Definitions ================================================================


def print_msg(message, limit):
    """Print message 'str' if verbosity level (opts.verbosity) >= limit."""

    if opts.verbosity >= limit:
        print(message)


def extract_key(item_list, key):
    """Extract the value a key from each item in a list.

    This returns a numpy array of the corresponding values. If an item is None,
    then np.nan is returned for that item's value.
    """
    result = []
    for item in item_list:
        if item is None:
            result.append(np.nan)
        else:
            result.append(item[key])
    result = np.array(result)
    return result


def write_fits(spectrum, filename=None, header=None):
    """
    Overrides pySNIFS_fit.spectrum.WR_fits_file. Allows full header
    propagation (including comments) and covariance matrix storage.
    """

    assert not (spectrum.start is None or spectrum.step is None or
                spectrum.data is None)

    # Primary HDU: signal
    hdusig = F.PrimaryHDU(spectrum.data, header=header)
    for key in ['EXTNAME', 'CTYPES', 'CRVALS', 'CDELTS', 'CRPIXS']:
        if key in hdusig.header:
            del(hdusig.header[key])  # Remove technical keys from E3D cube
    hdusig.header.set('CRVAL1', spectrum.start, after='NAXIS1')
    hdusig.header.set('CDELT1', spectrum.step, after='CRVAL1')

    hduList = F.HDUList([hdusig])

    # 1st extension 'VARIANCE': variance
    if spectrum.has_var:
        hduvar = F.ImageHDU(spectrum.var, name='VARIANCE')
        hduvar.header['CRVAL1'] = spectrum.start
        hduvar.header['CDELT1'] = spectrum.step

        hduList.append(hduvar)

    # 2nd (compressed) extension 'COVAR': covariance (lower triangle)
    if hasattr(spectrum, 'cov'):
        hducov = F.ImageHDU(np.tril(spectrum.cov), name='COVAR')
        hducov.header['CRVAL1'] = spectrum.start
        hducov.header['CDELT1'] = spectrum.step
        hducov.header['CRVAL2'] = spectrum.start
        hducov.header['CDELT2'] = spectrum.step

        hduList.append(hducov)

    if filename:                        # Save hduList to disk
        hduList.writeto(filename, output_verify='silentfix', clobber=True)

    return hduList                      # For further handling if needed


def create_2Dlog(opts, cube, params, dparams, chi2):
    """Dump an informative text log about the PSF (metaslice) 2D-fit."""

    logfile = file(opts.log2D, 'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    if opts.background_degree == -2:       # Step background
        npar_sky = 2
    else:                       # Polynomial background (or none)
        npar_sky = (opts.background_degree + 1) * (opts.background_degree + 2) / 2

    delta, theta = params[:2]
    xc, yc = params[2:4]
    xy, ell, alpha = params[4:7]
    intensity = params[-npar_sky - 1]
    sky = params[-npar_sky:]

    names = ['delta', 'theta', 'x0', 'y0', 'xy', 'ell', 'alpha', 'I'] + \
            ['sky%d' % d for d in xrange(npar_sky)]
    labels = '# lbda  ' + '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"] * ((8 + npar_sky) * 2 + 1)) + '\n'

    for n in xrange(cube.nslice):
        list2D = [cube.lbda[n],
                  delta[n], dparams[n][0],
                  theta[n], dparams[n][1],
                  xc[n], dparams[n][2],
                  yc[n], dparams[n][3],
                  xy[n], dparams[n][4],
                  ell[n], dparams[n][5],
                  alpha[n], dparams[n][6],
                  intensity[n], dparams[n][-npar_sky - 1]]
        if npar_sky:
            tmp = np.transpose((sky[:, n], dparams[n][-npar_sky:]))
            list2D += tmp.flatten().tolist()
        list2D += [chi2[n]]
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def create_3Dlog(opts, cube, cube_fit, fitpar, dfitpar, chi2):
    """Dump an informative text log about the PSF (full-cube) 3D-fit."""

    logfile = file(opts.log3D, 'w')

    logfile.write('# cube    : %s   \n' % os.path.basename(opts.input))
    logfile.write('# object  : %s   \n' % cube.e3d_data_header["OBJECT"])
    logfile.write('# airmass : %.3f \n' % cube.e3d_data_header["AIRMASS"])
    logfile.write('# efftime : %.3f \n' % cube.e3d_data_header["EFFTIME"])

    # Global parameters
    # lmin  lmax  delta +/- ddelta  ...  alphaN +/- dalphaN chi2|RSS
    names = ['delta', 'theta', 'xc', 'yc', 'xy'] + \
            ['ell%d' % d for d in xrange(ellDeg + 1)] + \
            ['alpha%d' % d for d in xrange(alphaDeg + 1)]
    labels = '# lmin  lmax' + \
        '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: Residual Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: true chi2
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  %6.0f  ' + \
          '  '.join(
              ["%10.4g"] * ((5 + (ellDeg + 1) + (alphaDeg + 1)) * 2 + 1)) + '\n'
    list3D = [cube.lstart, cube.lend,
              fitpar[0], dfitpar[0],
              fitpar[1], dfitpar[1],
              fitpar[2], dfitpar[2],
              fitpar[3], dfitpar[3],
              fitpar[4], dfitpar[4]]
    for i in xrange(ellDeg + 1):   # Ellipticity coefficiens
        list3D += [fitpar[5 + i], dfitpar[5 + i]]
    for i in xrange(alphaDeg + 1):  # Alpha coefficients
        list3D += [fitpar[6 + ellDeg + i], dfitpar[6 + ellDeg + i]]
    list3D += [chi2]             # chi2|RSS
    logfile.write(fmt % tuple(list3D))

    # Metaslice parameters
    # lbda  I -/- dI  sky0 +/- dsky0  sky1 +/- dsky1  ...  chi2|RSS
    npar_psf = 7 + ellDeg + alphaDeg
    if opts.background_degree == -2:       # Step background
        npar_sky = 2
    else:                       # Polynomial background (or none)
        npar_sky = (opts.background_degree + 1) * (opts.background_degree + 2) / 2

    names = ['I'] + ['sky%d' % d for d in range(npar_sky)]
    labels = '# lbda  ' + '  '.join('%8s +/- d%-8s' % (n, n) for n in names)
    if cube.var is None:        # Least-square fit: compute Res. Sum of Squares
        labels += '        RSS\n'
    else:                       # Chi2 fit: compute chi2 per slice
        labels += '        chi2\n'
    logfile.write(labels)
    fmt = '%6.0f  ' + '  '.join(["%10.4g"] * ((1 + npar_sky) * 2 + 1)) + '\n'
    for n in xrange(cube.nslice):       # Loop over metaslices
        # Wavelength, intensity and error on intensity
        list2D = [cube.lbda[n], fitpar[npar_psf + n], dfitpar[npar_psf + n]]
        for i in xrange(npar_sky):  # Add background parameters
            list2D.extend([fitpar[npar_psf + cube.nslice + n * npar_sky + i],
                           dfitpar[npar_psf + cube.nslice + n * npar_sky + i]])
        # Compute chi2|RSS
        chi2 = np.nan_to_num((cube.slice2d(n, coord='p') -
                             cube_fit.slice2d(n, coord='p')) ** 2)
        if cube.var is not None:    # chi2: divide by variance
            chi2 /= cube.slice2d(n, coord='p', var=True)
        list2D += [chi2.sum()]      # Slice chi2|RSS
        logfile.write(fmt % tuple(list2D))

    logfile.close()


def fill_header(hdr, psf, adr, cube, opts, chi2, seeing, posprior, fluxes):
    """Fill header *hdr* with PSF fit-related keywords."""

    # Convert reference position from lmid = (lmin+lmax)/2 to reference
    # wavelength
    lmin, lmax = cube.lstart, cube.lend   # 1st and last meta-slice wavelength
    xref, yref = adr.refract(psf['ref_center_x'], psf['ref_center_y'],
                             reference_wavelength, unit=cube.spxSize)
    print_msg("Ref. position [%.0f A]: %+.2f x %+.2f spx" %
              (reference_wavelength, xref, yref), 0)

    # "[short|long], classic[-powerlaw]" or "[long|short] [blue|red],
    # chromatic"
    # psfname = ', '.join((psf.name, psf.model))
    psfname = 'TODO'

    psf_items = psf.get_fits_header_items()

    for key, value, description in psf_items:
        hdr[key] = (value, description)

    hdr['ES_VERS'] = __version__
    hdr['ES_CUBE'] = (opts.input, 'Input cube')
    hdr['ES_LREF'] = (reference_wavelength, 'Lambda ref. [A]')
    hdr['ES_CHI2'] = (chi2, 'Chi2|RSS of 3D-fit')
    hdr['ES_AIRM'] = (adr.get_airmass(), 'Effective airmass')
    hdr['ES_PARAN'] = (adr.get_parangle(), 'Effective parangle [deg]')
    hdr['ES_LMIN'] = (lmin, 'Meta-slices minimum lambda')
    hdr['ES_LMAX'] = (lmax, 'Meta-slices maximum lambda')

    hdr['ES_METH'] = (opts.method, 'Extraction method')
    hdr['ES_PSF'] = (psfname, 'PSF model name')
    hdr['ES_SUB'] = (psf.subsampling, 'PSF subsampling')
    if opts.method.endswith('aperture'):
        hdr['ES_APRAD'] = (opts.radius, 'Aperture radius [" or sigma]')

    tflux, sflux = fluxes       # Total point-source and sky fluxes
    hdr['ES_TFLUX'] = (tflux, 'Total point-source flux')
    if sflux:
        hdr['ES_SFLUX'] = (sflux, 'Total sky flux/arcsec^2')

    hdr['SEEING'] = (seeing, 'Estimated seeing @lbdaRef ["] (extract_star2)')

    if opts.usePriors:
        hdr['ES_PRIOR'] = (opts.usePriors, 'PSF prior hyper-scale')
        if opts.seeingPrior is not None:
            hdr['ES_PRISE'] = (opts.seeingPrior, 'Seeing prior [arcsec]')
    if opts.psf3Dconstraints:
        for i, constraint in enumerate(opts.psf3Dconstraints):
            hdr['ES_BND%d' % (i + 1)] = (constraint, "Constraint on 3D-PSF")


# ########## MAIN ##############################

if __name__ == "__main__":

    import optparse

    # Options ================================================================

    usage = "[%prog] [options] incube.fits"

    parser = optparse.OptionParser(usage, version=__version__)

    parser.add_option("-i", "--in", type=str, dest="input",
                      help="Input datacube (or use argument)")
    parser.add_option("-o", "--out", type=str,
                      help="Output point source spectrum")
    parser.add_option("-s", "--sky", type=str,
                      help="Output sky spectrum")

    # PSF parameters
    parser.add_option("-S", "--skyDeg", type=int,
                      help="Sky polynomial background degree "
                      "(-1: none) [%default]",
                      default=0)

    # PSF model
    parser.add_option("--psf", choices=('old', 'fourier'), help="PSF model "
                      "(old|fourier) [%default]", default='fourier')

    # Extraction method and parameters
    parser.add_option("-N", "--nmeta", type=int,
                      help="Number of chromatic meta-slices [%default]",
                      default=12)
    parser.add_option("--subsampling", type=int,
                      help="Spaxel subsampling [%default]",
                      default=scene_model.config.default_subsampling)
    parser.add_option("--border", type=int,
                      help="Border used for model evaluation [%default]",
                      default=scene_model.config.default_border)

    parser.add_option("-m", "--method",
                      choices=('psf', 'optimal', 'aperture', 'subaperture'),
                      help="Extraction method "
                      "(psf|optimal|[sub]aperture) [%default]",
                      default="psf")
    parser.add_option("-r", "--radius", type=float,
                      help="Aperture radius for non-PSF extraction "
                           "(>0: in \", <0: in seeing sigma) [%default]",
                      default=-5.)
    parser.add_option("-L", "--leastSquares",
                      dest="chi2fit", action="store_false",
                      help="Least-square fit [default is a chi2 fit]",
                      default=True)

    # Plotting
    parser.add_option("-g", "--graph",
                      choices=('png', 'eps', 'pdf', 'svg', 'pylab'),
                      help="Graphic output format (png,eps,pdf,svg,pylab)")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (='-g pylab')")

    # Covariance management
    parser.add_option("-V", "--covariance", action='store_true',
                      help="Compute and store covariance matrix in extension")

    # Priors
    parser.add_option("--usePriors", type=float,
                      help="PSF prior hyper-scale, or 0 for none "
                      "(req. powerlaw-PSF) [%default]",
                      default=0.)
    parser.add_option("--seeingPrior", type=float,
                      help="Seeing prior (from Exposure.Seeing) [\"]")

    # Expert options
    parser.add_option("--no3Dfit", action='store_true',
                      help="Do not perform final 3D-fit")
    parser.add_option("--keepmodel", action='store_true',
                      help="Store meta-slices and adjusted model in 3D cubes")
    parser.add_option("--psf3Dconstraints", type=str, action='append',
                      help="Constraints on PSF parameters (n:val,[val])")

    # Debug options
    parser.add_option("-v", "--verbosity", type=int,
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    parser.add_option("-f", "--file", type=str, dest="log2D",
                      help="2D adjustment logfile name")
    parser.add_option("-F", "--File", type=str, dest="log3D",
                      help="3D adjustment logfile name")
    parser.add_option("--ignorePertinenceTests", action='store_true',
                      # help=optparse.SUPPRESS_HELP
                      help="Ignore tests on PSF pertinence (but DON'T!)")

    # Production options
    parser.add_option("--accountant",
                      help="Accountant output YAML file")

    opts, args = parser.parse_args()
    if not opts.input:
        if args:
            opts.input = args[0]
        else:
            parser.error("No input datacube specified.")

    if opts.graph:
        opts.plot = True
    elif opts.plot:
        opts.graph = 'pylab'

    if opts.skyDeg < -2:
        opts.skyDeg = -1        # No sky background
        if opts.sky:
            print("WARNING: Cannot extract sky spectrum in no-sky mode.")

    if opts.verbosity <= 0:
        np.seterr(all='ignore')

    if opts.usePriors:
        if opts.usePriors < 0:
            parser.error("Prior scale (--usePriors) must be positive.")

    if opts.seeingPrior and not opts.usePriors:
        parser.error("Seeing prior requires prior usage (--usePriors > 0).")

    # Input datacube ==========================================================

    print("Opening datacube %s" % opts.input)

    # The pySNIFS e3d_data_header dictionary is not enough for later
    # updates in fill_header, which requires a *true* pyfits header.
    try:
        try:                                    # Try to read a Euro3D cube
            inhdr = F.getheader(opts.input, 1)  # 1st extension
            full_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input)
            isE3D = True
        except ValueError:                      # Try to read a 3D FITS cube
            inhdr = F.getheader(opts.input, 0)  # Primary extension
            full_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input)
            isE3D = False
    except IOError:
        parser.error("Cannot access file '%s'" % opts.input)
    step = full_cube.lstep

    print_msg("Cube %s [%s]: %d slices [%.2f-%.2f], %d spaxels" %
              (os.path.basename(opts.input), 'E3D' if isE3D else '3D',
               full_cube.nslice,
               full_cube.lbda[0], full_cube.lbda[-1], full_cube.nlens), 1)

    objname = inhdr.get('OBJECT', 'Unknown')
    efftime = inhdr['EFFTIME']            # [s]
    airmass = inhdr['AIRMASS']
    try:
        parangle = inhdr['PARANG']        # Sky parallactic angle [deg]
    except KeyError:                      # Not in original headers
        print("WARNING: Computing PARANG from header ALTITUDE, AZIMUTH and "
              "LATITUDE.")
        # [deg]
        _, inhdr['PARANG'] = snifs_scene.estimate_zenithal_parallactic(inhdr)

    channel = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
    # Include validity tests and defaults
    pressure, temp = snifs_scene.read_pressure_and_temperature(inhdr)

    background_degree = opts.skyDeg
    hasSky = background_degree != -1       # Sky component

    # Test channel and set default output name
    if channel not in ('B', 'R'):
        parser.error(
            "Input datacube %s has no valid CHANNEL keyword (%s)" %
            (opts.input, channel))
    if not opts.out:                    # Default output
        opts.out = 'spec_%s.fits' % (channel)

    # Select the PSF
    scene_model_parameters = {
        'subsampling': opts.subsampling,
        'border': opts.border,
    }
    if opts.psf == 'old':
        scene_model_class = snifs_scene.SnifsOldSceneModel
        seeing_degree = 2
        scene_model_parameters['exposure_time'] = efftime
        scene_model_parameters['alpha_degree'] = seeing_degree
        seeing_key = 'alpha'
        seeing_powerlaw_keys = ['A%d' % i for i in range(seeing_degree + 1)]
        global_fit_keys = ['ell', 'xy']
    elif opts.psf == 'fourier':
        scene_model_class = snifs_scene.SnifsFourierSceneModel
        scene_model_parameters['background_degree'] = background_degree
        seeing_degree = 1
        seeing_key = 'seeing_width'
        seeing_powerlaw_keys = ['seeing_ref_power', 'seeing_ref_width']
        global_fit_keys = ['ell_coord_x', 'ell_coord_y']
    else:
        parser.error("Invalid PSF model '%s'" % opts.psf)

    print("  Object: %s, Efftime: %.1fs, Airmass: %.2f" %
          (objname, efftime, airmass))
    print("TODO: print PSF info")

    # print("  PSF: '%s', sub-sampled x%d" % \
    # (', '.join((psfFn.model, psfFn.name)), psfFn.subsampling))

    if background_degree > 0:
        print("  Sky: polynomial, degree %d" % background_degree)
    elif background_degree == 0:
        print("  Sky: uniform")
    elif background_degree == -1:
        print("  Sky: none")
    else:
        parser.error("Invalid sky degree '%d'" % background_degree)

    # Accounting
    if opts.accountant:
        try:
            from libRecord import Accountant
        except ImportError:
            print("WARNING: libRecord is not accessible, accounting disabled")
        else:
            import atexit

            accountant = Accountant(opts.accountant, opts.out)
            print(accountant)
            atexit.register(accountant.finalize)
    else:
        accountant = None

    # 2D-model fitting ========================================================

    # Meta-slice definition (min,max,step [px]) ------------------------------

    slices = metaslice(full_cube.nslice, opts.nmeta, trim=10)
    print("  Channel: '%s', extracting slices: %s" % (channel, slices))

    if isE3D:
        meta_cube = pySNIFS.SNIFS_cube(e3d_file=opts.input, slices=slices)
    else:
        meta_cube = pySNIFS.SNIFS_cube(fits3d_file=opts.input, slices=slices)
    meta_cube.flag_nans(name='meta-cube')
    meta_cube.x = meta_cube.i - 7       # From I,J to spx coords
    meta_cube.y = meta_cube.j - 7
    spxSize = meta_cube.spxSize
    nmeta = meta_cube.nslice

    print_msg("  Meta-slices before selection: %d from %.2f to %.2f by %.2f A"
              % (nmeta, meta_cube.lstart, meta_cube.lend, meta_cube.lstep), 0)

    if opts.keepmodel:                  # Store meta-slices in 3D-cube
        path, name = os.path.split(opts.out)
        outpsf = os.path.join(path, 'meta_' + name)
        print("Saving meta-slices in 3D-fits cube '%s'..." % outpsf)
        meta_cube.WR_3d_fits(outpsf)

    # Initial ADR model
    adr = snifs_scene.build_adr_model_from_header(inhdr)
    print_msg('  ' + str(adr), 1)

    # 2D-fit ------------------------------

    print("Meta-slice 2D-fitting (%s)..." %
          ('chi2' if opts.chi2fit else 'least-squares'))

    meta_cube_data = np.zeros((meta_cube.data.shape[0], 15, 15))
    meta_cube_var = np.zeros(meta_cube_data.shape)
    meta_cube_var[...] = np.nan

    meta_cube_data[:, meta_cube.i, meta_cube.j] = meta_cube.data
    meta_cube_var[:, meta_cube.i, meta_cube.j] = meta_cube.var

    # Fit positions with Gaussian psf models, and then fit the full PSF to each
    # image individually.
    valid = []
    gaussian_psf_models = []
    individual_psf_models = []
    individual_uncertainties = []

    for idx in range(len(meta_cube_data)):
        try:
            # Gaussian fit for initial position.
            gaussian_model = scene_model.GaussianSceneModel(
                meta_cube_data[idx], meta_cube_var[idx], subsampling=1,
                border=0
            )
            gaussian_model.fix(rho=0.)
            gaussian_model.fit()

            initial_position_x = gaussian_model['center_x']
            initial_position_y = gaussian_model['center_y']

            # Fit the full PSF model to each image individually, starting at
            # the Gaussian fit location.
            full_psf_model = scene_model_class(
                meta_cube_data[idx], meta_cube_var[idx],
                **scene_model_parameters
            )
            full_psf_model.set_parameters(
                center_x=initial_position_x,
                center_y=initial_position_y,
            )

            full_psf_model.fit()

            uncertainties = full_psf_model.calculate_uncertainties()

            if opts.verbosity >= 2:
                print("")
                full_psf_model.print_fit_info("Fit to metaslice %d" % idx,
                                              uncertainties=uncertainties)
                print("")

            gaussian_psf_models.append(gaussian_model)
            individual_psf_models.append(full_psf_model)
            individual_uncertainties.append(uncertainties)
            valid.append(True)
        except scene_model.PsfModelException as e:
            print("Fit on slice %d failed with error %s" % (idx, e))
            gaussian_psf_models.append(None)
            individual_psf_models.append(None)
            individual_uncertainties.append(None)
            valid.append(False)
            raise

    valid = np.array(valid)
    if not np.all(valid):
        print("WARNING: %d metaslices discarded due to invalid fits" % \
            np.sum(~valid))

    # Guess the reference positions for the ADR using the individual fits.
    xc_vec = extract_key(individual_psf_models, 'center_x')
    yc_vec = extract_key(individual_psf_models, 'center_y')

    xc_err = extract_key(individual_uncertainties, 'center_x')
    yc_err = extract_key(individual_uncertainties, 'center_y')

    # Back-propagate positions to lmid wavelength, and take the median value.
    valid_xmids, valid_ymids = adr.refract(
        xc_vec[valid], yc_vec[valid], meta_cube.lbda[valid], backward=True,
        unit=spxSize
    )
    xmids = np.empty(nmeta)
    ymids = np.empty(nmeta)
    xmids.fill(np.nan)
    ymids.fill(np.nan)
    xmids[valid] = valid_xmids
    ymids[valid] = valid_ymids

    xmid = np.median(valid_xmids)
    ymid = np.median(valid_ymids)

    # Cut out any images that didn't find PSFs at the right position.
    r = np.hypot(valid_xmids - xmid, valid_ymids - ymid)
    rmax = 4.4478 * np.median(r)     # Robust to outliers 3*1.4826
    good = np.zeros(nmeta, dtype=bool)
    good[valid] = (r <= rmax)           # Valid fit and reasonable position
    bad = np.zeros(nmeta, dtype=bool)
    bad[valid] = (r > rmax)             # Valid fit but discarded position
    if bad.any():
        print("WARNING: %d metaslices discarded after ADR selection" % \
              (len(np.nonzero(bad))))

    # Estimate the seeing coefficients
    individual_seeing_widths = extract_key(individual_psf_models, seeing_key)
    individual_seeing_errs = extract_key(individual_uncertainties, seeing_key)
    seeing_powerlaw_guesses = libES.powerLawFit(
        meta_cube.lbda[good] / reference_wavelength,
        individual_seeing_widths[good],
        seeing_degree
    )

    # Estimate the positions. Here we used a clipped mean rather than a median,
    # following what extract_star did.
    if good.any():
        print_msg("%d/%d centroids found within %.2f spx of (%.2f,%.2f)" %
                  (len(xmids[good]), len(xmids), rmax, xmid, ymid), 1)
        xc, yc = xmids[good].mean(), ymids[good].mean()  # Position at lmid
    else:
        raise ValueError('No position initial guess')

    print("3D fit...")
    fitter = scene_model.MultipleImageFitter(
        scene_model_class(wavelength_dependence=True, adr_model=adr,
                          spaxel_size=spxSize, **scene_model_parameters),
        meta_cube_data, meta_cube_var,
    )
    fitter.fix(wavelength=meta_cube.lbda)

    # Set initial guesses for the seeing
    for key, value in zip(seeing_powerlaw_keys, seeing_powerlaw_guesses):
        fitter.add_global_fit_parameter(key, value)
    guess_widths = libES.powerLawEval(
        seeing_powerlaw_guesses, meta_cube.lbda / reference_wavelength
    )

    # Do global fits for all other PSF parameters
    for key in global_fit_keys:
        fitter.add_global_fit_parameter(key)

    # ADR parameters
    delta0 = adr.delta           # ADR power = tan(zenithal distance)
    theta0 = adr.theta           # ADR angle = parallactic angle [rad]
    print_msg("  ADR guess: delta=%.2f (airmass=%.2f), theta=%.1f deg" %
              (delta0, adr.get_airmass(), theta0 * TA.RAD2DEG), 1)

    fitter.add_global_fit_parameter('ref_center_x', xc)
    fitter.add_global_fit_parameter('ref_center_y', yc)

    fitter.add_global_fit_parameter('adr_delta', delta0)
    fitter.add_global_fit_parameter('adr_theta', theta0)

    # Set up the ADR model
    # fitter.setup_adr(pressure, temp, spxSize)
    # fitter.add_adr_prior(channel, airmass, parangle)

    fit_psf = fitter.fit()

    # Calculate covariance matrix
    print("Calculating covariance...")
    covariance_names, covariance = fitter.calculate_covariance()
    uncertainties = fitter.calculate_uncertainties(
        names=covariance_names, covariance=covariance
    )

    # Print out fit facts
    if opts.verbosity >= 1:
        print("")
        fitter.print_fit_info("3D metaslice fit", uncertainties=uncertainties,
                              verbosity=opts.verbosity)
        print("")

    # Store guess and fit parameters ------------------------------

    fitpar = fitter.parameters
    chi2 = fitter.fit_chi_square
    dof = fitter.degrees_of_freedom

    print_msg("  Fit result: chi2/dof=%.2f/%d" % (chi2, dof), 1)
    for key, value in fitter.parameters.items():
        print_msg("  Fit result [%20s]: %s" % (key, value), 2)

    print_msg("  Ref. position fit @%.0f A: %+.2f±%.2f × %+.2f±%.2f spx" %
              (lmid, fitpar['ref_center_x'], uncertainties['ref_center_x'],
               fitpar['ref_center_y'], uncertainties['ref_center_y']), 1)

    # Update ADR params
    print_msg("  ADR fit: delta=%.2f±%.2f, theta=%.1f±%.1f deg" %
              (fitpar['adr_delta'], uncertainties['adr_delta'],
               fitpar['adr_theta'] * TA.RAD2DEG, uncertainties['adr_theta'] *
               TA.RAD2DEG), 1)
    adr.set_param(delta=fitpar['adr_delta'], theta=fitpar['adr_theta'])
    print("  Effective airmass: %.2f" % adr.get_airmass())

    # Estimated seeing (FWHM in arcsec)
    seeing = spxSize * fit_psf.calculate_fwhm(wavelength=reference_wavelength)
    print('  Seeing estimate @%.0f A: %.2f" FWHM' % (reference_wavelength,
                                                     seeing))

    # Estimated seeing parameters
    seeing_powerlaw_values = [fit_psf[i] for i in seeing_powerlaw_keys]
    fit_widths = libES.powerLawEval(
        seeing_powerlaw_values, meta_cube.lbda / reference_wavelength
    )

    # Check fit pertinence ------------------------------

    print("TODO: priors and checks of final values!")

    # Compute point-source and background spectra =============================
    if opts.method == 'psf':
        radius = None
        method = 'psf, %s' % ('chi2' if opts.chi2fit else 'least-squares')
    else:
        # Compute aperture radius
        if opts.radius < 0:     # Aperture radius [sigma]
            radius = -opts.radius * seeing / 2.355  # [arcsec]
            method = '%s r=%.1f sigma=%.2f"' % \
                     (opts.method, -opts.radius, radius)
        else:                   # Aperture radius [arcsec]
            radius = opts.radius        # [arcsec]
            method = '%s r=%.2f"' % (opts.method, radius)
        raise Exception("Aperture photometry not implemented!")

    print("Extracting the point-source spectrum (method=%s)..." % method)
    if not hasSky:
        print("WARNING: No background adjusted.")

    # Extraction
    full_cube_data = np.zeros((full_cube.data.shape[0], 15, 15))
    full_cube_var = np.zeros(full_cube_data.shape)
    full_cube_var[...] = np.nan

    full_cube_data[:, full_cube.i, full_cube.j] = full_cube.data
    full_cube_var[:, full_cube.i, full_cube.j] = full_cube.var

    extraction = fit_psf.extract(full_cube_data, full_cube_var,
                                 wavelength=full_cube.lbda)

    # Convert (mean) sky spectrum to "per arcsec**2"
    extraction['background_density'] = extraction['background'] / spxSize**2
    extraction['background_density_variance'] = \
        extraction['background_variance'] / spxSize**4

    # Full covariance matrix of point-source spectrum
    print("TODO: compute point-source spectrum covariance.")

    # Creating a standard SNIFS cube with the adjusted data
    cube_fit = pySNIFS.SNIFS_cube(lbda=meta_cube.lbda)  # Always 225 spx
    cube_fit.x = cube_fit.i - 7                        # x in spaxel
    cube_fit.y = cube_fit.j - 7                        # y in spaxel

    psf = fitter.evaluate()[:, cube_fit.i, cube_fit.j]
    bkg = np.array([np.ones(cube_fit.i.shape) * i['background'] for i in
                   fitter.scene_models])

    cube_fit.data = psf

    # Update header ------------------------------

    # Total point-source flux
    tflux = extraction['amplitude'].sum()
    # Total sky flux (per arcsec**2)
    sflux = extraction['background_density'].sum()

    fill_header(inhdr, fit_psf, adr, meta_cube, opts, chi2, seeing, None,
                (tflux, sflux))

    # Save point-source spectrum ------------------------------

    print("Saving output point-source spectrum to '%s'" % opts.out)

    # Store variance as extension to signal
    star_spec = pySNIFS.spectrum(data=extraction['amplitude'],
                                 var=extraction['amplitude_variance'],
                                 start=full_cube.lbda[0], step=step)

    if opts.covariance:  # Append covariance directly to pySNIFS.spectrum
        print("TODO: implement covariance")
        # star_spec.cov = covspec

    write_fits(star_spec, opts.out, inhdr)

    # Save background spectrum ------------------------------

    if not opts.sky:        # Use default sky spectrum name
        opts.sky = 'sky_%s.fits' % (channel)
    print("Saving output sky spectrum to '%s'" % opts.sky)
    # Store variance as extension to signal
    sky_spec = pySNIFS.spectrum(data=extraction['background_density'],
                                var=extraction['background_density_variance'],
                                start=full_cube.lbda[0], step=step)
    write_fits(sky_spec, opts.sky, inhdr)

    # Save 3D adjusted parameter file ------------------------------

    print("TODO: 3D log file")
    # if opts.log3D:
        # print("Producing 3D adjusted parameter logfile %s..." % opts.log3D)
        # create_3Dlog(opts, meta_cube, cube_fit, fitpar, dfitpar, chi2)

    # Save adjusted PSF ------------------------------

    if opts.keepmodel:
        path, name = os.path.split(opts.out)
        outpsf = os.path.join(path, 'psf_' + name)
        print("Saving adjusted meta-slice PSF in 3D-fits cube '%s'..." %
              outpsf)
        cube_fit.WR_3d_fits(outpsf, header=[])  # No header in cube_fit

    # Create output graphics =================================================

    if opts.plot:
        print("Producing output figures [%s]..." % opts.graph)

        import matplotlib as M
        backends = {'png': 'Agg', 'eps': 'PS', 'pdf': 'PDF', 'svg': 'SVG'}
        try:
            M.use(backends[opts.graph.lower()])
            basename = os.path.splitext(opts.out)[0]
            plot1 = os.path.extsep.join((basename + "_plt", opts.graph))
            plot2 = os.path.extsep.join((basename + "_fit1", opts.graph))
            plot3 = os.path.extsep.join((basename + "_fit2", opts.graph))
            plot4 = os.path.extsep.join((basename + "_fit3", opts.graph))
            plot5 = os.path.extsep.join((basename + "_fit4", opts.graph))
            plot6 = os.path.extsep.join((basename + "_fit5", opts.graph))
            plot7 = os.path.extsep.join((basename + "_fit6", opts.graph))
            plot8 = os.path.extsep.join((basename + "_fit7", opts.graph))
        except KeyError:
            opts.graph = 'pylab'
            plot1 = plot2 = plot3 = plot4 = plot5 = plot6 = plot7 = plot8 = ''
        import matplotlib.pyplot as P

        # Accomodate errorbar fmt API change at v2.2.0
        # (https://matplotlib.org/api/api_changes.html#function-signatures)
        if M.__version__ >= '2.2.0':
            warnings.warn("Matplotlib %s: accomodate new-style errorbar fmt" % M.__version__)
            NONE = 'none'
        else:
            NONE = None

        # Non-default colors
        from ToolBox import MPL
        blue = MPL.blue
        red = MPL.red
        green = MPL.green
        orange = MPL.orange
        purple = MPL.purple

        # Plot of the star and sky spectra -----------------------------------

        print_msg("Producing spectra plot %s..." % plot1, 1)

        fig1 = P.figure()

        if hasSky and sky_spec.data.any():
            axS = fig1.add_subplot(3, 1, 1)  # Point-source
            axB = fig1.add_subplot(3, 1, 2)  # Sky
            axN = fig1.add_subplot(3, 1, 3)  # S/N
        else:
            axS = fig1.add_subplot(2, 1, 1)  # Point-source
            axN = fig1.add_subplot(2, 1, 2)  # S/N

        axS.text(0.95, 0.8, os.path.basename(opts.input),
                 fontsize='small', ha='right', transform=axS.transAxes)

        axS.plot(star_spec.x, star_spec.data, blue)
        axS.errorband(star_spec.x, star_spec.data, np.sqrt(star_spec.var),
                      color=blue)
        axN.plot(star_spec.x, star_spec.data / np.sqrt(star_spec.var), blue)

        if hasSky and sky_spec.data.any():
            axB.plot(sky_spec.x, sky_spec.data, green)
            axB.errorband(sky_spec.x, sky_spec.data, np.sqrt(sky_spec.var),
                          color=green)
            axB.set(title=u"Background spectrum (per arcsec²)",
                    xlim=(sky_spec.x[0], sky_spec.x[-1]),
                    xticklabels=[])
            if background_degree == -2:
                axB.plot(sky_spec.x, sigspecs[:, 2] * 10, red,
                         label=u"Differential ×10")
                axB.errorband(
                    sky_spec.x, sigspecs[:, 2] * 10, np.sqrt(varspecs[:, 2]) * 10,
                    color=red)
                axB.legend(loc='upper right', fontsize='small')
            # Sky S/N
            axN.plot(sky_spec.x, sky_spec.data / np.sqrt(sky_spec.var), green)

        axS.set(title="Point-source spectrum [%s, %s]" % (objname, method),
                xlim=(star_spec.x[0], star_spec.x[-1]), xticklabels=[])
        axN.set(title="Signal/Noise", xlabel=u"Wavelength [Å]",
                xlim=(star_spec.x[0], star_spec.x[-1]))

        fig1.tight_layout()
        if plot1:
            fig1.savefig(plot1)

        # Plot of the fit on each slice --------------------------------------

        print_msg("Producing slice fit plot %s..." % plot2, 1)

        ncol = int(np.floor(np.sqrt(nmeta)))
        nrow = int(np.ceil(nmeta / float(ncol)))

        fig2 = P.figure()
        fig2.suptitle(
            "Slice plots [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        # Compute PSF & bkgnd models on incomplete cube
        mod = fitter.evaluate()[:, meta_cube.i, meta_cube.j]
        psf2 = fitter.evaluate(background=0.)[:, meta_cube.i, meta_cube.j]
        bkg2 = fitter.evaluate(amplitude=0.)[:, meta_cube.i, meta_cube.j]
        sno = np.sort(meta_cube.no)

        for i in xrange(nmeta):        # Loop over meta-slices
            data = meta_cube.data[i, :]
            fit = mod[i, :]
            ax = fig2.add_subplot(nrow, ncol, i + 1)
            ax.plot(sno, data, color=blue, ls='-')  # Signal
            if meta_cube.var is not None:
                ax.errorband(
                    sno, data, np.sqrt(meta_cube.var[i, :]), color=blue)
            ax.plot(sno, fit, color=red, ls='-')   # Model
            if hasSky and sky_spec.data.any():
                ax.plot(sno, psf2[i, :], color=green, ls='-')   # PSF alone
                ax.plot(sno, bkg2[i, :], color=orange, ls='-')  # Background
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)

            ax.set_ylim(data.min() / 1.2, data.max() * 1.2)
            ax.set_xlim(-1, 226)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel #", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')

        fig2.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot2:
            fig2.savefig(plot2)

        # Plot of the fit on rows and columns sum ----------------------------

        print_msg("Producing profile plot %s..." % plot3, 1)

        if not opts.covariance:     # Plot fit on rows and columns sum

            fig3 = P.figure()
            fig3.suptitle(
                "Rows and columns [%s, airmass=%.2f]" % (objname, airmass),
                fontsize='large')

            for i in xrange(nmeta):        # Loop over slices
                ax = fig3.add_subplot(nrow, ncol, i + 1)

                # Signal
                sigSlice = meta_cube.slice2d(i, coord='p', NAN=False)
                prof_I = sigSlice.sum(axis=0)  # Sum along rows
                prof_J = sigSlice.sum(axis=1)  # Sum along columns
                # Errors
                if opts.chi2fit:              # Chi2 fit: plot errorbars
                    varSlice = meta_cube.slice2d(
                        i, coord='p', var=True, NAN=False)
                    err_I = np.sqrt(varSlice.sum(axis=0))
                    err_J = np.sqrt(varSlice.sum(axis=1))
                    ax.errorbar(range(len(prof_I)), prof_I, err_I,
                                fmt='o', c=blue, ecolor=blue, ms=3)
                    ax.errorbar(range(len(prof_J)), prof_J, err_J,
                                fmt='^', c=red, ecolor=red, ms=3)
                else:            # Least-square fit
                    ax.plot(range(len(prof_I)), prof_I,
                            marker='o', c=blue, ms=3, ls='None')
                    ax.plot(range(len(prof_J)), prof_J,
                            marker='^', c=red, ms=3, ls='None')
                # Model
                modSlice = cube_fit.slice2d(i, coord='p')
                mod_I = modSlice.sum(axis=0)
                mod_J = modSlice.sum(axis=1)
                ax.plot(mod_I, ls='-', color=blue)
                ax.plot(mod_J, ls='-', color=red)

                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("I (blue) or J (red)", fontsize='small')
                    ax.set_ylabel("Flux", fontsize='small')

                fig3.subplots_adjust(left=0.06, right=0.96,
                                     bottom=0.06, top=0.95)
        else:                           # Plot correlation matrices

            # Parameter correlation matrix
            corrpar = covpar / np.outer(dfitpar, dfitpar)
            parnames = data_model.func[0].parnames  # PSF param names
            if background_degree >= 0:    # Add polynomial background param names
                coeffnames = ["00"] + \
                             [ "%d%d" % (d - j, j)
                               for d in range(1, background_degree + 1)
                               for j in range(d + 1) ]
                parnames += [ "b%02d_%s" % (s + 1, c)
                              for c in coeffnames for s in range(nmeta) ]
            elif background_degree == -2:
                coeffnames = ["mean", "diff"]
                parnames += [ "b%02d_%s" % (s + 1, c)
                              for c in coeffnames for s in range(nmeta) ]

            assert len(parnames) == corrpar.shape[0]
            # Remove some of the names for clarity
            parnames[npar_psf + 1::2] = [''] * len(parnames[npar_psf + 1::2])

            fig3 = P.figure(figsize=(7, 6))
            ax3 = fig3.add_subplot(1, 1, 1,
                                   title="Parameter correlation matrix")
            im3 = ax3.imshow(np.absolute(corrpar),
                             vmin=1e-3, vmax=1,
                             norm=P.matplotlib.colors.LogNorm(),
                             aspect='equal', origin='upper',
                             interpolation='nearest')
            ax3.set_xticks(range(len(parnames)))
            ax3.set_xticklabels(parnames,
                                va='top', fontsize='x-small', rotation=90)
            ax3.set_yticks(range(len(parnames)))
            ax3.set_yticklabels(parnames,
                                ha='right', fontsize='x-small')

            cb3 = fig3.colorbar(im3, ax=ax3, orientation='vertical')
            cb3.set_label("|Correlation|")

            fig3.tight_layout()

        if plot3:
            fig3.savefig(plot3)

        # Plot of the star center of gravity and adjusted center -------------

        print_msg("Producing ADR plot %s..." % plot4, 1)
        # Guessed and adjusted position at current wavelength
        adr_scale = adr.get_scale(meta_cube.lbda) / spxSize
        xguess = xc + delta0 * np.sin(theta0) * adr_scale
        yguess = yc - delta0 * np.cos(theta0) * adr_scale

        xfit, yfit = adr.refract(
            fitpar['ref_center_x'], fitpar['ref_center_y'], meta_cube.lbda,
            unit=spxSize
        )

        fig4 = P.figure()

        ax4c = fig4.add_subplot(2, 1, 1,
                                aspect='equal', adjustable='datalim',
                                xlabel="X center [spx]",
                                ylabel="Y center [spx]",
                                title="ADR plot [%s, airmass=%.2f]" %
                                (objname, airmass))
        ax4a = fig4.add_subplot(2, 2, 3,
                                xlabel=u"Wavelength [Å]",
                                ylabel="X center [spx]")
        ax4b = fig4.add_subplot(2, 2, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel="Y center [spx]")

        if good.any():
            ax4a.errorbar(meta_cube.lbda[good], xc_vec[good],
                          yerr=xc_err[good], fmt=NONE, ecolor=green)
            ax4a.scatter(meta_cube.lbda[good], xc_vec[good],
                         edgecolors='none', c=meta_cube.lbda[good],
                         cmap=M.cm.jet, zorder=3, label="Fit 2D")
        if bad.any():
            ax4a.plot(meta_cube.lbda[bad], xc_vec[bad],
                      mfc=red, mec=red, marker='.', ls='None', label='_')
        ax4a.plot(meta_cube.lbda, xguess, 'k--', label="Guess 3D")
        ax4a.plot(meta_cube.lbda, xfit, green, label="Fit 3D")
        P.setp(ax4a.get_xticklabels() + ax4a.get_yticklabels(),
               fontsize='xx-small')
        ax4a.legend(loc='best', fontsize='small', frameon=False)

        if good.any():
            ax4b.errorbar(meta_cube.lbda[good], yc_vec[good],
                          yerr=yc_err[good], fmt=NONE, ecolor=green)
            ax4b.scatter(meta_cube.lbda[good], yc_vec[good], edgecolors='none',
                         c=meta_cube.lbda[good], cmap=M.cm.jet, zorder=3)
        if bad.any():
            ax4b.plot(meta_cube.lbda[bad], yc_vec[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax4b.plot(meta_cube.lbda, yfit, green)
        ax4b.plot(meta_cube.lbda, yguess, 'k--')
        P.setp(ax4b.get_xticklabels() + ax4b.get_yticklabels(),
               fontsize='xx-small')

        if valid.any():
            ax4c.errorbar(xc_vec[valid], yc_vec[valid],
                          xerr=xc_err[valid], yerr=yc_err[valid],
                          fmt=NONE, ecolor=green)
        if good.any():
            ax4c.scatter(xc_vec[good], yc_vec[good], edgecolors='none',
                         c=meta_cube.lbda[good],
                         cmap=M.cm.jet, zorder=3)
            # Plot position selection process
            ax4c.plot(xmids[good], ymids[good], marker='.',
                      mfc=blue, mec=blue, ls='None')  # Selected ref. positions
        if bad.any():
            ax4c.plot(xmids[bad], ymids[bad], marker='.',
                      mfc=red, mec=red, ls='None')   # Discarded ref. positions
        ax4c.plot((xmid, xc), (ymid, yc), 'k-')
        ax4c.plot(xguess, yguess, 'k--')  # Guess ADR
        ax4c.plot(xfit, yfit, green)      # Adjusted ADR
        ax4c.set_autoscale_on(False)
        ax4c.plot((xc,), (yc,), 'k+')
        ax4c.add_patch(M.patches.Circle((xmid, ymid), radius=rmax,
                                        ec='0.8', fc='None'))  # ADR selection
        ax4c.add_patch(M.patches.Rectangle((-7.5, -7.5), 15, 15,
                                           ec='0.8', lw=2, fc='None'))  # FoV
        txt = u'Guess: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
              (xc, yc, airmass, theta0 * TA.RAD2DEG)
        txt += u'\nFit: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
               (fitpar['ref_center_x'], fitpar['ref_center_y'],
                adr.get_airmass(), adr.get_parangle())
        txtcol = 'k'
        if accountant:
            if accountant.test_warning('ES_PRIOR_POSITION'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_POSITION')
                txtcol = MPL.red
            if accountant.test_warning('ES_PRIOR_AIRMASS'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_AIRMASS')
                txtcol = MPL.red
            if accountant.test_warning('ES_PRIOR_PARANGLE'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_PARANGLE')
                txtcol = MPL.red
        ax4c.text(0.95, 0.8, txt, transform=ax4c.transAxes,
                  fontsize='small', ha='right', color=txtcol)

        fig4.tight_layout()
        if plot4:
            fig4.savefig(plot4)

        # Plot of the other model parameters ---------------------------------

        print_msg("Producing model parameter plot %s..." % plot6, 1)

        if opts.psf == 'old':
            key_6b = 'ell'
            key_6c = 'xy'
            label_6b = u'y² coeff.'
            label_6c = u'xy coeff.'
        elif opts.psf == 'fourier':
            key_6b = 'ell_coord_x'
            key_6c = 'ell_coord_y'
            label_6b = u'ell x'
            label_6c = u'ell y'

        val_6b = extract_key(individual_psf_models, key_6b)
        err_6b = extract_key(individual_uncertainties, key_6b)
        val_6c = extract_key(individual_psf_models, key_6c)
        err_6c = extract_key(individual_uncertainties, key_6c)

        guess_6b = np.median(val_6b[good])
        guess_6c = np.median(val_6c[good])

        fit_6b = fitpar[key_6b]
        fiterr_6b = uncertainties[key_6b]
        fit_6c = fitpar[key_6c]
        fiterr_6c = uncertainties[key_6c]

        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, green, label="Fit 3D")
            if dy is not None:
                ax.errorband(x, y, dy, color=green)

        fig6 = P.figure()

        ax6a = fig6.add_subplot(2, 1, 1,
                                title='Model parameters '
                                '[%s, seeing %.2f" FWHM]' % (objname, seeing),
                                xticklabels=[],
                                ylabel=u'α [spx]')
        ax6b = fig6.add_subplot(4, 1, 3,
                                xticklabels=[],
                                ylabel=label_6b)
        ax6c = fig6.add_subplot(4, 1, 4,
                                xlabel=u"Wavelength [Å]",
                                ylabel=label_6c)

        if good.any():
            ax6a.errorbar(meta_cube.lbda[good], individual_seeing_widths[good],
                          individual_seeing_errs[good], marker='.', mfc=blue,
                          mec=blue, ecolor=blue, capsize=0, ls='None',
                          label="Fit 2D")
        if bad.any():
            ax6a.plot(meta_cube.lbda[bad], individual_seeing_widths[bad],
                      marker='.', mfc=red, mec=red, ls='None', label="_")
        ax6a.plot(meta_cube.lbda, guess_widths, 'k--',
                  label="Guess 3D" if not opts.seeingPrior else "Prior 3D")
        # plot_conf_interval(ax6a, meta_cube.lbda, fit_alpha, err_alpha)
        plot_conf_interval(ax6a, meta_cube.lbda, fit_widths, None)
        txt = 'Guess: %s' % \
              (', '.join(['a%d=%.2f' % (i, a) for i, a in
                          enumerate(seeing_powerlaw_guesses)]))
        txt += '\nFit: %s' % \
               (', '.join(['a%d=%.2f' % (i, a) for i, a in
                           enumerate(seeing_powerlaw_values)]))
        txtcol = 'k'
        if accountant and accountant.test_warning('ES_PRIOR_SEEING'):
            txt += '\n%s' % accountant.get_warning('ES_PRIOR_SEEING')
            txtcol = MPL.red
        ax6a.text(0.95, 0.8, txt, transform=ax6a.transAxes,
                  fontsize='small', ha='right', color=txtcol)
        ax6a.legend(loc='upper left', fontsize='small', frameon=False)
        P.setp(ax6a.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax6b.errorbar(meta_cube.lbda[good], val_6b[good], err_6b[good],
                          marker='.',
                          mfc=blue, mec=blue, ecolor=blue, capsize=0, ls='None')
        if bad.any():
            ax6b.plot(meta_cube.lbda[bad], val_6b[bad],
                      marker='.', mfc=red, mec=red, ls='None')
        ax6b.plot(meta_cube.lbda, guess_6b*np.ones(len(meta_cube.lbda)), 'k--')
        # plot_conf_interval(ax6b, meta_cube.lbda, fit_ell, err_ell)
        plot_conf_interval(ax6b, meta_cube.lbda,
                           fit_6b*np.ones(len(meta_cube.lbda)),
                           fiterr_6b*np.ones(len(meta_cube.lbda)))
        txt = 'Guess: %s=%.2f' % (label_6b, guess_6b)
        txt += '\nFit: %s=%.2f' % (label_6b, fit_6b)

        ax6b.text(0.95, 0.1, txt, transform=ax6b.transAxes,
                  fontsize='small', ha='right', va='bottom')
        P.setp(ax6b.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax6c.errorbar(meta_cube.lbda[good], val_6c[good], err_6c[good],
                          marker='.', mfc=blue, mec=blue, ecolor=blue,
                          capsize=0, ls='None')
        if bad.any():
            ax6c.plot(meta_cube.lbda[bad], val_6c[bad], marker='.', mfc=red,
                      mec=red, ls='None')
        ax6c.plot([meta_cube.lstart, meta_cube.lend], [guess_6c] * 2, 'k--')
        plot_conf_interval(ax6c,
                           np.asarray([meta_cube.lstart, meta_cube.lend]),
                           np.ones(2) * fit_6c, np.ones(2) * fiterr_6c)
        ax6c.text(0.95, 0.1,
                  u'Guess: %s=%4.2f\nFit: %s=%4.2f' % (label_6c, guess_6c,
                                                       label_6c, fit_6c),
                  transform=ax6c.transAxes,
                  fontsize='small', ha='right', va='bottom')
        P.setp(ax6c.get_xticklabels() + ax6c.get_yticklabels(),
               fontsize='x-small')

        fig6.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.95)
        if plot6:
            fig6.savefig(plot6)

        # Plot of the radial profile -----------------------------------------

        print_msg("Producing radial profile plot %s..." % plot7, 1)

        fig7 = P.figure()
        fig7.suptitle(
            "Radial profile plot [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        def ellRadius(x, y, x0, y0, ell, xy):
            dx = x - x0
            dy = y - y0
            # BEWARE: can return NaN's if ellipse is ill-defined
            return np.sqrt(dx ** 2 + ell * dy ** 2 + 2 * xy * dx * dy)

        def radialbin(r, f, binsize=20, weighted=True):
            rbins = np.sort(r)[::binsize]  # Bin limits, starting from min(r)
            ibins = np.digitize(r, rbins)  # WARNING: ibins(min(r)) = 1
            ib = np.arange(len(rbins)) + 1  # Bin indices
            ib = [iib for iib in ib if r[ibins == iib].any()]
            rb = np.array([r[ibins == iib].mean() for iib in ib])  # Mean radius
            if weighted:
                fb = np.array([np.average(f[ibins == iib], weights=r[ibins == iib])
                              for iib in ib])  # Mean radius-weighted data
            else:
                fb = np.array([f[ibins == iib].mean()
                              for iib in ib])  # Mean data
            # Error on bin mean quantities
            # snb = np.sqrt([ len(r[ibins==iib]) for iib in ib ]) # sqrt(#points)
            # drb = np.array([ r[ibins==iib].std()/n for iib,n in zip(ib,snb) ])
            # dfb = np.array([ f[ibins==iib].std()/n for iib,n in zip(ib,snb) ])
            return rb, fb

        for i in xrange(nmeta):        # Loop over slices
            ax = fig7.add_subplot(nrow, ncol, i + 1, yscale='log')
            # Use adjusted elliptical radius instead of plain radius
            # r    = np.hypot(meta_cube.x-xfit[i], meta_cube.y-yfit[i])
            # rfit = np.hypot(cube_fit.x-xfit[i], cube_fit.y-yfit[i])
            if opts.psf == 'old':
                r = ellRadius(meta_cube.x, meta_cube.y,
                              xfit[i], yfit[i], fitpar['ell'], fitpar['xy'])
                rfit = ellRadius(cube_fit.x, cube_fit.y,
                                 xfit[i], yfit[i], fitpar['ell'], fitpar['xy'])
            else:
                r = np.hypot(meta_cube.x - xfit[i], meta_cube.y - yfit[i])
                rfit = np.hypot(cube_fit.x - xfit[i], cube_fit.y - yfit[i])

            ax.plot(r, meta_cube.data[i],
                    marker=',', mfc=blue, mec=blue, ls='None')  # Data
            ax.plot(rfit, cube_fit.data[i],
                    marker='.', mfc=red, mec=red, ms=1, ls='None')  # Model
            # ax.set_autoscale_on(False)
            if hasSky and sky_spec.data.any():
                ax.plot(rfit, psf[i], marker='.', mfc=green, mec=green,
                        ms=1, ls='None')  # PSF alone
                ax.plot(rfit, bkg[i], marker='.', mfc=orange, mec=orange,
                        ms=1, ls='None')  # Sky
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            if opts.method != 'psf':
                ax.axvline(radius / spxSize, color=orange, lw=2)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spx]", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')
            # ax.axis([0, rfit.max()*1.1,
            #          meta_cube.data[i][meta_cube.data[i]>0].min()/2,
            #          meta_cube.data[i].max()*2])
            ax.axis([0, rfit.max() * 1.1,
                     cube_fit.data[i].min() / 2, cube_fit.data[i].max() * 2])

            # Binned values
            rb, db = radialbin(r, meta_cube.data[i])
            ax.plot(rb, db, 'c.')
            rfb, fb = radialbin(rfit, cube_fit.data[i])
            ax.plot(rfb, fb, 'm.')

        fig7.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if plot7:
            fig7.savefig(plot7)

        # Missing energy (not activated by default)
        if opts.verbosity >= 3:
            print_msg("Producing missing energy plot...", 1)

            figB = P.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figB.add_subplot(nrow, ncol, i + 1, yscale='log')
                # Binned values
                rb, db = radialbin(r, meta_cube.data[i])
                rfb, fb = radialbin(rfit, cube_fit.data[i])
                tb = np.cumsum(rb * db)
                norm = tb.max()
                ax.plot(rb, 1 - tb / norm, 'c.')
                ax.plot(rfb, 1 - np.cumsum(rfb * fb) / norm, 'm.')
                if hasSky and sky_spec.data.any():
                    rfb, pb = radialbin(rfit, psf[i])
                    rfb, bb = radialbin(rfit, bkg[i])
                    ax.plot(rfb, 1 - np.cumsum(rfb * pb) / norm,
                            marker='.', mfc=green, mec=green, ls='None')
                    ax.plot(rfb, 1 - np.cumsum(rfb * bb) / norm,
                            marker='.', mfc=orange, mec=orange, ls='None')
                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius / spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel(
                        "Missing energy [fraction]", fontsize='small')

            figB.tight_layout()
            if opts.graph != 'pylab':
                figB.savefig(
                    os.path.extsep.join((basename + "_nrj", opts.graph)))

        # Radial Chi2 plot (not activated by default)
        if opts.verbosity >= 3:
            print_msg("Producing radial chi2 plot...", 1)

            figA = P.figure()
            for i in xrange(nmeta):        # Loop over slices
                ax = figA.add_subplot(nrow, ncol, i + 1, yscale='log')
                chi2 = (meta_cube.slice2d(i, coord='p') -
                        cube_fit.slice2d(i, coord='p')) ** 2 / \
                    meta_cube.slice2d(i, coord='p', var=True)
                ax.plot(rfit, chi2.flatten(),
                        marker='.', ls='none', mfc=blue, mec=blue)
                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                       fontsize='xx-small')
                ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                        fontsize='x-small', transform=ax.transAxes)
                if opts.method != 'psf':
                    ax.axvline(radius / spxSize, color=orange, lw=2)
                if ax.is_last_row() and ax.is_first_col():
                    ax.set_xlabel("Elliptical radius [spx]",
                                  fontsize='small')
                    ax.set_ylabel(u"χ²", fontsize='small')

            figA.tight_layout()
            if opts.graph != 'pylab':
                figA.savefig(
                    os.path.extsep.join((basename + "_chi2", opts.graph)))

        # Contour plot of each slice -----------------------------------------

        print_msg("Producing PSF contour plot %s..." % plot8, 1)

        fig8 = P.figure()
        fig8.suptitle(
            "Data and fit [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        extent = (meta_cube.x.min() - 0.5, meta_cube.x.max() + 0.5,
                  meta_cube.y.min() - 0.5, meta_cube.y.max() + 0.5)
        for i in xrange(nmeta):        # Loop over meta-slices
            ax = fig8.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p')
            fit = cube_fit.slice2d(i, coord='p')
            vmin, vmax = np.percentile(data[data > 0], (5., 95.))  # Percentiles
            lev = np.logspace(np.log10(vmin), np.log10(vmax), 5)
            ax.contour(data, lev, origin='lower', extent=extent,
                       cmap=M.cm.jet)                      # Data
            ax.contour(fit, lev, origin='lower', extent=extent,
                       linestyles='dashed', cmap=M.cm.jet)  # Fit
            if valid[i]:
                ax.errorbar((xc_vec[i],), (yc_vec[i],),
                            xerr=(xc_err[i],), yerr=(yc_err[i],),
                            fmt=NONE, ecolor=blue if good[i] else red)
            ax.plot((xfit[i],), (yfit[i],), marker='*', color=green)
            if opts.method != 'psf':
                ax.add_patch(M.patches.Circle((xfit[i], yfit[i]),
                                              radius / spxSize,
                                              fc='None', ec=orange, lw=2))
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                P.setp(ax.get_xticklabels(), visible=False)
            if not ax.is_first_col():
                P.setp(ax.get_yticklabels(), visible=False)

        fig8.subplots_adjust(left=0.05, right=0.96, bottom=0.06, top=0.95,
                             hspace=0.02, wspace=0.02)
        if plot8:
            fig8.savefig(plot8)

        # Residuals of each slice --------------------------------------------

        print_msg("Producing residual plot %s..." % plot5, 1)

        fig5 = P.figure()
        fig5.suptitle(
            "Residual plot [%s, airmass=%.2f]" % (objname, airmass),
            fontsize='large')

        images = []
        for i in xrange(nmeta):        # Loop over meta-slices
            ax = fig5.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = meta_cube.slice2d(i, coord='p')  # Signal
            fit = cube_fit.slice2d(i, coord='p')  # Model
            if opts.chi2fit:    # Chi2 fit: display residuals in units of sigma
                var = meta_cube.slice2d(i, coord='p', var=True, NAN=False)
                res = np.nan_to_num((data - fit) / np.sqrt(var))
            else:               # Least-squares: display relative residuals
                res = np.nan_to_num((data - fit) / fit) * 100  # [%]

            # List of images, to be commonly normalized latter on
            images.append(ax.imshow(res, origin='lower', extent=extent,
                                    cmap=M.cm.RdBu_r, interpolation='nearest'))

            ax.plot((xfit[i],), (yfit[i],), marker='*', color=green)
            P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                   fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            ax.axis(extent)

            # Axis management
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                ax.xaxis.set_major_formatter(M.ticker.NullFormatter())
            if not ax.is_first_col():
                ax.yaxis.set_major_formatter(M.ticker.NullFormatter())

        # Common image normalization
        vmin, vmax = np.percentile(
            [im.get_array().filled() for im in images], (3., 97.))
        norm = M.colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Colorbar
        cax = fig5.add_axes([0.90, 0.07, 0.02, 0.87])
        cbar = fig5.colorbar(images[0], cax, orientation='vertical')
        P.setp(cbar.ax.get_yticklabels(), fontsize='small')
        if opts.chi2fit:    # Chi2 fit
            cbar.set_label(u'Residuals [σ]', fontsize='small')
        else:
            cbar.set_label(u'Residuals [%]', fontsize='small')

        fig5.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.95,
                             hspace=0.02, wspace=0.02)
        if plot5:
            fig5.savefig(plot5)

        # Show figures -------------------------------------------------

        if opts.graph == 'pylab':
            P.show()

# End of extract_star.py ======================================================
