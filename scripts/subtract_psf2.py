#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""Generate (and subtract) extract_star PSF cube from extracted spectrum"""

import os
import numpy as np
import pySNIFS
from astropy.io import fits

import scene_model
from scene_model.snifs import SnifsFourierSceneModel

import warnings
# Ignore warnings from pyfits.writeto
# (https://github.com/spacetelescope/PyFITS/issues/43)
warnings.filterwarnings("ignore", "Overwriting existing file")


if __name__ == '__main__':
    import optparse

    usage = "Usage: [%prog] [options] inspec.fits"
    parser = optparse.OptionParser(
        usage, version=scene_model.config.__version__
    )

    parser.add_option("-r", "--ref",
                      help="Reference datacube")
    parser.add_option("-s", "--sky",
                      help="Sky spectrum to be removed from output datacube")
    parser.add_option("-o", "--out",
                      help="Output point-source subtracted datacube")

    parser.add_option("-k", "--keep", action="store_true",
                      help="Save point-source datacube (with --psfname)",
                      default=False)
    parser.add_option("--psfname",
                      help="Name of point-source datacube [psf_refcube]",
                      default=None)
    parser.add_option("-n", "--nosubtract",
                      dest="subtract", action="store_false",
                      help="Do *not* subtract point-source from datacube",
                      default=True)

    opts, args = parser.parse_args()

    if not opts.ref:
        parser.error("Reference cube not specified")
    if opts.subtract and not opts.out:
        parser.error("Name for output point-source subtracted cube "
                     "not specified")
    if opts.psfname is None:    # Default name for output PSF cube
        opts.psfname = 'psf_' + os.path.basename(opts.ref)
    else:                       # Assume that user wants to keep the PSF...
        opts.keep = True

    # Input spectrum
    print("Opening input spectrum %s" % args[0])
    spec = pySNIFS.spectrum(args[0])
    # pySNIFS doesn't keep the header, so read that separately.
    header = fits.getheader(args[0])
    print(spec)

    # Reference/input cube
    print("Opening reference cube %s" % opts.ref)
    try:                        # Try Euro3D
        cube = pySNIFS.SNIFS_cube(e3d_file=opts.ref)
        cube.writeto = cube.WR_e3d_file
        cubetype = "Euro3D"
    except ValueError:          # Try 3D
        cube = pySNIFS.SNIFS_cube(fits3d_file=opts.ref)
        cube.writeto = cube.WR_3d_fits
        cubetype = "3D"
    print("  %s, %d slices [%.2f-%.2f], %d spaxels" %
          (cubetype, cube.nslice, cube.lstart, cube.lend, cube.nlens))

    # Check spectral samplings are coherent
    assert (spec.len, spec.start, spec.step) == \
        (cube.nslice, cube.lstart, cube.lstep), \
        "Incompatible spectrum and reference cube"

    # Load the scene model
    psf = header['ES_PSF']
    if psf == 'fourier':
        model = SnifsFourierSceneModel.from_fits_header(header)
    else:
        parser.error("Unsupported PSF model ES_PSF=%s" % psf)
    model.setup_grid(grid_size=(15, 15))

    if opts.sky:
        sky = pySNIFS.spectrum(opts.sky)
        sky_degree = header['ES_SDEG']

        if sky_degree == -1:
            # No background
            sky_data = np.zeros(spec.len)
            sky_var = np.zeros(spec.len)
        elif sky_degree == 0:
            # Flat background
            sky_data = sky.data
            sky_var = sky.var
        else:
            raise NotImplementedError(
                'skyDeg %s not implemented!' % sky_degree
            )

        spaxel_size = header['ES_SPXSZ']

        # from arcsec^-2 into spaxels^-1
        sky_data *= spaxel_size**2
        sky_var *= spaxel_size**4
    else:
        sky_data = np.zeros(spec.len)
        sky_var = np.zeros(spec.len)

    # Evaluate the PSF components without the spectrum/sky applied to it. We
    # evaluate the components separately, so that we can pull out just the PSF.
    # The shape of eval_components will be (num_wavelengths, num_components,
    # num_spaxels_x, num_spaxels_y). The first component is the PSF, the
    # following ones are the background components.
    eval_components = model.evaluate_multi(
        wavelength=cube.lbda,
        separate_components=True,
        apply_coefficients=False,
    )

    # The cube has a flat version of the spaxels rather than a 2D image. There
    # are also some missing spaxels. Convert to that shape. flat_components
    # then has the shape (num_wavelengths, num_components, num_spaxels).
    flat_components = eval_components[..., cube.i, cube.j]

    psf_component = flat_components[:, 0]
    background_component = flat_components[:, 1]

    psf_model = psf_component * spec.data[:, None]
    psf_var = psf_component**2 * spec.var[:, None]

    background_model = background_component * sky_data[:, None]
    background_var = background_component**2 * sky_var[:, None]

    full_model = psf_model + background_model
    full_model_var = psf_var + background_var

    if opts.subtract:
        # Save point-source subtracted cube
        # Does it really make sense to add the variance here? I think that this
        # might be double counting, but the original script did it so I kept
        # it.
        cube.data -= full_model
        cube.var += full_model_var

        print("Saving point-source subtracted %s cube %s" % (cubetype,
                                                             opts.out))
        cube.writeto(opts.out)

    if opts.keep:
        # Save PSF cube
        cube.data = psf_model
        cube.var = psf_var
        print("Saving point-source %s cube %s" % (cubetype, opts.psfname))
        cube.writeto(opts.psfname)
