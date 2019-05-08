#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""3D fixed PSF/aperture extractor"""

import os
import numpy as np
import pySNIFS
from astropy.io import fits

import scene_model
from scene_model.snifs import SnifsFourierSceneModel, cube_to_arrays, \
    convert_radius_to_pixels, apply_variance_filter

if __name__ == '__main__':

    import optparse

    usage = "Usage: [%prog] [options] inspec.fits"
    parser = optparse.OptionParser(
        usage, version=scene_model.config.__version__
    )

    parser.add_option("-c", "--cube",
                      help="Datacube")
    parser.add_option("-s", "--spec",
                      help="Reference spectrum for PSF parameters")
    parser.add_option("-o", "--out",
                      help="Output spectrum [%default]", default='spec.fits')
    parser.add_option("-S", "--skyDeg", type="int",
                      help="Sky polynomial background degree [%default]",
                      default=-1)
    parser.add_option(
        "-m", "--method",
        help="Extraction method (psf|optimal|aperture|subaperture) "
        "['%default']",
        default="psf"
    )
    parser.add_option(
        "-r", "--radius", type="float",
        help="Aperture radius for non-PSF extraction "
        "(>0: in arcsec, <0: in seeing sigma) [%default]",
        default=-5.
    )
    parser.add_option("--filterVariance", dest="filter_variance",
                      action="store_true", help="Apply a filter in wavelength "
                      "to the variance estimate to avoid Poisson biases.",
                      default=False)

    opts, args = parser.parse_args()

    # Input spectrum
    print("Opening input spectrum %s" % opts.spec)
    spec = pySNIFS.spectrum(opts.spec)
    # pySNIFS doesn't keep the header, so read that separately.
    header = fits.getheader(opts.spec)
    print(spec)

    # Reference/input cube
    print("Opening cube %s" % opts.cube)
    try:                        # Try Euro3D
        cube = pySNIFS.SNIFS_cube(e3d_file=opts.cube)
        cubetype = "Euro3D"
    except ValueError:          # Try 3D
        cube = pySNIFS.SNIFS_cube(fits3d_file=opts.cube)
        cubetype = "3D"
    print("  %s, %d slices [%.2f-%.2f], %d spaxels" %
          (cubetype, cube.nslice, cube.lstart, cube.lend, cube.nlens))

    # Check spectral samplings are coherent
    assert (spec.len, spec.start, spec.step) == \
        (cube.nslice, cube.lstart, cube.lstep), \
        "Incompatible spectrum and reference cube"

    # Load the scene model. We override the sky degree parameter in the header
    # with the one passed in to this method.
    psf = header['ES_PSF']
    header['ES_SDEG'] = opts.skyDeg
    if psf == 'fourier':
        model = SnifsFourierSceneModel.from_fits_header(header)
    else:
        parser.error("Unsupported PSF model ES_PSF=%s" % psf)
    model.setup_grid(grid_size=(15, 15))

    # Compute aperture radius [arcsec]
    if opts.method == 'psf':
        radius = None
    else:
        radius = convert_radius_to_pixels(opts.radius, header['SEEING'],
                                          cube.spxSize)

    cube_data, cube_var = cube_to_arrays(cube)

    # Filter the variance if set
    if opts.filter_variance:
        cube_var = apply_variance_filter(cube_var)

    extraction = model.extract(cube_data, cube_var, wavelength=cube.lbda,
                               method=opts.method, radius=radius)

    out_spec = pySNIFS.spectrum(
        data=extraction['amplitude'],
        var=extraction['amplitude_variance'],
        start=cube.lbda[0],
        step=cube.lstep
    )
    out_spec.WR_fits_file(opts.out, header_list=header.items())
