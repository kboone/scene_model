#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
3D PSF-based point-source extractor. There are several PSF choices. The default
PSF is a exponential profile in Fourier space convolved with an instrumental
PSF and a model of the tracking.
"""

import os
import numpy as np

from ToolBox.Misc import warning2stdout

import scene_model
from scene_model import snifs as snifs_scene

import warnings
warnings.showwarning = warning2stdout   # Redirect warnings to stdout
warnings.filterwarnings("ignore", "Overwriting existing file")

# Numpy setup
np.set_printoptions(linewidth=999)       # X-wide lines

if __name__ == "__main__":
    import optparse

    usage = "[%prog] [options] incube.fits"

    parser = optparse.OptionParser(
        usage, version=scene_model.config.__version__
    )

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
    parser.add_option("--psf", choices=('classic', 'fourier'), help="PSF model"
                      " (classic|fourier) [%default]", default='fourier')

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
                      choices=('psf', 'aperture', 'subaperture'),
                      help="Extraction method (psf|[sub]aperture) [%default]",
                      default="psf")
    parser.add_option("-r", "--radius", type=float,
                      help="Aperture radius for non-PSF extraction "
                      "(>0: in \", <0: in seeing sigma) [%default]",
                      default=-5.)
    parser.add_option("-L", "--leastSquares",
                      dest="least_squares", action="store_true",
                      help="Least-square fit [default is a chi2 fit]",
                      default=False)
    parser.add_option("--filterVariance", dest="filter_variance",
                      action="store_true", help="Apply a filter in wavelength "
                      "to the variance estimate to avoid Poisson biases.",
                      default=False)

    # Plotting
    parser.add_option("-g", "--graph",
                      choices=('png', 'eps', 'pdf', 'svg', 'pylab'),
                      help="Graphic output format (png,eps,pdf,svg,pylab)")
    parser.add_option("-p", "--plot", action='store_true',
                      help="Plot flag (='-g pylab')")

    # Priors
    parser.add_option("--usePriors", type=float,
                      help="PSF prior hyper-scale, or 0 for none "
                      "(req. powerlaw-PSF) [%default]",
                      default=0.)
    parser.add_option("--seeingPrior", type=float,
                      help="Seeing prior (from Exposure.Seeing) [\"]")

    # Expert options
    parser.add_option("--keepmodel", action='store_true',
                      help="Store meta-slices and adjusted model in 3D cubes")

    # Debug options
    parser.add_option("-v", "--verbosity", type=int,
                      help="Verbosity level (<0: quiet) [%default]",
                      default=0)
    parser.add_option("-f", "--file", type=str, dest="log2D",
                      help="2D adjustment logfile name")
    parser.add_option("-F", "--File", type=str, dest="log3D",
                      help="3D adjustment logfile name")

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

    # Accounting
    if opts.accountant:
        try:
            from libRecord import Accountant
        except ImportError:
            print("WARNING: libRecord is not accessible, accounting disabled")
            accountant = None
        else:
            import atexit

            accountant = Accountant(opts.accountant, opts.out)
            print(accountant)
            atexit.register(accountant.finalize)
    else:
        accountant = None

    # Load the data cube into the fitter.
    try:
        fitter = snifs_scene.SnifsCubeFitter(
            opts.input,
            psf=opts.psf,
            background_degree=opts.skyDeg,
            subsampling=opts.subsampling,
            border=opts.border,
            least_squares=opts.least_squares,
            filter_variance=opts.filter_variance,
            prior_scale=opts.usePriors,
            seeing_prior=opts.seeingPrior,
            accountant=accountant,
            verbosity=opts.verbosity
        )
    except scene_model.SceneModelException as e:
        parser.error(e.message)

    # If the output paths weren't specified, build defaults.
    if not opts.out:
        opts.out = 'spec_%s.fits' % (fitter.channel)
    if not opts.sky:
        opts.sky = 'sky_%s.fits' % (fitter.channel)

    # Do the 2D fit
    fitter.fit_metaslices_2d(num_meta_slices=opts.nmeta)

    if opts.log2D:
        fitter.write_log_2d(opts.log2D)

    # Do the 3D fit
    fitter.fit_metaslices_3d()

    if opts.log3D:
        fitter.write_log_3d(opts.log3D)

    # Make sure that everything looks ok
    fitter.check_validity()

    # Save the meta-slice model if desired.
    if opts.keepmodel:
        path, name = os.path.split(opts.out)

        meta_path = os.path.join(path, 'meta_' + name)
        print("  Saving meta-slices in 3D-fits cube '%s'..." % meta_path)
        fitter.meta_cube.WR_3d_fits(meta_path)

        model_path = os.path.join(path, 'psf_' + name)
        print("  Saving adjusted meta-slice model in 3D-fits cube '%s'..." %
              model_path)
        fitter.meta_cube_model.WR_3d_fits(model_path, header=[])

    # Extract the point source spectrum
    fitter.extract(method=opts.method, radius=opts.radius)

    # Write the point source and background spectra to fits files.
    fitter.write_spectrum(opts.out, opts.sky)

    if opts.plot:
        print("Producing output figures [%s]..." % opts.graph)

        import matplotlib as M
        backends = {'png': 'Agg', 'eps': 'PS', 'pdf': 'PDF', 'svg': 'SVG'}
        try:
            M.use(backends[opts.graph.lower()])
            basename = os.path.splitext(opts.out)[0]
        except KeyError:
            opts.graph = 'pylab'

        def make_plot_path(extension):
            if opts.graph == 'pylab':
                # Don't save plots. Set paths to None so that they aren't
                # saved.
                return None
            else:
                return os.path.extsep.join((basename + extension, opts.graph))

        fitter.plot_spectrum(make_plot_path("_plt"))
        fitter.plot_slice_fit(make_plot_path("_fit1"))
        fitter.plot_row_column_sums(make_plot_path("_fit2"))
        fitter.plot_adr(make_plot_path("_fit3"))
        fitter.plot_residuals(make_plot_path("_fit4"))
        fitter.plot_seeing(make_plot_path("_fit5"))
        fitter.plot_radial_profile(make_plot_path("_fit6"))
        fitter.plot_contours(make_plot_path("_fit7"))
