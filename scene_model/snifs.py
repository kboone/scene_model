# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from copy import deepcopy
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import leastsq

from . import config
from .utils import SceneModelException, extract_key
from .element import ConvolutionElement, PsfElement, Pixelizer, RealPixelizer
from .scene import SceneModel
from .models import GaussianMoffatPsfElement, \
    PointSource, \
    Background, \
    PolynomialBackground, \
    GaussianPsfElement, \
    ExponentialPowerPsfElement, \
    ChromaticExponentialPowerPsfElement, \
    GaussianSceneModel
from .fit import MultipleImageFitter
from .prior import MultivariateGaussianPrior

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np

# Import SNIFS libraries if available.
try:
    from ToolBox.Arrays import metaslice
    from ToolBox.Astro import Coords
    from ToolBox.Atmosphere import ADR
    from ToolBox import MPL
    import pySNIFS
except ImportError as e:
    print("WARNING: Unable to load SNIFS libraries! (%s)" % e.message)
    print("Some functionality will be disabled.")


rad_to_deg = 180. / np.pi

# Definitions of reasonable parameters for SNIFS images. These are used to flag
# potential issues with the fits, and to reject really bad fits.
MAX_POSITION_PRIOR_OFFSET = 1    # Max position offset wrt. prior [spx]
MAX_SEEING_PRIOR_OFFSET = 40     # Max seeing offset wrt. prior [%]
MAX_AIRMASS_PRIOR_OFFSET = 20    # Max airmass offset wrt. prior [%]
MAX_PARANG_PRIOR_OFFSET = 20     # Max parangle offset wrt. prior [deg]
MAX_POSITION = 6                 # Max position wrt. FoV center [spx]

MIN_SEEING = 0.3                 # Min reasonable seeing ['']
MAX_SEEING = 4.0                 # Max reasonable seeing ['']
MAX_AIRMASS = 4.                 # Max reasonable airmass


def read_pressure_and_temperature(header, mauna_kea_pressure=616.,
                                  mauna_kea_temperature=2.):
    """
    Read pressure [mbar] and temperature [C] from header (or use default
    Mauna-Kea values), and check value consistency.

    modified from libExtractStar.py
    """
    if header is None:
        return mauna_kea_pressure, mauna_kea_temperature

    pressure = header.get('PRESSURE', np.nan)
    if not 550 < pressure < 650:        # Non-standard pressure
        print("WARNING: non-standard pressure (%.0f mbar) updated to %.0f mbar"
              % (pressure, mauna_kea_pressure))
        if isinstance(header, dict):    # pySNIFS.SNIFS_cube.e3d_data_header
            header['PRESSURE'] = mauna_kea_pressure
        else:                           # True fits header, add comment
            header['PRESSURE'] = (mauna_kea_pressure,
                                  "Default Mauna Kea pressure [mbar]")
        pressure = mauna_kea_pressure

    temperature = header.get('TEMP', np.nan)
    if not -20 < temperature < 20:      # Non-standard temperature
        print("WARNING: non-standard temperature (%.0f C) updated to %.0f C" %
              (temperature, mauna_kea_temperature))
        if isinstance(header, dict):    # pySNIFS.SNIFS_cube.e3d_data_header
            header['TEMP'] = mauna_kea_temperature
        else:                           # True pyfits header, add comment
            header['TEMP'] = (mauna_kea_temperature,
                              "Default Mauna Kea temperature [C]")
        temperature = mauna_kea_temperature

    return pressure, temperature


def estimate_zenithal_parallactic(header):
    """
    Estimate zenithal distance [deg] and parallactic angle [deg] from header.

    modified from libExtractStar.py
    """
    ha, dec = Coords.altaz2hadec(
        header['ALTITUDE'], header['AZIMUTH'], phi=header['LATITUDE'],
        deg=True
    )

    zd, parangle = Coords.hadec2zdpar(
        ha, dec, phi=header['LATITUDE'], deg=True
    )

    return zd, parangle


def estimate_adr_parameters(header, return_dispersions=False):
    """
    Estimate ADR parameters delta and theta [rad] from the airmass and
    parallactic angle in a SNIFS fits header.
    """
    channel = header['CHANNEL']
    header_airmass = header['AIRMASS']
    header_parang = header['PARANG']

    if channel == 'B':
        dispersion_delta, dispersion_theta = [0.0173, 0.02882]
        ddelta_zp, ddelta_slope = [0.00766, -0.00734]
        dtheta_zp, dtheta_slope = [3.027, -0.554]   # deg
    elif channel == 'R':
        dispersion_delta, dispersion_theta = [0.0122, 0.02536]
        ddelta_zp, ddelta_slope = [0.00075, 0.04674]
        dtheta_zp, dtheta_slope = [4.447, 3.078]    # deg
    else:
        raise SceneModelException("Unknown channel %s" % channel)

    header_delta = np.tan(np.arccos(1. / header_airmass))
    header_theta = header_parang / rad_to_deg

    sinpar = np.sin(header_theta)
    cospar = np.cos(header_theta)

    correction_delta = ddelta_zp + ddelta_slope * sinpar
    correction_theta = (dtheta_zp + dtheta_slope * cospar) / rad_to_deg

    predicted_delta = header_delta + correction_delta
    predicted_theta = header_theta + correction_theta

    if return_dispersions:
        return predicted_delta, predicted_theta, dispersion_delta, \
            dispersion_theta
    else:
        return predicted_delta, predicted_theta


def build_adr_model(pressure, temperature, delta=None, theta=None,
                    reference_wavelength=config.reference_wavelength):
    """Load an ADR model for the PSF."""
    adr_model = ADR(pressure, temperature, lref=reference_wavelength,
                    delta=delta, theta=theta)

    return adr_model


def build_adr_model_from_header(header, **kwargs):
    """Build an ADR model from a SNIFS fits header"""
    pressure, temperature = read_pressure_and_temperature(header)
    delta, theta = estimate_adr_parameters(header)

    adr_model = build_adr_model(pressure, temperature, delta, theta, **kwargs)

    return adr_model


def cube_to_arrays(cube):
    """Convert a SNIFS cube into two 3d numpy arrays with the data and
    variance.

    Any invalid spaxels or ones that aren't present are set to np.nan in the
    cube.
    """
    size_x = np.max(cube.i) + 1
    size_y = np.max(cube.j) + 1

    data = np.zeros((cube.data.shape[0], size_x, size_y))
    data[...] = np.nan
    data[:, cube.i, cube.j] = cube.data

    var = np.zeros(data.shape)
    var[...] = np.nan
    var[:, cube.i, cube.j] = cube.var

    return data, var


def evaluate_power_law(coefficients, x):
    """Evaluate (curved) power-law: coefficients[-1] * x**(coefficients[-2] +
    coefficients[-3]*(x-1) + ...)

    Note that f(1) = coefficients[-1] = f(lref) with x = lbda/lref.
    """
    return coefficients[-1] * x**np.polyval(coefficients[:-1], x - 1)


def power_law_jacobian(coeffs, x):
    ncoeffs = len(coeffs)                               # M
    jac = np.empty((ncoeffs, len(x)), dtype=x.dtype)    # M×N
    jac[-1] = x**np.polyval(coeffs[:-1], x - 1)         # df/dcoeffs[-1]
    jac[-2] = coeffs[-1] * jac[-1] * np.log(x)          # df/dcoeffs[-2]
    for i in range(-3, -ncoeffs - 1, -1):
        jac[i] = jac[i + 1] * (x - 1)

    return jac                                          # M×N


def fit_power_law(x, y, deg=2, guess=None):
    import ToolBox.Optimizer as TO
    if guess is None:
        guess = [0.] * (deg - 1) + [-1., 2.]
    else:
        assert len(guess) == (deg + 1)

    model = TO.Model(evaluate_power_law, jac=power_law_jacobian)
    data = TO.DataSet(y, x=x)
    fit = TO.Fitter(model, data)
    lsqPars, msg = leastsq(fit.residuals, guess, args=(x,))

    if msg <= 4:
        return lsqPars
    else:
        raise SceneModelException("fit_power_law did not converge")


def write_pysnifs_spectrum(spectrum, path=None, header=None):
    """Write a pySNIFS spectrum to a fits file at the given path.

    This was adapted from extract_star.py

    This allows full header propagation (including comments) and covariance
    matrix storage.
    """
    assert not (spectrum.start is None or spectrum.step is None or
                spectrum.data is None)

    # Primary HDU: signal
    hdusig = fits.PrimaryHDU(spectrum.data, header=header)
    for key in ['EXTNAME', 'CTYPES', 'CRVALS', 'CDELTS', 'CRPIXS']:
        if key in hdusig.header:
            del(hdusig.header[key])  # Remove technical keys from E3D cube
    hdusig.header.set('CRVAL1', spectrum.start, after='NAXIS1')
    hdusig.header.set('CDELT1', spectrum.step, after='CRVAL1')

    hduList = fits.HDUList([hdusig])

    # 1st extension 'VARIANCE': variance
    if spectrum.has_var:
        hduvar = fits.ImageHDU(spectrum.var, name='VARIANCE')
        hduvar.header['CRVAL1'] = spectrum.start
        hduvar.header['CDELT1'] = spectrum.step

        hduList.append(hduvar)

    # 2nd (compressed) extension 'COVAR': covariance (lower triangle)
    if hasattr(spectrum, 'cov'):
        hducov = fits.ImageHDU(np.tril(spectrum.cov), name='COVAR')
        hducov.header['CRVAL1'] = spectrum.start
        hducov.header['CDELT1'] = spectrum.step
        hducov.header['CRVAL2'] = spectrum.start
        hducov.header['CDELT2'] = spectrum.step

        hduList.append(hducov)

    if path:                        # Save hduList to disk
        # writeto uses the clobber keyword for astropy < 1.3, and the overwrite
        # keyword for astropy >= 1.3. Unfortunately the CC is on astropy 1.0,
        # so we need to pick the right one.
        import astropy
        from distutils.version import LooseVersion
        astropy_version = LooseVersion(astropy.__version__)
        change_version = LooseVersion('1.3')
        if astropy_version >= change_version:
            hduList.writeto(path, output_verify='silentfix', overwrite=True)
        else:
            hduList.writeto(path, output_verify='silentfix', clobber=True)

    return hduList                  # For further handling if needed


def convert_radius_to_pixels(radius, seeing, spaxel_size):
    """Parse a radius in arcseconds/sigmas and convert it to a radius in
    pixels.

    If the radius is less than 0, then it is interpreted as a multiple of the
    seeing sigma. If the radius is greater than 0, then it is interpreted as a
    radius in arcseconds.
    """
    if radius > 0:
        # Explicit radius in arcseconds
        pass
    else:
        # Radius in multiples of the seeing sigma
        radius = -radius * seeing / 2.355

    # Convert the radius from arcseconds to spaxels
    radius /= spaxel_size

    return radius


def get_background_element(background_degree):
    """Set up a background element for a given degree.

    The background degree is specified as follows:
    - background_degree = 0: a flat background
    - background_degree >= 1: a polynomial background of the specified order.
    - background_degree = -1: no background.

    Note that the background elements returned by this function are all applied
    after pixelization.
    """
    if background_degree == 0:
        # Flat background
        background_element = Background()
    elif background_degree > 0:
        # Polynomial background
        background_element = PolynomialBackground(
            background_degree=background_degree
        )
    elif background_degree == -1:
        # No background
        background_element = None
    else:
        raise SceneModelException("Unknown background_degree %d!" %
                                  background_degree)

    return background_element


class SnifsAdrElement(ConvolutionElement):
    """An element that applies atmospheric differential refraction to a scene.
    """
    def __init__(self, adr_model=None, spaxel_size=None, pressure=None,
                 temperature=None, **kwargs):
        super(SnifsAdrElement, self).__init__(**kwargs)

        if adr_model is not None:
            self.set_adr(adr_model, spaxel_size)
        elif pressure is not None:
            self.load_adr(pressure, temperature, spaxel_size)
        else:
            # Will be set up later.
            self.adr_model = None
            self.spaxel_size = None

    def _setup_parameters(self):
        """Setup ADR parameters

        The ADR model has delta and theta parameters that determine the
        wavelength variation of the position. We also need to specify the
        position at a reference wavelength to set the zeropoint of the model.
        """
        self._add_parameter('wavelength', None, (None, None), 'WAVE',
                            'wavelength [A]', fixed=True, apply_prefix=False)

        self._add_parameter('adr_delta', 1., (0., 100.), 'DELTA',
                            'ADR delta parameter')
        self._add_parameter('adr_theta', 0., (None, None), 'THETA',
                            'ADR theta parameter')

        # Add the shifts introduced by the ADR as explicit parameters so that
        # we can pull them out in other places.
        self._add_parameter('adr_shift_x', None, (None, None), 'ADRX',
                            'ADR shift in the X direction', derived=True)
        self._add_parameter('adr_shift_y', None, (None, None), 'ADRY',
                            'ADR shift in the Y direction', derived=True)

    def _calculate_derived_parameters(self, parameters):
        """Calculate the shifts introduced by the ADR"""
        p = parameters

        if p['wavelength'] is None:
            raise SceneModelException("Must set wavelength for %s!" %
                                      type(self))
        if self.adr_model is None:
            raise SceneModelException("Must setup the ADR model for %s!" %
                                      type(self))

        adr_scale = (self.adr_model.get_scale(p['wavelength']) /
                     self.spaxel_size)
        shift_x = p['adr_delta'] * np.sin(p['adr_theta']) * adr_scale
        shift_y = -p['adr_delta'] * np.cos(p['adr_theta']) * adr_scale

        p['adr_shift_x'] = shift_x
        p['adr_shift_y'] = shift_y

        # Update parameters from the superclass
        parent_parameters = super(SnifsAdrElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, adr_shift_x,
                          adr_shift_y, **kwargs):
        """Apply a translation to the model in Fourier space"""
        shift_fourier = np.exp(-1j * (adr_shift_x * kx + adr_shift_y * ky))

        return shift_fourier

    def set_parameters(self, update_derived=True, **kwargs):
        """Update the delta and theta parameters of the ADR model."""
        for key, value in kwargs.items():
            if key == 'adr_delta':
                self.adr_model.set_param(delta=value)
            elif key == 'adr_theta':
                self.adr_model.set_param(theta=value)

        # Do the standard set_parameters
        super(SnifsAdrElement, self).set_parameters(
            update_derived=update_derived, **kwargs
        )

    def set_adr(self, adr_model, spaxel_size):
        """Set the ADR model to one that has already been created."""
        # Make sure that we passed both the adr model and the spaxel size
        if adr_model is None or spaxel_size is None:
            raise SceneModelException(
                "If specifying a previously set up adr model, must also "
                "specify the spaxel size."
            )

        self.adr_model = adr_model
        self.spaxel_size = spaxel_size

    def load_adr(self, pressure, temperature, spaxel_size):
        """Load an ADR model for the PSF."""
        if pressure is None or temperature is None or spaxel_size is None:
            raise SceneModelException(
                "Must specify all of pressure, temperature and spaxel_size."
            )

        adr_model = build_adr_model(pressure, temperature,
                                    delta=self['adr_delta'],
                                    theta=self['adr_theta'])

        self.adr_model = adr_model
        self.spaxel_size = spaxel_size

    def from_fits_header(cls, header, spaxel_size, **kwargs):
        """Load the ADR model from a SNIFS fits header.

        This only uses header parameters that come directly from the
        instrument, and can be used for any SNIFS cube.
        """
        adr_model = build_adr_model_from_header(header)
        element = cls(adr_model=adr_model, spaxel_size=spaxel_size, **kwargs)

        return element


class SnifsClassicPsfElement(GaussianMoffatPsfElement):
    """Model of the PSF as implemented in the pipeline.

    Don't use this class directly, subclass it instead with the following
    parameters set.
    """
    def __init__(self, exposure_time, *args, **kwargs):
        """Initialize the PSF. The PSF parameters depend on the exposure time.
        """
        super(SnifsClassicPsfElement, self).__init__(*args, **kwargs)

        if exposure_time < 15:
            # Short exposure PSF
            self.beta0 = 1.395
            self.beta1 = 0.415
            self.sigma0 = 0.56
            self.sigma1 = 0.2
            self.eta0 = 0.6
            self.eta1 = 0.16
        elif exposure_time > 15:
            # Long exposure PSF
            self.beta0 = 1.685
            self.beta1 = 0.345
            self.sigma0 = 0.545
            self.sigma1 = 0.215
            self.eta0 = 1.04
            self.eta1 = 0.00

    def _setup_parameters(self):
        super(SnifsClassicPsfElement, self)._setup_parameters()
        self._modify_parameter('sigma', derived=True)
        self._modify_parameter('beta', derived=True)
        self._modify_parameter('eta', derived=True)

    def _calculate_derived_parameters(self, parameters):
        """Fix sigma, beta and eta to all be functions of alpha"""
        p = parameters

        # Derive all of the Gaussian and Moffat parameters from alpha
        p['beta'] = self.beta0 + self.beta1 * p['alpha']
        p['sigma'] = self.sigma0 + self.sigma1 * p['alpha']
        p['eta'] = self.eta0 + self.eta1 * p['alpha']

        # Update parameters from the superclass
        parent_parameters = super(SnifsClassicPsfElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


class ChromaticSnifsClassicPsfElement(SnifsClassicPsfElement):
    """A chromatic SnifsClassicPsfElement with seeing dependence on wavelength.

    The PSF alpha parameter takes the following form:

        x = wave / ref_wave
        alpha = A[-1] * x**(A[-2] + A[-3]*(x-1) + A[-4]*(x-1)**2 + ...)

    The number of terms in the polynomial power term is determined by
    alpha_degree. The default of alpha_degree=2 gives:

        alpha = A2 * x**(A1 + A0*(x-1))
    """
    def __init__(self, alpha_degree=2, *args, **kwargs):
        """Initialize the PSF."""
        self.alpha_degree = alpha_degree

        super(ChromaticSnifsClassicPsfElement, self).__init__(*args, **kwargs)

    def _setup_parameters(self):
        super(ChromaticSnifsClassicPsfElement, self)._setup_parameters()

        # Alpha is now a derived parameter
        self._modify_parameter('alpha', derived=True)

        self._add_parameter('wavelength', None, (None, None), 'WAVE',
                            'wavelength [A]', fixed=True, apply_prefix=False)

        # Powerlaw polynomial parameters
        for i in range(self.alpha_degree):
            self._add_parameter('A%d' % i, 0., (None, None), 'A%d' % i,
                                'Alpha coeff. a%d' % i)

        # Reference width parameter
        deg = self.alpha_degree
        self._add_parameter('A%d' % deg, 2.4, (0.01, 30.), 'A%d' % deg,
                            'Alpha coeff. a%d' % deg)

    def _calculate_derived_parameters(self, parameters):
        """Calculate the seeing width parameter using a power-law in wavelength
        """
        p = parameters

        wavelength = p['wavelength']
        alpha_params = [p['A%d' % i] for i in range(self.alpha_degree + 1)]

        # Ensure that all required variables have been set properly.
        if wavelength is None:
            raise SceneModelException("Must set wavelength!")

        # Calculate the alpha parameters as a function of wavelength.
        x = wavelength / config.reference_wavelength
        alpha = evaluate_power_law(alpha_params, x)

        # Add in the new parameters
        p['alpha'] = alpha

        # Update parameters from the superclass
        parent_parameters = super(ChromaticSnifsClassicPsfElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


class TrackingEllipticityPsfElement(PsfElement):
    """A PsfElement that represents non-chromatic ellipticity due to effects
    like tracking or wind-shake

    Effects like bad tracking and windshake effectively convolve the PSF with a
    Gaussian-like profile that is wavelength-independent. In practice, for
    SNIFS, we find that these effects mostly occur in one specific direction
    (typically the Y-direction), and the Gaussian in the convolution has a much
    longer major-axis compared to the minor-axis. We are unable to measure both
    the minor axis and the seeing separately with SNIFS, and we assume that the
    minor axis is infinitesimally small in order to proceed. The major axis is
    then defined by two points in Cartesian space that specify the width and
    angle. Angles are difficult for fitters as they lead to singularities when
    the width is near zero, so we fit the ellipticity with the two points in
    Cartesian space directly.

    Note that this definition means that flipping the signs of both the X and Y
    coordinates leads to the same model, so there is a degeneracy. The
    ellipticity is almost always aligned with one of the axes, so we don't want
    to put bounds near the axes. To handle this, we initialize the fitter at
    (0.1, 0.1) which almost always means that the fitter ends up either
    choosing the (+1, 0) direction or the (0, +1) direction.
    """
    def _setup_parameters(self):
        self._add_parameter('ellipticity_x', 0.1, (-30., 30.), 'ELLX',
                            'Ellipticity coordinate in X direction')
        self._add_parameter('ellipticity_y', 0.1, (-30., 30.), 'ELLY',
                            'Ellipticity coordinate in Y direction')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, ellipticity_x,
                          ellipticity_y, **kwargs):
        # Infinitesimally small minor axis
        rho = 0.99

        gaussian = np.exp(-0.5 * (
            kx**2 * ellipticity_x**2 +
            ky**2 * ellipticity_y**2 +
            2. * kx * ky * rho * ellipticity_x * ellipticity_y
        ))

        return gaussian


class SnifsClassicSeeingPrior(MultivariateGaussianPrior):
    """Add a prior on the seeing of this image to the model.

    This was adapted from libExtractStar.py. This only works when seeing_degree
    is 2.
    """
    def __init__(self, channel, seeing, **kwargs):
        # Predict the alpha parameters given a seeing value and a channel.
        if channel == 'B':
            central_values = np.array([
                -0.134 * seeing + 0.5720,       # p0
                -0.134 * seeing - 0.0913,       # p1
                +3.474 * seeing - 1.3880        # p2
            ])
            covariance = np.array([
                [0.0402503, 0.01114101, -0.0031652],
                [0.01114101, 0.00722763, -0.00232724],
                [-0.0031652, -0.00232724, 0.07978374]
            ])
        elif channel == 'R':
            central_values = np.array([
                -0.0777 * seeing + 0.1741,      # p0
                -0.0202 * seeing - 0.3434,      # p1
                +3.4000 * seeing - 1.352        # p2
            ])
            covariance = np.array([
                [2.32247418e-03, 1.36444214e-05, -4.67068339e-03],
                [1.36444214e-05, 1.66489725e-03, -1.69993230e-03],
                [-4.67068377e-03, -1.69993216e-03, 9.80626270e-02]
            ])
        else:
            raise SceneModelException("Unknown channel '%s'" % channel)

        keys = ['A0', 'A1', 'A2']

        super(SnifsClassicSeeingPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )

    @classmethod
    def from_fits_header(cls, header, seeing, **kwargs):
        """Load the prior from a SNIFS fits header.

        seeing should be a GS seeing prediction from SNIFS.
        """
        channel = header['CHANNEL']
        prior = cls(channel, channel, seeing, **kwargs)

        return prior


class SnifsClassicEllipticityPrior(MultivariateGaussianPrior):
    """Add a prior on the ellipticity to a classic SNIFS model.

    This is adapted from libExtractStar.py
    """
    def __init__(self, channel, airmass, **kwargs):
        if channel == 'B':
            intercept_ell, slope_ell, dispersion_ell = [1.730, -0.323, 0.221]
            dispersion_xy = 0.041
        elif channel == 'R':
            intercept_ell, slope_ell, dispersion_ell = [1.934, -0.442, 0.269]
            dispersion_xy = 0.050
        else:
            raise SceneModelException("Unknown channel %s" % channel)

        predicted_ell = intercept_ell + slope_ell * airmass
        predicted_xy = 0.

        keys = ['ell', 'xy']
        central_values = [predicted_ell, predicted_xy]
        covariance = [[dispersion_ell**2, 0], [0, dispersion_xy**2]]

        super(SnifsClassicEllipticityPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )

    @classmethod
    def from_fits_header(cls, header, **kwargs):
        """Load the prior from a SNIFS fits header.

        This only uses header parameters that come directly from the
        instrument, and can be used for any SNIFS cube.
        """
        channel = header['CHANNEL']
        airmass = header['AIRMASS']

        prior = cls(channel, airmass, **kwargs)

        return prior


class SnifsFourierSeeingPrior(MultivariateGaussianPrior):
    """Add a prior on the seeing of this image to the model.

    The seeing_width parameter sets the seeing of the model. The relationship
    between this parameter and the seeing estimate from the SNIFS instrument
    was measured from a large set of extractions of standard stars on all
    photometric nights.
    """
    def __init__(self, channel, exposure_time, seeing, **kwargs):
        # Seeing prediction. This is the conversion between the GS seeing value
        # to the width of the model. I fit a linear relation to all of the
        # long exposure standard star exposures (short exposures don't have a
        # GS seeing prediction). There is no noticeable difference between the
        # channels or exposure times.
        predicted_width = -0.378 + 0.895 * seeing
        dispersion_width = 0.0727

        # Power prediction. The value here depends on whether it was a long or
        # short exposure and the channel. I grouped everything into short
        # (<1.5s), intermediate (1.5s to 15s) and long (>15s) exposures and
        # estimated from that. I realized afterwards that only long exposures
        # have a seeing estimate and therefore a prior applied. This could in
        # principle be pulled out so that the power prior is always applied,
        # but that isn't the case right now.
        if exposure_time < 1.5:
            if channel == 'B':
                predicted_power, dispersion_power = (-0.497, 0.103)
            elif channel == 'R':
                predicted_power, dispersion_power = (-0.498, 0.141)
        elif exposure_time < 15.:
            if channel == 'B':
                predicted_power, dispersion_power = (-0.432, 0.165)
            elif channel == 'R':
                predicted_power, dispersion_power = (-0.418, 0.187)
        else:
            if channel == 'B':
                predicted_power, dispersion_power = (-0.394, 0.068)
            elif channel == 'R':
                predicted_power, dispersion_power = (-0.334, 0.089)

        # Seeing powerlaw power
        keys = ['seeing_ref_width', 'seeing_ref_power']
        central_values = [predicted_width, predicted_power]
        covariance = [[dispersion_width**2, 0], [0, dispersion_power**2]]

        super(SnifsFourierSeeingPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )

    @classmethod
    def from_fits_header(cls, header, seeing, **kwargs):
        """Load the prior from a SNIFS fits header.

        seeing should be a GS seeing prediction from SNIFS.
        """
        channel = header['CHANNEL']
        efftime = header['EFFTIME']
        prior = cls(channel, efftime, seeing, **kwargs)

        return prior


class SnifsFourierEllipticityPrior(MultivariateGaussianPrior):
    """Add a prior on the ellipticity to the model.

    The ellipticity is primarily affected by the airmass. I determined a
    relationship for this from a set of standard stars. Note that this relation
    should really only be applied to long exposures, short exposures have a
    very different dependence. It doesn't really matter because the short
    exposures are so bright that they blow away the prior anyway, but I should
    figure out how to deal with this.
    """
    def __init__(self, channel, airmass, **kwargs):
        if channel == 'B':
            intercept_x, slope_x, dispersion_x = [+0.600, -0.179, 0.222]
            intercept_y, slope_y, dispersion_y = [-0.110, +0.150, 0.147]
        elif channel == 'R':
            intercept_x, slope_x, dispersion_x = [+0.576, -0.185, 0.223]
            intercept_y, slope_y, dispersion_y = [-0.123, +0.166, 0.146]
        else:
            raise SceneModelException("Unknown channel %s" % channel)

        predicted_ell_x = intercept_x + slope_x * airmass
        predicted_ell_y = intercept_y + slope_y * airmass

        keys = ['ellipticity_x', 'ellipticity_y']
        central_values = [predicted_ell_x, predicted_ell_y]
        covariance = [[dispersion_x**2, 0], [0, dispersion_y**2]]

        super(SnifsFourierEllipticityPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )

    @classmethod
    def from_fits_header(cls, header, **kwargs):
        """Load the prior from a SNIFS fits header.

        This only uses header parameters that come directly from the
        instrument, and can be used for any SNIFS cube.
        """
        channel = header['CHANNEL']
        airmass = header['AIRMASS']

        prior = cls(channel, airmass, **kwargs)

        return prior


class SnifsAdrPrior(MultivariateGaussianPrior):
    """Add a prior on the ADR to the model.

    The values here were determined by Yannick in the "faint standard star
    ad-hoc analysis (adr.py and runaway.py)". airmass and parang should be
    values from a fits header, and can be automatically read with
    from_fits_header.
    """
    def __init__(self, center_delta, center_theta, dispersion_delta,
                 dispersion_theta, **kwargs):
        keys = ['adr_delta', 'adr_theta']
        central_values = [center_delta, center_theta]
        covariance = [[dispersion_delta**2, 0], [0, dispersion_theta**2]]

        super(SnifsAdrPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )

    @classmethod
    def from_fits_header(cls, header, **kwargs):
        """Load the prior from a SNIFS fits header.

        This only uses header parameters that come directly from the
        instrument, and can be used for any SNIFS cube.
        """
        center_delta, center_theta, dispersion_delta, dispersion_theta = \
            estimate_adr_parameters(header, return_dispersions=True)

        prior = cls(center_delta, center_theta, dispersion_delta,
                    dispersion_theta, **kwargs)

        return prior


class SnifsFourierSceneModel(SceneModel):
    """A model of the PSF for the SNIFS IFU.

    This model was empirically constructed, and it contains:
    - a Gaussian instrumental core
    - a wide tail for the instrumental core
    - a Kolmogorov-like seeing term (with a variable exponent that doesn't
    quite match the Kolmogorov prediction).
    - a term to capture the tracking and wind-shake jitter seen in SNIFS
    exposures.
    - a background

    The model can optionally be specified with wavelength dependence, in which
    case an ADR model is used to determine the position of the point source as
    a function of wavelength and a power law is used to determine the seeing
    variation with wavelength.

    The background degree specifies which background model to use. See
    get_background_element for details.
    """
    def __init__(self, image=None, variance=None,
                 use_empirical_parameters=True, wavelength_dependence=False,
                 background_degree=0, adr_model=None, spaxel_size=None,
                 pressure=None, temperature=None, **kwargs):
        """Build the elements of the PSF model.

        If use_empirical_parameters is True, then the instrumental parameters
        and the seeing power are set to empirically fitted parameters
        determined from the whole SNfactory standard star dataset.
        """
        self.wavelength_dependence = wavelength_dependence

        elements = []

        # Point source for the supernova
        point_source_element = PointSource()
        elements.append(point_source_element)

        # Gaussian instrumental core
        inst_core_element = GaussianPsfElement(
            prefix='inst_core',
            fits_keyword_prefix='C',
            fits_description_prefix='Instrumental core',
        )
        elements.append(inst_core_element)

        # Exponential power function for instrumental wings
        inst_wings_element = ExponentialPowerPsfElement(
            prefix='inst_wings',
            fits_keyword_prefix='I',
            fits_description_prefix='Instrumental wings'
        )
        elements.append(inst_wings_element)

        # ADR
        if wavelength_dependence:
            adr_element = SnifsAdrElement(
                adr_model=adr_model, spaxel_size=spaxel_size,
                pressure=pressure, temperature=temperature
            )
            self.adr_element = adr_element
            elements.append(adr_element)

            # Rename the center position parameters to specify that they are
            # the positions at the reference wavelength.
            point_source_element.rename_parameter('center_x', 'ref_center_x')
            point_source_element.rename_parameter('center_y', 'ref_center_y')

        # Seeing
        if wavelength_dependence:
            seeing_class = ChromaticExponentialPowerPsfElement
        else:
            seeing_class = ExponentialPowerPsfElement
        seeing_element = seeing_class(
            prefix='seeing',
            fits_keyword_prefix='S',
            fits_description_prefix='Seeing'
        )
        elements.append(seeing_element)

        # Tracking
        tracking_element = TrackingEllipticityPsfElement()
        elements.append(tracking_element)

        # Pixelize
        elements.append(Pixelizer)

        # Background. The background is a slowly varying function, and doesn't
        # need to be convolved with the PSF or with the pixel. We apply it
        # after pixelization.
        background_element = get_background_element(background_degree)
        if background_element is not None:
            elements.append(background_element)

        if wavelength_dependence:
            # Several elements need access to the wavelength parameter. Make
            # sure that it is shared across them.
            shared_parameters = ['wavelength']
        else:
            shared_parameters = []

        super(SnifsFourierSceneModel, self).__init__(
            elements=elements,
            image=image,
            variance=variance,
            shared_parameters=shared_parameters,
            **kwargs
        )

        if use_empirical_parameters:
            self.fix(
                inst_core_sigma_x=0.2376935 * np.sqrt(2),
                inst_core_sigma_y=0.154173 * np.sqrt(2),
                inst_core_rho=0.,
                inst_wings_power=1.07,
                inst_wings_width=0.209581,
                seeing_power=1.52,
            )

    def get_fits_header_items(self, prefix=config.default_fits_prefix):
        """Get a list of parameters to put into the fits header.

        Each entry has the form (fits_keyword, value, description).

        We add in all additional keywords here that are necessary to recreate
        the model from the fits header.
        """
        # Get the default list of keywords from the model elements.
        fits_list = super(SnifsFourierSceneModel, self)\
            .get_fits_header_items(prefix=prefix)

        # Add in additional keys
        fits_list.append((prefix + 'WDEP', self.wavelength_dependence,
                          'Wavelength dependent scene model'))
        if self.wavelength_dependence:
            fits_list.append((prefix + 'SPXSZ', self.adr_element.spaxel_size,
                              'Spaxel size'))

        return fits_list

    @classmethod
    def from_fits_header(cls, fits_header, prefix=config.default_fits_prefix):
        """Load the scene model from a fits header.

        The header keys are assumed to all have prefix before in the
        actual key used in the header (extract_star prefixes everything with
        ES_).
        """
        wavelength_dependence = fits_header[prefix + 'WDEP']
        background_degree = fits_header[prefix + 'SDEG']

        if wavelength_dependence:
            spaxel_size = fits_header[prefix + 'SPXSZ']
            pressure, temperature = read_pressure_and_temperature(fits_header)
            scene_model = cls(
                use_empirical_parameters=False,
                wavelength_dependence=True,
                background_degree=background_degree,
                spaxel_size=spaxel_size,
                pressure=pressure,
                temperature=temperature
            )
        else:
            scene_model = cls(
                use_empirical_parameters=False,
                wavelength_dependence=False,
                background_degree=background_degree
            )

        # Read in the rest of the parameters
        scene_model.load_fits_header(fits_header, prefix=prefix,
                                     skip_parameters=['wavelength'])

        return scene_model

    def _get_center_position(self, parameters):
        if self.wavelength_dependence:
            # Add in the ADR shift.
            center_x = parameters['ref_center_x'] + parameters['adr_shift_x']
            center_y = parameters['ref_center_y'] + parameters['adr_shift_y']
            return (center_x, center_y)
        else:
            # Use the default.
            return super(SnifsFourierSceneModel, self)._get_center_position(
                parameters
            )


class SnifsClassicSceneModel(SceneModel):
    """A scene model for the SNIFS IFU using a Gaussian + Moffat PSF.

    This implements the original PSF model for SNIFS which is the sum of a
    Gaussian and a Moffat profile.

    The model can optionally be specified with wavelength dependence, in which
    case an ADR model is used to determine the position of the point source as
    a function of wavelength and a power law is used to determine the seeing
    variation with wavelength.

    Note that extract_star's implementation worked only in real space while
    this implementation uses Fourier transforms. Technically, we should get a
    better result if we use the Pixelizer that convolves with the pixel in
    Fourier space, but we need to use RealPixelizer which simply sums up the
    subpixel values directly if we want to match the behavior of extract_star.

    To get behavior that matches what is currently in the pipeline, the
    subsampling needs to match (=3 by default in the pipeline).

    The background degree specifies which background model to use. See
    get_background_element for details.
    """
    def __init__(self, image=None, variance=None, exposure_time=None,
                 alpha_degree=2, wavelength_dependence=False,
                 background_degree=0, adr_model=None, spaxel_size=None,
                 pressure=None, temperature=None, **kwargs):
        """Build the elements of the PSF model."""
        elements = []

        if exposure_time is None:
            raise SceneModelException(
                "Must specify exposure time for SnifsClassicSceneModel"
            )

        # Point source for the supernova
        point_source_element = PointSource()
        elements.append(point_source_element)

        # Gaussian + Moffat PSF.
        if wavelength_dependence:
            psf_element = ChromaticSnifsClassicPsfElement(alpha_degree,
                                                          exposure_time)
        else:
            psf_element = SnifsClassicPsfElement(exposure_time)
        elements.append(psf_element)

        # ADR
        if wavelength_dependence:
            adr_element = SnifsAdrElement(
                adr_model=adr_model, spaxel_size=spaxel_size,
                pressure=pressure, temperature=temperature
            )
            elements.append(adr_element)

            # Rename the center position parameters to specify that they are
            # the positions at the reference wavelength.
            point_source_element.rename_parameter('center_x', 'ref_center_x')
            point_source_element.rename_parameter('center_y', 'ref_center_y')

        # Pixelize
        elements.append(RealPixelizer)

        # Background. The background is a slowly varying function, and doesn't
        # need to be convolved with the PSF or with the pixel. We apply it
        # after pixelization.
        background_element = get_background_element(background_degree)
        if background_element is not None:
            elements.append(background_element)

        if wavelength_dependence:
            shared_parameters = ['wavelength']
        else:
            shared_parameters = []

        super(SnifsClassicSceneModel, self).__init__(
            elements=elements,
            image=image,
            variance=variance,
            shared_parameters=shared_parameters,
            **kwargs
        )

    def _get_center_position(self, parameters):
        if self.wavelength_dependence:
            # Add in the ADR shift.
            center_x = parameters['ref_center_x'] + parameters['adr_shift_x']
            center_y = parameters['ref_center_y'] + parameters['adr_shift_y']
            return (center_x, center_y)
        else:
            # Use the default.
            return super(SnifsClassicSceneModel, self)._get_center_position(
                parameters
            )


class SnifsCubeFitter(object):
    """Fitter to fit a SNIFS cube.

    The fit happens in several parts:
    - Split the cube into several metaslices.
    - Fit a Gaussian to each metaslice to determine the initial position.
    - Fit the PSF to each metaslice to determine the initial PSF parameters.
    - Estimate global parameters for the PSF.
    - Fit a full chromatic model to the metaslices.
    - Extract the cube using the full chromatic model.
    """
    def __init__(self, path, psf="fourier", background_degree=0,
                 subsampling=config.default_subsampling,
                 border=config.default_border, least_squares=False,
                 prior_scale=0., seeing_prior=None, accountant=None,
                 verbosity=0):
        """Initialize the fitter.

        path is the path to the fits cube that will be extracted.
        """
        self.verbosity = verbosity

        self._read_cube(path)
        self._setup_psf(psf, prior_scale=prior_scale,
                        seeing_prior=seeing_prior)

        self.background_degree = background_degree
        self.subsampling = subsampling
        self.border = border
        self.least_squares = least_squares
        self.accountant = accountant

        # Meta cube variables. These are set in fit_metaslices_2d
        self.meta_cube = None
        self.metaslice_info = None
        self.metaslice_guesses = None

        # 3D fit results
        self.fitter_3d = None
        self.fit_parameters = None
        self.fit_uncertainties = None
        self.fit_scene_model = None
        self.meta_cube_model = None
        self.reference_seeing = None

        # Extraction
        self.extraction = None
        self.extraction_method = None
        self.extraction_radius = None
        self.point_source_spectrum = None
        self.background_spectrum = None

        self.print_cube_info()

    def print_message(self, message, minimum_verbosity=0):
        """Print message if verbosity level >= minimum_verbosity."""
        if self.verbosity >= minimum_verbosity:
            print(message)

    def _read_cube(self, path):
        """Read a SNIFS cube that is stored in a fits file.

        The cube can be either in Euro3D or Fits3D format. We detect the format
        and handle either transparently here.
        """
        # Load the cube
        self.print_message("Opening datacube %s" % path)
        try:
            try:
                # Check what kind of cube we have. pySNIFS raises a ValueError
                # if the cube is a Fits3d file that we tried to load as Euro3D.
                header = fits.getheader(path, 1)
                cube = pySNIFS.SNIFS_cube(e3d_file=path)
            except ValueError:
                # We have a 3D fits cube
                header = fits.getheader(path, 0)  # Primary extension
                cube = pySNIFS.SNIFS_cube(fits3d_file=path)
        except IOError:
            raise SceneModelException("Cannot access file '%s'" % path)

        # Update the parallactic angle in the header if it isn't there already
        if 'PARANG' not in header:
            print("WARNING: Computing PARANG from header ALTITUDE, AZIMUTH "
                  "and LATITUDE.")
            _, header['PARANG'] = estimate_zenithal_parallactic(header)

        # Make sure that the pressure and temperature in the header are valid,
        # and update them if necessary.
        pressure, temperature = read_pressure_and_temperature(header)

        # Only B and R channel extractions are implemented. Make sure that we
        # are on one of these.
        channel = header['CHANNEL'][0].upper()  # 'B' or 'R'
        if channel not in ('B', 'R'):
            raise SceneModelException(
                "Input datacube %s has no valid CHANNEL keyword (%s)" %
                (path, header['CHANNEL'])
            )

        # Set up an ADR model
        adr_model = build_adr_model_from_header(header)

        # Save the results
        self.path = path
        self.header = header
        self.cube = cube

        self.channel = channel
        self.adr_model = adr_model
        self.original_adr_model = deepcopy(adr_model)

    def _setup_psf(self, psf, prior_scale=0., seeing_prior=None):
        """Set up a PSF to be fit. All PSF specific configuration should happen
        here.
        """
        scene_model_kwargs = {}

        # Pick the PSF to fit
        if psf == "fourier":
            scene_model_class = SnifsFourierSceneModel
            seeing_degree = 1
            seeing_key = 'seeing_width'
            seeing_powerlaw_keys = ['seeing_ref_power', 'seeing_ref_width']
            global_fit_keys = ['ellipticity_x', 'ellipticity_y']
        elif psf == "classic":
            scene_model_class = SnifsClassicSceneModel
            seeing_degree = 2
            seeing_key = 'alpha'
            seeing_powerlaw_keys = ['A%d' % i for i in
                                    range(seeing_degree + 1)]
            global_fit_keys = ['ell', 'xy']

            scene_model_kwargs['exposure_time'] = self.header['EFFTIME']
            scene_model_kwargs['alpha_degree'] = seeing_degree
        else:
            raise SceneModelException("Unknown PSF %s!" % psf)

        # Build a list of priors to apply
        individual_priors = []
        global_priors = []
        if prior_scale > 0.:
            # ADR prior is always applied for global fits
            adr_prior = SnifsAdrPrior.from_fits_header(self.header)
            global_priors.append(adr_prior)

            # Ellipticity prior is always applied for both individual and
            # global fits.
            if psf == "fourier":
                ell_prior = SnifsFourierEllipticityPrior.from_fits_header(
                    self.header
                )
            elif psf == "classic":
                ell_prior = SnifsClassicEllipticityPrior.from_fits_header(
                    self.header
                )
            individual_priors.append(ell_prior)
            global_priors.append(ell_prior)

            # Seeing prior is applied only for global fits if seeing_prior is
            # set.
            if seeing_prior is not None:
                if psf == "fourier":
                    seeing_prior_object = SnifsFourierSeeingPrior.\
                        from_fits_header(self.header, seeing_prior)
                elif psf == "classic":
                    seeing_prior_object = SnifsClassicSeeingPrior.\
                        from_fits_header(self.header, seeing_prior)
                global_priors.append(seeing_prior_object)
                self.seeing_prior_object = seeing_prior_object

        self.psf = psf
        self.prior_scale = prior_scale
        self.seeing_prior = seeing_prior
        self.individual_priors = individual_priors
        self.global_priors = global_priors
        self.scene_model_class = scene_model_class
        self.seeing_degree = seeing_degree
        self.seeing_key = seeing_key
        self.seeing_powerlaw_keys = seeing_powerlaw_keys
        self.global_fit_keys = global_fit_keys
        self.scene_model_kwargs = scene_model_kwargs

    def print_cube_info(self):
        """Print information about the cube and the model that will be fit to
        it
        """
        cube = self.cube
        header = self.header

        self.print_message("Cube %s [%s]: [%.2f-%.2f], %d spaxels" %
                           (os.path.basename(self.path),
                            'E3D' if cube.from_e3d_file else '3D',
                            cube.lbda[0], cube.lbda[-1], cube.nlens), 1)

        self.print_message("  Object: %s, Efftime: %.1fs, Airmass: %.2f" %
                           (self.object_name, header['EFFTIME'],
                            header['AIRMASS']))

        self.print_message("  PSF: %s, subsampling: %d, border: %d" %
                           (self.psf, self.subsampling, self.border))

        if self.background_degree > 0:
            self.print_message("  Sky: polynomial, degree %d" %
                               self.background_degree)
        elif self.background_degree == 0:
            self.print_message("  Sky: uniform")
        elif self.background_degree == -1:
            self.print_message("  Sky: none")
        else:
            raise SceneModelException("Invalid sky degree '%d'" %
                                      self.background_degree)

        self.print_message("  Channel: '%s'" % self.channel)
        self.print_message("  Fit method: %s" % (
            'least-squares' if self.least_squares else 'chi2'))

        # Initial ADR model
        self.print_message(
            "  ADR guess: delta=%.2f (airmass=%.2f), theta=%.1f deg" % (
                self.adr_model.delta, self.adr_model.get_airmass(),
                self.adr_model.theta * rad_to_deg
            ), 1)

    def fit_metaslices_2d(self, num_meta_slices=12):
        """Fit the meta slices.

        This step produces initial guesses for the parameters of the 3D fit.
        """
        self.print_message("Meta-slice 2D-fitting...")

        # Create the metaslices
        slices = metaslice(self.cube.nslice, num_meta_slices, trim=10)
        if self.cube.from_e3d_file:
            meta_cube = pySNIFS.SNIFS_cube(e3d_file=self.path, slices=slices)
        else:
            meta_cube = pySNIFS.SNIFS_cube(fits3d_file=self.path,
                                           slices=slices)
        meta_cube.x = meta_cube.i - np.max(meta_cube.i) / 2.
        meta_cube.y = meta_cube.j - np.max(meta_cube.j) / 2.
        meta_cube_data, meta_cube_var = cube_to_arrays(meta_cube)

        self.print_message(
            "  Meta-slices before selection: %d from %.2f to %.2f by %.2f A" %
            (num_meta_slices, meta_cube.lstart, meta_cube.lend,
             meta_cube.lstep)
        )

        # Fit positions with Gaussian psf models, and then fit the full PSF to
        # each image individually.
        valid = []
        gaussian_scene_models = []
        individual_scene_models = []
        individual_uncertainties = []

        for idx in range(len(meta_cube_data)):
            try:
                if self.least_squares:
                    var = None
                else:
                    var = meta_cube_var[idx]

                # Crude Gaussian fit for initial position.
                gaussian_model = GaussianSceneModel(
                    meta_cube_data[idx], var, subsampling=1,
                    border=0
                )
                gaussian_model.fix(rho=0.)
                gaussian_model.fit()

                initial_position_x = gaussian_model['center_x']
                initial_position_y = gaussian_model['center_y']

                # Fit the full PSF model to each image individually, starting
                # at the Gaussian fit location.
                full_model = self.scene_model_class(
                    image=meta_cube_data[idx],
                    variance=var,
                    priors=self.individual_priors,
                    prior_scale=self.prior_scale,
                    wavelength_dependence=False,
                    background_degree=self.background_degree,
                    subsampling=self.subsampling,
                    border=self.border,
                    **self.scene_model_kwargs
                )
                full_model.set_parameters(
                    center_x=initial_position_x,
                    center_y=initial_position_y,
                )

                full_model.fit()

                uncertainties = full_model.calculate_uncertainties()

                if self.verbosity >= 2:
                    print("")
                    full_model.print_fit_info("Fit to metaslice %d" % idx,
                                              uncertainties=uncertainties)
                    print("")

                gaussian_scene_models.append(gaussian_model)
                individual_scene_models.append(full_model)
                individual_uncertainties.append(uncertainties)
                valid.append(True)
            except SceneModelException as e:
                print("    Fit on slice %d failed with error %s" % (idx, e))
                gaussian_scene_models.append(None)
                individual_scene_models.append(None)
                individual_uncertainties.append(None)
                valid.append(False)

        valid = np.array(valid)
        if not np.all(valid):
            print("  WARNING: %d metaslices discarded due to invalid fits" %
                  np.sum(~valid))

        # Guess the reference positions for the ADR using the individual fits.
        x_centers = extract_key(individual_scene_models, 'center_x')
        y_centers = extract_key(individual_scene_models, 'center_y')

        x_center_errs = extract_key(individual_uncertainties, 'center_x')
        y_center_errs = extract_key(individual_uncertainties, 'center_y')

        # Back-propagate positions to the reference wavelength, and take the
        # median value.
        valid_xrefs, valid_yrefs = self.original_adr_model.refract(
            x_centers[valid], y_centers[valid], meta_cube.lbda[valid],
            backward=True, unit=self.cube.spxSize
        )
        xrefs = np.empty(num_meta_slices)
        yrefs = np.empty(num_meta_slices)
        xrefs.fill(np.nan)
        yrefs.fill(np.nan)
        xrefs[valid] = valid_xrefs
        yrefs[valid] = valid_yrefs

        xref = np.median(valid_xrefs)
        yref = np.median(valid_yrefs)

        # Cut out any images that didn't find PSFs at the right position.
        r = np.hypot(valid_xrefs - xref, valid_yrefs - yref)
        rmax = 3 * 1.4826 * np.median(r)    # Robust to outliers (~3*NMAD)
        good = np.zeros(num_meta_slices, dtype=bool)
        good[valid] = (r <= rmax)           # Valid fit and reasonable position
        bad = np.zeros(num_meta_slices, dtype=bool)
        bad[valid] = (r > rmax)             # Valid fit but discarded position
        if bad.any():
            print("  WARNING: %d metaslices discarded after ADR selection" %
                  (len(np.nonzero(bad))))

        # Estimate the seeing coefficients. If we have a prior on the seeing
        # use that. If not, do a fit to the 2D slices.
        seeing_widths = extract_key(individual_scene_models, self.seeing_key)
        seeing_uncertainties = extract_key(individual_uncertainties,
                                           self.seeing_key)
        if self.prior_scale > 0. and self.seeing_prior:
            prior_predictions = self.seeing_prior_object.predicted_values
            seeing_powerlaw_guesses = [
                prior_predictions[i] for i in self.seeing_powerlaw_keys
            ]
        else:
            seeing_powerlaw_guesses = fit_power_law(
                meta_cube.lbda[good] / config.reference_wavelength,
                seeing_widths[good],
                self.seeing_degree
            )
        guess_seeing_widths = evaluate_power_law(
            seeing_powerlaw_guesses,
            meta_cube.lbda / config.reference_wavelength
        )

        # Estimate the positions. Here we used a clipped mean rather than a
        # median, following what extract_star did.
        if good.any():
            self.print_message("  %d/%d centroids found within %.2f spx of "
                               "(%.2f,%.2f)" % (len(xrefs[good]), len(xrefs),
                                                rmax, xref, yref), 1)
            ref_center_x_guess = xrefs[good].mean()
            ref_center_y_guess = yrefs[good].mean()
        else:
            raise ValueError('No position initial guess')

        # Build a table of the information for each slice.
        metaslice_info = Table({
            'wavelength': meta_cube.lbda,
            'center_x': x_centers,
            'center_y': y_centers,
            'center_x_uncertainty': x_center_errs,
            'center_y_uncertainty': y_center_errs,
            self.seeing_key: seeing_widths,
            '%s_uncertainty' % self.seeing_key: seeing_uncertainties,
            'guess_%s' % self.seeing_key: guess_seeing_widths,
            'guess_ref_center_x': xrefs,
            'guess_ref_center_y': yrefs,
            'guess_center_radius': np.ones(len(xrefs)) * rmax,
            'valid': valid,
            'good': good,
            'bad': bad,
            'gaussian_scene_model': gaussian_scene_models,
            'scene_model': individual_scene_models,
            'uncertainties': individual_uncertainties,
        })

        # Save guess information
        metaslice_guesses = {
            'ref_center_x': ref_center_x_guess,
            'ref_center_y': ref_center_y_guess,
            'adr_delta': self.original_adr_model.delta,
            'adr_theta': self.original_adr_model.theta,
        }
        for key, value in zip(self.seeing_powerlaw_keys,
                              seeing_powerlaw_guesses):
            metaslice_guesses[key] = value

        for key in self.global_fit_keys:
            # Take the median value of other parameters that will be fit
            # globally (eg: ellipticity)
            values = extract_key(individual_scene_models, key)
            guess_value = np.median(values[good])
            metaslice_guesses[key] = guess_value

        # For variables that we have priors on, override any guesses from the
        # data.
        for prior in self.global_priors:
            prior.update_initial_values(metaslice_guesses)

        # Save results
        self.meta_cube = meta_cube
        self.metaslice_info = metaslice_info
        self.metaslice_guesses = metaslice_guesses

    def write_log_2d(self, path):
        """Dump an informative text log about the PSF (metaslice) 2D-fit."""
        self.print_message("Producing 2D adjusted parameter logfile %s..." %
                           path)
        logfile = open(path, 'w')

        logfile.write('# cube    : %s   \n' % os.path.basename(self.path))
        logfile.write('# object  : %s   \n' % self.header["OBJECT"])
        logfile.write('# airmass : %.3f \n' % self.header["AIRMASS"])
        logfile.write('# efftime : %.3f \n' % self.header["EFFTIME"])

        first = True
        for row in self.metaslice_info:
            wavelength = row['wavelength']
            valid = row['valid']
            if not valid:
                # Fit failed, skip
                logfile.write("# Fit failed for wavelength %.2f" % wavelength)
                continue

            scene_model = row['scene_model']
            parameters = scene_model.parameters
            parameter_names = sorted(parameters.keys())
            uncertainties = row['uncertainties']

            if first:
                # Print a header.
                first = False

                header_str = "# wavelength"

                for key in parameter_names:
                    if key not in uncertainties:
                        continue
                    header_str += " %41s" % ("%s +/- d%s" % (key, key))

                if self.least_squares:
                    header_str += " %20s\n" % "RSS"
                else:
                    header_str += " %20s\n" % "chi2"
                logfile.write(header_str)

            # Print out each parameter
            row_str = "%12.2f" % wavelength
            for key in parameter_names:
                if key not in uncertainties:
                    continue
                row_str += " %20g %20g" % (parameters[key], uncertainties[key])
            row_str += " %20g\n" % scene_model.chi_square()

            logfile.write(row_str)

        logfile.close()

    def fit_metaslices_3d(self):
        self.print_message("Meta-slice 3D-fitting...")

        # Make sure that the meta cube has already been created. This is done
        # in fit_metaslices_2d which needs to be called before this function.
        if self.meta_cube is None:
            raise SceneModelException(
                "Must generate metaslices before calling fit_metaslices_3d"
            )

        # Model to use for the 3d fit
        scene_model = self.scene_model_class(
            wavelength_dependence=True,
            background_degree=self.background_degree,
            subsampling=self.subsampling,
            border=self.border,
            adr_model=self.adr_model,
            spaxel_size=self.cube.spxSize,
            **self.scene_model_kwargs
        )

        meta_cube_data, meta_cube_var = cube_to_arrays(self.meta_cube)

        if self.least_squares:
            var = None
        else:
            var = meta_cube_var

        fitter = MultipleImageFitter(
            scene_model,
            meta_cube_data,
            var,
            priors=self.global_priors,
            prior_scale=self.prior_scale,
        )
        fitter.fix(wavelength=self.meta_cube.lbda)

        # Set initial guesses from the metaslice fits
        for key, value in self.metaslice_guesses.items():
            fitter.add_global_fit_parameter(key, value)

        # Do global fits for all other model parameters with default starting
        # parameters.
        for key in self.global_fit_keys:
            fitter.add_global_fit_parameter(key)

        fit_scene_model = fitter.fit()

        # Calculate covariance matrix
        print("  Calculating covariance...")
        try:
            covariance_names, covariance = fitter.calculate_covariance()
        except SceneModelException as e:
            # Failed to calculate the covariance. This can happen for many
            # reasons. Print out the model that we failed on to help figure out
            # what happened.
            fitter.print_fit_info("3D metaslice fit", uncertainties=False)
            raise

        uncertainties = fitter.calculate_uncertainties(
            names=covariance_names, covariance=covariance
        )

        # Print out fit info
        parameters = fitter.parameters

        self.print_message(
            "  Ref. position fit @%.0f A: %+.2f±%.2f × %+.2f±%.2f spx" %
            (config.reference_wavelength, parameters['ref_center_x'],
             uncertainties['ref_center_x'], parameters['ref_center_y'],
             uncertainties['ref_center_y']), 1
        )

        # Update ADR params
        adr_delta = parameters['adr_delta']
        adr_theta = parameters['adr_theta']
        self.adr_model.set_param(delta=adr_delta, theta=adr_theta)
        self.print_message(
            "  ADR fit: delta=%.2f±%.2f, theta=%.1f±%.1f deg" %
            (adr_delta, uncertainties['adr_delta'], adr_theta * rad_to_deg,
             uncertainties['adr_theta'] * rad_to_deg), 1
        )
        self.print_message("  Effective airmass: %.2f" %
                           self.adr_model.get_airmass())

        # Calculate the fitted center positions in each metaslice
        fit_center_x, fit_center_y = self.adr_model.refract(
            parameters['ref_center_x'], parameters['ref_center_y'],
            self.meta_cube.lbda, unit=self.cube.spxSize
        )

        # Estimated seeing (FWHM in arcsec)
        reference_seeing = self.cube.spxSize * fit_scene_model.calculate_fwhm(
            wavelength=config.reference_wavelength
        )
        self.print_message('  Seeing estimate @%.0f A: %.2f" FWHM' %
                           (config.reference_wavelength, reference_seeing))

        # Estimated seeing parameters
        seeing_powerlaw_values = [parameters[i] for i in
                                  self.seeing_powerlaw_keys]
        fit_seeing_widths = evaluate_power_law(
            seeing_powerlaw_values,
            self.meta_cube.lbda / config.reference_wavelength
        )

        # Print out full fit info if requested
        if self.verbosity >= 1:
            print("")
            fitter.print_fit_info("3D metaslice fit",
                                  uncertainties=uncertainties,
                                  verbosity=self.verbosity)
            print("")

        # Create a standard SNIFS cube with the meta cube model
        meta_cube_model = pySNIFS.SNIFS_cube(lbda=self.meta_cube.lbda)
        meta_cube_model.x = meta_cube_model.i - np.max(meta_cube_model.i) / 2.
        meta_cube_model.y = meta_cube_model.j - np.max(meta_cube_model.j) / 2.
        model = fitter.evaluate()[:, meta_cube_model.i, meta_cube_model.j]
        meta_cube_model.data = model

        # Save results
        self.metaslice_info['fit_seeing_widths'] = fit_seeing_widths
        self.metaslice_info['fit_center_x'] = fit_center_x
        self.metaslice_info['fit_center_y'] = fit_center_y
        self.fitter_3d = fitter
        self.fit_parameters = fitter.parameters
        self.fit_uncertainties = uncertainties
        self.fit_scene_model = fit_scene_model
        self.meta_cube_model = meta_cube_model
        self.reference_seeing = reference_seeing

    def write_log_3d(self, path):
        """Dump an informative text log about the PSF (full-cube) 3D-fit."""
        self.print_message("Producing 3D adjusted parameter logfile %s..." %
                           path)
        logfile = open(path, 'w')

        logfile.write('# cube    : %s   \n' % os.path.basename(self.path))
        logfile.write('# object  : %s   \n' % self.header["OBJECT"])
        logfile.write('# airmass : %.3f \n' % self.header["AIRMASS"])
        logfile.write('# efftime : %.3f \n' % self.header["EFFTIME"])

        parameters = self.fit_parameters
        uncertainties = self.fit_uncertainties
        parameter_names = sorted(self.fit_parameters.keys())

        # Global parameters
        logfile.write("# Global parameters\n")
        logfile.write("# %18s %20s %20s\n" % ("name", "value", "uncertainty"))
        for name in parameter_names:
            value = parameters[name]
            if not np.isscalar(value):
                # A non-global parameter, skip it for now
                continue

            logfile.write("%20s %20g %20g\n" % (name, value,
                                                uncertainties[name]))
        if self.least_squares:
            name = "RSS"
        else:
            name = "chi2"
        logfile.write("%20s %20s\n" % (name, self.fitter_3d.fit_chi_square))

        # Individual parameters
        logfile.write("# Individual parameters\n")
        first = True
        for i in range(len(self.metaslice_info)):
            wavelength = self.metaslice_info[i]['wavelength']

            if first:
                # Print a header.
                first = False

                header_str = "# wavelength"

                for key in parameter_names:
                    if np.isscalar(parameters[key]):
                        continue
                    header_str += " %41s" % ("%s +/- d%s" % (key, key))

                if self.least_squares:
                    header_str += " %20s\n" % "RSS"
                else:
                    header_str += " %20s\n" % "chi2"
                logfile.write(header_str)

            # Print out each parameter
            row_str = "%12.2f" % wavelength
            for key in parameter_names:
                if np.isscalar(parameters[key]):
                    continue
                row_str += " %20g %20g" % (parameters[key][i],
                                           uncertainties[key][i])
            row_str += " %20g\n" % self.fitter_3d.scene_models[i].chi_square()

            logfile.write(row_str)

        logfile.close()

    def check_validity(self):
        """Check the validity of the model.

        This adds warnings to the accountant for parameters that are far from
        their predicted values, and raises a SceneModelException if the
        parameters are unphysical.
        """
        if self.prior_scale > 0.:
            # Test on seeing
            if self.seeing_prior:
                fac = (self.reference_seeing / self.seeing_prior - 1) * 1e2
                if abs(fac) > MAX_SEEING_PRIOR_OFFSET:
                    print("WARNING: Seeing %.2f\" is %+.0f%% away from "
                          "predicted %.2f\"" % (self.reference_seeing,
                                                fac, self.seeing_prior))
                    if self.accountant:
                        self.accountant.add_warning("ES_PRIOR_SEEING")

            # Tests on ADR parameters
            original_airmass = self.original_adr_model.get_airmass()
            fit_airmass = self.adr_model.get_airmass()
            fac = (fit_airmass / original_airmass - 1) * 1e2
            if abs(fac) > MAX_AIRMASS_PRIOR_OFFSET:
                print("WARNING: Airmass %.2f is %+.0f%% away from "
                      "predicted %.2f" % (fit_airmass, fac, original_airmass))
                if self.accountant:
                    self.accountant.add_warning("ES_PRIOR_AIRMASS")

            # Rewrap angle difference [rad]
            original_theta = self.original_adr_model.theta
            fit_theta = self.adr_model.theta
            err = (((fit_theta - original_theta) + np.pi) %
                   (2 * np.pi) - np.pi) * rad_to_deg
            if abs(err) > MAX_PARANG_PRIOR_OFFSET:
                print("WARNING: Parangle %.0fdeg is %+.0fdeg away from "
                      "predicted %.0fdeg" %
                      (self.adr_model.get_parangle(), err,
                       self.original_adr_model.get_parangle()))
                if self.accountant:
                    self.accountant.add_warning("ES_PRIOR_PARANGLE")

        ref_center_x = self.fit_parameters['ref_center_x']
        ref_center_y = self.fit_parameters['ref_center_y']
        if not (abs(ref_center_x) < MAX_POSITION and
                abs(ref_center_y) < MAX_POSITION):
            print("WARNING: Point-source %+.2f x %+.2f mis-centered" %
                  (ref_center_x, ref_center_y))
            if self.accountant:
                self.accountant.add_warning("ES_MIS-CENTERED")

        # Tests on seeing and airmass
        if not MIN_SEEING < self.reference_seeing < MAX_SEEING:
            raise SceneModelException("Unphysical seeing (%.2f\")" %
                                      self.reference_seeing)
        if not 1. <= self.adr_model.get_airmass() < MAX_AIRMASS:
            raise SceneModelException("Unphysical airmass (%.2f)" %
                                      self.adr_model.get_airmass())
        # Test positivity of alpha and ellipticity
        fit_seeing_widths = self.metaslice_info['fit_seeing_widths']
        if fit_seeing_widths.min() < 0:
            raise SceneModelException(
                "Seeing widths are negative (%.2f) at %.0f A" %
                (fit_seeing_widths.min(),
                 self.meta_cube.lbda[fit_seeing_widths.argmin()]))

    def extract(self, method='psf', radius=None, **kwargs):
        """Extract the PSF. See SceneModel.extract for details.

        If aperture photometry is being performed, radius is interpreted
        as a multiple of the seeing sigmas if it is less than 0, and a radius
        in arcseconds if it is greater than 0.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before extracting!"
            )

        if method == "psf":
            method_str = "psf, %s" % ("least-squares" if self.least_squares
                                      else "chi2")
        else:
            method_str = "%s, r=%.2f" % (method, radius)

        print("Extracting the point-source spectrum (%s)..." % method_str)

        cube_data, cube_var = cube_to_arrays(self.cube)

        if self.least_squares:
            cube_var = None

        # Convert the radius to pixels.
        if radius is not None:
            pixel_radius = convert_radius_to_pixels(
                radius, self.reference_seeing, self.cube.spxSize
            )
        else:
            pixel_radius = None

        extraction = self.fit_scene_model.extract(
            cube_data, cube_var, method=method, radius=pixel_radius,
            wavelength=self.cube.lbda
        )

        # Build a pySNIFS object for the point source
        point_source_spectrum = pySNIFS.spectrum(
            data=extraction['amplitude'],
            var=extraction['amplitude_variance'],
            start=self.cube.lbda[0],
            step=self.cube.lstep,
        )
        self.point_source_spectrum = point_source_spectrum

        if self.has_sky:
            # Convert (mean) sky spectrum to "per arcsec**2"
            spaxel_size = self.cube.spxSize
            extraction['background_density'] = \
                extraction['background'] / spaxel_size**2
            if self.has_sky:
                extraction['background_density_variance'] = \
                    extraction['background_variance'] / spaxel_size**4

            # Build a pySNIFS object for the background
            sky_spectrum = pySNIFS.spectrum(
                data=extraction['background_density'],
                var=extraction['background_density_variance'],
                start=self.cube.lbda[0],
                step=self.cube.lstep,
            )
            self.sky_spectrum = sky_spectrum

        # Save the information about the extraction.
        self.extraction_method = method
        if method == 'aperture' or method == 'subaperture':
            self.extraction_radius = radius

        self.extraction = extraction

    def write_spectrum(self, output_path, sky_output_path,
                       prefix=config.default_fits_prefix):
        """Write the spectrum out to a fits file at the given path"""
        # Make sure that the extraction has already been done
        if self.extraction is None:
            raise SceneModelException(
                "Must run the extraction to produce spectra!"
            )

        # Build the header for the output file. We start with the input header
        # and add in tons of new keys.
        header = self.header

        items = self.fit_scene_model.get_fits_header_items(prefix=prefix)
        for key, value, description in items:
            header[key] = (value, description)

        # Helper to prefix things
        def p(extension):
            return "%s%s" % (prefix, extension)

        header[p('VERS')] = config.__version__
        header[p('CUBE')] = (self.path, "Input cube")
        header[p('LREF')] = (config.reference_wavelength, "Lambda ref. [A]")
        header[p('SDEG')] = (self.background_degree, "Polynomial bkgnd degree")
        header[p('CHI2')] = (self.fitter_3d.fit_chi_square,
                             "Chi2|RSS of 3D-fit")
        header[p('AIRM')] = (self.adr_model.get_airmass(), "Effective airmass")
        header[p('PARAN')] = (self.adr_model.get_parangle(),
                              "Effective parangle [deg]")
        header[p('LMIN')] = (self.meta_cube.lstart,
                             "Meta-slices minimum lambda")
        header[p('LMAX')] = (self.meta_cube.lend, "Meta-slices maximum lambda")

        header[p('PSF')] = (self.psf, 'PSF model name')
        header[p('SUB')] = (self.subsampling, 'Model subsampling')
        header[p('BORD')] = (self.border, 'Model border')

        header[p('METH')] = (self.extraction_method, 'Extraction method')
        if self.extraction_radius is not None:
            header[p('APRAD')] = (self.extraction_radius,
                                  'Aperture radius [" or sigma]')

        tflux = self.extraction['amplitude'].sum()
        header[p('TFLUX')] = (tflux, 'Total point-source flux')

        if self.has_sky:
            sflux = self.extraction['background_density'].sum()
            header[p('SFLUX')] = (sflux, 'Total sky flux/arcsec^2')

        header['SEEING'] = (self.reference_seeing,
                            'Estimated seeing @lbdaRef ["] (extract_star2)')

        if self.prior_scale > 0.:
            header['ES_PRIOR'] = (self.prior_scale, 'Prior hyper-scale')
            if self.seeing_prior is not None:
                header['ES_PRISE'] = (self.seeing_prior,
                                      'Seeing prior [arcsec]')

        # Save the point source spectrum
        print("  Saving output point-source spectrum to '%s'" % output_path)
        write_pysnifs_spectrum(self.point_source_spectrum, output_path, header)

        if self.has_sky:
            # Save the sky spectrum
            print("  Saving output sky spectrum to '%s'" % sky_output_path)
            write_pysnifs_spectrum(self.sky_spectrum, sky_output_path, header)

    @property
    def has_sky(self):
        """Return True if the model has a sky component, False otherwise"""
        return self.background_degree != -1

    @property
    def object_name(self):
        """Return the name of the object"""
        object_name = self.header.get('OBJECT', 'Unknown')
        return object_name

    def plot_spectrum(self, path=None):
        """Plot the extracted spectrum.

        If path is set, the spectrum is written to a file at that path.
        """
        # Make sure that the extraction is done.
        if self.extraction is None:
            raise SceneModelException(
                "Must run the extraction before plotting spectra!"
            )

        point_spec = self.point_source_spectrum
        sky_spec = self.sky_spectrum

        if path is not None:
            self.print_message("Producing spectra plot %s..." % path, 1)

        from matplotlib import pyplot as plt

        fig = plt.figure()

        if self.has_sky and sky_spec.data.any():
            axS = fig.add_subplot(3, 1, 1)  # Point-source
            axB = fig.add_subplot(3, 1, 2)  # Sky
            axN = fig.add_subplot(3, 1, 3)  # S/N
        else:
            axS = fig.add_subplot(2, 1, 1)  # Point-source
            axN = fig.add_subplot(2, 1, 2)  # S/N

        axS.text(0.95, 0.8, os.path.basename(self.path), fontsize='small',
                 ha='right', transform=axS.transAxes)

        axS.plot(point_spec.x, point_spec.data, MPL.blue)
        axS.errorband(point_spec.x, point_spec.data, np.sqrt(point_spec.var),
                      color=MPL.blue)
        axN.plot(point_spec.x, point_spec.data / np.sqrt(point_spec.var),
                 MPL.blue)

        if self.has_sky and sky_spec.data.any():
            axB.plot(sky_spec.x, sky_spec.data, MPL.green)
            axB.errorband(sky_spec.x, sky_spec.data, np.sqrt(sky_spec.var),
                          color=MPL.green)
            axB.set(title=u"Background spectrum (per arcsec²)",
                    xlim=(sky_spec.x[0], sky_spec.x[-1]),
                    xticklabels=[])

            # Sky S/N
            axN.plot(sky_spec.x, sky_spec.data / np.sqrt(sky_spec.var),
                     MPL.green)

        axS.set(title="Point-source spectrum [%s]" % (self.object_name),
                xlim=(point_spec.x[0], point_spec.x[-1]), xticklabels=[])
        axN.set(title="Signal/Noise", xlabel=u"Wavelength [Å]",
                xlim=(point_spec.x[0], point_spec.x[-1]))

        fig.tight_layout()
        if path:
            fig.savefig(path)

    def plot_slice_fit(self, path=None):
        """Plot the fits to each slice.

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting slices!"
            )

        if path is not None:
            self.print_message("Producing slice fit plot %s..." % path, 1)

        from matplotlib import pyplot as plt

        num_meta_slices = self.meta_cube.nslice
        ncol = int(np.floor(np.sqrt(num_meta_slices)))
        nrow = int(np.ceil(num_meta_slices / float(ncol)))

        fig = plt.figure()
        fig.suptitle("Slice plots [%s, airmass=%.2f]" %
                     (self.object_name, self.header['AIRMASS']),
                     fontsize='large')

        # Compute all of the model components on an incomplete cube. components
        # will have the shape (num_meta_slices, num_components, num_spaxels)
        full_components = self.fitter_3d.evaluate(separate_components=True)
        components = full_components[..., self.meta_cube.i, self.meta_cube.j]
        model = np.sum(components, axis=1)
        spaxel_numbers = np.sort(self.meta_cube.no)
        num_components = components.shape[1]

        # Make a plot for each meta slice
        for i in range(num_meta_slices):
            data = self.meta_cube.data[i, :]
            fit = model[i, :]
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.plot(spaxel_numbers, data, color=MPL.blue, ls='-')  # Signal
            if self.meta_cube.var is not None:
                ax.errorband(spaxel_numbers, data,
                             np.sqrt(self.meta_cube.var[i, :]), color=MPL.blue)
            ax.plot(spaxel_numbers, fit, color=MPL.red, ls='-')   # Model

            # Plot the individual components.
            colors = [MPL.green, MPL.orange, MPL.purple, MPL.yellow, MPL.brown]
            for component_idx in range(num_components):
                ax.plot(spaxel_numbers, components[i][component_idx],
                        color=colors[component_idx % len(colors)], ls='-')
            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                     fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % self.meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)

            ax.set_ylim(data.min() / 1.2, data.max() * 1.2)
            ax.set_xlim(-1, 226)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Spaxel #", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')

        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if path:
            fig.savefig(path)

    def plot_row_column_sums(self, path=None):
        """Plot the sums of each row and column for each slice.

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting slices!"
            )

        if path is not None:
            self.print_message("Producing profile plot %s..." % path, 1)

        from matplotlib import pyplot as plt

        fig = plt.figure()
        fig.suptitle("Rows and columns [%s, airmass=%.2f]" %
                     (self.object_name, self.header['AIRMASS']),
                     fontsize='large')

        num_meta_slices = self.meta_cube.nslice
        ncol = int(np.floor(np.sqrt(num_meta_slices)))
        nrow = int(np.ceil(num_meta_slices / float(ncol)))

        for i in range(num_meta_slices):
            ax = fig.add_subplot(nrow, ncol, i + 1)

            # Signal
            signal_slice = self.meta_cube.slice2d(i, coord='p', NAN=False)
            row_profile = signal_slice.sum(axis=0)  # Sum along rows
            col_profile = signal_slice.sum(axis=1)  # Sum along columns

            # Errors
            if self.least_squares:
                # Least-squares, don't show errorbars.
                ax.plot(range(len(row_profile)), row_profile,
                        marker='o', c=MPL.blue, ms=3, ls='None')
                ax.plot(range(len(col_profile)), col_profile,
                        marker='^', c=MPL.red, ms=3, ls='None')
            else:
                # Chi-square, show errorbars
                var_slice = self.meta_cube.slice2d(i, coord='p', var=True,
                                                   NAN=False)
                row_err = np.sqrt(var_slice.sum(axis=0))
                col_err = np.sqrt(var_slice.sum(axis=1))
                ax.errorbar(range(len(row_profile)), row_profile, row_err,
                            fmt='o', c=MPL.blue, ecolor=MPL.blue, ms=3)
                ax.errorbar(range(len(col_profile)), col_profile, col_err,
                            fmt='^', c=MPL.red, ecolor=MPL.red, ms=3)

            # Model
            model_slice = self.meta_cube_model.slice2d(i, coord='p')
            row_model = model_slice.sum(axis=0)
            col_model = model_slice.sum(axis=1)
            ax.plot(row_model, ls='-', color=MPL.blue)
            ax.plot(col_model, ls='-', color=MPL.red)

            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                     fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % self.meta_cube.lbda[i],
                    fontsize='x-small', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I (blue) or J (red)", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')

            fig.subplots_adjust(left=0.06, right=0.96, bottom=0.06, top=0.95)

        if path:
            fig.savefig(path)

    def plot_adr(self, path=None):
        """Plot the ADR fit

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting the ADR!"
            )

        if path is not None:
            self.print_message("Producing ADR plot %s..." % path, 1)

        import matplotlib
        from matplotlib import pyplot as plt

        # Accomodate errorbar fmt API change at v2.2.0
        # (https://matplotlib.org/api/api_changes.html#function-signatures)
        if matplotlib.__version__ >= '2.2.0':
            NONE = 'none'
        else:
            NONE = None

        wavelengths = self.meta_cube.lbda
        spaxel_size = self.cube.spxSize
        fit_ref_center_x = self.fit_scene_model['ref_center_x']
        fit_ref_center_y = self.fit_scene_model['ref_center_y']
        guess_ref_center_x = self.metaslice_guesses['ref_center_x']
        guess_ref_center_y = self.metaslice_guesses['ref_center_y']

        # Calculate offsets to shift positions to the middle of the fit
        # wavelength range. The reference position can float, so it makes more
        # sense to anchor it at the middle wavelength rather than at the
        # reference wavelength for plotting purposes.
        mid_wavelength = np.median(wavelengths)
        guess_mid_x, guess_mid_y = self.original_adr_model.refract(
            guess_ref_center_x, guess_ref_center_y, mid_wavelength,
            unit=spaxel_size
        )
        fit_mid_x, fit_mid_y = self.adr_model.refract(
            fit_ref_center_x, fit_ref_center_y, mid_wavelength,
            unit=spaxel_size
        )

        # Retrieve positions from individual fits
        individual_x = self.metaslice_info['center_x']
        individual_y = self.metaslice_info['center_y']
        individual_x_err = self.metaslice_info['center_x_uncertainty']
        individual_y_err = self.metaslice_info['center_y_uncertainty']

        # Calculate positions using the original ADR model read from the
        # header.
        guess_x, guess_y = self.original_adr_model.refract(
            guess_ref_center_x, guess_ref_center_y, wavelengths,
            unit=spaxel_size
        )
        mid_individual_x = individual_x + guess_mid_x - guess_x
        mid_individual_y = individual_y + guess_mid_y - guess_y

        # Positions after the 3D fit
        fit_x = self.metaslice_info['fit_center_x']
        fit_y = self.metaslice_info['fit_center_y']

        fig = plt.figure()

        ax3 = fig.add_subplot(
            2, 1, 1,
            aspect='equal',
            adjustable='datalim',
            xlabel="X center [spx]",
            ylabel="Y center [spx]",
            title="ADR plot [%s, airmass=%.2f]" % (self.object_name,
                                                   self.header['AIRMASS'])
        )
        ax1 = fig.add_subplot(
            2, 2, 3,
            xlabel=u"Wavelength [Å]",
            ylabel="X center [spx]"
        )
        ax2 = fig.add_subplot(
            2, 2, 4,
            xlabel=u"Wavelength [Å]",
            ylabel="Y center [spx]"
        )

        valid = self.metaslice_info['valid']
        good = self.metaslice_info['good']
        bad = self.metaslice_info['bad']

        # Should all have the same radius, so we can pick the first one.
        good_radius = self.metaslice_info['guess_center_radius'][0]

        if good.any():
            ax1.errorbar(wavelengths[good], individual_x[good],
                         yerr=individual_x_err[good], fmt=NONE,
                         ecolor=MPL.green)
            ax1.scatter(wavelengths[good], individual_x[good],
                        edgecolors='none', c=wavelengths[good],
                        cmap=plt.cm.jet, zorder=3, label="Fit 2D")
        if bad.any():
            ax1.plot(wavelengths[bad], individual_x[bad], mfc=MPL.red,
                     mec=MPL.red, marker='.', ls='None', label='_')
        ax1.plot(wavelengths, guess_x, 'k--', label="Guess 3D")
        ax1.plot(wavelengths, fit_x, MPL.green, label="Fit 3D")
        plt.setp(ax1.get_xticklabels() + ax1.get_yticklabels(),
                 fontsize='xx-small')
        ax1.legend(loc='best', fontsize='small', frameon=False)

        if good.any():
            ax2.errorbar(wavelengths[good], individual_y[good],
                         yerr=individual_y_err[good], fmt=NONE,
                         ecolor=MPL.green)
            ax2.scatter(wavelengths[good], individual_y[good],
                        edgecolors='none', c=wavelengths[good],
                        cmap=plt.cm.jet, zorder=3)
        if bad.any():
            ax2.plot(wavelengths[bad], individual_y[bad], marker='.',
                     mfc=MPL.red, mec=MPL.red, ls='None')
        ax2.plot(wavelengths, guess_y, 'k--')
        ax2.plot(wavelengths, fit_y, MPL.green)
        plt.setp(ax2.get_xticklabels() + ax2.get_yticklabels(),
                 fontsize='xx-small')

        if valid.any():
            ax3.errorbar(individual_x[valid], individual_y[valid],
                         xerr=individual_x_err[valid],
                         yerr=individual_y_err[valid], fmt=NONE,
                         ecolor=MPL.green)

        if good.any():
            ax3.scatter(individual_x[good], individual_y[good],
                        edgecolors='none', c=wavelengths[good],
                        cmap=plt.cm.jet, zorder=3)
            # Plot position selection process
            ax3.plot(mid_individual_x[good], mid_individual_y[good],
                     marker='.', mfc=MPL.blue, mec=MPL.blue, ls='None')
        if bad.any():
            # Discarded ref. positions
            ax3.plot(mid_individual_x[bad], mid_individual_y[bad], marker='.',
                     mfc=MPL.red, mec=MPL.red, ls='None')
        ax3.plot((guess_mid_x, fit_mid_x), (guess_mid_y, fit_mid_y), 'k-')
        ax3.plot(guess_x, guess_y, 'k--')  # Guess ADR
        ax3.plot(fit_x, fit_y, MPL.green)      # Adjusted ADR
        ax3.set_autoscale_on(False)
        ax3.plot((fit_mid_x,), (fit_mid_y,), 'k+')
        ax3.add_patch(matplotlib.patches.Circle(
            (guess_mid_x, guess_mid_y), radius=good_radius, ec='0.8',
            fc='None'))
        # ADR selection
        ax3.add_patch(matplotlib.patches.Rectangle(
            (-7.5, -7.5), 15, 15, ec='0.8', lw=2, fc='None'))
        # FoV
        txt = u'Guess: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
              (guess_ref_center_x, guess_ref_center_y,
               self.original_adr_model.get_airmass(),
               self.original_adr_model.get_parangle())
        txt += u'\nFit: x0,y0=%+4.2f,%+4.2f  airmass=%.2f parangle=%+.0f°' % \
               (fit_ref_center_x, fit_ref_center_y,
                self.adr_model.get_airmass(), self.adr_model.get_parangle())
        txtcol = 'k'

        if self.accountant:
            accountant = self.accountant
            if accountant.test_warning('ES_PRIOR_POSITION'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_POSITION')
                txtcol = MPL.red
            if accountant.test_warning('ES_PRIOR_AIRMASS'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_AIRMASS')
                txtcol = MPL.red
            if accountant.test_warning('ES_PRIOR_PARANGLE'):
                txt += '\n%s' % accountant.get_warning('ES_PRIOR_PARANGLE')
                txtcol = MPL.red

        ax3.text(0.95, 0.8, txt, transform=ax3.transAxes, fontsize='small',
                 ha='right', color=txtcol)

        fig.tight_layout()
        if path:
            fig.savefig(path)

    def plot_seeing(self, path=None):
        """Plot the seeing and ellipticity parameters

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting seeing!"
            )

        if path is not None:
            self.print_message("Producing seeing plot %s..." % path, 1)

        from matplotlib import pyplot as plt

        # The first subplot is seeing. The other 2 subplots will show different
        # things for the different PSFs.
        if self.psf == 'classic':
            key_2 = 'ell'
            key_3 = 'xy'
            label_1 = u'α [spx]'
            label_2 = u'y² coeff.'
            label_3 = u'xy coeff.'
        elif self.psf == 'fourier':
            key_2 = 'ellipticity_x'
            key_3 = 'ellipticity_y'
            label_1 = 'seeing width [spx]'
            label_2 = 'ellipticity x'
            label_3 = 'ellipticity y'
        else:
            raise SceneModelException("Can't make seeing plot for PSF %s"
                                      % self.psf)

        wavelengths = self.meta_cube.lbda

        val_2 = extract_key(self.metaslice_info['scene_model'], key_2)
        err_2 = extract_key(self.metaslice_info['uncertainties'], key_2)
        val_3 = extract_key(self.metaslice_info['scene_model'], key_3)
        err_3 = extract_key(self.metaslice_info['uncertainties'], key_3)

        guess_2 = self.metaslice_guesses[key_2]
        guess_3 = self.metaslice_guesses[key_3]

        fit_2 = self.fit_parameters[key_2]
        fit_3 = self.fit_parameters[key_3]
        fit_err_2 = self.fit_uncertainties[key_2]
        fit_err_3 = self.fit_uncertainties[key_3]

        def plot_conf_interval(ax, x, y, dy):
            ax.plot(x, y, MPL.green, label="Fit 3D")
            if dy is not None:
                ax.errorband(x, y, dy, color=MPL.green)

        fig = plt.figure()

        ax1 = fig.add_subplot(
            2, 1, 1,
            title='Model parameters '
            '[%s, seeing %.2f" FWHM]' % (self.object_name,
                                         self.reference_seeing),
            xticklabels=[],
            ylabel=label_1
        )
        ax2 = fig.add_subplot(4, 1, 3, xticklabels=[], ylabel=label_2)
        ax3 = fig.add_subplot(4, 1, 4, xlabel=u"Wavelength [Å]",
                              ylabel=label_3)

        good = self.metaslice_info['good']
        bad = self.metaslice_info['bad']

        if good.any():
            ax1.errorbar(
                wavelengths[good],
                self.metaslice_info[self.seeing_key][good],
                self.metaslice_info['%s_uncertainty' % self.seeing_key][good],
                marker='.', mfc=MPL.blue, mec=MPL.blue, ecolor=MPL.blue,
                capsize=0, ls='None', label="Fit 2D"
            )
        if bad.any():
            ax1.plot(wavelengths[bad],
                     self.metaslice_info[self.seeing_key][bad], marker='.',
                     mfc=MPL.red, mec=MPL.red, ls='None', label="_")
        ax1.plot(
            wavelengths,
            self.metaslice_info['guess_%s' % self.seeing_key],
            'k--',
            label="Guess 3D" if not self.seeing_prior else "Prior 3D"
        )
        fit_widths = extract_key(self.fitter_3d.scene_models, self.seeing_key)
        plot_conf_interval(ax1, wavelengths, fit_widths, None)

        seeing_powerlaw_guesses = []
        seeing_powerlaw_values = []
        for key in self.seeing_powerlaw_keys:
            seeing_powerlaw_guesses.append(self.metaslice_guesses[key])
            seeing_powerlaw_values.append(self.fit_parameters[key])

        txt = 'Guess: %s' % \
              (', '.join(['A%d=%.2f' % (i, a) for i, a in
                          enumerate(seeing_powerlaw_guesses)]))
        txt += '\nFit: %s' % \
               (', '.join(['A%d=%.2f' % (i, a) for i, a in
                           enumerate(seeing_powerlaw_values)]))

        txtcol = 'k'
        if self.accountant and self.accountant.test_warning('ES_PRIOR_SEEING'):
            txt += '\n%s' % self.accountant.get_warning('ES_PRIOR_SEEING')
            txtcol = MPL.red
        ax1.text(0.95, 0.8, txt, transform=ax1.transAxes, fontsize='small',
                 ha='right', color=txtcol)
        ax1.legend(loc='upper left', fontsize='small', frameon=False)
        plt.setp(ax1.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax2.errorbar(wavelengths[good], val_2[good], err_2[good],
                         marker='.', mfc=MPL.blue, mec=MPL.blue,
                         ecolor=MPL.blue, capsize=0, ls='None')
        if bad.any():
            ax2.plot(wavelengths[bad], val_2[bad], marker='.', mfc=MPL.red,
                     mec=MPL.red, ls='None')
        ax2.plot(wavelengths, guess_2*np.ones(len(wavelengths)), 'k--')
        plot_conf_interval(ax2, wavelengths, fit_2*np.ones(len(wavelengths)),
                           fit_err_2*np.ones(len(wavelengths)))
        txt = 'Guess: %s=%.3f' % (label_2, guess_2)
        txt += '\nFit: %s=%.3f' % (label_2, fit_2)

        ax2.text(0.95, 0.1, txt, transform=ax2.transAxes, fontsize='small',
                 ha='right', va='bottom')
        plt.setp(ax2.get_yticklabels(), fontsize='x-small')

        if good.any():
            ax3.errorbar(wavelengths[good], val_3[good], err_3[good],
                         marker='.', mfc=MPL.blue, mec=MPL.blue,
                         ecolor=MPL.blue, capsize=0, ls='None')
        if bad.any():
            ax3.plot(wavelengths[bad], val_3[bad], marker='.', mfc=MPL.red,
                     mec=MPL.red, ls='None')
        ax3.plot(wavelengths, guess_3*np.ones(len(wavelengths)), 'k--')
        plot_conf_interval(ax3, wavelengths, fit_3*np.ones(len(wavelengths)),
                           fit_err_3*np.ones(len(wavelengths)))
        txt = 'Guess: %s=%.3f' % (label_3, guess_3)
        txt += '\nFit: %s=%.3f' % (label_3, fit_3)
        ax3.text(0.95, 0.1, txt, transform=ax3.transAxes, fontsize='small',
                 ha='right', va='bottom')
        plt.setp(ax3.get_xticklabels() + ax3.get_yticklabels(),
                 fontsize='x-small')

        fig.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.95)
        if path:
            fig.savefig(path)

    def plot_residuals(self, path=None):
        """Plot the residuals of the 3D fit in each meta-slice.

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting residuals!"
            )

        if path is not None:
            self.print_message("Producing residual plot %s..." % path, 1)

        import matplotlib
        from matplotlib import pyplot as plt

        fig = plt.figure()
        fig.suptitle("Residual plot [%s, airmass=%.2f]" %
                     (self.object_name, self.header['AIRMASS']),
                     fontsize='large')

        wavelengths = self.meta_cube.lbda
        num_meta_slices = self.meta_cube.nslice
        ncol = int(np.floor(np.sqrt(num_meta_slices)))
        nrow = int(np.ceil(num_meta_slices / float(ncol)))

        fit_center_x = self.metaslice_info['fit_center_x']
        fit_center_y = self.metaslice_info['fit_center_y']

        extent = (self.meta_cube.x.min() - 0.5, self.meta_cube.x.max() + 0.5,
                  self.meta_cube.y.min() - 0.5, self.meta_cube.y.max() + 0.5)

        images = []
        for i in range(num_meta_slices):        # Loop over meta-slices
            ax = fig.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = self.meta_cube.slice2d(i, coord='p')  # Signal
            fit = self.meta_cube_model.slice2d(i, coord='p')  # Model
            if self.least_squares:
                # Least-squares: display relative residuals
                res = np.nan_to_num((data - fit) / fit) * 100  # [%]
            else:
                # Chi2 fit: display residuals in units of sigma
                var = self.meta_cube.slice2d(i, coord='p', var=True, NAN=False)
                res = np.nan_to_num((data - fit) / np.sqrt(var))

            # List of images, to be commonly normalized latter on
            images.append(ax.imshow(res, origin='lower', extent=extent,
                                    cmap=plt.cm.RdBu_r,
                                    interpolation='nearest'))

            ax.plot((fit_center_x[i],), (fit_center_y[i],), marker='*',
                    color=MPL.green)
            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                     fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % wavelengths[i],
                    fontsize='x-small', transform=ax.transAxes)
            ax.axis(extent)

            # Axis management
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            if not ax.is_first_col():
                ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

        # Common image normalization
        vmin, vmax = np.percentile(
            [im.get_array().filled() for im in images], (3., 97.))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Colorbar
        cax = fig.add_axes([0.90, 0.07, 0.02, 0.87])
        cbar = fig.colorbar(images[0], cax, orientation='vertical')
        plt.setp(cbar.ax.get_yticklabels(), fontsize='small')
        if self.least_squares:
            # Chi2 fit
            cbar.set_label(u'Residuals [%]', fontsize='small')
        else:
            cbar.set_label(u'Residuals [σ]', fontsize='small')

        fig.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.95,
                            hspace=0.02, wspace=0.02)
        if path:
            fig.savefig(path)

    def plot_radial_profile(self, path=None):
        """Plot the radial profiles of the 3D fit in each meta-slice.

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting radial "
                "profiles!"
            )

        if path is not None:
            self.print_message("Producing radial profile plot %s..." % path, 1)

        from matplotlib import pyplot as plt

        wavelengths = self.meta_cube.lbda
        num_meta_slices = self.meta_cube.nslice
        ncol = int(np.floor(np.sqrt(num_meta_slices)))
        nrow = int(np.ceil(num_meta_slices / float(ncol)))

        # 3D fit positions
        fit_center_x = self.metaslice_info['fit_center_x']
        fit_center_y = self.metaslice_info['fit_center_y']

        fig = plt.figure()
        fig.suptitle(
            "Radial profile plot [%s, airmass=%.2f]" %
            (self.object_name, self.header['AIRMASS']),
            fontsize='large'
        )

        # Compute all of the model components on an incomplete cube. components
        # will have the shape (num_meta_slices, num_components, num_spaxels)
        full_components = self.fitter_3d.evaluate(separate_components=True)
        components = full_components[..., self.meta_cube_model.i,
                                     self.meta_cube_model.j]
        num_components = components.shape[1]

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
            # Mean radius
            rb = np.array([r[ibins == iib].mean() for iib in ib])
            if weighted:
                # Mean radius-weighted data
                fb = np.array(
                    [np.average(f[ibins == iib], weights=r[ibins == iib]) for
                     iib in ib]
                )
            else:
                fb = np.array([f[ibins == iib].mean()
                              for iib in ib])  # Mean data
            return rb, fb

        for i in range(num_meta_slices):        # Loop over slices
            ax = fig.add_subplot(nrow, ncol, i + 1, yscale='log')
            # Use adjusted elliptical radius instead of plain radius
            # r    = np.hypot(meta_cube.x-xfit[i], meta_cube.y-yfit[i])
            # rfit = np.hypot(cube_fit.x-xfit[i], cube_fit.y-yfit[i])
            if self.psf == 'classic':
                r = ellRadius(
                    self.meta_cube.x, self.meta_cube.y, fit_center_x[i],
                    fit_center_y[i], self.fit_parameters['ell'],
                    self.fit_parameters['xy']
                )
                rfit = ellRadius(
                    self.meta_cube_model.x, self.meta_cube_model.y,
                    fit_center_x[i], fit_center_y[i],
                    self.fit_parameters['ell'], self.fit_parameters['xy']
                )
            else:
                r = np.hypot(self.meta_cube.x - fit_center_x[i],
                             self.meta_cube.y - fit_center_y[i])
                rfit = np.hypot(self.meta_cube_model.x - fit_center_x[i],
                                self.meta_cube_model.y - fit_center_y[i])

            ax.plot(r, self.meta_cube.data[i],
                    marker=',', mfc=MPL.blue, mec=MPL.blue, ls='None')
            ax.plot(rfit, self.meta_cube_model.data[i],
                    marker='.', mfc=MPL.red, mec=MPL.red, ms=1, ls='None')
            if self.has_sky and self.sky_spectrum.data.any():
                colors = [MPL.green, MPL.orange, MPL.purple, MPL.yellow,
                          MPL.brown]
                for component_idx in range(num_components):
                    color = colors[component_idx % len(colors)]
                    ax.plot(rfit, components[i][component_idx], mfc=color,
                            mec=color, ms=1, ls='None')
            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                     fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % wavelengths[i],
                    fontsize='x-small', transform=ax.transAxes)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("Elliptical radius [spx]", fontsize='small')
                ax.set_ylabel("Flux", fontsize='small')
            ax.axis([0, rfit.max() * 1.1,
                     self.meta_cube_model.data[i].min() / 2,
                     self.meta_cube_model.data[i].max() * 2])

            # Binned values
            rb, db = radialbin(r, self.meta_cube.data[i])
            ax.plot(rb, db, 'c.')
            rfb, fb = radialbin(rfit, self.meta_cube_model.data[i])
            ax.plot(rfb, fb, 'm.')

        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94)
        if path:
            fig.savefig(path)

    def plot_contours(self, path=None):
        """Plot contours of each meta-slice and its 3D fit.

        If path is set, the plot is written to a file at that path.
        """
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before plotting radial "
                "profiles!"
            )

        if path is not None:
            self.print_message("Producing contour plot %s..." % path, 1)

        import matplotlib
        from matplotlib import pyplot as plt

        # Accomodate errorbar fmt API change at v2.2.0
        # (https://matplotlib.org/api/api_changes.html#function-signatures)
        if matplotlib.__version__ >= '2.2.0':
            NONE = 'none'
        else:
            NONE = None

        wavelengths = self.meta_cube.lbda
        num_meta_slices = self.meta_cube.nslice
        ncol = int(np.floor(np.sqrt(num_meta_slices)))
        nrow = int(np.ceil(num_meta_slices / float(ncol)))

        # Retrieve positions from individual fits
        individual_x = self.metaslice_info['center_x']
        individual_y = self.metaslice_info['center_y']
        individual_x_err = self.metaslice_info['center_x_uncertainty']
        individual_y_err = self.metaslice_info['center_y_uncertainty']

        # 3D fit positions
        fit_center_x = self.metaslice_info['fit_center_x']
        fit_center_y = self.metaslice_info['fit_center_y']

        valid = self.metaslice_info['valid']
        good = self.metaslice_info['good']

        fig = plt.figure()
        fig.suptitle(
            "Data and fit [%s, airmass=%.2f]" %
            (self.object_name, self.header['AIRMASS']),
            fontsize='large'
        )

        extent = (self.meta_cube.x.min() - 0.5, self.meta_cube.x.max() + 0.5,
                  self.meta_cube.y.min() - 0.5, self.meta_cube.y.max() + 0.5)

        for i in range(num_meta_slices):        # Loop over meta-slices
            ax = fig.add_subplot(ncol, nrow, i + 1, aspect='equal')
            data = self.meta_cube.slice2d(i, coord='p')
            good_data = data[np.isfinite(data)]
            fit = self.meta_cube_model.slice2d(i, coord='p')

            vmin, vmax = np.percentile(good_data[good_data > 0], (5., 95.))
            lev = np.logspace(np.log10(vmin), np.log10(vmax), 5)
            ax.contour(data, lev, origin='lower', extent=extent,
                       cmap=matplotlib.cm.jet)                          # Data
            ax.contour(fit, lev, origin='lower', extent=extent,
                       linestyles='dashed', cmap=matplotlib.cm.jet)     # Fit
            if valid[i]:
                ax.errorbar((individual_x[i],), (individual_y[i],),
                            xerr=(individual_x_err[i],),
                            yerr=(individual_y_err[i],), fmt=NONE,
                            ecolor=MPL.blue if good[i] else MPL.red)
            ax.plot((fit_center_x[i],), (fit_center_y[i],), marker='*',
                    color=MPL.green)
            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                     fontsize='xx-small')
            ax.text(0.05, 0.85, u"%.0f Å" % wavelengths[i], fontsize='x-small',
                    transform=ax.transAxes)
            ax.axis(extent)
            if ax.is_last_row() and ax.is_first_col():
                ax.set_xlabel("I [spx]", fontsize='small')
                ax.set_ylabel("J [spx]", fontsize='small')
            if not ax.is_last_row():
                plt.setp(ax.get_xticklabels(), visible=False)
            if not ax.is_first_col():
                plt.setp(ax.get_yticklabels(), visible=False)

        fig.subplots_adjust(left=0.05, right=0.96, bottom=0.06, top=0.95,
                            hspace=0.02, wspace=0.02)
        if path:
            fig.savefig(path)
