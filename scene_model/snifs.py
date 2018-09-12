# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import leastsq

# Import SNIFS libraries if available.
try:
    from ToolBox.Arrays import metaslice
    from ToolBox.Astro import Coords
    from ToolBox.Atmosphere import ADR
    import pySNIFS
except ImportError as e:
    print("WARNING: Unable to load SNIFS libraries! (%s)" % e.message)
    print("Some functionality will be disabled.")

from . import config
from .utils import SceneModelException, extract_key, nmad
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
from .prior import GaussianPrior, MultivariateGaussianPrior

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


rad_to_deg = 180. / np.pi


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
        hduList.writeto(path, output_verify='silentfix', clobber=True)

    return hduList                  # For further handling if needed


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
    elif background_element == -1:
        # No background
        background_element = None
    else:
        raise SceneModelException("Unknown background_degree %d!" %
                                  background_degree)

    return background_element


class SnifsAdrElement(ConvolutionElement):
    """A PsfElement that applies atmospheric differential refraction to a
    scene.
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

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, wavelength,
                          adr_delta, adr_theta, **kwargs):
        if wavelength is None:
            raise SceneModelException("Must set wavelength for %s!" %
                                      type(self))
        if self.adr_model is None:
            raise SceneModelException("Must setup the ADR model for %s!" %
                                      type(self))

        adr_scale = self.adr_model.get_scale(wavelength) / self.spaxel_size

        shift_x = adr_delta * np.sin(adr_theta) * adr_scale
        shift_y = -adr_delta * np.cos(adr_theta) * adr_scale

        shift_fourier = np.exp(-1j * (shift_x * kx + shift_y * ky))

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

        adr_model = build_adr_model(pressure, temperature, delta=self['delta'],
                                    theta=self['theta'])

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
        self._add_parameter('ell_coord_x', 0.1, (-30., 30.), 'ELLX',
                            'Ellipticity coordinate in X direction')
        self._add_parameter('ell_coord_y', 0.1, (-30., 30.), 'ELLY',
                            'Ellipticity coordinate in Y direction')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, ell_coord_x,
                          ell_coord_y, **kwargs):
        # Infinitesimally small minor axis
        rho = 0.99

        gaussian = np.exp(-0.5 * (
            kx**2 * ell_coord_x**2 +
            ky**2 * ell_coord_y**2 +
            2. * kx * ky * rho * ell_coord_x * ell_coord_y
        ))

        return gaussian


class SnifsFourierSeeingPrior(GaussianPrior):
    """Add a prior on the seeing of this image to the model.

    The seeing_width parameter sets the seeing of the model. The relationship
    between this parameter and the seeing estimate from the SNIFS instrument
    was measured from a large set of extractions of standard stars on all
    photometric nights.
    """
    def __init__(self, seeing, seeing_key='seeing_width', **kwargs):
        predicted_width = -0.342 + 0.914 * seeing
        dispersion = 0.079

        super(SnifsFourierSeeingPrior, self).__init__(
            seeing_key, predicted_width, dispersion, **kwargs
        )


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

        keys = ['ell_width_x', 'ell_width_y']
        central_values = [predicted_ell_x, predicted_ell_y]
        covariance = [[dispersion_x**2, 0], [0, dispersion_y**2]]

        super(SnifsFourierEllipticityPrior, self).__init__(
            keys, central_values, covariance, **kwargs
        )


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

        if wavelength_dependence:
            spaxel_size = fits_header[prefix + 'SPXSZ']
            pressure, temperature = read_pressure_and_temperature(fits_header)
            scene_model = cls(use_empirical_parameters=False,
                              wavelength_dependence=True,
                              spaxel_size=spaxel_size, pressure=pressure,
                              temperature=temperature)
        else:
            scene_model = cls(use_empirical_parameters=False,
                              wavelength_dependence=False)

        # Read in the rest of the parameters
        scene_model.load_fits_header(fits_header, prefix=prefix,
                                     skip_parameters=['wavelength'])

        return scene_model


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
                 verbosity=0, **kwargs):
        """Initialize the fitter.

        path is the path to the fits cube that will be extracted.
        """
        self.verbosity = verbosity

        self._read_cube(path)
        self._setup_psf(psf, **kwargs)

        self.background_degree = background_degree
        self.subsampling = subsampling
        self.border = border
        self.least_squares = least_squares

        # Meta cube variables. These are set in fit_metaslices_2d
        self.meta_cube = None
        self.metaslice_info = None
        self.metaslice_guesses = None

        # 3D fit results
        self.fitter_3d = None
        self.fit_scene_model = None
        self.meta_cube_model = None
        self.reference_seeing = None

        # Extraction
        self.extraction = None

        if least_squares:
            raise SceneModelException("Least-squares is not implemented!")

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

    def _setup_psf(self, psf, **kwargs):
        """Set up a PSF to be fit. All PSF specific configuration should happen
        here.
        """
        scene_model_kwargs = {}

        if psf == "fourier":
            scene_model_class = SnifsFourierSceneModel
            seeing_degree = 1
            seeing_key = 'seeing_width'
            seeing_powerlaw_keys = ['seeing_ref_power', 'seeing_ref_width']
            global_fit_keys = ['ell_coord_x', 'ell_coord_y']
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

        self.psf = psf
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
                           (header.get('OBJECT', 'Unknown'), header['EFFTIME'],
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
        self.print_message("  Fit method: %s" % ('chi2' if self.least_squares
                                                 else 'least-squares'))

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
                # Crude Gaussian fit for initial position.
                gaussian_model = GaussianSceneModel(
                    meta_cube_data[idx], meta_cube_var[idx], subsampling=1,
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
                    variance=meta_cube_var[idx],
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
                raise

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
        valid_xrefs, valid_yrefs = self.adr_model.refract(
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
        rmax = 3 * nmad(r)
        good = np.zeros(num_meta_slices, dtype=bool)
        good[valid] = (r <= rmax)           # Valid fit and reasonable position
        bad = np.zeros(num_meta_slices, dtype=bool)
        bad[valid] = (r > rmax)             # Valid fit but discarded position
        if bad.any():
            print("  WARNING: %d metaslices discarded after ADR selection" %
                  (len(np.nonzero(bad))))

        # Estimate the seeing coefficients
        seeing_widths = extract_key(individual_scene_models, self.seeing_key)
        seeing_uncertainties = extract_key(individual_uncertainties,
                                           self.seeing_key)
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
            'guess_seeing_widths': guess_seeing_widths,
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
            'adr_delta': self.adr_model.delta,
            'adr_theta': self.adr_model.theta,
        }
        for key, value in zip(self.seeing_powerlaw_keys,
                              seeing_powerlaw_guesses):
            metaslice_guesses[key] = value

        # Save results
        self.meta_cube = meta_cube
        self.metaslice_info = metaslice_info
        self.metaslice_guesses = metaslice_guesses

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

        fitter = MultipleImageFitter(
            scene_model,
            meta_cube_data,
            meta_cube_var,
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
        print("Calculating covariance...")
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
        self.print_message(
            "  ADR fit: delta=%.2f±%.2f, theta=%.1f±%.1f deg" %
            (parameters['adr_delta'], uncertainties['adr_delta'],
             parameters['adr_theta'] * rad_to_deg, uncertainties['adr_theta'] *
             rad_to_deg), 1
        )

        self.print_message("  Effective airmass: %.2f" %
                           self.adr_model.get_airmass())

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
        model = fitter.evaluate()[:, meta_cube_model.i, meta_cube_model.j]
        meta_cube_model.data = model

        # Save results
        self.metaslice_info['fit_seeing_widths'] = fit_seeing_widths
        self.fitter_3d = fitter
        self.fit_scene_model = fit_scene_model
        self.meta_cube_model = meta_cube_model
        self.reference_seeing = reference_seeing

    def extract(self):
        # Make sure that the 3D fit has already been done.
        if self.fit_scene_model is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before extracting!"
            )

        print("Extracting the point-source spectrum...")

        cube_data, cube_var = cube_to_arrays(self.cube)

        extraction = self.fit_scene_model.extract(
            cube_data, cube_var, wavelength=self.cube.lbda
        )

        # Convert (mean) sky spectrum to "per arcsec**2"
        spaxel_size = self.cube.spxSize
        extraction['background_density'] = \
            extraction['background'] / spaxel_size**2
        extraction['background_density_variance'] = \
            extraction['background_variance'] / spaxel_size**4

        self.extraction = extraction

    def write_spectrum(self, output_path, sky_output_path,
                       prefix=config.default_fits_prefix):
        """Write the spectrum out to a fits file at the given path"""
        # Make sure that the extraction has already been done
        if self.extraction is None:
            raise SceneModelException(
                "Must run the 3D metaslice fit before extracting!"
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

        # Total sky flux (per arcsec**2)
        tflux = self.extraction['amplitude'].sum()
        sflux = self.extraction['background_density'].sum()

        header[p('TFLUX')] = (tflux, 'Total point-source flux')
        header[p('SFLUX')] = (sflux, 'Total sky flux/arcsec^2')
        header['SEEING'] = (self.reference_seeing,
                            'Estimated seeing @lbdaRef ["] (extract_star2)')

        print("TODO: add prior info to header")

        # Save the point source spectrum
        print("Saving ouptut point-source spectrum to '%s'" % output_path)
        point_source_spectrum = pySNIFS.spectrum(
            data=self.extraction['amplitude'],
            var=self.extraction['amplitude_variance'],
            start=self.cube.lbda[0],
            step=self.cube.lstep,
        )
        write_pysnifs_spectrum(point_source_spectrum, output_path, header)

        # Save the sky spectrum
        print("Saving ouptut sky spectrum to '%s'" % sky_output_path)
        sky_spectrum = pySNIFS.spectrum(
            data=self.extraction['background_density'],
            var=self.extraction['background_density_variance'],
            start=self.cube.lbda[0],
            step=self.cube.lstep,
        )
        write_pysnifs_spectrum(sky_spectrum, sky_output_path, header)
