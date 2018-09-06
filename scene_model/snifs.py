# -*- coding: utf-8 -*-
from . import config
from .utils import SceneModelException
from .element import ConvolutionElement, PsfElement, Pixelizer, RealPixelizer
from .scene import SceneModel
from .models import GaussianMoffatPsfElement, \
    PointSource, \
    Background, \
    GaussianPsfElement, \
    ExponentialPowerPsfElement, \
    ChromaticExponentialPowerPsfElement

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


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

        from ToolBox.Atmosphere import ADR
        self.adr_model = ADR(pressure, temperature)
        self.spaxel_size = spaxel_size


class SnifsOldPsfElement(GaussianMoffatPsfElement):
    """Model of the PSF as implemented in the pipeline.

    Don't use this class directly, subclass it instead with the following
    parameters set.
    """
    def __init__(self, exposure_time, *args, **kwargs):
        """Initialize the PSF. The PSF parameters depend on the exposure time.
        """
        super(SnifsOldPsfElement, self).__init__(*args, **kwargs)

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
        super(SnifsOldPsfElement, self)._setup_parameters()
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
        parent_parameters = super(SnifsOldPsfElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


class ChromaticSnifsOldPsfElement(SnifsOldPsfElement):
    """A chromatic SnifsOldPsfElement with seeing dependence on wavelength.

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

        super(ChromaticSnifsOldPsfElement, self).__init__(*args, **kwargs)

    def _setup_parameters(self):
        super(ChromaticSnifsOldPsfElement, self)._setup_parameters()

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
        alpha = alpha_params[-1] * x**(np.polyval(alpha_params[:-1], x-1))

        # Add in the new parameters
        p['alpha'] = alpha

        # Update parameters from the superclass
        parent_parameters = super(ChromaticSnifsOldPsfElement, self).\
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


class SnifsFourierSceneModel(SceneModel):
    """A model of the PSF for the SNIFS IFU.

    This model was empirically determined, and it contains:
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
    """
    def __init__(self, image=None, variance=None,
                 use_empirical_parameters=True, wavelength_dependence=False,
                 adr_model=None, spaxel_size=None, pressure=None,
                 temperature=None, **kwargs):
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
        background_element = Background()
        elements.append(background_element)

        if wavelength_dependence:
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


class SnifsOldSceneModel(SceneModel):
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
    """
    def __init__(self, image=None, variance=None, exposure_time=None,
                 alpha_degree=2, wavelength_dependence=False, adr_model=None,
                 spaxel_size=None, pressure=None, temperature=None, **kwargs):
        """Build the elements of the PSF model."""
        elements = []

        if exposure_time is None:
            raise SceneModelException(
                "Must specify exposure time for SnifsOldSceneModel"
            )

        # Point source for the supernova
        point_source_element = PointSource()
        elements.append(point_source_element)

        # Gaussian + Moffat PSF.
        if wavelength_dependence:
            psf_element = ChromaticSnifsOldPsfElement(alpha_degree,
                                                      exposure_time)
        else:
            psf_element = SnifsOldPsfElement(exposure_time)
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
        background_element = Background()
        elements.append(background_element)

        if wavelength_dependence:
            shared_parameters = ['wavelength']
        else:
            shared_parameters = []

        super(SnifsOldSceneModel, self).__init__(
            elements=elements,
            image=image,
            variance=variance,
            shared_parameters=shared_parameters,
            **kwargs
        )
