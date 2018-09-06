# -*- coding: utf-8 -*-
from . import config
from .utils import SceneModelException
from .element import SubsampledModelComponent, PixelModelComponent, \
    PsfElement, Pixelizer, RealPixelizer
from .scene import SceneModel

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


class PointSource(SubsampledModelComponent):
    def _setup_parameters(self):
        self._add_parameter('amplitude', None, (None, None), 'AMP',
                            'Point source amplitude', coefficient=True)
        self._add_parameter('center_x', None, (None, None), 'POSX',
                            'Point source center position X')
        self._add_parameter('center_y', None, (None, None), 'POSY',
                            'Point source center position Y')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, amplitude,
                          center_x, center_y, **kwargs):
        # A delta function is a complex exponential in Fourier space.
        point_source_fourier = amplitude * np.exp(
            - 1j * (center_x * kx + center_y * ky)
        )

        return point_source_fourier


class GaussianPointSource(SubsampledModelComponent):
    """A point source convolved with a Gaussian.

    This can be evaluated entirely in real space, so it is useful to use this
    for simple Gaussian fits over a model where a point source is explicitly
    convolved with a Gaussian PSF.
    """
    def _setup_parameters(self):
        self._add_parameter('amplitude', None, (None, None), 'AMP',
                            'Point source amplitude', coefficient=True)
        self._add_parameter('center_x', None, (None, None), 'POSX',
                            'Point source center position X')
        self._add_parameter('center_y', None, (None, None), 'POSY',
                            'Point source center position Y')
        self._add_parameter('sigma_x', 1., (0.1, 20.), 'SIGX',
                            'Gaussian width in X direction')
        self._add_parameter('sigma_y', 1., (0.1, 20.), 'SIGY',
                            'Gaussian width in Y direction')
        self._add_parameter('rho', 0., (-1., 1.), 'RHO',
                            'Gaussian correlation')

    def _evaluate(self, x, y, subsampling, grid_info, amplitude, center_x,
                  center_y, sigma_x, sigma_y, rho, **kwargs):

        gaussian = np.exp(-0.5 / (1 - rho**2) * (
            (x - center_x)**2 / sigma_x**2 +
            (y - center_y)**2 / sigma_y**2 +
            -2. * x * y * rho / sigma_x / sigma_y
        ))

        # Normalize
        gaussian /= 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
        gaussian /= subsampling**2
        gaussian *= amplitude

        return gaussian


class Background(PixelModelComponent):
    def _setup_parameters(self):
        self._add_parameter('background', None, (None, None), 'BKG',
                            'Background', coefficient=True)

    def _evaluate(self, x, y, grid_info, background, **kwargs):
        return np.ones(x.shape) * background


class PolynomialBackground(PixelModelComponent):
    def __init__(self, background_degree=0, normalization_scale=10., *args,
                 **kwargs):
        self.background_degree = background_degree
        self.normalization_scale = normalization_scale
        super(PolynomialBackground, self).__init__(*args, **kwargs)

    def _setup_parameters(self):
        """Setup the polynomial background parameters.

        This is a 2-dimensional polynomial background. We label the flat
        background level parameters as background, and the higher order terms
        as background_x_y where x is the degree in the x-direction and y is the
        degree in the y direction. eg: background_1_2 is degree 1 in x and
        degree 2 in y.
        """
        self._add_parameter('background', None, (None, None), 'BKG',
                            'Background', coefficient=True)

        for x_degree in range(self.background_degree + 1):
            for y_degree in range(self.background_degree + 1 - x_degree):
                if x_degree == 0 and y_degree == 0:
                    # Already added the constant background.
                    continue

                self._add_parameter(
                    'background_%d_%d' % (x_degree, y_degree),
                    0.,
                    (None, None),
                    'BKG%d%d' % (x_degree, y_degree),
                    'Polynomial background, x-degree=%d, y-degree=%d' %
                    (x_degree, y_degree),
                    coefficient=True
                )

    def _evaluate(self, x, y, grid_info, background, **parameters):
        components = []

        # Zeroth order background
        components.append(background * np.ones(x.shape))

        # Normalize so that things vary on a reasonable scale
        norm_x = x / self.normalization_scale
        norm_y = y / self.normalization_scale

        # Polynomial background components
        for x_degree in range(self.background_degree + 1):
            for y_degree in range(self.background_degree + 1 - x_degree):
                if x_degree == 0 and y_degree == 0:
                    # Already added the constant background.
                    continue

                name = 'background_%d_%d' % (x_degree, y_degree)
                coefficient = parameters[name]
                component = (
                    coefficient * (norm_x**x_degree) * (norm_y**y_degree)
                )

                components.append(component)

        return components


class GaussianPsfElement(PsfElement):
    def _setup_parameters(self):
        self._add_parameter('sigma_x', 1., (0.1, 20.), 'SIGX',
                            'Gaussian width in X direction')
        self._add_parameter('sigma_y', 1., (0.1, 20.), 'SIGY',
                            'Gaussian width in Y direction')
        self._add_parameter('rho', 0., (-1., 1.), 'RHO',
                            'Gaussian correlation')

    def _evaluate(self, x, y, subsampling, grid_info, sigma_x, sigma_y, rho,
                  **kwargs):
        gaussian = np.exp(-0.5 / (1 - rho**2) * (
            x**2 / sigma_x**2 +
            y**2 / sigma_y**2 +
            -2. * x * y * rho / sigma_x / sigma_y
        ))

        # Normalize
        gaussian /= 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
        gaussian /= subsampling**2

        return gaussian

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, sigma_x,
                          sigma_y, rho, **kwargs):
        gaussian = np.exp(-0.5 * (
            kx**2 * sigma_x**2 +
            ky**2 * sigma_y**2 +
            2. * kx * ky * rho * sigma_x * sigma_y
        ))

        return gaussian


class GaussianMoffatPsfElement(PsfElement):
    def _setup_parameters(self):
        self._add_parameter('alpha', 2.5, (0.1, 15.), 'ALPHA', 'Moffat width')
        self._add_parameter('sigma', 1., (0.5, 5.), 'SIGMA', 'Gaussian width')
        self._add_parameter('beta', 2., (1.5, 50.), 'BETA', 'Moffat power')
        self._add_parameter('eta', 1., (0., None), 'ETA', 'Gaussian ratio')
        self._add_parameter('ell', 1., (0.2, 5.), 'E0', 'Ellipticity')
        self._add_parameter('xy', 0., (-0.6, 0.6), 'XY', 'XY coefficient')

    def _evaluate(self, x, y, subsampling, grid_info, alpha, sigma, beta, eta,
                  ell, xy, **kwargs):
        # Issue: with the pipeline parametrization, the radius can sometimes be
        # negative which is obviously not physical. If samples in that region
        # are chosen, set the signal to 0. The final result should never end up
        # here.
        if np.abs(xy) > np.sqrt(ell):
            return np.zeros(x.shape)

        r2 = x**2 + ell * y**2 + 2 * xy * x * y
        gaussian = np.exp(-0.5 * r2 / sigma**2)
        moffat = (1 + r2 / alpha**2)**(-beta)
        model = moffat + eta * gaussian

        model /= subsampling**2
        model /= (np.pi / np.sqrt(ell - xy**2) * (2 * eta * sigma**2 + alpha**2
                                                  / (beta - 1)))
        return model


class ExponentialPowerPsfElement(PsfElement):
    """A Psf model element that has a profile in Fourier space of
    exp(-(w * width)**power).

    When power is 5/3, this is a Kolmogorov PSF.
    """
    def _setup_parameters(self):
        self._add_parameter('power', 1.6, (0., 2.), 'POW', 'power')
        self._add_parameter('width', 0.5, (0.01, 30.), 'WID', 'width')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, power, width,
                          **kwargs):
        k = np.sqrt(kx*kx + ky*ky)

        fourier_profile = np.exp(-width**power * k**power)

        return fourier_profile


class KolmogorovPsfElement(ExponentialPowerPsfElement):
    """A Kolmogorov PSF.

    This is just an ExponentialPowerPsfElement with the power set to 5/3
    """
    def _setup_parameters(self):
        super(KolmogorovPsfElement, self)._setup_parameters()

        self.fix(power=5./3.)


class ChromaticExponentialPowerPsfElement(ExponentialPowerPsfElement):
    """A chromatic ExponentialPowerPsfElement to represent seeing.

    The width of the PSF takes the form:

        width = ref_width * (wave / ref_wave) ** (ref_power)

    So the full PSF profile in Fourier space is:

        width = exp(-(w * ref_width * (wave / ref_wave) ** (ref_power))**power)
    """
    def _setup_parameters(self):
        super(ChromaticExponentialPowerPsfElement, self)._setup_parameters()

        # Width is now a derived parameter
        self._modify_parameter('width', derived=True)

        self._add_parameter('wavelength', None, (None, None), 'WAVE',
                            'wavelength [A]', fixed=True, apply_prefix=False)
        self._add_parameter('ref_width', 1., (0.01, 30.), 'RWID',
                            'width at reference wavelength')
        self._add_parameter('ref_power', -0.3, (-2., 2.), 'RPOW',
                            'powerlaw power')

    def _calculate_derived_parameters(self, parameters):
        """Calculate the seeing width parameter using a power-law in wavelength
        """
        p = parameters

        # Ensure that all required variables have been set properly.
        if p['wavelength'] is None:
            raise SceneModelException("Must set wavelength!")

        p['width'] = (
            p['ref_width'] *
            (p['wavelength'] / config.reference_wavelength)**p['ref_power']
        )

        # Update parameters from the superclass
        parent_parameters = super(ChromaticExponentialPowerPsfElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


class GaussianSceneModel(SceneModel):
    """A scene model of a point source convolved with a Gaussian.

    We use a GaussianPointSource object which evaluates the convolution of the
    Gaussian with a point source in real space so that we never have to convert
    to Fourier space and back. This speeds up computation time, and means that
    no buffer is needed for this scene model. We do however sample using a
    RealPixelizer (that doesn't convolve with the pixel in Fourier space), so a
    fine subsampling is recommended if accurate results are desired.
    """
    def __init__(self, *args, **kwargs):
        elements = [
            GaussianPointSource,
            RealPixelizer,
            Background
        ]

        super(GaussianSceneModel, self).__init__(elements, *args, **kwargs)


class KolmogorovSceneModel(SceneModel):
    """A scene model of a point source convolved with a Kolmogorov PSF."""
    def __init__(self, *args, **kwargs):
        elements = [
            PointSource,
            KolmogorovPsfElement,
            Pixelizer,
            Background
        ]

        super(KolmogorovSceneModel, self).__init__(elements, *args, **kwargs)
