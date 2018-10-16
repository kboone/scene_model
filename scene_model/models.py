# -*- coding: utf-8 -*-
from __future__ import print_function

from . import config
from .utils import SceneModelException
from .element import SubsampledModelComponent, PixelModelComponent, \
    PsfElement, Pixelizer, RealPixelizer
from .scene import SceneModel

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


class PointSource(SubsampledModelComponent):
    def _setup_parameters(self):
        self._add_parameter('amplitude', 1., (None, None), 'AMP',
                            'Point source amplitude', coefficient=True)
        self._add_parameter('center_x', 0., (None, None), 'POSX',
                            'Point source center position X')
        self._add_parameter('center_y', 0., (None, None), 'POSY',
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
        self._add_parameter('center_x', None, (None, None), 'XC',
                            'Point source center position X')
        self._add_parameter('center_y', None, (None, None), 'YC',
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


class SimpleGaussianPointSource(GaussianPointSource):
    """A Gaussian point source with no ellipticity.

    See GaussianPointSource for details. We fix rho to 0, and force sigma_x to
    be equal to sigma_y, removing 2 parameters from the fit.
    """
    def _setup_parameters(self):
        super(SimpleGaussianPointSource, self)._setup_parameters()

        # sigma_y and sigma_x are now just one sigma parameter.
        self._add_parameter('sigma', 1., (0.1, 20.), 'SIG', 'Gaussian width')
        self._modify_parameter('sigma_x', derived=True)
        self._modify_parameter('sigma_y', derived=True)
        self.fix(rho=0)

    def _calculate_derived_parameters(self, parameters):
        p = parameters

        p['sigma_x'] = p['sigma']
        p['sigma_y'] = p['sigma']

        # Update parameters from the superclass
        parent_parameters = super(SimpleGaussianPointSource, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


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
        self._add_parameter('sigma_x', 1., (0.01, 20.), 'SIGX',
                            'Gaussian width in X direction')
        self._add_parameter('sigma_y', 1., (0.01, 20.), 'SIGY',
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
        self._add_parameter('width', 0.5, (0.001, 30.), 'WID', 'width')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, power, width,
                          **kwargs):
        k = np.sqrt(kx*kx + ky*ky)

        fourier_profile = np.exp(-width**power * k**power)

        return fourier_profile


class DeltaExponentialPsfElement(PsfElement):
    """A Psf model element that is the sum of a delta function and a
    Fourier exponential profile .
    """
    def _setup_parameters(self):
        self._add_parameter('delta_fraction', 0.5, (0., 1.), 'DELT',
                            'delta function fraction')
        self._add_parameter('power', 1.6, (0., 2.), 'POW', 'power')
        self._add_parameter('width', 0.5, (0.001, 30.), 'WID', 'width')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, delta_fraction,
                          power, width, **kwargs):
        k = np.sqrt(kx*kx + ky*ky)

        fourier_profile = (
            delta_fraction +
            (1 - delta_fraction) * np.exp(-width**power * k**power)
        )

        return fourier_profile


class FourierMoffatPsfElement(PsfElement):
    """A Psf model element that has a Fourier profile of a Moffat distribution.

    This is not theoretically motivated in any way.
    """
    def _setup_parameters(self):
        self._add_parameter('alpha', 1., (0.0001, 100.), 'ALPHA', 'alpha')
        self._add_parameter('beta', 2., (0., 5.), 'BETA', 'beta')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, alpha, beta,
                          **kwargs):
        k2 = kx*kx + ky*ky

        fourier_profile = (1 + k2 / alpha**2)**(-beta)

        return fourier_profile


class KolmogorovPsfElement(ExponentialPowerPsfElement):
    """A Kolmogorov PSF.

    This is just an ExponentialPowerPsfElement with the power set to 5/3
    """
    def _setup_parameters(self):
        super(KolmogorovPsfElement, self)._setup_parameters()

        self.fix(power=5./3.)


class VonKarmanPsfElement(PsfElement):
    """VonKarman PSF.

    In Fourier space, this has the form:

        r0^(-5/3) * (f^2 + L0^-2)^(-11/6)

    where r0 is the Fried parameter and L0 is the outer scale.
    """
    def _setup_parameters(self):
        self._add_parameter('r0', 1., (0., None), 'R0',
                            'von Karman Fried parameter')
        self._add_parameter('L0', 20., (1., None), 'L0',
                            'von Karman outer scale')

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, r0, L0,
                          **kwargs):
        k = np.sqrt(kx*kx + ky*ky)
        # fourier_profile = np.exp(-r0**(5/3.) * (k**2 + L0**-2)**(11/6.))
        # fourier_profile = np.exp(
            # -(L0 / r0)**(5/3.) * (
                # + 1.87439 * (k / L0)**(5/3.)
                # - 1.50845 * (k / L0)**2.
            # )
        # )
        from scipy.special import kv, gamma
        fourier_profile = np.exp(
            -(L0 / r0)**(5/3.) * (
                gamma(5/6.) / 2**(1/6.) -
                (k / L0)**(5/6.) * kv(5/6., k / L0)
            )
        )

        # Set the (0, 0) bin to 1. The profile has that as the limit, but the
        # above calculation will give nan.
        fourier_profile[0, 0] = 1.

        return fourier_profile


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


class SimpleGaussianSceneModel(SceneModel):
    """A scene model of a point source convolved with a Gaussian without
    ellipticity.

    See GaussianSceneModel for details.
    """
    def __init__(self, *args, **kwargs):
        elements = [
            SimpleGaussianPointSource,
            RealPixelizer,
            Background
        ]

        super(SimpleGaussianSceneModel, self).__init__(elements, *args,
                                                       **kwargs)


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
