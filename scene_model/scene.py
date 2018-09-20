# -*- coding: utf-8 -*-
from __future__ import print_function

from astropy.table import Table
from scipy.optimize import minimize
from collections import OrderedDict
from scipy.linalg import LinAlgError

from . import config
from . import fit
from .utils import DumbLRUCache, SceneModelException, nmad
from .utils import print_parameter_header, print_parameter
from .element import fourier_to_real, ModelElement, PsfElement

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np
if config.use_autograd:
    from autograd import grad


class SceneModel(object):
    """Model of a scene.

    This model represents everything that is in an image. It is built up of
    several elements, which can be point sources, backgrounds, detector
    artifacts, etc.

    The elements are applied sequentially, so they need to be defined in the
    proper order. Each element is applied to the output of the previous one,
    and can be applied however is appropriate.

    For example, the elements could be [Galaxy, PointSource, PSF, ADR,
    Background]. The Galaxy element creates a component representing a galaxy.
    The PointSource element adds a point source component to the model. The PSF
    element convolves the components of the model with a PSF. The ADR element
    applies a wavelength-dependent position shift to the model components.
    Finally, the Background element adds a background component to the model.
    """
    def __init__(self, elements, image=None, variance=None, grid_size=None,
                 priors=[], prior_scale=1.,
                 subsampling=config.default_subsampling,
                 border=config.default_border, shared_parameters=[], **kwargs):
        """Initialize a scene model.

        elements is a list of either ModelElement classes or instances which
        will be used to build the scene. If a class is passed, an instance of
        that class will be created with no parameters passed to the
        initializer.

        The scene model can be initialized either with a grid size over which
        the scene will be evaluated, or with an image that represents data that
        the scene model will be fit to. If a grid size is passed, then
        grid_size should be a tuple with the y and x sizes of the image. If an
        image is passed, then the variance of that image can optionally be
        specified. If neither an image nor a grid_size is specified, then the
        model can't be evaluated (although these can be set up later with
        methods like load_image or set_grid_size).

        The internal models will be calculated using the given subsampling and
        border relative to the initial grid.
        """
        self.fit_scale = 1.
        self.fit_result = None
        self.fit_initial_values = None

        self._chi_square_cache = DumbLRUCache()

        self.subsampling = subsampling
        self.border = border

        self.elements = []
        self.priors = priors
        self.prior_scale = prior_scale
        self.shared_parameters = shared_parameters

        for element in elements:
            if not isinstance(element, ModelElement):
                # We don't have a ModelElement instance. Assume that whatever
                # we were passed acts like a class and generates a ModelElement
                # instance when it is called.
                element = element()

            # Make sure that we don't duplicate any parameters.
            model_parameters = self.parameters
            for parameter in element._parameter_info:
                if parameter in model_parameters:
                    if parameter in self.shared_parameters:
                        # This parameter is a shared parameter and it has
                        # already been added by another element. This is fine.
                        continue

                    raise SceneModelException(
                        "Model has two parameters with the name %s! (second "
                        " in %s). Either add a prefix with "
                        "ModelElement(prefix='my_prefix') for two separate "
                        "parameters or add this to the shared_parameters list."
                        % (parameter, element)
                    )

            # Everything looks ok, add the element.
            self.elements.append(element)

        if image is not None:
            self.load_image(image, variance, **kwargs)
        elif grid_size is not None:
            self.setup_grid(grid_size)

        # Clear the cache to set up cache-related structures.
        self.clear_cache()

    def _create_grid(self, grid_size, border, subsampling):
        """Create a grid that the model can be evaluated on.

        grid_size should be a tuple with the y and x dimensions.

        This returns a dictionary with all of the different variables needed to
        evaluate the model in both real and Fourier space.
        """
        grid_width_y, grid_width_x = grid_size

        # Set up an ij grid which has integer indexing of the pixels starting
        # in the lower-left corner of the image.
        grid_i, grid_j = np.meshgrid(
            np.arange(grid_width_x), np.arange(grid_width_y), indexing='ij'
        )

        # Set up an xy grid so that the center pixel is at (0, 0). This
        # convention works with what SNIFS uses.
        reference_i = (grid_width_x - 1.) / 2.
        reference_j = (grid_width_y - 1.) / 2.
        remaining_x = grid_width_x - 1. - reference_i
        remaining_y = grid_width_y - 1. - reference_j

        grid_x = grid_i - reference_i
        grid_y = grid_j - reference_j

        # Build an oversampled and padded grid to use for the actual
        # calculations. This grid is defined in both real and Fourier space
        # since we will handle convolutions with Fourier transformations.
        step = 1. / subsampling
        subsampling_buffer = (subsampling - 1) / 2. * step

        pad_start_x = -subsampling_buffer - border - reference_i
        pad_start_y = -subsampling_buffer - border - reference_j
        pad_end_x = subsampling_buffer + border + remaining_x + step / 2.
        pad_end_y = subsampling_buffer + border + remaining_y + step / 2.

        pad_range_x = np.arange(pad_start_x, pad_end_x, step)
        pad_range_y = np.arange(pad_start_y, pad_end_y, step)

        pad_grid_x, pad_grid_y = np.meshgrid(pad_range_x, pad_range_y,
                                             indexing='ij')

        kx = 2 * np.pi * np.fft.fftfreq(pad_grid_x.shape[0], step)
        ky = 2 * np.pi * np.fft.fftfreq(pad_grid_x.shape[1], step)
        pad_grid_kx, pad_grid_ky = np.meshgrid(kx, ky, indexing='ij')

        subsampling_buffer = subsampling_buffer
        pad_grid_x = pad_grid_x
        pad_grid_y = pad_grid_y
        pad_grid_kx = pad_grid_kx
        pad_grid_ky = pad_grid_ky

        # Calculate the offset that the center position needs to be moved by to
        # be in the right location on the padded grid.
        pad_offset_x = border + subsampling_buffer + reference_i
        pad_offset_y = border + subsampling_buffer + reference_j

        # Calculate the Fourier shift that needs to be applied to get an FFTed
        # Fourier space model to the reference position and back.
        pad_ifft_shift = np.exp(
            - 1j * (pad_offset_x * pad_grid_kx + pad_offset_y * pad_grid_ky)
        )
        pad_fft_shift = np.conj(pad_ifft_shift)

        # Build a string that uniquely defines the grid so that two grid_infos
        # can be compared for equality.
        unique_str = '%d,%s,%d,%.6f,%.6f' % (
            subsampling, grid_size, border, reference_i, reference_j
        )

        # Build a structure that holds all of the grid information
        grid_info = {
            'grid_size': grid_size,
            'subsampling': subsampling,
            'border': border,

            'grid_i': grid_i,
            'grid_j': grid_j,
            'grid_x': grid_x,
            'grid_y': grid_y,

            'reference_i': reference_i,
            'reference_j': reference_j,

            'subsampling_buffer': subsampling_buffer,
            'pad_grid_x': pad_grid_x,
            'pad_grid_y': pad_grid_y,
            'pad_grid_kx': pad_grid_kx,
            'pad_grid_ky': pad_grid_ky,

            'pad_offset_x': pad_offset_x,
            'pad_offset_y': pad_offset_y,

            'pad_fft_shift': pad_fft_shift,
            'pad_ifft_shift': pad_ifft_shift,

            'unique_str': unique_str,
        }

        return grid_info

    def setup_grid(self, grid_size, border=None, subsampling=None):
        """Setup the model to be evaluated on a specific sized grid.

        If border and subsampling aren't specified, they are taken from
        self.border and self.subsampling.

        See _create_grid for details, this is a small wrapper around that
        method that saves the grid internally.
        """
        if border is None:
            border = self.border
        else:
            self.border = border

        if subsampling is None:
            subsampling = self.subsampling
        else:
            self.subsampling = subsampling

        grid_info = self._create_grid(grid_size, border, subsampling)

        self.grid_info = grid_info

        # Reset the cache. Everything will need to be recalculated after
        # setting a new grid.
        self.clear_cache()

    def set_prior_initial_values(self):
        """Set the initial values for parameters from the priors"""
        if not self.priors:
            # No priors, nothing to do.
            return

        parameters = self.parameters
        for prior in self.priors:
            parameters = prior.update_initial_values(parameters)
        self.set_parameters(update_derived=False, **parameters)

    def load_image(self, image, variance=None, saturation_level=None,
                   apply_scale=True, set_initial_guesses=True,
                   amplitude_key='amplitude', background_key='background',
                   center_x_key='center_x', center_y_key='center_y'):
        """Load an image to fit.

        This will load an image and guess some of the initial values.
        SceneModel_process_image will estimate the brightness and position of
        the brightest source in the image. If set_initial_guesses is set, these
        will be used to initialize the parameters of the model. By default, the
        center_x, center_y, amplitude and background parameters are set. These
        keywords can be changed with the parameters to this function. If some
        of these parameters don't exist in the model, then they are simply not
        set.
        """
        # Build a grid of x and y coordinates for the image
        self.setup_grid(image.shape)

        image_fit_data = self._process_image(image, variance, saturation_level)

        self.image = image_fit_data['image']
        self.variance = image_fit_data['variance']
        self.fit_mask = image_fit_data['mask']
        self.fit_x = image_fit_data['mask_x']
        self.fit_y = image_fit_data['mask_y']
        self.fit_data = image_fit_data['mask_data']
        self.fit_variance = image_fit_data['mask_variance']
        self.fit_scale = image_fit_data['scale']

        self.fit_model = None
        self.fit_result = None
        self.fit_initial_values = None

        # Set the initial amplitude and background parameters from the estimate
        if set_initial_guesses:
            guess_map = {
                amplitude_key: 'amplitude_guess',
                background_key: 'background_guess',
                center_x_key: 'center_x_guess',
                center_y_key: 'center_y_guess',
            }

            for parameter_name, guess_name in guess_map.items():
                if parameter_name in self._parameter_info:
                    self[parameter_name] = image_fit_data[guess_name]

        # Update initial values from the priors
        self.set_prior_initial_values()

    def _process_image(self, image, variance=None, saturation_level=None,
                       background_fraction=0.5):
        """Process an image to select valid pixels and estimate scales.

        This returns a dictionary with all of the relevant information from the
        image. Masked arrays for the image, variance and x and y coordinates
        are all included in this array.

        Additionally, this method estimates the brightness of the brightest
        object in the image. This is used to set a rough scale for the image
        when fitting (so that the amplitudes are O(1)). The position of that
        brightest object is also estimated.

        If variance is not specified, then a flat variance level is estimated
        for this image (equivalent to least-squares)

        If saturation_level is specified, pixels above that level will be
        masked out.
        """
        # Ensure that the model is set up to handle this size image.
        if self.grid_info['grid_size'] != image.shape:
            raise SceneModelException(
                'Image passed is a different size than the model is set up '
                'for! Run scene_model.load_image() to set up the model for '
                'this image size before whatever else you were trying to do.'
            )

        # Mask out invalid pixels.
        mask = ~np.isnan(image)

        # If the variance is None, then we are doing a least-squares
        # fit/extraction. The "variance" for fitting purposes will therefore
        # be a flat array, but we can pick a non-zero scale to multiply
        # everything by to make the fitting nicer. We choose the measured NMAD
        # of the image to be this scale, which should give somewhat reasonable
        # numbers.
        if variance is None:
            noise_scale = nmad(image[mask])**2
            use_variance = np.ones(image.shape) * noise_scale
        else:
            # Ignore nans here. nan > 0 evaluates to False which is what we
            # want.
            with np.errstate(invalid='ignore'):
                mask = mask & (variance > 0)
            use_variance = variance

        if saturation_level is not None:
            # Mask out saturated pixels in the fit.
            mask = mask & (image < saturation_level)

        grid_x = self.grid_info['grid_x']
        grid_y = self.grid_info['grid_y']

        mask_x = grid_x[mask]
        mask_y = grid_y[mask]
        mask_data = image[mask]
        mask_variance = use_variance[mask]

        # We need to have some estimate of the brightness of the image for
        # numerical reasons. The default levels are ~1e-14, which is
        # troublesome for numerical precision. Here, we estimate the brightness
        # which can be used as a scale parameter for other algorithms.

        # Estimate the background by taking the median of the pixels farthest
        # from the center of the image.
        r2 = mask_x**2 + mask_y**2
        background_mask = r2 > np.percentile(r2, 100*(1-background_fraction))
        source_mask = ~background_mask

        background_guess = np.median(mask_data[background_mask])

        # Estimate the amplitude by taking the sum of what is left over
        # after taking out the sky. This could be negative, so we set a
        # lower bound at a multiple of an estimate of the noise of the
        # image. Getting this exactly right is not important, we just need
        # order of magnitude.
        amplitude_guess = np.sum(mask_data - background_guess)
        min_amplitude_guess = nmad(mask_data) * 10

        scale = max(amplitude_guess, min_amplitude_guess)

        # Reestimate the center position using a weighted average of the
        # absolute flux in the region near the detected object.
        sub_data = mask_data - background_guess
        weights = np.abs(sub_data[source_mask])
        center_x_guess = np.average(mask_x[source_mask], weights=weights)
        center_y_guess = np.average(mask_y[source_mask], weights=weights)

        result = {
            'image': image,
            'variance': variance,

            'mask': mask,
            'mask_x': mask_x,
            'mask_y': mask_y,
            'mask_data': mask_data,
            'mask_variance': mask_variance,

            'scale': scale,
            'amplitude_guess': amplitude_guess,
            'background_guess': background_guess,
            'center_x_guess': center_x_guess,
            'center_y_guess': center_y_guess,
        }

        return result

    @property
    def parameters(self):
        """Return a list of the current parameter values.
        """
        full_parameters = dict()
        for element in self.elements:
            for name, value in element.parameters.items():
                # Use the first entry for shared parameters
                if name not in full_parameters:
                    full_parameters[name] = value

        return full_parameters

    @property
    def _parameter_info(self):
        """Return an ordered dictionary of the full information for each of the
        parameters.

        See SceneModel.parameters for details on the ordering.
        """
        parameter_info = OrderedDict()
        for element in self.elements:
            for name, parameter_dict in element._parameter_info.items():
                # Use the first entry for shared parameters
                if name not in parameter_info:
                    parameter_info[name] = parameter_dict

        return parameter_info

    @property
    def coefficients(self):
        """Return an ordered dictionary of the coefficient parameters that
        scale each component of the model.
        """
        coefficients = dict()
        for element in self.elements:
            for name, value in element.coefficients.items():
                # Use the first entry for shared parameters
                if name not in coefficients:
                    coefficients[name] = value

        return coefficients

    @property
    def _coefficient_info(self):
        """Return an ordered dictionary of the full information for each of the
        coefficient parameters
        """
        coefficient_info = OrderedDict()
        for element in self.elements:
            for name, parameter_dict in element._coefficient_info.items():
                # Use the first entry for shared parameters
                if name not in coefficient_info:
                    coefficient_info[name] = parameter_dict

        return coefficient_info

    def get_parameter_element(self, parameter_name):
        """Find the element that is responsible for a certain parameter"""
        for element in self.elements:
            if parameter_name in element._parameter_info:
                return element

        # Didn't find it.
        raise SceneModelException("No parameter %s found!" % parameter_name)

    def evaluate(self, grid_info=None, separate_components=False,
                 apply_coefficients=True, return_full_parameters=False,
                 apply_mask=False, **parameters):
        """Evaluate the scene model with the given parameters.

        This adds in the amplitude and background to the PSF model that comes
        out of evaluate_psf
        """
        # Use stored grid_info by default. A grid_info generated by
        # self.setup_grid can optionally be explicitly passed to evaluate the
        # model on a different grid than the stored one (eg: for producing
        # oversampled models).
        if grid_info is None:
            grid_info = self.grid_info

        # Set up the model
        if separate_components:
            model = []
        else:
            model = 0.
        mode = None

        # Add in any shared parameters if they aren't explicitly in the
        # parameters dictionary.
        for parameter_name in self.shared_parameters:
            if parameter_name not in parameters:
                element = self.get_parameter_element(parameter_name)
                value = element[parameter_name]
                parameters[parameter_name] = value

        for element in self.elements:
            if config.debug_fourier:
                print(type(element).__name__)

            old_mode = mode
            model, mode, parameters = element.update_model(
                grid_info, model, mode, parameters, separate_components,
                apply_coefficients
            )

            if config.debug_fourier:
                print("    %s -> %s" % (old_mode, mode))

        if config.debug_fourier:
            print("")

        # Warn the user if we don't end up with a pixelated image in the end.
        # If the image isn't pixelized, then something went wrong and the
        # output won't be usable.
        if mode != 'pixel':
            print("WARNING: SceneModel didn't produce a pixelated image! Got "
                  "%s instead" % mode)

        if apply_mask:
            # Apply the fit mask for this image
            model = model[self.fit_mask]

        if return_full_parameters:
            return model, parameters
        else:
            return model

    def set_parameters(self, update_derived=True, **kwargs):
        for key, value in kwargs.items():
            element = self.get_parameter_element(key)
            element.set_parameters(update_derived=update_derived,
                                   **{key: value})

    def __getitem__(self, parameter):
        """Return the value of a given parameter"""
        element = self.get_parameter_element(parameter)
        return element[parameter]

    def __setitem__(self, parameter, value):
        """Set the value of a given parameter"""
        self.set_parameters(**{parameter: value})

    def fix(self, **kwargs):
        for key, value in kwargs.items():
            element = self.get_parameter_element(key)
            element.fix(model_name=True, **{key: value})

    def _modify_parameter(self, name, **kwargs):
        element = self.get_parameter_element(name)
        element._modify_parameter(name, model_name=True, **kwargs)

    def clear_cache(self):
        """Clear the internal caches."""
        # Cache of the chi square output.
        self._chi_square_cache.clear()

        for element in self.elements:
            element.clear_cache()

    def chi_square(self, parameters={}, return_full_info=False,
                   do_analytic_coefficients=True, apply_fit_scale=False,
                   apply_priors=False):
        """Calculate the chi-square value for a given set of model parameters.

        If apply_fit_scale is True, then the image is scaled by a predetermined
        amount that was calculated in _process_image. This scale is set to keep
        the amplitude and similar parameters around 1 for numerical reasons
        during fitting. The chi_square will then have a minimum with an
        amplitude of approximately 1. The true amplitude can be recovered by
        multiplying the recovered amplitude by self.fit_scale.
        """
        # Cache the output of this function. If it is called again with the
        # same parameters, then it is not recalculated. This speeds up
        # computing time dramatically when fitting multiple images because we
        # can then modify the parameters of a single image's model without
        # having to recalculate all of the other models. In Python3, there is a
        # nice lru_cache decorator in the standard library, but it doesn't
        # exist in the old Python at the CC. We just roll our own simple single
        # element cache here which works fine.
        do_cache = True
        if config.use_autograd:
            # If we are using autograd, we need to make sure to not cache
            # anything if we are doing autograd calculations.
            for key, value in parameters.items():
                if isinstance(value, np.numpy_boxes.ArrayBox):
                    do_cache = False
                    break

        if do_cache:
            # Check if these parameters are in the LRU cache
            call_info = {
                'parameters': parameters,
                'return_full_info': return_full_info,
                'do_analytic_coefficients': do_analytic_coefficients,
                'apply_fit_scale': apply_fit_scale,
                'apply_priors': apply_priors,
            }
            cache_value = self._chi_square_cache[call_info]
            if cache_value is not None:
                return cache_value

        # Get the data to use in the chi square evaluation, and scale it if
        # desired.
        use_data = self.fit_data
        use_variance = self.fit_variance
        use_mask = self.fit_mask

        if apply_fit_scale:
            use_data = use_data / self.fit_scale
            use_variance = use_variance / self.fit_scale**2

        # Evaluate the model
        if do_analytic_coefficients:
            # The coefficients need to be evaluated analytically. We first
            # evaluate all of the model components without the coefficients
            # applied, and then we calculate the coefficients using an analytic
            # least-squares fit.
            components, eval_parameters = self.evaluate(
                separate_components=True, apply_coefficients=False,
                return_full_parameters=True, **parameters
            )

            # Evaluate the coefficients analytically
            coef_names, coef_values, coef_variances = \
                self._calculate_coefficients(
                    components, use_data, use_variance, use_mask,
                    **eval_parameters
                )
            coef_dict = dict(zip(coef_names, coef_values))
            full_parameters = dict(eval_parameters, **coef_dict)

            # Build the final model
            model = self._apply_coefficients(components, **full_parameters)
        else:
            model, full_parameters = self.evaluate(
                return_full_parameters=True, **parameters
            )

        fit_model = model[use_mask]

        chi_square = np.sum((use_data - fit_model)**2 / use_variance)

        if apply_priors:
            prior_penalty = self._evaluate_priors(full_parameters)
            chi_square += prior_penalty

        if return_full_info:
            return_val = chi_square, fit_model, full_parameters
        else:
            return_val = chi_square

        # Update the cache
        if do_cache:
            self._chi_square_cache[call_info] = return_val

        return return_val

    def _evaluate_priors(self, full_parameters=None):
        """Evaluate the priors and return the total penalty on the chi-square.

        This returns the sum of all of the prior effects multiplied by the
        prior scale.

        full_parameters should be a dictionary of parameters that at least
        contains every parameter that is used for the priors. If not specified,
        it will be pulled directly from the model.
        """
        if full_parameters is None:
            full_parameters = self.parameters

        prior_penalty = 0.
        for prior in self.priors:
            prior_penalty += prior.evaluate(full_parameters)
        prior_penalty *= self.prior_scale

        return prior_penalty

    def _calculate_coefficients(self, components, data, variance, mask,
                                **parameters):
        """Calculate the coefficients analytically given a list of components.

        This returns a list of component names, an array with the coefficient
        values and an array with the coefficient statistical variances. Note
        that the returned variances do not take into account systematic
        uncertainties from fitting the model!
        """
        coefficient_info = self._coefficient_info

        # Figure out which parameters need to be fit, and subtract any fixed
        # parameters from the data.
        model = 0.
        basis = []
        parameter_names = []
        for component, (parameter_name, parameter_dict) in \
                zip(components, coefficient_info.items()):
            mask_component = component[mask]
            if parameter_dict['fixed']:
                # Don't need to fit this parameter, add it to the model
                if parameter_name in parameters:
                    value = parameters[parameter_name]
                else:
                    value = parameter_dict['value']
                value = parameter_dict['value']
                model += mask_component * value
            else:
                # Need to fit this parameter, add it to the basis
                basis.append(mask_component)
                parameter_names.append(parameter_name)

        if len(parameter_names) == 0:
            # There are no coefficients that need to be analytically evaluated,
            # we're done.
            return {}, {}

        basis = np.vstack(basis).T
        weight = 1 / np.sqrt(variance)

        use_data = data - model

        A = basis * weight[..., np.newaxis]
        b = weight * use_data

        alpha = np.dot(A.T, A)
        beta = np.dot(A.T, b)

        try:
            cov = np.linalg.inv(alpha)
        except LinAlgError:
            # Invalid model where the model parts are degenerate with each
            # other. This shouldn't happen. One common scenario is that the PSF
            # position has drifted far out of the image compared to its width,
            # and it evaluates to 0 everywhere.
            raise SceneModelException("Found degenerate model for parameters: "
                                      "%s" % parameters)
        values = np.dot(cov, beta)
        variances = np.diag(cov)

        return parameter_names, values, variances

    def _apply_coefficients(self, components, **parameters):
        """Apply the coefficients for each component to get the full model."""
        coefficient_info = self._coefficient_info
        model = 0.
        for component, (parameter_name, parameter_dict) in \
                zip(components, coefficient_info.items()):
            if parameter_name in parameters:
                value = parameters[parameter_name]
            else:
                value = parameter_dict['value']
            model += component * value

        return model

    def _get_fit_parameters(self, do_analytic_coefficients=True,
                            apply_fit_scale=True):
        """Return the information needed to do a fit with a flat parameter
        vector (what scipy needs)

        If do_analytic_coefficients is True, then all coefficient parameters
        (eg: amplitude, background) are analytically solved for and aren't
        included in the list of fit parameters.

        If apply_fit_scale is True, then all scale parameters are scaled by the
        internal scale parameter (an estimate of the amplitude).

        This method returns an OrderedDict with the parameter names as keys and
        parameter dictionary as the values with all of the relevant
        information. See _add_parameter for details on the keys in that
        dictionary.
        """
        parameters = OrderedDict()

        if apply_fit_scale:
            fit_scale = self.fit_scale
        else:
            fit_scale = 1.

        for parameter_name, parameter_dict in self._parameter_info.items():
            try:
                if (parameter_dict['fixed'] or parameter_dict['derived'] or
                        (parameter_dict['coefficient'] and
                         do_analytic_coefficients)):
                    # Parameter that isn't fitted, ignore it.
                    continue
            except:
                from IPython import embed; embed()

            new_dict = parameter_dict.copy()

            if parameter_dict['coefficient']:
                # Parameter that is the coefficient for a component. Scale it
                # by the predetermined amount.
                if new_dict['value'] is not None:
                    new_dict['value'] = new_dict['value'] / fit_scale
                lower_bound, upper_bound = new_dict['bounds']
                if lower_bound is not None:
                    lower_bound = lower_bound / fit_scale
                if upper_bound is not None:
                    upper_bound = upper_bound / fit_scale
                new_dict['scale'] = fit_scale
            else:
                new_dict['scale'] = 1.

            parameters[parameter_name] = new_dict

        return parameters

    def fit(self, do_analytic_coefficients=True, verbose=False):
        fit_parameters = self._get_fit_parameters(do_analytic_coefficients)

        parameter_names = list(fit_parameters.keys())
        initial_values = [i['value'] for i in fit_parameters.values()]
        bounds = [i['bounds'] for i in fit_parameters.values()]
        scales = np.array([i['scale'] for i in fit_parameters.values()])

        self.fit_initial_values = dict(zip(parameter_names, initial_values))

        def chi_square_flat(x, return_full_info=False, apply_fit_scale=True,
                            apply_priors=True):
            map_parameters = {i: j for i, j in zip(parameter_names, x)}

            return self.chi_square(
                map_parameters,
                do_analytic_coefficients=do_analytic_coefficients,
                return_full_info=return_full_info,
                apply_fit_scale=apply_fit_scale,
                apply_priors=apply_priors,
            )

        res = minimize(
            chi_square_flat,
            initial_values,
            bounds=bounds,
            jac=grad(chi_square_flat) if config.use_autograd else None,
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-10},
        )

        if not res.success:
            raise SceneModelException(
                "Fit failed! (message=%s, params=%s)" %
                (res.message, dict(zip(parameter_names, res.x)))
            )

        # Retrieve the unscaled parameters.
        fit_parameters = res.x * scales
        chi_square, full_model, full_parameters = chi_square_flat(
            fit_parameters, return_full_info=True, apply_fit_scale=False,
            apply_priors=False,
        )

        self.set_parameters(update_derived=False, **full_parameters)
        prior_penalty = self._evaluate_priors()

        # Save the fit results.
        self.fit_model = full_model
        self.fit_result = res
        self.fit_chi_square = chi_square
        self.fit_prior_penalty = prior_penalty

        if verbose:
            self.print_fit_info()

        return full_parameters, full_model

    def calculate_covariance(self, verbose=False,
                             method=config.default_covariance_method):
        """Estimate the covariance matrix using a numerical estimate of the
        Hessian.

        See fit.calculate_covariance for details.
        """
        fit_parameters = self._get_fit_parameters(
            do_analytic_coefficients=False
        )

        parameter_names = list(fit_parameters.keys())
        values = np.array([i['value'] for i in fit_parameters.values()])
        bounds = [i['bounds'] for i in fit_parameters.values()]
        scales = np.array([i['scale'] for i in fit_parameters.values()])

        def evaluate_chi_square(flat_parameters):
            map_parameters = {i: j for i, j in zip(parameter_names,
                                                   flat_parameters)}

            return self.chi_square(
                map_parameters, do_analytic_coefficients=False,
                apply_fit_scale=True, apply_priors=True
            )

        cov = fit.calculate_covariance(evaluate_chi_square, method,
                                       parameter_names, values, bounds,
                                       verbose=verbose)

        # Rescale parameters.
        cov *= np.outer(scales, scales)

        return parameter_names, cov

    def plot_correlation(self, names=None, covariance=None, **kwargs):
        """Plot the correlation matrix between the fit parameters.

        See fit._plot_correlation for details, this is just a wrapper around
        it.
        """
        fit._plot_correlation(self, names, covariance, **kwargs)

    def calculate_uncertainties(self, names=None, covariance=None):
        """Estimate the uncertainties on all fit parameters.

        This is done by calculating the full covariance matrix and returning a
        dict matching parameters to the errors.

        This will throw a SceneModelException in the covariance matrix is not
        well defined.

        If the covariance matrix has already been calculated, then it can be
        passed in with the names and covariance parameters and it won't be
        recalculated.
        """
        if covariance is not None:
            if names is None:
                raise SceneModelException(
                    "Need to specify both names and covariance to used a "
                    "previously-calculated covariance matrix!"
                )
        else:
            names, covariance = self.calculate_covariance()

        uncertainties = np.sqrt(np.diag(covariance))

        result = {i: j for i, j in zip(names, uncertainties)}

        return result

    def print_fit_info(self, title=None, uncertainties=True):
        """Print a diagnostic of the fit.

        In uncertainties is set, then uncertainties will be printed out along
        with the fit info. uncertainties can either be True (in which case they
        are recalculated) or the output of a previous call to
        self.calculate_uncertainties.
        """
        do_uncertainties = False
        if self.fit_result is not None and uncertainties:
            # Uncertainties will only make sense if we are at the minimum, so
            # require that a fit has been done.
            do_uncertainties = True
            if uncertainties is True:
                # Need to calculate them.
                uncertainties = self.calculate_uncertainties()

        initial_values = self.fit_initial_values
        if initial_values is not None:
            do_initial_values = True
        else:
            do_initial_values = False

        print("="*70)

        if title is not None:
            print(title)
            print("="*70)

        print_parameter_header(do_uncertainties, do_initial_values)

        for parameter_name, parameter_dict in self._parameter_info.items():
            print_dict = parameter_dict.copy()
            if do_uncertainties and parameter_name in uncertainties:
                print_dict['uncertainty'] = uncertainties[parameter_name]
            if do_initial_values and parameter_name in initial_values:
                print_dict['initial_value'] = initial_values[parameter_name]

            print_parameter(parameter_name, print_dict, do_uncertainties,
                            do_initial_values)

        if self.fit_result is not None:
            print("-"*70)
            print("%20s: %g/%g" % ('chi2/dof', self.fit_chi_square,
                                   self.degrees_of_freedom))
            print("%20s: %d" % ('nfev', self.fit_result.nfev))

            if self.priors:
                print("%20s: %g (scale=%g)" % ('prior penalty',
                                               self.fit_prior_penalty,
                                               self.prior_scale))

        print("="*70)

    def generate_psf(self, grid_size=None, subsampling=11, **parameters):
        """Generate a model of the PSF.

        This is typically run with high subsampling to get a very detailed PSF
        model.

        Note this function only takes into account elements that are subclassed
        from PSF element. When working with the output of this function,
        sampling from the oversampled image is appropriate rather than
        integrating over it.

        This returns a grid_info dictionary and the resulting PSF model.
        """
        from .models import PointSource

        if grid_size is None:
            grid_size_y, grid_size_x = self.grid_info['grid_size']
            border = self.grid_info['border']
            grid_size = (grid_size_y + 2*border, grid_size_x + 2*border)

        grid_info = self._create_grid(grid_size, 0, subsampling)

        # Start with a point source at the center of the image.
        point_source = PointSource()
        point_source_parameters = {
            'center_x': 0.,
            'center_y': 0.,
            'amplitude': 1.
        }
        model, mode, point_source_parameters = point_source.update_model(
            grid_info, 0., None, point_source_parameters, False, True
        )

        # Use the current model parameters as a base, but override with
        # anything that is passed in.
        parameters = dict(self.parameters, **parameters)

        for element in self.elements:
            # Only use PsfElement elements. The PSF is considered to be the
            # convolution of all of those elements.
            if isinstance(element, PsfElement):
                model, mode, parameters = element.update_model(
                    grid_info, model, mode, parameters, False, True
                )

        if mode == 'fourier':
            model = fourier_to_real(model, grid_info)
            mode = 'real'
        elif mode != 'real':
            raise SceneModelException("PSF generation ended up in unknown "
                                      "mode %s. Can't handle!" % mode)

        return grid_info, model

    def calculate_fwhm(self, grid_size=None, subsampling=11, **parameters):
        """Calculate the FWHM in pixels of the PSF.

        This function generates a highly oversampled PSF, and then measures the
        peak amplitude of the PSF along with the number of pixels above the
        half-maximum. This count is then converted into an equivalent FWHM for
        a fully-circular PSF.
        """
        if grid_size is None:
            grid_size_y, grid_size_x = self.grid_info['grid_size']
            border = self.grid_info['border']
            grid_size = (grid_size_y + 2*border, grid_size_x + 2*border)

        if grid_size[0] % 2 == 0 or grid_size[1] % 2 == 0:
            raise SceneModelException(
                "grid sizes should be odd or the maximum value will not be "
                "found!"
            )

        if subsampling % 2 == 0:
            raise SceneModelException(
                "oversampling should be odd or the maximum value will not be "
                "found!"
            )

        psf_grid_info, psf = self.generate_psf(grid_size, subsampling,
                                               **parameters)
        scale_psf = psf / np.max(psf)

        # Count how many pixels are above the half-maximum.
        num_half_max = np.sum(scale_psf > 0.5)

        # Work out the FWHM for a circle with the corresponding area.
        fwhm = 2 * np.sqrt(num_half_max / np.pi) / subsampling

        return fwhm

    @property
    def degrees_of_freedom(self):
        """Calculate how many degrees of freedom there are in the fit"""
        num_data = len(self.fit_data)
        fit_parameters = self._get_fit_parameters(
            do_analytic_coefficients=False
        )
        num_parameters = len(fit_parameters)

        return num_data - num_parameters

    def extract(self, images, variances=None, **parameters):
        """Extract the coefficent parameters from a set of images.

        Coefficients refer to any parameters in the model that scale a
        component, including amplitudes, backgrounds, etc.

        If any parameters are modified between images, they can be passed as an
        array with the same length as the number of parameters (eg: wavelength
        dependence).

        images be either a single image or a list of images. Specifying the
        variance with variances is optional (a least-squares extraction will be
        done), but if done it must have the same shape as images.
        """
        # Handle the case where we want to extract a single image. After this,
        # images and variances will always be a list of images and variances.
        if len(np.shape(images)) == 2:
            # Single image was passed
            images = np.asarray(images)[np.newaxis, :, :]
            if variances is not None:
                variances = np.asarray(variances)[np.newaxis, :, :]
        else:
            images = np.asarray(images)
            if variances is not None:
                variances = np.asarray(variances)

        extraction_results = []

        for idx in range(len(images)):
            # Update the model with any parameters that were passed in.
            image_parameters = {}
            for parameter_name, parameter_value in parameters.items():
                parameter_dimension = len(np.shape(parameter_value))
                if parameter_dimension == 1:
                    # An array, one for each image
                    use_parameter = parameter_value[idx]
                elif parameter_dimension == 0:
                    # A single value, same for each image
                    use_parameter = parameter_value
                else:
                    # Shouldn't get here...
                    raise SceneModelException(
                        "Can't parse parameter %s! Too many dimensions" %
                        parameter_name
                    )
                image_parameters[parameter_name] = use_parameter

            # Process the image, and flag any bad pixels or things like that
            if variances is not None:
                variance = variances[idx]
            else:
                variance = None
            image_fit_data = self._process_image(images[idx], variance)

            # Calculate the components of the scene
            components, eval_parameters = self.evaluate(
                separate_components=True,
                apply_coefficients=False,
                return_full_parameters=True,
                **image_parameters
            )

            # Evaluate the coefficients analytically
            coef_names, coef_values, coef_variances = \
                self._calculate_coefficients(
                    components,
                    image_fit_data['mask_data'],
                    image_fit_data['mask_variance'],
                    image_fit_data['mask'],
                    **eval_parameters
                )

            result = {}
            for name, value, variance in zip(coef_names, coef_values,
                                             coef_variances):
                result[name] = value
                result["%s_variance" % name] = variance

            extraction_results.append(result)

        extraction_results = Table(extraction_results)

        return extraction_results

    def get_fits_header_items(self, prefix=config.default_fits_prefix):
        """Get a list of parameters to put into the fits header.

        Each entry has the form (fits_keyword, value, description).

        prefix is added to each key in the header.
        """
        fits_list = []
        for parameter_name, parameter_dict in self._parameter_info.items():
            # Only include parameters whose value is set.
            if parameter_dict['value'] is None:
                # Don't add in parameters whose values aren't set (eg: derived
                # parameters when we make a PSF out of a MultipleImageFitter
                # object. We don't want to include the parameters for a single
                # image, we want the global parameters).
                continue

            keyword = prefix + parameter_dict['fits_keyword']
            value = parameter_dict['value']
            description = parameter_dict['fits_description']

            fits_list.append((keyword, value, description))

        # Add in additional keys
        fits_list.append((prefix + 'MODEL', type(self).__name__,
                          'Scene model name'))
        fits_list.append((prefix + 'LREF', config.reference_wavelength,
                          'Reference wavelength [A]'))

        return fits_list

    def load_fits_header(self, header, prefix=config.default_fits_prefix,
                         skip_parameters=[]):
        """Load parameters from a fits header

        The header keys are prefix + the parameter's fits keyword.
        (extract_star prefixes everything with ES_).

        Parameters in skip_parameters are skipped and ignored. derived and
        coefficient parameters are loaded if they are in the header, but
        nothing is done if they aren't there. If any other parameter is
        missing, a warning is printed.
        """
        # Read in element parameters and set them.
        element_parameters = {}
        for name, parameter_dict in self._parameter_info.items():
            if name in skip_parameters:
                # Ignore this one
                continue

            fits_keyword = prefix + parameter_dict['fits_keyword']
            try:
                value = header[fits_keyword]
            except KeyError:
                if parameter_dict['derived'] or parameter_dict['coefficient']:
                    # It is normal if derived or coefficient parameters aren't
                    # in the header. For an extraction, the coefficients vary
                    # for each wavelength, so there is no single appropriate
                    # value.
                    pass
                else:
                    print("WARNING: parameter %s not found in header! "
                          "Skipping!" % fits_keyword)
                continue

            element_parameters[name] = value

        self.set_parameters(update_derived=False, **element_parameters)

    def plot(self):
        """Plot the data and model with various diagnostic plots."""
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm

        plt.figure(figsize=(8, 8))

        model = self.evaluate()
        image = self.image.copy()

        background_level = np.min(model)
        background = np.max([0, background_level])
        max_value = np.max(self.fit_data) - background

        if self.fit_variance is not None:
            error_level = np.median(self.fit_variance)
        else:
            error_level = nmad(self.fit_data)

        scale_min = background + np.max([error_level, max_value * 0.001])
        scale_max = background + np.max([5*error_level, max_value * 0.9])

        norm = LogNorm(vmin=scale_min, vmax=scale_max)

        image[np.isnan(image)] = -1
        image[image < 0] = scale_min / 2.
        model[model < 0] = scale_min / 2.

        plt.subplot(2, 2, 1)
        plt.imshow(image.T, origin='lower', norm=norm)
        plt.title('Data')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(model.T, origin='lower', norm=norm)
        plt.title('Model')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow((image - model).T, origin='lower')
        plt.title('Residuals')
        plt.colorbar()

        # Contour plot
        # Do the contours in log-space
        num_contours = 8
        contour_levels = np.linspace(
            np.log10(scale_min), np.log10(scale_max), num_contours
        )
        contour_colors = plt.cm.viridis(np.linspace(0, 1, num_contours))

        plt.subplot(2, 2, 4)
        data_contour = plt.contour(
            np.log10(image).T,
            levels=contour_levels,
            colors=contour_colors,
            origin='lower',
        )
        data_contour.collections[0].set_label('Data')

        # Need to chop out any levels that aren't used or the colors get messed
        # up...
        mask = []
        log_model = np.log10(model)
        for i in contour_levels:
            if np.all(i < log_model) or np.all(i > log_model):
                mask.append(False)
            else:
                mask.append(True)
        mask = np.array(mask)

        if np.sum(mask) > 0:
            # Make sure that there are overlapping entries. This isn't always
            # the case when models haven't been fit to data yet.
            model_contour = plt.contour(
                log_model.T,
                levels=contour_levels[mask],
                colors=contour_colors[mask],
                linestyles='dashed',
                origin='lower',
            )
            model_contour.collections[0].set_label('Model')
        plt.title('Contours')
        plt.xlim(0, self.image.shape[0]-1)
        plt.ylim(0, self.image.shape[1]-1)
        plt.legend()

    def plot_radial_profile(self, residuals=False, new_figure=True):
        """Plot the radial profile of the data and model"""
        from matplotlib import pyplot as plt

        p = self.parameters

        if new_figure:
            plt.figure()

        dx = self.grid_info['grid_x'] - p['center_x']
        dy = self.grid_info['grid_y'] - p['center_y']
        r2 = dx**2 + dy**2

        amplitude = p['amplitude']
        background = p['background']

        model = self.evaluate()

        scale_data = self.image / amplitude
        scale_model = model / amplitude
        scale_background = background / amplitude

        if residuals:
            # Show the residuals rather than the function directly.
            plt.scatter(np.sqrt(r2), scale_data - scale_model, s=5,
                        label='Residuals')

            plt.semilogx()
            plt.xlabel('Distance from center (spaxels)')
            plt.ylabel('Residuals (fraction of total flux)')
        else:
            plt.scatter(np.sqrt(r2), scale_data, s=5, label='Data')
            plt.scatter(np.sqrt(r2), scale_model - scale_background, s=5,
                        label='PSF model', c='C1')
            plt.scatter(np.sqrt(r2), scale_model, s=5, label='Full model',
                        c='C3')

            plt.axhline(background / amplitude, c='C2', label='Background')

            plot_data = scale_data[~np.isnan(scale_data)]
            neg_mask = plot_data < 0
            min_val = np.min(plot_data[~neg_mask])
            plot_data[neg_mask] = min_val / 2.

            plt.loglog()
            plt.xlabel('Distance from center (spaxels)')
            plt.ylabel('Relative flux')
            plt.ylim(min_val / 5., np.max(plot_data)*2)

        plt.legend()

    def fit_and_fix_position(self, verbose=False, center_x_key='center_x',
                             center_y_key='center_y', border=0, subsampling=3,
                             **kwargs):
        """Fit the object in the image with a Gaussian, and fix its position
        for future fits.

        As the position is typically easy to fit on high signal-to-noise
        images, this is useful to speed up fits to large datasets since the
        position fit doesn't affect anything else.

        Note that we use the GaussianSceneModel class which evaluates in real
        space. This means that we don't need a border on the model, but we do
        need higher subsampling to get a good result.
        """
        from .models import SimpleGaussianSceneModel
        gaussian_model = SimpleGaussianSceneModel(
            self.image, self.variance, border=border, subsampling=subsampling,
            **kwargs
        )

        parameters, model = gaussian_model.fit(verbose=verbose)
        center_x = parameters['center_x']
        center_y = parameters['center_y']

        self.fix(center_x=center_x, center_y=center_y)
