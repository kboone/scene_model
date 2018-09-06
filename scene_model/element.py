# -*- coding: utf-8 -*-
from collections import OrderedDict

from . import config
from .utils import is_overridden, DumbLRUCache, SceneModelException, \
    ModelParameterDictionary

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


def real_to_fourier(real_components, grid_info):
    """Convert real components to Fourier ones.

    This can handle either single components or lists of multiple
    components.
    """
    if config.debug_fourier:
        print("--  Applying FFT")

    if len(np.shape(real_components)) == 2:
        # Single element
        single = True
        real_components = [real_components]
    else:
        single = False

    fourier_components = []
    for real_component in real_components:
        fourier_component = (
            np.fft.fft2(real_component) *
            grid_info['pad_fft_shift']
        )
        fourier_components.append(fourier_component)

    if single:
        return fourier_components[0]
    else:
        return fourier_components


def fourier_to_real(fourier_components, grid_info):
    """Convert Fourier components to real ones.

    This can handle either single components or lists of multiple
    components.
    """
    if config.debug_fourier:
        print("--  Applying IFFT")

    if len(np.shape(fourier_components)) == 2:
        # Single element
        single = True
        fourier_components = [fourier_components]
    else:
        single = False

    real_components = []
    for fourier_component in fourier_components:
        real_component = (
            np.real(np.fft.ifft2(fourier_component *
                                 grid_info['pad_ifft_shift']))
        )
        real_components.append(real_component)

    if single:
        return real_components[0]
    else:
        return real_components


class ModelElement(object):
    """Element of a scene model.

    This class represents any object or transformation that is necessary to
    produce an image. This includes adding point sources or galaxies, applying
    PSFs, and telescope/detector artifacts. The final image is produced by
    applying a sequence of ModelElements sequentially to build a model.
    """
    def __init__(self, prefix=None, fits_keyword_prefix=None,
                 fits_description_prefix=None):
        """Initialize the model.

        If prefix is not None, then the prefix is added in front of each model
        parameter.

        If fits_prefix is not None, then it is added by default directly to the
        front the fits keyword for each parameter. If description_prefix is not
        None, then it is added in front of the description for each parameter.
        """
        self.prefix = prefix
        self.fits_keyword_prefix = fits_keyword_prefix
        self.fits_description_prefix = fits_description_prefix

        # Set up the cache
        self._cache = DumbLRUCache(max_size=5)

        self._parameter_info = OrderedDict()
        self._model_name_map = dict()
        self._setup_parameters()

    def _add_parameter(self, name, value, bounds, fits_keyword=None,
                       fits_description=None, fixed=False, apply_prefix=True,
                       coefficient=False, derived=False):
        """Add a parameter to the model."""
        # Apply prefixes if requested.
        model_name = name
        if apply_prefix:
            if self.prefix is not None:
                model_name = '%s_%s' % (self.prefix, name)
            if self.fits_keyword_prefix is not None:
                fits_keyword = self.fits_keyword_prefix + fits_keyword
            if self.fits_description_prefix is not None:
                fits_description = "%s %s" % (self.fits_description_prefix,
                                              fits_description)

        new_parameter = {
            'value': value,
            'bounds': bounds,
            'fixed': fixed,
            'coefficient': coefficient,
            'derived': derived,
            'fits_keyword': fits_keyword,
            'fits_description': fits_description,
        }

        self._parameter_info[model_name] = new_parameter
        self._model_name_map[name] = model_name

    def _setup_parameters(self):
        """Initialize the parameters for a given image.

        This function sets up the internal parameter structures with calls to
        _add_parameters. There are no parameters by default.
        """
        pass

    def _modify_parameter(self, name, model_name=False, **kwargs):
        """Modify a parameter.

        If internal_name is True, then the name passed in is the internal name
        of the parameter (eg: amplitude). Otherwise, the name passed in is the
        model name of the parameter (eg: star4_amplitude).
        """
        if 'derived' in kwargs and kwargs['derived'] is True:
            # When we change a parameter to derived, unset its value since it
            # is no longer applicable.
            kwargs['value'] = None

        if not model_name:
            # Get the model name
            name = self._model_name_map[name]

        self._parameter_info[name].update(kwargs)

    def rename_parameter(self, original_name, new_name):
        """Rename a parameter"""
        if original_name not in self._parameter_info:
            raise SceneModelException("%s has no parameter named %s." %
                                      (type(self), original_name))

        if new_name in self._parameter_info:
            raise SceneModelException("%s already has a parameter named %s." %
                                      (type(self), new_name))

        # Keep the order of _parameter_info. We need to recreate the
        # OrderedDict to do this.
        self._parameter_info = OrderedDict(
            (new_name if key == original_name else key, value) for key, value
            in self._parameter_info.items()
        )

        # Update the model name map dictionary.
        self._model_name_map[original_name] = new_name

    def fix(self, **kwargs):
        """Fix a set of parameters to a given set of values.

        If the value that a parameter is set to is None, then the parameter is
        unfixed.
        """
        for key, value in kwargs.items():
            parameter_dict = self._parameter_info[key]

            if parameter_dict['derived']:
                raise SceneModelException(
                    "Derived parameter %s can't be fixed! It is defined in "
                    "terms of other parameters, and is already effectively "
                    "fixed." % key
                )

            if value is None:
                # Unfix
                parameter_dict['fixed'] = False
            else:
                # Fix
                parameter_dict['value'] = value
                parameter_dict['fixed'] = True

        self.clear_cache()

    def is_fixed(self, parameter):
        """Return whether a parameter is fixed or not"""
        return self._parameter_info[parameter]['fixed']

    @property
    def parameters(self):
        """Return a list of the current parameter values."""
        full_parameters = dict()
        for parameter_name, parameter_dict in self._parameter_info.items():
            full_parameters[parameter_name] = parameter_dict['value']

        return full_parameters

    @property
    def coefficients(self):
        """Return an ordered dictionary of the parameters that are the
        coefficients for each component of the model.
        """
        coefficients = dict()
        for parameter_name, parameter_dict in self._parameter_info.items():
            if not parameter_dict['coefficient']:
                continue
            coefficients[parameter_name] = parameter_dict['value']

        return coefficients

    @property
    def _coefficient_info(self):
        """Return an ordered dictionary of the full information for each of the
        coefficients.
        """
        coefficient_info = OrderedDict()
        for parameter_name, parameter_dict in self._parameter_info.items():
            if not parameter_dict['coefficient']:
                continue
            coefficient_info[parameter_name] = parameter_dict

        return coefficient_info

    def set_parameters(self, update_derived=True, **kwargs):
        """Set parameters to the given values."""
        for key, value in kwargs.items():
            if key not in self._parameter_info:
                raise SceneModelException("%s has no parameter named %s." %
                                          (type(self), key))

            parameter_dict = self._parameter_info[key]

            # Make sure that we aren't changing any derived parameters unless
            # explicitly told to do so.
            if parameter_dict['derived'] and update_derived:
                raise SceneModelException(
                    "You shouldn't be setting %s, it is a derived parameter! "
                    "Pass update_derived=False to set_parameters if you "
                    "really intended to do this." % key
                )

            parameter_dict['value'] = value

        if update_derived:
            self._update_derived_parameters()

        self.clear_cache()

    def _calculate_derived_parameters(self, parameters):
        """Calculate parameters that are derived from other parameters.

        This function should be overridden in subclasses to perform whatever
        calculations are necessary.

        This is used for things like the pipeline PSF where several parameters
        are fixed to be a function of the alpha parameter. By default, this
        does nothing.
        """
        return parameters

    def _update_derived_parameters(self):
        """Recalculate derived parameters for the internally stored parameters.
        """
        # Get a dictionary with the remapped parameter names
        mapped_parameters = ModelParameterDictionary(self.parameters,
                                                     self._model_name_map)

        updated_parameters = \
            self._calculate_derived_parameters(mapped_parameters)

        self.set_parameters(update_derived=False,
                            **updated_parameters.model_dict)

    def __getitem__(self, parameter):
        """Return the value of a given parameter"""
        return self._parameter_info[parameter]['value']

    def __setitem__(self, parameter, value):
        """Set the value of a given parameter"""
        self.set_parameters(**{parameter: value})

    def update_model(self, grid_info, model, mode, parameters,
                     separate_components, apply_coefficients):
        """Apply the model element to a model.

        This method wraps and calls the _update_model method of subclasses that
        does all of the actual work. This function handles mapping all of the
        parameters to their prefixed/renamed versions transparently, and adds
        in any internal parameters.

        model is the model that has been built up so far from previous
        elements. mode indicates whether the model is currently in Fourier
        space or in real space.

        If separate_components is True, then model is a list of all of the
        individual components of the model (eg: a PSF, a galaxy, a background)
        rather than a single image. For separate components, if
        apply_coefficients is True, then the components are scaled by their
        respective coefficients. If not, they are returned with a scale of 1
        (this is used for fitting).

        The exact behavior here will vary. Some examples are:
        - SubsampledModelComponent: adds new components in to the model (eg: a
          point source, a galaxy)
        - ConvolutionElement: convolves the model with something (eg: a PSF).
        - Pixelizer: pixelizes the model and removes subsampling to obtain real
          pixels.

        This function returns the model with the element applied, a string
        representing the mode (eg: "real", "fourier" or "pixel") and a
        dictionary of updated parameters that includes both derived parameters
        and internal ones.
        """
        # Add in base parameters
        for key, parameter_dict in self._parameter_info.items():
            if key not in parameters:
                parameters[key] = parameter_dict['value']

            if not apply_coefficients:
                # Don't apply coefficients to the model. This means that the
                # scales of the components should be set to 1.
                if parameter_dict['coefficient']:
                    parameters[key] = 1.

        # Update the derived parameters. We use a ModelParameterDictionary
        # wrapper here to do the conversion between the internal parameter
        # names and the ones used in the SceneModel which can be arbitrarily
        # remapped.
        mapped_parameters = ModelParameterDictionary(parameters,
                                                     self._model_name_map)
        full_parameters = self._calculate_derived_parameters(mapped_parameters)

        # Get a dictionary with only the element specific parameters with the
        # internal names to pass to the _update_model function.
        element_parameters = mapped_parameters.get_element_parameters()

        # Do the actual model update
        model, mode = self._update_model(grid_info, model, mode,
                                         element_parameters,
                                         separate_components)

        # Get the full unmapped parameters to return
        scene_parameters = full_parameters.model_dict

        return model, mode, scene_parameters

    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Apply the model element to the model.

        This method should be implemented in subclasses. See update_model for
        details. This method should return the final model along with the mode
        that the model is in.

        parameters is a dictionary that only contains parameters that were
        defined for the specific element.
        """
        return model, mode

    def get_cache(self, item, calculator, **kwargs):
        """Use a cache to avoid reevaluating parts of an element that don't
        change.

        This cache is automatically reset whenever a major part of the PSF is
        changed, and it only holds the last call. The caller must explicitly
        specify the relevant variables, and the cached item will only be
        recalculated when one of those variables is changed. Note that
        variables like grids do not need to be specified because the cache is
        automatically reset whenever they change.

        calculator should be a lambda function that takes no arguments and that
        calculates the desired target if it isn't in the cache. It will only be
        evaluated if the cache hit fails.

        An example of how to use the cache is as follows:

        gaussian = self.get_cache(
            'gaussian',
            center_x=center_x,
            sigma_x=sigma_x,
            calculator=lambda: np.exp(-(grid_x-center_x)**2 / 2 / sigma_x**2)
        )

        Here, the cache will check if either of center_x or sigma_x has
        changed since the last call. If they haven't changed, it will return
        the previous value of gaussian in the cache. If they have, it will
        evaluate the lambda function given by calculator to evaluate the
        gaussian function. Note that the grid of x values specified by grid_x
        does not need to be included in the variables to watch since the cache
        will be cleared if it is ever changed.
        """
        cache = self._cache

        if config.use_autograd:
            # If we are using autograd, we need to make sure to not cache
            # anything if we are doing autograd calculations.
            for key, value in kwargs.items():
                if isinstance(value, np.numpy_boxes.ArrayBox):
                    return calculator()

        # Try to get the item from the cache.
        cache_value = cache[kwargs]
        if cache_value is not None:
            return cache_value

        # Not in the cache. We need to recalculate the item.
        new_value = calculator()
        cache[kwargs] = new_value

        return new_value

    def _grid_cache_parameters(self, grid_info):
        """Return a dictionary with the items necessary to uniquely specify a
        grid.

        This is used for caching. All of the items returned should be scalars.
        """
        grid_cache_parameters = {
            'subsampling': grid_info['subsampling'],
            'grid_size': grid_info['grid_size'],
            'border': grid_info['border'],
            'reference_i': grid_info['reference_i'],
            'reference_j': grid_info['reference_j'],
        }

        return grid_cache_parameters

    def clear_cache(self):
        """Clear the internal caches."""
        # Cache for specific parts of the PSF calculation.
        self._cache.clear()

    def print_cache_info(self):
        """Print info about the cache"""
        print("Hits: %d" % self._cache.hits)
        print("Misses: %d" % self._cache.misses)


class SubsampledModelElement(ModelElement):
    """Model element that is evaluated on a subsampled grid.

    The subsampled grid is defined in both real and Fourier space, and the
    model can be freely transformed between the two to make calculations
    easier.
    """
    def _evaluate(self, x, y, subsampling, grid_info, **parameters):
        """Evaluate the element at a set of x and y positions with the given
        parameters.

        The scene model evaluator will switch between Fourier and real space as
        appropriate in order to build the full PSF. The element can be defined
        in either Fourier or real space, or both. If one is not specified, then
        the other will be derived with an FFT.

        Note that all of the parameters will be passed in to this function,
        including fixed, derived, coefficient, etc. ones that are not
        necessarily used by the element. The parameters that are used can be
        explicitly added as function arguments, but a **parameters key should
        be used to capture extra ones.

        grid_info contains all of the grid information. Most of the time this
        isn't necessary and it can just be ignored. However, it is needed to do
        Fourier to real conversions and things like that. The x, y and
        subsampling values are actually included in grid_info as pad_grid_x and
        pad_grid_y, but they are passed as separate parameters for convenience
        so that they don't have to be explicitly pulled out every time.
        """
        # If _evaluate hasn't been defined, transform the Fourier space
        # evaluation of this element if it exists.
        if not self._has_fourier_evaluate():
            # Neither _evaluate nor _evaluate_fourier was defined, so there is
            # no way to evaluate the element!
            raise SceneModelException(
                "Must define at least one of _evaluate and _evaluate_fourier!"
            )

        fourier_element = self._evaluate_fourier(
            grid_info['pad_grid_kx'],
            grid_info['pad_grid_ky'],
            subsampling,
            grid_info,
            **parameters
        )

        real_element = fourier_to_real(fourier_element, grid_info)

        return real_element

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, **parameters):
        """Evaluate the element in Fourier space at a set of kx and ky
        frequencies with the given parameters.

        See _evaluate for details.
        """
        # If _evaluate_fourier hasn't been defined, transform the real space
        # evaluation of this element if it exists.
        if not self._has_real_evaluate():
            # Neither _evaluate nor _evaluate_fourier was defined, so there is
            # no way to evaluate the element!
            raise SceneModelException(
                "Must define at least one of _evaluate and _evaluate_fourier!"
            )

        real_element = self._evaluate(
            grid_info['pad_grid_x'],
            grid_info['pad_grid_y'],
            subsampling,
            grid_info,
            **parameters
        )

        fourier_element = real_to_fourier(real_element, grid_info)

        return fourier_element

    def _cache_evaluate(self, x, y, subsampling, grid_info, parameters,
                        mode='real'):
        """Cache evaluations of this element.

        Assuming that no parameters have changed, we should be able to reuse
        the last evaluation of this element. Because different parts of the
        model are typically being varied independently when doing things like
        fitting, there is a lot of reuse of model elements, so this gives a
        large speedup.

        This function can be used in place of _evaluate to directly gain
        caching functionality.
        """
        if mode == 'fourier':
            func = self._evaluate_fourier
            label = 'evaluate_fourier'
        elif mode == 'real':
            func = self._evaluate
            label = 'evaluate_real'
        else:
            raise SceneModelException("_cache_evaluate can't handle mode %s!" %
                                      mode)

        result = self.get_cache(
            label,
            calculator=lambda: func(x, y, subsampling, grid_info,
                                    **parameters),
            grid_info_str=grid_info['unique_str'],
            **parameters
        )

        return result

    def _cache_evaluate_fourier(self, kx, ky, subsampling, grid_info,
                                parameters):
        """Cache Fourier evaluations of this element.

        This is just a wrapper around _cache_evaluate with mode set to fourier.
        See it for details.
        """
        return self._cache_evaluate(kx, ky, subsampling, grid_info, parameters,
                                    mode='fourier')

    def _has_real_evaluate(self):
        """Check if the real evaluation is defined"""
        return is_overridden(SubsampledModelElement._evaluate, self._evaluate)

    def _has_fourier_evaluate(self):
        """Check if the real evaluation is defined"""
        return is_overridden(SubsampledModelElement._evaluate_fourier,
                             self._evaluate_fourier)


class SubsampledModelComponent(SubsampledModelElement):
    """A model element representing one or multiple components that are added
    into the model.

    Examples of components are point sources, galaxies, backgrounds, etc.
    """
    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Add the components defined by this class to a model."""
        # Apply any parent class updates
        model, mode = super(SubsampledModelComponent, self). _update_model(
            grid_info, model, mode, parameters, separate_components
        )

        # Figure out whether to add the component in real or Fourier space. We
        # just go with whatever mode the model is currently in.
        if mode is None:
            # No mode selected yet, pick whichever one the model has defined,
            # and fourier if it has defined both.
            if self._has_fourier_evaluate():
                mode = 'fourier'
            else:
                mode = 'real'

        if mode == 'real':
            components = self._cache_evaluate(
                grid_info['pad_grid_x'],
                grid_info['pad_grid_y'],
                grid_info['subsampling'],
                grid_info,
                parameters
            )
        elif mode == 'fourier':
            components = self._cache_evaluate_fourier(
                grid_info['pad_grid_kx'],
                grid_info['pad_grid_ky'],
                grid_info['subsampling'],
                grid_info,
                parameters
            )
        else:
            raise SceneModelException("SubsampledModelComponents can't be "
                                      "added to models in mode %s!" % mode)

        # Add the new components to the model
        if len(np.shape(components)) == 2:
            # Single component. Put it into a list so that we can handle
            # everything in the same way.
            components = [components]

        if separate_components:
            model.extend(components)
        else:
            for component in components:
                model += component

        return model, mode


class ConvolutionElement(SubsampledModelElement):
    """A model element representing a convolution of the model with something.

    Examples of convolutions are PSFs, ADR, etc.
    """
    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Add the components defined by this class to a model."""
        # Apply any parent class updates
        model, mode = super(ConvolutionElement, self). _update_model(
            grid_info, model, mode, parameters, separate_components
        )

        # Convolutions have to happen in Fourier space (where they are a
        # multiplication), so we convert the model to Fourier space if
        # necessary.
        if mode is None:
            # Can't convolve if there is no model yet! This means that
            # something went wrong in the model definition (defined convolution
            # before sources?)
            raise SceneModelException("Can't apply a convolution if there is "
                                      "no model yet. Is the order right?")
        elif mode == 'real':
            # Convert the model to Fourier space.
            model = real_to_fourier(model, grid_info)
            mode = 'fourier'
        elif mode != 'fourier':
            raise SceneModelException("ConvolutionElement can't be applied to "
                                      "models in mode %s!" % mode)

        convolve_image = self._cache_evaluate_fourier(
            grid_info['pad_grid_kx'],
            grid_info['pad_grid_ky'],
            grid_info['subsampling'],
            grid_info,
            parameters
        )

        # Convolve the model with the target. Note that we are operating in
        # Fourier space, so multiplying does a convolution in real space.
        if separate_components:
            # Do each component separately.
            convolved_model = []
            for component in model:
                convolved_model.append(component * convolve_image)
        else:
            convolved_model = model * convolve_image

        return convolved_model, mode


class PsfElement(ConvolutionElement):
    """Class for elements of a PSF model.

    This class is just a wrapper around ConvolutionElement, but is used to
    identify which components are part of the PSF. This identification is
    important for things like calculating the FWHM of the PSF where not all
    effects should be included.
    """
    pass


class Pixelizer(ModelElement):
    """Model element that pixelizes a model.

    After this is applied to the model, subsampling is removed and the model is
    pixelated. The model can no longer be transformed back to Fourier space,
    and any operations have to happen on the pixelated grid.
    """
    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Pixelate the image"""
        # Apply any parent class updates
        model, mode = super(Pixelizer, self)._update_model(
            grid_info, model, mode, parameters, separate_components
        )

        if mode == 'real':
            # Convert to fourier space
            model = real_to_fourier(model, grid_info)
            mode = 'fourier'
        elif mode != 'fourier':
            raise SceneModelException("Pixelizer can't be applied to models "
                                      "in mode %s!" % mode)

        kx = grid_info['pad_grid_kx']
        ky = grid_info['pad_grid_ky']
        subsampling = grid_info['subsampling']
        border = grid_info['border']

        # Convolve with the subpixel
        fourier_subpixel = self.get_cache(
            'fourier_pixel',
            calculator=lambda: (
                np.sinc(kx / np.pi / 2. / subsampling) *
                np.sinc(ky / np.pi / 2. / subsampling)
            ),
            grid_info_str=grid_info['unique_str']
        )

        if not separate_components:
            # Make a list with a single element that is the model to be able to
            # use the same code on everything. We'll pull it back out later.
            model = [model]

        final_model = []
        for component_model in model:
            fourier_subpixelated_model = component_model * fourier_subpixel

            # Convert to real space and sample
            subpixelated_model = fourier_to_real(fourier_subpixelated_model,
                                                 grid_info)

            pixelated_model = 0.
            for i in range(subsampling):
                for j in range(subsampling):
                    pixelated_model += subpixelated_model[
                        i::subsampling, j::subsampling
                    ]

            # Remove borders
            if border == 0:
                trimmed_model = pixelated_model
            else:
                trimmed_model = pixelated_model[border:-border, border:-border]
            final_model.append(trimmed_model)

        if not separate_components:
            final_model = final_model[0]

        return final_model, 'pixel'


class RealPixelizer(ModelElement):
    """Model element that pixelizes a model in real space.

    After this is applied to the model, subsampling is removed and the model is
    pixelated. The model can no longer be transformed back to Fourier space,
    and any operations have to happen on the pixelated grid.

    This pixelizer applies in real space only, and simply sums up all of the
    values of the subpixels. This should only be used if the subsampling scale
    is much smaller than the scale at which the model varies.
    """
    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Pixelate the image"""
        # Apply any parent class updates
        model, mode = super(RealPixelizer, self)._update_model(
            grid_info, model, mode, parameters, separate_components,
        )

        if mode == 'fourier':
            # Convert to real space
            model = fourier_to_real(model, grid_info)
            mode = 'real'
        elif mode != 'real':
            raise SceneModelException("RealPixelizer can't be applied to "
                                      "models in mode %s!" % mode)

        subsampling = grid_info['subsampling']
        border = grid_info['border']

        if not separate_components:
            # Make a list with a single element that is the model to be able to
            # use the same code on everything. We'll pull it back out later.
            model = [model]

        final_model = []
        for component_model in model:
            pixelated_model = 0.
            for i in range(subsampling):
                for j in range(subsampling):
                    pixelated_model += component_model[i::subsampling,
                                                       j::subsampling]

            # Remove borders
            if border == 0:
                trimmed_model = pixelated_model
            else:
                trimmed_model = pixelated_model[border:-border, border:-border]
            final_model.append(trimmed_model)

        if not separate_components:
            final_model = final_model[0]

        return final_model, 'pixel'


class PixelModelComponent(ModelElement):
    """A model element representing one or multiple components that are added
    into the model after pixelization.

    This is primarily intended to be used for backgrounds or detector
    artifacts.
    """
    def _update_model(self, grid_info, model, mode, parameters,
                      separate_components):
        """Add the components defined by this class to a model."""
        # Apply any parent class updates
        model, mode = super(PixelModelComponent, self)._update_model(
            grid_info, model, mode, parameters, separate_components,
        )

        if mode != 'pixel':
            raise SceneModelException(
                "PixelModelComponents can only be added to a SceneModel after "
                "pixelization. (current mode %s)" % mode
            )

        components = self._cache_evaluate(
            grid_info['grid_x'],
            grid_info['grid_y'],
            grid_info,
            parameters
        )

        # Add the new components to the model
        if len(np.shape(components)) == 2:
            # Single component. Put it into a list so that we can handle
            # everything in the same way.
            components = [components]

        if separate_components:
            model.extend(components)
        else:
            for component in components:
                model += component

        return model, mode

    def _cache_evaluate(self, x, y, grid_info, parameters):
        """Cache evaluations of this element.

        Assuming that no parameters have changed, we should be able to reuse
        the last evaluation of this element. Because different parts of the
        model are typically being varied independently when doing things like
        fitting, there is a lot of reuse of model elements, so this gives a
        large speedup.

        This function can be used in place of _evaluate to directly gain
        caching functionality.
        """
        result = self.get_cache(
            'evaluate',
            calculator=lambda: self._evaluate(x, y, grid_info, **parameters),
            grid_info_str=grid_info['unique_str'],
            **parameters
        )

        return result

    def _evaluate(self, x, y, grid_info, **parameters):
        """Evaluate the element at a set of x and y positions with the given
        parameters.

        This must be implemented in subclasses.
        """
        raise NotImplementedError()
