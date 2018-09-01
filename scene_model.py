# -*- coding: utf-8 -*-
from __future__ import print_function

from astropy.table import Table
import itertools
from scipy.optimize import minimize
from collections import OrderedDict
from scipy.linalg import pinvh, LinAlgError
from scipy.signal import medfilt2d
import functools
import time
from copy import deepcopy


try:
    import iminuit as minuit
    minuit_version = 'iminuit'
except ImportError:
    import minuit
    minuit_version = 'PyMinuit'


###############################################################################
# Hardcoded configuration for SNIFS. This might need to be updated for other
# instruments.
###############################################################################

# If available, this code can make use of the autograd package for calculating
# gradients and hessians. Unfortunately, the version of numpy at the CC is
# extremely old and not compatible with the autograd package, so we can't use
# it there. This can be enabled on private machines to speed things up a lot.
use_autograd = False

# Use a constant reference wavelength for easy comparisons.
reference_wavelength = 5000.

# By default, the scene model is evaluated on a subsampled grid with a border.
# The default parameters for this border and subsampling are set here, although
# they can be overriden in the SceneModel constructor. It is important that the
# grid contains the full PSF, since the PSf will be normalized to be 1 in the
# evaluated region!
default_subsampling = 2
default_border = 15

# Fits output
default_fits_prefix = 'ES_'

###############################################################################
# Debug flags. Enable to get a lot of output
###############################################################################

# Show all Fourier transformations.
debug_fourier = False

###############################################################################
# End of configuration.
###############################################################################

if use_autograd:
    from autograd import numpy as np
    from autograd import grad, hessian
else:
    import numpy as np


class PsfModelException(Exception):
    pass


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


def hessian_to_covariance(hessian):
    """Safely invert a Hessian matrix to get a covariance matrix.

    Sometimes, parameters can have wildly different scales from each other,
    even when I try to set them to be close to 1. The main offender seems to be
    the background level of an image: I normalize based off of the amplitude,
    so for very bright images, a very small change in the background level can
    dramatically affect the chi-square. When calculating the covariance matrix,
    this can cause major problems because the Hessian will have variables on
    wildly different scales.

    What we actually care about is having the same relative precision on the
    error of each parameter rather than the absolute precision. In that case,
    we can normalize the Hessian prior to inverting it, and then renormalize
    afterwards. This deals with the problem of varying scales of parameters
    gracefully.
    """
    # Choose scales to set the diagonal of the hessian to 1.
    scales = np.sqrt(np.diag(hessian))
    norm_hessian = hessian / np.outer(scales, scales)

    # Now invert the scaled Hessian using a safe inversion algorithm
    inv_norm_hessian = pinvh(norm_hessian)

    # Add the scales back in.
    covariance = 2. * inv_norm_hessian / np.outer(scales, scales)

    return covariance


def calculate_covariance_finite_difference(chisq_function, parameter_names,
                                           values, bounds, verbose=False):
    """Do a 2nd order finite difference estimate of the covariance matrix.

    This doesn't work very well because it doesn't use adaptive step sizes.
    Minuit does much better with adaptive step sizes.

    For this, the formula is:
    d^2f(dx1*dx2) = ((f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2))
                     / 4*e1*e2)
    So we need to calculate all the f(x +/-e1 +/-e2) terms (where e1 and
    e2 are small steps in 2 possibly different directions).
    """
    # The three terms here are the corresponding weight, the sign of e1 and
    # the sign of e2.
    difference_info = [
        (+1/4., +1., +1.),
        (-1/4., +1., -1.),
        (-1/4., -1., +1.),
        (+1/4., -1., -1.),
    ]

    num_variables = len(parameter_names)

    # Determine good step sizes. Since we have a chi-square function, a 1-sigma
    # change in a parameter corresponds to a 1 unit change in the output
    # chi-square function. We want our steps to change the chi-square function
    # by an amount of roughly 1e-5 (far from machine precision, but small
    # enough to be probing locally). We start by guessing a step size of 1e-5
    # (which is typically pretty reasonable for parameters that are of order 1)
    # and then bisect to find the right value.
    # We also cap the maximum step size for parameters at 1. For parameters of
    # order unity, this means that a change of 1 unit should make an
    # appreciable difference in the resulting chi-square. If this doesn't
    # happen, then something is wrong. An example of where this can happen is
    # when the
    steps = []
    ref_chisq = chisq_function(values)

    for parameter_idx in range(len(parameter_names)):
        step = 1e-5
        min_step = None
        max_step = None

        # Move away from the nearest bounds to avoid boundary issues.
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        if min_bound is None:
            if max_bound is None:
                # No bounds, doesn't matter what we pick.
                direction = +1.
            else:
                # Max bound only
                direction = -1.
        else:
            if max_bound is None:
                # Min bound only
                direction = +1.
            else:
                # Both bounds, move away from the nearest bound.
                if value - min_bound > max_bound - value:
                    direction = -1.
                else:
                    direction = 1.

        while True:
            # Estimate the second derivative numerator for a finite difference
            # calculation. We want to choose a step size that sets this to a
            # reasonable value. Note that we move only in the direction away
            # from the nearest boundary, so this isn't centered at the correct
            # position, but this is only to get an initial estimate of the
            # scale so it doesn't matter.
            step_values = values.copy()
            step_values[parameter_idx] += step * direction
            step_1_chisq = chisq_function(step_values)
            step_values[parameter_idx] += step * direction
            step_2_chisq = chisq_function(step_values)
            diff = 0.25 * step_2_chisq - 0.5 * step_1_chisq + 0.25 * ref_chisq

            if diff < -1e-4:
                # We found a minimum that is better than the supposed true
                # minimum. This indicates that something is wrong because the
                # minimizer failed.
                raise PsfModelException(
                    "Found better minimum while varying %s to calculate "
                    "covariance matrix! Fit failed!" %
                    parameter_names[parameter_idx]
                )

            if diff < 1e-6:
                # Too small step size, increase it.
                min_step = step
                if max_step is not None:
                    step = (step + max_step) / 2.
                else:
                    step = step * 2.
            elif diff > 1e-4:
                # Too large step size, decrease it.
                max_step = step
                if min_step is not None:
                    step = (step + min_step) / 2.
                else:
                    step = step / 2.
            elif step > 1e9:
                # Shouldn't need steps this large. This only happens if one
                # parameter doesn't affect the model at all, in which case we
                # can't calculate the covariance.
                raise PsfModelException(
                    "Parameter %s doesn't appear to affect the model! Cannot "
                    "estimate the covariance." % parameter_names[parameter_idx]
                )
            else:
                # Good step size, we're done.
                break

        steps.append(step)

    steps = np.array(steps)
    if verbose:
        print("Finite difference covariance step sizes: %s" % steps)

    difference_matrices = []

    # If we are too close to a boundary, shift the center position by the step
    # size. This isn't technically right, but we can't evaluate past bounds or
    # we run into major issues with parameters that have real physical bounds
    # (eg: airmass below 1.)
    original_values = values
    values = original_values.copy()
    for parameter_idx in range(len(parameter_names)):
        name = parameter_names[parameter_idx]
        min_bound, max_bound = bounds[parameter_idx]
        value = values[parameter_idx]
        step = steps[parameter_idx]

        direction_str = None
        if min_bound is not None and value - 2*step < min_bound:
            direction_str = "up"
            values[parameter_idx] = min_bound + 2*step
        elif max_bound is not None and value + 2*step > max_bound:
            direction_str = "down"
            values[parameter_idx] = max_bound - 2*step

        if direction_str is not None and verbose:
            print("WARNING: Parameter %s is at bound! Moving %s by %g to "
                  "calculate covariance!" % (name, direction_str, 2*step))

    # Calculate all of the terms that will be required to calculate the finite
    # differences. Note that there is a lot of reuse of terms, so here we
    # calculate everything that is needed and build a set of matrices for each
    # step combination.
    for weight, sign_e1, sign_e2 in difference_info:
        matrix = np.zeros((num_variables, num_variables))
        for i in range(num_variables):
            for j in range(num_variables):
                if i > j:
                    # Symmetric
                    continue

                step_values = values.copy()
                step_values[i] += sign_e1 * steps[i]
                step_values[j] += sign_e2 * steps[j]
                chisq = chisq_function(step_values)
                matrix[i, j] = chisq
                matrix[j, i] = chisq

        difference_matrices.append(matrix)

    # Hessian
    hessian = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            if i > j:
                continue

            val = 0.

            for (weight, sign_e1, sign_e2), matrix in \
                    zip(difference_info, difference_matrices):
                val += weight * matrix[i, j]

            val /= steps[i] * steps[j]

            hessian[i, j] = val
            hessian[j, i] = val

    # Invert the Hessian to get the covariance matrix
    try:
        cov = hessian_to_covariance(hessian)
    except LinAlgError:
        raise PsfModelException("Covariance matrix is not well defined!")

    variance = np.diag(cov)

    if np.any(variance < 0):
        raise PsfModelException("Covariance matrix is not well defined! "
                                "Found negative variances.")

    return cov


def calculate_covariance_minuit(chisq_function, parameter_names, start_values,
                                bounds, verbose=False):
    num_variables = len(parameter_names)

    minuit_kwargs = {}

    # PyMinuit requires that names be less than 10 characters. Relabel all of
    # our variables to p1, p2, etc. to deal with this.
    safe_names = ['p%d' % i for i in range(len(parameter_names))]

    for name, value, bound in zip(safe_names, start_values, bounds):
        minuit_kwargs[name] = value
        minuit_kwargs['error_%s' % name] = 0.5

        # PyMinuit can't handle when both of the bounds are None
        if bound[0] is not None:
            if bound[1] is None and minuit_version == 'PyMinuit':
                raise PsfModelException(
                    "PyMinuit can't handle one-sided bounds!"
                )
            minuit_kwargs['limit_%s' % name] = bound

    # Just using *args works in iminuit if forced_parameters is specified, but
    # PyMinuit needs explicit parameter names. Ugh. We build a function on the
    # fly with the right parameter names.
    minuit_args = ', '.join(safe_names)
    minuit_func_str = ('def func(%s):\n    return chisq_function([%s])\n' %
                       (minuit_args, minuit_args))
    exec_vars = {'chisq_function': chisq_function}
    exec(minuit_func_str, exec_vars)
    minuit_wrapper = exec_vars["func"]

    m = minuit.Minuit(
        minuit_wrapper,
        # forced_parameters=parameter_names,
        errordef=1.,
        print_level=0,
        **minuit_kwargs
    )

    # Do a search for the minimum. This shouldn't go anywhere if the fit
    # was already good.
    migrad_res = m.migrad()

    if minuit_version == 'iminuit':
        if not (migrad_res[0]['is_valid'] and
                migrad_res[0]['has_accurate_covar']):
            # Minuit failed.
            raise PsfModelException('Minuit estimate of covariance failed')
    else:
        # No estimate of fit success available with PyMinuit. We check that we
        # are at the same minimum and things like that below, so we should
        # still be fine.
        pass

    # Recalculate the covariance to get the best estimate possible.
    if verbose:
        if minuit_version == 'iminuit':
            m.set_print_level(1)
        elif minuit_version == 'PyMinuit':
            m.printMode = 1
        print("Minuit covariance estimate")
        print("Location:", start_values)
    m.hesse()

    # Make sure that the fit didn't change anything.
    orig_fmin = chisq_function(start_values)
    minuit_fmin = m.fval
    if np.abs(orig_fmin - minuit_fmin) > max(1e-4, 1e-6*orig_fmin):
        error_str = ("Minuit covariance estimate found a different minimum!"
                     " (minuit=%g, pipeline=%g). Covariance estimation failed!"
                     % (minuit_fmin, orig_fmin))
        raise PsfModelException(error_str)

    for name, start_value, minuit_value in zip(safe_names, start_values,
                                               m.args):
        diff = start_value - minuit_value
        if np.abs(diff) > 1e-3:
            error_str = ("Minuit covariance estimate found a different "
                         "best-fit value for %s! (minuit=%g, pipeline=%g). "
                         "Covariance estimation failed!"
                         % (name, minuit_value, start_value))
            raise PsfModelException(error_str)

    # Map the minuit covariance matrix (which is a dict) to a matrix.
    parameter_idx_map = {name: idx for idx, name in enumerate(safe_names)}
    cov = np.zeros((num_variables, num_variables))
    for (key_1, key_2), value in m.covariance.items():
        cov[parameter_idx_map[key_1], parameter_idx_map[key_2]] = value

    return cov


def _plot_correlation(instance, names=None, covariance=None, **kwargs):
    """Plot the correlation matrix between the fit parameters.

    If the covariance matrix has already been calculated, then it can be
    passed in with the names and covariance parameters and it won't be
    recalculated.
    """
    if covariance is not None:
        if names is None:
            raise PsfModelException(
                "Need to specify both names and covariance to used a "
                "previously-calculated covariance matrix!"
            )
    else:
        names, covariance = instance.calculate_covariance(**kwargs)

    uncertainties = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(uncertainties, uncertainties)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(correlation, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)

    plt.xticks(np.arange(len(names)), names, rotation='vertical')
    plt.yticks(np.arange(len(names)), names)
    plt.tight_layout()
    plt.colorbar()


def _evaluate_priors(parameters, priors):
    """Evaluate priors on a set of parameters.

    parameters should be a dictionary of parameters with their corresponding
    values.

    priors should be a dictionary of the priors, with the dictionary entries
    specifying the prior to use.
    """
    total_prior = 0.
    for parameter, prior_dict in priors.items():
        parameter_value = parameters[parameter]

        center = prior_dict['value']
        width = prior_dict['width']

        # Gaussian prior
        prior = (parameter_value - center)**2 / width**2
        total_prior += prior

    return total_prior


def print_parameter_header(do_uncertainties, do_initial_values):
    """Print a header for the parameter information"""
    if do_initial_values:
        initial_value_str = "--Guess--  "
    else:
        initial_value_str = ""

    if do_uncertainties:
        fit_str = "----------Fit---------  "
    else:
        fit_str = "----Fit---  "

    header = ("------Parameter-----  %s%s---Note---" %
              (initial_value_str, fit_str))
    print(header)


def print_parameter(parameter_name, parameter_dict, do_uncertainties,
                    do_initial_values):
    """Print out a parameter and related information in a standardized
    format.
    """
    value = parameter_dict['value']
    lower_bound, upper_bound = parameter_dict['bounds']

    if 'fixed' in parameter_dict and parameter_dict['fixed']:
        message = "fixed"
    elif 'derived' in parameter_dict and parameter_dict['derived']:
        message = "derived"
    elif lower_bound is None or upper_bound is None:
        message = ""
    elif lower_bound == upper_bound:
        message = "fixed"
    else:
        fraction = ((value - lower_bound) / (upper_bound - lower_bound))
        if fraction < 0:
            message = "<-- ERROR: BELOW LOWER BOUND!"
        elif fraction > 1:
            message = "<-- ERROR: ABOVE UPPER BOUND!"
        if fraction < 0.0001:
            message = "<-- lower bound"
        elif fraction > 0.9999:
            message = "<-- upper bound"
        else:
            message = ""

    # If there is no value specified, set to np.nan
    if value is None:
        value = np.nan

    if do_uncertainties:
        if 'uncertainty' in parameter_dict:
            uncertainty = parameter_dict['uncertainty']
            uncertainty_str = "± %-9.3g  " % uncertainty
        else:
            uncertainty_str = 13 * " "
    else:
        uncertainty_str = ""

    if do_initial_values:
        if 'initial_value' in parameter_dict:
            initial_value = parameter_dict['initial_value']
            initial_value_str = "%10.4g " % initial_value
        else:
            initial_value_str = 11 * " "
    else:
        initial_value_str = ""

    print("%20s %s %10.4g %s%s" % (parameter_name, initial_value_str, value,
                                   uncertainty_str, message))


def _real_to_fourier(real_components, grid_info):
    """Convert real components to Fourier ones.

    This can handle either single components or lists of multiple
    components.
    """
    if debug_fourier:
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


def _fourier_to_real(fourier_components, grid_info):
    """Convert Fourier components to real ones.

    This can handle either single components or lists of multiple
    components.
    """
    if debug_fourier:
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


class DumbLRUCache(object):
    """Dumb LRU cache

    This caches the last max_size calls to a function. When the cache is full,
    the least recently used call is discarded.

    In Python3 there is an lru_cache decorator in the standard library, but
    that doesn't exist in Python2 so I wrote this simple class. It is dumb, and
    not optimized at all, but it works.
    """
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.cache = []

        self.hits = 0
        self.misses = 0

    def __getitem__(self, key):
        """Return the cache value for a given key.

        This returns None in the key wasn't in the cache. The item is also
        moved to the back of the cache. We manually check each key against the
        cache, so the key doesn't need to be hashable. This isn't optimal for
        many use cases, but I want to use this for dicts and it works well for
        that.
        """
        for index, (cache_key, value) in enumerate(self.cache):
            if key == cache_key:
                # Found it. Move the entry to the end of the cache.
                self.cache.append(self.cache.pop(index))

                # Return the value
                self.hits += 1
                return value

        # Not in the cache
        self.misses += 1
        return None

    def __setitem__(self, key, value):
        """Add a value to the cache."""
        # Make sure that the item isn't in the cache already. Ideally I would
        # write something that does the getting and setting together, but I
        # didn't.
        test_value = self[key]
        if test_value is not None:
            raise PsfModelException(
                "key %s is already in the cache. This shouldn't happen!" % key
            )

        # Add the item
        self.cache.append((key, value))

        # Keep the cache within the maximum size.
        if len(self.cache) > self.max_size:
            self.cache.pop(0)

    def clear(self):
        """Clear the cache"""
        self.cache = []


class ModelParameterDictionary(object):
    """Wrap a dictionary with SceneModel names so that it can be accessed by an
    element with its internal names.

    This is used for transparently handling models where prefixes have been
    added or where parameters have been renamed. For example, the SceneModel
    could represent the instrumental PSF with a Gaussian with a prefix of
    "inst_". The SceneModel would call the width of the Gaussian
    "inst_sigma_x", but the ModelElement representing that part of the PSF
    would call it "sigma_x" without the prefix.

    This doesn't implement the full dict api... that should be probably be
    filled out later.
    """
    def __init__(self, model_dict, name_map):
        """Initialize the model parameter dictionary.

        model_dict is a dictionary of model parameters with the full names used
        in the SceneModel.

        name_map is a dictionary that maps the element's internal keys to the
        ones that are actually in the model.

        For example:

            model_dict = {'inst_sigma_x': 0.5}
            name_map = {'sigma_x': 'inst_sigma_x'}
        """
        self.model_dict = model_dict
        self.name_map = name_map

    def __getitem__(self, key):
        try:
            map_key = self.name_map[key]
        except KeyError:
            raise PsfModelException("Unknown parameter %s!" % key)

        try:
            return self.model_dict[map_key]
        except KeyError:
            raise PsfModelException("No value available for %s!" % map_key)

    def __setitem__(self, key, value):
        """Set the value of a given parameter"""
        try:
            map_key = self.name_map[key]
        except KeyError:
            raise PsfModelException("Unknown parameter %s!" % key)

        self.model_dict[map_key] = value

    def __contains__(self, key):
        try:
            map_key = self.name_map[key]
        except KeyError:
            raise PsfModelException("Unknown parameter %s!" % key)

        return map_key in self.model_dict

    def get_element_parameters(self):
        """Return a dictionary of the element parameter keys and values.

        Note that parameters shouldn't be modified in this dictionary as they
        won't propagate back to the original dictionary
        """
        result = {}
        for key, map_key in self.name_map.items():
            try:
                result[key] = self.model_dict[map_key]
            except KeyError:
                raise PsfModelException("No value available for %s!" % map_key)

        return result


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
                 subsampling=default_subsampling, border=default_border,
                 shared_parameters=[], **kwargs):
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

                    raise PsfModelException(
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

        wx = 2 * np.pi * np.fft.fftfreq(pad_grid_x.shape[0], step)
        wy = 2 * np.pi * np.fft.fftfreq(pad_grid_x.shape[1], step)
        pad_grid_wx, pad_grid_wy = np.meshgrid(wx, wy, indexing='ij')

        subsampling_buffer = subsampling_buffer
        pad_grid_x = pad_grid_x
        pad_grid_y = pad_grid_y
        pad_grid_wx = pad_grid_wx
        pad_grid_wy = pad_grid_wy

        # Calculate the offset that the center position needs to be moved by to
        # be in the right location on the padded grid.
        pad_offset_x = border + subsampling_buffer + reference_i
        pad_offset_y = border + subsampling_buffer + reference_j

        # Calculate the Fourier shift that needs to be applied to get an FFTed
        # Fourier space model to the reference position and back.
        pad_ifft_shift = np.exp(
            - 1j * (pad_offset_x * pad_grid_wx + pad_offset_y * pad_grid_wy)
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
            'pad_grid_wx': pad_grid_wx,
            'pad_grid_wy': pad_grid_wy,

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
            raise PsfModelException(
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

        # First, we estimate the position of the object using a median filter.
        # Then, we estimate the background by taking the median of the pixels
        # farthest away from the object.
        medfilt_image = medfilt2d(image, 3)
        med_center_i, med_center_j = np.unravel_index(
            np.nanargmax(medfilt_image, axis=None),
            medfilt_image.shape
        )
        med_center_x = grid_x[med_center_i, med_center_j]
        med_center_y = grid_y[med_center_i, med_center_j]

        # Estimate the background by taking the median of the pixels farthest
        # from the guessed object location in the image.
        r2 = (mask_x - med_center_x)**2 + (mask_y - med_center_y)**2
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

        # Reestimate the center position using a weighted average of the flux
        # in the region near the detected object.
        sub_data = mask_data - background_guess
        weights = sub_data[source_mask]
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
        raise PsfModelException("No parameter %s found!" % parameter_name)

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
            if debug_fourier:
                print(type(element).__name__)

            old_mode = mode
            model, mode, parameters = element.update_model(
                grid_info, model, mode, parameters, separate_components,
                apply_coefficients
            )

            if debug_fourier:
                print("    %s -> %s" % (old_mode, mode))

        if debug_fourier:
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
            element.fix(**{key: value})

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
                   apply_priors=True):
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
        if use_autograd:
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
            # TODO: priors
            # prior_penalty = self.evaluate_priors(full_parameters)
            # chi_square += prior_penalty
            pass

        if return_full_info:
            return_val = chi_square, fit_model, full_parameters
        else:
            return_val = chi_square

        # Update the cache
        if do_cache:
            self._chi_square_cache[call_info] = return_val

        return return_val

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
            raise PsfModelException("Found degenerate model for parameters: %s"
                                    % parameters)
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
            if (parameter_dict['fixed'] or parameter_dict['derived'] or
                    (parameter_dict['coefficient'] and
                     do_analytic_coefficients)):
                # Parameter that isn't fitted, ignore it.
                continue

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

        def chi_square_flat(x, return_full_info=False, apply_fit_scale=True):
            map_parameters = {i: j for i, j in zip(parameter_names, x)}

            return self.chi_square(
                map_parameters,
                do_analytic_coefficients=do_analytic_coefficients,
                return_full_info=return_full_info,
                apply_fit_scale=apply_fit_scale,
            )

        res = minimize(
            chi_square_flat,
            initial_values,
            bounds=bounds,
            jac=grad(chi_square_flat) if use_autograd else None,
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-12},
        )

        if not res.success:
            print(res)
            raise PsfModelException("Fit failed!", res.message)

        # Retrieve the unscaled parameters.
        fit_parameters = res.x * scales
        chi_square, full_model, full_parameters = chi_square_flat(
            fit_parameters, return_full_info=True, apply_fit_scale=False,
        )

        # Save the fit results.
        self.set_parameters(update_derived=False, **full_parameters)
        self.fit_model = full_model
        self.fit_result = res
        self.fit_chi_square = chi_square

        if verbose:
            self.print_fit_info()

        return full_parameters, full_model

    def calculate_covariance(self, verbose=False, method='autograd' if
                             use_autograd else 'finite_difference'):
        """Estimate the covariance matrix using a numerical estimate of the
        Hessian.

        This method should only be called when the function has been fit to an
        image and the current parameter values are at the minimum of the
        chi-square.

        This only works for chi-square fitting, and will produce unreliable
        results for least-squares fits.

        The covariance matrix can be calculated with three different methods:
        - finite_difference: a custom finite difference covariance calculator
        that uses adaptive step sizes.
        - minuit: a hook into iminuit or PyMinuit's Hesse algorithm which runs
        a finite difference routine.
        - autograd: an automatic differentiation package that analytically
        evaluates the Hessian. This is typically the fastest method for
        complicated models and is extremely accurate.
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
                apply_fit_scale=True
            )

        if method == 'minuit':
            cov = calculate_covariance_minuit(
                evaluate_chi_square, parameter_names, values, bounds=bounds,
                verbose=verbose
            )
        elif method == 'finite_difference':
            cov = calculate_covariance_finite_difference(
                evaluate_chi_square, parameter_names, values, bounds,
                verbose=verbose
            )
        elif method == 'autograd':
            self.clear_cache()
            hess = hessian(evaluate_chi_square)(values)
            cov = hessian_to_covariance(hess)
        else:
            raise PsfModelException('Unknown method %s' % method)

        # Rescale parameters.
        cov *= np.outer(scales, scales)

        return parameter_names, cov

    def plot_correlation(instance, names=None, covariance=None, **kwargs):
        """Plot the correlation matrix between the fit parameters.

        See _plot_correlation for details, this is just a wrapper around it.
        """
        _plot_correlation(instance, names, covariance, **kwargs)

    def calculate_uncertainties(self, names=None, covariance=None):
        """Estimate the uncertainties on all fit parameters.

        This is done by calculating the full covariance matrix and returning a
        dict matching parameters to the errors.

        This will throw a PsfModelException in the covariance matrix is not
        well defined.

        If the covariance matrix has already been calculated, then it can be
        passed in with the names and covariance parameters and it won't be
        recalculated.
        """
        if covariance is not None:
            if names is None:
                raise PsfModelException(
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
            print("%20s: %g/%g" % ('chi2/dof', self.fit_result.fun,
                                   self.degrees_of_freedom))
            print("%20s: %d" % ('nfev', self.fit_result.nfev))

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
            model = _fourier_to_real(model, grid_info)
            mode = 'real'
        elif mode != 'real':
            raise PsfModelException("PSF generation ended up in unknown mode "
                                    "%s. Can't handle!" % mode)

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
            raise PsfModelException(
                "grid sizes should be odd or the maximum value will not be "
                "found!"
            )

        if subsampling % 2 == 0:
            raise PsfModelException(
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
                    raise PsfModelException(
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

    def get_fits_header_items(self, prefix=default_fits_prefix):
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
        fits_list.append((prefix + 'LREF', reference_wavelength,
                          'Reference wavelength [A]'))

        return fits_list

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
            raise PsfModelException("%s has no parameter named %s." %
                                    (type(self), original_name))

        if new_name in self._parameter_info:
            raise PsfModelException("%s already has a parameter named %s." %
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
                raise PsfModelException(
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
                raise PsfModelException("%s has no parameter named %s." %
                                        (type(self), key))

            parameter_dict = self._parameter_info[key]

            # Make sure that we aren't changing any derived parameters unless
            # explicitly told to do so.
            if parameter_dict['derived'] and update_derived:
                raise PsfModelException(
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

        if use_autograd:
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


class MultipleImageFitter():
    def __init__(self, scene_model, images, variances=None, **kwargs):
        """Initialze a fitter for multiple images.

        The fitter will fit a scene model to every image. scene_model is either
        a previously set up SceneModel object, or a callable that generates a
        SceneModel. If a previously set up SceneModel is passed, it will be
        copied with copy.deepcopy for each image that is being fit.

        images is a list of 2-dimensional images to be fit. variances should
        have the same shape of images if it is specified.
        """
        scene_models = []
        for index in range(len(images)):
            image = images[index]
            if variances is not None:
                variance = variances[index]
            else:
                variance = None

            if callable(scene_model):
                new_scene_model = scene_model()
            else:
                new_scene_model = deepcopy(scene_model)

            new_scene_model.load_image(image, variance, **kwargs)

            scene_models.append(new_scene_model)

        self._base_scene_model = scene_model
        self.scene_models = scene_models

        self._global_fit_parameters = OrderedDict()
        self._global_fixed_parameters = OrderedDict()

        self.fit_result = None
        self.fit_initial_values = None

    def fix(self, **kwargs):
        """Fix a parameter to a specific value for all the models.

        Each key is a parameter name, and either a single value or a list can
        be specified for the key's value. If a single value is given, then
        all images are set to that value. If a list is given, then the list
        must have the same length as the number of images, and one entry will
        be assigned to each image.
        """
        for key, value in kwargs.items():
            if np.isscalar(value):
                for scene_model in self.scene_models:
                    scene_model.fix(**{key: value})
                self._global_fixed_parameters[key] = value
            else:
                for scene_model, model_value in zip(self.scene_models, value):
                    scene_model.fix(**{key: model_value})

    def add_global_fit_parameter(self, name, initial_guess=None, bounds=None):
        """Fix a parameter so that it is fit globally for all images"""

        # If initial_guess or bounds is None, pull it from the first image.
        if initial_guess is None:
            initial_guess = self.scene_models[0]._parameter_info[name]['value']
        if bounds is None:
            bounds = self.scene_models[0]._parameter_info[name]['bounds']

        new_parameter = {
            'value': initial_guess,
            'bounds': bounds,
        }

        self._global_fit_parameters[name] = new_parameter

        for scene_model in self.scene_models:
            scene_model._modify_parameter(name, fixed=True, **new_parameter)

    def _get_fit_parameters(self, do_analytic_coefficients=True,
                            apply_fit_scale=True):
        """Return the information needed to do a fit with a flat parameter
        vector (what scipy needs)

        If do_analytic_coefficients is True, then parameters are analytically
        solved for if possible (eg: amplitude, background) and aren't included
        as fit parameters.

        This method returns both an OrderedDict with the global parameter names
        and parameter dictionaries, and a list of OrderedDicts with the
        parameter information for each model as specified in
        PsfModel._get_fit_parameters.
        """
        # Get the global parameter information. These are all used in the fit,
        # so we can just use the dictionary that we have already.
        global_parameters = self._global_fit_parameters

        # Get the parameter information for each model.
        individual_parameters = []
        for scene_model in self.scene_models:
            model_fit_parameters = scene_model._get_fit_parameters(
                do_analytic_coefficients=do_analytic_coefficients,
                apply_fit_scale=apply_fit_scale
            )
            individual_parameters.append(model_fit_parameters)

        return global_parameters, individual_parameters

    def chi_square(self, global_parameters={}, individual_parameters=None,
                   do_analytic_coefficients=True, save_parameters=False,
                   apply_fit_scale=False, apply_priors=True):
        """Calculate the chi-square value for a given set of PSF parameters.

        global_parameters is a dictionary of parameters that are passed to
        every model.

        individual_parameters is a list with the same length as the number of
        models that are being fit. Each element of the list is a dictionary
        with the parameters to pass to the corresponding model's chi_square
        function.

        If do_analytic_coefficients is True, then the parameters that represent
        the scales of the model components are recalculated analytically.

        If save_parameters is True, the underlying models are updated to save
        the given parameters. Otherwise, they are unaffected.

        If apply_fit_scale is True, then the individual images are scaled by a
        predetermined amount that was calculated in _process_image, chosen such
        that the amplitude of the PSF is ~1. See SceneModel.chi_square for
        details.
        """
        if individual_parameters is None:
            # Use the default individual parameters.
            individual_parameters = [{} for i in range(len(self.scene_models))]

        # Add any manually specified parameters to the internal ones.
        global_parameters = dict(self.global_parameters, **global_parameters)

        total_chi_square = 0

        for scene_model, base_parameters in zip(self.scene_models,
                                                individual_parameters):
            parameters = dict(base_parameters, **global_parameters)

            model_chi_square = scene_model.chi_square(
                parameters,
                do_analytic_coefficients=do_analytic_coefficients,
                return_full_info=save_parameters,
                apply_fit_scale=apply_fit_scale
            )

            if save_parameters:
                model_chi_square, fit_model, full_parameters = model_chi_square
                scene_model.set_parameters(update_derived=False,
                                           **full_parameters)
                scene_model.fit_model = fit_model

            total_chi_square += model_chi_square

        if apply_priors:
            # TODO: priors
            # prior_penalty = self.evaluate_priors(global_parameters)
            # total_chi_square += prior_penalty
            pass

        if save_parameters:
            # Save global parameters:
            for parameter_name, value in global_parameters.items():
                self._global_fit_parameters[parameter_name]['value'] = value

        return total_chi_square

    def evaluate_priors(self, parameters):
        """Evaluate the priors that have been set"""
        return _evaluate_priors(parameters, self._priors)

    def _parse_start_fit_parameters(self, global_parameters,
                                    individual_parameters):
        """Parse fit parameters information and return a list of initial values
        and bounds that can be passed to a fitter.
        """
        parameter_names = []
        initial_values = []
        all_bounds = []
        scales = []

        # Global parameters
        for key, parameter_dict in global_parameters.items():
            parameter_names.append(key)
            initial_values.append(parameter_dict['value'])
            all_bounds.append(parameter_dict['bounds'])
            scales.append(1.)

        # Individual parameters for each model
        for index, model_dict in enumerate(individual_parameters):
            for key, parameter_dict in model_dict.items():
                initial_value = parameter_dict['value']
                bounds = parameter_dict['bounds']
                scale = parameter_dict['scale']

                parameter_names.append('%s[%d]' % (key, index))
                initial_values.append(initial_value)
                all_bounds.append(bounds)
                scales.append(scale)

        initial_values = np.array(initial_values)

        return parameter_names, initial_values, all_bounds, scales

    def _map_fit_parameters(self, flat_parameters, global_parameters,
                            individual_parameters):
        """Map a flat array of fit parameters to what is needed to calculate a
        chi-square.
        """
        index = 0

        fit_global_parameters = {}
        for key, parameter_dict in global_parameters.items():
            fit_global_parameters[key] = flat_parameters[index]
            index += 1

        fit_individual_parameters = []
        for model_parameters in individual_parameters:
            fit_model_parameters = {}
            for key, parameter_dict in model_parameters.items():
                fit_model_parameters[key] = flat_parameters[index]
                index += 1
            fit_individual_parameters.append(fit_model_parameters)

        return fit_global_parameters, fit_individual_parameters

    def evaluate(self, *args, **kwargs):
        """Evaluate each model and return a stacked array."""
        models = []
        for scene_model in self.scene_models:
            model = scene_model.evaluate(*args, **kwargs)
            models.append(model)
        models = np.array(models)
        return models

    def fit(self, do_analytic_coefficients=True, verbose=False):
        global_parameters, individual_parameters = \
            self._get_fit_parameters(do_analytic_coefficients)

        parameter_names, initial_values, bounds, scales = \
            self._parse_start_fit_parameters(global_parameters,
                                             individual_parameters)

        def chi_square_flat(flat_parameters, save_parameters=False,
                            apply_fit_scale=True):
            flat_parameters = flat_parameters
            fit_global_parameters, fit_individual_parameters = \
                self._map_fit_parameters(flat_parameters, global_parameters,
                                         individual_parameters)

            chi_square = self.chi_square(
                fit_global_parameters,
                fit_individual_parameters,
                do_analytic_coefficients=do_analytic_coefficients,
                save_parameters=save_parameters,
                apply_fit_scale=apply_fit_scale,
            )

            return chi_square

        res = minimize(
            chi_square_flat,
            initial_values,
            bounds=bounds,
            jac=grad(chi_square_flat) if use_autograd else None,
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-12},
        )

        if not res.success:
            raise PsfModelException("Fit failed!", res.message)

        # Retrieve the unscaled parameters and save them.
        fit_parameters = res.x * scales
        chi_square = chi_square_flat(fit_parameters, save_parameters=True,
                                     apply_fit_scale=False)

        self.fit_result = res
        self.fit_initial_values = dict(zip(parameter_names, initial_values))
        self.fit_chi_square = chi_square

        if verbose:
            self.print_fit_info()

        # Return a fitted version of the PSF with global parameters fixed.
        if callable(self._base_scene_model):
            fit_scene_model = self._base_scene_model()
        else:
            fit_scene_model = deepcopy(self._base_scene_model)

        grid_info = self.scene_models[0].grid_info
        fit_scene_model.setup_grid(
            grid_info['grid_size'],
            grid_info['border'],
            grid_info['subsampling']
        )

        fix_parameters = self._global_fixed_parameters.copy()
        for name, parameter_dict in self._global_fit_parameters.items():
            fix_parameters[name] = parameter_dict['value']
        fit_scene_model.fix(**fix_parameters)

        return fit_scene_model

    def print_fit_info(self, title=None, uncertainties=True, verbosity=1):
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

        if self.fit_result is not None:
            print("%20s: %s" % ("Fit result", self.fit_result.message))
            print("-"*70)

        print("Global parameters:\n")

        global_parameters, individual_parameters = \
            self._get_fit_parameters(do_analytic_coefficients=False,
                                     apply_fit_scale=False)

        print_parameter_header(do_uncertainties, do_initial_values)

        for parameter_name, parameter_dict in global_parameters.items():
            print_dict = parameter_dict.copy()
            if do_uncertainties and parameter_name in uncertainties:
                print_dict['uncertainty'] = uncertainties[parameter_name]
            if do_initial_values and parameter_name in initial_values:
                print_dict['initial_value'] = initial_values[parameter_name]

            print_parameter(parameter_name, print_dict, do_uncertainties,
                            do_initial_values)

        if verbosity >= 2:
            print("-"*70)
            print("Individual parameters:\n")

            print_parameter_header(do_uncertainties, do_initial_values)

            for idx, individual_dict in enumerate(individual_parameters):
                for base_parameter_name, parameter_dict in \
                        individual_dict.items():
                    print_dict = parameter_dict.copy()
                    if do_uncertainties and base_parameter_name in \
                            uncertainties:
                        uncertainty = uncertainties[base_parameter_name][idx]
                        print_dict['uncertainty'] = uncertainty
                    parameter_name = '%s[%d]' % (base_parameter_name, idx)
                    if do_initial_values and parameter_name in initial_values:
                        print_dict['initial_value'] = \
                            initial_values[parameter_name]

                    print_parameter(parameter_name, print_dict,
                                    do_uncertainties, do_initial_values)

        if self.fit_result is not None:
            print("-"*70)
            print("%20s: %g/%g" % ('chi2/dof', self.fit_result.fun,
                                   self.degrees_of_freedom))
            print("%20s: %d" % ('nfev', self.fit_result.nfev))

        print("="*70)

    def fit_and_fix_positions(self, **kwargs):
        """Fit the positions of each scene with a Gaussian, and fix them for
        future fits"""
        print("TODO: make this work!")
        for scene_model in self.scene_models:
            scene_model.fit_and_fix_position(**kwargs)

    def calculate_covariance(self, verbose=False, method='autograd' if
                             use_autograd else 'finite_difference'):
        """Estimate the covariance matrix using a numerical estimate of the
        Hessian.

        See scene_model.calculate_covariance for details.
        """
        global_parameters, individual_parameters = \
            self._get_fit_parameters(do_analytic_coefficients=False)

        parameter_names, parameter_values, bounds, scales = \
            self._parse_start_fit_parameters(global_parameters,
                                             individual_parameters)

        def chi_square_flat(flat_parameters, save_parameters=False):
            scaled_parameters = flat_parameters
            fit_global_parameters, fit_individual_parameters = \
                self._map_fit_parameters(scaled_parameters, global_parameters,
                                         individual_parameters)

            return self.chi_square(
                fit_global_parameters,
                fit_individual_parameters,
                do_analytic_coefficients=False,
                save_parameters=save_parameters,
                apply_fit_scale=True,
            )

        if method == 'minuit':
            cov = calculate_covariance_minuit(
                chi_square_flat, parameter_names, parameter_values,
                bounds=bounds, verbose=verbose
            )
        elif method == 'finite_difference':
            cov = calculate_covariance_finite_difference(
                chi_square_flat, parameter_names, parameter_values, bounds,
                verbose=verbose,
            )
        elif method == 'autograd':
            hess = hessian(chi_square_flat)(parameter_values)
            cov = hessian_to_covariance(hess)
        else:
            raise PsfModelException('Unknown method %s' % method)

        # Rescale amplitude and background parameters.
        cov *= np.outer(scales, scales)

        return parameter_names, cov

    def plot_correlation(instance, names=None, covariance=None, **kwargs):
        """Plot the correlation matrix between the fit parameters.

        See _plot_correlation for details, this is just a wrapper around it.
        """
        _plot_correlation(instance, names, covariance, **kwargs)

    def calculate_uncertainties(self, names=None, covariance=None, **kwargs):
        """Estimate the uncertainties on all fit parameters.

        This is done by calculating the full covariance matrix and returning a
        dict matching parameters to the errors.

        This method will throw a PsfModelException in the covariance matrix is
        not well defined.

        When fitting multiple images, terms that exist for every image are
        given variable names like amplitude[5]. We restack those here to
        give arrays.

        If the covariance matrix has already been calculated, then it can be
        passed in with the names and covariance parameters and it won't be
        recalculated.
        """
        if covariance is not None:
            if names is None:
                raise PsfModelException(
                    "Need to specify both names and covariance to used a "
                    "previously-calculated covariance matrix!"
                )
        else:
            names, covariance = self.calculate_covariance(**kwargs)

        uncertainties = np.sqrt(np.diag(covariance))

        array_keys = []

        result = {}
        for name, uncertainty in zip(names, uncertainties):
            if name[-1] == ']':
                # Part of an array. The elements will always be in order, so
                # we can just split on the [ character.
                key = name.split('[')[0]
                if key not in result:
                    array_keys.append(key)
                    result[key] = []
                result[key].append(uncertainty)
            else:
                result[name] = uncertainty

        for key in array_keys:
            result[key] = np.array(result[key])

        return result

    @property
    def global_parameters(self):
        """Return the current value of all of the global parameters"""
        global_parameters = {}

        for name, parameter_dict in self._global_fit_parameters.items():
            global_parameters[name] = parameter_dict['value']

        return global_parameters

    @property
    def parameters(self):
        """Return the current value of all parameters that were included in the
        fit.
        """
        global_parameters, individual_parameters = \
            self._get_fit_parameters(do_analytic_coefficients=False)

        parameters = {}

        for name, parameter_dict in global_parameters.items():
            parameters[name] = parameter_dict['value']

        individual_keys = individual_parameters[0].keys()
        for key in individual_keys:
            values = []
            for individual_dict in individual_parameters:
                values.append(individual_dict[key]['value'])
            values = np.array(values)
            parameters[key] = values

        return parameters

    @property
    def degrees_of_freedom(self):
        """Calculate how many degrees of freedom there are in the fit"""
        # Figure out how many degrees of freedom are present in each sub-model
        dof = 0
        for scene_model in self.scene_models:
            dof += scene_model.degrees_of_freedom

        # Remove global fit parameters
        dof -= len(self._global_fit_parameters)

        return dof


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
            raise PsfModelException(
                "Must define at least one of _evaluate and _evaluate_fourier!"
            )

        fourier_element = self._evaluate_fourier(
            grid_info['pad_grid_wx'],
            grid_info['pad_grid_wy'],
            subsampling,
            grid_info,
            **parameters
        )

        real_element = _fourier_to_real(fourier_element, grid_info)

        return real_element

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, **parameters):
        """Evaluate the element in Fourier space at a set of wx and wy
        frequencies with the given parameters.

        See _evaluate for details.
        """
        # If _evaluate_fourier hasn't been defined, transform the real space
        # evaluation of this element if it exists.
        if not self._has_real_evaluate():
            # Neither _evaluate nor _evaluate_fourier was defined, so there is
            # no way to evaluate the element!
            raise PsfModelException(
                "Must define at least one of _evaluate and _evaluate_fourier!"
            )

        real_element = self._evaluate(
            grid_info['pad_grid_x'],
            grid_info['pad_grid_y'],
            subsampling,
            grid_info,
            **parameters
        )

        fourier_element = _real_to_fourier(real_element, grid_info)

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
            raise PsfModelException("_cache_evaluate can't handle mode %s!" %
                                    mode)

        result = self.get_cache(
            label,
            calculator=lambda: func(x, y, subsampling, grid_info,
                                    **parameters),
            grid_info_str=grid_info['unique_str'],
            **parameters
        )

        return result

    def _cache_evaluate_fourier(self, wx, wy, subsampling, grid_info,
                                parameters):
        """Cache Fourier evaluations of this element.

        This is just a wrapper around _cache_evaluate with mode set to fourier.
        See it for details.
        """
        return self._cache_evaluate(wx, wy, subsampling, grid_info, parameters,
                                    mode='fourier')

    def _has_real_evaluate(self):
        """Check if the real evaluation is defined"""
        return self._evaluate.__func__ is not \
            SubsampledModelElement._evaluate.__func__

    def _has_fourier_evaluate(self):
        """Check if the real evaluation is defined"""
        return self._evaluate_fourier.__func__ is not \
            SubsampledModelElement._evaluate_fourier.__func__


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
                grid_info['pad_grid_wx'],
                grid_info['pad_grid_wy'],
                grid_info['subsampling'],
                grid_info,
                parameters
            )
        else:
            raise PsfModelException("SubsampledModelComponents can't be added "
                                    "to models in mode %s!" % mode)

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
            raise PsfModelException("Can't apply a convolution if there is no "
                                    "model yet. Is the order right?")
        elif mode == 'real':
            # Convert the model to Fourier space.
            model = _real_to_fourier(model, grid_info)
            mode = 'fourier'
        elif mode != 'fourier':
            raise PsfModelException("ConvolutionElement can't be applied to "
                                    "models in mode %s!" % mode)

        convolve_image = self._cache_evaluate_fourier(
            grid_info['pad_grid_wx'],
            grid_info['pad_grid_wy'],
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
            model = _real_to_fourier(model, grid_info)
            mode = 'fourier'
        elif mode != 'fourier':
            raise PsfModelException("Pixelizer can't be applied to models in "
                                    "mode %s!" % mode)

        wx = grid_info['pad_grid_wx']
        wy = grid_info['pad_grid_wy']
        subsampling = grid_info['subsampling']
        border = grid_info['border']

        # Convolve with the subpixel
        fourier_subpixel = self.get_cache(
            'fourier_pixel',
            calculator=lambda: (
                np.sinc(wx / np.pi / 2. / subsampling) *
                np.sinc(wy / np.pi / 2. / subsampling)
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
            subpixelated_model = _fourier_to_real(fourier_subpixelated_model,
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
            model = _fourier_to_real(model, grid_info)
            mode = 'real'
        elif mode != 'real':
            raise PsfModelException("RealPixelizer can't be applied to models "
                                    "in mode %s!" % mode)

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
            raise PsfModelException(
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


class PointSource(SubsampledModelComponent):
    def _setup_parameters(self):
        self._add_parameter('amplitude', None, (None, None), 'AMP',
                            'Point source amplitude', coefficient=True)
        self._add_parameter('center_x', None, (None, None), 'POSX',
                            'Point source center position X')
        self._add_parameter('center_y', None, (None, None), 'POSY',
                            'Point source center position Y')

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, amplitude,
                          center_x, center_y, **kwargs):
        # A delta function is a complex exponential in Fourier space.
        point_source_fourier = amplitude * np.exp(
            - 1j * (center_x * wx + center_y * wy)
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

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, sigma_x,
                          sigma_y, rho, **kwargs):
        gaussian = np.exp(-0.5 * (
            wx**2 * sigma_x**2 +
            wy**2 * sigma_y**2 +
            2. * wx * wy * rho * sigma_x * sigma_y
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
    exp(-(width * w)**power).

    When power is 5/3, this is a Kolmogorov PSF.
    """
    def _setup_parameters(self):
        self._add_parameter('power', 1.6, (0., 2.), 'POW', 'power')
        self._add_parameter('width', 0.5, (0.01, 30.), 'WID', 'width')

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, power, width,
                          **kwargs):
        wr = np.sqrt(wx*wx + wy*wy)

        fourier_profile = np.exp(-width**power * wr**power)

        return fourier_profile


class ChromaticExponentialPowerPsfElement(ExponentialPowerPsfElement):
    """A chromatic ExponentialPowerPsfElement to represent seeing.

    The width of the PSF takes the form:

        width = ref_width * (wave / ref_wave) ** (ref_power)
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
            raise PsfModelException("Must set wavelength!")

        p['width'] = (
            p['ref_width'] *
            (p['wavelength'] / reference_wavelength)**p['ref_power']
        )

        # Update parameters from the superclass
        parent_parameters = super(ChromaticExponentialPowerPsfElement, self).\
            _calculate_derived_parameters(p)

        return parent_parameters


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

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, wavelength,
                          adr_delta, adr_theta, **kwargs):
        if wavelength is None:
            raise PsfModelException("Must set wavelength for %s!" % type(self))
        if self.adr_model is None:
            raise PsfModelException("Must setup the ADR model for %s!" %
                                    type(self))

        adr_scale = self.adr_model.get_scale(wavelength) / self.spaxel_size

        shift_x = adr_delta * np.sin(adr_theta) * adr_scale
        shift_y = -adr_delta * np.cos(adr_theta) * adr_scale

        shift_fourier = np.exp(-1j * (shift_x * wx + shift_y * wy))

        return shift_fourier

    def set_adr(self, adr_model, spaxel_size):
        """Set the ADR model to one that has already been created."""
        # Make sure that we passed both the adr model and the spaxel size
        if adr_model is None or spaxel_size is None:
            raise PsfModelException(
                "If specifying a previously set up adr model, must also "
                "specify the spaxel size."
            )

        self.adr_model = adr_model
        self.spaxel_size = spaxel_size

    def load_adr(self, pressure, temperature, spaxel_size):
        """Load an ADR model for the PSF."""

        if pressure is None or temperature is None or spaxel_size is None:
            raise PsfModelException(
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
            raise PsfModelException("Must set wavelength!")

        # Calculate the alpha parameters as a function of wavelength.
        x = wavelength / reference_wavelength
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

    def _evaluate_fourier(self, wx, wy, subsampling, grid_info, ell_coord_x,
                          ell_coord_y, **kwargs):
        # Infinitesimally small minor axis
        rho = 0.99

        gaussian = np.exp(-0.5 * (
            wx**2 * ell_coord_x**2 +
            wy**2 * ell_coord_y**2 +
            2. * wx * wy * rho * ell_coord_x * ell_coord_y
        ))

        return gaussian


class GaussianSceneModel(SceneModel):
    """A scene model of a point source convovled with a Gaussian."""
    def __init__(self, *args, **kwargs):
        elements = [
            GaussianPointSource,
            RealPixelizer,
            Background
        ]

        super(GaussianSceneModel, self).__init__(elements, *args, **kwargs)


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

    def get_fits_header_items(self, prefix=default_fits_prefix):
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
    def from_fits_header(cls, fits_header, prefix=default_fits_prefix):
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

        # Read in element parameters and set them.
        skip_parameters = ['wavelength']
        element_parameters = {}
        for name, parameter_dict in scene_model._parameter_info.items():
            if name in skip_parameters:
                # Ignore this one
                continue

            fits_keyword = prefix + parameter_dict['fits_keyword']
            try:
                value = fits_header[fits_keyword]
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

        scene_model.set_parameters(update_derived=False, **element_parameters)

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
            raise PsfModelException(
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
