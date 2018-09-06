# -*- coding: utf-8 -*-
from scipy.optimize import minimize
from collections import OrderedDict
from scipy.linalg import pinvh, LinAlgError
from copy import deepcopy

from . import config
from .utils import SceneModelException
from .utils import print_parameter_header, print_parameter

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np
if config.use_autograd:
    from autograd import grad, hessian

# If we are using minuit for covariance estimation, figure out which version we
# have. This packages is compatible with both iminuit and PyMinuit. iminuit is
# strongly preferred. Neither is necessary to run the code, just use a
# different method to calculate the covariance matrix.
try:
    import iminuit as minuit
    minuit_version = 'iminuit'
except ImportError:
    import minuit
    minuit_version = 'PyMinuit'
except ImportError:
    minuit = None
    minuit_version = 'No minuit found'


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
                raise SceneModelException(
                    "Second derivative is negative when varying %s to "
                    "calculate covariance matrix! Something is very wrong! "
                    "(step=%f, second derivative=%f)" %
                    (parameter_names[parameter_idx], step, diff)
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
                raise SceneModelException(
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
        raise SceneModelException("Covariance matrix is not well defined!")

    variance = np.diag(cov)

    if np.any(variance < 0):
        raise SceneModelException("Covariance matrix is not well defined! "
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
                raise SceneModelException(
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
            raise SceneModelException('Minuit estimate of covariance failed')
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
        raise SceneModelException(error_str)

    for name, start_value, minuit_value in zip(safe_names, start_values,
                                               m.args):
        diff = start_value - minuit_value
        if np.abs(diff) > 1e-3:
            error_str = ("Minuit covariance estimate found a different "
                         "best-fit value for %s! (minuit=%g, pipeline=%g). "
                         "Covariance estimation failed!"
                         % (name, minuit_value, start_value))
            raise SceneModelException(error_str)

    # Map the minuit covariance matrix (which is a dict) to a matrix.
    parameter_idx_map = {name: idx for idx, name in enumerate(safe_names)}
    cov = np.zeros((num_variables, num_variables))
    for (key_1, key_2), value in m.covariance.items():
        cov[parameter_idx_map[key_1], parameter_idx_map[key_2]] = value

    return cov


def calculate_covariance(chi_square_function, method, names, values, bounds,
                         verbose=False):
        """Estimate the covariance matrix using a numerical estimate of the
        Hessian.

        This method should only be called when the function has been fit to an
        image and the current parameter values are at the minimum of the
        chi-square.

        This only works for chi-square fitting, and will produce unreliable
        results for least-squares fits.

        The covariance matrix can be calculated with one of several different
        methods that are specified with the method keyword:
        - finite_difference: a custom finite difference covariance calculator
        that uses adaptive step sizes.
        - minuit: a hook into iminuit or PyMinuit's Hesse algorithm which runs
        a finite difference routine.
        - autograd: an automatic differentiation package that analytically
        evaluates the Hessian. This is typically the fastest method for
        complex models and is the most accurate. It does have some limitations
        on what elements it can be applied to (not all numpy/scipy functions
        are supported), and requires up-to-date packages.

        The other parameters are:
        - chi_square_function is a function that takes a flat array of numbers
        and outputs a single number that is a chi-square.
        - names is a list of names for each of the parameters of the chi-square
        function in order.
        - values is the set of parameter values around which the
        Hessian/covariance will be calculated.
        - bounds is a list of tuples representing bounds for each parameter
        that shouldn't be crossed when evaluating the function.
        """
        if method == 'minuit':
            cov = calculate_covariance_minuit(
                chi_square_function, names, values, bounds=bounds,
                verbose=verbose
            )
        elif method == 'finite_difference':
            cov = calculate_covariance_finite_difference(
                chi_square_function, names, values, bounds,
                verbose=verbose
            )
        elif method == 'autograd':
            hess = hessian(chi_square_function)(values)
            cov = hessian_to_covariance(hess)
        else:
            raise SceneModelException("Unknown method for covariance "
                                      "calculation %s" % method)

        return cov


def _plot_correlation(instance, names=None, covariance=None, **kwargs):
    """Plot the correlation matrix between the fit parameters.

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
        SceneModel._get_fit_parameters.
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
            jac=grad(chi_square_flat) if config.use_autograd else None,
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-12},
        )

        if not res.success:
            raise SceneModelException("Fit failed!", res.message)

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

    def calculate_covariance(self, verbose=False,
                             method=config.default_covariance_method):
        """Estimate the covariance matrix using a numerical estimate of the
        Hessian.

        See fit.calculate_covariance for details.
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

        cov = calculate_covariance(chi_square_flat, method, parameter_names,
                                   parameter_values, bounds, verbose=verbose)

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

        This method will throw a SceneModelException in the covariance matrix
        is not well defined.

        When fitting multiple images, terms that exist for every image are
        given variable names like amplitude[5]. We restack those here to
        give arrays.

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
