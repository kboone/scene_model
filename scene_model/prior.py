# -*- coding: utf-8 -*-
from __future__ import print_function

from scipy.linalg import pinvh

from . import config
from .utils import SceneModelException

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


class Prior(object):
    """Class to represent a prior on the model.

    A prior adds some kind of penalty to the fitting. Any subclass must
    implement evaluate() which takes a dictionary of the model parameters as an
    argument and returns the penalty to add.

    Priors are also often used to initialize the model. When a SceneModel is
    created, it will call update_initial_values on all of the priors with a
    dictionary of initial values. To prevent this, pass set_initial_values as
    True to the initializer or set Prior.set_initial_values to False before
    creating the SceneModel.
    """
    def __init__(self, set_initial_values=True):
        """Initialize the prior.

        By default, parameters are initialized by the prior (with
        update_initial_values). This makes sense in a lot of cases, but can be
        turned off with the set_initial_values flag if necessary.
        """
        self.set_initial_values = set_initial_values

    @property
    def predicted_values(self):
        """Return a dictionary of values that are predicted by the prior.

        This is primarily used for setting initial guesses, and should be
        implemented by subclasses.
        """
        return {}

    def update_initial_values(self, parameters):
        """Update the parameters dictionary and return it.

        It is fine for this function to edit the parameters dictionary in
        place, but it still must return the dictionary.

        By default, this pulls a list of predicted values from
        Prior.predicted_values and adds them in, so only that function needs to
        be overridden.
        """
        if not self.set_initial_values:
            # Skip setting initial values.
            return parameters

        parameters.update(self.predicted_values)

        return parameters

    def evaluate(self, parameters):
        """Evaluate the prior.

        This should return a number which is the penalty to add to the
        chi-square.
        """
        return 0.


class GaussianPrior(Prior):
    def __init__(self, parameter_name, central_value, sigma, **kwargs):
        super(GaussianPrior, self).__init__(**kwargs)

        self.parameter_name = parameter_name
        self.central_value = central_value
        self.sigma = sigma

    @property
    def predicted_values(self):
        return {self.parameter_name: self.central_value}

    def evaluate(self, parameters):
        """Evaluate a Gaussian prior"""
        current_value = parameters[self.parameter_name]

        return (current_value - self.central_value)**2 / self.sigma**2


class MultivariateGaussianPrior(Prior):
    def __init__(self, parameter_names, central_values, covariance, **kwargs):
        super(MultivariateGaussianPrior, self).__init__(**kwargs)

        self.parameter_names = parameter_names
        self.central_values = np.asarray(central_values)

        self.update_covariance(covariance)

    def update_covariance(self, covariance):
        """Save the covariance and derived terms that are needed to evaluate
        the PDF of this prior
        """
        self.covariance = covariance

        self.inv_covariance = pinvh(covariance)

    @property
    def predicted_values(self):
        result = {}
        for parameter_name, central_value in zip(self.parameter_names,
                                                 self.central_values):
            result[parameter_name] = central_value

        return result

    def evaluate(self, parameters):
        """Evaluate a Multivariate Gaussian prior.

        As we work with chi-squares here, we use the analog for a multivariate
        Gaussian of dx' * cov^-1 * dx.
        """
        current_values = []
        for parameter_name in self.parameter_names:
            current_value = parameters[parameter_name]
            current_values.append(current_value)

        current_values = np.asarray(current_values)

        diff = current_values - self.central_values
        chisq = self.inv_covariance.dot(diff).dot(diff)

        return chisq
