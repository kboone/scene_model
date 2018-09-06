# -*- coding: utf-8 -*-
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

    def update_initial_values(self, instance, parameters):
        """Update the parameters dictionary and return it.

        instance is the object that the prior is associated to. This can be
        used to pull out whatever further information is necessary to get the
        initial values.

        It is fine for this function to edit the parameters dictionary in
        place, but it still must return the dictionary.
        """
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

    def update_initial_values(self, instance, parameters):
        parameters[self.parameter_name] = self.central_value

        return parameters

    def evaluate(self, parameters):
        """Evaluate a Gaussian prior"""
        current_value = parameters[self.parameter_name]

        return (current_value - self.central_value)**2 / self.sigma**2
