# -*- coding: utf-8 -*-
from __future__ import print_function

import sys

# If we are using autograd, then we need to use a special version of numpy.
from .config import numpy as np


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


class SceneModelException(Exception):
    pass


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
            raise SceneModelException(
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
            raise SceneModelException("Unknown parameter %s!" % key)

        try:
            return self.model_dict[map_key]
        except KeyError:
            raise SceneModelException("No value available for %s!" % map_key)

    def __setitem__(self, key, value):
        """Set the value of a given parameter"""
        try:
            map_key = self.name_map[key]
        except KeyError:
            raise SceneModelException("Unknown parameter %s!" % key)

        self.model_dict[map_key] = value

    def __contains__(self, key):
        try:
            map_key = self.name_map[key]
        except KeyError:
            raise SceneModelException("Unknown parameter %s!" % key)

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
                raise SceneModelException("No value available for %s!" %
                                          map_key)

        return result


def is_overridden(base_method, instance_method):
    """Check if a method is overridden.

    This needs to be handled differently in Python 2 and Python 3. It turns out
    that they implement methods in very different ways. We take care of that
    here.
    """
    if (sys.version_info > (3, 0)):
        # Python 3
        return base_method is not instance_method.__func__
    else:
        # Python 2
        return base_method.__func__ is not instance_method.__func__


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


def extract_key(item_list, key):
    """Extract the value a key from each item in a list.

    This returns a numpy array of the corresponding values. If an item is None,
    then np.nan is returned for that item's value.
    """
    result = []
    for item in item_list:
        if item is None:
            result.append(np.nan)
        else:
            result.append(item[key])
    result = np.array(result)
    return result
