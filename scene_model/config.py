# -*- coding: utf-8 -*-
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

# If we are using autograd, then we need to use a special version of numpy.
if use_autograd:
    from autograd import numpy
    default_covariance_method = 'autograd'
else:
    import numpy
    default_covariance_method = 'finite_difference'
