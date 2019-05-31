# scene_model
Scene modeling for IFUs

This code was designed to generate forward models of SNIFS IFU images to extract fluxes of point sources. It models the scene at each wavelength and can simulaneously fit for parameters across many different wavelengths or images.

The scene is composed of multiple elements that represent all of the transformations that need to be applied to the image to generate the final model. Examples of elements are:
- Adding a point source, or a galaxy to the image.
- Convolving with the atmospheric PSF.
- Atmospheric differential refraction.
- Telescope tracking
- Pixelization
- Detector artifacts

Every element can have its own set of parameters that can be fit either on single images or on a set of images. Parameters can also be shared across elements. Elements can be defined either in real or Fourier space, and the code automatically handles swapping between these two modes.

All of the model parameters can be fit to real data, using any kind of priors on the parameters. The covariance matrix can be computed for all of the parameters using several different methods. This code is a full forward model, and could be incorporated into an MCMC fitter to get Bayesian posteriors over parameters if desired. If [autograd](https://github.com/HIPS/autograd) is installed, it can be used to analytically compute derivatives of the model with respect to any model parameter.

This code was heavily inspired by the extract_star.py code written for the SNIFS instrument by Yannick Copin, Clement Buton and Emmanuel Pecontal and borrows code in several places from it.

Documentation is currently very lacking. Contact me if you are interested in using this code.
