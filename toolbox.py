import numpy as np
from scipy.stats import norm

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def gauss_pdf(x,c,mu,sig):
  """
  probability distribution function for a normal or Gaussian
  distribution:
    gauss_pdf(x,mu,sig) = 1/(sqrt(2*sig**2*np.pi))*
                             exp(-((x-mu)**2/(2*sig**2)))
  
  Parameters:
  -----------
  mu : mean of Gaussian distribution
  sig : sigma of Gaussian distribution
  """
  # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
  # and y = (x - loc) / scale
  # -> loc = mu, scale = sig
  # gauss_fit(x,mu,sig) = norm.pdf(x,mu,sig)/sig
  return c+norm.pdf(x,mu,sig)/sig

def qgauss_pdf(x,c,mu,sig):
  """
  probability distribution function for a q-Gaussian
  distribution:
    qgauss_pdf(x,) = 1/(sqrt(2*sig**2*np.pi))*
                             exp(-((x-mu)**2/(2*sig**2)))
  
  Parameters:
  -----------
  mu : mean of Gaussian distribution
  sig : sigma of Gaussian distribution
  """
  # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
  # and y = (x - loc) / scale
  # -> loc = mu, scale = sig
  # gauss_fit(x,mu,sig) = norm.pdf(x,mu,sig)/sig
  return c+norm.pdf(x,mu,sig)/sig

def gauss_cdf(x,mu,sig):
  """
  cumulative distribution function for a normal or Gaussian
  distribution:
    gauss_pdf(x,meam,sig) = 1/(sqrt(2*sig**2*np.pi))*
                                 exp(-((x-mu)**2/(2*sig**2)))
    gauss_cdf(x,mu,sig) = 1/2*(1+erf((x-mu)/(sig*sqrt(2)))
  
  Parameters:
  -----------
  mu : mean of Gaussian distribution
  sig : sigma of Gaussian distribution
  """
  # loc = mu, scale = sig
  # gauss_pdf(x,mu,sig) = norm.pdf(x,mu,sig)/sig
  # -> norm.cdf(x,mu,sig)/sig 
  return norm.cdf(x,mu,sig)

def movingaverage(data,navg):
  """calculates the moving average over
  *navg* data points"""
  weights = np.repeat(1.0, navg)/navg
  dataavg = np.convolve(data, weights, 'valid')
  return dataavg
