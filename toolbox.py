import numpy as np
from scipy.stats import norm

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def gauss_pdf(x,mean,sigma):
  """
  probability distribution function for a normal or Gaussian
  distribution:
    gauss_pdf(x,mean,sigma) = 1/(sqrt(2*sigma**2*np.pi))*
                             exp(-((x-mean)**2/(2*sigma**2)))
  
  Parameters:
  -----------
  mean : mean of Gaussian distribution
  sigma : sigma of Gaussian distribution
  """
  # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
  # and y = (x - loc) / scale
  # -> loc = mean, scale = sigma
  # gauss_fit(x,mean,sigma) = norm.pdf(x,mean,sigma)/sigma
  return norm.pdf(x,mean,sigma)/sigma

def gauss_cdf(x,mean,sigma):
  """
  cumulative distribution function for a normal or Gaussian
  distribution:
    gauss_pdf(x,meam,sigma) = 1/(sqrt(2*sigma**2*np.pi))*
                                 exp(-((x-mean)**2/(2*sigma**2)))
    gauss_cdf(x,mean,sigma) = 1/2*(1+erf((x-mean)/(sigma*sqrt(2)))
  
  Parameters:
  -----------
  mean : mean of Gaussian distribution
  sigma : sigma of Gaussian distribution
  """
  # loc = mean, scale = sigma
  # gauss_pdf(x,mean,sigma) = norm.pdf(x,mean,sigma)/sigma
  # -> norm.cdf(x,mean,sigma)/sigma 
  return norm.cdf(x,mean,sigma)

def movingaverage(data,navg):
  """calculates the moving average over
  *navg* data points"""
  weights = np.repeat(1.0, navg)/navg
  dataavg = np.convolve(data, weights, 'valid')
  return dataavg
