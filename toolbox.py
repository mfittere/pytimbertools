import numpy as np

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def gauss_fit(x,mean,sigma):
  """
  function for a normal or Gaussian distribution
    f(x,mu,sig) = 1/(sqrt(2*sig**2*pi))*exp(-(x-mu)**2/(2*sig**2))
  """
  return 1/(np.sqrt(2*sigma**2*np.pi))*np.exp(-((x-mean)**2/(2*sigma**2)))

def movingaverage(data,navg):
  """calculates the moving average over
  *navg* data points"""
  weights = np.repeat(1.0, navg)/navg
  dataavg = np.convolve(data, weights, 'valid')
  return dataavg
