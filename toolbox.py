import numpy as np

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def gauss_fit(x,a,mean,sigma):
  return a*np.exp(-((x-mean)**2/(2*sigma**2)))

def movingaverage(data,navg):
  """calculates the moving average over
  *navg* data points"""
  weights = np.repeat(1.0, navg)/navg
  dataavg = np.convolve(data, weights, 'valid')
  return dataavg
