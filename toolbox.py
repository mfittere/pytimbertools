import numpy as np
from scipy.stats import norm
from scipy.special import gamma

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def gauss_pdf(x,c,a,mu,sig):
  """
  probability distribution function for a normal or Gaussian
  distribution:
    gauss_pdf(x,mu,sig) = c+a*1/(sqrt(2*sig**2*np.pi))*
                             exp(-((x-mu)**2/(2*sig**2)))
  
  Parameters:
  -----------
  c : constant offset to fit background of profiles
  a : amplitude to compensate for *c*. a should be close to 1 if c is
      small. a should be equal to 1 if c is zero.
  mu : mean of Gaussian distribution
  sig : sigma of Gaussian distribution
  """
  # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
  # and y = (x - loc) / scale
  # -> loc = mu, scale = sig
  # gauss_fit(x,mu,sig) = norm.pdf(x,mu,sig)/sig
#  return c+a*norm.pdf(x,mu,sig)/sig
  return c+a*norm.pdf(x,mu,sig)

def qgauss_pdf(x,c,a,q,mu,beta):
  """
  Probability distribution function for a q-Gaussian distribution. A 
  q-Gaussian models Gaussian distribution with heavier tails. For q=1 
  the q-Gaussian converges versus a Gaussian.
 
    qgauss_pdf(x,c,a,mu,beta,q) = 
       c+a*(sqrt(beta)/c_q)*eq(-beta*(x-mu)**2)
  with
    c_q(q) = (sqrt(pi)*gamma((3-q)/(2*(q-1))))/(sqrt(q-1)*gamma(1/(q-1)))  
  where gamma is the gamma-function, and
    eq(x) = (1+(1-q)*x)**(1/(1-q))
  The variance of a q-Gaussian is given by:
    sigma = 1/(beta*(5-3*q)) for q < 5/3
          = infinity         for 5/3 < q < 2
          = undefined        for 2 <= q < 3
  The larger q, the heavier the tails of the distribution.

  Parameters:
  -----------
  c : constant offset to fit background of profiles
  a : amplitude to compensate for *c*. a should be close to 1 if c is
      small. a should be equal to 1 if c is zero.
  mu : mean of q-Gaussian distribution
  beta : beta of q-Gaussian distribution
  q : q of a q-Gaussian
  """
  if 1 < q and q < 3:
    c_q = ( ( np.sqrt(np.pi)*gamma((3-q)/(2*(q-1))) ) / 
            (np.sqrt(q-1)*gamma(1/(q-1))) )
    return c+a*(np.sqrt(beta)/c_q)*( (1+beta*(q-1)*(x-mu)**2)**(1/(1-q)) )
  else:
    raise ValueError('qgauss_pdf only defined for 1 < q < 3')

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

def median(x,w):
  """
  calculates the median weighted with the weights w. This is equivalent
  to calculating the 50% value of the cumulative sum of x*w.

  Example:
  --------
  x = [0.1,0.3,0.5]
  w = [1,0.5,1.5]
  median = np.median([0.1,0.1,0.3,0.5,0.5,0.5])
  """
  # each value of x has to be present w times in xm where the weighted
  # median of x is then median(xm)
  # e.g. x = [0.1,0.3,0.5],w = [1,0.5,1.5] -> w/w.min() = [2,1,3]
  # -> xm = [0.1,0.1,0.3,0.5,0.5,0.5]
  #    median(x,w) = median(xm)
  #    len(xm) = w.sum()
  # convert to integers
  wmin = w.min()
  w = w/wmin
  # for floating point numbers dividing by wmin still doesn't yield
  # integer numbers -> just round the number
  wlen = round(w.sum(),0)
  cs = (x*w).cumsum()
  cs_tot = cs[-1]
  idx_cs_50 = np.argmin(np.abs(cs-0.5*cs_tot))
  if wlen%2 == 1:
    return x[idx_cs_50]
  if wlen%2 == 0:
    return (x[idx_cs_50]+x[idx_cs_50+1])/2.
