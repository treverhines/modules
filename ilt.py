#!/usr/bin/env python
from __future__ import division
import mpmath 
import numpy as np
from scipy.special import factorial

# Poke methods into the mpmath.mpf class so that numpy functions 
# know how to handle mpf instances as arguments. functions will be
# added as needed
mpmath.mpf.exp = mpmath.exp
mpmath.mpf.sin = mpmath.sin
mpmath.mpf.cos = mpmath.cos
mpmath.mpf.tan = mpmath.tan

class ILT(object):
  '''                                                                                                            
  Evaluates the inverse laplace transform of a function, fhat(s), through a 
  Taylor series expansion.

  Initiating the class with fhat(s) will compute f^(n)(0) for n up to the 
  specified N, where f^(n)(t) is the nth derivative of the inverse Laplace 
  transform of fhat(s). f^(n)(0) is computed numerically using an extension of
  the initial value theroem, which involves evaluating nested limits. This 
  requires arbitrarily high precision arithmetic through the mpmath package.  

  Calling an instance with t will evaluate the Taylor series of f(t) about t=0.

  Note: Having gmpy installed on your system will significantly improve the 
    performance of this function.  I was not able to install gmpy through 
    pip or conda and so I built it manually following the instructions here:

    https://gmpy2.readthedocs.org/en/latest/intro.html#installing-gmpy2-on-unix-linux  

  Usage
  -----
  Compute the inverse Laplace transform of 1/(s**2+1) which is sin(t):

    In [1]: import ilt

    In [2]: fhat = lambda s:1/(s**2+1) 

    In [3]: f = ilt.ILT(fhat,20)

    In [4]: t = np.linspace(np.pi,np.pi)

    In [5]: np.linalg.norm(f(t)-np.sin(t))
    Out[5]: 3.7400241737311949e-09

  '''
  def __init__(self,fhat,N,f_args=None,f_kwargs=None,s_min=1e6):
    '''
    PARAMETERS:                                                    

      fhat: function which takes arguments: (s, *f_args, **f_kwargs) and returns
        an array type object

      N: number of terms in the Taylor series expansion of f. The memory 
        requirements for this function increase exponentially with N, so 
        be careful.

      fhat_args: (default None) arguments to fhat

      fhat_kwargs: (default None) key word arguments to fhat

      s_min: (default 1e6) the smallest approximation to inifinity. This is the 
        approximate infinity used to compute f^(n)(0). the next lower derivative
        uses an approximation to infinity which is s_min**2.  Hence, the largest 
        approximation to infinity is s_min**(2**(N-1)).  The machine precision 
        adjusts accordingly to accurately represent such a large number.  You
        should run this function varying s by orders of magnitude to make sure
        that your solution is converging for large s.  When the timescales of
        f(t) are >> 1 then s needs to be increased significantly.

    '''
    assert N >= 1, (
      'at least one term is needed in the Taylor series expansion') 

    if f_args is None:
      f_args = ()
    if f_kwargs is None:
      f_kwargs = {}

    workdps = 15 + int(np.log10(s_min)*2**(N-1))
    c = []
    with mpmath.workdps(workdps):
      s_min = mpmath.mpf(s_min)
      for n in range(N):
        s = s_min**(2.0**((N-1)-n))
        a = s**(n+1)*fhat(s,*f_args,**f_kwargs)
        b = sum(s**(m+1)*c[n-m-1] for m in range(n))
        c += [a - b]

    # convert c to numpy float arrays
    self.c = []
    for c_i in c:
      c_i = np.asarray(c_i)
      c_i_flat = c_i.flatten()
      c_i_flat = [float(i) for i in c_i_flat]
      self.c += [np.reshape(c_i_flat,np.shape(c_i))]

    self.N = N

  def __call__(self,t,diff=0):
    '''
    evaluates the inverse Laplace transform of fhat at t

    Parameters
    ----------
      t: scalar or array of times where f is evaluated
      diff: (default 0) returns t evaluated at this order derivative f
 
    Returns
    -------
      array object whos dimensions are the shape of the output from fhat(s)
      plus an additional time dimension.
    '''    
    assert diff < self.N, (
      'Derivative order must be less than than the Taylor series order')
    
    t = np.asarray(t)    
    term = lambda n,val:val[...,None]*t**n/factorial(n)
    return sum(term(n,val) for n,val in enumerate(self.c[diff:]))
