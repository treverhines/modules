#!/usr/bin/env python
import mpmath 
import numpy as np

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
                                                          
  '''
  def __init__(self,fhat,N,f_args=None,f_kwargs=None,log_s_min=4):
    '''
    PARAMETERS:                                                    

      fhat: function which takes arguments: (s, *f_args, **f_kwargs)

      N: number of terms in the Taylor series expansion of f. The memory 
         requirements for this function increase exponentially with N, so 
         be careful.

      fhat_args: (default None) arguments to fhat

      fhat_kwargs: (default None) key word arguments to fhat

      log_s_min: (default 4) the base 10 log of the smallest approximation to 
        inifinity. This is the approximate infinity used to compute f^(n)(0).
        the next lower derivative uses an approximation to infinity which is 
        a factor of two larger in log_10 space.  Hence, the largest 
        approximation to infinity is 10**(log_s_min*2**(N-1)).  The machine 
        precision adjusts accordingly to accurately represent such a large 
        number.

    '''
    if f_args is None:
      f_args = ()
    if f_kwargs is None:
      f_kwargs = {}

    workdps = 15 + int(log_s_min*2**(N-1))
    derivatives = []
    self.coefficients = []
    with mpmath.workdps(workdps):
      s = mpmath.mpf(10**(log_s_min*2**(N-1)))
      for n in range(N):
        a = s**(n+1)*fhat(s,*f_args,**f_kwargs)
        b = sum(s**(m+1)*derivatives[n-m-1] for m in range(n))
        derivatives += [a - b]
        c = derivatives[n]/mpmath.factorial(n)
        # convert c to a numpy array with double precision floats                                                  
        c_shape = np.shape(c)
        c = np.reshape([float(i) for i in c.flatten()],c_shape)
        self.coefficients += [c]
        s = mpmath.sqrt(s)

  def __call__(self,t):

    t = np.asarray(t)
    return sum(C[...,None]*t**n for n,C in enumerate(self.coefficients))
