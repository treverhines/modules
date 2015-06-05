#!/usr/bin/env python
from __future__ import division
import numpy as np
from scipy.special import factorial


def two_layer(y,D,slip,mu1,mu2,H,terms=5):
  '''
  surface displacements predicted by eq. 8 from Rybicki 1971

  Parameters
  ----------
    y: lateral distance from the dislocation   

    D: bottom of the screw dislocation (down is positive)  

    slip: The total left-lateral displacement across the fault

    mu1: Shear modulus for the top layer containing the dislocation

    mu2: Shear modulus for the bottom layer

    H: Thickness of the top layer
    
    terms: number of terms used to approximate the infinite series in
      the solution. terms=1 produces the solution for a homogeneous
      elastic halfspace and higher.


                         y=0,z=0                 
  -------------------------------------------------------
                        |                       |
                        |         mu1           |
                        |                       | H 
                         y=0,z=D                | 
                                                |
  -------------------------------------------------------

                                  mu2

  '''
  assert terms > 1

  u = np.zeros(len(y))
  r = mu2/mu1
  gamma = (1-r)/(1+r)
  u += np.arctan(D/y)
  for m in range(1,terms):
    W = np.arctan((D - 2*m*H)/y) + np.arctan((D + 2*m*H)/y) 
    u = u + gamma**m*W

  u = u*slip/(np.pi)
  return u 


def three_layer(y,p,slip,mu3,mu2,mu1,h3,h2,terms=5):
  '''
  surface displacements in a layered earth from Chinnery and Jovanovich
  1972.  This is taken from part iii and simplified with z-> 0, and
  mu4 -> 0 so that the displacements are at a free surface.

                         y=0,z=0                 
  -------------------------------------------------------
                        |                       |
                        |         mu3           |
                        |                       | h3 
                         y=0,z=p                | 
                                                |
  -------------------------------------------------------
                                                |
                                  mu2           | h2
                                                |
  -------------------------------------------------------
                                  
                                  mu1
                                                                         
  Parameters 
  ---------- 
    
    y: lateral distance from the dislocation

    p: bottom of the screw dislocation (down is positive)  

    slip: The total left-lateral displacement across the fault

    mu3: Shear modulus for the top layer containing the dislocation

    mu2: Shear modulus for the middle layer

    mu1: Shear modulus for the bottom layer

    h3: Thickness of the top layer

    h2: Thickness of the middle layer

    terms: number of terms used to approximate the infinite series in
      the solution. terms=1 produces the solution for a homogeneous
      elastic halfspace and higher.

  '''
  assert terms > 1

  u    = np.zeros(len(y))

  l_max = terms
  m_max = terms
  n_max = terms
  a1    = (mu2 - mu1) / (mu2 + mu1)
  a2    = (mu3 - mu2) / (mu3 + mu2)
  a3    = -1
  b1    = 2*mu1 / (mu1 + mu2)
  b2    = 2*mu2 / (mu2 + mu3)
  b3    = 2
  d1    = 2*mu2 / (mu1 + mu2)
  d2    = 2*mu3 / (mu2 + mu3)

  for n in range(n_max):
    W1    =           np.arctan( (  2*n*h3      - p ) / y )
    W2    =      a2 * np.arctan( ( -2*(n+1)*h3  + p ) / y )
    W3    =      a3 * np.arctan( (  2*n*h3      + p ) / y )
    W4    = a2 * a3 * np.arctan( ( -2*(n+1)*h3  - p ) / y )
    C     = (-a2*a3)**n 
    u     = u - ( C * ( W1 - W2 + W3 - W4 ) )

  for l in range(1,l_max):
    for m in range(m_max):
      for n in range(n_max):
        W1    =           np.arctan( (  2*(l+m)*h2  +  2*(l+n)*h3    - p ) / y )
        W2    =      a2 * np.arctan( ( -2*(l+m)*h2  -  2*(l+n+1)*h3  + p ) / y )
        W3    =      a3 * np.arctan( (  2*(l+m)*h2  +  2*(l+n)*h3    + p ) / y )
        W4    = a2 * a3 * np.arctan( ( -2*(l+m)*h2  -  2*(l+n+1)*h3  - p ) / y )
        C     = (-a1*a2)**m * (-a2*a3)**n * (-a1*a3*d2*b2)**l * _P(l,m,n)
        u     = u - C*( W1 - W2 + W3 - W4 )

  for l in range(l_max):
    for m in range(m_max):
      for n in range(n_max):
        W1    =      np.arctan( ( -2*(l+m+1)*h2  -  2*(l+n+1)*h3  +  p ) / y )
        W2    = a3 * np.arctan( ( -2*(l+m+1)*h2  -  2*(l+n+1)*h3  -  p ) / y )
        C     = a1 * d2 * b2 * (-a1*a2)**m * (-a2*a3)**n * (-a1*a3*d2*b2)**l * _Q(l,m,n)
        u     = u + C*(W1 + W2)

  u = u * slip / (2*np.pi)
  return u

def _P(l,m,n):
  N = factorial(n+1) * factorial(l+m-1)
  D = factorial(l) * factorial(n) * factorial(l-1) * factorial(m)
  return N / D

def _Q(l,m,n):
  N = factorial(n+l) * factorial(l+m)
  D = factorial(l) * factorial(n) * factorial(l) * factorial(m)
  return N / D

