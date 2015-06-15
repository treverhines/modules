#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from tplot.axes3d import Axes3D

def _augment(point):
  point = np.array(point,dtype=float,copy=True)
  s = np.shape(point) 
  a = np.ones(s[:-1]+(1,))
  return np.concatenate((point,a),axis=-1)


def _unaugment(point):
  point = np.array(point,dtype=float,copy=True)    
  return point[...,:-1]


class Transform:
  def __init__(self,M):
    M = np.asarray(M)
    self._M = M

  def __call__(self,point):
    point = _augment(point)
    out = np.einsum('ij,...j->...i',self._M,point)
    out = _unaugment(out) 
    return out

  def inverse(self):
    Minv = np.linalg.inv(self._M)
    return Transform(Minv)

  def get_M(self):
    return self._M
 
  def set_M(self,M):
    self._M = M

  def get_transformed_origin(self):
    return self._M[[0,1,2],3]

  def get_transformed_bases(self):
    return (self._M[[0,1,2],0],
            self._M[[0,1,2],1],
            self._M[[0,1,2],2])

  def __add__(self,other):
    return Transform(other.M.dot(self._M))

  def __sub__(self,other):
    return self + other.inverse()


class PointRotationX(Transform):
  '''point rotation about the x axis'''
  def __init__(self,arg):
    M = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, np.cos(arg), -np.sin(arg), 0.0],
                  [0.0, np.sin(arg), np.cos(arg), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class PointRotationY(Transform):
  '''point rotation about the y axis'''
  def __init__(self,arg):
    M = np.array([[np.cos(arg), 0.0, np.sin(arg), 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [-np.sin(arg), 0.0, np.cos(arg), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class PointRotationZ(Transform):
  '''point rotation about the z axis'''
  def __init__(self,arg):
    M = np.array([[np.cos(arg), -np.sin(arg), 0.0, 0.0],
                  [np.sin(arg), np.cos(arg), 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class PointStretch(Transform):
  '''point stretch with respect to origin'''
  def __init__(self,S):
    M = np.array([[S[0], 0.0, 0.0, 0.0],
                  [0.0, S[1], 0.0, 0.0],
                  [0.0, 0.0, S[2], 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class PointTranslation(Transform):
  '''point translation'''
  def __init__(self,T):
    M = np.array([[1.0, 0.0, 0.0, T[0]],
                  [0.0, 1.0, 0.0, T[1]],
                  [0.0, 0.0, 1.0, T[2]],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


if __name__ == '__main__':
  p = np.array([[0.0,0.0,0.0],
                [1.0,0.0,0.0],
                [1.0,-1.0,0.0],
                [0.0,-1.0,0.0]])
  p = np.array([[0.0,0.0,0.0],
                [1.0,0.0,0.0],
                [2.0,0.0,0.0]])
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.plot(p[:,0],p[:,1],p[:,2],'bo') 

  #p1 = T1(p)
  T = PointTranslation([-1.0,0.0,0.0])
  print(T.get_transformed_origin())
  print(T.get_transformed_bases())
  p2 = T(p)
  ax.plot(p2[:,0],p2[:,1],p2[:,2],'ro') 


  #T3 = T2 + T1 
  #p3 = T3(p)
  #ax.plot(p3[:,0],p3[:,1],p3[:,2],'go') 
  #print(p)

  ax.set_xlim((-3,3))
  ax.set_ylim((-3,3))
  ax.set_zlim((-3,3)) 
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.draw_basis(length=1,mutation_scale=10)
  #plt.show()
