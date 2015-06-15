#!/usr/bin/env python
import numpy as np


def augment(point):
  point = np.array(point,dtype=float,copy=True)
  s = np.shape(point) 
  a = np.ones(s[:-1]+(1,))
  return np.concatenate((point,a),axis=-1)


def unaugment(point):
  point = np.array(point,dtype=float,copy=True)    
  return point[...,:-1]


class Transform:
  def __init__(self,M):
    self.M = M

  def __call__(self,point):
    point = augment(point)
    out = np.einsum('ij,...j->...i',self.M,point)
    out = unaugment(out) 
    return out

  def inverse(self):
    Minv = np.linalg.inv(self.M)
    return Transform(Minv)

  def get_translation(self):
      

  def __add__(self,other):
    return Transform(self.M.dot(other.M))

  def __radd__(self,other):
    return Transform(other.M.dot(self.M))

  def __sub__(self,other):
    otherinv = other.inverse()
    return Transform(self.M.dot(otherinv.M))

  def __rsub__(self,other):
    selfinv = self.inverse()
    return Transform(other.M.dot(selfinv.M))

  
class RotationX(Transform):
  def __init__(self,arg):
    M = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, np.cos(arg), -np.sin(arg), 0.0],
                  [0.0, np.sin(arg), np.cos(arg), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class RotationY(Transform):
  def __init__(self,arg):
    M = np.array([[np.cos(arg), 0.0, np.sin(arg), 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [-np.sin(arg), 0.0, np.cos(arg), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class RotationZ(Transform):
  def __init__(self,arg):
    M = np.array([[np.cos(arg), -np.sin(arg), 0.0, 0.0],
                  [np.sin(arg), np.cos(arg), 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class Stretch(Transform):
  def __init__(self,S):
    M = np.array([[S[0], 0.0, 0.0, 0.0],
                  [0.0, S[1], 0.0, 0.0],
                  [0.0, 0.0, S[2], 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


class Translate(Transform):
  def __init__(self,T):
    M = np.array([[1.0, 0.0, 0.0, T[0]],
                  [0.0, 1.0, 0.0, T[1]],
                  [0.0, 0.0, 1.0, T[2]],
                  [0.0, 0.0, 0.0, 1.0]])
    Transform.__init__(self,M)


if __name__ == '__main__':
  T1 = Translate([0.0,1.0,0.0])
  T2 = Translate([0.0,2.0,0.0])
  S = Stretch([2.0,2.0,2.0])
  T3 = S 
  p = np.array([1.0,2.0,1.0])
  print(T3(p))
  print('hello')
