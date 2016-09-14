import os as os

try:
  import numpy as np
  import matplotlib.pyplot as pl
except ImportError:
  print "No module found: numpy matplotlib and scipy modules should be present to run pytimbertools"

try:
  import pytimber
except ImportError:
  print "pytimber can not be found!"

class BSRT(object):
  def __init__(self,beam,energy,t1,t2):
    self.data=[]
  @classmethod
  def get(db,ltr_dir='.',ltr_file='lhc.ltr',plt_dir='.'):

