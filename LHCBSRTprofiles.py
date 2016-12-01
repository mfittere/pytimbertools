#! /usr/bin/env python


#import GaussFit as gfit
try:
  import numpy as np
  import matplotlib.pyplot as pl
except ImportError:
  print("""No module found: numpy, matplotlib and scipy modules should 
      be present to run pytimbertools""")

import glob
import BinaryFileIO as bio


class BSRTprofiles(object):
  """
  class to analyze BSRT profiles
  """
  @staticmethod
  def read_header(r):
    """
    read header of binary file r
    """
    time_stamp              = r.read_long(1)
    num_bunches             = r.read_int(1)
    num_acq_per_bunch       = r.read_int(1)
    sz= num_bunches
    if sz != 1:
      bunch_list          = r.read_int(sz)[:num_bunches]
    else:
      bunch_list          = [r.read_int(1)]
    acquiredImageRectangle  = r.read_int(4)
    return (time_stamp, num_bunches, num_acq_per_bunch, bunch_list, acquiredImageRectangle)

  @staticmethod
  def get_record_size(num_bunches, num_acq_per_bunch, acquiredImageRectangle):
    """
    returns the size of a record in a BSRTprofile binary file
    """
    bunch_list_sz= num_bunches
    width = acquiredImageRectangle[2]
    height = acquiredImageRectangle[3]
    prof_h_sz= width*num_bunches*num_acq_per_bunch*4
    prof_v_sz= height*num_bunches*num_acq_per_bunch*4
    r_size = 8 + 4 + 4 + bunch_list_sz*4 + 16 + (width+height)*8 + prof_h_sz + prof_v_sz
    return (width,height,prof_h_sz,prof_v_sz,r_size)

  @staticmethod
  def read_array_int(r, sz):
    return np.array(r.read_int(sz)[:sz])

  @staticmethod
  def read_array_double(r, sz):
    return np.array(r.read_double(sz)[:sz])

  def __init__(self,fn=None,records=None,profiles_h=None,profiles_v=None):
    self.records   = records
    if self.records is None:
      self.filenames = None
    else:
      self.filenames = set(self.records.keys())
    self.profiles_h = profiles_h
    self.profiles_v = profiles_v

  @classmethod
  def load_files(cls,fn=None):
    """
    fn : filename(s), options are:
           fn = type list giving a list of filenames e.g. 
                ['LHC.profiles.1.bindata'] for a single file or 
                ['LHC.profiles.1.bindata','LHC.profiles.2.bindata'] 
                for multiple files.
           fn = type string giving a search string for glob, e.g.
                '/myprofiledir/*.bindata'
    """
    profiles_h = {}
    profiles_v = {}
    records = {}
    if fn is not None:
      # open the binary file
      r = bio.BinaryReader(fn)
      # read the file
      records[fn] = [] # list of records in binary file
      offset= 0
      while True:
        try:
          # header
          (time_stamp, num_bunches, num_acq_per_bunch, bunch_list, acquiredImageRectangle)= BSRTprofiles.read_header(r)
          (width,height,prof_h_sz,prof_v_sz,r_size) = BSRTprofiles.get_record_size(num_bunches, num_acq_per_bunch, acquiredImageRectangle)
          records[fn].append((offset, time_stamp, num_bunches, num_acq_per_bunch, bunch_list,width,height,prof_h_sz,prof_v_sz,r_size))
          offset+= r_size
          # profiles
          projPositionSet1 = BSRTprofiles.read_array_double(r, width)
          projPositionSet2 = BSRTprofiles.read_array_double(r, height)
          profilesSet1 = BSRTprofiles.read_array_int(r, width*num_bunches*num_acq_per_bunch)
          profilesSet2 = BSRTprofiles.read_array_int(r, height*num_bunches*num_acq_per_bunch)
          # profiles are stored in the order [bunch_1 acquisition_1,bunch_1 acquisition_2,...,bunch_1 acquisition_m,...,bunch_n acquisition_1,...,bunch_n acquisition_m]
          for i in xrange(num_bunches):
            slotID = bunch_list[i]
            if slotID not in profiles_h.keys():
              profiles_h[slotID]=[]
            if slotID not in profiles_v.keys():
              profiles_v[slotID]=[]
            for j in xrange(num_acq_per_bunch):
              profiles_h[slotID].append((time_stamp,projPositionSet1,np.array(profilesSet1[width*(i*num_bunches+j):width*(i*num_bunches+(j+1))], dtype= np.float_)))
              profiles_v[slotID].append((time_stamp,projPositionSet2,np.array(profilesSet2[height*(i*num_bunches+j):height*(i*num_bunches+(j+1))], dtype= np.float_)))
          if r.tell() != offset:
            print('ERROR: wrong file position %s != %s = offset!'%(r.tell(),offset))
        except IOError:
          break
      r.seek(0) # resume file
    else: 
      print('ERROR: You have to give either a list of files or a search ' +
      'string for the filenames')
      return
    # convert to a structured array
    ftype=[('time_stamp',int),('pos',np.ndarray),('amp',np.ndarray)]
    for prof in profiles_h,profiles_v:
      for k in prof.keys():
        prof[k] = np.array(prof[k],dtype=ftype)
    return cls(fn=fn,records=records,profiles_h=profiles_h,profiles_v=profiles_v)
