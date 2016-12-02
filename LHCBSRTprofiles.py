#! /usr/bin/env python


#import GaussFit as gfit
try:
  import numpy as np
  import matplotlib.pyplot as pl
except ImportError:
  print('No module found: numpy, matplotlib and scipy modules ' +
        'should be present to run pytimbertools')

import glob
import BinaryFileIO as bio
import localdate as ld

class BSRTprofiles(object):
  """
  class to analyze BSRT profiles

  Example:
  --------
  Load profiles from list of files (see BSRTprofiles.load_files for 
  details):
    BSRTprofiles.load_files(files='/myprofiledir/*.bindata')
  
  Attributes:
  -----------
  records : dictionary containing binary file information with the 
            format:
              {binary_file_name : [record_1,...,record_n]}
            For each record the following information is saved:
              record_* = [(offset, time_stamp, num_bunches, 
                           num_acq_per_bunch, bunch_list, width,
                           height, prof_h_sz, prof_v_sz, r_size),...]
            with:
              offset      = start position of record in the binary file
              time_stamp  = timestamp of record in unix time [ns]
              num_bunches = number of bunches
              num_acq_per_bunch = number of acquisitions per bunch
              bunch_list  = list of bunches in terms of 
                           slotID (= #bucket/10)
              width       = width of image
              height      = height of image
              prof_h_sz, prof_v_sz = size of hor./vert. profile
              r_size      = size of record
  filenames : list of binary files loaded
  profiles: dictionary containing horizontal ('h') and vertical ('v')
            profiles with the format:
                {slotID : profile }
              where profile is a structured array with the following
              fields:
                time_stamp : time stamp in unix time [ns]
                pos : position [um]
                amp : amplitude of profile [a.u.]
              
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
    return (time_stamp, num_bunches, num_acq_per_bunch, bunch_list,
            acquiredImageRectangle)

  @staticmethod
  def get_record_size(num_bunches, num_acq_per_bunch,
                      acquiredImageRectangle):
    """
    returns the size of a record in a BSRTprofile binary file
    """
    bunch_list_sz= num_bunches
    width = acquiredImageRectangle[2]
    height = acquiredImageRectangle[3]
    prof_h_sz= width*num_bunches*num_acq_per_bunch*4
    prof_v_sz= height*num_bunches*num_acq_per_bunch*4
    r_size = (8 + 4 + 4 + bunch_list_sz*4 + 16 + (width+height)*8 +
              prof_h_sz + prof_v_sz)
    return (width,height,prof_h_sz,prof_v_sz,r_size)

  @staticmethod
  def read_array_int(r, sz):
    return np.array(r.read_int(sz)[:sz])

  @staticmethod
  def read_array_double(r, sz):
    return np.array(r.read_double(sz)[:sz])

  def __init__(self, fn=None, records=None, profiles=None):
    self.records   = records
    if self.records is None:
      self.filenames = None
    else:
      self.filenames = self.records.keys()
    self.profiles = profiles

  @classmethod
  def load_files(cls,files=None,verbose=False):
    """
    load BSRT profile files

    Example:
    --------
    Load a single file:
        BSRTprofiles.load_files(files=['fn_profile_1'])
      or
        BSRTprofiles.load_files(files='fn_profile_1')
      if filename is unique.
    Load a list of filenames:
      BSRTprofiles.load_files(files=['fn_profile_1','fn_profile_2'])
    Provide a search string:
      BSRTprofiles.load_files(files='/myprofiledir/*.bindata')

    Parameters:
    -----------
    files : either list of filename(s) or search string
    verbose : verbose option for additional output

    Returns:
    --------
    class: BSRTprofiles instance
    """
    if isinstance(files,list):
      fns = files
    elif isinstance(files,str):
      fns = glob.glob(files)
    else:
      raise ValueError('Give either a list of filenames or a search ' +
                       'string!')
      return
    profiles = {'h': {}, 'v': {}}
    records = {}
    for fn in fns:
      if verbose: print '... loading file %s'%fn
      # open the binary file
      r = bio.BinaryReader(fn)
      # read the file
      records[fn] = [] # list of records in binary file
      offset= 0
      while True:
        try:
          header = BSRTprofiles.read_header(r)
          (time_stamp, num_bunches, num_acq_per_bunch, bunch_list, 
             acquiredImageRectangle) = header
          profsize = BSRTprofiles.get_record_size(num_bunches, 
                        num_acq_per_bunch, acquiredImageRectangle)
          (width,height,prof_h_sz,prof_v_sz,r_size) = profsize
          records[fn].append((offset, time_stamp, num_bunches, 
                              num_acq_per_bunch, bunch_list, width,
                              height, prof_h_sz, prof_v_sz, r_size))
          offset+= r_size
          # profiles
          projPositionSet1 = BSRTprofiles.read_array_double(r, width)
          projPositionSet2 = BSRTprofiles.read_array_double(r, height)
          profilesSet1 = BSRTprofiles.read_array_int(r,
                             width*num_bunches*num_acq_per_bunch)
          profilesSet2 = BSRTprofiles.read_array_int(r,
                             height*num_bunches*num_acq_per_bunch)
          # profiles are stored in the order [bunch_1 acquisition_1,
          # bunch_1 acquisition_2, ..., bunch_1 acquisition_m, ...,
          # bunch_n acquisition_1, ..., bunch_n acquisition_m]
          for i in xrange(num_bunches):
            slotID = bunch_list[i]
            for plane in ['h','v']:
              if slotID not in profiles[plane].keys():
                profiles[plane][slotID]=[]
            for j in xrange(num_acq_per_bunch):
              profiles['h'][slotID].append( (time_stamp, projPositionSet1,
                  np.array(profilesSet1[width*(i*num_bunches+j):
                                       width*(i*num_bunches+(j+1))],
                           dtype= np.float_)) )
              profiles['v'][slotID].append( (time_stamp,projPositionSet2,
                  np.array(profilesSet2[height*(i*num_bunches+j):
                                        height*(i*num_bunches+(j+1))],
                           dtype= np.float_)) )
          # check the file position
          if r.tell() != offset:
            raise ValueError('wrong file position %s != '%(r.tell()) +
                             '%s = offset!'%(offset))
        except IOError:
          break
      # resume file
      r.seek(0)
    # convert to a structured array
    ftype=[('time_stamp',int), ('pos',np.ndarray), ('amp',np.ndarray)]
    for plane in ['h','v']:
      prof = profiles[plane]
      for k in prof.keys():
        prof[k] = np.array(prof[k], dtype=ftype)
    return cls(fn=fn, records=records, profiles=profiles)
  def plot_profile(self, slot = None, time_stamp = None, plane = 'h'):
    """
    Plot all profiles for specific bunch and time. Plot title displays 
    'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    plane : plane of profile, either 'h' or 'v'
    """
    mask = self.profiles[plane][slot]['time_stamp'] == time_stamp
    profs = self.profiles[plane][slot][mask]
    for i in xrange(len(profs)):
      pl.plot(profs[i]['pos'],profs[i]['amp'],label = 'profile %s'%i)
    pl.grid(b=True)
    pl.legend(loc='best')
    ts = ld.dumpdate(t=time_stamp*1.e-9,fmt='%Y-%m-%d %H:%M:%S.SSS',
                     zone=ld.myzones['cern']) # convert ns -> s
    pl.gca().set_title('%s plane, %s'%(plane.upper(),ts))
