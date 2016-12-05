#! /usr/bin/env python


#import GaussFit as gfit
try:
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as pl
except ImportError:
  print('No module found: numpy, matplotlib and scipy modules ' +
        'should be present to run pytimbertools')
import os
import shutil
import glob
import BinaryFileIO as bio
import localdate as ld
from statsmodels.nonparametric.smoothers_lowess import lowess

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
  Methods:
  --------
  plot_profile : plot profile for specific bunch and timestamp    
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
  def smooth_profile_stefania(self, slot = None, time_stamp = None, plane = 'h'):
    #select profile for slot and time stamp                             
    profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                             plane=plane)
    profs_smooth = profs.copy() # new array with smoothened profile
    for i in xrange(len(profs)):
      x = profs[i]['pos']
      y = profs[i]['amp']
      # smooth profile with lowess method, values are already sorted
      xs,ys = lowess(endog=y,exog=x,frac=0.1,it=3,delta=0,
                     is_sorted=True,missing='drop',
                     return_sorted=True).T 
      # find x where y is max in order to define left and right side 
      # of each distribution
  def get_timestamps(self, slot = None, plane = 'h'):
    """
    get all time stamps in unix time [ns] for slot *slot* and
    plane *plane*
    """
    return np.sort(list(set(self.profiles[plane][slot]['time_stamp'])))
  def get_profile(self, slot = None, time_stamp = None, plane = 'h'):
    """
    get profile data for slot *slot*, time stamp *time_stamp* as
    unix time [ns] and plane *plane*
    """
    mask = self.profiles[plane][slot]['time_stamp'] == time_stamp
    return self.profiles[plane][slot][mask]
  def get_profile_norm(self, slot = None, time_stamp = None, 
                       plane = 'h'):
    """
    get normalized profile data for slot *slot*, time stamp
    *time_stamp* as unix time [ns] and plane *plane*. Profiles are 
    normalized to represent a probability distribution, explicitly:
      norm_data = raw_data/(int(raw_data))
    so that:
      int(norm_data) = 1
    
    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                             plane=plane) 
    for i in xrange(len(profs)):
      # assume equal spacing
      try:
        dx = profs[i]['pos'][1]-profs[i]['pos'][0]
        prof_int = (dx*profs[i]['amp']).sum()
        profs[i]['amp'] = profs[i]['amp']/prof_int
      except ValueError:
        if verbose:
          print('ERROR: no data found for bunch %s, '%(slot) +
          'profile %s, time stamp %s.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
           len(profs[i]['amp'])))
          pass
    return profs 
  def plot_profile(self, slot = None, time_stamp = None, plane = 'h',
                   norm = True, verbose = False):
    """
    Plot all profiles for specific bunch and time. Plot title displays 
    'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    plane : plane of profile, either 'h' or 'v'
    norm : if norm = false raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    verbose : verbose option for additional output
    """
    # list for failed timestamps (used for mk_profile_video)
    profs_failed = False
    #select profile for slot and time stamp
    if norm:
      profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                    plane=plane)
    else:
      profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                               plane=plane)
    for i in xrange(len(profs)):
      try:
        pl.plot(profs[i]['pos'],profs[i]['amp'],label = 'profile %s'%i)
      except ValueError:
        if verbose:
          print('ERROR: plotting of bunch %s, '%(slot) +
          'profile %s, time stamp %s failed.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs_ts[i]['pos']),
           len(profs_ts[i]['amp'])))
          profs_failed = True
          pass
    pl.grid(b=True)
    pl.legend(loc='best')
    # convert ns -> s before creating date string
    ts = ld.dumpdate(t=time_stamp*1.e-9,
             fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
    pl.gca().set_title('bunch %s, %s plane, %s'%(slot,
                             plane.upper(),ts))
    return profs_failed
  def mk_profile_video(self, slot = None, plt_dir='BSRTprofile_gifs',
                       delay=20, norm = True, export=False, 
                       verbose=False):
    """
    Generates a video of the profiles of bunch with *slot*

    Parameters:
    -----------
    slot : slot or list of slots of bunches. If slot = None all bunches
           are selected. In case of several bunch, on video is 
           generated per bunch
    plt_dir : directory to save videos
    delay : option for convert to define delay between pictures
    norm : if norm = false raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    export : If True do not delete png files
    verbose : verbose option for additional output
    """
    tmpdir = os.path.join(plt_dir,'tmp_bsrtprofiles')
    # dictionary to store failed profiles
    profs_failed = {}
    for plane in ['h','v']:
      # set slot and plane, initialize variables
      profs_failed[plane] = {}
      pl.figure(plane,figsize=(8,8))
      # all bunches
      if slot is None:
        slots = self.profiles[plane].keys()
      # if single bunch 
      elif not hasattr(slot,"__iter__"):
        slots = [slot]
      for slot in slots:
        if os.path.exists(tmpdir) == False:
          os.makedirs(tmpdir)
        profs_failed[plane][slot] = []
        if verbose:
          print('... generating plots for bunch %s'%slot)
        time_stamps = self.get_timestamps(slot=slot,plane=plane)
        pngcount = 0
        if verbose: 
          print( '... total number of profiles %s'%(len(time_stamps)))
        # generate png of profiles for each timestamps
        for time_stamp in time_stamps:
          pl.clf()
          # 1) profile plot
          flaux = self.plot_profile(slot=slot,time_stamp=time_stamp,
                                    plane=plane)
          # if plot failed, flaux = True -> append t profs_failed
          if flaux: profs_failed[plane][slot].append(time_stamp)
          fnpl = os.path.join(tmpdir,'bunch_%s_plane_%s_%05d.png'%(slot,
                              plane,pngcount))
          if verbose: print '... save png %s'%(fnpl)
          pl.savefig(fnpl)
          pngcount += 1
        # create video from pngs
        if verbose: print '... creating .gif file with convert'
        cmd="convert -delay %s %s %s"%(delay,
             os.path.join(tmpdir,'bunch_%s_plane_%s_*.png'%(slot,plane)),
             os.path.join(plt_dir,'bunch_%s_plane_%s.gif'%(slot,plane)))
        os.system(cmd)
        # delte png files already
        if (export is False) and (os.path.exists(tmpdir) is True):
          shutil.rmtree(tmpdir)
    # print a warning for all failed plots
    for plane in profs_failed.keys():
      for slot in profs_failed[plane].keys():
         if len(profs_failed[plane][slot])>0:
            print('WARNING: plotting of profiles for plane %s'%(plane) +
            'and slot %s failed for timestamps:'%(slot))
            ts = tuple(set(profs_failed[plane][slot]))
            lts = len(ts)
            print(('%s, '*lts)%ts)
    # delete temporary directory
    if (export is False) and (os.path.exists(tmpdir) is True):
      shutil.rmtree(tmpdir)
