#! /usr/bin/env python


#import GaussFit as gfit
try:
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as pl
  from scipy.optimize import curve_fit
except ImportError:
  print('No module found: numpy, matplotlib and scipy modules ' +
        'should be present to run pytimbertools')
import os
import shutil
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import gridspec

import BinaryFileIO as bio
import localdate as ld
import toolbox as tb

class BSRTprofiles(object):
  """
  class to analyze BSRT profiles

  Example:
  --------
  Load profiles from list of files (see BSRTprofiles.load_files for 
  details):
    BSRTprofiles.load_files(files='/myprofiledir/*.bindata')
  Load profiles from list (load_files), discard noisy profiles 
  (clean_data), calculate statistical parameters (stats):
    (BSRTprofiles.load_files(files='/myprofiledir/*.bindata')
      .clean_data().stats()
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

  def __init__(self, records=None, profiles=None,
               profiles_stat=None):
    self.records   = records
    if self.records is None:
      self.filenames = None
    else:
      self.filenames = self.records.keys()
    self.profiles = profiles
    self.profiles_stat = profiles_stat

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
              # check the data
              # h plane
              if len(projPositionSet1) == 0:
                check_data = False
                print("WARNING: len(projPositionSet1) = 0 for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) + 
                      "Data is discarded (both *Set1 and *Set2)!")
              elif (len(profilesSet1[width*(i*num_bunches+j):
                         width*(i*num_bunches+(j+1))]) == 0):
                check_data = False
                print("WARNING: len(profilesSet1[...]) = 0 for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                      " Data is discarded (both *Set1 and *Set2)!")
              elif (len(projPositionSet1)!=
                     len(profilesSet1[width*(i*num_bunches+j):
                           width*(i*num_bunches+(j+1))])):
                print("WARNING: len(projPositionSet1) != " + 
                      "len(profilesSet1[...]) for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                      "Data is discarded (both *Set1 and *Set2)!")
                check_data = False
              # v plane
              if len(projPositionSet2) == 0:
                check_data = False
                print("WARNING: len(projPositionSet2) = 0 for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) + 
                      "Data is discarded (both *Set1 and *Set2)!")
              elif (len(profilesSet2[height*(i*num_bunches+j):
                         height*(i*num_bunches+(j+1))]) == 0):
                check_data = False
                print("WARNING: len(profilesSet2[...]) = 0 for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                      " Data is discarded (both *Set1 and *Set2)!")
              elif (len(projPositionSet2)!=
                     len(profilesSet2[height*(i*num_bunches+j):
                           height*(i*num_bunches+(j+1))])):
                print("WARNING: len(projPositionSet2) != " + 
                      "len(profilesSet2[...]) for " +
                      "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                      "Data is discarded (both *Set1 and *Set2)!")
                check_data = False
              else:
                check_data = True
              if check_data:
                # h values go from -x to +x
                profiles['h'][slotID].append( (time_stamp, projPositionSet1,
                    np.array(profilesSet1[width*(i*num_bunches+j):
                                         width*(i*num_bunches+(j+1))],
                             dtype= np.float_)) )
                # v values go from x to -x
                # -> reverse the order
                if projPositionSet2[0] > projPositionSet2[-1]:
                  profiles['v'][slotID].append( (time_stamp,
                    np.fliplr([projPositionSet2])[0],
                    np.fliplr([np.array(
                      profilesSet2[height*(i*num_bunches+j):
                                   height*(i*num_bunches+(j+1))],
                             dtype= np.float_)])[0]) )
                # v values go from -x to x
                # -> keep the order
                else:
                  profiles['v'][slotID].append( (time_stamp,
                    projPositionSet2,
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
    return cls(records=records, profiles=profiles)
  def clean_data(self,stdamp = 3000):
    """
    removes all profiles considered to be just noise, explicitly
    profiles with:
    1) std(self.profiles[plane][slot]['amp']) < stdamp
    """
    for plane in 'h','v':
      for slot in self.profiles[plane].keys():
        # delete all entries with std(profs['amp']) < stdamp)
        rm_ts_std = []
        for i in xrange(len(self.profiles[plane][slot])):
          std_aux = (self.profiles[plane][slot][i]['amp']).std()
          if std_aux < stdamp:
            rm_ts_std.append(self.profiles[plane][slot][i]['time_stamp'])
        rm_ts = list(set(rm_ts_std))
        for ts in rm_ts:
          print("... removing entry for plane %s, "%(plane) +
                "slot %s, time stamp "%(slot) +
                "%s"%(self.profiles[plane][slot][i]['time_stamp']))
        rm_mask = np.in1d(self.profiles[plane][slot]['time_stamp'],
                          rm_ts,invert=True)
        self.profiles[plane][slot] = self.profiles[plane][slot][rm_mask]
    return self
  def stats(self,verbose=False):
    """
    calculate statistical parameters for the average over all profiles
    for each timestamp.
    """
    if verbose:
      if self.profile_stat is None:
        print('... calculate statistical parameters')
      else:
        print('... delete old data and recalculate statistical ' + 
            'parameters')
    self.profiles_stat={}
    for plane in ['h','v']:
      self.profiles_stat[plane]={}
      for slot in self.profiles[plane].keys():
        self.profiles_stat[plane][slot] = []
        for time_stamp in self.get_timestamps(slot=slot, plane=plane):
          # average profile
          profs_norm_avg = self.get_profile_norm_avg(slot=slot,                  
                             time_stamp=time_stamp, plane=plane)            
          # 1) estimate centroid with three different methods:
          # 1a) Gaussian fit (cent_gauss)
          # 1b) center of gravity sum(x*w) (cent_stat)
          # 1c) median (cent_stat_median)
          # 1d) 50 % of cummulative sum (cent_cumsum)
          # 2) estimate of distribution width with three different 
          #    methods:
          # 2a) Gaussian fit (sigma_gauss)
          # 2b) statistical rms sum(x*w) (sigma_stat)
          # 2c) median absolute deviation = median(|x-median(x)|)
          # 2d) 26 % and 84% of cummulative sum (sigma_cumsum)

          # a) Gaussian fit
          # assume initial values of
          # mean0=0,sigma0=2,a0=1/sqrt(2*sigma0**2*pi)=0.2
          try:
            p,pcov = curve_fit(tb.gauss_fit,profs_norm_avg['pos'],
                               profs_norm_avg['amp'],p0=[0,2])
            # error on p
            psig = [ np.sqrt(pcov[i,i]) for i in range(len(p)) ]
            cent_gauss, sigma_gauss = p[0],p[1]
            cent_gauss_err, sigma_gauss_err = psig[0],psig[1]
          except RuntimeError:
            if verbose:
              print("WARNING: fit failed for plane %s, "%(plane) +
                    "slotID %s, timestamp %s"%(slot,time_stamp))
            cent_gauss, sigma_gauss = 0,0
            cent_gauss_err, sigma_gauss_err = 0,0
            pass
          # b) statistical parameters
          cent_stat = np.average(profs_norm_avg['pos'],
                                 weights=profs_norm_avg['amp'])
          sigma_stat = np.average((profs_norm_avg['pos']-cent_stat)**2,
                                weights=profs_norm_avg['amp'])
          cent_stat_median = np.average(profs_norm_avg['pos'],
                                 weights=profs_norm_avg['amp'])
          sigma_stat_median = np.average((profs_norm_avg['pos']
                                -cent_stat)**2,
                                weights=profs_norm_avg['amp'])
          #print 'append slot %s'%slot
          self.profiles_stat[plane][slot].append((time_stamp,cent_gauss,
            cent_gauss_err,cent_stat,cent_stat_median,sigma_gauss,
            sigma_gauss_err,sigma_stat,sigma_stat_median))
    # convert to a structured array
    ftype=[('time_stamp',int), ('cent_gauss',float), 
      ('cent_gauss_err',float), ('cent_stat',float), 
      ('cent_stat_median',float), ('sigma_gauss',float),
      ('sigma_gauss_err',float), ('sigma_stat',float),
      ('sigma_stat_median',float)]
    for plane in ['h','v']:
      for k in self.profiles_stat[plane].keys():
        self.profiles_stat[plane][k] = np.array(
          self.profiles_stat[plane][k], dtype=ftype)
    return self
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
  def get_profile_norm_avg(self, slot = None, time_stamp = None,
                           plane = 'h'):                                    
    """
    returns averaged normalized profile for slot *slot*, time stamp
    *time_stamp* as unix time [ns] and plane *plane*.
    """
    #select profile for slot and time stamp
    profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                 plane=plane)
    # no profiles found
    if len(profs) == 0:
      raise ValueError('slot or timestamp not found')
    # one profile
    if len(profs) == 1:
      profs_avg = profs[0].copy()
      return profs_avg
    # more than one profile
    # take the average over the profiles
    # 1) check that x-axis are the same
    check_x = 0
    for i in xrange(len(profs)):
      if (np.abs(profs[0]['pos']-profs[1]['pos'])).sum() != 0:
        check_x +=1
    #     check_x = 0 if x-axis of profiles are the same
    #     check_x > 0 if x-axis differ
    if check_x ==0: 
      # 2) take the average. Integral is normalized to 1
      #    -> to normalize avg divide by number of profiles
      profs_avg = profs[0].copy() 
      profs_avg['amp'] = profs['amp'].sum(axis=0)/len(profs)
      return profs_avg
    else:
      return None
  def get_profile_stat(self, slot = None, time_stamp = None, plane = 'h'):
    """
    get profile data for slot *slot*, time stamp *time_stamp* as
    unix time [ns] and plane *plane*
    """
    mask = self.profiles_stat[plane][slot]['time_stamp'] == time_stamp
    return self.profiles_stat[plane][slot][mask]
  def _plot_profile(self, slot = None, time_stamp = None, plane = 'h',
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

    Returns:
    --------
    check_plot : bool, flag if profile plot failed
                 (used for mk_profile_video)
    """
    # flag if profile plot failed (used for mk_profile_video)
    check_plot = True
    # select profile for slot and time stamp
    # normalized profiles
    if norm:
      profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                    plane=plane)
      profs_avg = self.get_profile_norm_avg(slot=slot,
                          time_stamp=time_stamp,plane=plane)
      if self.profiles_stat is not None:
        stat_aux = self.get_profiles_stat(slot=slot,
                     time_stamp=time_stamp,plane=plane)
        centroid_gauss = stat_aux['centroid_gauss']
        sigma_gauss    = stat_aux['sigma_gauss']
    # raw data profile
    else:
      profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                               plane=plane)
    for i in xrange(len(profs)):
      try:
        pl.plot(profs[i]['pos'],profs[i]['amp'],
                label='profile %s'%(i+1))
        if norm:
          # plot Gaussian fit in addition
          pl.plot(profs[i]['pos'],tb.gauss_fit(profs[i]['pos'],
                  centroid_gauss,sigma_gauss),color='gray',linestyle='--')
          pl.ylabel(r'probability (integral normalized to 1) [a.u.]')
        else:
          pl.ylabel('intensity [a.u.]')
        pl.xlabel('position [mm]')
        pl.grid(b=True)
        pl.legend(loc='best')
        # convert ns -> s before creating date string
        ts = ld.dumpdate(t=time_stamp*1.e-9,
                 fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
        pl.gca().set_title('bunch %s, %s plane, %s'%(slot,
                             plane.upper(),ts))
      except ValueError:
        if verbose:
          print('ERROR: plotting of bunch %s, '%(slot) +
          'profile %s, time stamp %s failed.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
           len(profs[i]['amp'])))
        check_plot = False
        pass
    if norm and check_plot:
      pl.plot(profs_avg['pos'],profs_avg['amp'],
              label = 'average profile',color='k',linestyle='--')

    return check_plot
  def plot_profile(self, slot = None, time_stamp = None, plane = 'h',
                   verbose = False):
    """
    Plot raw data profiles for specific bunch and time. Plot title displays 
    'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output
    """
    self._plot_profile(slot=slot,time_stamp=time_stamp,plane=plane,
                       norm=False,verbose=verbose)
  def plot_profile_norm(self, slot = None, time_stamp = None,
                        plane = 'h', verbose = False):
    """
    Plot all profiles for specific bunch and time. Profiles are
    normalized to represent a probability distribution, explicitly:
      norm_data = raw_data/(int(raw_data))
    so that:
      int(norm_data) = 1
    Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output
    """
    self._plot_profile(slot=slot,time_stamp=time_stamp,plane=plane,
                       norm=True,verbose=verbose)
  def plot_cumsum(self, slot = None, time_stamp = None, plane = 'h',
                  verbose = False):
    """
    Plot cumulative distribution function of normalized profiles for 
    a specific bunch and time. Profiles are normalized to represent
    a probability distribution, explicitly:
      norm_data = raw_data/(int(raw_data)) 
    so that:
      int(norm_data) = 1
    Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output

    Returns:
    --------
    check_plot : bool, flag if profile plot failed
                 (used for mk_profile_video)
    """
    #select profile for slot and time stamp
    profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                  plane=plane)
    profs_avg = self.get_profile_norm_avg(slot=slot,
                       time_stamp=time_stamp,plane=plane)
    # flag if profile plot failed
    check_plot = True
    # individual profiles
    for i in xrange(len(profs)):
      try:
        pl.plot(profs[i]['pos'],profs[i]['amp'].cumsum(),
                label='profile %s'%(i+1))
      except ValueError:
        if verbose:
          print('ERROR: plotting of bunch %s, '%(slot) +
          'profile %s, time stamp %s failed.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
           len(profs[i]['amp'])))
        check_plot = False
        pass
    # average profile
    if check_plot:
      pl.plot(profs_avg['pos'],profs_avg['amp'].cumsum(),
            label = 'average profile',color='k',linestyle='-')
      pl.xlabel('position [mm]')
      pl.ylabel(r'cumulative distribution functions [a.u.]')
      pl.grid(b=True)
      pl.legend(loc='best')
      # convert ns -> s before creating date string
      ts = ld.dumpdate(t=time_stamp*1.e-9,
                       fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
      pl.gca().set_title('bunch %s, %s plane, %s'%(slot,
                          plane.upper(),ts))
    return check_plot
  def plot_residual(self, slot = None, time_stamp = None, 
                    time_stamp_ref = None, plane = 'h',
                    verbose = False):
    """
    Plot residual of normalized profiles for a specific bunch and time.
    Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    time_stamp_ref : reference time stamp in unix time [ns], if None
                     use first time stamp
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output
    """
    self._plot_residual_ratio(flag='residual', flagprof='all', slot=slot,
            time_stamp=time_stamp, time_stamp_ref=time_stamp_ref,
            plane=plane, verbose=verbose)
  def plot_ratio(self, slot = None, time_stamp = None, 
                 time_stamp_ref = None, plane = 'h',
                 verbose = False):
    """
    Plot ratio of normalized profiles for a specific bunch and time.
    Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    time_stamp_ref : reference time stamp in unix time [ns], if None    
                     use first time stamp                               
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output
    """
    self._plot_residual_ratio(flag='ratio', flagprof = 'all',slot=slot,
            time_stamp=time_stamp, time_stamp_ref=time_stamp_ref,
            plane=plane, verbose=verbose)
  def _plot_residual_ratio(self, flag, flagprof, slot = None,
                           time_stamp = None, time_stamp_ref = None, 
                           plane = 'h', verbose = False):
    """
    Plot residual or ratio of normalized profiles for a specific bunch and time.
    Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    flag : flag = 'residual' plot the residual
           flag = 'ratio' plot the ratio
    flagprof: flagprof = 'all' plot all profiles + average
              flagprof = 'avg' plot only average profile
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    time_stamp_ref : reference time stamp in unix time [ns], if None
                     use first time stamp
    plane : plane of profile, either 'h' or 'v'
    verbose : verbose option for additional output

    Returns:
    --------
    check_plot : bool, flag if profile plot failed 
                 (used for mk_profile_video)
    """
    # if time_stamp_ref = None use first timestamp
    if time_stamp_ref is None:
      time_stamp_ref = self.get_timestamps(slot=slot,plane=plane)[0]
    # select profile for slot and time stamp
    # individual profiles
    if flagprof == 'all':
      profs         = self.get_profile_norm(slot=slot,
                             time_stamp=time_stamp, plane=plane)
      profs_ref     = self.get_profile_norm(slot=slot,
                             time_stamp=time_stamp_ref, plane=plane)
    # average profiles
    profs_avg     = self.get_profile_norm_avg(slot=slot,
                           time_stamp=time_stamp, plane=plane)
    profs_ref_avg = self.get_profile_norm_avg(slot=slot,
                           time_stamp=time_stamp_ref, plane=plane)
    # take only values for which x-range of profile coincides
    xval = list(set(profs_avg['pos']) and set(profs_ref_avg['pos']))
    mask = np.array([ pos in xval for pos in profs_avg['pos'] ],
                    dtype=bool)
    mask_ref = np.array([ pos in xval for pos in profs_ref_avg['pos'] ],
                        dtype=bool)
    # individual profiles
    # check_plot = flag if profile plot failed
    check_plot = True
    if flagprof == 'all':
      for i in xrange(len(profs)):
        try:
          if flag == 'residual':
            pl.plot(profs[i]['pos'][mask],
                    profs[i]['amp'][mask]-profs_ref[i]['amp'][mask_ref],
                    label='profile %s'%(i+1))
          if flag == 'ratio':
            pl.plot(profs[i]['pos'][mask],
                    profs[i]['amp'][mask]/profs_ref[i]['amp'][mask_ref],
                    label='profile %s'%(i+1))
        except ValueError:
          if verbose:
            print('ERROR: plotting of bunch %s, '%(slot) +
            'profile %s, time stamp %s failed.'%(i,time_stamp) +
            ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
             len(profs[i]['amp'])))
          check_plot = False
          pass
    #  average profile
    if flag == 'residual':
      try:
        pl.plot(profs_avg['pos'][mask],
                profs_avg['amp'][mask]-profs_ref_avg['amp'][mask_ref],
                label = 'average profile',color='k',linestyle='-')
      except (ValueError,IndexError):
        check_plot = False
        pass
    if flag == 'ratio':
      try:
        pl.plot(profs_avg['pos'][mask],
                profs_avg['amp'][mask]/profs_ref_avg['amp'][mask_ref],
                label = 'average profile',color='k',linestyle='-')
      except (ValueError,IndexError):
        check_plot = False
        pass
    if check_plot:
      pl.xlabel('position [mm]')
      if flag == 'residual':
        pl.ylabel(r'residual $A-A_{\mathrm{ref}}$ [a.u.]')
      if flag == 'ratio':
        pl.ylabel(r'ratio $A/A_{\mathrm{ref}}$ [a.u.]')
      pl.grid(b=True)
      pl.legend(loc='best')
      # convert ns -> s before creating date string
      ts = ld.dumpdate(t=time_stamp*1.e-9,
               fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
      pl.gca().set_title('bunch %s, %s plane, %s'%(slot,
                               plane.upper(),ts))
    return check_plot
  def plot_all(self,slot = None, time_stamp = None,
               time_stamp_ref = None, plane = 'h', norm = True, 
               verbose = False):
    """
    plot normalized or raw data profiles, cumulative distribution 
    function, residual and ratio in respect to reference distribution

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    time_stamp_ref : reference time stamp in unix time [ns], if None
                     use first time stamp
    plane : plane of profile, either 'h' or 'v'
    norm : if norm = False raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    verbose : verbose option for additional output
    
    Returns:
    --------
    flaux : bool, flag to check if profile plots have failed
    """
    pl.clf()
    fig = pl.gcf()
    nsub = 4 # number of subplots
    for i in xrange(nsub):
      fig.add_subplot(2,2,i+1)
    ts = ld.dumpdate(t=time_stamp*1.e-9,
             fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
    pl.suptitle('bunch %s, %s plane, %s'%(slot,
                             plane.upper(),ts))
    # 1) profile plot
    pl.subplot(223)
    # flaux = flag for checking if profile plots have failed
    flaux = self._plot_profile(slot=slot,time_stamp=time_stamp,
                              plane=plane,norm=norm,verbose=verbose)
    pl.gca().set_yscale('log')
    if norm:
      pl.gca().set_ylabel('probability [a.u.]')
      pl.gca().set_ylim(1.e-5,0.32)
    # 2) cumulative sum
    pl.subplot(224)
    self.plot_cumsum(slot=slot,time_stamp=time_stamp,plane=plane,
                     verbose=verbose)
    pl.gca().set_ylabel('cumulative dist. [a.u.]')
    pl.gca().set_ylim(-1,25)
#     3) residual, only average profile
    pl.subplot(221)
    self._plot_residual_ratio(flag='residual', flagprof='avg', 
           slot=slot, time_stamp=time_stamp,
           time_stamp_ref=time_stamp_ref, plane=plane, verbose=verbose)
    pl.gca().set_ylim(-0.05,0.05)
#     4) ratio
    pl.subplot(222)
    self._plot_residual_ratio(flag='ratio', flagprof='avg', 
           slot=slot, time_stamp=time_stamp,
           time_stamp_ref=time_stamp_ref, plane=plane, verbose=verbose)
    pl.gca().set_ylim(-1,6)
    # remove subplot titles, shring legend size and put it on top of
    # the subplot
    for i in xrange(4):
      pl.subplot(2,2,i+1)
      pl.gca().set_title('')
      pl.gca().legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                    ncol=2, mode="expand", borderaxespad=0.,
                    fontsize=10)
      pl.gca().set_xlim(-8,8)
    fig.subplots_adjust(top=0.5)
    fig.tight_layout()
    return flaux
  def mk_profile_video(self, slot = None, time_stamp_ref=None,plt_dir='BSRTprofile_gifs',
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
    check_plot = {}
    for plane in ['h','v']:
      # set slot and plane, initialize variables
      check_plot[plane] = {}
      # 1) all bunches
      if slot is None:
        slots = self.profiles[plane].keys()
      # 2) if single bunch 
      elif not hasattr(slot,"__iter__"):
        slots = [slot]
      # generate the figure and subplot
      pl.figure(plane)
      for slot in slots:
        if os.path.exists(tmpdir) == False:
          os.makedirs(tmpdir)
        check_plot[plane][slot] = []
        if verbose:
          print('... generating plots for bunch %s'%slot)
        time_stamps = self.get_timestamps(slot=slot,plane=plane)
        pngcount = 0
        if verbose: 
          print( '... total number of profiles %s'%(len(time_stamps)))
        # generate png of profiles for each timestamps
        for time_stamp in time_stamps:
          pl.clf()
          flaux = self.plot_all(slot=slot,time_stamp=time_stamp,
                                time_stamp_ref=time_stamp_ref,
                                plane=plane,norm=norm)
          # if plot failed, flaux = False -> append t check_plot
          if flaux is False: check_plot[plane][slot].append(time_stamp)
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
        # delete png files already
        if (export is False) and (os.path.exists(tmpdir) is True):
          shutil.rmtree(tmpdir)
        pl.figure(plane)
        pl.close()
    # print a warning for all failed plots
    for plane in check_plot.keys():
      for slot in check_plot[plane].keys():
         if len(check_plot[plane][slot])>0:
            print('WARNING: plotting of profiles for plane %s'%(plane) +
            ' and slot %s failed for timestamps:'%(slot))
            ts = tuple(set(check_plot[plane][slot]))
            lts = len(ts)
            print(('%s, '*lts)%ts)
    # delete temporary directory
    if (export is False) and (os.path.exists(tmpdir) is True):
      shutil.rmtree(tmpdir)
