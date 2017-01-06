#! /usr/bin/env python


try:
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as pl
  from scipy.optimize import curve_fit
  from scipy.special import erf
  from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
  print('No module found: numpy, matplotlib, scipy modules and ' +
        'statsmodels should be present to run pytimbertools')
import os
import shutil
import glob
# from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import gridspec

import BinaryFileIO as bio
import localdate as ld
import toolbox as tb

# plot colors
# individual profiles
profile_colors = {'cyan':'#00FFFF','LightCyan':'#E0FFFF',
  'PaleTurquoise':'#AFEEEE','Aquamarine':'#7FFFD4',
  'Turquoise':'#40E0D0','CadetBlue':'#5F9EA0','SteelBlue':'#4682B4',
  'LightSteelBlue':'#B0C4DE','PowderBlue':'#B0E0E6',
  'SkyBlue':'#87CEEB','DeepSkyBlue':'#00BFFF','DodgerBlue':'#1E90FF',
  'CornflowerBlue':'#6495ED','MediumBlue':'#0000CD',
  'DarkBlue':'#00008B'}
# fits
fit_colors = {'Red':'#FF0000','DarkRed':'#8B0000'}

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
  profiles_norm : same format as profiles. Contains all profiles are
      normalized to represent a probability distribution with an
      integral of 1.
  profiles_norm_avg : same fromat as profile. Contains for each plane,
      slot and timestamp the average of all normalized profiles for
      this time stamp.
  profiles_stat: statistical parameters calculated for average profile
      for each timestamp

  Methods:
  --------
  plot_profile : plot profile for specific slot and timestamp    
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

  def __init__(self, records=None, profiles=None, profiles_norm=None,
               profiles_norm_avg=None, profiles_stat=None):
    self.records   = records
    if self.records is None:
      self.filenames = None
    else:
      self.filenames = self.records.keys()
    self.profiles = profiles
    self.profiles_norm = profiles_norm
    self.profiles_norm_avg = profiles_norm_avg
    self.profiles_stat = profiles_stat
    if self.profiles_stat is None:
      self.profiles_stat_var = None
    else:
      # take first plane and slot in order to get profile_stat named fields
      plane_aux = self.profiles_stat.keys()[0]
      slot_aux = self.profiles_stat[plane_aux].keys()[0] 
      self.profiles_stat_var = self.profiles_stat[plane_aux][slot_aux].dtype.names
  @classmethod
  def load_files(cls,files=None,verbose=True):
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
                if verbose:
                  print("WARNING: len(projPositionSet1) = 0 for " +
                        "slotID %s, timestamp %s! "%(slotID,time_stamp) + 
                        "Data is discarded (both *Set1 and *Set2)!")
              elif (len(profilesSet1[width*(i*num_bunches+j):
                         width*(i*num_bunches+(j+1))]) == 0):
                check_data = False
                if verbose:
                  print("WARNING: len(profilesSet1[...]) = 0 for " +
                        "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                        " Data is discarded (both *Set1 and *Set2)!")
              elif (len(projPositionSet1)!=
                     len(profilesSet1[width*(i*num_bunches+j):
                           width*(i*num_bunches+(j+1))])):
                if verbose:
                  print("WARNING: len(projPositionSet1) != " + 
                        "len(profilesSet1[...]) for " +
                        "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                        "Data is discarded (both *Set1 and *Set2)!")
                check_data = False
              # v plane
              if len(projPositionSet2) == 0:
                check_data = False
                if verbose:
                  print("WARNING: len(projPositionSet2) = 0 for " +
                        "slotID %s, timestamp %s! "%(slotID,time_stamp) + 
                        "Data is discarded (both *Set1 and *Set2)!")
              elif (len(profilesSet2[height*(i*num_bunches+j):
                         height*(i*num_bunches+(j+1))]) == 0):
                check_data = False
                if verbose:
                  print("WARNING: len(profilesSet2[...]) = 0 for " +
                        "slotID %s, timestamp %s! "%(slotID,time_stamp) +
                        " Data is discarded (both *Set1 and *Set2)!")
              elif (len(projPositionSet2)!=
                     len(profilesSet2[height*(i*num_bunches+j):
                           height*(i*num_bunches+(j+1))])):
                if verbose:
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
    # convert to a structured array and sort by time stamp
    ftype=[('time_stamp',int), ('pos',np.ndarray), ('amp',np.ndarray)]
    for plane in ['h','v']:
      for k in profiles[plane].keys():
        profiles[plane][k] = np.array(profiles[plane][k], dtype=ftype)
    return cls(records=records, profiles=profiles)
  def clean_data(self,stdamp = 3000,verbose = True):
    """
    removes all profiles considered to be just noise, explicitly
    profiles with std(self.profiles[plane][slot]['amp']) < stdamp
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
          if verbose:
            print("... removing entry for plane %s, "%(plane) +
                  "slot %s, time stamp "%(slot) +
                  "%s"%(self.profiles[plane][slot][i]['time_stamp']))
        rm_mask = np.in1d(self.profiles[plane][slot]['time_stamp'],
                          rm_ts,invert=True)
        self.profiles[plane][slot] = self.profiles[plane][slot][rm_mask]
    return self
  def _norm_profiles(self,verbose = True):
    """
    Generate normalized profiles for each plane, slot and timestamp. 
    Profiles are normalized to represent a probability distribution, 
    explicitly:
      norm_data = raw_data/(int(raw_data))
    so that:
      int(norm_data) = 1
    The average is taken over all profiles for each plane, slot and 
    time stamp.

    Parameters:
    -----------
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm (normalized profiles) and 

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    if verbose:
      if self.profiles_norm is not None:
        print('... delete self.profiles_norm')
    self.profiles_norm={}
    if verbose:
      print('... normalize profiles. Normalized profiles are saved' +
            ' in self.profiles_norm')
    for plane in 'h','v':
      self.profiles_norm[plane]={}
      for slot in self.profiles[plane].keys():
        self.profiles_norm[plane][slot] = []
        for idx in xrange(len(self.profiles[plane][slot])):
          try:
            prof = self.profiles[plane][slot][idx].copy()
            # assume equal spacing
            dx = prof['pos'][1]-prof['pos'][0]
            prof_int = (dx*prof['amp']).sum()
            self.profiles_norm[plane][slot].append(
                (prof['time_stamp'],np.array(prof['pos']),
                 np.array(prof['amp']/prof_int)) )
          except ValueError:
            pass
    # convert to a structured array and sort by time stamp
    ftype=[('time_stamp',int), ('pos',np.ndarray), ('amp',np.ndarray)]
    for plane in ['h','v']:
      for k in self.profiles_norm[plane].keys():
        self.profiles_norm[plane][k] = np.array(
            self.profiles_norm[plane][k], dtype=ftype)
    return self
  def _norm_avg_profiles(self,verbose = True):
    """
    Generate average of normalized profiles for each
    plane, slot and timestamp. The average is taken over all profiles 
    for each plane, slot and 
    time stamp.

    Parameters:
    -----------
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm_avg (average normalized profile)

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    # generate normalized profiles if missing
    if self.profiles_norm is None:
      self = self._norm_profiles(verbose=verbose)
    if verbose:
      if self.profiles_norm_avg is not None:
        print('... delete self.profiles_norm_avg')
    self.profiles_norm_avg={}
    if verbose:
      print('... average normalized profiles. Averaged normalized ' +
            'profiles are saved in self.profiles_norm_avg')
    for plane in 'h','v':
      self.profiles_norm_avg[plane]={}
      for slot in self.profiles_norm[plane].keys():
        self.profiles_norm_avg[plane][slot]=[]
        ts = self.get_timestamps(slot=slot,plane=plane)
        for time_stamp in ts:
          profs = self.get_profile_norm(plane=plane,slot=slot,
                                         time_stamp=time_stamp)
          # no profiles found -> skip
          if len(profs) == 0:
            if verbose:
              print('slot %s or timestamp %s not found'%(slot,time_stamp))
            continue
          # one profile
          elif len(profs) == 1:
            profs_avg = profs[0].copy()
          # more than one profile
          elif len(profs) > 1:
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
          self.profiles_norm_avg[plane][slot].append((int(time_stamp),
              profs_avg['pos'],profs_avg['amp']))
    # convert to a structured array and sort by time stamp
    ftype=[('time_stamp',int), ('pos',np.ndarray), ('amp',np.ndarray)]
    for plane in ['h','v']:
      for k in self.profiles_norm_avg[plane].keys():
        self.profiles_norm_avg[plane][k] = np.array(
            self.profiles_norm_avg[plane][k], dtype=ftype)
    return self
  def norm(self,verbose = True):
    """
    Generate normalized and average of normalized profiles for each
    plane, slot and timestamp. Profiles are normalized to represent a 
    probability distribution, explicitly:
      norm_data = raw_data/(int(raw_data))
    so that:
      int(norm_data) = 1
    The average is taken over all profiles for each plane, slot and 
    time stamp.

    Parameters:
    -----------
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm (normalized profiles) and 
        self.profiles_norm_avg (average normalized profile)

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    # self.profiles_norm
    self._norm_profiles(verbose=verbose)
    # self.profiles_norm_avg
    self._norm_avg_profiles(verbose=verbose)
    return self
  def remove_background(self,verbose=False):
    """
    Removes the background from all normalized profiles.
    Estimate background by:
      1) averaging the amplitude over the first and last 10 bins for
         each bin
      2) averaging over all profiles for each slot as background seems
         to stay rather constant

    Parameters:
    -----------
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object where the background is removed
        from all normalized profiles.
    """
    if (self.profiles_norm is None) or (self.profiles_norm_avg is None):
      self = self.norm()
    if verbose:
      print('... remove background from normalized profiles')
    for plane in 'h','v':
      for slot in self.profiles_norm[plane].keys():
        # estimate background
        bgnavg = 10
        ts = self.get_timestamps(slot=slot,plane=plane)
        bg_avg_left = np.mean([ self.get_profile_norm_avg(slot=slot,
                          time_stamp=t,plane=plane)['amp'][:bgnavg] 
                          for t in ts ],axis=None)
        bg_avg_right = np.mean([ self.get_profile_norm_avg(slot=slot,
                          time_stamp=t,plane=plane)['amp'][-bgnavg:] 
                          for t in ts ])
        bg_avg = (bg_avg_left+bg_avg_right)/2
        # remove it from the normalized profiles
        self.profiles_norm[plane][slot]['amp'] = (
            self.profiles_norm[plane][slot]['amp']-bg_avg )
        self.profiles_norm_avg[plane][slot]['amp'] = (
            self.profiles_norm_avg[plane][slot]['amp']-bg_avg )
        # renormalize profiles to integral(profile)=1
        for idx in xrange(len(self.profiles_norm[plane][slot])):
          try:
            prof = self.profiles_norm[plane][slot][idx]
            # assume equal spacing
            dx = prof['pos'][1]-prof['pos'][0]
            prof_int = (dx*prof['amp']).sum()
            self.profiles_norm[plane][slot][idx]['amp'] = (prof['amp']/
                                                           prof_int)
          except ValueError:
            pass
        for idx in xrange(len(self.profiles_norm_avg[plane][slot])):
          try:
            prof = self.profiles_norm_avg[plane][slot][idx]
            # assume equal spacing
            dx = prof['pos'][1]-prof['pos'][0]
            prof_int = (dx*prof['amp']).sum()
            self.profiles_norm_avg[plane][slot][idx]['amp'] = (prof['amp']/
                                                           prof_int)
          except ValueError:
            pass

    return self
  def _set_slots(self,plane,slots):
    """
    set slot numbers, handles the case of slots = None and only one 
    slot.
    """
    # 1) all slots
    if slots is None:
      slots = self.profiles[plane].keys()
    # 2) if single slot 
    elif not hasattr(slots,"__iter__"):
      slots = [slot]
    return np.sort(slots,axis=None)
  def stats(self,slots=None,force=False,verbose=False):
    """
    calculate statistical parameters for the average over all profiles
    for each timestamp.

    Parameters:
    -----------
    slot: slot numbers (single value or list)
    force: force recalculation
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_stat
    """
    # constants for cumulative distribution later
    # - mean = 50 % of cumulative distribution
    # - in n sigma of the distribution erf(n/sqrt(2)) per cent are 
    #   contained:
    #     1 sigma = erf(1/sqrt(2)) (68 %)
    #             = 1 - erf(1/sqrt(2)) (32%)
    cumsum_1sigma = erf(1/np.sqrt(2))
    # generate normalized profile if missing
    if self.profiles_norm_avg is None:
      self.norm(verbose=verbose)
    # initialize if data is empty
    if self.profiles_stat is None:
      self.profiles_stat={}
      self.profiles_stat['h']={}
      self.profiles_stat['v']={}
    if verbose:
      if force is False:
        print('... calculate statistical parameters.')
      elif force is True:
        print('... delete old data and recalculate statistical ' + 
            'parameters')
    for plane in ['h','v']:
      if verbose:
        print('... start plane %s'%plane.upper())
      for slot in self._set_slots(plane=plane,slots=slots):
        if verbose:
          print('... start slot %s'%slot)
        # 1) data is already calculated + force = False-> go to next slot
        if (slot in self.profiles_stat[plane].keys()) and (force is False):
          continue
        # 2) calculate/recalculate statistical parameters
        self.profiles_stat[plane][slot] = []
        for time_stamp in self.get_timestamps(slot=slot, plane=plane):
          # average profile
          profs_norm_avg = self.get_profile_norm_avg(slot=slot,                  
                             time_stamp=time_stamp, plane=plane)            
          # 1) estimate centroid with three different methods:
          # 1a) Gaussian fit (cent_gauss)
          # 1b) Gaussian fit (cent_gauss)
          # 1c) center of gravity sum(x*w) (cent_stat)
          # 1d) median (cent_median)
          # 1e) 50 % of cummulative sum (cent_cumsum)
          # 1f) peak of distribution
          # 2) estimate of distribution width with three different 
          #    methods:
          # 2a) Gaussian fit (sigma_gauss)
          # 2b) Gaussian fit (sigma_gauss)
          # 2c) statistical rms sum(x*w) (sigma_stat)
          # 2d) median absolute deviation = median(|x-median(x)|)
          # 2e) 26 % and 84% of cummulative sum (sigma_cumsum)

          # a) Gaussian fit
          # assume initial values of
          # c=0,a=1,mu=0,sig=2
          # fit function =
          #   c+a*1/sqrt(2*sig**2*pi)*exp(((x-mu)/sig)**2/2)
          # c is also an estimate for the background
          # a compensate for c0 so that the integral over the
          # the distribution is equal to 1. If c is small, a should be
          # close to 1
          try:
            p,pcov = curve_fit(tb.gauss_pdf,profs_norm_avg['pos'],
                               profs_norm_avg['amp'],p0=[0,1,0,2],
                               bounds=([0,0,-np.inf,0],
                                       [1,np.inf,np.inf,np.inf]))
            # error on p
            psig = [ np.sqrt(pcov[i,i]) for i in range(len(p)) ]
            c_gauss, a_gauss = p[0], p[1]
            cent_gauss, sigma_gauss = p[2],p[3]
            c_gauss_err, a_gauss_err = psig[0], psig[1]
            cent_gauss_err, sigma_gauss_err = psig[2],psig[3]
          except RuntimeError:
            if verbose:
              print("WARNING: Gaussian fit failed for " +
                    "plane %s, slotID %s and "%(plane,slot) +
                    "timestamp %s"%(time_stamp))
            c_gauss, a_gauss, cent_gauss, sigma_gauss = 0,0,0,0
            c_gauss_err, a_gauss_err = 0,0
            cent_gauss_err, sigma_gauss_err = 0,0
            pass
          # b) qGaussian fit, note that 1 < q < 3 -> constraint fit
          #    parameters
          # fit function =
          #   c+a*sqrt(beta)/cq*eq(-beta*(x-mu)**2)
          # cq is also an estimate for the background
          # a compensate for c0 so that the integral over the
          # the distribution is equal to 1. If c0 is small, a should be
          # close to 1
          try:
            p,pcov = curve_fit(tb.qgauss_pdf,profs_norm_avg['pos'],
                               profs_norm_avg['amp'],
                               bounds=([0,0,1,-np.inf,0],
                                 [1,np.inf,3,np.inf,np.inf]))
            # error on p
            psig = [ np.sqrt(pcov[i,i]) for i in range(len(p)) ]
            pcov_beta_q = pcov[2,4]
            c_qgauss, a_qgauss = p[0], p[1]
            q_qgauss = p[2]
            cent_qgauss, beta_qgauss = p[3], p[4]
            c_qgauss_err, a_qgauss_err = psig[0], psig[1]
            q_qgauss_err = psig[2]
            cent_qgauss_err, beta_qgauss_err = psig[3],psig[4]
            # sigma**2 = 1/(beta*(5-3q)) for q < 5/3
            # sigma**2 = infty for 5/3 <= q < 2
            # undefined for 2<= q < 3
            if (q_qgauss > 1) & (q_qgauss < 5/3.):
              sigma_qgauss = np.sqrt( 1/(beta_qgauss*(5-3*q_qgauss)) )
              # sigma_f**2 = 
              #   |df/da|**2*sigma_a**2+|df/db|**2*sigma_b**2
              #   + 2*(df/da)*(df/db)*sigma_ab
              # pcov_beta_q = covariance entry beta_q 
              var_qgauss_err = ( (1/(4*beta_qgauss**3*(5-3*q_qgauss))
                   *beta_qgauss_err**2) +
                (9/(4*beta_qgauss*(5-3*q_qgauss)**3)
                   *q_qgauss_err**2) +
                (3/(4*beta_qgauss**2*(5-3*q_qgauss)**2))*pcov_beta_q )
              sigma_qgauss_err = np.sqrt(var_qgauss_err)
            elif (q_qgauss >= 5/3.) & (q_qgauss < 2):
              sigma_qgauss = np.inf
              sigma_qgauss_err = np.inf
            else:
              sigma_qgauss = 0
              sigma_qgauss_err = 0
          except RuntimeError,RuntimeWarning:
            if verbose:
              print("WARNING: q-Gaussian fit failed for " +
                    "plane %s, slotID %s and "%(plane,slot) +
                    "timestamp %s"%(time_stamp))
            c_qgauss, a_qgauss, q_qgauss = 0,0,0
            cent_qgauss, sigma_qgauss = 0,0
            c_qgauss_err, a_qgauss_err, q_qgauss_err = 0,0,0
            cent_qgauss_err, sigma_qgauss_err = 0,0
            pass
          x,y = profs_norm_avg['pos'],profs_norm_avg['amp']
          sum_y = y.sum()
          dx = x[1] - x[0]
          # c) statistical parameters
          # -- weighted average
          cent_stat  = (x*y).sum()/sum_y
          sigma_stat = np.sqrt((y*(x-cent_stat)**2).sum()/sum_y)
          # d) median
          cent_median  = tb.median(x,y)
          # assume normal distributed
          # -> sigma = 1.4826*MAD (median absolute deviation)
          mad = tb.mad(x,cent_median,y)
          sigma_median = 1.4826*mad
          # e) cumulative sum
          cumsum_sigma_left = 0.5 - cumsum_1sigma/2
          cumsum_sigma_right = 0.5 + cumsum_1sigma/2
          cumsum_y = dx*y.cumsum()
          cumsum_idx_50 = (np.abs(cumsum_y-0.5)).argmin()
          cumsum_idx_sigma_32 = (np.abs(cumsum_y-
          cumsum_sigma_left)).argmin()
          cumsum_idx_sigma_68 = (np.abs(cumsum_y-
          cumsum_sigma_right)).argmin()
          cent_cumsum = x[cumsum_idx_50]
          sigma_cumsum_32 = cent_cumsum-x[cumsum_idx_sigma_32]
          sigma_cumsum_68 = x[cumsum_idx_sigma_68] - cent_cumsum
          # f) peak of distribution
          cent_peak = x[y.argmax()]
          # background estimate
          # average the last 10 bins
          bgnavg = 10
          bg_avg_left = y[:bgnavg].mean()
          bg_avg_right = y[-bgnavg:].mean()
          bg_avg = (bg_avg_left+bg_avg_right)/2
          self.profiles_stat[plane][slot].append((time_stamp,
            # Gaussian fit
            c_gauss, a_gauss, cent_gauss, sigma_gauss,
            c_gauss_err, a_gauss_err, cent_gauss_err, sigma_gauss_err,
            # q-Gaussian fit
            c_qgauss, a_qgauss, q_qgauss, cent_qgauss, beta_qgauss,
            sigma_qgauss,
            c_qgauss_err, a_qgauss_err, q_qgauss_err, cent_qgauss_err,
            beta_qgauss_err, sigma_qgauss_err,
            # statistical parameters
            cent_stat, sigma_stat, cent_median, mad, sigma_median,
            cent_cumsum, sigma_cumsum_32, sigma_cumsum_68,
            cent_peak,
            # background estimate
            bg_avg_left,bg_avg_right,bg_avg))
    # convert to a structured array
    ftype=[('time_stamp',int),('c_gauss',float),('a_gauss',float),
           ('cent_gauss',float),('sigma_gauss',float),
           ('c_gauss_err',float),('a_gauss_err',float),
           ('cent_gauss_err',float),('sigma_gauss_err',float),
           ('c_qgauss',float),('a_qgauss',float),('q_qgauss',float),
           ('cent_qgauss',float),('beta_qgauss',float),
           ('sigma_qgauss',float),('c_qgauss_err',float),
           ('a_qgauss_err',float),('q_qgauss_err',float),
           ('cent_qgauss_err',float),('beta_qgauss_err',float),
           ('sigma_qgauss_err',float),('cent_stat',float),
           ('sigma_stat',float),('cent_median',float),('mad',float),
           ('sigma_median',float),('cent_cumsum',float),
           ('sigma_cumsum_32',float),('sigma_cumsum_68',float),
           ('cent_peak',float),('bg_avg_left',float),
           ('bg_avg_right',float),('bg_avg',float)]
    for plane in ['h','v']:
      for k in self.profiles_stat[plane].keys():
        self.profiles_stat[plane][k] = np.array(
          self.profiles_stat[plane][k], dtype=ftype)
    # variable for statistical variable names
    self.profiles_stat_var = [ ftype[k][0] for k in xrange(len(ftype)) ]
    return self
  def get_slots(self):
    """
    get sorted list of slot numbers
    """
    slotsh = np.sort(list(set(self.profiles['h'].keys())))
    slotsv = np.sort(list(set(self.profiles['v'].keys())))
    if np.all(slotsh == slotsv):
      return slotsh
    else:
      print('WARNING: slot numbers in H and V differ!')
      return (slotsh,slotsv)
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
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles[plane][slot][mask][0]
    else:
      return self.profiles[plane][slot][mask]
  def get_profile_norm(self, slot = None, time_stamp = None, 
                       plane = 'h'):
    """
    get normalized profile data for slot *slot*, time stamp
    *time_stamp* as unix time [ns] and plane *plane*.
    """
    ts = self.get_timestamps(slot=slot,plane=plane)
    if time_stamp not in ts:
      raise ValueError('time stamp %s '%(time_stamp) + 'is not in ' +
          'list of time stamps for slot %s and plane %s'%(slot,plane))
    mask = self.profiles_norm[plane][slot]['time_stamp'] == time_stamp
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm[plane][slot][mask][0]
    else:
      return self.profiles_norm[plane][slot][mask]
  def get_profile_norm_avg(self, slot = None, time_stamp = None,
                           plane = 'h'):                                    
    """
    returns averaged normalized profile for slot *slot*, time stamp
    *time_stamp* as unix time [ns] and plane *plane*.
    """
    mask = (self.profiles_norm_avg[plane][slot]['time_stamp'] 
                == time_stamp)
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm_avg[plane][slot][mask][0]
    else:
      return self.profiles_norm_avg[plane][slot][mask]
  def get_profile_stat(self, slot = None, time_stamp = None, plane = 'h'):
    """
    get profile data for slot *slot*, time stamp *time_stamp* as
    unix time [ns] and plane *plane*
    """
    mask = self.profiles_stat[plane][slot]['time_stamp'] == time_stamp
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_stat[plane][slot][mask][0]
    else:
      return self.profiles_stat[plane][slot][mask]
  def _plot_profile(self, slot = None, time_stamp = None, plane = 'h',
                    norm = True, smooth = True, verbose = False):
    """
    Plot all profiles for specific slot and time. Plot title displays 
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
    smooth: if True the averaged profiles are smoothed using lowess
            function from the statsmodels module
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
        stat_aux = self.get_profile_stat(slot=slot,
                     time_stamp=time_stamp,plane=plane)
        # Gaussian fit
        c_gauss, a_gauss = stat_aux['c_gauss'], stat_aux['a_gauss']
        cent_gauss = stat_aux['cent_gauss']
        sigma_gauss    = stat_aux['sigma_gauss']
        # q-Gaussian fit
        c_qgauss, a_qgauss = stat_aux['c_qgauss'], stat_aux['a_qgauss']
        q_qgauss = stat_aux['q_qgauss']
        cent_qgauss = stat_aux['cent_qgauss']
        beta_qgauss    = stat_aux['beta_qgauss']
    # raw data profile
    else:
      profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                               plane=plane)
    for i in xrange(len(profs)):
      try:
        pl.plot(profs[i]['pos'],profs[i]['amp'],
                label='profile %s'%(i+1),linestyle='-',
                color=profile_colors.values()[i])
        if norm:
          pl.ylabel(r'probability (integral normalized to 1) [a.u.]')
          pl.ylim(5.e-5,0.5)
        else:
          pl.ylabel('intensity [a.u.]')
        pl.xlabel('position [mm]')
        pl.grid(b=True)
        pl.legend(loc='best')
        # convert ns -> s before creating date string
        ts = ld.dumpdate(t=time_stamp*1.e-9,
                 fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
        pl.gca().set_title('slot %s, %s plane, %s'%(slot,
                             plane.upper(),ts))
      except ValueError:
        if verbose:
          print('ERROR: plotting of slot %s, '%(slot) +
          'profile %s, time stamp %s failed.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
           len(profs[i]['amp'])))
        check_plot = False
        pass
    if norm and check_plot:
      # smooth data
      if smooth:
        #profs_avg['pos'],profs_avg['amp'] = 
        prof_smooth = lowess(endog=profs_avg['amp'],
            exog=profs_avg['pos'],frac=0.025,it=3,delta=0,
            is_sorted=True,missing='drop')
        profs_avg['pos'] = prof_smooth[:,0]
        profs_avg['amp'] = prof_smooth[:,1]
      # plot average over profiles
      pl.plot(profs_avg['pos'],profs_avg['amp'],
              label = 'average profile',color='k',linestyle='-')
      if self.profiles_stat is not None:
        # plot Gaussian fit
        pl.plot(profs[i]['pos'],tb.gauss_pdf(profs[i]['pos'],
                c_gauss,a_gauss,cent_gauss,sigma_gauss),
                color=fit_colors['Red'],
                linestyle='--',linewidth=1,label='Gaussian fit')
        # plot q-Gaussian fit
        pl.plot(profs[i]['pos'],tb.qgauss_pdf(profs[i]['pos'],
                c_qgauss,a_qgauss,q_qgauss,cent_qgauss,beta_qgauss),
                color=fit_colors['DarkRed'],
                linestyle='-',linewidth=1,label='q-Gaussian fit')
    return check_plot
  def plot_profile(self, slot = None, time_stamp = None, plane = 'h',
                   verbose = False):
    """
    Plot raw data profiles for specific slot and time. Plot title displays 
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
    Plot all profiles for specific slot and time. Profiles are
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
    a specific slot and time. Profiles are normalized to represent
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
    if self.profiles_stat is not None:
      sta = self.get_profile_stat(slot=slot,
                   time_stamp=time_stamp,plane=plane)
    # flag if profile plot failed
    check_plot = True
    # individual profiles
    for i in xrange(len(profs)):
      try:
        dx = profs[i]['pos'][1]-profs[i]['pos'][0]
        pl.plot(profs[i]['pos'],(dx*profs[i]['amp']).cumsum(),
                label='profile %s'%(i+1),
                color=profile_colors.values()[i])
      except ValueError:
        if verbose:
          print('ERROR: plotting of slot %s, '%(slot) +
          'profile %s, time stamp %s failed.'%(i,time_stamp) +
          ' len(x) =%s, len(y) = %s'%(len(profs[i]['pos']),
           len(profs[i]['amp'])))
        check_plot = False
        pass
    # average profile
    if check_plot:
      dx = profs[i]['pos'][1]-profs[i]['pos'][0]
      pl.plot(profs_avg['pos'],(dx*profs_avg['amp']).cumsum(),
            label = 'average profile',color='k',linestyle='-')
      if self.profiles_stat is not None:
        pl.plot(profs_avg['pos'],(dx*tb.gauss_pdf(profs_avg['pos'],
                sta['c_gauss'],sta['a_gauss'],sta['cent_gauss'],
                sta['sigma_gauss'])).cumsum(),
                label = 'Gaussian fit',color = fit_colors['Red'],
                linestyle = '--')
        pl.plot(profs_avg['pos'],(dx*tb.qgauss_pdf(profs_avg['pos'],
                sta['c_qgauss'],sta['a_qgauss'],sta['q_qgauss'],
                sta['cent_qgauss'],sta['beta_qgauss'])).cumsum(),label = 'Gaussian fit', 
                color = fit_colors['DarkRed'], linestyle = '-')
      pl.xlabel('position [mm]')
      pl.ylabel(r'cumulative distribution functions [a.u.]')
      pl.ylim(-0.05,1.05)
      pl.grid(b=True)
      pl.legend(loc='best')
      # convert ns -> s before creating date string
      ts = ld.dumpdate(t=time_stamp*1.e-9,
                       fmt='%Y-%m-%d %H:%M:%S.SSS',zone='cern')
      pl.gca().set_title('slot %s, %s plane, %s'%(slot,
                          plane.upper(),ts))
    return check_plot
  def plot_residual(self, slot = None, time_stamp = None, 
                    time_stamp_ref = None, plane = 'h',
                    verbose = False):
    """
    Plot residual of normalized profiles for a specific slot and time.
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
    Plot ratio of normalized profiles for a specific slot and time.
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
                           plane = 'h', smooth = False,
                           verbose = False):
    """
    Plot residual or ratio of normalized profiles for a specific slot and time.
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
            print('ERROR: plotting of slot %s, '%(slot) +
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
        if self.profiles_stat is not None:
          sta = self.get_profile_stat(slot=slot,
                       time_stamp=time_stamp,plane=plane)
          # Gaussian fit
          c_gauss, a_gauss = sta['c_gauss'], sta['a_gauss']
          cent_gauss = sta['cent_gauss']
          sigma_gauss    = sta['sigma_gauss']
          # q-Gaussian fit
          c_qgauss, a_qgauss = sta['c_qgauss'], sta['a_qgauss']
          q_qgauss = sta['q_qgauss']
          cent_qgauss = sta['cent_qgauss']
          beta_qgauss    = sta['sigma_qgauss']
          pl.plot(profs_avg['pos'][mask],
                  profs_avg['amp'][mask]-
                  tb.gauss_pdf(profs_avg['pos'][mask],sta['c_gauss'],
                    sta['a_gauss'],sta['cent_gauss'],
                    sta['sigma_gauss']),
                  label = 'Gaussian fit',
                  color=fit_colors['Red'],linestyle='--')
          pl.plot(profs_avg['pos'][mask],
                  profs_avg['amp'][mask]-
                  tb.qgauss_pdf(profs_avg['pos'][mask],sta['c_qgauss'],
                    sta['a_qgauss'],sta['q_qgauss'],
                    sta['cent_qgauss'],sta['beta_qgauss']), 
                  label = 'q-Gaussian fit',
                  color=fit_colors['DarkRed'],linestyle='-')
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
      pl.gca().set_title('slot %s, %s plane, %s'%(slot,
                               plane.upper(),ts))
    return check_plot
  def plot_all(self,slot = None, time_stamp = None,
               time_stamp_ref = None, plane = 'h', norm = True, 
               smooth = True, verbose = False):
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
    smooth: if True the averaged profiles are smoothed using lowess
            function from the statsmodels module
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
    pl.suptitle('slot %s, %s plane, %s'%(slot,
                             plane.upper(),ts))
    # 1) profile plot
    pl.subplot(223)
    # flaux = flag for checking if profile plots have failed
    flaux = self._plot_profile(slot=slot,time_stamp=time_stamp,
                              plane=plane,norm=norm,smooth=smooth,
                              verbose=verbose)
    pl.gca().set_yscale('log')
    if norm:
      pl.gca().set_ylabel('probability [a.u.]')
    # 2) cumulative sum
    pl.subplot(224)
    self.plot_cumsum(slot=slot,time_stamp=time_stamp,plane=plane,
                     verbose=verbose)
    pl.gca().set_ylabel('cumulative dist. [a.u.]')
#     3) residual, only average profile
    pl.subplot(221)
    self._plot_residual_ratio(flag='residual', flagprof='avg', 
           slot=slot, time_stamp=time_stamp,
           time_stamp_ref=time_stamp_ref, plane=plane, smooth=smooth,
           verbose=verbose)
    pl.gca().set_ylim(-0.05,0.05)
#     4) ratio
    pl.subplot(222)
    self._plot_residual_ratio(flag='ratio', flagprof='avg', 
           slot=slot, time_stamp=time_stamp,
           time_stamp_ref=time_stamp_ref, plane=plane, smooth=smooth,
           verbose=verbose)
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
    fig.subplots_adjust(top=0.38)
    fig.tight_layout()
    return flaux
  def mk_profile_video(self, slots = None, t1=None, t2=None,
                       plt_dir='BSRTprofile_gifs',
                       delay=20, norm = True, export=False, 
                       verbose=False):
    """
    Generates a video of the profiles of slot with *slot*

    Parameters:
    -----------
    slot : slot or list of slots of slots. If slot = None all slots
           are selected. In case of several slot, on video is 
           generated per slot
    t1 : start time, if None first time stamp for slot is used
    t2 : end time, if None last time stamp for slot is used
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
      slots = self._set_slots(plane=plane,slots=slots)
      # generate the figure and subplot
      pl.figure(plane)
      for slot in slots:
        if os.path.exists(tmpdir) == False:
          os.makedirs(tmpdir)
        check_plot[plane][slot] = []
        if verbose:
          print('... generating plots for slot %s'%slot)
        time_stamps = self.get_timestamps(slot=slot,plane=plane)
        if t1 is None:
          t1 = time_stamps[0]
        if t2 is None:
          t2 = time_stamps[-1]
        time_stamps = time_stamps[(time_stamps >= t1) & 
                                  (time_stamps <= t2)]
        if len(time_stamps) > 0:
          time_stamp_ref = time_stamps[0]
        else:
          if verbose:
            print('WARNING: no time stamps found for slot %s, '%slot +
                  'plane %s and (t1,t2)=(%s,%s)'%(plane,t1,t2))
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
          fnpl = os.path.join(tmpdir,'slot_%s_plane_%s_%05d.png'%(slot,
                              plane,pngcount))
          if verbose: print '... save png %s'%(fnpl)
          pl.savefig(fnpl)
          pngcount += 1
        # create video from pngs
        if len(time_stamps) > 0:
          if verbose:
            print '... creating .gif file with convert'
          cmd="convert -delay %s %s %s"%(delay,
               os.path.join(tmpdir,'slot_%s_plane_%s_*.png'%(slot,plane)),
               os.path.join(plt_dir,'slot_%s_plane_%s.gif'%(slot,plane)))
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
