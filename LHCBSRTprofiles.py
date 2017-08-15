#! /usr/bin/env python

try:
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as pl
  from scipy.optimize import curve_fit
  from scipy.special import erf
  from statsmodels.nonparametric.smoothers_lowess import lowess
  import cPickle
  import glob
except ImportError:
  print('No module found: numpy, matplotlib, scipy, cPickle, glob and' +
        ' statsmodels module should be present to run pytimbertools')

try:
  import pytimber
except ImportError:
  print('No module pytimber found!')

import os
import shutil
import glob
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
    BSRTprofiles.load_files(files='*.bindata')
  Load profiles from list (load_files), discard noisy profiles 
  (clean_data), calculate statistical parameters (stats):
    BSRTprofiles.load_files(files='*.bindata').clean_data().stats()
  Attributes:
  -----------
  beam: either 'B1' or 'B2'
  db: pytimber data base to get lsf factor, beta etc.
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

  Only filled after running self.norm():
    profiles_norm : same format as profiles. Contains all profiles are
        normalized to represent a probability distribution with an
        integral of 1.
    profiles_norm_avg : same fromat as profile. Contains for each plane,
        slot and timestamp the average of all normalized profiles for
        this time stamp .
    profiles_norm_mvavg: same format as profiles. Contains for each plane,
        slot and timestamp the moving average over several normalized. The
        default is 11 profiles (5 previous + 5 after).

  Only filled after running self.get_stats():
    profiles_norm_avg_stat: statistical parameters calculated for average profile
        for each timestamp
    profiles_norm_mvavg_stat: statistical parameters calculated for moving 
        average profile for each timestamp

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

  def __init__(self, beam=None,db=None,
               records=None, profiles=None, profiles_norm=None,
               profiles_norm_avg=None, profiles_norm_mvavg=None, 
               profiles_norm_avg_stat=None, profiles_norm_mvavg_stat=None,
               split='full',fitsplit='full',bgnavg = None, nmvavg = None):
    self.beam = beam
    self.db = db
    self.records   = records
    if self.records is None:
      self.filenames = None
    else:
      self.filenames = self.records.keys()
    self.profiles = profiles
    self.profiles_norm = profiles_norm
    self.profiles_norm_avg = profiles_norm_avg
    self.profiles_norm_mvavg = profiles_norm_avg
    self.profiles_norm_avg_stat = profiles_norm_avg_stat
    self.profiles_norm_mvavg_stat = profiles_norm_avg_stat
    if self.profiles_norm_avg_stat is None or self.profiles_norm_mvavg_stat is None:
      self.profiles_norm_avg_stat_var = None
      self.profiles_norm_mvavg_stat_var = None
      # index for covariance matrix for Gaussian and q-Gaussian fit 
      self.fit_var_gauss = None
      self.fit_var_qgauss = None
    else:
      # take first plane and slot in order to get profile_stat named fields
      plane_aux = self.profiles_norm_avg_stat.keys()[0]
      slot_aux = self.profiles_norm_avg_stat[plane_aux].keys()[0] 
      self.profiles_norm_avg_stat_var = self.profiles_norm_avg_stat[plane_aux][slot_aux].dtype.names
      self.profiles_norm_mvavg_stat_var = self.profiles_norm_mvavg_stat[plane_aux][slot_aux].dtype.names
      # index for covariance matrix for Gaussian and q-Gaussian fit 
      # c,a,mu,sig
      self.fit_var_gauss = {0:'c',1:'a',2:'mu',3:'sig'}
      # c,a,q,mu,beta
      self.fit_var_qgauss = {0:'c',1:'a',2:'q',3:'mu',4:'beta'}
    # -- take full, left or right side of profile
    self.split = split
    # -- do not weigh fit (full), or weight left or right side of profile
    self.fitsplit = fitsplit
    # -- background estimate
    # internal flag to check if background has been removed
    self._rm_bg = False
    # bgnavg = average over bgnavg bins on left and bgnavg bins on right to
    # obtain estimate of background level
    self.bgnavg = bgnavg
    # -- estimate noise (uncertainty) with the std over *nmvavg* 
    # measurements, needed to calculate xi**2
    self.nmvavg = nmvavg
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
  def _norm_profiles(self,slot=None,verbose = True):
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
    slot: slot number, list of slots or if slot = None takes all slots
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
      for sl in self._set_slots(plane=plane,slot=slot):
        self.profiles_norm[plane][sl] = []
        for idx in xrange(len(self.profiles[plane][sl])):
          try:
            prof = self.profiles[plane][sl][idx].copy()
            # assume equal spacing
            dx = prof['pos'][1]-prof['pos'][0]
            prof_int = (dx*prof['amp']).sum()
            self.profiles_norm[plane][sl].append(
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
  def _norm_mvavg_profiles(self,slot=None,verbose = True):
    """
    Generate moving average over self.nmvavg+1 timestamps of normalized 
    profiles for each plane, slot and timestamp. The moving average is 
    taken over all profiles for each plane, slot and time stamp.

    Parameters:
    -----------
    slot: slot number, list of slots or if slot = None takes all slots
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm_mvavg (moving average normalized profile)

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    if verbose:
      if self.profiles_norm_mvavg is not None:
        print('... delete self.profiles_norm_mvavg')
    if verbose:
      print('... moving average over self.nmvavg = %s '%(self.nmvavg) +
            'normalized profiles. Averaged normalized profiles are' +
            ' saved in self.profiles_norm_mvavg')
    self.profiles_norm_mvavg = self._get_norm_avg_mvavg_profiles(
                                 slot=slot,nmvavg = self.nmvavg,verbose=verbose)
    return self
  def _norm_avg_profiles(self,slot=None,verbose = True):
    """
    Generate average of normalized profiles for each
    plane, slot and timestamp. The average is taken over all profiles 
    for each plane, slot and 
    time stamp.

    Parameters:
    -----------
    slot: slot number, list of slots or if slot = None takes all slots
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm_avg (average normalized profile)

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    if verbose:
      if self.profiles_norm_avg is not None:
        print('... delete self.profiles_norm_avg')
    if verbose:
      print('... average normalized profiles. Averaged normalized ' +
            'profiles are saved in self.profiles_norm_avg')
    self.profiles_norm_avg = self._get_norm_avg_mvavg_profiles(
                             slot=slot,nmvavg = None,verbose=verbose)
    return self
  def _get_norm_avg_mvavg_profiles(self, slot=None,nmvavg = None, 
                                   verbose = True):
    """
    function to calculate average over all profiles with the same 
    timestamp (nmvavg = None) or moving average over nmvavg time stamps
    (nmvavg = even integer) using all profiles for each time stamp
    
    Parameters:
    -----------
    slot: slot number, list of slots or if slot = None takes all slots
    nmvavg: a) nmvavg = None:
               calculate the average over all profiles with the same
               time stamp
            b) nmvavg = even integer:
               calculate the moving average over all profiles and
               nmvavg+1 time stamps (take previous/following nmvavg/2
               time stamps + profile itself. This means, that if e.g.
               3 profiles are taken per time stamp and nmvavg = 10, the
               average and std is taken over in total 
               3*(nmavg+1) = 33 profiles.
    Returns:
    --------
    profs_norm_avg: structured array of average profiles either calculated
               with a) or b). Profiles are here always probability
               distributions, meaning that they are normalized to 1
               pos = position in mm (bin)
               amp = mean amplitude of bin over a) or b) profiles
               ampstd = standard deviation of amp (estimate of noise)
               amperr = error of mean value (= ampstd/sqrt(nmvavg) )
    """
    # check values
    if nmvavg is not None:
      if nmvavg % 2 != 0:
         raise ValueError('nmvavg must be None or an even integer!')
    # abbreviate profs_norm_avg with pna
    pna = {}
    for plane in 'h','v':
      if verbose:
        print('... start plane = %s'%plane)
      pna[plane]={}
      for sl in self._set_slots(plane=plane,slot=slot):
        if verbose:
          print('... normalizing slot = %s'%sl)
        pna[plane][sl]=[]
        ts = self.get_timestamps(slot=sl,plane=plane)
        # check that there are enough time stamps for nmvavg
        if (nmvavg is not None) and (nmvavg + 1 > len(ts)):
          print('ERROR: not enough profiles for nmvavg = %s!'%nmvavg
          +'slot = %s, plane = %s'%(sl,plane))
          continue
        for time_stamp,ts_idx in zip(ts,xrange(len(ts))):
          # a) nmvavg = None: average over all profiles with the same
          #    time stamp -> time_stamp already correctly set
          if nmvavg is None: 
            ts_profs = time_stamp
          # b) nmvavg = even integer:
          else:
            nmvavg_2 = int(round(nmvavg/2))
            if ts_idx <= nmvavg_2:
              ts_profs = [ts[0],ts[nmvavg]]
            elif ts_idx >= len(ts)-nmvavg_2: 
              ts_profs = [ts[-nmvavg-1],ts[-1]]
            else:
              ts_profs = [ts[ts_idx-nmvavg_2],ts[ts_idx+nmvavg_2]]
          profs = self.get_profile_norm(plane=plane,slot=sl,
                                         time_stamp=ts_profs)
          # no profiles found -> skip
          if len(profs) == 0:
            if verbose:
              print('slot %s or timestamp %s not found'%(sl,time_stamp))
            continue
          # one profile
          elif len(profs) == 1:
            pos = profs[0]['pos']
            amp = profs[0]['amp']
            ampstd = np.zeros(len(pos))
            amperr = np.zeros(len(pos))
          # more than one profile
          elif len(profs) > 1:
            # take the average over the profiles
            # 1) a) shorten all profiles to the minimum length if they do
            #       not already have the same length
            #    b) rebin to binning of first profile (note sometimes
            #       two values fall in the same bin -> peaks in 
            #       distribution)
            xmin = np.max([np.min(n) for n in profs['pos']])
            xmax = np.min([np.max(n) for n in profs['pos']])
            # assume equal bin size
            dbin = profs[0]['pos'][1]-profs[0]['pos'][0]
            mask = np.logical_and(profs[0]['pos'] >= xmin,
                                  profs[0]['pos'] <= xmax)
            bin_edges = [profs[0]['pos'][0]-dbin/2]+list(profs[0]['pos'][mask]+dbin/2)
            for i in xrange(len(profs)):
              mask = np.logical_and(profs[i]['pos'] >= xmin,
                                    profs[i]['pos'] <= xmax)
              profs[i]['pos'] = profs[i]['pos'][mask]
              profs[i]['amp'] = profs[i]['amp'][mask]
              dbin_i = profs[i]['pos'][1]-profs[i]['pos'][0]
              # rebin to binning of first profile, make linear interpolation
              # in each bin
              if dbin != dbin_i:
                profs[i]['amp'] = np.interp(x=profs[0]['pos'],xp=profs[i]['pos'], 
                                            fp=profs[i]['amp'])                   
                profs[i]['pos'] = profs[0]['pos']                                 
            # 2) check that x-axis of all profiles are the same
            #    check_x = 0 if x-axis of profiles are the same
            #    check_x > 0 if x-axis differ
            check_x = 0
            for i in xrange(len(profs)):
              if ((len(profs[0]['pos'])-len(profs[i]['pos']) !=0)
                 or (len(profs[0]['amp'])-len(profs[i]['amp']) !=0)):
                check_x +=1
            if check_x != 0:
              print('ERROR: Rebinning of profile failed for ' +
                    'plane %s, slot %s, timestamp %s'%(plane,sl,time_stamp))
              continue
            # 2) if only left/right profile is taken we need to find
            # the peak, mirror left/right profile, renormalize
            if self.split == 'left' or self.split == 'right':
              # find peak using average profile
              pos,amp = profs[0]['pos'],profs['amp'].mean(axis=0)
              di = 40 
              ilim=[len(pos)/2-di,len(pos)/2+di] # take central region
              x,y=pos[ilim[0]:ilim[1]],amp[ilim[0]:ilim[1]]
              popt, pcov = curve_fit(f=tb.gauss_pdf, xdata=x, ydata=y,
                             p0=[0,1,0,2],bounds=([-0.1,0,-8,0],[0.1,2,8,16]))
              imax = np.argmax(tb.gauss_pdf(x,*popt))+ilim[0]
              # mirror profile
              for i in xrange(len(profs)):
                if self.split == 'left':
                  profs['pos'][i] = np.concatenate((profs['pos'][i][:imax+1],
                    2*profs['pos'][i][imax]-profs['pos'][i][:imax][::-1]),axis=0)
                  profs['amp'][i] = np.concatenate((profs['amp'][i][:imax+1],
                    profs['amp'][i][:imax][::-1]),axis=0)
                if self.split == 'right':
                  profs['pos'][i] = np.concatenate(
                    (2*profs['pos'][i][imax]-profs['pos'][i][imax+1:][::-1],
                    profs['pos'][i][imax:]),axis=0)
                  profs['amp'][i] = np.concatenate(
                    (profs['amp'][i][imax+1:][::-1],profs['amp'][i][imax:]),axis=0)
                dx = profs['pos'][i][1]-profs['pos'][i][0]
                prof_int = (dx*profs['amp'][i]).sum()
                profs['amp'][i] = profs['amp'][i]/prof_int
            # recheck position
            check_x = 0
            for i in xrange(len(profs)):
              if (np.abs(profs[0]['pos']-profs[i]['pos'])).sum() != 0:
                check_x +=1
            if check_x != 0:
              print('ERROR: Mirroring of profile failed. Skipping ' +
                    'plane %s, slot %s, timestamp %s'%(plane,sl,time_stamp))
              continue
            # take the average. Integral is normalized to 1
            pos = profs[0]['pos']
            # mean amplitude of bin
            amp = profs['amp'].mean(axis=0)
            # standard deviation of amp (estimate of noise)
            ampstd = profs['amp'].std(axis=0)
            # error of mean value (= ampstd/sqrt(nmvavg) )
            amperr = ampstd/np.sqrt(len(profs))
            pna[plane][sl].append((int(time_stamp),
                                     pos,amp,ampstd,amperr))
    # convert to a structured array and sort by time stamp
    ftype=[('time_stamp',int), ('pos',np.ndarray), ('amp',np.ndarray),
           ('ampstd',np.ndarray), ('amperr',np.ndarray)]
    for plane in ['h','v']:
      for k in pna[plane].keys():
        pna[plane][k] = np.array(pna[plane][k], dtype=ftype)
    return pna
  def norm(self, slot = None, nmvavg = 10, split = 'full', verbose = False):
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
    slot: slot number, list of slots or if slot = None takes all slots
    nmvavg : number of timestamps used for:
                - moving average over profiles
                - estimate of noise for each bin for profiles i = std over
                  previous/following nmvavg profiles (-> in total
                  nmvavg + 1 profiles)
                Note: nmvavg must be an even number!
    split: take only half of the profile - needed at top energy
           where profiles are diffraction limited. The profile is split by
           detecting the peak and then mirroring it on the other side.
           'left': take left profile
           'right': take right profile
           'full': (default) take full profile
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
        self.profiles_norm (normalized profiles) and 
        self.profiles_norm_avg (average normalized profile)
        self.profiles_norm_mvavg (moving average over nmvavg +1 
                                  normalized profile)

    Note:
    -----
    profile data is assumed to be equally spaced in x
    """
    self.split = split
    # self.profiles_norm
    self._norm_profiles(slot=slot,verbose=verbose)
    # self.profiles_norm_avg
    self._norm_avg_profiles(slot=slot,verbose=verbose)
    # self.profiles_norm_mvavg
    if self.nmvavg is not None:
      print('WARNING: Changing self.nmvavg from %s '%(self.nmvavg) +
            'to %s'%nmvavg)
    self.nmvavg = nmvavg
    if self.nmvavg is not None:
      self._norm_mvavg_profiles(slot=slot,verbose=verbose)
    return self
  def remove_background(self,bgnavg = 10,verbose=False):
    """
    Removes the background from all normalized profiles.
    Estimate background by:
      1) averaging the amplitude over the first and last 10 bins for
         each slot
      2) averaging over all profiles for each slot as background seems
         to stay rather constant

    Parameters:
    -----------
    bgnavg : estimate background by taking the average value over
             the bgnavg most left and bgnavg most right bins
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object where the background is removed
        from all normalized profiles.
    """
    if self.bgnavg is not None:
      print('WARNING: Setting bgnavg to %s '%bgnavg +
            'from previously %s'%self.bgnavg)
    self.bgnavg = bgnavg
    if (self.profiles_norm is None) or (self.profiles_norm_avg is None):
      self = self.norm()
    if verbose:
      print('... remove background from normalized profiles')
    for plane in 'h','v':
      for slot in self.profiles_norm[plane].keys():
        # estimate background
        ts = self.get_timestamps(slot=slot,plane=plane)
        bg_avg_left = np.mean([ self.get_profile_norm_avg(slot=slot,
                          time_stamp=t,plane=plane)['amp'][:self.bgnavg] 
                          for t in ts ],axis=None)
        bg_avg_right = np.mean([ self.get_profile_norm_avg(slot=slot,
                          time_stamp=t,plane=plane)['amp'][-self.bgnavg:] 
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
    # flag to save status that background has been removed
    self._rm_bg = True
    return self
  def _set_slots(self,plane,slot):
    """
    set slot numbers, handles the case of slot = None and only one 
    slot.
    """
    # 1) all slots
    if slot is None:
      slot = self.profiles[plane].keys()
    # 2) if single slot 
    elif not hasattr(slot,"__iter__"):
      slot = [slot]
    return np.sort(slot,axis=None)
  def get_beta_lsf_variable_names(self):
    """
    get variables names in logging database for lsf correction factor,
    beta function at BSRT location and beam energy.
    """
    db   = self.db
    beam = self.beam
    [lsf_h_var, lsf_v_var]   = db.search('%LHC%BSRT%'+beam.upper()      
                                         +'%LSF_%')                     
    [beta_h_var, beta_v_var] = db.search('%LHC%BSRT%'+beam.upper()      
                                         +'%BETA%')                     
    energy_var = u'LHC.BOFSU:OFC_ENERGY'                                    
    return lsf_h_var,lsf_v_var,beta_h_var,beta_v_var,energy_var
  def get_beta_lsf_energy(self):
    """
    get energy, lsf correction factor and beta function at BSRT for
    profiles using pytimber
    
    Returns:
    --------
    dictionary with energy, lsf correction and beta function in the
    format:
      {key: (t,v)
    where t is an array with the timestamps and v is the array with the
    values. All timestamps are in [ns] (note that logging database is
    in [s]!)
    """
    db   = self.db
    beam = self.beam
    if db is None:
      try:
        db = pytimber.LoggingDB()
      except NameError:
        print('ERROR: Trying to use db = pytimber.LoggingDB(), but ' +
              'pytimber is not imported! You can use ' +
              'pagestore database instead for offline analysis!')
        return
    # get min/max time of profile data
    (t1,t2) = self._get_range_timestamps()
    # convert [ns] -> [s]
    (t1,t2) = (t1*1.e-9,t2*1.e-9)
    bsrt_lsf_var = self.get_beta_lsf_variable_names()
    lsf_h_var,lsf_v_var,beta_h_var,beta_v_var,energy_var = bsrt_lsf_var
    t1_lsf = t1
    bsrt_lsf = db.get(bsrt_lsf_var, t1_lsf, t2)
    # only logged rarely, loop until array is not empty, print warning
    # if time window exceeds one month
    # check that time stamp of lsf,beta,energy is before first sigma
    # timestamp
    for var in bsrt_lsf_var:
      while (bsrt_lsf[var][0].size  == 0):
        if (np.abs(t1_lsf-t1) > 30*24*60*60):
          raise ValueError(('Last logging time for ' + ', %s'*5
          + ' exceeds 1 month! Check your data!!!')%tuple(bsrt_lsf_var))
          return
        else:
          t1_lsf = t1_lsf-24*60*60
          bsrt_lsf = db.get(bsrt_lsf_var, t1_lsf, t2)
      while (bsrt_lsf[var][0][0] > t1):
        if (np.abs(t1_lsf-t1) > 30*24*60*60):
          raise ValueError(('Last logging time for ' + ', %s'*5
          + ' exceeds 1 month! Check your data!!!')%tuple(bsrt_lsf_var))
          return
        else:
          t1_lsf = t1_lsf-24*60*60
          bsrt_lsf = db.get(bsrt_lsf_var, t1_lsf, t2)
    # convert [s] -> [ns] for all time stamps 
    for k in bsrt_lsf.keys():
      bsrt_lsf[k] = list(bsrt_lsf[k])
      bsrt_lsf[k][0] = bsrt_lsf[k][0]*1.e9
    return bsrt_lsf
  def update_beta_lsf_energy(self,t1=None,t2=None,beth=None,betv=None,                   
                        lsfh=None,lsfv=None,energy=None,verbose=True):                             
    """
    update beta function and lsf factor and recalculate emittances

    beth,betv: hor.,vert. beta [m]
    lsfh,lsfv: lsf conversion factor [mm]?
    energy: energy [GeV]
    """
    if self.db is not None:
      bsrt_lsf_db = self.get_beta_lsf_energy()
    bsrt_lsf_var = self.get_beta_lsf_variable_names()
    lsf_var  = {}
    beta_var = {}
    (lsf_var['h'],lsf_var['v'],beta_var['h'],beta_var['v'],
         energy_var) = bsrt_lsf_var
    print 'energy',energy,np.min(bsrt_lsf_db[energy_var][1])
    if verbose:
      for k,v,dbk in zip(['beth','betv','lsfh','lsfv','energy'],
                         [beth,betv,lsfh,lsfv,energy],
                         [beta_var['h'],beta_var['v'],
                          lsf_var['h'],lsf_var['v'],energy_var]):
        if v is not None:
          print(k,': old=',bsrt_lsf_db[dbk][1],', new=',v)
    for stat in ['profiles_norm_avg_stat','profiles_norm_mvavg_stat']: 
      profs = self.__dict__[stat]
      # loop over plane
      for p in profs.keys():
        if (p.lower() == 'h' and beth is None and lsfh is None and 
            energy is None):
          continue
        if (p.lower() == 'v' and betv is None and lsfv is None and 
            energy is None):
          continue
        # loop over slots
        for s in profs[p].keys():
          times = profs[p][s]['time_stamp']
          if t1 is None:
            t1 = times[0]
          if t2 is None:
            t2 = times[-1]
          # select time stamps
          mask  = np.logical_and(times >= t1, times <= t2)
          # loop over time stamps
          for ii in xrange(len(times)):
            if not mask[ii]:
              continue
            time_stamp = profs[p][s]['time_stamp'][ii]
            if self.db is not None: 
              idx = (np.where(time_stamp-
                              bsrt_lsf_db[lsf_var[p]][0]>=0.)[0][-1])
              beta = bsrt_lsf_db[beta_var[p]][1][idx]
              lsf  = bsrt_lsf_db[lsf_var[p]][1][idx]
              egev = bsrt_lsf_db[energy_var][1][idx]
            elif None in [beth,lsfh,betv,lsfv,energy]:
              print('ERROR: no timber database defined and not all '+
                    'beta, lsf and energy values are defined!')
              print('  beth,lsfh,betv,lsfv,energy = '+'%4.2f, '*3 +
                    '%4.2f'%(beth,lsfh,betv,lsfv,energy))
            if p.lower == 'h':
               if beth is not None:
                 beta = beth
                 profs[p][s]['beta'][ii] = beta
               if lsfh is not None:
                 lsf = lsfh
                 profs[p][s]['lsf'][ii] = lsf
            if p.lower == 'v':
               if betv is not None:
                 beta = betv
                 profs[p][s]['beta'][ii] = beta
               if lsfv is not None:
                 lsf = lsfv
                 profs[p][s]['lsf'][ii] = lsf
            if energy is not None:
              egev = energy
              profs[p][s]['energy'][ii] = egev
            # use sigma_gauss to later rescale plot x-axis to sigma
            sigma = profs[p][s]['sigma_gauss'][ii]
            profs[p][s]['sigma_beam'][ii] = ((sigma**2 - lsf**2)/beta)
            for key in ['sigma_gauss','sigma_qgauss','sigma_stat',
                        'sigma_median','sigma_cumsum_32',
                        'sigma_cumsum_68']:
              sigma = profs[p][s][key][ii]
              # geometric emittance
              emit_geom = ((sigma**2 - lsf**2)/beta)
              emit_norm = pytimber.toolbox.emitnorm(emit_geom,
                            EGeV=egev,m0=938.272046)
              profs[p][s][key.replace('sigma','emit')][ii] = emit_norm 
      self.__dict__[stat] = profs
    return None
#  def sigma_prof_to_sigma_beam(self,sigma_prof,plane,time_stamp):
#    """
#    convert profile sigma to beam sigma
#      sigma_beam = sqrt(sigma_prof**2-lsf**2)
#
#    Parameter:
#    ----------
#    sigma_prof: profile sigma [mm]
#    plane: 'h' or 'v'
#    time_stamp: time stamp
#
#    Returns:
#    --------
#    sigma_beam: beam sigma [mm]
#    """
#    # get variable names
#    bsrt_lsf_var = self.get_beta_lsf_variable_names()
#    lsf_var={};beta_var={}
#    (lsf_var['h'],lsf_var['v'],beta_var['h'],beta_var['v'],
#         energy_var) = bsrt_lsf_var
#    # extract the data
#    bsrt_lsf_db = self.get_beta_lsf_energy()
#    idx = np.where(time_stamp-bsrt_lsf_db[lsf_var[plane]][0]>=0.)[0][-1]
#    lsf = bsrt_lsf_db[lsf_var[plane]][1][idx]
#    return np.sqrt(sigma_prof**2-lsf**2)
      
  def get_stats(self,slot=None,beam=None,db=None,force=False,
                verbose=False,bgnavg = 10,fitsplit = 'full'):
    """
    calculate statistical parameters for the average over all profiles
    for each timestamp.
    
    To use without conversion to emittance, just do:
      self.get_stats()

    Parameters:
    -----------
    slot: slot number, list of slots or if slot = None takes all slots
    beam: beam, either 'B1' or 'B2' or None for no conversion.
          The specification of the beam is needed to convert picture 
          sigma to beam normalized emittance using 
            self.get_beta_lsf_energy to
          extract variables from logging data base. If beam=None
    db: database to extract data. Can be either directly from the
        logging database (also in case of default db=None if no 
        database is given):
          db = pytimber.LoggingDB()
        or a local one from pagestore:
          db=pagestore.PageStore(dbfile,datadir,readonly=True)
    bgnavg: average over first and last bgnavg bins of each
            average profile (self.profiles_norm_avg) to obtain an
            estimate for the background
    fitsplit: put additional weights in the fitting of the Gaussian
              and qGaussian.
              'full': do not put any weights
              'left': put higher errors on the right 1/3 of the
                      distribution
              'right': put higher errors on the left 1/3 of the
                      distribution
    force: force recalculation
    verbose : verbose option for additional output

    Returns:
    --------
    self : BSRTprofiles class object with recalculated 
           self.profiles_norm_avg_stat

    Conversion from beam sigma to emittance (only valid for Gaussian
    distributions):
             *energy* = beam energy
             *beta*   = beta function at BSRT
             *lsf*    = correction factor for system optical resolution
             BSRT profile sigma are then converted to beam normalized
             emittance with:
                sigma_beam = sqrt((sigma_bsrt)**2-(lsf)**2)
                eps_beam = sigma_beam**2/beta/(beta_rel*gamma_rel)
    """
    self.fitsplit = fitsplit
    # check of input parameters (make that better)
    if beam is None:
      if verbose:
        print("WARNING: no conversion of sigma to beam emittance!" +
              "Specify the parameter 'beam' and the database 'db'" + 
              "to be used for data extraction.")
    elif beam.upper() == 'B1' or beam.upper() == 'B2':
      self.beam = beam.upper()
    else:
      raise ValueError('beam must be either None, b1 or b2!')
    if db is None:
      try:
        self.db = pytimber.LoggingDB()
      except NameError:
        print('ERROR: Trying to use db = pytimber.LoggingDB(), but ' +
              'pytimber is not imported! You can use ' +
              'pagestore database instead for offline analysis!')
        pass
    else:
      self.db = db
    if self.bgnavg is not None and verbose:
      print('... setting bgnavg to %s '%bgnavg +
            'from previously %s'%self.bgnavg)
    self.bgnavg = bgnavg
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
    if self.profiles_norm_avg_stat is None:
      self.profiles_norm_avg_stat={}
      self.profiles_norm_avg_stat['h']={}
      self.profiles_norm_avg_stat['v']={}
    if self.profiles_norm_mvavg_stat is None and self.nmvavg is not None:
      self.profiles_norm_mvavg_stat={}
      self.profiles_norm_mvavg_stat['h']={}
      self.profiles_norm_mvavg_stat['v']={}
    # get lsf factor, beta@BSRT and energy from logging databse
    # get lsf correction factor, beta function and beam energy
    if ((self.beam) is not None) and ((self.beam).upper() in ['B1','B2']):
      # variable names
      bsrt_lsf_var = self.get_beta_lsf_variable_names()
      lsf_var={};beta_var={}
      (lsf_var['h'],lsf_var['v'],beta_var['h'],beta_var['v'],
           energy_var) = bsrt_lsf_var
      # data from database
      bsrt_lsf_db = self.get_beta_lsf_energy()
    if verbose:
      if force is False:
        print('... calculate statistical parameters.')
      elif force is True:
        print('... delete old data and recalculate statistical ' + 
            'parameters')
    for plane in 'hv':
      if verbose:
        print('... start plane %s'%plane.upper())
      if self.profiles_norm_avg_stat is None:
        self.profiles_norm_avg_stat[plane] = {}
      if self.profiles_norm_mvavg_stat is None:
        self.profiles_norm_mvavg_stat[plane] = {}
      for sl in self._set_slots(plane=plane,slot=slot):
        if verbose:
          print('... start slot %s'%sl)
        # 2) calculate/recalculate statistical parameters
        # initialize/delete old data
        self.profiles_norm_avg_stat[plane][sl] = []
        if self.nmvavg is not None:
          self.profiles_norm_mvavg_stat[plane][sl] = []
        for time_stamp in self.get_timestamps(slot=sl,plane=plane):
          # 1) estimate centroid with three different methods:
          # 1a) Gaussian fit (cent_gauss)
          # 1b) qGaussian fit (cent_qgauss)
          # 1c) center of gravity sum(x*w) (cent_stat)
          # 1d) median (cent_median)
          # 1e) 50 % of cummulative sum (cent_cumsum)
          # 1f) peak of distribution

          # 2) estimate of distribution width with three different 
          #    methods:
          # 2a) Gaussian fit (sigma_gauss)
          # 2b) qGaussian fit (sigma_qgauss)
          # 2c) statistical rms sum(x*w) (sigma_stat)
          # 2d) median absolute deviation = median(|x-median(x)|)
          # 2e) 26 % and 84% of cummulative sum (sigma_cumsum)

          # 3) calculate beam normalized emittance for sigma values
          # g) emittance for all statistical parameters in 2)

          # 4) Estimate of halo
          # 4y) sum over bins between 3 mm and 6 mm
          # 4z) entropie sum(p_k ln(p_k) ) where p_k is the bin height
          
          # 5) goodness of fit parameters:
          # 5h) calculate xi^2
          # 5i) calculate correlation matrix
          
          # average profile
          profs_norm_avg = self.get_profile_norm_avg(slot=sl,                  
                             time_stamp=time_stamp, plane=plane)
          # moving average profile
          if self.nmvavg is not None:
            profs_norm_mvavg = self.get_profile_norm_mvavg(slot=sl,                  
                             time_stamp=time_stamp, plane=plane)
          else:
            profs_norm_mvavg = None
          for pna,avgflag in zip([profs_norm_avg,profs_norm_mvavg],
                                 ['avg','mvavg']):
            if avgflag == 'mvavg' and self.nmvavg is None:
              break
            # 1) data is already calculated + force = False-> go to next slot
            if ((avgflag == 'avg') and 
                (sl in self.profiles_norm_avg_stat[plane].keys()) and 
                (force is False)):
              continue
            if ((avgflag == 'mvavg') and 
                (sl in self.profiles_norm_mvavg_stat[plane].keys()) and 
                (force is False)):
              continue
            # first: set some initial fit paramters for Gaussian and qGaussian fit
            # background
            if self._rm_bg is False:
              cmin,cmax = [0,0.1]
            else:
              cmin,cmax = [-0.1,0.1]
            # sigma used for fit
            nsample = len(pna['amp'])
            if avgflag == 'mvavg': # take errors into account for moving average fit
              ampstd = pna['ampstd']
            else:
              ampstd = np.ones(nsample)
            if fitsplit != 'full':
              cent_median  = tb.median(pna['pos'],pna['amp'])
              idx = int(round(np.where( pna['pos'] == cent_median)[0]))
              if fitsplit == 'left':
                didx = (nsample-idx)/3
                ampstd = ampstd*np.array([1]*(idx+didx)+[100]*(nsample - (idx+didx))) 
              if fitsplit == 'right':
                didx = 2*idx/3
                ampstd = ampstd*np.array([100]*(didx)+[1]*(nsample - didx))
            # q for q-Gaussian fit
            if plane == 'h': # first guess for q-Gaussian
              q0 = 1.1
            else:
              q0 = 0.9
            # a) Gaussian fit
            # assume initial values of
            # c=0,a=1,mu=0,sig=2
            # fit function =
            #   c+a*1/sqrt(2*sig**2*pi)*exp(((x-mu)/sig)**2/2)
            # c is also an estimate for the background
            # a compensate for c0 so that the integral over the
            # the distribution is equal to 1. If c is small, a should be
            # close to 1
            # -- fit is done without errors for profs_norm_avg 
            #    as the std over this few measurements (e.g. 3) 
            #    is to noisy.
            # -- limits for c,a:
            #    background is usually around 0.008 
            #        -> limit c to [-0.1,0.1]
            #        -> limit a to [0,2]
            #    allow for negative values if background is subtracted
            #    as background values could then also become negative
            #    as the average over the whole fill for each bunch is
            #    subtracted
            # -- limits for cent,sigma:
            #    centroid shouldn't exceed maximum range of profile, which
            #    is approximately [-8,8]
            try:
              p,pcov_g = curve_fit(f=tb.gauss_pdf,xdata=pna['pos'],
                                 ydata=pna['amp'],p0=[0,1,0,2],
                                 sigma=ampstd,
                                 bounds=([cmin,0,-8,0],
                                         [cmax,2,8,16]))
              profs_norm_gauss = tb.gauss_pdf(pna['pos'],*p)
              # h) calculate xi-squared
              # - estimate noise by the standard deviation in each bin
              #   over self.nmvavg measurements.
              # - normalize by the nbins - nparam -> xisq_g in [0,1]
              xisq_g = (((pna['amp']-profs_norm_gauss)/
                       ampstd)**2).sum()
              # nparam = 4 for Gaussian distribution
              xisq_g = xisq_g/(len(pna['amp'])-4)
              # i) calculate the correlation matrix
              # corr(x,y) = sig_xy/(sigx*sigy)
              #           = sqrt(pcov(x,y)/(pcov(x,x)*pcov(y,y)))
              pcorr_g = np.array([ [ 
                         pcov_g[i,j]/np.sqrt(pcov_g[i,i]*pcov_g[j,j]) 
                         for j in range(len(p))] for i in range(len(p))])
              # error on p
              psig = [ np.sqrt(pcov_g[i,i]) for i in range(len(p)) ]
              c_gauss, a_gauss = p[0], p[1]
              cent_gauss, sigma_gauss = p[2],p[3]
              c_gauss_err, a_gauss_err = psig[0], psig[1]
              cent_gauss_err, sigma_gauss_err = psig[2],psig[3]
            except RuntimeError:
              if verbose:
                print("WARNING: Gaussian fit failed for " +
                      "plane %s, slotID %s and "%(plane,sl) +
                      "timestamp %s"%(time_stamp))
              c_gauss, a_gauss, cent_gauss, sigma_gauss = 0,0,0,0
              c_gauss_err, a_gauss_err = 0,0
              cent_gauss_err, sigma_gauss_err = 0,0
              pass
            # b) qGaussian fit, note that 1 < q < 3 -> constraint fit
            #    parameters
            # fit function =
            #   c+a*sqrt(beta)/cq*eq(-beta*(x-mu)**2)
            # -- fit is done without errors for profs_norm_avg 
            #    as the std over this few measurements (e.g. 3) 
            #    is to noisy.
            # -- cq is also an estimate for the background
            #    a compensate for c0 so that the integral over the
            #    the distribution is equal to 1. If c0 is small, a should be
            #    close to 1
            # -- limits for c,a:
            #    background is usually around 0.008 
            #        -> limit c to [-0.1,0.1]
            #        -> limit a to [0,2]
            #    allow for negative values if background is subtracted
            #    as background values could then also become negative
            #    as the average over the whole fill for each bunch is
            #    subtracted
            # -- limits for cent:
            #    centroid shouldn't exceed maximum range of profile, which
            #    is approximately [-8,8]
            try:
              p,pcov_qg = curve_fit(f=tb.qgauss_pdf,xdata=pna['pos'],
                               ydata=pna['amp'],sigma=ampstd,
                               p0=[1.e-3,1,q0,0,1],
                               bounds=([cmin,0,1.e-2,-6,1.e-2],
                               [cmax,2,5/3.,6,4.0]))
              profs_norm_qgauss = tb.qgauss_pdf(pna['pos'],*p)
              # h) calculate xi-squared
              # - estimate noise by the standard deviation in each bin
              #   over self.nmvavg measurements.
              # - normalize by the nbins - nparam -> xisq_qg in [0,1]
              xisq_qg = (((pna['amp']-profs_norm_qgauss)/
                       ampstd)**2).sum()
              # nparam = 4 for Gaussian distribution
              xisq_qg = xisq_qg/(len(pna['amp'])-5)
              # i) calculate the correlation matrix
              # corr(x,y) = sig_xy/(sigx*sigy)
              #           = sqrt(pcov(x,y)/(pcov(x,x)*pcov(y,y)))
              pcorr_qg = np.array([ [ 
                         pcov_qg[i,j]/np.sqrt(pcov_qg[i,i]*pcov_qg[j,j]) 
                         for j in range(len(p))] for i in range(len(p))])
              # error on p
              psig = [ np.sqrt(pcov_qg[i,i]) for i in range(len(p)) ]
              pcov_beta_q = pcov_qg[2,4]
              c_qgauss, acq_qgauss = p[0], p[1]
              # normalizaton factor cq is absorbed in acq = p[1]
              # -> extract it again to have a = acq/cq with a approximately
              #    one and modeling only the background as also for Gaussian
              q_qgauss = p[2]
              cq_qgauss = tb.qgauss_cq(q_qgauss)
              a_qgauss = acq_qgauss/cq_qgauss
              cent_qgauss, beta_qgauss = p[3], p[4]
              c_qgauss_err, acq_qgauss_err = psig[0], psig[1]
              a_qgauss_err = acq_qgauss_err/np.abs(cq_qgauss) # assume cq is constant, neglect q-dependence
              q_qgauss_err = psig[2]
              cent_qgauss_err, beta_qgauss_err = psig[3],psig[4]
              sigma_qgauss = tb.qgauss_sigma(q_qgauss,beta_qgauss)
              if (q_qgauss < 5/3.):
                var_qgauss_err = ( (1/(4*beta_qgauss**3*(5-3*q_qgauss))
                     *beta_qgauss_err**2) +
                  (9/(4*beta_qgauss*(5-3*q_qgauss)**3)
                     *q_qgauss_err**2) +
                  (3/(4*beta_qgauss**2*(5-3*q_qgauss)**2))*pcov_beta_q )
                sigma_qgauss_err = np.sqrt(var_qgauss_err)
              elif (q_qgauss >= 5/3.) & (q_qgauss < 2):
                sigma_qgauss_err = np.inf
              else:
                sigma_qgauss_err = 0
            except RuntimeError,RuntimeWarning:
              if verbose:
                print("WARNING: q-Gaussian fit failed for " +
                      "plane %s, slotID %s and "%(plane,sl) +
                      "timestamp %s"%(time_stamp))
              c_qgauss, a_qgauss, q_qgauss = 0,0,0
              cent_qgauss, sigma_qgauss = 0,0
              c_qgauss_err, a_qgauss_err, q_qgauss_err = 0,0,0
              cq_qgauss,acq_qgauss,acq_qgauss_err = 0,0,0
              cent_qgauss_err, sigma_qgauss_err = 0,0
              pass
            x,y = pna['pos'],pna['amp']
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
            # average the first/last bgnavg bins
            bg_avg_left = y[:self.bgnavg].mean()
            bg_avg_right = y[-self.bgnavg:].mean()
            bg_avg = (bg_avg_left+bg_avg_right)/2
            # g) emittance
            # if no beam is given
            (sigma_beam,emit_gauss,emit_qgauss,emit_stat,emit_median,
            emit_cumsum_32,emit_cumsum_68) = (0,)*7
            (beta,lsf,energy) = (0,)*3
            # if beam given convert picture sigma to beam sigma and then 
            # to normalized emittance
            if (beam is not None) and (beam.upper() in ['B1','B2']):
              # get lsf,beta and energy with time stamp closest to
              # time_stamp and smaller than time_stamp
              bsrt_lsf = {}
              for k in lsf_var[plane],beta_var[plane],energy_var:
                idx = np.where(time_stamp-bsrt_lsf_db[k][0]>=0.)[0][-1] 
                bsrt_lsf[k] = bsrt_lsf_db[k][1][idx]
              # convert sigma to emittance
              emit_norm = []
              beta  = bsrt_lsf[beta_var[plane]]
              lsf = bsrt_lsf[lsf_var[plane]]
              energy = bsrt_lsf[energy_var]
              for sigma in [sigma_gauss,sigma_qgauss,sigma_stat,
                sigma_median,sigma_cumsum_32,sigma_cumsum_68]:
                # geometric emittance
                emit_geom = ((sigma**2 - lsf**2)/beta) 
                emit_norm.append(pytimber.toolbox.emitnorm(emit_geom,
                                energy,m0=938.272046))
              [emit_gauss,emit_qgauss,emit_stat,emit_median,
                            emit_cumsum_32,emit_cumsum_68] = emit_norm
              # use sigma to scale later x [mm] to x [sigma]
              sigma_beam = np.sqrt(sigma_gauss**2 - lsf**2)
            # estimate of halo
            # y) sum bins between x_min mm and x_max mm
            #    sum bins between -x_min mm and -x_max mm (remember bump  
            #      on right side of profile, so better do individual sums)
            x_min,x_max = 3,6
            mask_r = np.logical_and(pna['pos'] < x_max,
                              pna['pos'] > x_min)
            mask_l = np.logical_and(pna['pos'] < -x_min,
                              pna['pos'] > -x_max)
            sum_bin_left = pna['amp'][mask_l].sum() 
            sum_bin_right = pna['amp'][mask_r].sum() 
            # z) calculate the entropy and divide by the total entropy
            #      entropie = -sum_(k=0)^(nbins) p_k ln(p_k)
            #    where p_k is the bin height
            #    Now normalize in addition to the maximum entropie = 
            #    entropie for which all bins have the same height
            #      entropie_max = -sum_(k=0)^(nbins) 1/nbins * ln(1/nbins)
            #                   = -nbins*1/nbins*ln(1/nbins)
            #                   = ln(nbins)
            #    => entropie_norm = -(1/ln(nbins))*
            #                               sum_(k=0)^(nbins) p_k ln(p_k)
            #    For the case with removed background the bin amplitudes
            #    can be negative, so this only works if the background is
            #    not removed.
            #    Even without background removal bins can be negative
            #    -> take as quick fix the absolute value
            if self._rm_bg is False:
              nbins = len(pna['pos'])
              entropie = -(1/np.log(nbins))*(pna['amp']*
                             np.log(np.abs(pna['amp']))).sum()
            else:
              entropie = 0
            data = (time_stamp,
                # Gaussian fit
                c_gauss, a_gauss, cent_gauss, sigma_gauss,
                pcov_g, pcorr_g, xisq_g,
                c_gauss_err, a_gauss_err, cent_gauss_err, sigma_gauss_err,
                # q-Gaussian fit
                c_qgauss, a_qgauss, q_qgauss, cent_qgauss, beta_qgauss,
                sigma_qgauss,cq_qgauss,acq_qgauss,
                pcov_qg, pcorr_qg, xisq_qg,
                c_qgauss_err, a_qgauss_err, q_qgauss_err, cent_qgauss_err,
                beta_qgauss_err, sigma_qgauss_err,
                acq_qgauss_err,
                # statistical parameters
                cent_stat, sigma_stat, cent_median, mad, sigma_median,
                cent_cumsum, sigma_cumsum_32, sigma_cumsum_68,
                cent_peak,
                # beta,lsf,energy for conversion
                beta,lsf,energy,
                # beam sigma from Gaussian fit
                sigma_beam,
                # normalized emittance
                emit_gauss,emit_qgauss,emit_stat,emit_median,
                emit_cumsum_32,emit_cumsum_68,
                # background estimate
                bg_avg_left,bg_avg_right,bg_avg,
                # estimate of halo
                sum_bin_left,sum_bin_right,entropie
                )
            if avgflag == 'avg': 
              self.profiles_norm_avg_stat[plane][sl].append(data)
            if avgflag == 'mvavg' and self.nmvavg is not None: 
              self.profiles_norm_mvavg_stat[plane][sl].append(data)
    # convert to a structured array
    ftype=[('time_stamp',int),('c_gauss',float),('a_gauss',float),
           ('cent_gauss',float),('sigma_gauss',float),
           ('pcov_gauss',float,(4,4)),('pcorr_gauss',float,(4,4)),
           ('xisq_gauss',float),('c_gauss_err',float),
           ('a_gauss_err',float),
           ('cent_gauss_err',float),('sigma_gauss_err',float),
           ('c_qgauss',float),('a_qgauss',float),('q_qgauss',float),
           ('cent_qgauss',float),('beta_qgauss',float),
           ('sigma_qgauss',float),('cq_qgauss',float),
           ('acq_qgauss',float),
           ('pcov_qgauss',float,(5,5)),('pcorr_qgauss',float,(5,5)),
           ('xisq_qgauss',float),('c_qgauss_err',float),
           ('a_qgauss_err',float),('q_qgauss_err',float),
           ('cent_qgauss_err',float),('beta_qgauss_err',float),
           ('sigma_qgauss_err',float),('acq_qgauss_err',float),
           ('cent_stat',float),
           ('sigma_stat',float),('cent_median',float),('mad',float),
           ('sigma_median',float),('cent_cumsum',float),
           ('sigma_cumsum_32',float),('sigma_cumsum_68',float),
           ('cent_peak',float),('beta',float),('lsf',float),
           ('energy',float),('sigma_beam',float),
           ('emit_gauss',float),('emit_qgauss',float),
           ('emit_stat',float),('emit_median',float),
           ('emit_cumsum_32',float),('emit_cumsum_68',float),
           ('bg_avg_left',float),('bg_avg_right',float),
           ('bg_avg',float),
           ('sum_bin_left',float),('sum_bin_right',float),
           ('entropie',float)
           ]
    for plane in ['h','v']:
      for k in self.profiles_norm_avg_stat[plane].keys():
        self.profiles_norm_avg_stat[plane][k] = np.array(
          self.profiles_norm_avg_stat[plane][k], dtype=ftype)
    # variable for statistical variable names
    self.profiles_norm_avg_stat_var = [ ftype[k][0] for k in xrange(len(ftype)) ]
    if self.nmvavg is not None:
      for plane in ['h','v']:
        for k in self.profiles_norm_avg_stat[plane].keys():
          self.profiles_norm_mvavg_stat[plane][k] = np.array(
            self.profiles_norm_mvavg_stat[plane][k], dtype=ftype)
    # variable for statistical variable names
      self.profiles_norm_mvavg_stat_var = [ ftype[k][0] for k in xrange(len(ftype)) ]
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
  def _get_range_timestamps(self):
    """
    Returns:
    --------
    (t1,t2) = minimum/maximum timestamps of all profiles in unix
              time [ns]
    """
    lmin,lmax = [[],[]]
    for pl in self.profiles.keys():
      for sl in self.profiles[pl].keys():
        if len(self.profiles[pl][sl]['time_stamp']) != 0:
            lmin.append((self.profiles[pl][sl]['time_stamp']).min())
            lmax.append((self.profiles[pl][sl]['time_stamp']).max())
    return (np.min(lmin),np.max(lmax))
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

    Parameter:
    ----------
    slot : slot number
    plane : plane
    time_stamp :
       None : take first time stamp
       float : one timestamp in ns
       [t1,t2] : range of time stamps including t1,t2 (closed interval)
    """
    ts = self.get_timestamps(slot=slot,plane=plane)
    # convert single time stamp to [t1,t2] = [time_stamp,time_stamp]
    if not hasattr(time_stamp,"__iter__"):
      if time_stamp is None:
        time_stamp = ts[0]
      if time_stamp not in ts:
        raise ValueError('time stamp %s '%(time_stamp) + 'is not in ' +
            'list of time stamps for slot %s and plane %s'%(slot,plane))
      t1,t2 = time_stamp,time_stamp
    # interval of time stamps
    elif len(time_stamp) == 2:
      t1,t2 = time_stamp
    else:
      raise ValueError('Only None, single time stamp or interval of ' +
              'time stamps allowed for time_stamp')
    # extract data
    pn = self.profiles_norm[plane][slot]
    mask = np.logical_and(pn['time_stamp']>=t1, pn['time_stamp']<=t2)
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
    try:
      mask = (self.profiles_norm_avg[plane][slot]['time_stamp'] 
                == time_stamp)
    except TypeError:
      print('in get_profile_norm_avg: Data could not be extracted! Have you run ' +
            'self.norm() to normalize profiles?')
      return
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm_avg[plane][slot][mask][0]
    else:
      return self.profiles_norm_avg[plane][slot][mask]
  def get_profile_norm_mvavg(self, slot = None, time_stamp = None,
                           plane = 'h'):                                    
    """
    returns moving averag of normalized profile for slot *slot*, time stamp
    *time_stamp* as unix time [ns] and plane *plane*.
    """
    try:
      mask = (self.profiles_norm_mvavg[plane][slot]['time_stamp'] 
                == time_stamp)
    except TypeError:
      print('in get_profile_norm_mvavg: Data could not be extracted! Have you run ' +
            'self.norm() to normalize profiles?')
      return
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm_mvavg[plane][slot][mask][0]
    else:
      return self.profiles_norm_mvavg[plane][slot][mask]
  def get_profile_avg_stat(self, slot = None, time_stamp = None, plane = 'h'):
    """
    get profile data for slot *slot*, time stamp *time_stamp* as
    unix time [ns] and plane *plane*
    """
    mask = self.profiles_norm_avg_stat[plane][slot]['time_stamp'] == time_stamp
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm_avg_stat[plane][slot][mask][0]
    else:
      return self.profiles_norm_avg_stat[plane][slot][mask]
  def get_profile_mvavg_stat(self, slot = None, time_stamp = None, plane = 'h'):
    """
    get profile data for slot *slot*, time stamp *time_stamp* as
    unix time [ns] and plane *plane*
    """
    mask = self.profiles_norm_mvavg_stat[plane][slot]['time_stamp'] == time_stamp
    if len(np.where(mask==True)[0]) == 1:
      return self.profiles_norm_mvavg_stat[plane][slot][mask][0]
    else:
      return self.profiles_norm_mvavg_stat[plane][slot][mask]
  def _plot_profile(self, slot = None, time_stamp = None, plane = 'h',
                    xaxis = 'mm', norm = True, mvavg = False,
                    smooth = 0.025, verbose = False):
    """
    Plot all profiles for specific slot and time. Plot title displays 
    'Europe/Zurich' time.

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plane : plane of profile, either 'h' or 'v'
    norm : if norm = false raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    smooth: parameter to smooth (only!) average profiles with lowess.
            smooth = 0 or smooth = None: no smoothing
            smooth > 0: 'frac' parameter in lowess (Between 0 and 1.
            The fraction of the data used when estimating each y-value.)
    verbose : verbose option for additional output

    Returns:
    --------
    check_plot : bool, flag if profile plot failed
                 (used for mk_profile_video)
    """
    # flag if profile plot failed (used for mk_profile_video)
    check_plot = True
    xscale = 1 #scaling for mm or sigma for x-axis, is reset in case of sigma
    # select profile for slot and time stamp
    # normalized profiles
    if norm:
      profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                    plane=plane)
      if mvavg is True:
        profs_avg = self.get_profile_norm_mvavg(slot=slot,
                          time_stamp=time_stamp,plane=plane)
      else:
        profs_avg = self.get_profile_norm_avg(slot=slot,
                          time_stamp=time_stamp,plane=plane)
      if self.profiles_norm_avg_stat is not None:
        if mvavg is True:
          stat_aux = self.get_profile_mvavg_stat(slot=slot,
                     time_stamp=time_stamp,plane=plane)
        else:
          stat_aux = self.get_profile_avg_stat(slot=slot,
                     time_stamp=time_stamp,plane=plane)
        # Gaussian fit
        c_gauss, a_gauss = stat_aux['c_gauss'], stat_aux['a_gauss']
        cent_gauss = stat_aux['cent_gauss']
        sigma_gauss = stat_aux['sigma_gauss']
        sigma_beam  = stat_aux['sigma_beam']
        # q-Gaussian fit
        c_qgauss, acq_qgauss = stat_aux['c_qgauss'], stat_aux['acq_qgauss']
        q_qgauss = stat_aux['q_qgauss']
        cent_qgauss = stat_aux['cent_qgauss']
        beta_qgauss    = stat_aux['beta_qgauss']
        if xaxis == 'sigma':
          if sigma_beam !=0:
            xscale = 1/sigma_beam
          else:
            xscale = 1/sigma_gauss
    # raw data profile
    else:
      profs = self.get_profile(slot=slot,time_stamp=time_stamp,
                               plane=plane)
    for i in xrange(len(profs)):
      try:
        pl.plot(xscale*profs[i]['pos'],profs[i]['amp'],
                label='profile %s'%(i+1),linestyle='-',
                color=profile_colors.values()[i])
        if norm:
          pl.ylabel(r'probability (integral normalized to 1) [a.u.]')
          pl.ylim(1.e-3,1.0)
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
      if (smooth > 0) and (smooth < 1):
        prof_smooth = lowess(endog=profs_avg['amp'],
            exog=profs_avg['pos'],frac=smooth,it=3,delta=0,
            is_sorted=True,missing='drop')
        profs_avg['pos'] = prof_smooth[:,0]
        profs_avg['amp'] = prof_smooth[:,1]
      # plot average over profiles
      pl.plot(profs_avg['pos']*xscale,profs_avg['amp'],
              label = 'average profile',color='k',linestyle='-')
      if self.profiles_norm_avg_stat is not None:
        # plot Gaussian fit
        pl.plot(profs_avg['pos']*xscale,tb.gauss_pdf(profs_avg['pos'],
                c_gauss,a_gauss,cent_gauss,sigma_gauss),
                color=fit_colors['Red'],
                linestyle='--',linewidth=1,label='Gaussian fit')
        # plot q-Gaussian fit
        pl.plot(profs_avg['pos']*xscale,tb.qgauss_pdf(profs_avg['pos'],
                c_qgauss,acq_qgauss,q_qgauss,cent_qgauss,beta_qgauss),
                color=fit_colors['DarkRed'],
                linestyle='-',linewidth=1,label='q-Gaussian fit')
        if xaxis == 'sigma':
          pl.gca().set_xlim(-6,6)
          if self.db is not None:
            lbl=(r'position [$\sigma_{\rm{Beam}}$], '+
                 r'$\sigma_{\rm{Beam}}$ = %2.2f mm'%sigma_beam)
          else:
            lbl=(r'position [$\sigma_{\rm{Prof}}$], '+
                 r'$\sigma_{\rm{Prof}}$ = %2.2f mm'%sigma_gauss)
          pl.gca().set_xlabel(lbl,fontsize=10)
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
                        mvavg=False,smooth=None,plane = 'h', verbose = False):
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
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    smooth: parameter to smooth (only!) average profiles with lowess.
            smooth = 0 or smooth = None: no smoothing
            smooth > 0: 'frac' parameter in lowess (Between 0 and 1.
            The fraction of the data used when estimating each y-value.)
    verbose : verbose option for additional output
    """
    self._plot_profile(slot=slot,time_stamp=time_stamp,plane=plane,
                       norm=True,mvavg=mvavg,smooth=smooth,verbose=verbose)
  def plot_cumsum(self, slot = None, time_stamp = None,xaxis='mm', 
                  plane = 'h', mvavg = False, verbose = False):
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
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plane : plane of profile, either 'h' or 'v'
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    verbose : verbose option for additional output

    Returns:
    --------
    check_plot : bool, flag if profile plot failed
                 (used for mk_profile_video)
    """
    # scaling for mm or sigma for x-axis, is reset in case of sigma
    xscale = 1
    sta = None
    # select profile for slot and time stamp
    profs = self.get_profile_norm(slot=slot,time_stamp=time_stamp,
                                  plane=plane)
    if mvavg is True:
      profs_avg = self.get_profile_norm_mvavg(slot=slot,
                       time_stamp=time_stamp,plane=plane)
      sta = self.get_profile_mvavg_stat(slot=slot,
                   time_stamp=time_stamp,plane=plane)
    else:
      profs_avg = self.get_profile_norm_avg(slot=slot,
                       time_stamp=time_stamp,plane=plane)
      sta = self.get_profile_avg_stat(slot=slot,
                   time_stamp=time_stamp,plane=plane)
    if xaxis == 'sigma' and sta is not None:
      sigma_gauss = sta['sigma_gauss']
      sigma_beam  = sta['sigma_beam']
      if sigma_beam != 0:
        xscale = 1/sigma_beam
      else:
        xscale = 1/sigma_gauss
    # flag if profile plot failed
    check_plot = True
    # individual profiles
    for i in xrange(len(profs)):
      try:
        dx = profs[i]['pos'][1]-profs[i]['pos'][0]
        pl.plot(profs[i]['pos']*xscale,(dx*profs[i]['amp']).cumsum(),
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
      dx = profs_avg['pos'][1]-profs_avg['pos'][0]
      pl.plot(profs_avg['pos']*xscale,(dx*profs_avg['amp']).cumsum(),
            label = 'average profile',color='k',linestyle='-')
      pl.xlabel('position [mm]')
      if self.profiles_norm_avg_stat is not None:
        pl.plot(profs_avg['pos']*xscale,(dx*tb.gauss_pdf(profs_avg['pos'],
                sta['c_gauss'],sta['a_gauss'],sta['cent_gauss'],
                sta['sigma_gauss'])).cumsum(),
                label = 'Gaussian fit',color = fit_colors['Red'],
                linestyle = '--')
        pl.plot(profs_avg['pos']*xscale,(dx*tb.qgauss_pdf(profs_avg['pos'],
                sta['c_qgauss'],sta['acq_qgauss'],sta['q_qgauss'],
                sta['cent_qgauss'],sta['beta_qgauss'])).cumsum(),
                label = 'q-Gaussian fit', 
                color = fit_colors['DarkRed'], linestyle = '-')
        if xaxis == 'sigma':
          pl.gca().set_xlim(-6,6)
          if self.db is not None:
            lbl=(r'position [$\sigma_{\rm{Beam}}$], '+
                 r'$\sigma_{\rm{Beam}}$ = %2.2f mm'%sigma_beam)
          else:
            lbl=(r'position [$\sigma_{\rm{Prof}}$], '+
                 r'$\sigma_{\rm{Prof}}$ = %2.2f mm'%sigma_gauss)
          pl.gca().set_xlabel(lbl,fontsize=10)
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
  def _plot_residual_ratio(self, flag, flagprof, slot = None,
          time_stamp = None, xaxis='mm',slot_ref = None,
          time_stamp_ref = None, 
          plane = 'h', mvavg = False, errbar = False, smooth = 0.025,
          verbose = False):
    """
    Plot residual or ratio of normalized profiles for a specific slot 
    and time. Plot title displays 'Europe/Zurich' time.

    Parameters:
    -----------
    flag : flag = 'residual' plot the residual
           flag = 'ratio' plot the ratio
    flagprof: flagprof = 'all' plot all profiles + average
              flagprof = 'avg' plot only average profile
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    slot_ref: slot number of reference bunch, if slot_ref = None use
              *slot* as reference slot
    time_stamp_ref : reference time stamp in unix time [ns], if None
                     use first time stamp of slot *slot_ref*
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plane : plane of profile, either 'h' or 'v'
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    errbar : if errbar = True show error bars for residual
             if errbar = False do not show error bars for residual
    smooth: parameter to smooth (only!) the average profile with lowess.
            smooth = 0 or None: no smoothing
            smooth > 0: 'frac' parameter in lowess (Between 0 and 1.
            The fraction of the data used when estimating each y-value.)
    verbose : verbose option for additional output

    Returns:
    --------
    check_plot : bool, flag if profile plot failed 
                 (used for mk_profile_video)
    """
    xscale = 1 #scaling for mm or sigma for x-axis, is reset in case of sigma
    # initialize for default values = None
    if slot_ref is None:
      slot_ref = slot
    if time_stamp_ref is None:
      time_stamp_ref = self.get_timestamps(slot=slot_ref,plane=plane)[0]
    # check that timestamps are correct
    if time_stamp not in self.get_timestamps(slot=slot,plane=plane):
      print('ERROR: timestamp %s for slot %s not found!'%(time_stamp,slot))
      return
    if time_stamp_ref not in self.get_timestamps(slot=slot_ref,plane=plane):
      print('ERROR: timestamp %s for slot %s not found!'%(time_stamp_ref,slot_ref))
      return
    if smooth is None:
      smooth = 0
    # get statistical parameters
    if self.profiles_norm_avg_stat is not None:
      if mvavg is True:
        sta = self.get_profile_mvavg_stat(slot=slot,
                   time_stamp=time_stamp,plane=plane)
      else:
        sta = self.get_profile_avg_stat(slot=slot,
                       time_stamp=time_stamp,plane=plane)
      if xaxis == 'sigma':
        sigma_gauss = sta['sigma_gauss']
        sigma_beam  = sta['sigma_beam']
        if sigma_beam !=0:
          xscale = 1/sigma_beam
        else:
          xscale = 1/sigma_gauss
    # select profile for slot and time stamp
    # individual profiles
    if flagprof == 'all':
      profs         = self.get_profile_norm(slot=slot,
                             time_stamp=time_stamp, plane=plane)
      profs_ref     = self.get_profile_norm(slot=slot_ref,
                             time_stamp=time_stamp_ref, plane=plane)
    # average profiles
    if mvavg is True:
      profs_avg     = self.get_profile_norm_mvavg(slot=slot,
                             time_stamp=time_stamp, plane=plane)
      profs_ref_avg = self.get_profile_norm_mvavg(slot=slot_ref,
                           time_stamp=time_stamp_ref, plane=plane)
    else:
      profs_avg     = self.get_profile_norm_avg(slot=slot,
                             time_stamp=time_stamp, plane=plane)
      profs_ref_avg = self.get_profile_norm_avg(slot=slot_ref,
                           time_stamp=time_stamp_ref, plane=plane)
    # smooth profiles
    if (smooth < 0) or (smooth >= 1):
      print('ERROR in _plot_residual_ratio: smooth parameters has to' +
      'be between 0 and 1! smooth = %s'%smooth)
    if (smooth > 0) and (smooth < 1):
      for pp in [profs_avg,profs_ref_avg]:
        prof_smooth = lowess(endog=pp['amp'],
            exog=pp['pos'],frac=smooth,it=3,delta=0,
            is_sorted=True,missing='drop')
        pp['pos'] = prof_smooth[:,0]
        pp['amp'] = prof_smooth[:,1]
    # take only values for which x-range of profile coincides
    xval = list(set(profs_avg['pos']).intersection(set(profs_ref_avg['pos'])))
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
            pl.plot(profs[i]['pos'][mask]*xscale,
                    profs[i]['amp'][mask]-profs_ref[i]['amp'][mask_ref],
                    label='profile %s'%(i+1))
          if flag == 'ratio':
            pl.plot(profs[i]['pos'][mask]*xscale,
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
    # shorten the names
    pos_avg = profs_avg['pos'][mask]
    amp_avg = profs_avg['amp'][mask]
    sig_avg = profs_avg['amperr'][mask]
    amp_ref = profs_ref_avg['amp'][mask_ref]
    sig_ref = profs_ref_avg['amperr'][mask_ref]
    if flag == 'residual':
      try:
        # plot error of residual as one sigma envelope
        # res_error = sqrt(sig(profs_avg)**2+sig(profs_avg_ref)**2)
        pl.plot(pos_avg*xscale,amp_avg-amp_ref,
                label = 'average profile',color='k',linestyle='-')
        if errbar:
          sig = np.sqrt(sig_avg**2+sig_ref**2)
          res = amp_avg - amp_ref
          pl.fill_between(profs_avg['pos'][mask]*xscale,res-sig,res+sig,
                  alpha=0.2,color='k')
          # Gaussian fit
          c_gauss, a_gauss = sta['c_gauss'], sta['a_gauss']
          cent_gauss = sta['cent_gauss']
          sigma_gauss    = sta['sigma_gauss']
          # q-Gaussian fit
          c_qgauss, a_qgauss = sta['c_qgauss'], sta['a_qgauss']
          q_qgauss = sta['q_qgauss']
          cent_qgauss = sta['cent_qgauss']
          beta_qgauss    = sta['sigma_qgauss']
          pl.plot(pos_avg*xscale,amp_avg-
                  tb.gauss_pdf(pos_avg,sta['c_gauss'],
                    sta['a_gauss'],sta['cent_gauss'],
                    sta['sigma_gauss']),
                  label = 'Gaussian fit',
                  color=fit_colors['Red'],linestyle='--')
          pl.plot(pos_avg*xscale,amp_avg-
                  tb.qgauss_pdf(pos_avg,sta['c_qgauss'],
                    sta['acq_qgauss'],sta['q_qgauss'],
                    sta['cent_qgauss'],sta['beta_qgauss']), 
                  label = 'q-Gaussian fit',
                  color=fit_colors['DarkRed'],linestyle='-')
      except (ValueError,IndexError):
        check_plot = False
        pass
    if flag == 'ratio':
      try:
        pl.plot(pos_avg*xscale,amp_avg/amp_ref,
                label = 'average profile',color='k',linestyle='-')
        if errbar:
          rat = amp_avg/amp_ref
          sig = np.abs(rat)*np.sqrt((sig_avg/amp_avg)**2
                                    +(sig_ref/amp_ref)**2)
          pl.fill_between(pos_avg*xscale,rat-sig,rat+sig,
                  alpha=0.2,color='k')
      except (ValueError,IndexError):
        check_plot = False
        pass
    if check_plot:
      if xaxis == 'sigma' and self.profiles_norm_avg_stat is not None:
        pl.gca().set_xlim(-6,6)
        if self.db is not None:
          lbl=(r'position [$\sigma_{\rm{Beam}}$], '+
              r'$\sigma_{\rm{Beam}}$ = %2.2f mm'%sigma_beam)
        else:
          lbl=(r'position [$\sigma_{\rm{Prof}}$], '+
              r'$\sigma_{\rm{Prof}}$ = %2.2f mm'%sigma_gauss)
        pl.gca().set_xlabel(lbl,fontsize=10)
      else:
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
  def plot_all(self,slot = None, time_stamp = None, slot_ref = None,
               time_stamp_ref = None, xaxis = 'mm', plane = 'h', norm = True, 
               mvavg = False, errbar = True, smooth = None, log = True,
               verbose = False):
    """
    plot normalized or raw data profiles, cumulative distribution 
    function, residual and ratio in respect to reference distribution

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time [ns]
    slot_ref : reference slot, if None the same slot is used
    time_stamp_ref : time stamp in unix time [ns] of reference slot,
                     if None first time stamp is used
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plane : plane of profile, either 'h' or 'v'
    norm : if norm = False raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    smooth: parameter to smooth profiles with lowess.
            smooth = 0 or smooth = None: no smoothing
            smooth > 0: 'frac' parameter in lowess (Between 0 and 1.
            The fraction of the data used when estimating each y-value.)
            A good choice for smooth is 0.025.
    errbar : if errbar = True show error bars for residual
             if errbar = False do not show error bars for residual
    log: plot profile in log scale
    verbose : verbose option for additional output
    
    Returns:
    --------
    flaux : bool, flag to check if profile plots have failed
    """
    # check that options are compatible
    if ((mvavg is True) or (errbar is True)) and (norm is False):
      print('ERROR: norm must be True for options mvavg = True or ' +
            'errbar = True!')
      return
    pl.clf()
    if time_stamp is not None:
      ts = ld.dumpdate(t=time_stamp*1.e-9,
                       fmt='%Y-%m-%d %H:%M:%S',zone='cern')
    else:
      raise ValueError('You have to specify a time stamp in unix ' +
                       'time [ns]!')
    if slot_ref is None:
      slot_ref = slot
    if time_stamp_ref is None:
      time_stamp_ref = self.get_timestamps(slot=slot_ref,plane=plane)[0]
    ts_ref = ld.dumpdate(t=time_stamp_ref*1.e-9,
                           fmt='%Y-%m-%d %H:%M:%S',zone='cern')
    pl.suptitle('%s plane, slot %s - %s, '%(plane.upper(),slot,ts) +
                'ref slot %s - %s'%(slot_ref,ts_ref))
    # 1) profile plot
    pl.subplot(223)
    # flaux = flag for checking if profile plots have failed
    flaux = self._plot_profile(slot=slot,time_stamp=time_stamp,
                              plane=plane,norm=norm,smooth=smooth,
                              mvavg=mvavg,xaxis=xaxis,verbose=verbose)
    if norm:
      pl.gca().set_ylabel('probability [a.u.]')
    if log:
      pl.gca().set_yscale('log')
    # 2) cumulative sum
    pl.subplot(224)
    self.plot_cumsum(slot=slot,time_stamp=time_stamp,plane=plane,
                     xaxis=xaxis,verbose=verbose)
    pl.gca().set_ylabel('cumulative dist. [a.u.]')
#     3) residual, only average profile
    pl.subplot(221)
    self._plot_residual_ratio(flag='residual', flagprof='avg', 
           slot=slot, time_stamp=time_stamp, slot_ref=slot_ref,
           time_stamp_ref=time_stamp_ref, plane=plane, mvavg = mvavg,
           errbar=errbar, smooth=smooth,xaxis=xaxis,
           verbose=verbose)
    pl.gca().set_ylim(-0.07,0.07)
#     4) ratio
    pl.subplot(222)
    self._plot_residual_ratio(flag='ratio', flagprof='avg', 
           slot=slot, time_stamp=time_stamp, slot_ref=slot_ref,
           time_stamp_ref=time_stamp_ref, plane=plane, mvavg=mvavg, 
           errbar=errbar, smooth=smooth,xaxis=xaxis,
           verbose=verbose)
    pl.gca().set_ylim(-1,5)
    # remove subplot titles, shrink legend size and put it on top of
    # the subplot
    for i in xrange(4):
      pl.subplot(2,2,i+1)
      pl.gca().set_title('')
      if i == 2 or i == 3:
        h,l = pl.gca().get_legend_handles_labels()
        l[-4] = u'profile 1-%s'%l[-4].split()[-1]
        pl.gca().legend(h[-4:],l[-4:],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                        ncol=2, mode="expand", borderaxespad=0.,
                        fontsize=8)
      else:
        pl.gca().legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                        ncol=2, mode="expand", borderaxespad=0.,
                        fontsize=8)
    pl.tight_layout()
    pl.subplots_adjust(hspace=0.7,wspace=0.25,top=0.85,bottom=0.1)
    return flaux
  def mk_profile_video(self, slot = None, t1=None, t2=None,
                       slot_ref=None, norm = True, 
                       mvavg = True, errbar = True, smooth=None,
                       xaxis='mm',
                       plt_dir='BSRTprofile_gifs', delay=20, 
                       export=False,verbose=False):
    """
    Generates a video of the profiles of slot with *slot*

    Parameters:
    -----------
    slot : slot or list of slots of slots. If slot = None all slots
           are selected. In case of several slot, on video is 
           generated per slot
    slot_ref: reference slot number
           slot_ref = None: the first time stamp of the same slot *slot*
               is used as reference
           slot_ref = integer: slot number *slot_ref* is used as
               reference and the time stamps closest to the time stamps
               of slot *slot* are used for reference (note that the 
               timestamps are then not exactly the same as BSRT loops 
               through bunches when taking data)
    t1 : start time in unix time [ns], if None first time stamp for
         slot is used
    t2 : end time in unix time [ns], if None last time stamp for slot
         is used
    norm : if norm = false raw profiles are plotted
           if norm = True profiles are normalized to represent a 
                     probability distribution, explicitly:
                        norm_data = raw_data/(int(raw_data))
                     so that:
                        int(norm_data) = 1
    mvavg : if mvavg = False plot average profile for each time stamp
                       (self.profiles_norm_avg)
            if mvavg = True plots moving average for each time stamp
                       (self.profiles_norm_mvavg)
    errbar : if errbar = True show error bars for residual
             if errbar = False do not show error bars for residual
    smooth: parameter to smooth profiles with lowess.
            smooth = 0: no smoothing
            smooth > 0: 'frac' parameter in lowess (Between 0 and 1.
            The fraction of the data used when estimating each y-value.)
            Good values are achieved for smooth =0.025
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plt_dir : directory to save videos
    delay : option for convert to define delay between pictures
    export : If True do not delete png files
    verbose : verbose option for additional output
    """
    tmpdir = os.path.join(plt_dir,'tmp_bsrtprofiles')
    # dictionary to store failed profiles
    check_plot = {}
    for plane in ['h','v']:
      # set slot and plane, initialize variables
      check_plot[plane] = {}
      # generate the figure and subplot
      for slot in self._set_slots(plane=plane,slot=slot):
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
        if (len(time_stamps) == 0):                                                 
          print('WARNING: no time stamps found for slot %s, '%slot +
                'plane %s and (t1,t2)=(%s,%s)'%(plane,t1,t2))
          continue       
        if slot_ref is None:
          time_stamp_ref = time_stamps[0]
        elif slot_ref in self.get_slots():
          time_stamps_ref = self.get_timestamps(slot=slot_ref,
                                                plane=plane)
          time_stamps_ref = time_stamps_ref[(time_stamps_ref >= t1) &
                                  (time_stamps_ref <= t2)]
          if (len(time_stamps_ref) == 0) and verbose:
            print('WARNING: no time stamps found for reference ' +
                  'slot %s, '%slot +
                  'plane %s and (t1,t2)=(%s,%s)!'%(plane,t1,t2) +
                  'No reference used!')
        else:
          if slot_ref is None:
            slot_ref = slot
          print('ERROR: slot number %s not found!'%slot_ref) 
          return
        pngcount = 0
        if verbose: 
          print( '... total number of profiles %s'%(len(time_stamps)))
        # generate png of profiles for each timestamps
        for time_stamp in time_stamps:
          pl.close('all')
          # find closest time stamp for reference bunch
          if slot_ref is not None:
            idx = np.argmin(np.abs(time_stamp-time_stamps_ref))
            time_stamp_ref = time_stamps_ref[idx]
          pl.clf()
          flaux = self.plot_all(slot=slot,time_stamp=time_stamp,
                      slot_ref=slot_ref,time_stamp_ref=time_stamp_ref,
                      plane=plane,norm=norm,mvavg=mvavg,errbar=errbar,
                      smooth=smooth,xaxis=xaxis)
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
  def plot_all_ipac17(self,slot = None, time_stamp = None, 
               time_stamp_ref = None, xaxis = 'sigma', plane = 'h',
               resavg=False,resfit=True,verbose = False):
    """
    simplified version of plot_all for IPAC. Takes as reference bunch
    the same bunch.
    Assumes the following plot options in plot_all:
      norm = True, smooth = 0, mvavg = True, errbar = True,
      log = True

    Parameters:
    -----------
    slot : slot number
    time_stamp : time stamp in unix time [ns], if None take last time stamp
    time_stamp_ref : time stamp in unix time [ns] of reference slot,
                     if None first time stamp is used
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    plane : plane of profile, either 'h' or 'v'
    resavg: plot residual of distribution (time_stamp) in respect to
            reference distribution (time_stamp_ref)
    resfit: plot deviation from Gaussian fit
    verbose : verbose option for additional output
    
    Returns:
    --------
    flaux : bool, flag to check if profile plots have failed
    """
    # hardcode plot options
    norm = True; smooth = 0; mvavg = True; errbar = True; log=True;
    # set default values
    if slot is None:
      slot = self.get_slots()[0]
      if verbose: print("Using first slot %s"%slot)
    time_stamp_all = self.get_timestamps(slot=slot,plane=plane)
    if time_stamp is None:
      time_stamp = time_stamp_all[-1]
    if time_stamp_ref is None:
      time_stamp_ref = time_stamp_all[0]
    ts = ld.dumpdate(t=time_stamp*1.e-9,
                     fmt='%H:%M:%S',zone='cern')
    ts_ref = ld.dumpdate(t=time_stamp_ref*1.e-9,
                         fmt='%H:%M:%S',zone='cern')
    # adjust figure size
    num=pl.gcf().number
    pl.clf()
    f,(ax1,ax2) = pl.subplots(2,1, sharex=True,num=num,gridspec_kw={'hspace':0.45})
    f.set_size_inches(5, 6, forward=True)
    pl.suptitle(r'%s plane, slot %s, $t$=%s, '%(plane.upper(),slot,ts) +
                r'$t_{\rm ref}$ = %s'%(ts_ref),fontsize=14)
    # rescale position if xaxis=='sigma'
    if xaxis == 'sigma' and self.profiles_norm_avg_stat is not None:
      stat_aux = self.get_profile_mvavg_stat(slot=slot,
                 time_stamp=time_stamp,plane=plane)
      sigma_gauss = stat_aux['sigma_gauss']
      sigma_beam  = stat_aux['sigma_beam']
      if sigma_beam != 0:
        xscale = 1/sigma_beam 
      else:
        xscale = 1/sigma_gauss
    else:
      xscale = 1
    # 1) plot moving average profile + Gaussian fit
    # profile
    prof = self.get_profile_norm_mvavg(slot=slot,
             time_stamp = time_stamp, plane = plane)
    ax1.plot(prof['pos']*xscale,prof['amp'],'k-',label='profile')
    ax1.fill_between(x=prof['pos']*xscale,y1=prof['amp']-prof['amperr'],
                    y2=prof['amp']+prof['amperr'],color='k',alpha=0.3)
    # Gaussian fit
    stat_aux = self.get_profile_mvavg_stat(slot=slot,
                 time_stamp=time_stamp,plane=plane)
    c_gauss, a_gauss = stat_aux['c_gauss'], stat_aux['a_gauss']
    cent_gauss = stat_aux['cent_gauss']
    sigma_gauss    = stat_aux['sigma_gauss']
    amp_gauss = tb.gauss_pdf(prof['pos'],c_gauss,a_gauss,cent_gauss,
                             sigma_gauss)
    ax1.plot(prof['pos']*xscale,amp_gauss,color='r',
             linewidth=1,label='Gaussian fit')
    ax1.set_ylim(1.e-3,0.7)
    ax1.grid(b=True)
    ax1.set_ylabel(r'probability $A$ [a.u.]',fontsize=14)
    ax1.set_yscale('log')
    ax1.set_title('')
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=2, mode="expand", borderaxespad=0.,
                  fontsize=14)
    # 3) residual, only average profile
    if resavg:
      # get overlapping positions
      prof_ref = self.get_profile_norm_mvavg(slot=slot,
                     time_stamp = time_stamp_ref, plane = plane)
      xval = list(set(prof['pos']) and set(prof_ref['pos']))
      mask     = np.array([ pos in xval for pos in prof['pos'] ],
                          dtype=bool)
      mask_ref = np.array([ pos in xval for pos in prof_ref['pos'] ],
                          dtype=bool)
      pos = prof['pos'][mask]*xscale
      res = 1.e2*(prof['amp'][mask]-prof_ref['amp'][mask])
      sig = 1.e2*(np.sqrt(prof['amperr'][mask]**2+prof_ref['amperr'][mask]**2)) 
      ax2.plot(pos,res,'k-',label=r'Profile($t$) - Profile($t_{\rm{ref}}$)')
      ax2.fill_between(x=pos,y1=res-sig,y2=res+sig,color='k',alpha=0.3)
    #    Gaussian fit residual
    if resfit:
      res = 1.e2*(prof['amp']-amp_gauss)
      ax2.plot(prof['pos']*xscale,res,'r-',label='Profile($t$) - Gaussian fit($t$)')
    if resfit or resavg:
      ax2.set_ylim(-5,5)
      ax2.grid(b=True)
      ax2.set_ylabel(r'residual [$10^{-2}$]',fontsize=14)
      ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=1, mode="expand", borderaxespad=0.,
                  fontsize=14)
    if xaxis == 'sigma' and xscale != 1:
      # now rescale the xaxis
      for ax in ax1,ax2:
        ax.set_xlim(-6,6)
        ax.yaxis.set_tick_params(labelsize=14)
      if self.db is not None:
        lbl=(r'position [$\sigma_{\rm{Beam}}$], '+
            r'$\sigma_{\rm{Beam}}$ = %2.2f mm'%sigma_beam)
      else:
        lbl=(r'position [$\sigma_{\rm{Prof}}$], '+
            r'$\sigma_{\rm{Prof}}$ = %2.2f mm'%sigma_gauss)
      ax2.set_xlabel(lbl,fontsize=10)
      ax2.xaxis.set_tick_params(labelsize=14)
      #otherwise leave unchanged
    else:
      for ax in ax1,ax2:
        ax.set_xlim(-8,8)
      ax2.set_xlabel('position [mm]',fontsize=14)
    pl.gcf().subplots_adjust(left=0.18,top=0.87,hspace=0.05)
  def mk_profile_video_ipac17(self, slots = None, t1=None, t2=None,
        xaxis='mm',resavg=False,resfit=True,
        plt_dir='BSRTprofile_gifs', delay=20, 
        export=False,verbose=False):
    """
    Generates a video of the profiles of slot with *slot* using 
    plot_all_ipac17

    Parameters:
    -----------
    slot : slot or list of slots of slots. If slot = None all slots
           are selected. In case of several slot, on video is 
           generated per slot
    t1 : start time in unix time [ns], if None first time stamp for
         slot is used
    t2 : end time in unix time [ns], if None last time stamp for slot
         is used
    xaxis: if mm = leave xaxis in mm as for raw profiles
           if sigma = normalize to sigma calculated with Gaussian fit
                      (due to LSF conversion only Gaussian fit should be
                      used for absolute emittance and beam sigma 
                      calculation)
    resavg: plot residual of distribution (time_stamp) in respect to
            reference distribution (time_stamp_ref)
    resfit: plot deviation from Gaussian fit
    plt_dir : directory to save videos
    delay : option for convert to define delay between pictures
    export : If True do not delete png files
    verbose : verbose option for additional output
    """
    tmpdir = os.path.join(plt_dir,'tmp_bsrtprofiles')
    # dictionary to store failed profiles
    check_plot = {}
    for plane in ['h','v']:
      # set slot and plane, initialize variables
      check_plot[plane] = {}
      # generate the figure and subplot
      for slot in self._set_slots(plane=plane,slot=slot):
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
        if (len(time_stamps) == 0):                                                 
          print('WARNING: no time stamps found for slot %s, '%slot +
                'plane %s and (t1,t2)=(%s,%s)'%(plane,t1,t2))
          continue       
        pngcount = 0
        if verbose: 
          print( '... total number of profiles %s'%(len(time_stamps)))
        # generate png of profiles for each timestamps
        for time_stamp in time_stamps:
          pl.close('all')
          # find closest time stamp for reference bunch
          pl.clf()
          self.plot_all_ipac17(slot=slot,time_stamp=time_stamp,
            time_stamp_ref=time_stamps[0],plane=plane,xaxis=xaxis,
            resavg=resavg,resfit=resfit)
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
    # delete temporary directory
    if (export is False) and (os.path.exists(tmpdir) is True):
      shutil.rmtree(tmpdir)
  def plot_stats(self,slot,plane='h',t1=None,t2=None,param=None,paramidx=None,
                 avgprof=None,mvavg=None,norm=False,log=False,kwargs=None):
    """ 
    plot the statistical parameter *param* for slot *slot* within
    [t1,t2].

    Parameters:
    -----------
    slot: slot number
    plane: either 'h' or 'v'
    t1,t2: start/end as unix time [ns], if None plot full range
    param: parameter to plot, all paramters are list with self.
    paramidx: for the covariance and correlation matrix also the indices
              of the matrix have to be specified as a tuple (i,j).
    avgprof: if avgprof = 'avg' or None: use statistical paramters
                 calculated from average profile (self.profiles_avg_stat)
             if avgprof = 'mvavg': use statistical paramters calculated 
                 from moving average profiles (self.profiles_mvavg_stat)
    mvavg: do a moving average over *mvavg* data points, if None do not average
    norm: if True normalize to initiale value, if False plot raw data
    log: log scale, in this case the absolute value of the data is plotted
    kwargs: dictionary of keyword arguments controlling the plot options, see
              matplotlib.pyplot.plot
    """
    # set default values and check parameters
    if slot not in self.get_slots():
      print('Slot number %s not found!'%slot)
      return
    ts = self.get_timestamps(slot)
    if t1 is None: t1 = ts[0]
    if t2 is None: t2 = ts[-1]
    if avgprof is None: avgprof = 'avg'
    if avgprof == 'avg':
      if param not in self.profiles_norm_avg_stat_var:
        print('ERROR: Give statistical parameter to plot, options are:')
        print self.profiles_norm_avg_stat_var
        return
    elif avgprof == 'mvavg':
      if param not in self.profiles_norm_mvavg_stat_var:
        print('ERROR: Give statistical parameter to plot, options are:')
        print self.profiles_norm_mvavg_stat_var
        return
    else:
      print("ERROR: avgprof must be None,'avg' or 'mvavg'!")
      return
    # c,a,mu,sig
    self.fit_var_gauss = {0:'c',1:'a',2:'mu',3:'sig'}
    # c,a,q,mu,beta
    self.fit_var_qgauss = {0:'c',1:'a',2:'q',3:'mu',4:'beta'}
    print 
    if ('pcov' in param) or ('pcorr' in param) and paramidx == None:
      print('ERROR: Please specify indices of matrix in paramidx!')
      print('for Gaussian:');print(self.fit_var_gauss)
      print('for qGaussian:');print(self.fit_var_qgauss)
      return
    # get data
    if avgprof == 'avg':
      profs_aux = self.profiles_norm_avg_stat[plane][slot]
    elif avgprof == 'mvavg':
      profs_aux = self.profiles_norm_mvavg_stat[plane][slot]
    mask = np.logical_and(profs_aux['time_stamp'] >= t1,
                                profs_aux['time_stamp'] <= t2)
    time_stamps = tb.movingaverage(profs_aux['time_stamp'][mask],navg=mvavg)
    if 'pcov' in param or 'pcorr' in param:
      data = tb.movingaverage(
                profs_aux[param][mask][:,paramidx[0],paramidx[1]],navg=mvavg)
    else:
      data = tb.movingaverage(profs_aux[param][mask],navg=mvavg)
    if log:
      data = np.abs(data)
    if norm:
      data = data/data[0]
    if kwargs is None:
      pl.plot(time_stamps*1.e-9,data)
    else:
      pl.plot(time_stamps*1.e-9,data,**kwargs)
    pytimber.set_xaxis_date()
    pl.grid(b=True)
    pl.ylabel(param)
    pl.legend(loc='best')
    if log: pl.yscale('log')
  def dump(self,path='.',filename='bsrt.pickle',verbose=False):
    """
    dump data with pickle. Used also to combine data from different
    analysis

    Note: disable autoreload before using pickle!
      %autoreload 0

    Parameters:
    -----------
    path: directory to store data
    filename: filename
    verbose : verbose option for additional output
    """
    with open(os.path.join(path,filename),'wb') as pickle_file:
      cPickle.dump(self,pickle_file)
  def load_pickle(self,fn='./bsrt.pickle',verbose=False):
    """
    load data from pickle file and combine it with existing data.
    Used also to combine data from different analysis.

    Note: the class defined in cPickle must be the same as the 
    one loaded in the ipython session or the python file. Check
    autoreload in ipython shell!

    Parameters:
    -----------
    fn: glob search pattern
    verbose : verbose option for additional output
    """
    if os.path.isfile(fn):
      try:
        bsrt_new = cPickle.load(fn)
      except TypeError:
        print('ERROR: File %s is not a pickle file!'%fn)
        return
    else:
      print('ERROR: file %s does not exist!')
      return
    return bsrt_new
  def add_profiles(self,bsrt_new=None,force=False,verbose=False):
    """
    load data from another BSRT class to combine it with existing data.
    Used also to combine data from different analysis.

    This assumes that
    1) BSRTprofiles.load_files().clean_data() has been run in advance.
    2) self.norm().get_stats() has been run for individual bunches, e.g.
        self.norm(slot=20).get_stats(slot=20)

    Parameters:
    -----------
    bsrt_new: class to be loaded
    verbose : verbose option for additional output
    force: overwrite existing data
    """
    # check that profiles attribute is not empty -> at least load_files
    # was run
    if bsrt_new.profiles is None:
      print('ERROR: Please run BSRTprofiles.load_files(...)' +
            '.clean_data(...) for class in file %s!'%fn)
    if self.profiles is None:
      print('ERROR: Please run BSRTprofiles.load_files(...)' +
            '.clean_data(...) on your BSRTprofile class!')
    # check parameters which should all agree if same data was read in
    for at in self.__dict__.keys():
      if 'profile' not in at and 'records' not in at:
        if self.__dict__[at] != bsrt_new.__dict__[at]:
          print('ERROR: classes differ in attribute %s!'%(at) +
                'class: %s, '%(self.__dict__[at]) +
                'pickle: %s'%(bsrt_new.__dict__[at]))
          return
      if at == 'records':
        if self.__dict__[at] != bsrt_new.__dict__[at]:
          print('ERROR: records of class and pickled class differ! ' +
               'Check that you took the same input file!')
          return
    # now combine the profile data
    prof_vars = ([b for b in [ a for a in self.__dict__.keys() 
                  if 'profiles_norm' in a ] if 'var' not in b])
    for plane in 'hv':
      for var in prof_vars:
        bsrt_new_prof = ((bsrt_new.__dict__)[var])[plane]
        for slot in bsrt_new_prof.keys(): # loop over bunches
          # overwrite data only if force = True
          if slot in bsrt_new_prof.keys() and force:
            if verbose:
              print('WARNING: Slot already in database, deleting '+
                    'data %s for plane %s, slot %s!'%(var,plane,slot))
            (self.__dict__)[var][plane][slot] = None
          else:
            if verbose:
              print('WARNING: Slot already in database, skipping '+
                    'data %s for plane %s, slot %s!'%(var,plane,slot))
            continue
          (self.__dict__)[var][plane][slot] = bsrt_new_prof[slot]
