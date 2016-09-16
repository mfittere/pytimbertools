import os as os

try:
  import numpy as np
  import matplotlib.pyplot as pl
  from scipy.optimize import curve_fit
except ImportError:
  print "No module found: numpy, matplotlib and scipy modules should be present to run pytimbertools"

try:
  import pytimber
except ImportError:
  print "pytimber can not be found!"

try:
  from beams import *
except ImportError:
  print "beams module can not be found!"

def exp_fit(x,a,tau):
  return a*np.exp(x/tau)

def _get_timber_data(beam,t1,t2,db=None):
  """retrieve data from timber needed for
  emittance calculation
  
  Parameters:
  ----------
  db: timber database
  beam: either 'B1' or 'B2'
  EGeV: beam energy [GeV]
  t1,t2: start and end time of extracted data
         in unix time
  Returns:
  -------
  """
  # if no databse is given create dummy database to extract data
  if db==None: db=pytimber.LoggingDB()
  # extract the data from timber
  fit_sig_h,fit_sig_v,gate_delay=db.search('%LHC%BSRT%'+beam.upper()+'%FIT_SIGMA_%')+db.search('%LHC%BSRT%'+beam.upper()+'%GATE_DELAY%')
  bsrt_var=[fit_sig_h,fit_sig_v,gate_delay]
  lsf_h,lsf_v,beta_h,beta_v=db.search('%LHC%BSRT%'+beam.upper()+'%LSF_%')+db.search('%LHC%BSRT%'+beam.upper()+'%BETA%')
  bsrt_lsf_var=[lsf_h,lsf_v,beta_h,beta_v]
  bsrt_data=db.get(bsrt_var,t1,t2)
  bsrt_lsf=db.get(bsrt_lsf_var,t1-24*60*60,t2)
  #check that all timestamps are the same for bsrt_var
  for k in bsrt_var:
      if np.any(bsrt_data[bsrt_var[0]][0]!=bsrt_data[k][0]):
          print "ERROR: time stamps for %s and %s differ!"%(bsrt_var[0],bsrt_var[k])
          return
  # create list containing all the data (bsrt_list), then save data in structured array bsrt_array
  bsrt_list=[]
  var=zip(bsrt_data[gate_delay][0],bsrt_data[gate_delay][1],
          bsrt_data[fit_sig_h][1],bsrt_data[fit_sig_v][1])
  for t,gate,sigh,sigv in var:
  #find closest timestamp with t_lsf<t
      lsf_t={};lsf_v={}
      for k in bsrt_lsf_var:
          idx=np.where(t-bsrt_lsf[k][0]>=0.)[0][-1]
          lsf_t[k],lsf_v[k]=bsrt_lsf[k][0][idx],bsrt_lsf[k][1][idx]
      for i in range(len(gate)):
          bsrt_list.append(tuple([t,gate[i],sigh[i],sigv[i]]+
             [lsf_t[k] for k in bsrt_lsf_var]+[lsf_v[k] for k in bsrt_lsf_var]))
  ftype=[('time',float),('gate',float),('sigh',float),('sigv',float),('lsfh_t',float),('lsfv_t',float),('beth_t',float),('betv_t',float),('lsfh',float),('lsfv',float),('beth',float),('betv',float)]
  bsrt_array=np.array(bsrt_list,dtype=ftype)
  return bsrt_array

class BSRT(object):
  """class to analyze data from BSRT
  Example:
  --------
    To extract the data from timber:

      t1=pytimber.parsedate("2016-08-24 00:58:00.000")
      t2=pytimber.parsedate("2016-08-24 00:59:00.000")
      bsrt=BSRT.get('B1',t1,t2)

  Attributes:
  -----------
  timber_variables: timber variables needed to calculate
                    normalized emittance
  emit: dictionary of normalized emittances
        {slot: [time [s],emith [um],emitv[um]]}
  get_timber_data: array with data from timber to calculate 
        emittances (usually not stored)
  get: extract data from timber and return dictionary
       with time and normalized emittances for each slot."""
  timber_variables={}
  timber_variables['B1']=[u'LHC.BSRT.5R4.B1:FIT_SIGMA_H', u'LHC.BSRT.5R4.B1:FIT_SIGMA_V', u'LHC.BSRT.5R4.B1:GATE_DELAY',u'LHC.BSRT.5R4.B1:LSF_H', u'LHC.BSRT.5R4.B1:LSF_V', u'LHC.BSRT.5R4.B1:BETA_H', u'LHC.BSRT.5R4.B1:BETA_V']
  timber_variables['B2']=[u'LHC.BSRT.5L4.B2:FIT_SIGMA_H', u'LHC.BSRT.5L4.B2:FIT_SIGMA_V', u'LHC.BSRT.5L4.B2:GATE_DELAY',u'LHC.BSRT.5L4.B2:LSF_H', u'LHC.BSRT.5L4.B2:LSF_V', u'LHC.BSRT.5L4.B2:BETA_H', u'LHC.BSRT.5L4.B2:BETA_V']
  def __init__(self,emit=None):
    if emit==None:
      self.emit=np.array([])
    else:
      self.emit=emit
  @classmethod
  def get(cls,beam,EGeV,t1,t2,db=None):
    """retrieve data using timber and calculate
    normalized emittances from extracted values.
    Parameters:
    ----------
    db: timber database
    beam: either 'B1' or 'B2'
    EGeV: beam energy [GeV]
    t1,t2: start and end time of extracted data
           in unix time
    Returns:
    -------
    emit: dictionary of normalized emittances
          sorted after slot number
          {slot: [time [s],emith [um],emitv[um]]}
    """
    bsrt_array=_get_timber_data(beam,t1,t2,db)
# create dictionary indexed with slot number
    bsrt_dict={}
    for j in set(bsrt_array['gate']):#loop over slots
        bsrt_slot=bsrt_array[bsrt_array['gate']==j]#data for slot j
        bsrt_emit=[]
        for k in set(bsrt_slot['time']):#loop over all timestamps for slot j
            bsrt_aux=bsrt_slot[bsrt_slot['time']==k] #data for slot j and timestamp k
            emith=beamparam.emitnorm(np.mean((bsrt_aux['sigh']**2-bsrt_aux['lsfh']**2)/bsrt_aux['beth']),EGeV)
            emitv=beamparam.emitnorm(np.mean((bsrt_aux['sigv']**2-bsrt_aux['lsfv']**2)/bsrt_aux['betv']),EGeV)
            bsrt_emit.append((k,emith,emitv))
        bsrt_dict[j]=np.sort(np.array(bsrt_emit,dtype=[('time',float),('emith',float),('emitv',float)]),axis=0)#sort after the time
    return cls(emit=bsrt_dict)
  def get_timber_data(self,beam,t1,t2,db=None):
    """retrieve data from timber needed for
    emittance calculation
    
    Parameters:
    ----------
    db: timber database
    beam: either 'B1' or 'B2'
    EGeV: beam energy [GeV]
    t1,t2: start and end time of extracted data
           in unix time
    Returns:
    -------
    bsrt_array: structured array with the format 
                ['time','gate','sigh','sigv','lsfh_t',
                'lsfv_t','beth_t','betv_t','lsfh','lsfv',
                'beth','betv']
    """
    return _get_timber_data(beam,t1,t2,db)
  def fit(self,t1,t2):
    """fit the emittance between t1 and t2
    with an exponential function:
      a*exp(t/tau)
    Paramters:
    ----------
    t1: start time in unix time
    t2: end time in unix time

    Returns:
    --------
    a,tau: fit parameters a (initial value [um]) and
           tau (growth time [s])
    """
    # check that the data has been extracted
    if len(self.emit)==0:
      print "ERROR: first extract the emittance data using BSRT.get(beam,EGeV,t1,t2,db)"
      return
    # loop over all slots
    bsrt_fit_dict={}
    for slot in self.emit.keys():
       mask = (self.emit[slot]['time']>=t1) & (self.emit[slot]['time']<=t2)
       data = self.emit[slot][mask]
       try:
         # subtract initial time to make fitting easier
         data['time']=data['time']-data['time'][0]
         # give a guess for the initial paramters
         # assume eps(t1)=a*exp(t1/tau)
         #        eps(t2)=a*exp(t2/tau)
         fit_data = [t1,t2]
         for plane in ['h','v']:
           t2_fit=data['time'][-1]-data['time'][0]
           epst2_fit=data['emit%s'%plane][-1]
           epst1_fit=data['emit%s'%plane][0]
           a_init=epst1_fit
           tau_init=t2_fit/(np.log(epst2_fit)-np.log(epst1_fit))
           p,pcov=curve_fit(exp_fit,data['time'],data['emit%s'%plane],p0=[a_init,tau_init])
           psig=[np.sqrt(pcov[i,i]) for i in range(len(p))]
           fit_data+=[p[0],psig[0],p[1],psig[1]]
         ftype=[('t1',float),('t2',float),('ah',float),('sigah',float),('tauh',float),('sigtauh',float),('av',float),('sigav',float),('tauv',float),('sigtauv',float)]
         bsrt_fit_dict[slot] = np.array([tuple(fit_data)],dtype=ftype)
       except IndexError:
         print 'no data found for slot %s'%slot
    return bsrt_fit_dict

