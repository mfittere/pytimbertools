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

try:
  from beams import *
except ImportError:
  print "beams module can not be found!"

class BSRT(object):
  """class to analyze data from BSRT
  Example:
  --------
    To extract the data from timber:

      t1=pytimber.parsedate("2016-08-24 00:58:00.000")
      t2=pytimber.parsedate("2016-08-24 00:59:00.000")
      bsrt=BSRT.get('B1',t1,t2)"""
  timber_variables={}
  timber_variables['B1']=[u'LHC.BSRT.5R4.B1:FIT_SIGMA_H', u'LHC.BSRT.5R4.B1:FIT_SIGMA_V', u'LHC.BSRT.5R4.B1:GATE_DELAY',u'LHC.BSRT.5R4.B1:LSF_H', u'LHC.BSRT.5R4.B1:LSF_V', u'LHC.BSRT.5R4.B1:BETA_H', u'LHC.BSRT.5R4.B1:BETA_V']
  timber_variables['B2']=[u'LHC.BSRT.5L4.B2:FIT_SIGMA_H', u'LHC.BSRT.5L4.B2:FIT_SIGMA_V', u'LHC.BSRT.5L4.B2:GATE_DELAY',u'LHC.BSRT.5L4.B2:LSF_H', u'LHC.BSRT.5L4.B2:LSF_V', u'LHC.BSRT.5L4.B2:BETA_H', u'LHC.BSRT.5L4.B2:BETA_V']
  def __init__(self,data=None,emit=None):
    if data==None:
      self.emit=np.array([])
    else:
      self.emit=data
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
          {slot: [time [s],emith [um],emitv[um]])
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
    print bsrt_data,bsrt_lsf
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
        bsrt_dict[j]=np.array(bsrt_emit,dtype=[('time',float),('emith',float),('emitv',float)])
    return cls(data=bsrt_dict)


