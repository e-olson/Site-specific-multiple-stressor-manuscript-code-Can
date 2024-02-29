""" General use functions
"""
import numpy as np
import subprocess
import re
import datetime as dt
import cftime
import netCDF4 as nc
import os
import warnings
#import xarray as xr
import pandas as pd
from scipy.stats.distributions import t as scpst_t
import scipy.stats as scst

fdate_torig=re.compile('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

# horizontal grid file paths:
path_gfdl_static = '/work/Elise.Olson/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/ocean_daily_gfdl/ocean_daily_gfdl.static.nc'
path_1x1_static = '/work/Elise.Olson/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/ocean_daily_1x1deg/ocean_daily_1x1deg.static.nc'

class statgrid:
    vlist=['geolon','geolat','deptho','wet','sftof','areacello'] # list of variables to extract
    def __init__(self,path=path_gfdl_static):
        with nc.Dataset(path) as fstat:
            # use setattr to assign variables to their names
            vlist=self.vlist
            for ivar in vlist:
                setattr(self, ivar, fstat.variables[ivar][:])
            self.units={ivar:fstat.variables[ivar].units for ivar in vlist}
            self.long_name={ivar:fstat.variables[ivar].long_name for ivar in vlist}
            self.sizes={ivar:np.shape(fstat.variables[ivar][:]) for ivar in vlist}
            self.dims= {ivar:fstat.variables[ivar].dimensions for ivar in vlist}
    def __repr__(self):
        nv=max([len(ivar) for ivar in self.vlist]+[8])
        nu=max([len(val)  for val  in self.units.values()]+[5])
        nn=max([len(val)  for val  in self.long_name.values()]+[9])
        ns=max([len(str(val))  for val  in self.sizes.values()]+[4])
        nd=max([len(str(val))  for val  in self.dims.values()]+[10])
        return f"{'variable':>{nv}}  {'units':{nu}}  {'long_name':{nn}}  {'size':{ns}}  {'dimensions':{nd}}\n"+\
            "\n".join([f"{ivar:>{nv}}  {self.units[ivar]:{nu}}  {self.long_name[ivar]:{nn}}  {str(self.sizes[ivar]):{ns}}  {str(self.dims[ivar]):{nd}}" for ivar in self.vlist])

def noLeapFromNC(ff,varname='time'):
    # ff should be netCDF4 file handle 
    # returns model time as cftime noleap
    return cftime.num2date(ff.variables[varname][:],ff.variables[varname].units,calendar='noleap')

def cftnoleap_to_dt(x0):
    if isinstance(x0,cftime.DatetimeNoLeap):
        return dt.datetime(int(x0.year),int(x0.month),int(x0.day),
                           int(x0.hour),int(x0.minute),int(x0.second))
    else: # assume array
        return np.array([cftnoleap_to_dt(el) for el in x0])

def decyr_to_dt(decy):
    # convert decimal year to datetime.datetime
    if isinstance(decy, float):
        yr=int(decy)
        yrfrac=decy-yr
        yrlendays=(dt.datetime(yr+1,1,1)-dt.datetime(yr,1,1)).days
        return dt.datetime(yr,1,1,0,0)+dt.timedelta(days=yrlendays*yrfrac)
    else: # assume array-like
        return np.array([decyr_to_dt(el) for el in decy])

def decyr(ti):
    # convert cftime noleap or datetime.datetime to decimal year
    yr=ti.year
    if cf.isNoLeap(ti):
        yrlen = 365
        yrref = cftime.datetime(yr,1,1,0,0,0,calendar='noleap')
    else:
        yrlen = (dt.datetime(yr+1,1,1)-dt.datetime(yr,1,1)).days
        yrref = dt.datetime(yr,1,1,0,0,0)
    return yr+(ti-yrref).total_seconds()/(24*3600*yrlen)

def dt_to_cftnoleap(x):
    if isinstance(x,dt.datetime):
        if x.month==2 and x.day==29:
            return cftime.datetime(x.year,x.month,x.day-1,x.hour,x.minute,x.second,calendar='noleap')
        else:
            return cftime.datetime(x.year,x.month,x.day,x.hour,x.minute,x.second,calendar='noleap')
    else: # assume array
        return np.array([dt_to_cftnoleap(el) for el in x])

def dayindex(dts):
    return [(ii-dt.datetime(1900,1,1)).days for ii in dts]

class timeindex(np.ndarray):
    def __new__(cls, dts, freq='daily',torig=dt.datetime(1900,1,1)):
        # https://numpy.org/doc/stable/user/basics.subclassing.html
        # Input array is an already formed ndarray instance
        if isNoLeap(dts):
            torig=cftime.datetime(torig.year,torig.month,torig.day,calendar='noleap')
        if freq=='daily' or freq=='day':
            # We first cast to be our class type
            if type(dts) in [dt.datetime,cftime.datetime]:
                obj = np.asarray([(dts-torig).days,]).view(cls)
            else:
                obj = np.asarray([(ii-torig).days for ii in dts]).view(cls)
        elif freq=='monthly' or freq=='month':
            obj = np.asarray([int((ii.year-torig.year)*12+ii.month-torig.month) for ii in dts]).view(cls)
        # add the new attribute to the created instance
        obj.freq=freq
        obj.torig=torig
        obj.calendar='noleap' if isNoLeap(dts) else 'standard'
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.freq = getattr(obj, 'freq', None)
        self.torig = getattr(obj, 'torig', None)
        self.calendar = getattr(obj, 'calendar', None)

class inttimeind(int):
    def __new__(cls,ti,freq='daily',torig=dt.datetime(1900,1,1),calendar='infer',fixval=False):
        if calendar=='infer':
            calendar = 'noleap' if isNoLeap(ti) else 'standard'
        if calendar=='noleap':
            torig=cftime.datetime(torig.year,torig.month,torig.day,calendar='noleap')
        if fixval: # for adding attributes to precalculated value
            ival=int(ti)
        elif freq=='daily' or freq=='day':
            ival = (ti-torig).days
        elif freq=='monthly' or freq=='month':
            ival= (ti.year-torig.year)*12+ti.month-torig.month
        rval = super(inttimeind,cls).__new__(cls,ival)
        rval.calendar=calendar
        rval.torig=torig
        rval.freq=freq
        return rval
    #def __getnewargs__(self): # for unpickling
    #    return (str(self),self.freq,self.torig,self.calendar,True)

def to_int_tind(ti,freq='daily',torig=dt.datetime(1900,1,1),calendar='infer'):
    if hasattr(ti,'__len__') and ~isinstance(ti,str): # assume array of datetimes
        return np.array([to_int_tind(el,freq=freq,torig=torig,calendar=calendar) for el in ti],dtype='object')
    else:
        return inttimeind(ti,freq=freq,torig=torig,calendar=calendar)

class exacttimeind(np.float64):
    #note: does not pickle/unpickle correctly
    def __new__(cls,ti,torig=dt.datetime(1900,1,1),calendar='infer',fixval=False):
        if calendar=='infer':
            calendar='noleap' if isNoLeap(ti) else 'standard'
        elif calendar=='noleap' or calendar=='standard':
            if ((calendar=='standard' and isNoLeap(ti)) or (calendar=='noleap' and not isNoLeap(ti))) and not fixval:
                raise TypeError('Wrong calendar')
        else:
            raise ValueError('calendar can be infer or standard or noleap')
        if calendar=='noleap':
            torig=cftime.datetime(torig.year,torig.month,torig.day,calendar='noleap')
        if fixval: # for setting attributes on calc'd val
            ival=ti
        else: # normal behavior
            ival=(ti-torig).total_seconds()/(24*3600)
        #ival=(ti-torig).total_seconds()/(24*3600)
        rval = super(exacttimeind,cls).__new__(cls,ival)
        rval.calendar=calendar
        rval.torig=torig
        return rval
    #def __getnewargs__(self):
    #    # to pass correct value to __new__ upon unpickling
    #    return (float(self),self.torig,self.calendar,True)

def to_exact_tind(ti,torig=dt.datetime(1900,1,1),calendar='infer'):
    if hasattr(ti,'__len__') and ~isinstance(ti,str): # assume array of datetimes
        return np.array([to_exact_tind(el,torig=torig,calendar=calendar) for el in ti],dtype='object')
    else:
        return exacttimeind(ti,torig=torig,calendar=calendar)

class linreg:
    # make this a class in order to return object with descriptive attributes
    def __init__(self,X,Y,alpha=0.05):
        A=np.concatenate([np.ones((len(X),1)), np.expand_dims(X,1)],1)
        if len(np.shape(np.squeeze(Y)))==1:
            self.coef, self.SumResSq, self.p, self.n, self.cov, self.CI, self.SE, self.Rsq = lsqfit(A,Y)
        else:
            self.coef, self.SumResSq, self.p, self.n, self.cov, self.CI, self.SE, self.Rsq = lsqfit_md(A,Y)
    def __repr__(self):
        t = type(self)
        return f"{t.__module__}.{t.__qualname__}(\n\tcoef={self.coef},\n\tSumResSq={self.SumResSq},\n\tp={self.p},\n\tn={self.n},\n\tcov={self.cov},\n\tCI={self.CI}\n\tRsq={self.Rsq})"

def lsqfit(X,Y,alpha=0.05):
    # X is nxp covariables; Y is nx1 response variable
    # calculate linear fit and 95% confidence intervals, stats
    # remove all rows for any bad values:
    if len(np.shape(Y))==2:
        if np.shape(Y)[1]==1:
            Y=Y[:,0]
    ii=~np.any(np.isnan(X),axis=1)&~np.isnan(Y[:])
    X=X[ii,:]
    Y=Y[ii]
    b,res,p,svs=np.linalg.lstsq(X,Y,rcond=None) # res=np.sum((np.dot(X,b)-Y)**2)
    n=len(Y)
    sig2=res/(n-p)
    cov=sig2*np.linalg.inv(np.dot(X.T,X))
    se=np.sqrt(np.diag(cov))
    sT=scpst_t.ppf(1.0-alpha/2.0,n-p) # use Student's t distribution in case of small sample sizes
    CI=sT*se
    Rsq=1-res/np.sum((Y-np.mean(Y))**2)
    return b,res,p,n,cov,CI,se,Rsq[0] # Rsq has multiple dimensions because res does

def linfitstats(X,Y,b,alpha=.05):
    # b is [intercept, slope]
    if len(np.shape(Y))==2:
        if np.shape(Y)[1]==1:
            Y=Y[:,0]
    X=np.concatenate([np.ones((len(X),1)), np.expand_dims(X,1)],1)
    ii=~np.any(np.isnan(X),axis=1)&~np.isnan(Y[:])
    X=X[ii,:]
    Y=Y[ii]
    res=np.sum((np.dot(X,b)-Y)**2)
    n=len(Y)
    p=len(b)
    sig2=res/(n-p)
    cov=sig2*np.linalg.inv(np.dot(X.T,X))
    se=np.sqrt(np.diag(cov))
    sT=scpst_t.ppf(1.0-alpha/2.0,n-p) # use Student's t distribution in case of small sample sizes
    CI=sT*se
    Rsq=1-res/np.sum((Y-np.mean(Y))**2)
    return {'res':res,'sig2':sig2,'p':p,'n':n,'cov':cov,'CI':CI,'se':se,'Rsq':Rsq}

def basicfitstats(Yest,Ytrue,b):
    # b is parameters
    # res is sum square error
    ii=~np.any(np.isnan(Yest))&~np.isnan(Ytrue)
    Yest=Yest[ii]
    Ytrue=Ytrue[ii]
    res=np.sum((Ytrue-Yest)**2)
    n=len(Ytrue)
    p=len(b)
    se=np.sqrt(res/(n-p))
    return {'res':res,'p':p,'n':n,'se':se,}

def nanxcor(x,y,lagmax=0,Nmin=0,method='Pearson'):
    # wrap scipy.stats correlation functions to allow lags and handle NaN values but still return p-values
    #x and y should initially be aligned in time
    if method=='Pearson' or method=='pearson':
        func=scst.pearsonr
    elif method=='Spearman' or method=='spearman':
        func=scst.spearmanr
    if lagmax==0:
        iii=~np.isnan(x)&~np.isnan(y)
        if np.sum(iii)>=Nmin:
            r=func(x[iii],y[iii])
            return r[0],r[1],np.sum(iii)
        else:
            return np.nan, np.nan, np.nan
    else:
        rs=list()
        ps=list()
        Ns=list()
        for ii in np.arange(-1*lagmax,lagmax+1):
            px=int(np.abs(ii))
            mx=int(-1*np.abs(ii))
            if ii<0:
                r,p,N=nanxcor(x[px:],y[:mx],lagmax=0,Nmin=Nmin,method=method)
            elif ii==0:
                r,p,N=nanxcor(x,y,lagmax=0,Nmin=Nmin,method=method)
            else:
                r,p,N=nanxcor(x[:mx],y[px:],lagmax=0,Nmin=Nmin,method=method)
            rs.append(r)
            ps.append(p)
            Ns.append(N)
        return np.array(rs),np.array(ps),np.array(Ns)


def lsqfit_md(X,data,alpha=0.05):
    # assume no NaN values; this is for model results
    # X is nxp covariables; data is nxmxr response variable
    # dimension to do regression on must be 0!
    # calculate linear fit and 95% confidence intervals, stats
    # adapt reshaping code from scipy.signal.detrend
    # put new dimensions at end, except for coefs at front
    data=np.asarray(data)
    dshape = data.shape
    N=dshape[0]
    assert N==len(X) # check correct dimensions
    newdata = np.reshape(data,(N, np.prod(dshape, axis=0) // N)) # // is floor division
    newdata = newdata.copy()  # make sure we have a copy
    b,res,p,svs=np.linalg.lstsq(X,newdata,rcond=None) # res=np.sum((np.dot(X,b)-Y)**2)
    sig2=res/(N-p)
    cova=np.linalg.inv(np.dot(X.T,X))
    cov=np.expand_dims(sig2,(-2,-1))*np.expand_dims(np.linalg.inv(np.dot(X.T,X)),0)
    se=np.expand_dims(np.sqrt(np.diag(cova)),0)*np.expand_dims(np.sqrt(sig2),-1)
    sT=scpst_t.ppf(1.0-alpha/2.0,N-p) # use Student's t distribution in case of small sample sizes
    CI=sT*se
    nd0=newdata.copy(); res0=res.copy()
    Rsq=1-res/np.sum((newdata-np.mean(newdata,0,keepdims=True))**2,0)
    #reshape arrays
    bshp=tuple([2]+list(dshape)[1:])
    newshp0=list(dshape)[1:]
    newshp1=newshp0+[b.shape[0]]
    newshp2=newshp0+list(cov.shape[1:])
    newshp0=tuple(newshp0)
    newshp1=tuple(newshp1)
    newshp2=tuple(newshp2)
    b=np.reshape(b,bshp)
    res=np.reshape(res,newshp0)
    cov=np.reshape(cov,newshp2)
    CI=np.reshape(CI,newshp1)
    se=np.reshape(se,newshp1)
    Rsq=np.squeeze(np.reshape(Rsq,newshp0))
    return b,res,p,N,cov,CI,se,Rsq

def ydfromdt(dts):
    if isinstance(dts,dt.datetime):
        return (dts-dt.datetime(dts.year-1,12,31)).days
    elif hasattr(dts,'__len__') and ~isinstance(dts,str): # assume array of datetimes
        return np.array([ydfromdt(el) for el in dts])
    elif pd.isnull(dts):
        return np.nan
    else:
        raise TypeError('bad type: ', type(dts))

def exactydfromdt(dts):
    if isinstance(dts,dt.datetime):
        return (dts-dt.datetime(dts.year-1,12,31)).total_seconds()/(24*3600)
    elif hasattr(dts,'__len__') and ~isinstance(dts,str): # assume array of datetimes
        return np.array([ydfromdt(el) for el in dts])
    elif pd.isnull(dts):
        return np.nan
    else:
        raise TypeError('bad type: ', type(dts))

def monthfromdt(dts):
    if isinstance(dts,dt.datetime):
        return dts.month
    else: # assume array of datetimes
        return np.array([monthfromdt(el) for el in dts])

def isNoLeap(dts):
    if isinstance(dts,cftime.datetime):
        return dts.calendar=='noleap'
    try:
        if len(dts)>=1:
            return isinstance(dts[0],cftime.datetime) and dts[0].calendar=='noleap'
    except TypeError:
        return False

def ydNLfromcftNL(dts):
    if isNoLeap(dts):
        if type(dts)==list or type(dts)==np.ndarray or type(dts)==np.ma.core.MaskedArray:
            return np.array([ydNLfromcftNL(el) for el in dts])
        else:
            return (dts-cftime.datetime(dts.year-1,12,31,calendar='noleap')).days
    else:
        raise TypeError('input not cftime noleap')

def exactydNLfromcftNL(dts):
    if isNoLeap(dts):
        if type(dts)==list or type(dts)==np.ndarray or type(dts)==np.ma.core.MaskedArray:
            return np.array([ydNLfromcftNL(el) for el in dts])
        else:
            return (dts-cftime.datetime(dts.year-1,12,31,calendar='noleap')).total_seconds()/(24*3600)
    else:
        raise TypeError('input not cftime noleap')

def read_torig(fhandle,varname='time',rtype='datetime'):
    # fhandle should be netCDF4 Dataset object or file path
    try:
        ustring=fhandle.variables[varname].units
    except AttributeError:
        with nc.Dataset(fhandle) as ff:
            ustring=ff.variables[varname].units
    datestr=fdate_torig.search(ustring).group()
    if rtype=='string':
        return datestr
    elif rtype=='datetime':
        return dt.datetime.strptime(datestr,'%Y-%m-%d %H:%M:%S')
    else:
        raise Exception('rtype not defined')

def prepb19(text):
    """ prepend instructions to load fre/bronx-19 module before executing command 
    for use with subprocess + shell=True
    arg text is command to be executed requiring fre/bronx-19
    """
    text = 'source /usr/local/Modules/default/init/bash; module load fre/bronx-19; '+text
    return text

def prepmod(text,modname):
    """ prepend a generic module"""
    return f'source /usr/local/Modules/default/init/bash; module load {modname}; '+text

def prep_py39d(text):
    """ prepend instructions to activate py39d environment before executing command 
    for use with subprocess + shell=True
    arg text is command to be executed
    """
    text = 'source /usr/local/Modules/default/init/bash; source /home/Elise.Olson/.bashrc; conda activate py39d; '+text
    return text

class pidlist(list):
    """ class to add fxns on top of list for storing pids from subprocess
    """
    def __init__(self,data=None):
        if data is None:
            super().__init__()
        else:
            super().__init__(data)

    def wait(self,maxpid=0,verb=False):
        #pids should be a list of output from subprocess.Popen
        #maxpid is length pid list should reach before continuing
        #       generally maxproc-1 or 0
        # remember lists are mutable so original is changed; no need to return
        ind=0
        while len(self)>maxpid:
            ind=(ind+1)%len(self)
            if self[ind].poll() is not None:
                cpid=self.pop(ind)
                pidclose(cpid,verb)
        return
    
def pidclose(pid,verb=False):
    # make sure stdout and stderr are closed and display any output there
    if verb:
        for line in pid.stdout:
            print(line)
    for line in pid.stderr:
        print(line)
    if pid.returncode!=0:
        print('returncode:',pid.returncode)
    pid.stdout.close()
    pid.stderr.close()
    return

def subprocrun(cmdlist,maxproc=1,verb=False,prepfun=None):
    # verb=True prints Popen stdout
    # cmdlist should be list of commands to run in shell
    # prepfun is function to apply to each elemnt in cmdlist, eg prepb19 to load bronx-19 before running
    if type(cmdlist)==str: # if single path string passed, convert to list
        cmdlist=[cmdlist,]
    if prepfun is not None:
        cmdlist=[prepfun(el) for el in cmdlist]
    pids=pidlist()
    for icmd in cmdlist:
        if verb:
            print(icmd)
        pids.wait(maxproc-1,verb=verb)
        pids.append(subprocess.Popen(icmd, shell=True, stdout=subprocess.PIPE,  stderr=subprocess.PIPE))
    pids.wait()
    return 

def fmtVarName(strx):
    """ transform string into one that meets python naming conventions
    arg: str
    returns: str suitable for file name
    """
    vName=re.sub('[^a-zA-Z0-9_\-\s/]','',strx.strip())
    vName=re.sub('[\s/]','_',vName)
    vName=re.sub('-','_',vName)
    if re.match('[0-9]',vName):
        vName='_'+vName
    return vName

def haversine(la0,lo0,la1,lo1):
    """ haversine formula with numpy array handling
    Calculates spherical distance between points on Earth in meters
    Compares elements of (la0,lo0) with (la1,lo1)
    Shapes must be compatible with numpy array broadcasting
    args: lats and lons in decimal degrees
    returns: distance on sphere with volumetric mean Earth radius in meters
    """
    rEarth=6371*1e3 # 
    # convert to radians
    la0=np.radians(la0)
    la1=np.radians(la1)
    lo0=np.radians(lo0)
    lo1=np.radians(lo1)
    theta=2*np.arcsin(np.sqrt(np.sin((la0-la1)/2)**2+np.cos(la0)*np.cos(la1)*np.sin((lo0-lo1)/2)**2))
    d=rEarth*theta
    return d

def nearest_point(la,lo,gridla,gridlo,mask=None,tol=2,thresh=40,badval=-999999):
    """ find nearest point to la,lo on arbitrary horizontal grid with points at gridla, gridlo
    with optional masking (eg for ocean values) 
    args: 
        la,lo: coordinates of point(s) to find
        gridla,gridlo: 2-d array of grid latitude, longitude
        mask: boolean mask of same shape ad gridla and gridlo with ones at valid grid points
        tol: number of degrees lat/lon within which to search; could be problematic near poles
        thresh: only accept grid points within this many km of target 
                default 40 km based on .5 deg grid and half diagonal grid distance:
                        .5*111km/deg*sqrt(2)/2 ~ 39.2 km
    returns:
        j,i model indices along y and x dimensions, respectively
        if mask is provided, j and i are set to badval; note that if negative integer, might index array w/o err
    """
    if not isinstance(la,(int,float,np.int64,np.float64,np.int32,np.float32)):
        try:
            len(la)
        except TypeError as err:
            raise Exception(f'TypeError:{err}\nAdd type to list? '+repr(type(la))) from None
        ji=np.array([nearest_point(ila,ilo,gridla,gridlo,mask,tol,thresh,badval) for ila, ilo in zip(la,lo)])
        return ji[:,0],ji[:,1]
    jjj,iii=np.where(((gridlo>lo-tol)&(gridlo<lo+tol)&(gridla>la-tol)&(gridla<la+tol))|\
                     ((gridlo+360>lo-tol)&(gridlo+360<lo+tol)&(gridla>la-tol)&(gridla<la+tol))|\
                     ((gridlo>lo+360-tol)&(gridlo<lo+360+tol)&(gridla>la-tol)&(gridla<la+tol)))
    #jjj,iii=np.where((gridlo>lo-tol)&(gridlo<lo+tol)&(gridla>la-tol)&(gridla<la+tol))
    ddd=np.minimum(np.minimum(haversine(la,lo+360,gridla[jjj,iii],gridlo[jjj,iii]),
                            haversine(la,lo,gridla[jjj,iii],gridlo[jjj,iii])),
                            haversine(la,lo,gridla[jjj,iii],gridlo[jjj,iii]+360))

    ind=np.argmin(ddd)
    if ddd[ind]<=thresh*1e3:
        j=jjj[ind]
        i=iii[ind]
        if mask is not None:
            if mask[j,i]==0:
                j=badval
                i=badval
    else:
        j=badval
        i=badval
    return [j,i]

def nearest_valid_point(la,lo,gridla,gridlo,mask,tol=2,thresh=40,badval=-999999):
    """ find nearest point to la,lo on arbitrary horizontal grid with points at gridla, gridlo
    with optional masking (eg for ocean values) 
    args: 
        la,lo: coordinates of point(s) to find
        gridla,gridlo: 2-d array of grid latitude, longitude
        mask: boolean mask of same shape ad gridla and gridlo with ones at valid grid points
        tol: number of degrees lat/lon within which to search; could be problematic near poles
        thresh: only accept grid points within this many km of target 
                default 40 km based on .5 deg grid and half diagonal grid distance:
                        .5*111km/deg*sqrt(2)/2 ~ 39.2 km
    returns:
        j,i model indices along y and x dimensions, respectively
        nearest valid (mask=1) point is returned
        if none is found within thresh, badval is returned; note that if negative integer, might index array w/o err

    """
    if not isinstance(la,(int,float,np.int64,np.float64,np.int32,np.float32)):
        try:
            len(la)
        except TypeError as err:
            raise Exception(f'TypeError:{err}\nAdd type to list? '+repr(type(la))) from None
        ji=np.array([nearest_valid_point(ila,ilo,gridla,gridlo,mask,tol,thresh,badval) for ila, ilo in zip(la,lo)])
        return ji[:,0],ji[:,1]
    jjj,iii=np.where((((mask==1) & (gridlo>lo-tol)&(gridlo<lo+tol)&(gridla>la-tol)&(gridla<la+tol))|\
                     ((gridlo+360>lo-tol)&(gridlo+360<lo+tol)&(gridla>la-tol)&(gridla<la+tol))|\
                     ((gridlo>lo+360-tol)&(gridlo<lo+360+tol)&(gridla>la-tol)&(gridla<la+tol))))
    #jjj,iii=np.where((gridlo>lo-tol)&(gridlo<lo+tol)&(gridla>la-tol)&(gridla<la+tol))
    if len(jjj)==0: # none found
        return [badval,badval]
    ddd=np.minimum(np.minimum(haversine(la,lo+360,gridla[jjj,iii],gridlo[jjj,iii]),
                            haversine(la,lo,gridla[jjj,iii],gridlo[jjj,iii])),
                            haversine(la,lo,gridla[jjj,iii],gridlo[jjj,iii]+360))

    ind=np.argmin(ddd)
    if ddd[ind]<=thresh*1e3:
        j=jjj[ind]
        i=iii[ind]
    else:
        j=badval
        i=badval
    return [j,i]


def slidingWindowEval(x,func,window,axis=0):
    # x is input array
    # func is function to carry out over window
    # window is window size
    # axis is axis to act along, in case of multiple
    # if window is even, results will be shifted left by 1/2 unit
    x1=np.lib.stride_tricks.sliding_window_view(x, window, axis)
    b=func(x1,-1)

    # the rest of the code pads the front and back to return an array of the same shape as the original
    nfront=np.floor((window-1)/2)
    nback=np.floor((window-1)/2)+(window-1)%2
    inxf=[slice(None)]*np.ndim(b)
    inxf[axis]=slice(0,1,1)
    inxb=[slice(None)]*np.ndim(b)
    inxb[axis]=slice(np.shape(b)[axis]-1,np.shape(b)[axis],1)
    repsf=np.ones(np.ndim(b),dtype=int)
    repsf[axis]=int(nfront)
    repsb=np.ones(np.ndim(b),dtype=int)
    repsb[axis]=int(nback)
    x2=np.concatenate((np.tile(b[tuple(inxf)],repsf),b,np.tile(b[tuple(inxb)],repsb)),axis=axis)
    return x2

def mkdirs(fsave):
    saveloc=os.path.dirname(fsave)
    if not os.path.exists(saveloc):
        try:
            os.makedirs(saveloc)
        except FileExistsError:
            pass # in case other code running at the same time got to it first
    return

def stdFilt(tdt,vals,window_days,nstdev=3):
    """
    input:
    tdt is array of datetimes or in units of days past reference time
    vals is array of floats
    window_days is window half-width (search from t0-window_days to t0+window_days)
    nstdev is the number of standard deviations beyond which to exclude the data point
    returns:
    val with outliers converted to NaN
    """
    omeans=np.empty(len(vals))
    ostdvs=np.empty(len(vals))
    if isinstance(tdt[0],dt.datetime):
        tdelt=dt.timedelta(days=window_days)
    else:
        warnings.warn('Warning: assuming tdt is float representing elapsed time in days')
        #raise TypeError('Unexpected type:',type(tdt[0]))
        tdelt=window_days
    for ind in range(0,len(vals)):
        sel=(tdt>=tdt[ind]-tdelt)&(tdt<=tdt[ind]+tdelt)
        sel[ind]=False # exclude actual point
        selection=vals[sel]
        if np.sum(~np.isnan(selection))>0:
            omeans[ind]=np.nanmean(selection)
            ostdvs[ind]=np.nanstd(selection)
        else:
            omeans[ind]=np.nan
            ostdvs[ind]=np.nan
    return np.where(np.abs(vals-omeans)<=nstdev*ostdvs,vals,np.nan)


def boxcar(tdd,vals,window,fun,mindays=None):
    """
    input:
    tdd is array of datetimes as decimal days or equivalent
    vals is array of floats
    window is bocar width in days
    fun is function to apply to vals within window
    mindays is minimum days required in window to return a value (otherwise NaN); if None, set to 2*tdelta+1 (window length)
    returns:
    filtered times, values arrays
    """
    if mindays==None:
        mindays=window
    tday=np.array([int(el) for el in tdd])
    tlist=[]
    vlist=[]
    for ind in range(0,len(vals)):
        # window should extend back one less for even window size
        sel=(tday>=tday[ind]-(int(window/2)-(window+1)%2))&(tday<=tday[ind]+int(window/2))
        if np.sum(sel)>=mindays:
            tlist.append(np.nanmean(tdd[sel]))
            vlist.append(fun(vals[sel]))
    return np.array(tlist),np.array(vlist)


def boxcar2(tdd,vals,window,fun,preserveShape=False):
    # faster version (>10x) but requires regular sampling: no data gaps;
    # use fillblank fxn first to fill in missing entries with nan
    # if preserveShape, return array of same size as input padded with NaNs
    #    - only works for odd windows
    if preserveShape and window%2==0:
        raise Exception('preserveShape option requires odd window length')
    #tday=np.array([int(el) for el in tdd])
    tdd=tdd.reshape((len(tdd),1))
    vals=vals.reshape((len(vals),1))
    #print(np.shape(tdd),np.shape(vals))
    tlist=[]
    vlist=[]
    ix=0
    while ix<window:
        tlist.append(tdd[ix:len(tdd)-1*(window-1-ix)])
        vlist.append(vals[ix:len(tdd)-1*(window-1-ix)])
        ix+=1
    ct=np.concatenate(tlist,1)
    cv=np.concatenate(vlist,1)
    #print(np.shape(ct),np.shape(cv))
    #return ct, cv
    #try:
    #    ctm=np.nanmean(ct,axis=1)
    #    cfv=fun(cv,axis=1)
    #    cfsum=np.sum(~np.isnan(cv),axis=1)
    #    rt=np.where(cfsum>4/5*window,ctm,np.nan*np.ones(np.shape(ctm)))
    #    rval=np.where(cfsum>4/5*window,cfv,np.nan*np.ones(np.shape(cfv)))
    #except:
    #    print(type(ct))
    #    for el in ct.flatten():
    #        print(type(el))
    #    raise
    ### back to original version that requires all values in window to be real
    if preserveShape:
        rt0=np.mean(ct,axis=1,keepdims=True)
        rval0=fun(cv,axis=1,keepdims=True)
        iadd=np.nan*np.ones((int(window/2),1))
        rt=np.concatenate([iadd,rt0,iadd])
        rval=np.concatenate([iadd,rval0,iadd]).reshape(np.shape(vals))
    else:
        rt=np.mean(ct,axis=1)
        rval=fun(cv,axis=1)
    return rt,rval

def fillblank(tdd,vals): # faster!
    dmin=int(np.min(tdd))
    dmax=int(np.max(tdd))
    tday=np.arange(dmin,dmax+1)
    ix=0
    tlist=[]
    vlist=[]
    for ii in tday:
        if int(tdd[ix])==ii:
            tlist.append(tdd[ix])
            vlist.append(vals[ix])
            ix+=1
        else:
            tlist.append(np.nan)
            vlist.append(np.nan)
    return np.array(tlist),np.array(vlist),tday
