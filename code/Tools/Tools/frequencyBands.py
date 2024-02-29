import numpy as np
import warnings


bxfbase='/work/ebo/calcs/buoyCompTS/bxfs/'
optstr={None:'',
        'equiv':'.equiv',
        'HDgrid':'.HDgrid',}

def bxfpath(dsid,mvar,Tvec,freq,base=None,opt=None):#equiv=False):
    if base is None:
        base=bxfbase
    return f"{base}bxf.{mvar}.{dsid}.{'_'.join([str(el) for el in Tvec])}.{freq}{optstr[opt]}.pkl"

def boxcar(times0,vals0,window,fun=np.nanmean):
    # first boxcar filter, recording number of points contributing to nan values
    # fun is function to be applied to windowed data(vals0); defaults to np.nanmean
    # returns: rt: weighted times (always mean), 
    #          rval: filtered values (fun applied), 
    #          notnan: numer of contributing points (always sum), 
    #            filtered values masked where there was no original data
    
    tdd=np.array([*np.nan*np.ones(int(window/2)),*times0,*np.nan*np.ones(int(window/2))])
    vals=np.array([*np.nan*np.ones(int(window/2)),*vals0,*np.nan*np.ones(int(window/2))])
    
    if window%2==0:
        raise Exception('for correct shapes, requires odd window length')
    if len(vals)<window:
        return tdd,np.full(np.shape(vals),np.nan)
    tdd=tdd.reshape((len(tdd),1))
    vals=vals.reshape((len(vals),1))
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
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')
        rt=np.nanmean(ct,axis=1,keepdims=True)
        notnan=np.sum(~np.isnan(cv),axis=1,keepdims=True)
        rval=fun(cv,axis=1,keepdims=True)
    assert len(rval)==len(vals0)

    return rt,notnan,np.squeeze(rval),np.where(np.isnan(np.squeeze(vals0)),np.nan,np.squeeze(rval))

def updateFilt(vals0,window,nns0):
    # running mean weighted by number of points contributing to each previous mean
    vals=np.array([*np.nan*np.ones(int(window/2)),*np.squeeze(vals0),*np.nan*np.ones(int(window/2))])
    nns=np.array([*np.nan*np.ones(int(window/2)),*np.squeeze(nns0),*np.nan*np.ones(int(window/2))])
    
    if window%2==0:
        raise Exception('for correct shapes, requires odd window length')
    if len(vals)<window:
        return np.full(np.shape(vals),np.nan)
    vals=vals.reshape((len(vals),1))
    nns=nns.reshape((len(nns),1))
    vlist=[]
    nlist=[]
    ix=0
    while ix<window:
        vlist.append(vals[ix:len(vals)-1*(window-1-ix)])
        nlist.append(nns[ix:len(vals)-1*(window-1-ix)])
        ix+=1
    cv=np.concatenate(vlist,1)
    cn=np.concatenate(nlist,1)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
        nsum=np.nansum(cn,axis=1,keepdims=True)
        rval=np.nansum(cv*cn,axis=1,keepdims=True)/nsum
    assert len(rval)==len(vals0) 

    return nsum,rval

def boxcar2Step(times0,vals0,window,fun=np.nanmean):
    # first, apply boxcar filter, recording number of points contributing to each filtered value
    # then, update with a smoothing pass of the same window length, 
    # with a weighted average of the filtered values weighted by contributing number of points
    rt,nn, rval, __ = boxcar(times0,vals0,window,fun=fun)
    nsum,rvalf = updateFilt(np.squeeze(rval),window,nn)
    diags={'rt':rt,'nn':nn,'rval':rval,'nsum':nsum}
    rvalf[np.isnan(vals0)]=np.nan # don't extrapolate
    return rvalf, diags

def bandcalcs(Tvec,tt,val):
    # apply boxcar filters/differencing at each time scale to a given time series
    # input Tvec contains periods to consider, eg Tvec=[7,31,365]
    # tt is time in days
    # returns cs: filtered time series (boxcar mean applied sequentially from highest to lowest freq)
    #      bands: differenced time series
    c3=dict()
    v3=dict() # collect variance of each band at next timescale
    band3=dict()
    last3=val
    for ii in range(0,len(Tvec)):
        c3[ii], __ = boxcar2Step(tt,last3,Tvec[ii],fun=np.nanmean)
        band3[ii]=last3-np.squeeze(c3[ii])
        last3=np.squeeze(c3[ii])
        __,__,__,v3=boxcar(tt,band3[ii],Tvec[ii],fun=np.nanvar)
    band3[ii+1]=np.squeeze(c3[ii])
    return c3,band3,v3

