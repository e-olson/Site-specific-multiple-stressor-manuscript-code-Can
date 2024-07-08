import sys
import os
import socket
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as scopt
import commonfxns as cf, OAPBuoyData as OAP, diagsPP, panfxns as pf
import netCDF4 as nc
import xarray as xr
import dask.array as da
import cftime
import datetime as dt
import cmocean
import gsw
from sklearn.linear_model import TheilSenRegressor
import warnings

# adapt fits from evalfxns.py
#### define fits ############################
class basicfit:
    def fromdt(self,tdt):
        if self.calendar=='noleap':
            tdays=cf.to_exact_tind(cf.dt_to_cftnoleap(tdt),torig=self.torig,calendar=self.calendar)
        elif self.calendar=='standard':
            tdays=cf.to_exact_tind(tdt,torig=self.torig,calendar=self.calendar)
        return tdays
    def checktime(self,tdays):
        tchk= tdays[0] if hasattr(tdays,'__len__') else tdays
        if type(tchk)==dt.datetime:
            tdays=self.fromdt(tdays)
            tchk= tdays[0] if hasattr(tdays,'__len__') else tdays
        if not (tchk.calendar==self.calendar and tchk.torig==self.torig):
            raise TypeError('mismatched calendars')
        return tdays
    
    def linpred(self,tdays):
        tdays=self.checktime(tdays)
        rt=len(tdays.shape)
        rc=len(np.shape(self.coef[1,...]))
        return self.coef[0,...]+np.expand_dims(self.coef[1,...],0)*np.expand_dims(tdays,tuple(np.arange(rt,rt+rc)))
    def seaspred(self,yrind):
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([self.seas[iyd-1,...] for iyd in yrind])
    def fullpred(self,tdays,yrind):
        # linear trend + seasonality
        tdays=self.checktime(tdays)
        #if not (tdays[0].calendar==self.calendar and tdays[0].torig==self.torig):
        #    raise TypeError('mismatched calendars')
        yrind[yrind==366]=365 # handle possible leap years
        return self.linpred(tdays)+self.seaspred(yrind)

    def deseas(self,yrind,vals):
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([ival-self.seas[iyd-1,...] for iyd, ival in zip(yrind,vals)])
    def detrend(self,tdays,vals):
        tdays=self.checktime(tdays)
        #if not (tdays[0].calendar==self.calendar and tdays[0].torig==self.torig):
        #    raise TypeError('mismatched calendars')
        return vals-self.linpred(tdays)
    def targetdetrend(self,tdays,vals,target):
        tdays=self.checktime(tdays)
        target=self.checktime(target)
        rt=len(tdays.shape)
        rc=len(np.shape(self.coef[1,...]))
        return np.asarray(vals)-np.expand_dims(self.coef[1,...],0)*np.expand_dims(tdays-target,tuple(np.arange(rt,rt+rc)))
    def targetdetrenddeseas(self,tdays,yrind,vals,target):
        tdays=self.checktime(tdays)
        target=self.checktime(target)
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,self.targetdetrend(tdays,vals,target))])
    
    def __repr__(self):
        t = type(self)
        attlist=[el for el in self.__dict__.keys()]
        return (f"{t.__module__}.{t.__qualname__}(\n\t coef={self.coef}, "
                    f"fit output={self.out}, time calendar={self.calendar}, "
                    f"\n\t all attributes:{','.join(attlist)})")
    
class linfit(basicfit):
    def __init__(self,tdays,vals):
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        fit=cf.linreg(tdays.astype(float),vals)
        self.out=fit
        self.coef=fit.coef
        
class binseas(basicfit):
    def __init__(self,yind,vals,yrlen):
        self.yrlen=yrlen
        seas=binclim(yind,vals,yrlen)
        self.seas=seas-np.mean(seas,0) # keep the constant in the linear fit, not the seasonality
        
class binseas2(basicfit):
    def __init__(self,vals,yrlen):
        self.yrlen=yrlen
        seas=binclim2(vals,yrlen)
        self.seas=seas-np.mean(seas,0) # keep the constant in the linear fit, not the seasonality
        
def binclim(yind,vals,yrfac):
    # yind: day or month of year
    # vals: values at each day or month in series; first index must be time
    # yrfac: yind units per year (12 if yind is month of year)
    vals=np.asarray(vals)
    yind=np.asarray(yind)
    sh=list(vals.shape)
    clim=np.empty([yrfac]+sh[1:])
    for ii in range(0,yrfac):
        clim[ii]=np.mean(vals[yind==ii+1,...],0)
    return clim

def binclim2(vals,yrfac):
    # vals: values at each day or month in series; first index must be time
    # yrfac: yind units per year (12 if yind is month of year)
    vals=np.asarray(vals)
    sh=list(vals.shape)
    clim=np.empty([yrfac]+sh[1:])
    for ii in range(0,yrfac):
        clim[ii]=np.mean(vals[ii::yrfac,...],0)
    return clim

targettime_hist=dt.datetime(1988,1,1,0)+dt.timedelta(days=.286*365)
targettime_future=dt.datetime(2061,1,1,0)+(targettime_hist-dt.datetime(2014,1,1,0))
yrspan_future=[2061,2100]
scenDefaults={'ESM4_historical_D1':([1975,2014],targettime_hist),
            'ESM4_ssp119_D1':(yrspan_future,targettime_future),
            'ESM4_ssp126_D1':(yrspan_future,targettime_future),
            'ESM4_ssp245_D1':(yrspan_future,targettime_future),
            'ESM4_ssp245_D151':(yrspan_future,targettime_future),
            'ESM4_ssp245_D201':(yrspan_future,targettime_future),
            'ESM4_ssp370_D1':(yrspan_future,targettime_future),
            'ESM4_ssp534-over_D1':(yrspan_future,targettime_future),
            'ESM4_ssp585_D1':(yrspan_future,targettime_future)}

def calc_MMM_NOAA(iscen0,yrspan0,targettime):
    # iscen0 is scenario, eg 'ESM4_historical_D1'
    # yrspan0 contains first and last year (inclusive), eg [1975,2014]
    # targettime is date used in trend correction
    # (tref is just a date for the time index)
    ivar='tos'
    freq='monthly'
    subdir='ocean_monthly_1x1deg'
    yrspan1=diagsPP.dictScenDates[iscen0]
    files=diagsPP.genflist(ivar, freq, iscen0, yrspan0, subdir)
    clfiles = not os.path.isfile('/work/Elise.Olson'+files[0])
#     try: # check my run first
#         files=diagsPP.listFiles(iscen0,ftype='pp',segment='slice',ppvar=ivar,
#                                 freq=freq,subdir=subdir,yrlims=yrspan0)
#     except: # if not go to archive
#         files=diagsPP.listFiles(iscen0,ftype='pp',segment='base',ppvar=ivar,
#                                 freq=freq,subdir=subdir,yrlims=yrspan0)

    filesonwork=pf.cpfromTape(files)

    ylen=12
    tref=dt.datetime(yrspan0[0],1,1) # times referenced to start of series
    with xr.open_mfdataset(filesonwork).sel(time=slice(cftime.datetime(yrspan0[0],1,1,calendar='noleap'),
                                                       cftime.datetime(yrspan0[-1]+1,1,1,calendar='noleap'))) as ds:
        mod_tnl=ds.time.values # already on noleap calendar
        mod_val=np.array(ds['tos'])
    pf.clearFromWork(filesonwork)
    mod_tdt=cf.cftnoleap_to_dt(mod_tnl) # datetimes for plotting
    mod_tind=cf.to_exact_tind(mod_tnl,torig=tref)
    mod_ym=cf.monthfromdt(mod_tdt)
    mod_yind=mod_ym

    seas0=binseas(mod_ym,mod_val,ylen)
    dsea=seas0.deseas(mod_ym,mod_val)
    lf=linfit(mod_tind,dsea)
    newvals=lf.targetdetrend(mod_tind,mod_val,targettime)
    seasF=binclim(mod_ym,newvals,ylen)
    MMM=np.max(seasF,0)

    # save:
    dataar=xr.DataArray(MMM,coords=(ds.lat,ds.lon),name='MMM',
                        attrs={'long_name':'Maximum Monthly Mean','units':'degC'})
    fname=f'/work/ebo/calcs/extremes/MMM_NOAA/MMM.{iscen0}.{yrspan0[0]}_{yrspan0[1]}.{targettime:%Y%m%d}.nc'
    print(fname)
    dataar.to_netcdf(fname)
    
    if clfiles:
        pf.clearFromWork(filesonwork,verb=True)
        
    return mod_tdt, mod_tind, mod_val,newvals,lf, seasF, MMM

def load_MMM_NOAA(iscen0,yrspan0,targettime):
    fname=f'/work/ebo/calcs/extremes/MMM_NOAA/MMM.{iscen0}.{yrspan0[0]}_{yrspan0[1]}.{targettime:%Y%m%d}.nc'
    if os.path.isfile(fname):
        with xr.open_dataset(fname) as xf:
            MMM=np.array(xf.MMM)
    else:
        mod_tdt, mod_tind, mod_val,newvals,lf, seasF, MMM = calc_MMM_NOAA(iscen0,yrspan0,targettime)
    return MMM


def calcDHW(iscen0,iscenMMM,yrspanMMM,dtargetMMM=[],save=True):
    ivar='tos'
    subdir='ocean_daily_1x1deg'
    freq='daily'
    try: # check my run first
        files=diagsPP.listFiles(iscen0,ftype='pp',segment='slice',ppvar=ivar,
                                freq=freq,subdir=subdir,yrlims=yrspan0)
    except: # if not go to archive
        files=diagsPP.listFiles(iscen0,ftype='pp',segment='base',ppvar=ivar,
                                freq=freq,subdir=subdir,yrlims=yrspan0)
    filesonwork=pf.cpfromTape(files)
    if len(yrspan0)>0:
        ds=xr.open_mfdataset(filesonwork).sel(time=slice(f"{yrspan0[0]}-01-01",f"{yrspan0[1]+1}-01-01"))
    else:
        ds=xr.open_mfdataset(filesonwork)
    MMM=load_MMM_NOAA(iscenMMM,yrspanMMM,yrspanMMM)
    HS1=HS(ds.tos[:5,:,:],MMM)
    # calculate:
    DHW1=DHW(ds.tos,MMM)
    time=ds.time[12*7-1:].data
    dsout = xr.Dataset(data_vars=dict(DHW=(["time", "lat", "lon"], DHW1)),
                       coords=dict(lon=(["lon"],ds.lon.data),lat=(["lat"],ds.lat.data),time=(["time"],time)),
                       attrs=dict(description=f"Degree heating weeks: {iscen0} with MMM based on {iscenMMM}"))
    fname=f'/work/ebo/calcs/extremes/MMM_NOAA/DHW.{iscen0}.{yrspan0[0]}_{yrspan0[1]}.MMM{iscenMMM}.nc'
    print(fname)
    if save:
        dsout.to_netcdf()
    return DHW1

############################### saving means, variance, etc ###################################
listScenDates = diagsPP.listScenDates # moved
dictScenDates = diagsPP.dictScenDates # moved

def xrload(ivar, freq, iscen, yrspan, subdir, mfds_kw):
    # note: returned set is inclusive of end year yrspan[1] as well as start year yrspan[0]
    files=diagsPP.genflist(ivar, freq, iscen, yrspan, subdir)
    clfiles = not os.path.isfile('/work/Elise.Olson'+files[0])
    filesonwork=pf.cpfromTape(files)
    ds=xr.open_mfdataset(filesonwork,**mfds_kw).sel(time=slice(cftime.datetime(yrspan[0],1,1,calendar='noleap'),
                                                              cftime.datetime(yrspan[-1]+1,1,1,calendar='noleap')))
    return ds, filesonwork, clfiles

def fnameMultiVar(iscen,yrspan,freq):
    if type(iscen)==list or type(iscen)==tuple:
        clist=[]
        for ind, iiscen in enumerate(iscen):
            clist.append(iiscen)
            clist.append(f"{yrspan[2*ind]}_{yrspan[2*ind+1]}")
    else:
        clist=[iscen,f"{yrspan[0]}_{yrspan[-1]}"]
    return '.'.join(clist)+'.'+diagsPP.trfreq(freq)

sliceStatsPath='/work/ebo/calcs/extremes/sliceStats/'
def getvarpath(iscen,yrspan,freq):
    return sliceStatsPath+'variance/variance.'+fnameMultiVar(iscen,yrspan,freq)+'.nc'
def getmeanpath(iscen,yrspan,freq):
    return sliceStatsPath+'mean/mean.'+fnameMultiVar(iscen,yrspan,freq)+'.nc'

def nchasvar(ivar,fname):
    with xr.open_dataset(fname) as ff:
        s=ivar in ff.keys()
    return s

def calcMeanVar(ivar,freq,iscen,yrspan,subdir,recalc=False):
    meanpath=getmeanpath(iscen,yrspan,freq)
    varpath=getvarpath(iscen,yrspan,freq)
    calcMean=recalc or not os.path.isfile(meanpath) or not nchasvar(ivar,meanpath)
    calcVar=recalc or not os.path.isfile(varpath) or not nchasvar(ivar,varpath)
    print(f"{dt.datetime.now()}\n{meanpath}\n{varpath}")
    if calcMean or calcVar:
        ds, filesonwork, clfiles=xrload(ivar,freq,iscen,yrspan,subdir,mfds_kw={'chunks':{'time':-1,'lon':180,'lat':180}})
        print(ds[ivar])
    else:
        clfiles=False
        print(f"Mean and variance for {iscen}, {ivar}, {freq}, {yrspan} already stored.")
    if calcMean:
        print(f"{dt.datetime.now()} Calculating mean: {iscen}, {ivar}, {freq}, {yrspan}")
        imean=ds[ivar].mean(dim='time',keepdims=True)
        imean.to_netcdf(meanpath,mode='a')
    if calcVar:
        print(f"{dt.datetime.now()} Calculating variance: {iscen}, {ivar}, {freq}, {yrspan}")
        ivar=ds[ivar].var(dim='time',keepdims=True)
        ivar.to_netcdf(varpath,mode='a')
    if clfiles:
        pf.clearFromWork(filesonwork,verb=True)
    print(f"{dt.datetime.now()} - done")
    return

def save_bcfitvar(ivar,iscen):
    # set up for daily files
    scenName=diagsPP.scenNameDict[iscen]
    _vardirdict={'tos':'ocean_daily_1x1deg',
                'sos':'ocean_daily_1x1deg',
                'spco2':'ocean_cobalt_daily_sfc_1x1deg',
                'phos':'ocean_cobalt_daily_sfc_1x1deg',
                'o2os':'ocean_cobalt_daily_sfc_1x1deg',
                'chlos':'ocean_cobalt_omip_daily_1x1deg',}
    ds1,flist,__=xrload(ivar,'daily',iscen,diagsPP.dictScenDates[iscen],_vardirdict[ivar],
                        mfds_kw={'chunks':{'time':17155,'lon':90,'lat':90}})
    mod_tnl=ds1.time.values # already on noleap calendar
    mod_val=ds1[ivar].data
    mod_tdt=cf.cftnoleap_to_dt(mod_tnl) # datetimes for plotting
    mod_tind=cf.to_exact_tind(mod_tnl,torig=mod_tdt[0])
    mod_yd=cf.ydNLfromcftNL(mod_tnl)

    fit0=linfit(mod_tind,mod_val)
    yest=da.from_array(fit0.coef[0,...]+np.expand_dims(fit0.coef[1,...],0)*\
                        np.expand_dims(mod_tind.astype(float),(1,2)),chunks=(17155,90,90))
    fitseas=binseas2(mod_val-yest,365)
    
    def _add_dims(arr,tarr):
        while len(np.shape(arr))<len(np.shape(tarr)):
            arr=np.expand_dims(arr,-1)
        return arr
    def _gsmooth(YD,vals,L=30,yearlen=365):
        # YD is day of year (can include fractional component)
        # dim 0 must be time
        allt=np.arange(1,yearlen+1)
        fil=np.empty((len(allt),)+np.shape(vals)[1:])
        s=L/2.355
        for t in allt:
            diff=np.min([np.abs(YD-t),np.abs(YD-t+yearlen), np.abs(YD-t-yearlen)],0)
            weight=_add_dims(np.array([np.exp(-.5*x**2/s**2) if x <=3*L else 0.0 for x in diff]),vals)
            fil[t-1,...]=np.divide(np.nansum(weight*vals,0),np.nansum(weight*~np.isnan(vals),0),
                                   out=np.nan*da.array(np.ones(np.shape(vals)[1:])),
                                   where=np.nansum(weight*~np.isnan(vals),0)>0)
        return fil-np.nanmean(fil)
    
    seasgs=_gsmooth(np.arange(1,366),fitseas.seas)
    yds=np.array([ival-seasgs[iyd-1,...] for iyd, ival in zip(mod_yd,mod_val)])
    fit=linfit(mod_tind,yds)
    yest2=da.from_array(fit.coef[0,...]+np.expand_dims(fit.coef[1,...],0)*\
                        np.expand_dims(mod_tind.astype(float),(1,2)),chunks=(17155,90,90))
    var_dtds=np.var(yds-yest2,axis=0)
    var_seas=np.var(seasgs,axis=0)
    dsout=xr.Dataset(dict(linfitcoef=(('b','lat','lon'),fit.coef),binseas=(('YD','lat','lon'),fitseas.seas),
                         gsmoothseas=(('YD','lat','lon'),seasgs),var_seas=(('lat','lon'),var_seas),
                          var_dtds=(('lat','lon'),var_dtds)),
                    attrs={'base units':ds1[ivar].units,'trend units':ds1[ivar].units+' day-1',
                           'form':'yest=gsmoothseas(YD)+linfitcoef[0]+linfitcoef[1]*time_days',
                           'sourcefiles':flist,'savetime':f'{dt.datetime.now():"%Y%m%d %H:%M:%s"}'},
                    coords={'lat':ds1.lat,'lon':ds1.lon,'YD':np.arange(1,366)})
    savepath=f'/work/ebo/calcs/extremes/bcfitvar/bcfitvar.{scenName}.{ivar}.nc'
    cf.mkdirs(savepath)
    dsout.to_netcdf(savepath)
    ds1.close()
    return

if __name__ == '__main__':
    if sys.argv[1]=='save_bcfitvar':
        iscen=('ESM4_historical_D1', 'ESM4_ssp245_D1_histCont')
        for ivar in ['tos','sos','spco2','phos','o2os','chlos']:
            save_bcfitvar(ivar,iscen)
    else:
        raise NotImplementedError("input:",sys.argv)
