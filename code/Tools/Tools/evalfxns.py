import sys
import os
import socket
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as scopt
from Tools import commonfxns as cf, OAPBuoyData as OAP
import netCDF4 as nc
import cftime
import datetime as dt
import cmocean
import gsw
from sklearn.linear_model import TheilSenRegressor
import warnings

version = '0.3' # start storing version info in pickles'
maxSeasGapDays = 60 # maximum gap in year-days allowed to still compute seasonal cycle unless ignoreSeasCrit=True

dayperyro=365.25
dayperyrm=365 # no leap years

if 'stellar' in socket.gethostname():
    defaultSaveLoc='/scratch/cimes/eo2651/calcs/timeSeriesComps/'
else:
    defaultSaveLoc='/work/Elise.Olson/calcs/timeSeriesComps/'
locFinalFigs='/work/Elise.Olson/calcs/evalFigs/' # location to save figs for presentation

def figpath(figtype,figsaveloc,stid,mvar=None,freq=None):
    if figtype=='plot_ts':
        fname=f'plot_ts.{stid}.{mvar}.{freq}.png'
        fpath=figsaveloc+'plot_ts/'+fname
    elif figtype=='map':
        fname=f'map.{stid}.png'
        fpath=figsaveloc+'map/'+fname
    elif figtype=='plot_seas_b':
        fname=f'plot_seas_b.{stid}.{mvar}.{freq}.png'
        fpath=figsaveloc+'plot_seas_b/'+fname
    elif figtype=='plot_trendhist':
        fname=f'plot_trendhist.{stid}.{mvar}.{freq}.png'
        fpath=figsaveloc+'plot_trendhist/'+fname
    else:
        raise NotImplementedError('need to define fig path for this figtype')
    return fpath

def getcompsavepath(compsaveloc,stationID,modvar,freq):
    return compsaveloc+f"timeSeriesComp.{stationID}.{modvar}.{freq}.pkl"

def loadComp(compsaveloc,stationID,modvar,freq='daily'):
    ppath=getcompsavepath(compsaveloc,stationID,modvar,freq)
    with open(ppath, 'rb') as hh:
        mmm=pickle.load(hh)
    return mmm
        
# WSS
def WSS(obs,mod):
    # Willmott skill core, cannot include any NaN values
    return 1.0-np.sum((mod-obs)**2)/np.sum((np.abs(mod-np.mean(obs))+np.abs(obs-np.mean(obs)))**2)

def RMSE(obs,mod):
    # root mean square error, cannot include any NaN values
    return np.sqrt(np.sum((mod-obs)**2)/len(mod))

def Rsq(obs,mod):
    fit=cf.linreg(obs,mod)
    Rsq=fit.Rsq
    return Rsq

def binseas(YD,vals,ilen=366):
    YD=np.array(YD)
    vals=np.array(vals)
    mcycle=np.empty([ilen,])
    scycle=np.empty([ilen,])
    ncycle=np.empty([ilen,])
    for ii in range(1,ilen+1):
        iind=YD==ii
        if np.sum(iind)>0:
            mcycle[ii-1]=np.nanmean(vals[YD==ii])
            scycle[ii-1]=np.nanstd(vals[YD==ii])
            ncycle[ii-1]=np.sum(YD==ii)
        else:
            mcycle[ii-1]=np.nan
            scycle[ii-1]=np.nan
            ncycle[ii-1]=0
    return {'mean':mcycle,'std':scycle,'N':ncycle}
    #return mcycle
    # require 95% coverage:
    #if sum(~np.isnan(mcycle))>=.95*len(mcycle):
    #    return mcycle-np.nanmean(mcycle)
    #else:
    #    return np.nan*mcycle


#        with warnings.catch_warnings():
#            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# def gsmooth(YD,vals,L,yearlen=365,freq='daily'):
#     # for daily frequency YD should be year day; for monthly, month
#     iii=~np.isnan(vals)
#     iYD=YD[iii]
#     ivals=vals[iii]
#     allt=np.arange(1,yearlen+1)
#     fil=np.empty(np.size(allt))
#     if ((len(np.unique(iYD))<365-60) and (freq=='daily')) or ((len(np.unique(iYD))<10) and (freq=='monthly')): # more than 60 days missing
#         return np.nan*fil
#     else:
#         s=L/2.355
#         for t in allt:
#             diff=[min(abs(x-t),abs(x-t+yearlen), abs(x-t-yearlen)) for x in iYD]
#             weight=[np.exp(-.5*x**2/s**2) if x <= 3*L else 0.0 for x in diff]
#             fil[t-1]=np.sum(weight*ivals)/np.sum(weight) if np.sum(weight)>0 else np.nan
#         return fil-np.nanmean(fil)

class varDef: # for defining variable info: original data name, display name, display name, display name+units
    def __init__(self,dfkey,dispName,dispUnits,dispNameUnits):
        self.dfkey=dfkey
        self.dispName=dispName
        self.dispUnits=dispUnits
        self.dispNameUnits=dispNameUnits
        
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
        return self.coef[0]+self.coef[1]*tdays
    def seaspred(self,yrind):
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([self.seas[iyd-1] for iyd in yrind])
    def fullpred(self,tdays,yrind):
        # linear trend + seasonality
        tdays=self.checktime(tdays)
        #if not (tdays[0].calendar==self.calendar and tdays[0].torig==self.torig):
        #    raise TypeError('mismatched calendars')
        yrind[yrind==366]=365 # handle possible leap years
        return self.coef[0]+self.coef[1]*tdays+np.array([self.seas[iyd-1] for iyd in yrind])
    def deseas(self,yrind,vals):
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,vals)])
    def detrend(self,tdays,vals):
        tdays=self.checktime(tdays)
        #if not (tdays[0].calendar==self.calendar and tdays[0].torig==self.torig):
        #    raise TypeError('mismatched calendars')
        return vals-(self.coef[0]+self.coef[1]*tdays)
    def targetdetrend(self,tdays,vals,target):
        tdays=self.checktime(tdays)
        target=self.checktime(target)
        return vals+self.coef[1]*(target-tdays)
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
# not finished:         
# class linfit0(basicfit):
#     def __init__(self,tdays,yrind,vals,yrlen):
#         # 1) calculate seasonal climatology
#         # 2) calculate trend based on deseasoned data
#         # 3) re-calculate climatology
#         #yrlen is # of yrind units (days or months) per year
#         self.calendar=tdays[0].calendar
#         self.torig=tdays[0].torig
#         self.yrlen=yrlen
#         # now strip calendar info
#         tdays=tdays.astype(float)
#         yrind[yrind==366]=365 # a way to handle leap years
#         # first cut seasonal climatology
#         clim0=np.empty(yrlen)
#         for ii in range(0,12):
#             clim0[ii]=np.mean(vals[yrind==ii+1])
#         deseas0=np.array([el-clim0[ym-1] for el,ym in zip(vals,yrind)])
#         fit=cf.linreg(tdays.astype(float),deseas0)
#         self.out=fit
#         self.coef=fit.coef
#         self.seas=clim0
#         self.statsout=cf.linfitstats(tdays,np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,vals)]),fit.coef)

class linfit1(basicfit):
    def __init__(self,tdays,yrind,vals,yrlen,):
        # 1) starting and ending in same season, estimate linear fit
        # 2) calculate seasonal climatology
        # re-calculate linear fit
        #yrlen is # of yrind units (days or months) per year
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        self.yrlen=yrlen
        # now strip calendar info
        tdays=tdays.astype(float)
        yrind[yrind==366]=365 # a way to handle leap years
        # first cut for estimating seasonal cycle; start and end in same season
        iii=tdays<(tdays[-1]-(tdays[-1]-tdays[0])%yrlen)
        fit0=cf.linreg(tdays[iii],vals[iii])
        seas=gsmooth(yrind,vals-(fit0.coef[0]+fit0.coef[1]*tdays),L=yrlen/12,yearlen=yrlen)
        deseas=np.array([ival-seas[iyd-1] for iyd, ival in zip(yrind,vals)])
        fit=cf.linreg(tdays,deseas)
        self.out=fit
        self.coef=fit.coef
        self.seas=seas
        self.statsout=cf.linfitstats(tdays,np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,vals)]),fit.coef)
        
class optfit(basicfit):
    def res_slope_seas(self,coefs,xtind,yrind,yval,yrlen):
        yrind[yrind>365]=365 # only occurs if daily and leap year
        detr=yval-(coefs[0]+coefs[1]*xtind)
        gs=gsmooth(yrind,detr,L=yrlen/12,yearlen=yrlen)
        # subtract mean (in gsmooth), otherwise can't fit intercept
        resid=[ival - gs[int(iy)-1] for iy, ival in zip(yrind,detr)]
        return np.array(resid)
    def __init__(self,tdays,yrind,vals,yrlen):
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        self.yrlen=yrlen
        # now strip calendar info
        tdays=tdays.astype(float)
        optf=scopt.least_squares(self.res_slope_seas,[0,0],
                    args=(tdays,yrind,vals,yrlen),
                    verbose=1,ftol=1e-15)
        self.out=optf
        self.coef=optf.x
        self.seas=gsmooth(yrind,vals-(optf.x[0]+optf.x[1]*tdays),L=yrlen/12,yearlen=yrlen)
        self.statsout=cf.linfitstats(tdays,np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,vals)]),optf.x)

class TheilSen(basicfit):
    def __init__(self,tdays,yrind,vals,yrlen):
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        self.yrlen=yrlen
        # now strip calendar info
        tdays=tdays.astype(float)
        yrind[yrind==366]=365 # a way to handle leap years
        # first cut for estimating seasonal cycle; start and end in same season
        iii=tdays<(tdays[-1]-(tdays[-1]-tdays[0])%365)
        mod_thsfit0=TheilSenRegressor(random_state=0,copy_X=True,
                      fit_intercept=False).fit(tdays[iii].reshape(-1, 1), vals[iii])
        seas=gsmooth(yrind,vals-mod_thsfit0.coef_*tdays,L=yrlen/12,yearlen=yrlen)
        deseas=np.array([ival-seas[iyd-1] for iyd, ival in zip(yrind,vals)])
        mod_thsfit=TheilSenRegressor(random_state=0,copy_X=True,
                      fit_intercept=False).fit(tdays.reshape(-1, 1), deseas)
        self.out=mod_thsfit
        self.coef=[np.nan,mod_thsfit.coef_]
        self.seas=seas

class mixfitobs(basicfit):
    def __init__(self,modfit,tdays_b,yrind,vals,yrlen):
        self.calendar=modfit.calendar
        self.torig=modfit.torig
        self.yrlen=yrlen
        if not (tdays_b[0].calendar==self.calendar and tdays_b[0].torig==self.torig):
            raise TypeError('mismatched calendars')
        tdays_b=tdays_b.astype(float)
        self.seas=gsmooth(yrind,vals-modfit.coef[1]*tdays_b,L=yrlen/12,yearlen=yrlen) # don't worry about intercept
        self.coef=modfit.coef
        self.out='' # avoid error
        
class defseaslinfit(basicfit):
    def __init__(self,tdays,yrind,vals,yrlen,seas):
        #yrlen is # of yrind units (days or months) per year
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        self.yrlen=yrlen
        # now strip calendar info
        tdays=tdays.astype(float)
        yrind[yrind==366]=365 # a way to handle leap years
        # first cut for estimating seasonal cycle; start and end in same season
        deseas=np.array([ival-seas[iyd-1] for iyd, ival in zip(yrind,vals)])
        fit=cf.linreg(tdays,deseas)
        self.out=fit
        self.coef=fit.coef
        self.seas=seas
        self.statsout=cf.linfitstats(tdays,np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,vals)]),fit.coef)

class quadfit(basicfit):
    def poly2(self,b,x):
        return b[0]+b[1]*x+b[2]*x**2
    def res(self,b,x,y):
        return y-self.poly2(b,x)
    def poly2fix(self,b,b2,x):
        return self.poly2([b[0],b[1],b2],x)
    def resfix(self,b,b2,x,y):
        return y-self.poly2fix(b,b2,x)
    def __init__(self,tdays,yrind,vals,yrlen,b2=None):
        self.calendar=tdays[0].calendar
        self.torig=tdays[0].torig
        self.yrlen=yrlen
        # now strip calendar info
        tdays=tdays.astype(float)
        yrind[yrind==366]=365 # a way to handle leap years
        if b2 is None: # fit all 3 coefs
            test=self.res(np.array([0,0,0]),tdays,vals)
            mopt0=scopt.least_squares(self.res,[0,0,0],args=(tdays,vals),verbose=1)
            mseas0=gsmooth(yrind,vals-self.poly2(mopt0.x,tdays),L=yrlen/12,yearlen=yrlen)
            mds=np.array([ival-mseas0[iyd-1] for iyd, ival in zip(yrind,vals)])
            opt=scopt.least_squares(self.res,mopt0.x,args=(tdays,mds),verbose=1)
            seas=gsmooth(yrind,vals-self.poly2(opt.x,tdays),L=yrlen/12,yearlen=yrlen)
            self.coef=opt.x
        else: # fix quadratic term coef to b2 value specified
            oopt0=scopt.least_squares(self.resfix,[0,0],args=(b2,tdays,vals),verbose=1)
            oseas0=gsmooth(yrind,vals-self.poly2fix(oopt0.x,b2,tdays),L=yrlen/12,yearlen=yrlen)
            ods=np.array([ival-oseas0[iyd-1] for iyd, ival in zip(yrind,vals)])
            opt=scopt.least_squares(self.resfix,oopt0.x,args=(b2,tdays,ods),verbose=1)
            seas=gsmooth(yrind,vals-self.poly2fix(opt.x,b2,tdays),L=yrlen/12,yearlen=yrlen)
            self.coef=np.array([opt.x[0],opt.x[1],b2])
        self.out=opt
        self.seas=seas
        self.statsout=cf.basicfitstats(self.poly2(self.coef,tdays)+np.array([seas[iyd-1] for iyd in yrind]),vals,self.coef)
    # override parent class methods to support nonlinear equation:
    def linpred(self,tdays): # no longer linear, but annoying to change name now
        tdays=self.checktime(tdays)
        return self.poly2(self.coef,tdays.astype(float))
    def fullpred(self,tdays,yrind):
        # trend + seasonality
        tdays=self.checktime(tdays)
        yrind[yrind==366]=365 # handle possible leap years
        return self.poly2(self.coef,tdays.astype(float))+np.array([self.seas[iyd-1] for iyd in yrind])
    def detrend(self,tdays,vals):
        tdays=self.checktime(tdays)
        return vals-self.poly2(self.coef,tdays.astype(float))
    def targetdetrend(self,tdays,vals,target):
        tdays=self.checktime(tdays)
        target=self.checktime(target)
        return vals-self.poly2(self.coef,tdays.astype(float))+self.poly2(self.coef,target.astype(float))
    def targetdetrenddeseas(self,tdays,yrind,vals,target):
        tdays=self.checktime(tdays)
        target=self.checktime(target)
        yrind[yrind==366]=365 # handle possible leap years
        return np.array([ival-self.seas[iyd-1] for iyd, ival in zip(yrind,self.targetdetrend(tdays,vals,target))])

fitmethods={'linfit1':linfit1,'optfit':optfit,'TheilSen':TheilSen,'defseaslinfit':defseaslinfit,'quadfit':quadfit}
###################################################################################3

def gsmooth(YD,vals,L=30,yearlen=365):
    # YD is day of year (can include fractional component)
    iii=~np.isnan(vals)
    iYD=YD[iii]
    ivals=vals[iii]
    allt=np.arange(1,yearlen+1)
    fil=np.empty(np.size(allt))
    s=L/2.355
    for t in allt:
        diff=np.min([np.abs(iYD-t),np.abs(iYD-t+yearlen), np.abs(iYD-t-yearlen)],0)
        weight=[np.exp(-.5*x**2/s**2) if x <=3*L else 0.0 for x in diff]
        fil[t-1]=np.sum(weight*ivals)/np.sum(weight) if np.sum(weight)>0 else np.nan
    return fil-np.nanmean(fil)

def largestSeasGap(yd):
    # for checking seasonal coverage of data
    yd=np.sort(np.unique(yd))
    yd=np.concatenate([yd-365,yd,yd+365])
    return np.max(np.diff(yd))

#def res_slope_seas(coefs,xtind,yval,yrind,mlen,yrfacmod,freq):
#    yrind[yrind>365]=365 # only occurs if daily
#    detr=yval-(coefs[0]+coefs[1]*xtind)
#    gs=gsmooth(yrind,detr)#,mlen,yearlen=yrfacmod,freq=freq)
#    #gs=gs-np.mean(gs) # otherwise can't fit intercept
#    resid=[ival - gs[int(iyd)-1] for iyd, ival in zip(yrind,detr)]
#    return np.array(resid)

class ModError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ObsError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class timeSeriesComp:
    
    def __init__(self,modvar,stationID,staname,shortID,lat,lon,obsloadfun, obsloadkwargs, modloadfun, modloadkwargs, 
                 freq='daily',tref=dt.datetime(1975,1,1),savepath=defaultSaveLoc,figsavepath=None,compsavepath=None):
        # modvar: variable ID use throughout; may correspond to calculated value
        # remember that obs_tind and mod_tind are in different units due to no_leap calendar
        self.modvar=modvar
        self.version=version # version of this file
        self.stationID=stationID
        self.staname=staname
        self.shortTitle=shortID # keep old name to not break code
        self.shortID=shortID
        self.lat=lat
        self.lon=lon
        self.freq=freq if not freq=='month' else 'monthly'
        self.savepath=savepath
        self.figsavepath=figsavepath if figsavepath else savepath+'figs/'
        self.compsavepath=compsavepath if compsavepath else savepath+'comps/'
        #self.yrfaco=365.25 # obs days per year
        #self.yrfacm=365 # noleap days per year
        #self.mlen=30 # month is ~30 days
        self.tref=tref
        if self.freq=='daily':
            self.yrfaco=365.25
            self.yrfacm=365 # noleap
            self.mlen=30 # month is 30 index intervals
        elif self.freq=='monthly':
            self.yrfaco=12
            self.yrfacm=12
            self.mlen=1 # month is 1 index interval
        else: 
             raise NotImplementedError('frequency not accounted for:'+self.freq)
        if obsloadfun is None: # hack for running on model data only without rewriting code
            critO=False
        else:
            critO=True
        # step 1: loadvar_obs, loadvar_mod
        # load obs
        if critO:
            self.obs_tdt,self.obs_val, self.dispName, self.dispUnits, self.dispNameUnits, self.obsvar = obsloadfun(modvar,**obsloadkwargs)
        else:
            self.obs_tdt=np.array([dt.datetime(2010,1,1),])
            self.obs_val=np.array([np.nan,])
            self.dispName = ''
            self.dispUnits = ''
            self.dispNameUnits = ''
            self.obsvar = 'None'
        if np.sum(~np.isnan(self.obs_val))==0:
            critO=False # handle case where Null obs values passed through function to obtain Name/Units variables
        # load model
        self.mod_tnl, self.mod_tdt, self.mod_val = modloadfun(modvar,**modloadkwargs)
        if type(self.mod_val)==np.ma.core.MaskedArray:
            if np.ma.is_masked(self.mod_val[0]):
                raise ModError(f'Model is masked at this location. ({stationID},{modvar})')
            # if masked array, fill:
            self.mod_val=self.mod_val.filled(np.nan)
        self.__settimes__()
        obs_trange=[self.obs_tdt[0],self.obs_tdt[-1]]
        mod_trange=[self.mod_tdt[0],self.mod_tdt[-1]]
        OL_trange=[max(obs_trange[0],mod_trange[0]),min(obs_trange[1],mod_trange[1])]
        trange=[min(obs_trange[0],mod_trange[0]),max(obs_trange[1],mod_trange[1])]

        self.obs_yd=cf.ydfromdt(self.obs_tdt)
        self.obs_ym=cf.monthfromdt(self.obs_tdt)

        if (OL_trange[1]-OL_trange[0]).days>365.25*3: # require 3 yrs overlap & seas cycle res
            iiOLo=(self.obs_tdt>=OL_trange[0])&(self.obs_tdt<=OL_trange[1])
            if largestSeasGap(self.obs_yd[iiOLo])>maxSeasGapDays:
                self.iiOLo=None
            else:
                self.iiOLo=iiOLo
        else:
            self.iiOLo=None
        
        self.mod_yd=cf.ydNLfromcftNL(self.mod_tnl)
        self.mod_ym=cf.monthfromdt(self.mod_tdt)
        if freq=='daily':
            self.mod_yind=self.mod_yd
            self.obs_yind=self.obs_yd
            self.fityrlen=365 # year length to pass to fits (units of yind)
        elif freq=='monthly':
            self.mod_yind=self.mod_ym
            self.obs_yind=self.obs_ym
            self.fityrlen=12 # year length to pass to fits (units of yind)
        if self.iiOLo is not None:# require 3 yrs overlap; use previous check
            self.iiOLm=(self.mod_tdt>=OL_trange[0])&(self.mod_tdt<OL_trange[1])
        else:
            self.iiOLm=None
        self.obs_trange=obs_trange
        self.mod_trange=mod_trange
        self.OL_trange=OL_trange
        self.trange=trange
    def __settimes__(self): # separate so easy to call while unpickling
        self.mod_tind=cf.to_exact_tind(self.mod_tnl,torig=self.tref)
        self.obs_tind=cf.to_exact_tind(self.obs_tdt,torig=self.tref)
        self.obs_tind_b=cf.to_exact_tind(cf.dt_to_cftnoleap(self.obs_tdt),torig=self.tref)
        self.mod_intt=cf.to_int_tind(self.mod_tnl,torig=self.tref,freq=self.freq)
        self.obs_intt=cf.to_int_tind(self.obs_tdt,torig=self.tref,freq=self.freq)

    def calc_fits(self,fitlist=['optfit','linfit1'],fitlistOL=['linfit1'],defaultfit='optfit',
                  predefined=None,ignoreSeasCrit=False):
        # fits
        # predefined should be dictionary of attributes and values to be used instead of calculating those values
        # eg predefined={'obs_gsmooth':np.array([...])}
        # currently the only implemented options are for the key 'obs_gsmooth'
        # or 'obs_b2':'mod_b2' (or alternative set value) as switch to apply model (or other) quadratic term to obs fit for 'quadfit'
        if 'defseaslinfit' in fitlist and 'obs_gsmooth' in predefined.keys():
            do_defseaslinfit=True
            fitlist.remove('defseaslinfit')
        else:
            do_defseaslinfit=False
            
        self.defaultfit=defaultfit
        self.modfits={}
        self.obsfits={}
        obscrit=np.median(np.diff(self.obs_intt))*len(self.obs_val)>self.yrfaco*2.5 \
                    and (self.obs_trange[1]-self.obs_trange[0]).days>365.25*3 \
                    and (largestSeasGap(self.obs_yd)<maxSeasGapDays or ignoreSeasCrit)#len(np.unique(self.obs_yd))>=self.yrfaco-self.mlen*2
        #obscrit=(len(self.obs_val)>self.yrfaco*2.5 or (self.obs_trange[1]-self.obs_trange[0]).days>365.25*3) \
        #            and largestSeasGap(self.obs_yd)<maxSeasGapDays #len(np.unique(self.obs_yd))>=self.yrfaco-self.mlen*2
        obscrit2 = (largestSeasGap(self.obs_yd)<maxSeasGapDays or ignoreSeasCrit) 
                                            #len(np.unique(self.obs_yd))>=self.yrfaco-self.mlen*2
        critO = np.sum(~np.isnan(self.obs_val))>0 # allow code to work without obs
        #obscrit=True
        #obscrit2=True
        for ifit in fitlist:
            self.modfits[ifit]=fitmethods[ifit](self.mod_tind,self.mod_yind,self.mod_val,self.fityrlen)
            if (ifit=='quadfit') and (predefined is not None) and ('obs_b2' in predefined.keys()):
                b2=self.modfits['quadfit'].coef[-1] if predefined['obs_b2']=='mod_b2' else predefined['obs_b2']
                self.obsfits[ifit]=fitmethods[ifit](self.obs_tind,self.obs_yind,self.obs_val,self.fityrlen,b2) if obscrit else None
            else:
                self.obsfits[ifit]=fitmethods[ifit](self.obs_tind,self.obs_yind,self.obs_val,self.fityrlen) if obscrit else None
        if do_defseaslinfit:
            self.obsfits['defseaslinfit']=defseaslinfit(self.obs_tind,self.obs_yind,self.obs_val,
                                                        self.fityrlen,predefined['obs_gsmooth'])
        self.obsfits['mixfitobs']=mixfitobs(self.modfits[defaultfit],self.obs_tind_b,self.obs_yind,
                                            self.obs_val,self.fityrlen) if obscrit2 else None
        if self.freq=='daily':
            self.modfits['binseas']=binseas(self.mod_yind,self.modfits[self.defaultfit].detrend(self.mod_tind,self.mod_val))
            if obscrit:
                self.obsfits['binseas']=binseas(self.obs_yind,self.obsfits[self.defaultfit].detrend(self.obs_tind,self.obs_val))
            elif critO:
                self.obsfits['binseas']=binseas(self.obs_yind,self.modfits[self.defaultfit].\
                                                detrend(self.obs_tind_b,self.obs_val))
        if self.freq=='monthly':
            self.modfits['binseas']=binseas(self.mod_yind,self.modfits[self.defaultfit].\
                                            detrend(self.mod_tind,self.mod_val),ilen=12)
            if obscrit:
                self.obsfits['binseas']=binseas(self.obs_yind,self.obsfits[self.defaultfit].\
                                                detrend(self.obs_tind,self.obs_val),ilen=12)
            elif critO:
                self.obsfits['binseas']=binseas(self.obs_yind,self.modfits[self.defaultfit].\
                                                detrend(self.obs_tind_b,self.obs_val),ilen=12)
        # don't subtract mean from binseas
        #elif self.freq=='monthly':
        #    self.modfits['binseas']=binseas(self.mod_ym,self.mod_val,ilen=12)
        #    self.obsfits['binseas']=binseas(self.obs_ym,self.obs_val,ilen=12)
        self.mod_targetdind=cf.exacttimeind(np.mean(self.obs_tind_b),
                            torig=self.obs_tind_b[0].torig,calendar=self.obs_tind_b[0].calendar,
                                            fixval=True) # mean obs date in model tind units
        self.obs_targetdind=cf.exacttimeind(np.mean(self.obs_tind),
                            torig=self.obs_tind[0].torig,calendar=self.obs_tind[0].calendar,
                                            fixval=True) # mean obs date in model tind units
        self.target_datetime=self.obs_tind[0].torig+dt.timedelta(days=float(np.mean(self.obs_tind)))
        self.mod_target=self.modfits[defaultfit].targetdetrend(self.mod_tind,self.mod_val,self.mod_targetdind)
        if obscrit:
            self.obs_target=self.obsfits[defaultfit].targetdetrend(self.obs_tind,self.obs_val,
                                                               self.obs_targetdind)
        elif do_defseaslinfit:
            self.obs_target=self.obsfits['defseaslinfit'].targetdetrend(self.obs_tind,self.obs_val,
                                                               self.obs_targetdind)
        else:
            self.obs_target=np.nan*self.obs_val
        self.obs_target_b=self.obsfits['mixfitobs'].targetdetrend(self.obs_tind_b,self.obs_val,
                                                                  self.mod_targetdind) if obscrit2 else np.nan*self.obs_val
        self.mod_targetdeseas=self.modfits[defaultfit].targetdetrenddeseas(self.mod_tind,self.mod_yind,self.mod_val,
                                                                           self.mod_targetdind)
        if obscrit:
            self.obs_targetdeseas=self.obsfits[defaultfit].targetdetrenddeseas(self.obs_tind,self.obs_yind,
                                                                          self.obs_val,self.obs_targetdind)
        elif do_defseaslinfit:
            self.obs_targetdeseas=self.obsfits['defseaslinfit'].targetdetrenddeseas(self.obs_tind,self.obs_yind,
                                                                          self.obs_val,self.obs_targetdind)
        else:
            self.obs_targetdeseas=np.nan*self.obs_val
        self.obs_targetdeseas_b=self.obsfits['mixfitobs'].targetdetrenddeseas(self.obs_tind_b,self.obs_yind,
                                                                         self.obs_val,self.mod_targetdind) \
                                                                        if obscrit2 else np.nan*self.obs_val
        if self.iiOLo is not None: # avoid repetition of criteria
            self.modfits_OL=dict()
            self.obsfits_OL=dict()
            for ifit in fitlistOL:
                self.modfits_OL[ifit]=fitmethods[ifit](self.mod_tind[self.iiOLm],self.mod_yind[self.iiOLm],
                                                       self.mod_val[self.iiOLm],self.fityrlen)
                self.obsfits_OL[ifit]=fitmethods[ifit](self.obs_tind[self.iiOLo],self.obs_yind[self.iiOLo],
                                                       self.obs_val[self.iiOLo],self.fityrlen)
        # for convenience:
        self.mod_gsmooth=self.modfits[defaultfit].seas
        self.obs_gsmooth=self.obsfits[defaultfit].seas if self.obsfits[defaultfit] is not None \
                                                        else np.nan*self.mod_gsmooth
        #if do_defseaslinfit:
        #    self.obs_gsmooth=predefined['obs_gsmooth']
        self.obs_gsmooth_b=self.obsfits['mixfitobs'].seas if self.obsfits['mixfitobs'] is not None \
                                                        else np.nan*self.mod_gsmooth
        if critO:
            self.modfitlist = makeSlopesList(self)

    def calc_stats(self):
        if not 'defaultfit' in self.__dict__.keys():
            print('executing calc_fits()')
            self.calc_fits()
        # calc comparison stats
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            bias=np.mean(self.mod_targetdeseas)-np.mean(self.obs_targetdeseas)
            bias_b=np.mean(self.mod_targetdeseas)-np.mean(self.obs_targetdeseas_b)
        mod_SeasAmp=np.nanmax(self.mod_gsmooth)-np.nanmin(self.mod_gsmooth)
        obs_SeasAmp=np.nanmax(self.obs_gsmooth)-np.nanmin(self.obs_gsmooth) \
                    if np.sum(~np.isnan(self.obs_gsmooth))>300 else np.nan
        seas_AmpDiffPerc=(mod_SeasAmp-obs_SeasAmp)/obs_SeasAmp*100
        xcorr_Seas=np.corrcoef(self.obs_gsmooth[:365],self.mod_gsmooth)[0,1] # 365 is for daily case and 
                                                                             # has no effect if len<=365
        seasWSS=WSS(self.obs_gsmooth[:365],self.mod_gsmooth)
        seasRMSE=RMSE(self.obs_gsmooth[:365],self.mod_gsmooth)
        mod_FullMean=np.mean(self.mod_val)
        # if self.freq=='daily': 
        #     lags=np.arange(-183,182)
        # elif self.freq=='monthly':
        #     lags=np.arange(-6,6)
        # cors=np.empty(np.shape(lags))
        # for ii,ind in enumerate(lags):
        #     cors[ii]=np.corrcoef(np.roll(self.mod_gsmooth,ind),self.obs_gsmooth[:365])[1,0]
        # imax=np.argmax(cors)
        # xcorr_Seas_max=cors[imax]
        # xcorr_Seas_lag=lags[imax]
        
        self.stats={'bias':bias,'bias_b':bias_b,'mod_FullMean':mod_FullMean,'mod_SeasAmp':mod_SeasAmp,'obs_SeasAmp':obs_SeasAmp,'seas_AmpDiffPerc':seas_AmpDiffPerc,
                    'xcorr_Seas':xcorr_Seas,'seasWSS':seasWSS,'seasRMSE':seasRMSE,'obs_dts_std':np.std(self.obs_targetdeseas),
                    'mod_dts_std':np.std(self.mod_targetdeseas),'obs_dts_std_b':np.std(self.obs_targetdeseas_b),}
        
    def statsSummary(self):
        if not 'stats' in self.__dict__.keys():
            print('executing calc_stats()')
            self.calc_stats()
        sd=self.stats.copy()
        # extra rows list:
        rlist=['stationID', 'modvar', 'staname', 'lat', 'lon',]
        for el in rlist:
            sd[el]=getattr(self,el)
        sd['mod_opt_coef0']=self.modfits['optfit'].coef[0] if self.modfits['optfit'] is not None else np.nan
        sd['obs_opt_coef0']=self.obsfits['optfit'].coef[0] if self.obsfits['optfit'] is not None else np.nan
        sd['mod_opt_coef1']=self.modfits['optfit'].coef[1] if self.modfits['optfit'] is not None else np.nan
        sd['obs_opt_coef1']=self.obsfits['optfit'].coef[1] if self.obsfits['optfit'] is not None else np.nan
        sd['mod_ols_coef0']=self.modfits['linfit1'].coef[0] if self.modfits['linfit1'] is not None else np.nan
        sd['obs_ols_coef0']=self.obsfits['linfit1'].coef[0] if self.obsfits['linfit1'] is not None else np.nan
        sd['mod_ols_coef1']=self.modfits['linfit1'].coef[1] if self.modfits['linfit1'] is not None else np.nan
        sd['obs_ols_coef1']=self.obsfits['linfit1'].coef[1] if self.obsfits['linfit1'] is not None else np.nan
        sd['obs_N']=len(self.obs_val)
        return sd

    #for pickling/unpickling:
    def __getstate__(self):
        tclasslist=['obs_tind','obs_tind_b', 'mod_tind', 'mod_intt', 'obs_intt']
        print("Pickling, removing cf time classes:",tclasslist)
        mdict=self.__dict__
        for ikey in tclasslist:
            del mdict[ikey]
        return mdict 
    def __setstate__(self,mdict):
        self.__dict__.update(mdict)
        self.__settimes__() # calculated from saved entries
    def topickle(self):
        ppath=getcompsavepath(self.compsavepath,self.stationID,self.modvar,self.freq)
        cf.mkdirs(ppath)
        with open(ppath, 'wb') as hh:
            pickle.dump(self, hh, protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        t = type(self)
        attlist=[el for el in self.__dict__.keys()]
        for el in ['modvar','stationID','staname','freq']:
            attlist.remove(el)
        return (f"{t.__module__}.{t.__qualname__}(\n\t modvar={self.modvar}, "
                    f"stationID={self.stationID}, staname={self.staname}, freq={self.freq}"
                    f"\n\t other instance variables:{','.join(attlist)})")

    #def plot_decomp_mod(self):
    #    return plot_decomp_mod(modvar=self.modvar,
    #                       staname=self.staname,
    #                       lat=self.lat,
    #                       lon=self.lon,
    #                       tind=self.mod_tind,
    #                       tdt=self.mod_tdt,
    #                       val=self.mod_val,
    #                       ll='Model',
    #                       olsfit=self.modfits['linfit1'],
    #                       thsfit=self.mod_thsfit,
    #                       olsfit_OL=self.mod_olsfit_OL,
    #                       OL_trange=self.OL_trange,
    #                       yd=self.mod_yd,
    #                       detrended=self.mod_detrended,
    #                       binned=self.mod_bin,
    #                       gsmoothed=self.mod_gsmooth,
    #                       dtarget=self.mod_target,
    #                       dtargetdeseas=self.mod_targetdeseas)

    #def plot_decomp_obs(self):
    #    return plot_decomp(modvar=self.modvar,
    #                    staname=self.staname,
    #                    lat=self.lat,
    #                    lon=self.lon,
    #                    tind=self.obs_tind,
    #                    tdt=self.obs_tdt,
    #                    val=self.obs_val,
    #                    ll='Observed',
    #                    olsfit=self.obs_olsfit,
    #                    thsfit=self.obs_thsfit,
    #                    olsfit_OL=self.obs_olsfit_OL,
    #                    OL_trange=self.OL_trange,
    #                    yd=self.obs_yd,
    #                    detrended=self.obs_detrended,
    #                    binned=self.obs_bin,
    #                    gsmoothed=self.obs_gsmooth,
    #                    dtarget=self.obs_target,
    #                    dtargetdeseas=self.obs_targetdeseas)
    def plot_ts(self,includefits=['default',],save=False,dpi=None):
        fig,ax=plot_ts(self,includefits)
        if save:
            fsave=figpath('plot_ts',self.figsavepath,self.stationID,self.modvar,self.freq)
            saveloc=os.path.dirname(fsave)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            if dpi:
                fig.savefig(fsave,bbox_inches='tight',dpi=dpi)
            else: 
                fig.savefig(fsave,bbox_inches='tight')
        return fig,ax
    def plot_seas_b(self,save=False,dpi=None):
        fig,ax=plot_seas_b(self)
        if save:
            fsave=figpath('plot_seas_b',self.figsavepath,self.stationID,self.modvar,self.freq)
            saveloc= os.path.dirname(fsave)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            if dpi:
                fig.savefig(fsave,bbox_inches='tight',dpi=dpi)
            else: 
                fig.savefig(fsave,bbox_inches='tight')
        return fig,ax
    def plot_trendhist(self,save=False,dpi=None):
        fig,ax=plot_trendhist(self)
        if save:
            fsave=figpath('plot_trendhist',self.figsavepath,self.stationID,self.modvar,self.freq)
            saveloc=os.path.dirname(fsave)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            if dpi:
                fig.savefig(fsave,bbox_inches='tight',dpi=dpi)
            else:
                fig.savefig(fsave,bbox_inches='tight')
        return fig,ax


def plot_ts(mmm,includefits=['default',]):
    if not hasattr(mmm,'defaultfit'): # calc_fits has not been run
        includefits=[]
    if 'default' in includefits:
        includefits.remove('default')
        includefits=includefits+[mmm.defaultfit,]
    mcols=['g','c','teal','mediumspringgreen','cornflowerblue','lightsteelblue','blueviolet']
    ocols=['orange','m','gold','tomato','chocolate','goldenrod','deeppink']
    stys=['-','--',':']
    fig,ax=plt.subplots(1,1,figsize=(8,2))#(16,4))
    ax.plot(mmm.mod_tdt,mmm.mod_val,'b-',label='ESM4')
    ax.plot(mmm.obs_tdt,mmm.obs_val,'r.',markersize=2,label='Obs.')
    if hasattr(mmm,'modfits'):
        for i1,ifit in enumerate(includefits):
            if ifit in mmm.modfits.keys() and mmm.modfits[ifit] is not None:
                ilabel='fit' if len(includefits)==1 else ifit
                ax.plot(mmm.trange,mmm.modfits[ifit].linpred(mmm.trange),lw=1,
                        ls=stys[i1%len(stys)],color=mcols[i1],label=f'ESM4 {ilabel}')
    if hasattr(mmm,'obsfits'):
        for i1,ifit in enumerate(includefits):
            if ifit in mmm.obsfits.keys() and mmm.obsfits[ifit] is not None:
                coef=mmm.obsfits[ifit].coef
                ilabel='fit' if len(includefits)==1 else ifit
                ax.plot(mmm.trange,mmm.obsfits[ifit].linpred(mmm.trange),lw=1,
                        ls=stys[i1%len(stys)],color=ocols[i1],label=f'Obs. {ilabel}')
    ax.legend(loc=1,bbox_to_anchor=[1.25,1])
    ax.set_ylabel(mmm.dispName+(f' ({mmm.dispUnits})' if \
                                         len(mmm.dispUnits)>0 else ''));
    ax.set_title(f"{mmm.shortID} ({mmm.lat},{mmm.lon})")
    ax.set_xlim(mmm.trange)
    if hasattr(mmm,'target_datetime') and mmm.target_datetime is not None:
        if not np.isnan(np.mean(mmm.obs_targetdeseas)):
            ax.plot(mmm.target_datetime,np.mean(mmm.obs_targetdeseas),'+',color='darkred')
        else:
            ax.plot(mmm.target_datetime,np.mean(mmm.obs_targetdeseas_b),'+',color='darkred')
        ax.plot(mmm.target_datetime,np.mean(mmm.mod_targetdeseas),'x',color='navy')
    return fig, ax

def plot_trendhist(mmm):
    trends=[ifit.coef[1]*dayperyrm for ifit in mmm.modfitlist]
    fig,ax=plt.subplots(1,1,figsize=(3,2))
    ax.hist(trends,40,color='dodgerblue',alpha=.2);
    ax.axvline(mmm.obsfits['optfit'].coef[1]*dayperyro,color='r',label='Obs. trend')
    ax.axvline(mmm.modfits['optfit'].coef[1]*dayperyrm,color='b',label='Model trend')
    ax.set_xlabel(f'Subsample Model Trends \n ({mmm.dispUnits}/yr)')
    ax.set_ylabel(f'Count')
    ax.set_title(f'{mmm.shortID} {mmm.dispName}')
    ax.legend()
    return fig, ax

#seasonality plot example:
def plot_seas_b(mmm):
    fig,ax=plt.subplots(1,1,figsize=(6,2.5))#(16,6))
    ax.plot(np.arange(1,len(mmm.obs_gsmooth)+1),mmm.obs_gsmooth,'r-',label='obs')
    ax.plot(np.arange(1,len(mmm.obs_gsmooth)+1),mmm.obs_gsmooth_b,'m--',label='obs-mod trend')
    ax.plot(np.arange(1,len(mmm.mod_gsmooth)+1),mmm.mod_gsmooth,'b-',label='mod')
    ax.errorbar(np.arange(1,len(mmm.modfits['binseas']['mean'])+1),
                mmm.modfits['binseas']['mean']-np.nanmean(mmm.modfits['binseas']['mean']),
                yerr=mmm.modfits['binseas']['std'],color='b',alpha=.2,label='mod binned±stdev',capsize=2)
    ax.errorbar(np.arange(1,len(mmm.obsfits['binseas']['mean'])+1),
                mmm.obsfits['binseas']['mean']-np.nanmean(mmm.obsfits['binseas']['mean']),
                yerr=mmm.obsfits['binseas']['std'],color='r',alpha=.2,label='obs binned±stdev',capsize=2)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel(f'Mean Seasonal Component\nof {mmm.dispNameUnits}')
    ax.set_title(f'{mmm.shortID} Mean Seasonal Cycle After Removal of Linear Trend')
    ax.legend()
    return fig,ax

#seasonality plot example:
def plot_seas(mmm):
    fig,ax=plt.subplots(1,1,figsize=(16,6))
    ax.plot(np.arange(1,len(mmm.obs_gsmooth)+1),mmm.obs_gsmooth,'r-',label='obs')
    ax.plot(np.arange(1,len(mmm.mod_gsmooth)+1),mmm.mod_gsmooth,'b-',label='mod')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel(f'Mean Seasonal Component\nof {mmm.dispNameUnits}')
    ax.set_title('Mean Seasonal Cycle After Removal of Linear Trend')
    ax.legend()
    return fig,ax

#def plot_Trends(mmm):
#    fig,ax=plt.subplots(1,1,figsize=(1.5,1.5))
#    ax.errorbar(1,mmm.obs_olsfit.coef[1]*dayperyro,yerr=mmm.obs_olsfit.CI[1]*dayperyro,color='r',capsize=4,marker='o',markersize=2,label='obs OLS',linestyle='none')
#    ax.errorbar(1,mmm.modfits['linfit1'].coef[1]*dayperyrm,yerr=mmm.modfits['linfit1'].CI[1]*dayperyrm,color='b',capsize=4,marker='o',markersize=2,label='mod OLS',linestyle='none')
#    ax.plot(1,mmm.obs_thsfit.coef_*dayperyro,'ro',markersize=4,label='obs Theil-Sen',fillstyle='none')
#    ax.plot(1,mmm.mod_thsfit.coef_*dayperyrm,'bo',markersize=4,label='mod Theil-Sen',fillstyle='none')
#    ax.errorbar(2,mmm.obs_olsfit_OL.coef[1]*dayperyro,yerr=mmm.obs_olsfit_OL.CI[1]*dayperyro,color='r',capsize=4,marker='o',markersize=2,linestyle='none')
#    ax.errorbar(2,mmm.modfits['linfit1']_OL.coef[1]*dayperyrm,yerr=mmm.modfits['linfit1']_OL.CI[1]*dayperyrm,color='b',capsize=4,marker='o',markersize=2,linestyle='none')
#    ax.tick_params(axis='both', labelsize=6)
#    ax.set_xticks([1,2],['Full\nSeries','Overlapping\nRegion'])
#    ax.set_xlim(.3,2.7)
#    trendsuffix='yr$^{-1}$'
#    ax.set_ylabel(f'{mmm.dispName} Slope ({mmm.dispUnits} {trendsuffix})',fontsize=6)
#    ax.set_title(f'Linear Trend in {mmm.dispName} with 95% CI',fontsize=6)
#    ax.legend(fontsize=6,bbox_to_anchor=[1.1,.8])
#    return fig,ax

def makeSlopesList(icomp):
    stadd=np.zeros(30).astype(bool)
    stencil=np.zeros(len(icomp.mod_val))
    nn=len(icomp.obs_val)
    for ii in [el-icomp.obs_intt[0] for el in icomp.obs_intt]:
        stencil[ii]=1
    stencil=stencil.astype(bool)
    fitlist=[]
    ds=np.array([mv-icomp.modfits[icomp.defaultfit].seas[iyd-1] for mv, iyd in zip(icomp.mod_val,icomp.mod_yind)])
    ivals=ds[stencil]
    itind=icomp.mod_tind[stencil].astype(float)
    while np.sum(stencil)==nn:
        mod_olsfit=cf.linreg(itind,ivals)
        fitlist.append(mod_olsfit)
        stencil=np.concatenate((stadd,stencil[:-30]))
        ivals=ds[stencil]
        itind=icomp.mod_tind[stencil].astype(float)
    return fitlist

def ClusterList(j,i):
    llist=[(j,i)]
    for ind in range(1,6):
        llist.append((j-ind,i-ind))
        llist.append((j-ind,i))
        llist.append((j-ind,i+ind))
        llist.append((j,i-ind))
        llist.append((j,i+ind))
        llist.append((j+ind,i-ind))
        llist.append((j+ind,i))
        llist.append((j+ind,i+ind))
    return llist
