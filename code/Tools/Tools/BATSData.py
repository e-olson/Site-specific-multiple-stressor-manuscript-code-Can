import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from Tools import commonfxns as cf, OAPBuoyData as OAP, evalfxns as ev, OAPBuoyComp as OBC, diagsPP, ccfxns as cc
import datetime as dt
import gsw
import dateutil
import cftime
import matplotlib.image as mpimg
from scipy.stats.distributions import t as scpst_t
import sys
#sys.path.append('/nbhome/ebo/mocsy/')
#import mocsy
#import PyCO2SYS as pyco2
import pickle

lat=31.7
lon=-64.2
dsid='BATS'
OAPdsid=OAP.getID('BTM')

# definte variable names:
sSil='Si1'
sPO4='PO41'
sTem='Temp'
sDIC='CO2'
sAlk='Alk'
sSal='Sal'
sPress='Press'
# vardict={'phos':'pH_sw_nanmean','spco2':'pCO2_sw_nanmean','co2dryair':'xCO2_air_nanmean',
#          'tos':'SST_nanmean','sos':'SSS_nanmean','chlos':'CHL_nanmean',
#          'dpco2':'dpco2','l10chlos':'l10CHL','o2os':'DOXY_nanmean','hplusos':'hplus_nanmean'}

def _getdt(yyyymmdd,time,decy):
    yr=int(yyyymmdd[:4])
    mn=int(yyyymmdd[4:6])
    dy=int(yyyymmdd[6:])
    if len(time[-4:-2])>0:
        hr=int(time[-4:-2])
        mi=int(time[-2:])
        tdt = dt.datetime(yr,mn,dy,hr,mi)
    else:
        yr=int(decy)
        yrfrac=decy-yr
        yrlendays=(dt.datetime(yr+1,1,1)-dt.datetime(yr,1,1)).days
        tdt = dt.datetime(yr,1,1,0,0)+dt.timedelta(days=yrlendays*yrfrac)
    return tdt

# def _mocsywrap(T,S,TA,DIC,SIL,PHOS,patm,press,lat,fillNuts=False):
#     """
#     T: deg C (in situ)
#     S: psu
#     TA: mol/kg
#     DIC: mol/kg
#     SIL: mol/kg
#     PHOS: mol/kg
#     patm: atm
#     press: dbar
#     lat: dec deg N
#     returns:
#     pH,pco2,fco2,co2,hco3,co3,OmegaA,OmegaC
#     """
#     def isnum(x):
#         return isinstance(x,float) or isinstance(x,int)
#     # adjustments to permit use of floats for constant values:
#     if fillNuts:
#         SIL[pd.isnull(SIL)]=np.nanmean(SIL)
#         PHOS[pd.isnull(PHOS)]=np.nanmean(PHOS)
#     T=T*np.ones(np.shape(DIC)) if isnum(T) else T
#     S=S*np.ones(np.shape(DIC)) if isnum(S) else S
#     TA=TA*np.ones(np.shape(DIC)) if isnum(TA) else TA
#     SIL=SIL*np.ones(np.shape(DIC)) if isnum(SIL) else SIL
#     PHOS=PHOS*np.ones(np.shape(DIC)) if isnum(PHOS) else PHOS
#     patm=patm*np.ones(np.shape(DIC)) if isnum(patm) else patm
#     press=press*np.ones(np.shape(DIC)) if isnum(press) else press
#     lat=lat*np.ones(np.shape(DIC)) if isnum(lat) else lat
#     # function call
#     pH,pco2,fco2,co2,hco3,co3,OmegaA,OmegaC,BetaD,DENis,p,Tis = \
#             mocsy.mvars(temp=T, sal=S, 
#                  alk=TA, dic=DIC, 
#                  sil=SIL, phos=PHOS, 
#                  patm=patm, depth=press, lat=lat, 
#                  optcon='mol/kg', optt='Tinsitu', optp='db', 
#                  optb="l10", optk1k2='m10', optkf="dg", optgas='Pinsitu')
#     badvals=(np.isnan(T))|(np.isnan(S))|(np.isnan(TA))|(np.isnan(DIC))|(np.isnan(SIL))|(np.isnan(PHOS))|\
#             (np.isnan(patm))|(np.isnan(press))|(np.isnan(lat))
#     return  np.where(~badvals,pH,np.nan), \
#             np.where(~badvals,pco2,np.nan),\
#             np.where(~badvals,fco2,np.nan),\
#             np.where(~badvals,co2,np.nan),\
#             np.where(~badvals,hco3,np.nan),\
#             np.where(~badvals,co3,np.nan),\
#             np.where(~badvals,OmegaA,np.nan),\
#             np.where(~badvals,OmegaC,np.nan)

def load(surface=True,freq='original',depthRange=None,cCalcs=True):
    header=['Id', 'yyyymmdd', 'decy', 'time', 'latN', 'lonW', 'Depth', 'Temp', 'CTD_S', 
            'Sal1', 'Sig-th', 'O2(1)', 'OxFix', 'Anom1', 'CO2', 'Alk', 'NO31', 'NO21', 'PO41',
             'Si1', 'POC', 'PON', 'TOC', 'TN', 'Bact', 'POP', 'TDP', 'SRP', 'BSi', 'LSi', 
            'Pro', 'Syn', 'Piceu', 'Naneu']
    if freq!='original' and ((not surface) and depthRange is None):
        raise Exception('If data is to be averaged, choose a depth range or load surface values')
    if cCalcs and not surface:
        raise Exception('carb chem calculations were defined for surface only; check and implement')
    # Id     = Sample Id                  
    #  A unique bottle id which identifies cruise, cast, and Nisken number
    #  8 digit number !###$$$@@, where,
    #  !   =Cruise type, 1=BATS core, 2=BATS Bloom a, and 3=BATS Bloom b, etc.
    #  ### =Cruise number
    #  $$$  =Cast number, 1-80=CTD casts, 81-99=Hydrocasts (i.e. 83 = Data from Hydrocast number 3)
    #  @@  =Niskin number
    #  e.g. 10480410 is BATS core 48, cast 4, Niskin 10
    # yymmdd = Year Month Day   
    # decy   = Decimal Year     
    # time   = Time (hhmm)      
    # latN   = Latitude (Deg N) 
    # lonW   = Longitude (Deg W)
    # Depth  = Depth (m)                  
    # Temp   = Temperature ITS-90 (C)    
    # CTD_S  = CTD Salinity (PSS-78)      
    # Sal1   = Salinity-1 (PSS-78)        
    # Sig-th = Sigma-Theta (kg/m^3)       
    # O2(1)  = Oxygen-1 (umol/kg)          
    # OxFixT = Oxygen Fix Temp (C)        
    # Anom1  = Oxy Anomaly-1 (umol/kg)    
    # CO2    = CO2 (umol/kg)              
    # Alk    = Alkalinity (uequiv)        
    # NO31   = Nitrate+Nitrite-1 (umol/kg)
    # NO21   = Nitrite-1 (umol/kg)        
    # PO41   = Phosphate-1 (umol/kg)      
    # Si1    = Silicate-1 (umol/kg)       
    # POC    = POC (ug/kg)                
    # PON    = PON (ug/kg)                
    # TOC    = TOC (umol/kg)                
    # TN     = TN (umol/kg)  
    # Bact   = Bacteria (cells*10^8/kg)   
    # POP    = POP (umol/kg)
    # TDP    = Total dissolved Phosphorus (nmol/kg)
    # SRP    = Low-level phosphorus (nmol/kg)
    # BSi    = Particulate biogenic silica (umol/kg)
    # LSi    = Particulate lithogenic silica  (umol/kg)
    # Pro    = Prochlorococcus (cells/ml)
    # Syn    = Synechococcus (cells/ml)
    # Piceu  = Picoeukaryotes (cells/ml)
    # Naneu  = Nanoeukaryotes (cells/ml)
    excludeList=['dtUTC','Id', 'yyyymmdd', 'time', 'latN', 'lonW', 'CTD_S',
            'Sal1', 'Si1', 'POC', 'PON', 'TOC', 'TN', 'Bact', 'POP', 'TDP', 'SRP', 'BSi', 'LSi',
            'Pro', 'Syn', 'Piceu', 'Naneu']
    df=pd.read_csv('/work/ebo/obs/BATS/bats_bottle.txt',skiprows=60,sep='\t+',names=header,engine='python',
                   converters={'yyyymmdd':lambda x: str(x).strip(),'time':lambda x: str(x).strip()},
                   na_values=['NaN','-9999.0','-9999','-9','-9.0','-9.00','-9.000','-00009','-999','-999.0',
                              '-999.00','-999.000',])
    # some QC:
    df.loc[(df.Alk>3000)|(df.Alk<2000),['Alk']]=np.nan
    df.loc[(df.CO2<1800),['CO2']]=np.nan
    # calcs
    df['dtUTC']=pd.Series([_getdt(yyyymmdd,time,decy) for ind,(yyyymmdd,time,decy) \
                            in df.loc[:,['yyyymmdd','time','decy']].iterrows()],dtype='object')
    tref=dt.datetime(1975,1,1,0,0) # use start of model sections for consistency (alt: obsdtUTC[0])
    df['Lat']=df['latN']
    df['Lon']=-1*df['lonW']
    df['Z']=df['Depth']
    df['Press']=[-1*gsw.p_from_z(z,lat) for z,lat in zip(df['Depth'],df['Lat'])]
    df['Sal']=np.where(pd.isnull(df['CTD_S']),df['Sal1'],df['CTD_S'])

    if freq=='original':
        df2=df
    if surface:
        df2=df.loc[df.Z<=10].reset_index().copy(deep=True)
    elif depthRange is not None:
        df2=df.loc[(df.Z<=depthRange[1])&(df.Z>depthRange[0])].reset_index().copy(deep=True)
    if surface or (depthRange is not None):
        df2['nDIC'] = np.nanmean(df2['Sal'].values)/df2['Sal'].values * df2['CO2'].values
#     if mocsyCalcs: # before averaging; should only be called if surface=True
#         # Alk-S fit:
#         sDIC='CO2'; sAlk='Alk';sSal='Sal';sSil='Si1';sPO4='PO41';sTem='Temp';
#         alksalfit=cf.linreg(df2[sSal],df2[sAlk])
#         pH2,pco22,fco22,co22,hco32,co32,OmegaA2,OmegaC2=_mocsywrap(df2[sTem].values, df2[sSal].values, 
#                  np.where(pd.isnull(df2[sAlk].values),
#                     df2[sSal].values*alksalfit.coef[1]+alksalfit.coef[0],df2[sAlk].values)*1.e-6, 
#                  df2[sDIC].values*1.e-6, 
#                  df2[sSil].values*1.e-6, df2[sPO4].values*1.e-6, 
#                  1, df2[sPress].values, df2['Lat'].values,fillNuts=True)
#         df2['pH_calc']=pH2
#         df2['pco2_calc']=pco22
#         df2['fco2_calc']=fco22
#         df2['co2_calc']=co22
#         df2['hco3_calc']=hco32
#         df2['co3_calc']=co32
#         df2['omegaA_calc']=OmegaA2
#         df2['OmegaC_calc']=OmegaC2
    if cCalcs:#(also requires surface==True)
        def _meanfill(x):
            return np.where(~pd.isnull(x),x,np.nanmean(x))
        sDIC='CO2'; sAlk='Alk';sSal='Sal';sSil='Si1';sPO4='PO41';sTem='Temp';
        alksalfit=cf.linreg(df2[sSal],df2[sAlk])
        df2['TA_est'] = df2[sSal].values*alksalfit.coef[1]+alksalfit.coef[0]
        df2['TA_best']=np.where(pd.isnull(df2[sAlk]),df2['TA_est'],df2[sAlk])
        pyco2out=cc.pyco2_surf_all_from_TA_DIC(df2['TA_best'],df2[sDIC],df2[sSal],df2[sTem],
                                              _meanfill(df2['Si1']),
                                               _meanfill(df2['PO41']))
        savelist=['pH_total','pCO2','saturation_calcite','saturation_aragonite','HCO3','CO3','CO2','Hfree']
        for el in savelist:
            df2['csys_'+el]=pyco2out[el]
    if freq!='original':
        def countreal(x):
            return np.sum(~np.isnan(x))
        agdic={'exactDay':[np.nanmean]}
        for vname in df2.keys():
            if vname not in excludeList:
                agdic[vname]=agdic[vname]=[np.nanmean,np.nanstd,countreal]
        df3=df2.copy(deep=True)
        df3['timeind']=pd.Series(cf.to_int_tind(df3['dtUTC'],freq=freq,torig=tref),dtype='object')
        df3['exactDay']=pd.Series(cf.to_exact_tind(df3['dtUTC'],torig=tref),dtype='object')
        df3.drop(columns=['dtUTC'],inplace=True)
        df2=df3.groupby(['timeind']).agg(agdic).reset_index()
        df2.columns = [ii+'_'+jj if len(jj)>0 else ii for ii,jj in \
                  zip(df2.columns.get_level_values(0),df2.columns.get_level_values(1))]
        df2['dtUTC']=pd.Series([tref+dt.timedelta(days=ii) for ii in df2['exactDay_nanmean']],dtype='object')
    df2['YD']=cf.ydfromdt(df2['dtUTC'].values)
    return df2

def obsload(modvar,freq,df=None):
    vd={'tos':ev.varDef('Temp_nanmean','SST','°C','SST (°C)'),
        'sos':ev.varDef('Sal_nanmean','SSS','psu','SSS (psu)'),
        'phos':ev.varDef('csys_pH_total_nanmean','pH','','pH'),
        'talkos':ev.varDef('Alk_nanmean','TA','umol kg$^{-1}$','TA (µmol kg$^{-1}$)'),
        'no3os':ev.varDef('NO31_nanmean','NO$_3$','umol kg$^{-1}$','NO$_3$ (µmol kg$^{-1}$)')}
    if df is None:
        df=load(surface=True,freq=freq)
    if modvar in vd.keys():
        ik=vd[modvar].dfkey
        ix=~pd.isnull(df[ik])
        obs_tdt=np.squeeze(df.loc[ix,['dtUTC']].values)
        obs_val=np.squeeze(df.loc[ix,[ik]].values)
    return np.squeeze(df.loc[ix,['dtUTC']].values),np.squeeze(df.loc[ix,[ik]].values),\
                vd[modvar].dispName,vd[modvar].dispUnits,vd[modvar].dispNameUnits, vd[modvar].dfkey

# def modload(modvar,freq):
#     dsid=OAP.getID('BTM')
#     with nc.Dataset(OAP.modpath(dsid)) as f1:
#         mod_tnl=cf.noLeapFromNC(f1)
#         mod_tdt=cf.cftnoleap_to_dt(mod_tnl)
#         mod_val=np.mean(f1.variables['ph'][:,:2,...],1)
#     if freq=='monthly':
#         mod_tind=cf.timeindex(mod_tnl,freq=freq)
#         if len(np.unique(mod_tind))<len(mod_tind):
#             new_tind=np.unique(mod_tind)
#             new_val=np.array([np.nanmean(mod_val[mod_tind==iind]) for iind in new_tind])
#             temptdt=[mod_tdt[mod_tind==iind][0] for iind in new_tind]
#             yrmon=[[iii.year,iii.month] for iii in temptdt]
#             new_tnl=np.array([cftime.datetime(iyr,imon,15,calendar='noleap') for iyr,imon in yrmon])
#             new_tdt=np.array([dt.datetime(iyr,imon,15) for iyr, imon in yrmon])
#             return new_tnl, new_tdt, new_val
#     return mod_tnl, mod_tdt, np.squeeze(mod_val)


def comp(mvar,freq,obsdf=None,recalc=False):
    saveloc='/work/ebo/calcs/buoyCompTS/GFDL-ESM4.1.1975_2022/'
    ppath=ev.getcompsavepath(saveloc+'comps/',dsid,mvar,freq)
    if recalc:
        with nc.Dataset(diagsPP.searchExtracted('GFDL-ESM4.1.1975_2022',OAPdsid)) as f1, \
                nc.Dataset(diagsPP.searchExtracted('GFDL-ESM4.1.1975_2022',OAPdsid,'288grid')) as f2:
            icomp = ev.timeSeriesComp(mvar,dsid,'Bermuda Atlantic Time Series','BATS',lat,lon,
                            obsloadfun=obsload,obsloadkwargs={'freq':freq,'df':obsdf},
                            modloadfun=OBC.modload,modloadkwargs={'f1':f1,'f2':f2,'lon':lon,'lat':lat,'freq':freq},
                            freq=freq,savepath=saveloc)
        icomp.calc_fits()
        icomp.topickle()
        with open(ppath, 'rb') as hh:
            mmm=pickle.load(hh)
        return mmm
    else:
        try:
            with open(ppath, 'rb') as hh:
                mmm=pickle.load(hh)
            return mmm
        except:
            comp(mvar,freq,obsdf,recalc=True)
            return icomp



