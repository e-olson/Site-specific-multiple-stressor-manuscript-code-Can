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
import PyCO2SYS as pyco2
import pickle

# Station Aloha location: 22' 45'N, 158' 00'W
lat=22+45/60
lon=-158.0
dsid='HOT'
OAPdsid=OAP.getID('MOSEAN/WHOTS')

# # definte variable names:
# sSil='Si1'
# sPO4='PO41'
# sTem='Temp'
# sDIC='CO2'
# sAlk='Alk'
# sSal='Sal'
# sPress='Press'
# vardict={'phos':'pH_sw_nanmean','spco2':'pCO2_sw_nanmean','co2dryair':'xCO2_air_nanmean',
#          'tos':'SST_nanmean','sos':'SSS_nanmean','chlos':'CHL_nanmean',
#          'dpco2':'dpco2','l10chlos':'l10CHL','o2os':'DOXY_nanmean','hplusos':'hplus_nanmean'}

def _get_dt(yr,mon,day,timestr):
    hh,mm=timestr.split(':')
    return dt.datetime(int(yr),int(mon),int(day),int(hh),int(mm))

def load(surface=False,freq='original',depthRange=None,cCalcs=True):
    # options:
    # surface: load surface data (<=10m depth)
    # freq: original- don't average; daily: group by day; monthly: group by month
    # cCalcs: Do carbonate system calculations to get additional variables
    # some checking on option consistency:
    if freq!='original' and ((not surface) and depthRange is None):
        raise Exception('If data is to be averaged, choose a depth range or load surface values')
    if cCalcs and not surface:
        raise Exception('carbonate system calculations are set up for surface only')
    fpath='/work/ebo/obs/HOT/HOT-DOGSoutput.csv'
    header=['botid', 'date', 'time', 'press', 'theta', 'sigma', 'temp', 'csal', 'bsal', 'boxy', 
        'dic', 'ph', 'alk', 'phos', 'nit', 'sil', 'doc', 'pc', 'chl', 'hplc', 
        'l12', 'd12'] 
    units=['#', 'mmddyy', 'hhmmss', 'dbar', 'ITS-90', 'kg/m3', 'ITS-90', 'PSS-78', 'PSS-78', 'umol/kg', 
       'umol/kg','', 'ueq/kg', 'umol/kg', 'umol/kg', 'umol/kg', 'umol/kg', 'umol/kg', 'ug/l', 'ng/l', 
       'mg C/m3', 'mg C/m3']
    excludeList=['dtUTC','botid', 'date', 'time']
    df=pd.read_csv(fpath,na_values=['NaN','-9999.0','-9999','-9','-9.0','-9.00','-9.000','-00009'],
               skiprows=5,names=header + ['None'],index_col=None,
                converters={'date':lambda x: str(x).strip(),'time':lambda x: str(x).strip()})
    df.drop(columns='None',inplace=True) # get rid of weird extra column needed because of commas at end of line
    unitsdict=dict(zip(header,units))
    # fix -9's that weren't converted to nan:
    df.loc[df.alk==-9,['alk']]=np.nan
    # convert -9s in date, time columns:
    iii=df.date=='-00009'
    df.loc[iii,['date']]=np.nan
    df.loc[iii,['time']]=np.nan
    # if date but no time recorded, set time to mid-day:
    df.loc[df.time=='-00009',['time']]='120000'
    # drop na's
    df.dropna(axis=0,how='all',subset=['date'],inplace=True)
    df.reset_index(inplace=True)
    df['dtUTC']=pd.Series([dt.datetime.strptime(el,'%m%d%y-%H%M%S') for el in ['-'.join([a,b]) \
                               for a,b in zip(df['date'].values,df['time'].values)]],dtype='object')
    df['Z']=[-1*gsw.z_from_p(p,lat) for p in df['press'].values]
    df['YD']=cf.ydfromdt(df['dtUTC'].values)
    df['Sal']=[s1 if not pd.isnull(s1) else s2 for ind, (s1,s2) \
                   in df.loc[:,['bsal','csal']].iterrows()]
    # apply depth restrictions
    if surface:
        df=df.loc[df.Z<=10].reset_index().copy(deep=True)
    elif depthRange is not None:
        df=df.loc[(df.Z>depthRange[0])&(df.Z<=depthRange[1])].reset_index().copy(deep=True)
    # carb chem calculations
    if cCalcs: #(also requires surface==True)
        def _meanfill(x):
            return np.where(~pd.isnull(x),x,np.nanmean(x))
        df['nDIC'] = np.nanmean(df['Sal'].values)/df['Sal'].values * df['dic'].values
        alksalfit=cf.linreg(df.Sal,df.alk)
        df['TA_est'] = df['Sal'].values*alksalfit.coef[1]+alksalfit.coef[0]
        df['TA_best']=np.where(pd.isnull(df['alk']),df['TA_est'],df['alk'])
        pyco2out=cc.pyco2_surf_all_from_TA_DIC(df['TA_best'],df['dic'],df['Sal'],df['temp'],
                                              _meanfill(df['sil']),
                                               _meanfill(df['phos']))
        savelist=['pH_total','pCO2','saturation_calcite','saturation_aragonite','HCO3','CO3','CO2','Hfree']
        for el in savelist:
            df['csys_'+el]=pyco2out[el]
        
    # apply temporal averaging (or not)
    if freq=='original':
        dfout=df
    else: # freq!='original'
        def countreal(x):
            return np.sum(~np.isnan(x))
        agdic={'exactDay':[np.nanmean]}
        for vname in df.keys():
            if vname not in excludeList:
                agdic[vname]=agdic[vname]=[np.nanmean,np.nanstd,countreal]
        df3=df.copy(deep=True)
        tref=dt.datetime(1975,1,1,0,0) # use start of model sections for consistency
        df3['timeind']=pd.Series(cf.to_int_tind(df3['dtUTC'],freq=freq,torig=tref),dtype='object')
        df3['exactDay']=pd.Series(cf.to_exact_tind(df3['dtUTC'],torig=tref),dtype='object')
        df3.drop(columns=['dtUTC'],inplace=True)
        dfout=df3.groupby(['timeind']).agg(agdic).reset_index()
        dfout.columns = [ii+'_'+jj if len(jj)>0 else ii for ii,jj in \
                  zip(dfout.columns.get_level_values(0),dfout.columns.get_level_values(1))]
        dfout['dtUTC']=pd.Series([tref+dt.timedelta(days=ii) for ii in dfout['exactDay_nanmean']],dtype='object')
    dfout['YD']=cf.ydfromdt(dfout['dtUTC'].values)
    return dfout
    
    
def obsload(modvar,freq,df=None):
    # modvar: model variable name
    # freq: frequency
    # df: optionally pre-load obs dataframe (at correct frequency!) to avoid re-calculation
    vd={'tos':ev.varDef('temp_nanmean','SST','°C','SST (°C)'),
        'sos':ev.varDef('Sal_nanmean','SSS','psu','SSS (psu)'),
        'phos':ev.varDef('csys_pH_total_nanmean','pH','','pH'),
        'talkos':ev.varDef('alk_nanmean','TA','umol kg$^{-1}$','TA (µmol kg$^{-1}$)'),
        'no3os':ev.varDef('nit_nanmean',
                          'NO$_3$','µmol kg$^{-1}$','NO$_3$ (µmol kg$^{-1}$)')}
    if df is None:
        df=load(surface=True,freq=freq)
    if modvar in vd.keys():
        ik=vd[modvar].dfkey
        ix=~pd.isnull(df[ik])
        obs_tdt=np.squeeze(df.loc[ix,['dtUTC']].values)
        obs_val=np.squeeze(df.loc[ix,[ik]].values)
    return obs_tdt,obs_val,vd[modvar].dispName,vd[modvar].dispUnits,\
                vd[modvar].dispNameUnits, vd[modvar].dfkey


def modload(modvar,freq):
    scen='GFDL-ESM4.1.1975_2022'
    with nc.Dataset(diagsPP.searchExtracted(scen,OAPdsid)) as f1, \
         nc.Dataset(diagsPP.searchExtracted(scen,OAPdsid,'288grid')) as f2:
        return OBC.modload(modvar,f1,f2,lon,lat,freq)
    
def comp(mvar,freq,obsdf=None,recalc=False):
    saveloc='/work/ebo/calcs/buoyCompTS/GFDL-ESM4.1.1975_2022/'
    ppath=ev.getcompsavepath(saveloc+'comps/',dsid,mvar,freq)
    if recalc:
        with nc.Dataset(diagsPP.searchExtracted('GFDL-ESM4.1.1975_2022',OAPdsid)) as f1, \
             nc.Dataset(diagsPP.searchExtracted('GFDL-ESM4.1.1975_2022',OAPdsid,'288grid')) as f2:
            icomp = ev.timeSeriesComp(mvar,dsid,'HOT','HOT',lat,lon,
                            obsloadfun=obsload,obsloadkwargs={'freq':freq,'df':obsdf},
                            modloadfun=OBC.modload,
                            modloadkwargs={'f1':f1,'f2':f2,'lon':lon,'lat':lat,'freq':freq},
                            freq=freq,savepath=saveloc)
        # load OAP comp at OSP to get seasonal cycle
        mmmOAP=OBC.loadMoorComp(OAPdsid,mvar,freq)
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



