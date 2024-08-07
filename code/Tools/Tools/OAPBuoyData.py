#import socket
import os
import glob
# import re
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import datetime as dt
import dateutil
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import cartopy
import PyCO2SYS as pyco2
import warnings
import viz, commonfxns as cf, diagsPP, AlkFits, ccfxns as cc

# note: xarray failed to load ERDDAP nc files from this site
# path for info table on ERDDAP:
#url3='https://data.pmel.noaa.gov/pmel/erddap/tabledap/allDatasets.csv?datasetID' 
# urlInfo='https://data.pmel.noaa.gov/pmel/erddap/tabledap/allDatasets.csv?'+\
#         'datasetID%2Ctitle%2CminLongitude%2CmaxLongitude%2CminLatitude%2CmaxLatitude%2CminTime%2CmaxTime'
# start of path for actual data sets on ERDDAP:
# erdstr0='https://data.pmel.noaa.gov/pmel/erddap/tabledap/'

# define regions, almost all based on Sutton et al 2019
regions={
    'Iceland':'North Atlantic Ocean',
    'Gulf of Maine':'US east coast', 
    'First Landing':'US east coast', # Chesapeak Bay mouth
    'Hog Reef':'Atlantic coral reef',
    'Crescent Reef':'Atlantic coral reef', 
    'BTM':'North Atlantic Ocean', 
    "Gray's Reef":'US east coast', 
    'Coastal MS':'Gulf of Mexico coast', 
    'Coastal LA':'Gulf of Mexico coast',
    'Cheeca Rocks':'Caribbean coral reef', 
    'Stratus':'Southeast Pacific Ocean', 
    'La Parguera':'Caribbean coral reef', 
    'BOBOA':'Indian Ocean', 
    'CCE1':'Northeast Pacific Ocean', 
    'CCE2':'US west coast',
    'Cape Arago':'US west coast', 
    'NH-10':'US west coast', 
    'Cape Elizabeth':'US west coast', 
    'Cha ba':'US west coast', 
    'SEAK':'Alaskan coast', 
    'GAKOA':'Alaskan coast',
    'Kodiak':'Alaskan coast', 
    'M2':'Bering Sea coastal shelf', 
    'Papa':'Northeast Pacific Ocean', 
    'SOFS':'Southern Ocean', 
    'JKEO':'Northwest Pacific Ocean', 
    'KEO':'Northwest Pacific Ocean', 
    'MOSEAN/WHOTS':'Central Pacific Ocean',
    'Kaneohe':'Pacific island coral reef', 
    'CRIMP2':'Pacific island coral reef', 
    'CRIMP1':'Pacific island coral reef', 
    'Kilo Nalu':'Pacific island coral reef', 
    'Ala Wai':'Pacific island coral reef', 
    'TAO170W':'Equatorial Pacific Ocean',
    'TAO165E':'Equatorial Pacific Ocean', 
    'TAO8S165E':'Equatorial Pacific Ocean', 
    'TAO155W':'Equatorial Pacific Ocean', 
    'Chuuk':'Pacific island coral reef', 
    'TAO140W':'Equatorial Pacific Ocean', 
    'TAO125W':'Equatorial Pacific Ocean',
    'TAO110W':'Equatorial Pacific Ocean'}
localPath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/data/obs/OAPBuoy/'
# if 'stellar' in socket.gethostname():
#     localPath='/scratch/gpfs/eo2651/obs/OAPBuoys/'
#     calcsPath='/scratch/cimes/eo2651/calcs/extracted_OAPBuoy/'
# else:
#     # localPath='/net2/Elise.Olson/obs/OAPBuoys/'
#     # if not os.path.exists(localPath):
#     #     localPath='/work/Elise.Olson/obs/OAPBuoys/' # /net2 not accessible from PP nodes 
#     localPath='/work/Elise.Olson/obs/OAPBuoys/' # /net2 not accessible from PP nodes 
#     calcsPath='/work/Elise.Olson/calcs/extracted_OAPBuoy/'

def loadOAPInfo(modelgrid=False,modMeans=False):
    # if modMeans:
    #     t2=xr.open_dataset(calcsPath+'dfInfoBuoyWithModMeans.nc')
    #     dfInfoBuoy=t2.to_dataframe()
    #     dfInfoBuoy.index.name = None
    #     t2.close()
    # else:
    dfInfoBuoy=pd.read_csv('dfInfoBuoy.csv',index_col=0)
    if modelgrid:
        mask1x1=np.ceil(viz.wet1x1)
        dfInfoBuoy['jj'],dfInfoBuoy['ii']=cf.nearest_valid_point(dfInfoBuoy['Lat'],dfInfoBuoy['Lon'],
                                                              viz.glat1x1,viz.glon1x1,mask1x1,thresh=180)#just use HD
        dfInfoBuoy['Lat1x1']=[viz.glat1x1[jj,ii] for jj, ii in dfInfoBuoy[['jj','ii']].values]
        dfInfoBuoy['Lon1x1']=[viz.glon1x1[jj,ii] for jj, ii in dfInfoBuoy[['jj','ii']].values]
        # choose atmos grid point according to location of ocean grid point rather than original data location
        # mask not necessary
        # dfInfoBuoy['jj288'],dfInfoBuoy['ii288']=cf.nearest_point(dfInfoBuoy['Lat1x1'],dfInfoBuoy['Lon1x1'],
        #                                                          viz.glat288,viz.glon288,thresh=100,tol=2)
        dfInfoBuoy['jjHD'],dfInfoBuoy['iiHD']=cf.nearest_valid_point(dfInfoBuoy['Lat1x1'],dfInfoBuoy['Lon1x1'],
                                                                viz.glat1x1,viz.glon1x1,viz.wet1x1) # defaults are for half degree grid
        dfInfoBuoy['modBathy']=[viz.deptho1x1[jj,ii] for jj, ii in dfInfoBuoy[['jj','ii']].values]
        dfInfoBuoy['modBathyHD']=[viz.deptho1x1[jj,ii] for jj, ii in dfInfoBuoy[['jjHD','iiHD']].values]
    dfInfoBuoy['shortTitle']=[el.split(' NOAA Surface Ocean CO2 and Ocean Acidification Mooring Time Series')[0] for el in dfInfoBuoy['title']]
    return dfInfoBuoy

#def slidingstdthreshold(val, window, nstdev_thresh=3):
#    # note: relies on evenly spaced data (no gaps, but can have NaNs)
#    xmean=cf.slidingWindowEval(val,np.nanmean,window)
#    xstd=cf.slidingWindowEval(val,np.nanstd,window)
#    x2=np.where(np.abs(x-xmean)<=nstdev_thresh*xstd,val,np.nan)
#    return x2
# def loadOAPInfoERDDAP():
#     dfInfo=pd.read_csv(urlInfo,skiprows=[1,])
#     # make lists of buoy and other data sets
#     dflistBuoy=list()
#     dflistOther=list()
#     buoyRegEx=re.compile('NOAA Surface Ocean CO2 and Ocean Acidification Mooring Time Series')
#     for ind, row in dfInfo.iterrows():
#         #print(row['title'])
#         if buoyRegEx.search(row['title']):
#             dflistBuoy.append(row['datasetID'])
#         else:
#             dflistOther.append(row['datasetID'])
#     #remove special case from buoy data list:
#     dflistBuoy.remove('all_pmel_co2_moorings')
#     titledict=dict(zip(dfInfo['datasetID'],dfInfo['title']))
#     return dfInfo, dflistBuoy, dflistOther, titledict
# 
# def OAPInfoERDDAPBuoys(dfInfo, dflistBuoy, dflistOther):
#     # make smaller table containing only buoy datasets:
#     dfInfoBuoy=dfInfo.loc[dfInfo.datasetID.isin(dflistBuoy)].\
#             reset_index(drop=True).copy(deep=True)
#     # check that min and max lon/lat are the same for all and if so,
#     # clean up df to simplify:
#     x1=len(dfInfoBuoy.loc[np.abs(dfInfoBuoy['minLongitude']-dfInfoBuoy['maxLongitude'])>0])
#     x2=len(dfInfoBuoy.loc[np.abs(dfInfoBuoy['minLatitude']-dfInfoBuoy['maxLatitude'])>0])
#     if x1+x2==0:
#         dfInfoBuoy.drop(columns=['maxLongitude','maxLatitude'],inplace=True)
#         dfInfoBuoy.rename(columns={'minLongitude':'Lon','minLatitude':'Lat'},inplace=True)
#     # make Lon deg East: will throw error if test above failed
#     dfInfoBuoy['Lon']=-1*dfInfoBuoy['Lon']
#     return dfInfoBuoy
# 
# def buoySort(dfInfoBuoy):
#     lon1=-95; lon2=-139; lat1=15
#     z0=dfInfoBuoy.loc[dfInfoBuoy['Lon']>lon1,['datasetID','Lon','Lat']].\
#         sort_values(by='Lat',ascending=False).copy(deep=True)
#     z0['ind']=np.arange(0,len(z0))
#     z1=dfInfoBuoy.loc[(dfInfoBuoy.Lon<=lon1)&(dfInfoBuoy.Lon>lon2)&(dfInfoBuoy.Lat>lat1),
#                       ['datasetID','Lon','Lat']].\
#             sort_values(by='Lat',ascending=True).copy(deep=True)
#     z1['ind']=np.arange(len(z0),len(z0)+len(z1))
#     z2=dfInfoBuoy.loc[(dfInfoBuoy.Lon<=lon2)&(dfInfoBuoy.Lat>lat1),['datasetID','Lon','Lat']].\
#             sort_values(by='Lat',ascending=False).copy(deep=True)
#     z2['ind']=np.arange(len(z0)+len(z1),len(z0)+len(z1)+len(z2))
#     z3=dfInfoBuoy.loc[(dfInfoBuoy.Lon<=lon1)&(dfInfoBuoy.Lat<=lat1),['datasetID','Lon','Lat']].\
#             sort_values(by='Lon',ascending=True).copy(deep=True)
#     z3['ind']=np.arange(len(z0)+len(z1)+len(z2),len(z0)+len(z1)+len(z2)+len(z3))
#     zs=pd.concat([z0,z1,z2,z3])
#     dfInfoBuoy=dfInfoBuoy.merge(zs[['datasetID','ind']],on='datasetID').sort_values(by=['ind']).\
#             reset_index(drop=True).drop(columns='ind') 
#     return dfInfoBuoy

def countreal(x):
    return np.sum(~np.isnan(x))

def mapBuoyLocs(dfInfoBuoy,figsize=(5,5)):
    proj=ccrs.Mercator(central_longitude=-90,min_latitude=-10,max_latitude=70)
    tran = ccrs.Geodetic(); #PlateCarree()
    fig,ax=plt.subplots(1,1,figsize=figsize, subplot_kw=dict(projection=proj))
    ax.set_extent([np.min(dfInfoBuoy['Lon'])-5,np.max(dfInfoBuoy['Lon'])+5,
                   np.min(dfInfoBuoy['Lat'])-5,np.max(dfInfoBuoy['Lat'])+5])
    ax.coastlines()
    ax.plot(dfInfoBuoy['Lon'].values,dfInfoBuoy['Lat'].values,'r+',transform=tran)
    ax.plot(dfInfoBuoy['Lon'].values,dfInfoBuoy['Lat'].values,'b-',transform=tran)
    return fig, ax

# def loadOAPBuoyERDDAP(dfInfoBuoy,buoyName,freq='Daily'):
#     iB=dfInfoBuoy.title.str.contains(buoyName)
#     tit=dfInfoBuoy.loc[iB,['title']].values[0][0]
#     print(f'loading {tit}')
#     lat,lon=dfInfoBuoy.loc[iB,['Lat','Lon']].values[0]
#     with nc.Dataset(erdstr0+dfInfoBuoy.loc[iB,['datasetID']].values[0][0]) as f:
#         torig=dt.datetime.strptime(f.variables['s.time'].time_origin,'%d-%b-%Y %H:%M:%S')
#         obsdtUTC=[torig+dt.timedelta(seconds=ii) for ii in f.variables['s.time'][:]]
#         dayind=cf.dayindex(obsdtUTC)
#         yd=cf.ydfromdt(obsdtUTC)
#         df0=pd.DataFrame(data=np.array([obsdtUTC,dayind,yd]).T,columns=['dtUTC','dayind','yd'])
#         varlist=[el for el in f.variables.keys() if el not in ['s.station_id', 's.longitude', 's.latitude', 's.time']]
#         agdic=dict()
#         for ivar in varlist:
#             vname=ivar[2:]
#             agdic[vname]=[np.nanmean,np.nanstd,countreal]
#             df0[vname]=f.variables[ivar][:]
#     df0['SSSq']=df0['SSS']**2
#     agdic['SSSq']=[np.nanmean,np.nanstd,countreal]
#     df=df0.groupby(['dayind','yd']).agg(agdic).reset_index()
#     df0['dtUTC']=obsdtUTC
#     df['dtUTC']=[dt.datetime(1900,1,1)+dt.timedelta(days=ii)+dt.timedelta(hours=12) for ii in df['dayind']]
#     return [lat,lon], df0, df

# def loadOAPBuoyERDDAPNew(dfInfoBuoy,buoyID,freq='daily'):
#     iB=dfInfoBuoy.datasetID.str.contains(buoyID)|dfInfoBuoy.title.str.contains(buoyID)
#     tit=dfInfoBuoy.loc[iB,['title']].values[0][0]
#     #print(f'loading {tit}')
#     lat,lon=dfInfoBuoy.loc[iB,['Lat','Lon']].values[0]
#     with nc.Dataset(erdstr0+dfInfoBuoy.loc[iB,['datasetID']].values[0][0]) as f:
#         torig=dt.datetime.strptime(f.variables['s.time'].time_origin,'%d-%b-%Y %H:%M:%S')
#         obsdtUTC=[torig+dt.timedelta(seconds=ii) for ii in f.variables['s.time'][:]]
#         timeind=cf.timeindex(obsdtUTC,freq)
#         df0=pd.DataFrame(data=np.array([obsdtUTC,timeind]).T,columns=['dtUTC','timeind'])
#         varlist=[el for el in f.variables.keys() if el not in ['s.station_id', 's.longitude', 's.latitude', 's.time']]
#         agdic=dict()
#         for ivar in varlist:
#             vname=ivar[2:]
#             agdic[vname]=[np.nanmean,np.nanstd,countreal]
#             df0[vname]=f.variables[ivar][:]
#     df=df0.groupby(['timeind']).agg(agdic).reset_index()
#     df0['dtUTC']=obsdtUTC
#     if freq=='daily':
#         df['dtUTC']=[dt.datetime(1900,1,1)+dt.timedelta(days=ii)+dt.timedelta(hours=12) for ii in df['timeind']]
#     elif freq=='monthly':
#         df['dtUTC']=[dt.datetime(1900,1,1)+dateutil.relativedelta.relativedelta(months=ii)+dt.timedelta(days=14) for ii in df['timeind']]
#     return [lat,lon], df0, df

def loadOAPBuoy(dfInfoBuoy,buoyID,freq='daily',filtSSS=True,cCalcs=True):
    iB=dfInfoBuoy.datasetID.str.contains(buoyID)|dfInfoBuoy.title.str.contains(buoyID)
    lat,lon,dsid,shortTitle=dfInfoBuoy.loc[iB,['Lat','Lon','datasetID','shortTitle']].values[0]
    with nc.Dataset(localPath+'ncfiles/'+dfInfoBuoy.loc[iB,['datasetID']].values[0][0]+'.nc') as f:
        torigNC=dt.datetime.strptime(f.variables['time'].time_origin,'%d-%b-%Y %H:%M:%S')
        obsdtUTC=np.array([dt.datetime(1900,1,1)+dt.timedelta(seconds=(el-np.datetime64('1900-01-01')).astype(int)*1e-9) for el in fch['time'].values])
        tref=dt.datetime(1975,1,1,0,0) # use start of model sections for consistency (alt: obsdtUTC[0])
        timeind=cf.to_int_tind(obsdtUTC,freq=freq,torig=tref)
        exday=cf.to_exact_tind(obsdtUTC,torig=tref)
        df0=pd.DataFrame(data=np.array([obsdtUTC,timeind,exday]).T,columns=['dtUTC','timeind','exactDay'],dtype='object')
        varlist=[el for el in f.variables.keys() if el not in ['station_id', 'longitude', 'latitude', 'time']]
        agdic=dict()
        udict=dict()
        ldict=dict()
        agdic['exactDay']=[np.nanmean]
        for ivar in varlist:
            vname=ivar
            #note: some units are incorrect. long_name appears to have correct units.
            udict[vname]=f.variables[ivar].units if 'units' in f.variables[ivar].ncattrs() else ''
            ldict[vname]=f.variables[ivar].long_name if 'long_name' in f.variables[ivar].ncattrs() else ''
            agdic[vname]=[np.nanmean,np.nanstd,countreal]
            df0[vname]=f.variables[ivar][:]
    # some QC:
    ## SST:
    df0['SSTorig']=[val for val in df0['SST']]
    df0['SST']=[val if val>-8 else np.nan for val in df0['SSTorig']]
    if dsid=='pmel_co2_moorings_c2e7_ecb9_4565': # Gulf of Maine 
        dftemp=df0.copy(deep=True)
        dftemp['month']=[el.month for el in dftemp['dtUTC']]
        iii1=df0['dtUTC']<dt.datetime(2008,9,1)
        iii2=(df0.SST<1)&(dftemp.month.isin([6,7,8,9,10]))
        df0.loc[iii1|iii2,['SST']]=np.nan
        del dftemp
    ## SSS:
    if filtSSS: #time-consuming
        # since times are not regular, use time window
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',message="Warning: assuming tdt is float representing elapsed time in days")
            sval2=cf.stdFilt(exday,np.array(df0['SSS'].values),window_days=45,nstdev=3)
        df0['SSSorig']=df0['SSS'].values
        df0['SSS']=sval2
    ## pH:
    if shortTitle=='Stratus':
        df0['pH_sworig']=[val for val in df0['pH_sw']]
        df0.loc[obsdtUTC<dt.datetime(2014,1,1),['pH_sw']]=np.nan # the few points before 2014 are very far from the S-Alk based estimate                  
    # calculate H+ from pH before averaging
    df0['hplus']=1e6*10**(-1*df0['pH_sw']) # umol/l
    udict['hplus']='μmol/l'
    ldict['hplus']='Hydrogen Ion Concentration Converted From pH'
    agdic['hplus']=[np.nanmean,np.nanstd,countreal]
    #cCalcs
    if cCalcs:
        df0['TA_est']=AlkFits.OAP_Alk(shortTitle,df0['SSS'],df0['SST'],lon,lat)
        agdic['TA_est']=[np.nanmean,np.nanstd,countreal]
        res=cc.pyco2_surf_all_from_TA_pCO2(df0['TA_est'],df0['pCO2_sw'],df0['SSS'],df0['SST'])
        res15C=cc.pyco2_surf_all_from_TA_pCO2(df0['TA_est'],df0['pCO2_sw'],df0['SSS'],df0['SST'],temperature_out=15)
        resphpc=cc.pyco2_surf_all_from_pH_pCO2(df0['pH_sw'],df0['pCO2_sw'],df0['SSS'],df0['SST'])
        for ikey in cc.savelist:
            df0['cc_'+ikey]=res[ikey]
            agdic['cc_'+ikey]=[np.nanmean,np.nanstd,countreal]
        df0['cc_pH15C']=res15C['pH_total']
        agdic['cc_pH15C']=[np.nanmean,np.nanstd,countreal]
        if 'saturation_aragonite' in cc.savelist:
            df0['cc_saturation_aragonite_phpc']=resphpc['saturation_aragonite']
            agdic['cc_saturation_aragonite_phpc']=[np.nanmean,np.nanstd,countreal]
    # resample to requested time interval
    df=df0.groupby(['timeind']).agg(agdic).reset_index()
    # added 10/3/22  because flattened dataset is easier to work with:
    df.columns = [ii+'_'+jj if len(jj)>0 else ii for ii,jj in \
              zip(df.columns.get_level_values(0),df.columns.get_level_values(1))]
    df0['dtUTC']=pd.Series(obsdtUTC,dtype='object')
    # return mean datetime instead of center of interval
    df['dtUTC']=pd.Series([tref+dt.timedelta(days=ii) for ii in df['exactDay_nanmean']],dtype='object')
    # if freq=='daily':
    #     df['dtUTC']=[dt.datetime(1900,1,1)+dt.timedelta(days=ii)+dt.timedelta(hours=12) for ii in df['timeind']]
    # elif freq=='monthly':
    #     df['dtUTC']=[dt.datetime(1900,1,1)+dateutil.relativedelta.relativedelta(months=ii)+dt.timedelta(days=14) for ii in df['timeind']]
    return [lat,lon], ldict, udict, df0, df

def modpath(dsetID,q='1x1grid',scen='GFDL-ESM4.1.1975_2022',basedir=None):
    warnings.warn("change from OAPBuoyData.modpath(dsetID,q='1x1grid',scen='ESM4_historical_D1',basedir=None)"\
                    " to diagsPP.searchExtracted(scen,dsetID,'_1x1grid',basedir)")
    if basedir is None:
        basedir=calcsPath
    return diagsPP.searchExtracted(scen,dsetID,q,basedir)
    # # q can be 1x1grid or 288grid; defaults to 1x1
    # # srchstr=f"{basedir}*{diagsPP.scenNameDict[scen]}.{dsetID}*{q}*.nc"
    # srchstr=diagsPP.extractedTSPath(scen,dsetID,'*','*',q,basedir)
    # fpath=glob.glob(srchstr)
    # if not len(fpath)==1:
    #     raise ValueError(f'fpath has wrong size: searchstring={srchstr}; fpath={fpath}')
    # return fpath[0]

def getID(titlephrase):
    dfb=loadOAPInfo()
    ind=dfb.shortTitle==titlephrase
    if np.sum(ind)==0:
        ind=dfb.title.str.contains(titlephrase)
        if np.sum(ind)==0:
            raise RuntimeError('no matching title')
        elif np.sum(ind)==1:
            return dfb.loc[ind,['datasetID']].values[0][0]
        else:
            display(dfb.loc[ind])
            return input('Enter ID choice:')
    elif np.sum(ind)==1:
        return dfb.loc[ind,['datasetID']].values[0][0]
    else:
        display(dfb.loc[ind])
        return input('Enter ID choice:')


#Sutton et al 2019 regions from Table 1
regionDictSutton2019={'CCE1':'Northeast Pacific Ocean',
    'Papa':'Northeast Pacific Ocean',
    'KEO':'Northwest Pacific Ocean',
    'JKEO':'Northwest Pacific Ocean',
    'MOSEAN/WHOTS':'Central Pacific Ocean',
    'TAO110W':'Equatorial Pacific Ocean',
    'TAO125W':'Equatorial Pacific Ocean',
    'TAO140W':'Equatorial Pacific Ocean',
    'TAO155W':'Equatorial Pacific Ocean',
    'TAO170W':'Equatorial Pacific Ocean',
    'TAO165E':'Equatorial Pacific Ocean',
    'TAO8S165E':'Equatorial Pacific Ocean',
    'Stratus':'Southeast Pacific Ocean',
    'BTM':'North Atlantic Ocean',
    'Iceland':'North Atlantic Ocean',
    'BOBOA':'Indian Ocean',
    'SOFS':'Southern Ocean',
    'GAKOA':'Alaskan coast',
    'Kodiak':'Alaskan coast',
    'SEAK':'Alaskan coast',
    'M2':'Bering Sea coastal shelf',
    'Cape Elizabeth':'US west coast',
    'Cha ba':'US west coast',
    'CCE2':'US west coast',
    'Dabob':'US west coast',
    'NH-10':'US west coast',
    'Twanoh':'US west coast', # excluded
    'Cape Arago':'US west coast',
    'Ala Wai':'Pacific island coral reef',
    'Chuuk':'Pacific island coral reef',
    'CRIMP1':'Pacific island coral reef',
    'CRIMP2':'Pacific island coral reef',
    'Kaneohe':'Pacific island coral reef',
    'Kilo Nalu':'Pacific island coral reef',
    "Gray's Reef":'US east coast',
    'Gulf of Maine':'US east coast',
    'Crescent Reef':'Atlantic coral reef',
    'Hog Reef':'Atlantic coral reef',
    'Coastal MS':'Gulf of Mexico coast',
    'Coastal LA':'Gulf of Mexico coast',
    'Cheeca Rocks':'Caribbean coral reef',
    'La Parguera':'Caribbean coral reef',
    'First Landing':'US east coast', # added
    }
