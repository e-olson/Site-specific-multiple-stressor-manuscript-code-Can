#from multiprocessing import Pool
import sys
import os
import socket
import pickle
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as scopt
import commonfxns as cf, OAPBuoyData as OAP, evalfxns as ev, viz, diagsPP, conversions
import netCDF4 as nc
import cftime
import datetime as dt
# import cmocean
import gsw
# from sklearn.linear_model import TheilSenRegressor
import warnings

vardict={'phos':'pH_sw_nanmean','spco2':'pCO2_sw_nanmean','apco2':'pCO2_air_nanmean','co2dryair':'xCO2_air_nanmean',
         'tos':'SST_nanmean','sos':'SSS_nanmean','chlos':'CHL_nanmean',
         'dpco2':'dpco2','l10chlos':'l10CHL','o2os':'DOXY_nanmean','hplusos':'hplus_nanmean',
         'phosC':'cc_pH_total_nanmean','talkos':'TA_est_nanmean',
         'omega_a_0':'cc_saturation_aragonite_nanmean',
         'omega_a_P_0':'cc_saturation_aragonite_phpc_nanmean',
         'omega_c_0':'cc_saturation_calcite_nanmean',
         'dissicos':'cc_dic_nanmean', 'o2percos':'o2perc','AOUos':'AOU',}
dispName={'phos':'Surface pH','phosC':'Est. Surface pH','spco2':'Surface pCO$_2$','apco2':'Surface Air pCO$_2$',
          'co2dryair':'Air CO$_2$','dpco2':'$\Delta$ pCO$_2$',
        'tos':'SST','sos':'SSS','chlos':'Surface Chl','l10chlos':'log$_{10}$(Surface Chl)','o2os':'DO','hplusos':'[H$^{+}$]',
        'MLD_003':'Mixed layer depth (delta rho = 0.03)',
        'intpp':'Int. PP',
        'fgco2':'Surface Downward Flux of Total CO2',
        'talkos':'TA',
        'dissicos':'DIC',
        'dic_kw':'Gas Exchange piston velocity for dic',
        'dic_sc_no':'Ocean surface Schmidt Number for dic',
        'pso':'Sea Water Pressure at Sea Water Surface',
        'friver':'Water Flux into Sea Water From Rivers',
        'nsmp_100':'Small phytoplankton, upper 100m',
        'nlgp_100':'Large phytoplankton, upper 100m',
        'ndi_100':'Diazotrophs, upper 100m',
        'o2satos':'Surface DO at Saturation',
        'no3os':'NO$_3$',
        'ph':'pH',
        'o2':'DO',
        'o2sat':'DO at Saturation',
        'o2percos':'Percent Oxygen Saturation',
        'AOUos':'AOU',
        'chl':'Chl',
        'thetao':'Potential Temperature',
        'so':'Sea Water Salinity',
        'talk':'Total Alkalinity',
        'dissic':'DIC',
        'co3':'CO$_3^{2-}$',
        'co3satarag':'Mole Concentration of Carbonate Ion in Equilibrium with Pure Aragonite in sea water',
        'co3satcalc':'Mole Concentration of Carbonate Ion in Equilibrium with Pure Calcite in sea water',
        'no3':'Dissolved Nitrate Concentration',
        'mesozoo_200':'Int. Mesozooplankton, 200m',
        'omega_a_0':'$\Omega_a$',
        'omega_a_P_0':'$\Omega_a$(pH,pCO$_2$)',
        'omega_a_50':'50 m $\Omega_a$',
        'omega_c_0':'$\Omega_c$',
        'omega_c_50':'50 m $\Omega_c$',}

dispUnits={'phos':'','phosC':'','spco2':'μatm','apco2':'μatm','co2dryair':'μmol mol$^{-1}$','dpco2':'μatm',
        'tos':'°C','sos':'psu','chlos':'μg l$^{-1}$','o2os':'μmol kg$^{-1}$','l10chlos':'log(μg l$^{-1}$)','hplusos':'μmol l$^{-1}$',
        'MLD_003':'m',
        'intpp':'mol m$^{-2}$ s$^{-1}$',
        'fgco2':'kg m$^{-2}$ s$^{-1}$',
        'talkos':'μmol kg$^{-1}$',
        'dissicos':'μmolkg$^{-1}$',
        'dic_kw':'m/sec',
        'dic_sc_no':'mol/kg',
        'pso':'Pa',
        'friver':'kg m-2 s-1',
        'nsmp_100':'mol m$^{-2}$',
        'nlgp_100':'mol m$^{-2}$',
        'ndi_100':'mol m$^{-2}$',
        'phos':'1',
        'o2satos':'mol m$^{-3}$',
        'o2percos':'%',
        'AOUos':'μmol kg$^{-1}$',
        'no3os':'μmol kg$^{-1}$',
        'ph':'1',
        'z_i':'meters',
        'z_l':'meters',
        'o2':'mol m$^{-3}$',
        'o2sat':'mol m$^{-3}$',
        'chl':'kg m$^{-3}$',
        'thetao':'degC',
        'so':'psu',
        'talk':'mol m$^{-3}$',
        'dissic':'mol m$^{-3}$',
        'co3':'mol m$^{-3}$',
        'co3satarag':'mol m$^{-3}$',
        'co3satcalc':'mol m$^{-3}$',
        'no3':'mol m$^{-3}$',
        'mesozoo_200':'mol m$^{-2}$',
        'omega_a_0':' ',
        'omega_a_P_0':' ',
        'omega_a_50':' ',
        'omega_c_0':' ',
        'omega_c_50':' ',}
dispNameUnits={'phos':'Surface pH','phosC':'Surface pH','spco2':'Surface pCO$_2$ (μatm)','apco2':'Surface Air pCO$_2$ (μatm)',
               'co2dryair':'Air CO$_2$ (μmol mol$^{-1}$)',
        'tos':'SST (°C)','sos':'SSS (psu)','chlos':'Surface Chl (μg l$^{-1}$)','l10chlos':'log$_{10}$[Chl (μg l$^{-1}$)]',
        'o2os':'DO (μmol kg$^{-1}$)','dpco2':'$\Delta$ pCO$_2$ (μatm)','hplusos':'[H$^{+}$] (μmol l$^{-1}$)',
        'MLD_003':'Mixed layer depth (delta rho = 0.03) (m)',
        'intpp':'Primary Organic Carbon Production by All Types of Phytoplankton (mol m-2 s-1)',
        'fgco2':'Surface Downward Flux of Total CO2 (kg m-2 s-1)',
        'talkos':'Surface Total Alkalinity (μmol kg$^{-1}$)',
        'dissicos':'Surface Dissolved Inorganic Carbon Concentration (μmolkg$^{-1}$)',
        'dic_kw':'Gas Exchange piston velocity for dic (m/sec)',
        'dic_sc_no':'Ocean surface Schmidt Number for dic (mol/kg)',
        'pso':'Sea Water Pressure at Sea Water Surface (Pa)',
        'friver':'Water Flux into Sea Water From Rivers (kg m-2 s-1)',
        'nsmp_100':'Small phytoplankton nitrogen biomass in upper 100m (mol m-2)',
        'nlgp_100':'Large phytoplankton nitrogen biomass in upper 100m (mol m-2)',
        'ndi_100':'Diazotroph nitrogen biomass in upper 100m (mol m-2)',
        'o2satos':'Surface Dissolved Oxygen Concentration at Saturation (mol m-3)',
        'no3os':'Surface Dissolved Nitrate Concentration (μmol kg$^{-1}$)', 
        'ph':'pH (1)',
        'o2':'Dissolved Oxygen Concentration (mol m-3)',
        'o2sat':'Dissolved Oxygen Concentration at Saturation (mol m-3)',
        'o2percos':'Oxygen Saturation (%)',
        'AOUos':'AOU (μmol kg$^{-1}$)',
        'chl':'Mass Concentration of Total Phytoplankton expressed as Chlorophyll in sea water (kg m-3)',
        'thetao':'Sea Water Potential Temperature (degC)',
        'so':'Sea Water Salinity (psu)',
        'talk':'Total Alkalinity (mol m-3)',
        'dissic':'Dissolved Inorganic Carbon Concentration (mol m-3)',
        'co3':'Carbonate Ion Concentration (mol m-3)',
        'co3satarag':'Mole Concentration of Carbonate Ion in Equilibrium with Pure Aragonite in sea water (mol m-3)',
        'co3satcalc':'Mole Concentration of Carbonate Ion in Equilibrium with Pure Calcite in sea water (mol m-3)',
        'no3':'Dissolved Nitrate Concentration (mol m-3)',
        'mesozoo_200':'Integrated Mesozooplankton (mol m-2)',
        'omega_a_0':'Surface $\Omega_a$',
        'omega_a_P_0':'Surface $\Omega_a$(pH,pCO$_2$)',
        'omega_a_50':'50 m $\Omega_a$',
        'omega_c_0':'Surface $\Omega_c$',
        'omega_c_50':'50 m $\Omega_c$',}

vproclist=['AOUos',]#['tos','sos','spco2','o2os','chlos',]#'phos','phosC','talkos','spco2','dpco2','o2os','co2dryair','chlos','l10chlos',
           #'hplusos','omega_a_0','omega_a_P_0','omega_c_0','dissicos','AOUos','o2percos','apco2']
vproclistM=[]#['intpp','mesozoo_200','omega_a_50','omega_c_50','no3os'] # model only variables
# spco2: 1 atm = 101325 Pa; 1 μatm=0.101325 Pa; Pa*1/(0.101325)->μatm
# chl: ug/l=mg/m3   kg/m3*1e6=mg/m3
# DO: # mol/m3 -> umol/kg:
varMult={'phos':1,'phosC':1,'spco2':1/0.101325,'apco2':1/0.101325,'dpco2':1/0.101325,'co2dryair':1,'tos':1,'sos':1,'chlos':1e6,
         'hplusos':1,'talkos':1,'dissicos':1e6}
varUncert={'tos':0.01,'sos':0.05,'spco2':2,'apco2':1,'co2dryair':0.02,'dpco2':2.24,'o2os':np.nan,
            'phos':0.02,'phosC':np.nan,'chlos':0.013}

HDgrid=False # default to false
# if 'stellar' in socket.gethostname():
#     savebase='/scratch/cimes/eo2651/calcs/buoyCompTS/'
# else:
#     savebase='/work/Elise.Olson/calcs/buoyCompTS/'
savebase='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/'
figsaveloc=savebase+'figs/'
compsaveloc=savebase+'comps/'

def figpath(figtype,stid,mvar=None,freq='daily'):
    return ev.figpath(figtype,figsaveloc,stid,mvar,freq)

def compPath(stationID,modvar,freq='daily',csaveloc=None):
    if csaveloc is None:
        csaveloc=compsaveloc
    return ev.getcompsavepath(csaveloc,stationID,modvar,freq)

def loadMoorComp(stationID,modvar,freq='daily',path=None):
    if path is None:
        path=compsaveloc
    ppath=compPath(stationID,modvar,freq,path)
    with open(ppath,'rb') as ff:
        comp=pickle.load(ff)
    return comp

def obsload(modvar,df):
    # df should be loaded from OAP.loadOAPBuoy
    modvarobs=vardict[modvar]
    if modvarobs in ('pH_sw_nanmean','pCO2_sw_nanmean','xCO2_air_nanmean','pCO2_air_nanmean',
                       'SST_nanmean','SSS_nanmean','CHL_nanmean','DOXY_nanmean',
                      'cc_pH_total_nanmean','TA_est_nanmean','hplus_nanmean','cc_dic_nanmean',
                      'cc_saturation_aragonite_nanmean','cc_saturation_aragonite_phpc_nanmean',
                      'cc_saturation_calcite_nanmean'): 
        # list of known obs vars not requiring additional calculations
        ii1=~pd.isnull(df[modvarobs]).values
        obs_val=np.squeeze(df[modvarobs].values[ii1])
    elif modvarobs=='dpco2':
        ii1=(~pd.isnull(df['pCO2_sw_nanmean']).values)&(~pd.isnull(df['pCO2_air_nanmean']).values)
        obs_val=np.squeeze(df['pCO2_sw_nanmean'].values[ii1])-\
            np.squeeze(df['pCO2_air_nanmean'].values[ii1])
    elif modvarobs=='l10CHL':
        ii1=~pd.isnull(df['CHL_nanmean']).values
        obs_val=np.squeeze(df['CHL_nanmean'].values[ii1])
        obs_val=np.log10(obs_val)
    elif modvarobs=='o2perc':
        ii1=~pd.isnull(df['DOXY_nanmean']*df['SST_nanmean']*df['SSS_nanmean'])
        obs_val=conversions.oxyperc_surf_from_o2umolkg(df['DOXY_nanmean'].values[ii1],
                        df['SST_nanmean'].values[ii1],df['SSS_nanmean'].values[ii1])
        obs_val=np.squeeze(obs_val)
    elif modvarobs=='AOU':
        ii1=~pd.isnull(df['DOXY_nanmean']*df['SST_nanmean']*df['SSS_nanmean'])
        o2sat=gsw.O2sol_SP_pt(df['SSS_nanmean'].values[ii1],df['SST_nanmean'].values[ii1]) # S is psu, T is pt, umol/kg
        obs_val=np.squeeze(o2sat-df['DOXY_nanmean'].values[ii1])
    else:
        raise NotImplementedError
    if np.sum(ii1)==0:
        raise ObsError(f"No obs data for {modvarobs} at site.")
    #obs_tdt=np.array([pd.Timestamp(ii[0]).to_pydatetime() for ii in df.loc[ii1,['dtUTC']].values])
    obs_tdt=np.squeeze(df.loc[ii1,['dtUTC']].values)
    return obs_tdt, obs_val, dispName[modvar], dispUnits[modvar], dispNameUnits[modvar], modvarobs
    
def modload(modvar,f1,f2,lon,lat,freq='daily'):
    # particular conversions to these obs data units
    # lon,lat needed for oxygen conversions
    mod_tnl=cf.noLeapFromNC(f1)
    mod_tdt=cf.cftnoleap_to_dt(mod_tnl)
    if modvar in ('phos','phosC','spco2','tos','sos','chlos','co2s','dpco2','intpp','mesozoo_200'):
        if modvar=='phosC': # use to match to calculated ph
            modvar='phos' 
        if modvar in varMult.keys():
            # list of mod vars requiring only multiplication by constant
            mod_val=f1.variables[modvar][:]*varMult[modvar]
        else: # stop having to add ones to varMult dict when no multiplication necessary
            mod_val=f1.variables[modvar][:]
    elif modvar in ('o2os','talkos','no3os','dissicos'):
        SA=gsw.SA_from_SP(f1.variables['sos'][:],0,lon,lat)
        CT=gsw.CT_from_t(SA,f1.variables['tos'][:],0)
        rho=gsw.rho(SA,CT,0) #kg/m3
        if modvar=='o2os':
            omod=f1.variables['o2os'][:]
            # DO: # mol/m3 -> umol/kg:   mol/m3/(rho kg/m3)*1e6 = umol/kg
            mod_val=omod/rho*1e6
        elif modvar=='talkos':
            omod=f1.variables['talkos'][:] # mol m-3
            mod_val=omod/rho*1e6 # mol m-3 -> umol/kg
        elif modvar=='dissicos':
            omod=f1.variables['dissicos'][:] # mol m-3
            mod_val=omod/rho*1e6 # mol m-3 -> umol/kg
        elif modvar=='no3os':
            omod=f1.variables['no3os'][:] # mol m-3
            mod_val=omod/rho*1e6 # mol m-3 -> umol/kg
    elif modvar=='AOUos':
        o2sat=gsw.O2sol_SP_pt(f1.variables['sos'][:],f1.variables['tos'][:]) # S is psu, T is pt, umol/kg
        SA=gsw.SA_from_SP(f1.variables['sos'][:],0,lon,lat)
        CT=gsw.CT_from_t(SA,f1.variables['tos'][:],0)
        rho=gsw.rho(SA,CT,0) #kg/m3
        omod=f1.variables['o2os'][:]
        # DO: # mol/m3 -> umol/kg:   mol/m3/(rho kg/m3)*1e6 = umol/kg
        mod_val=o2sat-omod/rho*1e6
    elif modvar=='o2percos':
        mod_val=f1.variables['o2os'][:]/f1.variables['o2satos'][:]*100
    elif modvar=='l10chlos':
        omodvar='chlos'
        mod_val=f1.variables[omodvar][:]*varMult[omodvar]
        mod_val=np.log10(mod_val)
    elif modvar=='hplusos':
        mod_val=1e6*10**(-1*f1.variables['phos'][:])*varMult[modvar]
    elif modvar.startswith('omega_a_') or modvar.startswith('omega_c_'): # format omega_a/c_depth
        try:
            satvar='co3satarag' if modvar.startswith('omega_a_') else 'co3satcalc'
            depth_m=modvar.split('_')[-1]
            k=viz.k_from_z(float(depth_m))
            co3=f1.variables['co3'][:,k,...]
            sat=f1.variables[satvar][:,k,...]
            mod_val=co3/sat
        except:
            raise # for now
    else:
        raise NotImplementedError
    # this function loads from extractedc daily series, so if monthly, need to calculated averages
    mod_tind=cf.to_int_tind(mod_tnl,freq=freq,torig=dt.datetime(1975,1,1,0,0))
    mod_tex=cf.to_exact_tind(mod_tnl,torig=dt.datetime(1975,1,1,0,0))
    if len(np.unique(mod_tind))<len(mod_tind):
        new_tind=np.unique(mod_tind)
        new_val=np.array([np.nanmean(mod_val[mod_tind==iind]) for iind in new_tind])
        new_tex=np.array([np.nanmean(mod_tex[mod_tind==iind]) for iind in new_tind])
        new_tnl=np.array([mod_tex[0].torig+dt.timedelta(days=iii) for iii in new_tex]) 
        # mod_tex[0].torig should have been converted to correct type (cftime) regardless of torig input format
        new_tdt=cf.cftnoleap_to_dt(new_tnl)
        # temptdt=[mod_tdt[mod_tind==iind][0] for iind in new_tind]
        # yrmon=[[iii.year,iii.month] for iii in temptdt]
        # new_tnl=np.array([cftime.datetime(iyr,imon,15,calendar='noleap') for iyr,imon in yrmon])
        # new_tdt=np.array([dt.datetime(iyr,imon,15) for iyr, imon in yrmon])
        return new_tnl, new_tdt, new_val
    return mod_tnl, mod_tdt, np.squeeze(mod_val)
      
ModError=ev.ModError
ObsError=ev.ObsError 

def makeSlopesList(icomp):
    stadd=np.zeros(30).astype(bool)
    nn=len(icomp.mod_val)
    stencil=np.zeros(nn)
    for ii in icomp.obs_tind-icomp.obs_tind[0]:
        stencil[ii]=1
    stencil=stencil.astype(bool)
    fitlist=[]
    ivals=icomp.mod_val[stencil]
    itind=icomp.mod_tind[stencil]
    while np.sum(stencil)==nn:
        mod_olsfit=cf.linreg(itind,ivals)
        fitlist.append(mod_olsfit)
        stencil=np.concatenate((stadd,stencil[:-30]))
        ivals=icomp.mod_val[stencil]
        itind=icomp.mod_tind[stencil]
    return fitlist

def compileStats(save=True,path=None,vlist=None,freq='daily'):
    if path is None:
        path=compsaveloc
    if vlist is None:
        vlist=[*vproclist,*vproclistM]
    dfInfoBuoy=OAP.loadOAPInfo()
    statsdf=list()
    for dsid in dfInfoBuoy.datasetID:
        locName=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title']].values[0][0]
        print(locName)
        for mvar in vlist:
            print(mvar)
            try:
                mm=loadMoorComp(dsid,mvar,freq=freq,path=path)
                statsdf.append(mm.statsSummary())
            except FileNotFoundError:
                print('no file')
    statsdf=pd.DataFrame(statsdf)
    if save:
        #with open(savebase+'statsdfs.pkl', 'wb') as hh: # old version: save in directory above comps
        with open(path+'statsdfs.pkl', 'wb') as hh: # new version: save in comps dir
            pickle.dump(statsdf, hh, protocol=pickle.HIGHEST_PROTOCOL)
        # if statsdfsMerged.pkl exists in this directory, delete it so that it will be remade when it is next loaded
        if os.path.isfile(path+'statsdfsMerged.pkl'):
            os.remove(path+'statsdfsMerged.pkl')
    return statsdf

def loadStats(merged=False,path=None,vlist=None):
    print(path)
    if path is None:
        path=compsaveloc
    if merged:
        try:
            with open(path+'statsdfsMerged.pkl', 'rb') as hh:
                statsdf=pickle.load(hh)
        except:
            statsdf0=loadStats(merged=False,path=path)
            dfInfoBuoy=OAP.loadOAPInfo(modMeans=False)
            # if path==compsaveloc:
            #     dfInfoBuoy=OAP.loadOAPInfo(modMeans=True) #make sure this isn't necessary
            # else:
            #     dfInfoBuoy=OAP.loadOAPInfo(modMeans=False)
            #     print('No means loaded for this path')
            #if not path==compsaveloc:
            #    raise Error('Not Implemented; means would be from historical simulation; path=',path)
            statsdf=dfInfoBuoy.merge(statsdf0,left_on='datasetID',right_on='stationID')
            with open(path+'statsdfsMerged.pkl', 'wb') as hh:
                pickle.dump(statsdf,hh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(path+'statsdfs.pkl', 'rb') as hh:
                statsdf=pickle.load(hh)
        except:
            statsdf=compileStats(path=path)
    return statsdf

def makePlots():
    comps = loadAllComps()
    for im in comps.keys():
        #print(im)
        mm=comps[im]
        fig,ax=mm.plot_seas_b(save=True,dpi=300)
        plt.close(fig)
        fig,ax=mm.plot_ts(includefits=['optfit','linfit1'],save=True,dpi=300)
        plt.close(fig)
    print('done')
    return

def loadAllComps(compsdir=None,freq='daily',mvar=None):
    if compsdir is None:
        compsdir=compsaveloc
    listf = sorted( filter( lambda x: os.path.isfile(os.path.join(compsdir, x)),
                        os.listdir(compsdir) ) )
    if freq=='daily' or freq =='monthly':
        listf = [el for el in listf if freq in el]
    if mvar is not None:
        listf = [el for el in listf if mvar in el]
    def getparts(mcname):
        sp=mcname.split('.')
        mname=sp[1]
        vname=sp[2]
        freq=sp[3]
        return (mname,vname,freq)
    
    comps=dict()
    for el in listf:
        ikey=getparts(el)
        comps[ikey]=loadMoorComp(*ikey,path=compsdir)
    return comps

def setupClusterDF(staid):
    dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
    dsid=OAP.getID(staid)
    shortTitle=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['shortTitle']].values[0][0]
    baseTitle=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title']].values[0][0]
    jj0,ii0=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['jj','ii']].values[0]

    llist=ev.ClusterList(jj0,ii0)
    newdflist=[]
    for ix, (jj,ii) in enumerate(llist):
        idsid=dsid+f'_{ix:02d}'
        ilat=viz.glat1x1[jj,ii]
        ilon=viz.glon1x1[jj,ii]
        jj288,ii288=cf.nearest_point(ilat,ilon,
                               viz.glat288,viz.glon288,thresh=100,tol=2)
        newdflist.append([idsid,shortTitle+f' {ix:02d}',jj,ii,jj288,ii288,ilat,ilon,dsid,baseTitle])
    dfb=pd.DataFrame(newdflist,columns=['datasetID','shortTitle','jj','ii','jj288','ii288','Lat1x1','Lon1x1','base_datasetID','base_title'])
    Extsaveloc=OAP.calcsPath+f'Clusters/{shortTitle}/'
    return dfb, Extsaveloc

def obsloadNull(modvar,extras=None):
    # df should be loaded from OAP.loadOAPBuoy
    #modvarobs=vardict[modvar]
    obs_val=np.array([np.nan,])
    obs_tdt=np.array([dt.datetime(2010,1,1),])
    return obs_tdt, obs_val, dispName[modvar], dispUnits[modvar], dispNameUnits[modvar], 'None' #modvarobs

def run1Comp(ind):
    ind=int(ind)
    dfInfoBuoy=OAP.loadOAPInfo()
    dsid=dfInfoBuoy.datasetID[ind]
    locName,shortName=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title','shortTitle']].values[0]
    print(dsid,locName)
    [lat,lon], ldict, udict, df0, df = OAP.loadOAPBuoy(dfInfoBuoy,dsid,freq='daily')
    with nc.Dataset(OAP.modpath(dsid)) as f1:#, nc.Dataset(OAP.modpath(dsid,'288grid')) as f2:
        for mvar in vproclist:
            print(mvar)
            try:
                mmm=ev.timeSeriesComp(mvar,dsid,locName,shortName,lat,lon,
                        obsloadfun=obsload,obsloadkwargs={'df':df},
                        modloadfun=modload,modloadkwargs={'f1':f1,'f2':None,'lon':lon,'lat':lat,'freq':'daily'},
                        freq='daily',savepath=savebase,figsavepath=figsaveloc,compsavepath=compsaveloc)
                if mvar in ['co2dryair','apco2']:
                    mmm.calc_fits(fitlist=['quadfit','optfit','linfit1'],fitlistOL=['linfit1'],
                                     defaultfit='quadfit',predefined={'obs_b2':'mod_b2'})
                else:
                    mmm.calc_fits()
                mmm.calc_stats()
                mmm.topickle()
            except ObsError as err:
                print('ObsError:',err.value)
                print(f'Failed:({dsid},{mvar})')
            except ModError as err:
                print('ModError:',err.value)
                print(f'Failed:({dsid},{mvar})')
    return

# def run1CompV2(ind,dfInfoBuoy):
#     dsid=dfInfoBuoy.datasetID[ind]
#     locName=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title']].values[0][0]
#     print(dsid,locName)
#     [lat,lon], ldict, udict, df0, df = OAP.loadOAPBuoy(dfInfoBuoy,dsid,freq='daily')
#     with nc.Dataset(OAP.modpath(dsid)) as f1, nc.Dataset(OAP.modpath(dsid,'288grid')) as f2:
#         for mvar in vproclist:
#             print(mvar)
#             try:
#                 mmm=moorComp(mvar,dsid,locName,lat,lon,df,f1,f2)
#                 mmm.calc_fits()
#                 mmm.calc_stats()
#                 mmm.topickle()
#             except ObsError as err:
#                 print('ObsError:',err.value)
#                 print(f'Failed:({dsid},{mvar})')
#             except ModError as err:
#                 print('ModError:',err.value)
#                 print(f'Failed:({dsid},{mvar})')
#     return

def run1CompMonth(ind):
    ind=int(ind)
    dfInfoBuoy=OAP.loadOAPInfo()
    dsid=dfInfoBuoy.datasetID[ind]
    locName,shortName=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title','shortTitle']].values[0]
    print(dsid,locName)
    [lat,lon], ldict, udict, df0, df = OAP.loadOAPBuoy(dfInfoBuoy,dsid,freq='monthly')
    with nc.Dataset(OAP.modpath(dsid)) as f1, nc.Dataset(OAP.modpath(dsid,'288grid')) as f2:
        for mvar in vproclist:
            print(mvar)
            try:
                mmm=ev.timeSeriesComp(mvar,dsid,locName,shortName,lat,lon,
                        obsloadfun=obsload,obsloadkwargs={'df':df},
                        modloadfun=modload,modloadkwargs={'f1':f1,'f2':f2,'lon':lon,'lat':lat,'freq':'monthly'},
                        freq='monthly',savepath=savebase,figsavepath=figsaveloc,compsavepath=compsaveloc)
                if mvar in ['co2dryair','apco2']:
                    mmm.calc_fits(fitlist=['quadfit','optfit','linfit1'],fitlistOL=['linfit1'],
                                     defaultfit='quadfit',predefined={'obs_b2':'mod_b2'})
                else:
                    mmm.calc_fits()
                #mmm.calc_fits()
                mmm.calc_stats()
                mmm.topickle()
            except ObsError as err:
                print('ObsError:',err.value)
                print(f'Failed:({dsid},{mvar})')
            except ModError as err:
                print('ModError:',err.value)
                print(f'Failed:({dsid},{mvar})')
    return

def run1CompCluster(ind,staid):
    ind=int(ind)
    dfInfoBuoy,saveLoc=setupClusterDF(staid)
    print(saveLoc)
    dsid=dfInfoBuoy.datasetID[ind]
    locName,shortName,lat,lon=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['base_title',
                                                            'shortTitle','Lat1x1','Lon1x1']].values[0]
    print(dsid,locName)
    with nc.Dataset(OAP.modpath(dsid,'1x1grid',basedir=saveLoc)) as f1, nc.Dataset(OAP.modpath(dsid,'288grid',basedir=saveLoc)) as f2:
        for mvar in vproclistM: # add vrproclist back in later
            print(mvar)
            try:
                mmm=ev.timeSeriesComp(mvar,dsid,locName,shortName,lat,lon,
                        obsloadfun=obsloadNull,obsloadkwargs={},
                        modloadfun=modload,modloadkwargs={'f1':f1,'f2':f2,'lon':lon,'lat':lat,'freq':'daily'},
                        freq='daily',savepath=savebase,figsavepath=saveLoc+'figs/',compsavepath=saveLoc)
                if mvar in ['co2dryair','apco2']:
                    mmm.calc_fits(fitlist=['quadfit','optfit','linfit1'],fitlistOL=['linfit1'],
                                     defaultfit='quadfit',predefined={'obs_b2':'mod_b2'})
                else:
                    mmm.calc_fits()
                #mmm.calc_fits()
                mmm.calc_stats()
                mmm.topickle()
            except ObsError as err:
                print('ObsError:',err.value)
                print(f'Failed:({dsid},{mvar})')
            except ModError as err:
                print('ModError:',err.value)
                print(f'Failed:({dsid},{mvar})')
    return

def recalcStats(dsid,scen,freq,saveLoc,dfInfoBuoy=None):
    if dfInfoBuoy is None:
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
    locName,shortName,lat,lon=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title',
                                                            'shortTitle','Lat1x1','Lon1x1']].values[0]
    print(dsid,locName,saveLoc)
    for mvar in [*vproclist,]:#*vproclistM]:
        print(mvar)
        try:
            mmm=loadMoorComp(dsid,mvar,freq,saveloc)
            mmm.calc_stats()
            mmm.topickle()
        except FileNotFoundError as err:
            print('ObsError:',err.value)
            print(f'Pickle file not found:({dsid},{mvar})')
    return

def run1CompScen(dsid,scen,freq,saveLoc,dfInfoBuoy=None):
    if dfInfoBuoy is None:
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
    locName,shortName,lat,lon=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['title',
                                                            'shortTitle','Lat1x1','Lon1x1']].values[0]
    print(dsid,locName,saveLoc)
    if HDgrid:
        f1path=diagsPP.searchExtracted(scen,dsid,'_HDgrid')
        compPath=saveLoc+'compsHD/'
    else:
        f1path=diagsPP.searchExtracted(scen,dsid)
        compPath=saveLoc+'comps/'
    with nc.Dataset(f1path) as f1:#, nc.Dataset(diagsPP.searchExtracted(scen,dsid,'288grid')) as f2:
        for mvar in vproclistB:#'no3os',]:#[*vproclist,*vproclistM]:
            print(mvar)
            try:
                try:
                    [lat,lon], ldict, udict, df0, df = OAP.loadOAPBuoy(dfInfoBuoy,dsid,freq)
                    mmm=ev.timeSeriesComp(mvar,dsid,locName,shortName,lat,lon,
                            obsloadfun=obsload,obsloadkwargs={'df':df},
                            modloadfun=modload,modloadkwargs={'f1':f1,'f2':None,'lon':lon,'lat':lat,'freq':freq},
                            freq=freq,savepath=saveLoc,figsavepath=saveLoc+'figs/',compsavepath=compPath)
                except ObsError:
                    mmm=ev.timeSeriesComp(mvar,dsid,locName,shortName,lat,lon,
                            obsloadfun=obsloadNull,obsloadkwargs={},
                            modloadfun=modload,modloadkwargs={'f1':f1,'f2':None,'lon':lon,'lat':lat,'freq':freq},
                            freq=freq,savepath=saveLoc,figsavepath=saveLoc+'figs/',compsavepath=compPath)
                print(len(mmm.obs_val))
                if mvar in ['co2dryair','apco2']:
                    mmm.calc_fits(fitlist=['quadfit','linfit1'],fitlistOL=['linfit1'],
                                     defaultfit='quadfit',predefined={'obs_b2':'mod_b2'})
                else:
                    mmm.calc_fits(fitlist=['linfit1',],fitlistOL=['linfit1',],defaultfit='linfit1')
                #mmm.calc_fits()
                print(len(mmm.obs_val))
                mmm.calc_stats()
                print(len(mmm.obs_val))
                mmm.topickle()
            except ObsError as err:
                print('ObsError:',err.value)
                print(f'Failed:({dsid},{mvar})')
            except ModError as err:
                print('ModError:',err.value)
                print(f'Failed:({dsid},{mvar})')
    return

if __name__=="__main__":
    if sys.argv[1]=='run1CompScen':
        import diagsPP 
        ind=int(sys.argv[2])
        iscen=sys.argv[3]
        print(f"run1CompScen: ind={ind},scen={iscen}")
        yrspan = diagsPP.dictScenDates[iscen]
        freq='daily' 
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
        dsid=dfInfoBuoy.datasetID[ind]
        saveloc=savebase + f"{diagsPP.scenNameDict[iscen]}/"
        cf.mkdirs(saveloc)
        print(dsid,iscen,freq)
        #vproclist=['o2percos','AOUos']
        #vproclist=['phosC', 'omega_a_0', 'omega_c_0', 'dissicos']
        #vproclist=['dissicos','apco2']
        #vproclist=['co2dryair','apco2']
        vproclistB=['tos','sos']
        vproclistM=[]
        run1CompScen(dsid,iscen,freq,saveloc,dfInfoBuoy)
    elif sys.argv[1]=='run1CompScenM':
        import diagsPP 
        ind=int(sys.argv[2])
        iscen=sys.argv[3]
        print(f"run1CompScen: ind={ind},scen={iscen}")
        yrspan = diagsPP.dictScenDates[iscen]
        freq='monthly' 
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
        dsid=dfInfoBuoy.datasetID[ind]
        saveloc=savebase + f"{diagsPP.scenNameDict[iscen]}/"
        cf.mkdirs(saveloc)
        print(dsid,iscen,freq)
        #vproclist=['o2percos','AOUos']
        #vproclist=['phosC', 'omega_a_0', 'omega_c_0', 'dissicos']
        #vproclist=['dissicos','apco2']
        #vproclist=['co2dryair','apco2']
        vproclistB=['tos','sos','chlos','spco2','o2os','AOUos']
        vproclistM=[]
        run1CompScen(dsid,iscen,freq,saveloc,dfInfoBuoy)
    elif sys.argv[1]=='run1CompScenHD':
        import diagsPP 
        ind=int(sys.argv[2])
        scenInd=int(sys.argv[3])
        HDgrid=True
        print(f"run1CompScen: ind={ind},scenInd={scenInd}")
        iscen,yrspan = diagsPP.listScenDates[scenInd]
        freq='daily' 
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
        dsid=dfInfoBuoy.datasetID[ind]
        saveloc=savebase + f"{diagsPP.scenNameDict[iscen]}/"
        cf.mkdirs(saveloc)
        print(dsid,iscen,freq)
        vproclistB=['chlos','tos','sos']
        vproclistM=[]
        run1CompScen(dsid,iscen,freq,saveloc,dfInfoBuoy)

    elif sys.argv[1]=='recalcStats':
        import diagsPP 
        ind=int(sys.argv[2])
        scenInd=int(sys.argv[3])
        iscen,yrspan = diagsPP.listScenDates[scenInd]
        freq='daily' 
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
        dsid=dfInfoBuoy.datasetID[ind]
        saveloc=savebase + f"{diagsPP.scenNameDict[iscen]}/comps/"
        recalcStats(dsid,iscen,freq,saveloc,dfInfoBuoy)
        
    elif sys.argv[1]=='run1CompCluster':
        run1CompCluster(sys.argv[2],sys.argv[3])
        
    elif sys.argv[1]=='run1CompMonth':
        bc.run1CompMonth(sys.argv[2])
        
    else:
        raise NotImplementedError("input:",sys.argv)
