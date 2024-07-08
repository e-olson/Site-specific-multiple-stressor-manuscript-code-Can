import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import pandas as pd
import commonfxns as cf, OAPBuoyData as OAP, OAPBuoyComp as bc,viz, evalfxns as ev, frequencyBands as fb
import netCDF4 as nc
import cftime
import datetime as dt
import cmocean
import gsw
from collections import OrderedDict
import pickle
import os
import warnings

vlist0=['tos','sos','spco2','chlos','o2os',]
def loadDetrendAnomObs(icomp,deseas=False):
    # get data, detrend, and subtract mean
    otdt=icomp.obs_tdt
    mtdt=icomp.mod_tdt
    ott0=icomp.obs_tind.astype(float)
    # use pre-calculated deseas and detrend vals: 
    # obs_targetdeseas_b,obs_targetdeseas,obs_target, obs_target_b,mod_target,mod_target_deseas
    if deseas:
        oval0=icomp.obs_targetdeseas.astype(float)-np.mean(icomp.obs_targetdeseas)
        oval0_b=icomp.obs_targetdeseas_b.astype(float)-np.mean(icomp.obs_targetdeseas_b)
    else:
        oval0=icomp.obs_target.astype(float)-np.mean(icomp.obs_target)
        oval0_b=icomp.obs_target_b.astype(float)-np.mean(icomp.obs_target_b)

    ix=[np.argmin([np.abs((ii-el).total_seconds()/(24*3600)) for ii in icomp.mod_tdt]) for el in icomp.obs_tdt]
    
    ott=np.nan*np.ones(np.shape(icomp.mod_tdt))
    oval=np.nan*np.ones(np.shape(icomp.mod_tdt))
    oval_b=np.nan*np.ones(np.shape(icomp.mod_tdt))
    ott[ix]=icomp.obs_tind
    oval[ix]=oval0
    oval_b[ix]=oval0_b
    
    return otdt, ott, oval, oval_b # orig code called detrended quantities ovalds and ovalds_b

def loadDetrendAnomMod(icomp,deseas=False):
    # get data
    mtdt=icomp.mod_tdt
    mtt=icomp.mod_tind.astype(float)
    # use pre-calculated deseas and detrend vals: 
    # obs_targetdeseas_b,obs_targetdeseas,obs_target, obs_target_b,mod_target,mod_target_deseas
    if deseas:
        mval=icomp.mod_targetdeseas.astype(float)-np.mean(icomp.mod_targetdeseas)
    else:
        mval=icomp.mod_target.astype(float)-np.mean(icomp.mod_target)
    return mtdt, mtt, mval

def getBandStats(ivar,Tvec,icomp,bando3,bando3_b,bandm3,banddso3,banddso3_b,banddsm3):
    def _varcalc(bandts):
        vlist=[]
        for ii in range(0,len(Tvec)+1):
            crit=1 if ii==0 else Tvec[ii-1]
            vlist.append(np.nanvar(bandts[ii]) if np.sum(~np.isnan(bandts[ii]))>=crit else np.nan)
        return vlist
    def _ncalc(bandts):
        nlist=[]
        for ii in range(0,len(Tvec)+1):
            nlist.append(np.sum(~np.isnan(bandts[ii])))
        return nlist
    
    # seasonal cycle calcs
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')
        seasvarobs=np.nanvar(icomp.obs_gsmooth)
        seasvarmod=np.nanvar(icomp.mod_gsmooth)
        seasvarobs_b=np.nanvar(icomp.obs_gsmooth_b)
    NSeasobs=np.sum(~np.isnan(icomp.obs_gsmooth))
    NSeasmod=np.sum(~np.isnan(icomp.mod_gsmooth))
    NSeasobs_b=np.sum(~np.isnan(icomp.obs_gsmooth_b))
    # band calcs
    # add full variance to variance list for each group (append to end)
    vvar={'obs':_varcalc(bando3)+[np.nanvar(icomp.obs_target).astype(float) \
                                     if hasattr(icomp,'obs_target') else np.nan,],
          'mod':_varcalc(bandm3)+[np.nanvar(icomp.mod_target).astype(float),],
          'obs_ds':_varcalc(banddso3)+[np.nanvar(icomp.obs_targetdeseas.astype(float)) \
                                     if hasattr(icomp,'obs_targetdeseas') else np.nan,],
          'mod_ds':_varcalc(banddsm3)+[np.nanvar(icomp.mod_targetdeseas.astype(float)),],
          'obs_b':_varcalc(bando3_b)+[np.nanvar(icomp.obs_target_b.astype(float)) \
                                     if hasattr(icomp,'obs_target_b') else np.nan,],
          'obs_ds_b':_varcalc(banddso3_b)+[np.nanvar(icomp.obs_targetdeseas_b.astype(float)) \
                                     if hasattr(icomp,'obs_targetdeseas_b') else np.nan,]}
    Nvar={'obs':_ncalc(bando3)+[np.nansum(~np.isnan(icomp.obs_target.astype(float))) \
                                     if hasattr(icomp,'obs_target') else np.nan,],
          'mod':_ncalc(bandm3)+[np.nansum(~np.isnan(icomp.mod_target.astype(float))),],
          'obs_ds':_ncalc(banddso3)+[np.nansum(~np.isnan(icomp.obs_targetdeseas.astype(float))) \
                                     if hasattr(icomp,'obs_target') else np.nan,],
          'mod_ds':_ncalc(banddsm3)+[np.nansum(~np.isnan(icomp.mod_targetdeseas.astype(float))),],
          'obs_b':_ncalc(bando3_b)+[np.nansum(~np.isnan(icomp.obs_target_b.astype(float))) \
                                     if hasattr(icomp,'obs_target_b') else np.nan,],
          'obs_ds_b':_ncalc(banddso3_b)+[np.nansum(~np.isnan(icomp.obs_targetdeseas_b.astype(float))) \
                                     if hasattr(icomp,'obs_targetdeseas_b') else np.nan,]}
    # title=icomp.shortTitle+' '+ bc.dispNameUnits[ivar]
    # ofit=np.sum(~np.isnan(oval))>0
    
    return vvar, seasvarobs, seasvarmod, seasvarobs_b, Nvar, NSeasobs, NSeasmod, NSeasobs_b

def calcbands(Tvec=[13,],freq='monthly',cpath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/CanESM51_1975_2022/comps/',opt=None,vlist=None): #bc.compsaveloc):
    if vlist is None:
        vlist=vlist0
    equiv=True if opt=='equiv' else False
    if opt=='HDgrid' and not ('HD' in cpath):
        raise Error('Check cpath; should be path to half degree (HD) comps')
    bxfbase=cpath[:-6]+'bxfs/'
    dfInfoBuoy=OAP.loadOAPInfo()
    #dfInfoBuoy=OAP.loadOAPInfo(modMeans=True) check: was modMeans necessary?
    #comps=bc.loadAllComps()
    statsdf=bc.loadStats(merged=True,path=cpath)
    # create dataframe to accumulate results
    labels=OrderedDict()
    labelStr=OrderedDict()
    labels[0]=f"1_{Tvec[0]}"
    labelStr[0]=f"1 to {Tvec[0]} months"
    if len(Tvec)>1:
        for ii in range(1,len(Tvec)):
            labels[ii]=f"{Tvec[ii-1]}_{Tvec[ii]}"
            labelStr[ii]=f"{Tvec[ii-1]} to {Tvec[ii]} months"
    labels[len(Tvec)]=f"g{Tvec[-1]}"
    labelStr[len(Tvec)]=f"> {Tvec[-1]} months"
    labels[len(Tvec)+1]="total"
    labelStr[len(Tvec)+1]="Total"
    mcols=['ivar','datasetID',]
    ocols=['ivar','datasetID',]
    for ix in labels.keys():
        mcols.append('var_'+labels[ix]+'_mod')
        ocols.append('var_'+labels[ix]+'_obs')
    for ix in labels.keys():
        mcols.append('vards_'+labels[ix]+'_mod')
        ocols.append('vards_'+labels[ix]+'_obs')
    mcols.append('var_seas'+'_mod')
    ocols.append('var_seas'+'_obs')
    ocols.append('var_b_seas'+'_obs')
    for ix in labels.keys():
        ocols.append('var_b_'+labels[ix]+'_obs')
    for ix in labels.keys():
        ocols.append('vards_b_'+labels[ix]+'_obs')
    dfObsList=[]
    dfModList=[]
    dfModListN=[]
    dfObsListN=[]
    mcolsN=[el+"_N" if not el in ['ivar','datasetID'] else el for el in mcols ]
    ocolsN=[el+"_N" if not el in ['ivar','datasetID'] else el for el in ocols]
    
    # now run calculations and store outputs
    for ivar in vlist:
        print(ivar)
        dslist=statsdf.loc[(statsdf.modvar==ivar)&(statsdf.obs_N>1),
                           ['datasetID','shortTitle','obs_N']]['datasetID'].values
        for dsid in dslist:
            print(dsid)
            icomp=bc.loadMoorComp(dsid,ivar,freq,cpath)#comps[dsid,ivar,freq]
            mtdt, mtt, mvalds = loadDetrendAnomMod(icomp,deseas=True)
            otdt, ott, ovalds, ovalds_b = loadDetrendAnomObs(icomp,deseas=True)
            mtdt, mtt, mval = loadDetrendAnomMod(icomp,deseas=False)
            otdt, ott, oval, oval_b = loadDetrendAnomObs(icomp,deseas=False)
            if equiv:# resample model like obs
                mday=cf.dayindex(mtdt)
                oday=cf.dayindex(otdt)
                iii=[True if el in oday else False for el in mday ]
                #mtdt=mtdt[iii]; mtt=mtt[iii]; mvalds=mvalds[iii];mval=mval[iii]
                mtdt[~iii]=np.nan; mtt[~iii]=np.nan; mvalds[~iii]=np.nan;mval[~iii]=np.nan;

            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')
                bxfiltdsm,banddsm,variancedsm=fb.bandcalcs(Tvec,mtt,mvalds)
                bxfiltdso,banddso,variancedso=fb.bandcalcs(Tvec,ott,ovalds)
                bxfiltdso_b,banddso_b,variancedso_b=fb.bandcalcs(Tvec,ott,ovalds_b)

                bxfiltm,bandm,variancem=fb.bandcalcs(Tvec,mtt,mval)
                bxfilto,bando,varianceo=fb.bandcalcs(Tvec,ott,oval)
                bxfilto_b,bando_b,varianceo_b=fb.bandcalcs(Tvec,ott,oval_b)

            try:
                vvar, seasvarobs, seasvarmod, seasvarobs_b, Nvar, NSeasobs, NSeasmod, NSeasobs_b = getBandStats(ivar,Tvec,icomp,bando,bando_b,bandm,
                                                                                  banddso,banddso_b,banddsm)
            except:
                print(icomp.shortTitle,icomp.mod_target)
                raise

            # accumulate stats in summary dataframe:
            dfObsList.append([ivar,dsid,]+vvar['obs']+vvar['obs_ds']+[seasvarobs,]+[seasvarobs_b,]+vvar['obs_b']+vvar['obs_ds_b'])
            dfModList.append([ivar,dsid,]+vvar['mod']+vvar['mod_ds']+[seasvarmod,])

            dfObsListN.append([ivar,dsid,]+Nvar['obs']+Nvar['obs_ds']+[NSeasobs,]+[NSeasobs_b,]+Nvar['obs_b']+Nvar['obs_ds_b'])
            dfModListN.append([ivar,dsid,]+Nvar['mod']+Nvar['mod_ds']+[NSeasmod,])

            # save bands and stats by (variable,site)
            bxf=dict()
            savelist=['Tvec','otdt','ott','mtdt','mtt','vvar',
                      'seasvarobs','seasvarmod','Nvar','NSeasobs','NSeasmod',]
            for el in savelist:
                bxf[el]=eval(el)
            bxf['oval']={'full':oval,'deseas':ovalds}
            bxf['oval_b']={'full':oval_b,'deseas':ovalds_b}
            bxf['mval']={'full':mval,'deseas':mvalds}
            bxf['bxfiltm']={'full':bxfiltm,'deseas':bxfiltdsm}
            bxf['bandm']={'full':bandm,'deseas':banddsm}
            bxf['variancem']={'full':variancem,'deseas':variancedsm}
            bxf['bxfilto']={'full':bxfilto,'deseas':bxfiltdso}
            bxf['bando']={'full':bando,'deseas':banddso}
            bxf['varianceo']={'full':varianceo,'deseas':variancedso}
            bxf['bxfilto_b']={'full':bxfilto_b,'deseas':bxfiltdso_b}
            bxf['bando_b']={'full':bando_b,'deseas':banddso_b}
            bxf['varianceo_b']={'full':varianceo_b,'deseas':variancedso_b}
            ppath=fb.bxfpath(dsid,ivar,Tvec,freq,bxfbase,opt=opt)
            print(ppath)
            cf.mkdirs(ppath)
            with open(ppath, 'wb') as hh:
                pickle.dump(bxf, hh, protocol=pickle.HIGHEST_PROTOCOL)

    dfObs=pd.DataFrame(data=dfObsList,columns=ocols)
    dfMod=pd.DataFrame(data=dfModList,columns=mcols)
    dfa=pd.merge(left=dfObs,right=dfMod,on=['ivar','datasetID'])
    dfNObs=pd.DataFrame(data=dfObsListN,columns=ocolsN)
    dfNMod=pd.DataFrame(data=dfModListN,columns=mcolsN)
    dfN=pd.merge(left=dfNObs,right=dfNMod,on=['ivar','datasetID'])
    df1=pd.merge(left=dfa,right=dfN,on=['ivar','datasetID'])
    df=pd.merge(left=df1,right=dfInfoBuoy,on=['datasetID'])
    isequiv='.equiv' if equiv else ''
    df.to_csv(bxfbase+f"bxf_df.{'_'.join([str(el) for el in Tvec])}.{freq}{fb.optstr[opt]}.csv")
    return

if __name__=="__main__":
    print('starting calcs')
    # first rerun compile stats:
    bc.compileStats(save=True,path='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/CanESM51_1975_2022_monthly/comps/',vlist=vlist0,freq='monthly')
    calcbands(cpath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/CanESM51_1975_2022_monthly/comps/')

    # # equiv option:
    # calcbands(opt='equiv');#cpath='/work/ebo/calcs/buoyCompTS/GFDL-ESM4.1.1975_2022/comps/');

    # # HDgrid option (T and S):
    # vlistHD=['tos','sos','chlos']
    # bc.compileStats(save=True,path='/work/Elise.Olson/calcs/buoyCompTS/GFDL-ESM4.1.1975_2022/compsHD/',vlist=vlistHD)
    # calcbands(cpath='/work/Elise.Olson/calcs/buoyCompTS/GFDL-ESM4.1.1975_2022/compsHD/',
    #           opt='HDgrid',vlist=vlistHD)
