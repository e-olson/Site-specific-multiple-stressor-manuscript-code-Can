import glob
import os
import pandas as pd
import numpy as np
import socket
import cftime

# dictionaries containing information about diag and postprocessing tables for runs and function for checking presence of files
# also includes code to load files from various directories

# based on fre/xml/xanadu_esm4_20190304/ESM4/compileDiags.xml
diags_orig={'ocean_daily':["pso","MLD_003","friver", ],
'ocean_month':["deta_dt","MLD_003_min","MLD_003_max"],
'ocean_daily_z':["thetao", "so"],
'ocean_month_z':["thetao_min","so_min","rhopot0_min","thetao_max","so_max","rhopot0_max","agessc","age_200", ],
'ocean_scalar_month':["masso","volo","ssh_ga","precip_ga",],
'ocean_annual_z':["agessc", ],
'ocean_scalar_annual':["masso","volo", "soga","sosga","ssh_ga","precip_ga",],
 
'ice_month':["EXT_min","EXT_max","siconc_min","siconc_max"],
 
'ocean_cobalt_daily_sfc':[ "talkos","dissicos","spco2","dpco2","fgco2","dic_kw","dic_sc_no", "runoff_flux_alk",
                          "runoff_flux_dic", "runoff_flux_no3", "runoff_flux_po4",],
'ocean_cobalt_daily_2d':["intpp","intppnitrate","mesozoo_200", "sfc_irr",
                         "fcadet_calc_100","fcadet_arag_100","nsmp_100","nlgp_100","ndi_100",],
'ocean_cobalt_month_2d':["spco2_max","dpco2_max","fgco2_max","nsmp_100_max","nlgp_100_max","ndi_100_max",
                         "intpp_max","intppnitrate_max","jprod_mesozoo_200_max","epc100_max","fndet_btm_max",
                         "spco2_min","dpco2_min","fgco2_min","nsmp_100_min","nlgp_100_min","ndi_100_min",
                         "intpp_min","intppnitrate_min","jprod_mesozoo_200_min","epc100_min","fndet_btm_min",],
'ocean_cobalt_daily_car_z':["dissic","talk","ph",],
'ocean_cobalt_daily_sat_z':["co3","co3satcalc","co3satarag",],
'ocean_cobalt_daily_bio_z':["o2","o2sat","no3","chl",],
'ocean_cobalt_month_z':["dissic_max","phyc_max","zooc_max","bacc_max","talk_max","ph_max","o2_max",
                        "o2sat_max","no3_max","nh4_max","po4_max","dfe_max","si_max","chl_max","co3_max",
                        "co3satcalc_max","co3satarag_max","pp_max","irr_inst_max",
                        "dissic_min","phyc_min","zooc_min","bacc_min","talk_min","ph_min","o2_min",
                        "o2sat_min","no3_min","nh4_min","po4_min","dfe_min","si_min","chl_min","co3_min",
                        "co3satcalc_min","co3satarag_min","pp_min",
                        "irr_inst",],
 
'atmos_month_cmip':["sfcWind_min","sfcWind_max","pr_min","pr_max","evspsbl_min","evspsbl_max",],
'atmos_daily_cmip':["evspsbl",],
'aerosol_daily_cmip':["co2s",],
'aerosol_month_cmip':["co2s","co2s_min","co2s_max"],
'atmos_scalar':['rrvco2']
}

# based on fre/xml/xanadu_esm4_20190304/ESM4/compileDiagsV2.xml
diags_v2={'ocean_daily':["pso","MLD_003","friver", ],
'ocean_month':["deta_dt","MLD_003_min","MLD_003_max","tos_min","tos_max","sos_min","sos_max",],
'ocean_daily_z':["thetao", "so"],
'ocean_month_z':["thetao_min","so_min","rhopot0_min","thetao_max","so_max","rhopot0_max","agessc","age_200", ],
'ocean_scalar_month':["masso","volo","ssh_ga","precip_ga",],
'ocean_annual_z':["agessc", ],
'ocean_scalar_annual':["masso","volo", "soga","sosga","ssh_ga","precip_ga",],
 
'ice_month':["EXT_min","EXT_max","siconc_min","siconc_max"],
 
'ocean_cobalt_daily_sfc':[ "talkos","dissicos","spco2","dpco2","fgco2","dic_kw","dic_sc_no", "runoff_flux_alk",
                          "runoff_flux_dic", "runoff_flux_no3", "runoff_flux_po4",
                         "phos","o2os","o2satos","no3os",],
'ocean_cobalt_daily_2d':["intpp","intppnitrate","mesozoo_200", "sfc_irr",
                         "fcadet_calc_100","fcadet_arag_100","nsmp_100","nlgp_100","ndi_100",],
'ocean_cobalt_month_2d':["spco2_max","dpco2_max","fgco2_max","nsmp_100_max","nlgp_100_max","ndi_100_max",
                         "intpp_max","intppnitrate_max","jprod_mesozoo_200_max","epc100_max","fndet_btm_max",
                         "spco2_min","dpco2_min","fgco2_min","nsmp_100_min","nlgp_100_min","ndi_100_min",
                         "intpp_min","intppnitrate_min","jprod_mesozoo_200_min","epc100_min","fndet_btm_min",
                         "phos_max","o2os_max","chlos_max","no3os_max","phycos_max",
                         "phos_min","o2os_min","chlos_min","no3os_min","phycos_min",],
'ocean_cobalt_daily_car_z':["dissic","talk","ph",],
'ocean_cobalt_daily_sat_z':["co3","co3satcalc","co3satarag",],
'ocean_cobalt_daily_bio_z':["o2","o2sat","no3","chl",],
'ocean_cobalt_month_z':["dissic_max","phyc_max","zooc_max","bacc_max","talk_max","ph_max","o2_max",
                        "o2sat_max","no3_max","nh4_max","po4_max","dfe_max","si_max","chl_max","co3_max",
                        "co3satcalc_max","co3satarag_max","pp_max","irr_inst_max",
                        "dissic_min","phyc_min","zooc_min","bacc_min","talk_min","ph_min","o2_min",
                        "o2sat_min","no3_min","nh4_min","po4_min","dfe_min","si_min","chl_min","co3_min",
                        "co3satcalc_min","co3satarag_min","pp_min",
                        "irr_inst",],
 
'atmos_month_cmip':["sfcWind_min","sfcWind_max","pr_min","pr_max","evspsbl_min","evspsbl_max",],
'atmos_daily_cmip':["evspsbl",],
'aerosol_daily_cmip':["co2s",],
'aerosol_month_cmip':["co2s","co2s_min","co2s_max"],
'atmos_scalar':['rrvco2']
}

# original postprocessing from fre/xml/xanadu_esm4_20190304/ESM4/compilePP.xml
# (source, destination, [variables])
pp_orig=[('ocean_daily', 'ocean_daily_1x1deg',    []),
    ('ocean_month',      'ocean_monthly',         ['deta_dt',]),
    ('ocean_month',      'ocean_monthly_1x1deg',  ['MLD_003_min', 'MLD_003_max']),
    ('ocean_daily_z',    'ocean_daily_z',         ['thetao', 'so']),
    ('ocean_daily_z',    'ocean_daily_z_1x1deg',  []),
    ('ocean_month_z',    'ocean_monthly_z',       ['thetao_min', 'so_min', 'thetao_max', 'so_max']),
    ('ocean_month_z',    'ocean_monthly_z_1x1deg',[]),
    ('ocean_annual_z',   'ocean_annual_z_1x1deg', []),
    ('ocean_scalar_month',  'ocean_scalar_monthly',[]),
    ('ocean_scalar_annual', 'ocean_scalar_annual', []),
    ('ice_month',           'ice_1x1deg',  []),
    ('atmos_month_cmip',    'atmos_cmip',  []),
    ('atmos_daily_cmip',    'atmos_cmip',  []),
    ('atmos_scalar',        'atmos_scalar',[]),
    ('aerosol_month_cmip',  'aerosol_month_cmip', []),
    ('aerosol_daily_cmip',  'aerosol_daily_cmip', []),
    ('ocean_cobalt_daily_sfc',  'ocean_cobalt_daily_sfc_1x1deg',  []),
    ('ocean_cobalt_daily_2d',   'ocean_cobalt_daily_2d_1x1deg',   []),
    ('ocean_cobalt_month_2d',   'ocean_cobalt_monthly_2d_1x1deg', []),
    ('ocean_cobalt_daily_car_z','ocean_cobalt_daily_car_z',       ['ph',]),
    ('ocean_cobalt_daily_car_z','ocean_cobalt_daily_car_z_1x1deg',[]),
    ('ocean_cobalt_daily_sat_z','ocean_cobalt_daily_sat_z_1x1deg',[]),
    ('ocean_cobalt_daily_bio_z','ocean_cobalt_daily_bio_z_1x1deg',[]),
    ('ocean_cobalt_month_z',    'ocean_cobalt_monthly_z',         ['ph_max', 'ph_min']),
    ('ocean_cobalt_month_z',    'ocean_cobalt_monthly_z_1x1deg',  [])]
# not used:
#atmos_global_cmip,monthly
#atmos_tracer,tracer_level
#atmos_co2_month,atmos_co2_level
#ocean_cobalt_omip_sfc,ocean_cobalt_omip_sfc_1x1deg

# V2 postprocessing from fre/xml/xanadu_esm4_20190304/ESM4/compilePPV2.xml
# (source, destination, [variables])
pp_v2=[('ocean_daily', 'ocean_daily_1x1deg',[]),
    ('ocean_month', 'ocean_monthly',        ['deta_dt',]),
    ('ocean_month', 'ocean_monthly_1x1deg', ['MLD_003_min', 'MLD_003_max','tos_min','tos_max','sos_min','sos_max']),
    ('ocean_daily_z',   'ocean_daily_z',        ['thetao', 'so']),
    ('ocean_daily_z',   'ocean_daily_z_1x1deg', []),
    ('ocean_month_z',    'ocean_monthly_z',       ['thetao_min', 'so_min', 'thetao_max', 'so_max']),
    ('ocean_month_z',    'ocean_monthly_z_1x1deg',[]),
    ('ocean_annual_z',   'ocean_annual_z_1x1deg', []),
    ('ocean_scalar_month',  'ocean_scalar_monthly',[]),
    ('ocean_scalar_annual', 'ocean_scalar_annual', []),
    ('ice_month',           'ice_1x1deg',  []),
    ('atmos_month_cmip',    'atmos_cmip',  []),
    ('atmos_daily_cmip',    'atmos_cmip',  []),
    ('atmos_scalar',        'atmos_scalar',[]),
    ('aerosol_month_cmip',  'aerosol_month_cmip', []),
    ('aerosol_daily_cmip',  'aerosol_daily_cmip', []),
    ('ocean_cobalt_daily_sfc',  'ocean_cobalt_daily_sfc_1x1deg',  []),
    ('ocean_cobalt_daily_2d',   'ocean_cobalt_daily_2d_1x1deg',   []),
    ('ocean_cobalt_month_2d',   'ocean_cobalt_monthly_2d_1x1deg', []),
    ('ocean_cobalt_daily_car_z','ocean_cobalt_daily_car_z',       ['ph',]),
    ('ocean_cobalt_daily_car_z','ocean_cobalt_daily_car_z_1x1deg',[]),
    ('ocean_cobalt_daily_sat_z','ocean_cobalt_daily_sat_z_1x1deg',[]),
    ('ocean_cobalt_daily_bio_z','ocean_cobalt_daily_bio_z_1x1deg',[]),
    ('ocean_cobalt_month_z',    'ocean_cobalt_monthly_z',         ['ph_max', 'ph_min']),
    ('ocean_cobalt_month_z',    'ocean_cobalt_monthly_z_1x1deg',  [])]

yrspan_future = [2061,2100]
listScenDates = [('ESM4_historical_D1',[1975,2014]), #0
            ('ESM4_ssp119_D1',yrspan_future), #1
            ('ESM4_ssp126_D1',yrspan_future), #2
            ('ESM4_ssp245_D1',yrspan_future), #3
            ('ESM4_ssp245_D151',yrspan_future), #4
            ('ESM4_ssp245_D201',yrspan_future), #5
            ('ESM4_ssp370_D1',yrspan_future), #6
            ('ESM4_ssp534-over_D1',yrspan_future), #7
            ('ESM4_ssp585_D1',yrspan_future), #8
            ('ESM4_piControl_D_0226_0265',[226,265]), #9
            ('ESM4_piControl_D_0312_0351',[312,351]), #10
            (('ESM4_historical_D1','ESM4_ssp245_D1_histCont'),[1975,2014,2015,2022]), #11
            ('ESM4_1pctCO2_D1_start',[1,40]), #12
            (('ESM4_1pctCO2_D1_peak','ESM4_1pctCO2-cdr_D1_peak'),[121,140,141,160]), #13
            ('ESM4_1pctCO2-cdr_D1_end',[241,280]), #14
                 ('CanESM51_1975_2022',[1975,2022]),
                 ('CanESM51_1975_2022_monthly',[1975,2022]),
                 ('CanESM5CanOE_1975_2022_monthly',[1975,2022])]
dictScenDates=dict(listScenDates)
dictScenDates['GFDL-ESM4.1.1975_2022']=dictScenDates[('ESM4_historical_D1','ESM4_ssp245_D1_histCont')] # handle alias

scenNameDict={'CanESM51_1975_2022':'CanESM51_1975_2022',
              'CanESM51_1975_2022_monthly':'CanESM51_1975_2022_monthly',
              'CanESM5CanOE_1975_2022_monthly':'CanESM5CanOE_1975_2022_monthly',
              ('ESM4_historical_D1','ESM4_ssp245_D1_histCont'):'1975_2022',
              ('ESM4_1pctCO2_D1_peak','ESM4_1pctCO2-cdr_D1_peak'):'1pctCO2peak',
               'GFDL-ESM4.1.1975_2022':'GFDL-ESM4.1.1975_2022',
                '1pctCO2peak':'1pctCO2peak'}#,
               #'ESM4_historical_D1':'ESM4_historical_D1_1974_2014'} # disambiguation
for el in dictScenDates.keys():
    if not el in scenNameDict.keys():
        if type(el)==tuple:
            raise Error(f'Add {el} manually to scenNameDict - need special names for compound keys')
        scenNameDict[el]=el

zerodates={ 'ESM4_historical_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp119_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp126_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp245_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp245_D151':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp245_D201':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp370_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp534-over_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_ssp585_D1':cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_piControl_D_0226_0265':cftime.datetime(1,1,1,calendar='noleap'),
            'ESM4_piControl_D_0312_0351':cftime.datetime(1,1,1,calendar='noleap'),
            ('ESM4_historical_D1', 'ESM4_ssp245_D1_histCont'):cftime.datetime(1850,1,1,calendar='noleap'),
            'ESM4_1pctCO2_D1_start':cftime.datetime(1,1,1,calendar='noleap'),
            ('ESM4_1pctCO2_D1_peak', 'ESM4_1pctCO2-cdr_D1_peak'):cftime.datetime(1,1,1,calendar='noleap'),
            'ESM4_1pctCO2-cdr_D1_end':cftime.datetime(1,1,1,calendar='noleap'),}

# derived file names
def extractedTSPath(iscen,dsid,jj,ii,qualifier='_1x1grid',basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/extracted/'):
    scenName=scenNameDict[iscen]
    return basepath+f"{scenName}/{scenName}.{dsid}.j{jj}i{ii}{qualifier}.nc"

def searchExtracted(iscen,dsid,qualifier='_1x1grid',basepath='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/extracted/'):
    srchstr=extractedTSPath(iscen,dsid,'*','*',qualifier,basepath)
    fpath=glob.glob(srchstr)
    if not len(fpath)==1:
        raise ValueError(f'fpath has wrong size: searchstring={srchstr}; fpath={fpath}')
    return fpath[0]

def checkPPFiles(startdate, duration, pp_path, diags, pp):
    datestrlist=[f"{('000'+str(yr))[-4:]}*-{('000'+str(yr+4))[-4:]}*" for yr in range(startdate,startdate+duration,5)]
    errdict=dict()
    for (source,dest,varb) in pp:
        if len(varb)==0:
            varb=diags[source]
        for ivar in varb:
            for idatestr in datestrlist:
                pth=f'{pp_path}/{dest}/ts/*/5yr/{dest}.{idatestr}.{ivar}.nc'
                test=glob.glob(pth)
                if not len(test)==1:
                    if dest+'/'+ivar in errdict.keys():
                        errdict[dest+'/'+ivar].append(idatestr)
                    else:
                        errdict[dest+'/'+ivar]=[idatestr,]
    return errdict

# rdf definitions:
spincol='spinup output path'
basecol='base run output path'
runcol='run output path'
scencol='Scenario'
if 'stellar' in socket.gethostname():
    runcsvLoc='/home/eo2651/OAPMSE/analysis/Tools/Tools/Runs.csv'
else:
    runcsvLoc='/home/Elise.Olson/OAPMSE/analysis/Tools/Tools/RunsV2.csv'

def listFiles(runid,segment="slice",ftype='pp',yrlims=[],ppvar=None,subdir=None,freq=None):
    # segment options: slice, spinup, base (original)
    # ftype: pp, history, restart
    # yrlims should be integer years; can be single year
    # check input:
    if not (segment in ('slice','spinup','base')):
        raise ValueError(f'segement undefined: {segment},\n Choose from: (slice, spinup, base)')
    if not (ftype in ('pp','history','restart')):
        raise ValueError(f'ftype undefined:{ftype},\n Choose from: (pp, history, restart)')
    if type(yrlims)==int:
        yrlims=[yrlims]
    if len(yrlims)>0 and not (type(yrlims[0])==int and type(yrlims[-1]==int)):
        raise ValueError(f'incorrect input for yrlims: {yrlims}')
    rdf=pd.read_csv(runcsvLoc)
    xcolmap={'slice':runcol,'spinup':spincol,'base':basecol}
    if not (runid in set(rdf[scencol].values)):
        raise ValueError(f'runid undefined:{runid},\n Choose from:{rdf[scencol].values}')
    path0=rdf.loc[rdf[scencol]==runid,[xcolmap[segment]]].values[0][0]
    path1=os.path.join(path0,ftype)
    if ftype=='pp': # handle complicated pp case
        if ppvar is None: raise ValueError(f'ftype is pp and ppvar is not defined')
        flist=listFilesPP(path1,ppvar,subdir,freq,yrlims) 
    else: # history or restart
        flist=np.sort(os.listdir(path1))
        if len(yrlims)>0:
            yrs=[int(el[:4]) for el in flist]
            flist=flist[yrs>=yrlims[0]&yrs<=yrlims[-1]]
    return flist

def genflist(ivar, freq, iscen, yrspan, subdir,work=False):
    # return a list of files; 
    # allow iscen to be a list, in which cast yrlims is a list with 2x 
    #       the length specifying first and last for each segment
   
    # 'GFDL-ESM4.1.1975_2022' is alias for ('ESM4_historical_D1','ESM4_ssp245_D1_histCont') run:
    iscen = ('ESM4_historical_D1','ESM4_ssp245_D1_histCont') if iscen=='GFDL-ESM4.1.1975_2022' else iscen
    if type(iscen)==list or type(iscen)==tuple:
        files=list()
        for ind, iiscen in enumerate(iscen):
            iyrspan=yrspan[2*ind:2*(ind+1)]
            ifiles=genflist(ivar,freq,iiscen,iyrspan,subdir)
            for el in ifiles:
                files.append(el)
        files=np.asarray(files)
    else:
        try: # check my run first
            files=listFiles(iscen,ftype='pp',segment='slice',ppvar=ivar,
                                    freq=freq,subdir=subdir,yrlims=yrspan)
        except: # if not go to archive
            files=listFiles(iscen,ftype='pp',segment='base',ppvar=ivar,
                                    freq=freq,subdir=subdir,yrlims=yrspan)
    if work==True:
        files=['/work/ebo'+el for el in files]
    return files

def listFilesPP(basepath,ppvar,subdir=None,freq=None,yrlims=[]):
    # to load hist and ssp585 files on Stellar:
    if 'stellar' in socket.gethostname():
        basepath='/scratch/gpfs/eo2651/modelOutput'+basepath
        if not ('historical' in basepath or '585' in basepath):
            raise ValueError('this run was not transferred to stellar')
    dirsep='*' if subdir is None else subdir
    freqsep='*' if freq is None else freq
    list0=np.array(glob.glob(f'{basepath}/{dirsep}/ts/{freqsep}/*/*.{ppvar}.nc'))
    if len(list0)==0:
        print(f'{basepath}/{dirsep}/ts/{freqsep}/*/*.{ppvar}.nc')
        raise ValueError(f"The input specifications did not match any files: \n basepath={basepath},\n ppvar={ppvar},\n subdir={subdir},\n freq={freq},\n yrlims={yrlims}")
    list1=np.array([el[len(basepath)+1:].split('/') for el in list0])
    ind1=np.ones(np.shape(list1[:,0]))==1
    ind2=np.ones(np.shape(list1[:,0]))==1
    print('listFilesPP, varname:',ppvar)
    print(f"basepath: {basepath}")
    if subdir is None:
        setdir=list(set(list1[:,0])); 
        if len(setdir)==1:
            subdir=setdir[0]
            print(f"subdirectory set to only available option: {subdir}")
        else:
            print(f"subdirectory options: {setdir}")
    else:
        print(f"subdirectory: {subdir}")
    if freq is None:
        setfreq=list(set(list1[:,2])); 
        if len(setfreq)==1:
            freq=setfreq[0]
            print(f"frequency set to only available option: {freq}")
        else:
            print(f"frequency options: {setfreq}")
    else:
        print(f"frequency: {freq}")
    if subdir is None:
        subdir=input('Enter subdirectory choice:')
        ind1=list1[:,0]==subdir
    if freq is None:
        freq=input('Enter frequency choice:')
        ind2=list1[:,2]==subdir
    flist=list0[ind1&ind2]
    setinterv=list(set(list1[ind1&ind2,3]))
    assert len(setinterv)==1, f"Expected only one element in list setinterv. Current values: {list1[ind1&ind2]}"     
    if len(yrlims)>0:
        fnames=list1[ind1&ind2,4]
        datestrs=[el.split('.')[1].split('-') for el in fnames]
        yrs=np.array([[int(x0[:4]),int(x1[:4])] for [x0,x1] in datestrs])
        flist=flist[(yrs[:,1]>=yrlims[0])&(yrs[:,0]<=yrlims[-1])]
    flist=np.sort(flist)
    return flist

def trfreq(freq):
    # clean up likely mistakes
    if freq in ['Monthly','monthly','month','Month']:
        return 'monthly'
    elif freq in ['Daily','daily','Day','day']:
        return 'daily'
    elif freq in ['Yearly','yearly','Year','year','annual','Annual']:
        return 'yearly'
