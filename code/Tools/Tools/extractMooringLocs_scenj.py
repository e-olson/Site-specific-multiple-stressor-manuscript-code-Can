# make work on any scenario with integer as input to select scenario from list
import sys
import os
import numpy as np
import netCDF4 as nc
from Tools import commonfxns as cf, panfxns as pf, OAPBuoyData as OAP, evalfxns as ev, diagsPP, viz
import cftime
import pandas as pd


# vardict should contain variable names as keys and subdirectory as values
# can be 2d or 3d variables
vardirdict={# 2d vars
          'tos':'ocean_daily_1x1deg', #base
          'sos':'ocean_daily_1x1deg', #base
          'spco2':'ocean_cobalt_daily_sfc_1x1deg',
          #'co2s':'aerosol_daily_cmip',
          'dpco2':'ocean_cobalt_daily_sfc_1x1deg',
          #'MLD_003':'ocean_daily_1x1deg',
          'chlos':'ocean_cobalt_omip_daily_1x1deg', #base
          #'intpp':'ocean_cobalt_daily_2d_1x1deg',
          #288'huss':'atmos_cmip', #base
          #'fgco2':'ocean_cobalt_daily_sfc_1x1deg',
          #'talkos':'ocean_cobalt_daily_sfc_1x1deg',
          #'dissicos':'ocean_cobalt_daily_sfc_1x1deg',
          #'dic_kw':'ocean_cobalt_daily_sfc_1x1deg',
          #'dic_sc_no':'ocean_cobalt_daily_sfc_1x1deg',
          #'pso':'ocean_daily_1x1deg',
          #'friver':'ocean_daily_1x1deg',
          #'pr':'atmos_cmip', #base
          #'uas':'atmos_cmip', #base
          #'vas':'atmos_cmip', #base
          #'nsmp_100':'ocean_cobalt_daily_2d_1x1deg',
          #'nlgp_100':'ocean_cobalt_daily_2d_1x1deg',
          #'ndi_100':'ocean_cobalt_daily_2d_1x1deg',
          #'mesozoo_200':'ocean_cobalt_daily_2d_1x1deg',
          'phos':'ocean_cobalt_daily_sfc_1x1deg',
          'o2os':'ocean_cobalt_daily_sfc_1x1deg',
          'o2satos':'ocean_cobalt_daily_sfc_1x1deg',
          #'no3os':'ocean_cobalt_daily_sfc_1x1deg',
          # 3d vars
          #'ph':'ocean_cobalt_daily_car_z_1x1deg',
          #'o2':'ocean_cobalt_daily_bio_z_1x1deg',
          #'o2sat':'ocean_cobalt_daily_bio_z_1x1deg',
          #'chl':'ocean_cobalt_daily_bio_z_1x1deg',
          #'thetao':'ocean_daily_z_1x1deg',
          #'so':'ocean_daily_z_1x1deg',
          #'talk':'ocean_cobalt_daily_car_z_1x1deg',
          #'dissic':'ocean_cobalt_daily_car_z_1x1deg',
          'co3':'ocean_cobalt_daily_sat_z_1x1deg',
          'co3satarag':'ocean_cobalt_daily_sat_z_1x1deg',
          #'co3satcalc':'ocean_cobalt_daily_sat_z_1x1deg',
          #'no3':'ocean_cobalt_daily_bio_z_1x1deg'
          }

vardirdict180x288={}#'co2s': 'aerosol_daily_cmip',
    #'huss': 'atmos_cmip', 
    #'pr': 'atmos_cmip', 
    #'uas': 'atmos_cmip', 
    #'vas': 'atmos_cmip'}

vardirdictHD={'tos': 'ocean_daily_cmip',
              'sos': 'ocean_daily_cmip',
              'chlos': 'ocean_cobalt_omip_daily',}

# def destoutPath(dirpath,iscen,mooringID,jj,ii,qualifier=''):
#     return dirpath+f"{iscen}.{mooringID}.j{jj}i{ii}{qualifier}.nc"

def checkFiles(filelist):
    for ifile in filelist:
        if not os.path.isfile(ifile):
            return False # a file is missing
    return True # no files are missing

def buoyExtract(files,mooringID,jj,ii,iscen,ivar,maxproc=6,qualifier=''):
    # mooringID is datasetID from OAP.loadOAPInfo()
    # jj is lat index
    # ii is lon index
    # iscen is scenario
    # ivar is variable name
    # isubdir is subdirectory to find variable in
    # first check if target file already exists and contains variable; in that case skip
    
    # specify first and last+1 date values
    startdate=(cftime.datetime(yrspan[0],1,1,calendar='noleap')-diagsPP.zerodates[iscen]).days # first day in model time units
    startstr=f"{float(startdate):.1f}"
    enddate=(cftime.datetime(yrspan[-1]+1,1,1,calendar='noleap')-diagsPP.zerodates[iscen]).days # first day of next year in model time units
    endstr=f"{float(enddate):.1f}"
    
    scenName=diagsPP.scenNameDict[iscen]
    # if iscen==('ESM4_historical_D1','ESM4_ssp245_D1_histCont'):
    #     scenName='ESM4_historical_D1' # exception to new naming convention to avoid rewriting existing code and avoid storing duplicate data
    cf.mkdirs(saveloctemp)
    destout=diagsPP.extractedTSPath(iscen,mooringID,jj,ii,qualifier,savelocFinal) #destoutPath(savelocFinal,iscen,mooringID,jj,ii,qualifier)
    cf.mkdirs(destout)
    if os.path.isfile(destout):
        with nc.Dataset(destout) as ff:
            infiles=ff.variables.keys()
        if ivar in infiles:
            #print(f"{ivar} already present in {destout}")
            return [] # don't continue
    filesonwork=pf.cpfromTape(files)
    print(ivar,files[0])
    cmdlist=list()
    tempflist=list()
    if qualifier=='_HDgrid':
        hdims=['yh','xh'] # account for different horizontal dimension names
    else: 
        hdims=['lat','lon']
    for ifile in filesonwork:
        daterange=ifile.split('.')[-3]
        tempfile=saveloctemp+f'temp.{scenName}.{mooringID}.{ivar}.{daterange}.j{jj}i{ii}.nc'
        tempflist.append(tempfile)
        cmd=f'ncks -d time,{startstr},{endstr} -d {hdims[0]},{jj} -d {hdims[1]},{ii} {ifile} {tempfile}'
        cmdlist.append(cmd)
    # for each variable extract location to file:
    prepnco= lambda x:cf.prepmod(x,'nco')
    print(cmdlist)
    cf.subprocrun(cmdlist,maxproc=maxproc,prepfun=prepnco,verb=False)
    assert checkFiles(tempflist) # there seem to be cases where files weren't created: debug

    # concatenate temp files
    tempout=saveloctemp+f'temp.{scenName}.{mooringID}.{ivar}.j{jj}i{ii}.nc'
    cmd=f"ncrcat {' '.join(tempflist)} {tempout}"
    print(cmd)
    cf.subprocrun(cmd,maxproc=1,prepfun=prepnco)

    # if destout exists, append variable to it; otherwise, copy file to it
    if os.path.isfile(destout):
        cmd=f"ncks -A {tempout} {destout}"
        print(cmd)
        cf.subprocrun(cmd,maxproc=1,prepfun=prepnco)
    else:
        cmd=f"cp {tempout} {destout}"
        print(cmd)
        cf.subprocrun(cmd,maxproc=1)

    # clear temp dir
    for ifile in os.listdir(saveloctemp):
        if os.path.isfile(saveloctemp+ifile):
            os.remove(saveloctemp+ifile)
            
    return filesonwork

def main():

    # outer loop through variables, vardirdict, ii, jj
    for ivar in list(vardirdict.keys()):
        print(ivar)
        files=diagsPP.genflist(ivar, freq, iscen, yrspan, subdir=vardirdict[ivar])
        clfiles = not os.path.isfile('/work/Elise.Olson'+files[0])
        
        # loop through moorings
        for ind,[mooringID,jj,ii] in dfb.loc[:,['datasetID','jj','ii']].iterrows():
            filesonwork=buoyExtract(files,mooringID,jj,ii,iscen,ivar,maxproc=8,qualifier='_1x1grid')
    
        # clear variable from /work/ebo if first file in list was not already present (above)
        if clfiles:
            pf.clearFromWork(filesonwork,verb=True)

    # outer loop through variables, vardirdict180x288, ii288, jj288
    for ivar in list(vardirdict180x288.keys()):
        print(ivar)
        files=diagsPP.genflist(ivar, freq, iscen, yrspan, subdir=vardirdict180x288[ivar])
        clfiles = not os.path.isfile('/work/Elise.Olson'+files[0])
        
        # loop through moorings
        for ind,[mooringID,jj,ii] in dfb.loc[:,['datasetID','jj288','ii288']].iterrows():
            filesonwork=buoyExtract(files,mooringID,jj,ii,iscen,ivar,maxproc=8,qualifier='_288grid')
    
        # clear variable from /work/ebo if first file in list was not already present (above)
        if clfiles:
            pf.clearFromWork(filesonwork,verb=True)

    # outer loop through variables, vardirdict180x288, ii288, jj288
    for ivar in list(vardirdictHD.keys()):
        print(ivar)
        files=diagsPP.genflist(ivar, freq, iscen, yrspan, subdir=vardirdictHD[ivar])
        clfiles = not os.path.isfile('/work/Elise.Olson'+files[0])

        # loop through moorings
        for ind,[mooringID,jj,ii] in dfb.loc[:,['datasetID','jjHD','iiHD']].iterrows():
            filesonwork=buoyExtract(files,mooringID,jj,ii,iscen,ivar,maxproc=8,qualifier='_HDgrid')

        # clear variable from /work/ebo if first file in list was not already present (above)
        if clfiles:
            pf.clearFromWork(filesonwork,verb=True)

    return


if __name__=="__main__":
    option=sys.argv[1] # first argument should be type of extraction: scenario, cluster, (obs?)
    
    # scenj case:
    if option=='scenario':
        j=int(sys.argv[2])
        iscen,yrspan = diagsPP.listScenDates[j]
        scenName=diagsPP.scenNameDict[iscen]
        savelocFinal='/work/ebo/calcs/extracted_OAPBuoy/' # final results loc
        saveloctemp=f'/work/ebo/calcs/scratch/{scenName}/' # intermediate files loc (cleared frequently)
        # dfreq={365:'daily',12:'monthly',1:'yearly'} # represent frequency as records per year in input
        freq='daily' #dfreq[int(sys.argv[2])]
        dfb=OAP.loadOAPInfo(modelgrid=True)
        main()
    
    # cluster case:
    elif option=='cluster':
        sloc=sys.argv[2] #'BTM' or WHOTS or Papa
        savelocFinal=f'/work/ebo/calcs/extracted_OAPBuoy/Clusters/{sloc}/' # final results loc
        saveloctemp='/work/ebo/calcs/scratch/Clusters/' # intermediate files loc (cleared frequently)
        iscen,yrspan=diagsPP.listScenDates[11]# presentDay
        freq='daily' #dfreq[int(sys.argv[2])]
        dfInfoBuoy=OAP.loadOAPInfo(modelgrid=True)
        dsid=OAP.getID(sloc)
        shortTitle=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['shortTitle']].values[0][0]
        jj0,ii0=dfInfoBuoy.loc[dfInfoBuoy.datasetID==dsid,['jj','ii']].values[0]
    
        llist=ev.ClusterList(jj0,ii0)
        newdflist=[]
        for ix, (jj,ii) in enumerate(llist):
            idsid=dsid+f'_{ix:02d}'
            ilat=viz.glat1x1[jj,ii]
            ilon=viz.glon1x1[jj,ii]
            jj288,ii288=cf.nearest_point(ilat,ilon,
                                   viz.glat288,viz.glon288,thresh=100,tol=2)
            newdflist.append([idsid,shortTitle,jj,ii,jj288,ii288,ilat,ilon,dsid])
        dfb=pd.DataFrame(newdflist,columns=['datasetID','shortTitle','jj','ii','jj288','ii288','Lat1x1','Lon1x1','base_datasetID'])
        main() 
    
