import xarray as xr
import numpy as np
import os
import shutil
import pandas as pd
import OAPBuoyData as OAP, commonfxns as cf, diagsPP, viz
import cftime

cmip6data='/space/hall6/sitestore/eccc/crd/ccrn/model_output/CMIP6/final/CMIP6/'
h5ss='/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/'
canesm5assim='/space/hall5/sitestore/eccc/crd/ccrn/model_output/CMIP6/final/CMIP6/DCPP/CCCma/CanESM5/dcppA-assim/'

tlistHist= [ 'tos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19710101-19801231.nc',
             'tos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19810101-19901231.nc',
             'tos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19910101-20001231.nc',
             'tos_Oday_CanESM5-1_historical_r1i1p2f1_gn_20010101-20101231.nc',
             'tos_Oday_CanESM5-1_historical_r1i1p2f1_gn_20110101-20141231.nc',
           ]

tlist245=['tos_Oday_CanESM5-1_ssp245_r1i1p2f1_gn_20150101-20201231.nc',
          'tos_Oday_CanESM5-1_ssp245_r1i1p2f1_gn_20210101-20301231.nc',
         ]

slistHist= [ 'sos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19710101-19801231.nc',
             'sos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19810101-19901231.nc',
             'sos_Oday_CanESM5-1_historical_r1i1p2f1_gn_19910101-20001231.nc',
             'sos_Oday_CanESM5-1_historical_r1i1p2f1_gn_20010101-20101231.nc',
             'sos_Oday_CanESM5-1_historical_r1i1p2f1_gn_20110101-20141231.nc',
           ]

slist245=['sos_Oday_CanESM5-1_ssp245_r1i1p2f1_gn_20150101-20201231.nc',
          'sos_Oday_CanESM5-1_ssp245_r1i1p2f1_gn_20210101-20301231.nc',
         ]

flist={'tos': [os.path.join(h5ss,'work',el) for el in tlistHist]+[os.path.join(h5ss,'work',el) for el in tlist245],
       'sos': [os.path.join(h5ss,'work',el) for el in slistHist]+[os.path.join(h5ss,'work',el) for el in slist245],
      }

dfb=OAP.loadOAPInfo(modelgrid=True)

flongT=xr.open_mfdataset(flist['tos'],parallel=True).sel(time=slice(cftime.DatetimeNoLeap(1975,1,1),cftime.DatetimeNoLeap(2023,1,1)))
flongS=xr.open_mfdataset(flist['sos'],parallel=True).sel(time=slice(cftime.DatetimeNoLeap(1975,1,1),cftime.DatetimeNoLeap(2023,1,1)))

scen='CanESM51_1975_2022'

indlist=[]
for ind, row in dfb.iterrows():
    if row.modBathy>0:
        fnameout=diagsPP.extractedTSPath(scen,row.datasetID,row.jj,row.ii)
        print(fnameout)
        indlist.append(ind)
        flongT.isel(j=row.jj,i=row.ii)
        flongS.isel(j=row.jj,i=row.ii)
        dsout=xr.merge([flongT,flongS])
        print('writing')
        dsout.to_netcdf(fnameout,'w')
        print('done')

print('end')
flongT.close()
flongS.close()
