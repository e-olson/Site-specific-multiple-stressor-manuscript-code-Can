import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
#import socket

# # for working on stellar:
# if 'stellar' in socket.gethostname():
#     gprefix='/scratch/cimes/eo2651/grid/'
# else:
#     gprefix='/work/Elise.Olson/grid/'
with nc.Dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/deptho_Ofx_CanESM5-1_historical_r1i1p2f1_gn.nc') as fstat:
    glon1x1=fstat.variables['longitude'][:]
    glat1x1=fstat.variables['latitude'][:]
    deptho1x1=fstat.variables['deptho'][:]
    wet1x1=fstat.variables['deptho'][:].mask==False
with nc.Dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/reo000/work/areacello_Ofx_CanESM5-1_historical_r1i1p2f1_gn.nc') as fstat:
    areacello1x1=fstat.variables['areacello'][:]

# fstatic1x1=gprefix+'archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/ocean_daily_1x1deg/ocean_daily_1x1deg.static.nc'
# with nc.Dataset(fstatic1x1) as fstat:
#     glon1x1=fstat.variables['geolon'][:]
#     glat1x1=fstat.variables['geolat'][:]
#     deptho1x1=fstat.variables['deptho'][:]
#     wet1x1=fstat.variables['wet'][:]
#     areacello1x1=fstat.variables['areacello'][:]

# f3d1x1=gprefix+'archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/ocean_annual_z_1x1deg/ts/annual/5yr/ocean_annual_z_1x1deg.2010-2014.thetao.nc'
# with nc.Dataset(f3d1x1) as fstat:
#     z_l=fstat.variables['z_l'][:]
#     z_i=fstat.variables['z_i'][:]
#     lonvec1x1=fstat.variables['lon'][:]
#     latvec1x1=fstat.variables['lat'][:]

# fstaticHD=gprefix+'archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/ocean_daily_cmip/ocean_daily_cmip.static.nc'
# with nc.Dataset(fstaticHD) as fstat:
#     glonHD=fstat.variables['geolon'][:]
#     glatHD=fstat.variables['geolat'][:]
#     depthoHD=fstat.variables['deptho'][:]
#     wetHD=fstat.variables['wet'][:]
#     areacelloHD=fstat.variables['areacello'][:]

# fstatic180x288=gprefix+'archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/atmos_cmip/atmos_cmip.static.nc'
# with nc.Dataset(fstatic180x288) as fstat:
#     lonvec288 = fstat.variables['lon'][:]
#     latvec288 = fstat.variables['lat'][:]
#     glon288,glat288=np.meshgrid(lonvec288,latvec288)
#     landmask288=fstat.variables['land_mask'][:]
#     area288=fstat.variables['area'][:]

# def k_from_z(z):
#     return np.sum(z_i<=z)-1

# def z_from_k(k,astype='num'):
#     if astype=='num':
#         return z_l[k]
#     elif astype=='str':
#         return f'z={z_l[k]} m'
#     else:
#         raise ValueError("astype must be 'num' or 'str'")

def taylorView(cors,stdrats):
    r=stdrats
    theta=(1-cors)*np.pi/2
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r,'r*')
    ax.set_rmax(2)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ticklocs=np.pi*np.array([0,1/8,1/4,3/8,1/2]);
    ticklabels=1-ticklocs/np.pi*2
    ax.set_xticks(ticklocs);
    ax.set_xticklabels(ticklabels);
    #ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    #ax.grid(True)
    
    #ax.set_title("A line plot on a polar axis", va='bottom')
    #plt.show()
    
    ax.set_xlabel('std model/std obs',labelpad=20)
    ax.annotate("correlation", 
                xy=[.4,.65], xycoords="figure fraction",rotation=np.rad2deg(-np.pi/4),
                ha="left", va="bottom",fontsize=16)
    return fig, ax

def squareax(ax,line=True,origin=False,linespecs=None):
    if linespecs is None:
        linespecs={'linestyle':'-','color':'lightgray'}
    xl=ax.get_xlim()
    yl=ax.get_ylim()
    al=[min(0 if origin else xl[0],0 if origin else yl[0]),max(xl[1],yl[1])]
    if line:
        ax.plot(al,al,**linespecs)
    ax.set_xlim(al)
    ax.set_ylim(al)
    ax.set_aspect(1)
    return 

def gfdlLon(lon):
    """ Function to convert arbitrary longitude into the range used by gfdl model grid,
    for display purposes. Input can be a single longitude or array/list/dataarray of values. """
    if hasattr(lon,"__len__"): # assume array-like
        lontype=type(lon)
        newlon=[gfdlLon(el) for el in lon]
        if lontype==np.ndarray:
            return np.array(newlon)
        elif lontype==pd.core.series.Series:
            return pd.core.series.Series(newlon,index=lon.index)
        else:
            return newlon
    else:
        if lon<0:
            return lon+360
        elif lon>360:
            return lon-360
        else:
            return lon
