import numpy as np
import xarray as xr
import os
import glob
import cmocean
import cftime
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely import geometry as geo
import matplotlib.path as mpath
import PyCO2SYS as pyco2

"""
Reproduce and document Alkalinity(T,S) relationships used by Sutton et al.

from Adrienne Sutton: Here are the TA-S relationships we use:

Washington State coast and Puget Sound (Fassbender et al. 2016a) California Current Ecosystem (Cullison Gray et al. 2011) South Atlantic Bight (Xue et al. 2006) Gulf of Maine (Hunt, University of New Hampshire, personal communication: TA = SSS * 52.5 + 476) Kuroshio Extension (Fassbender et al. 2017) Northeast Pacific (Fassbender et al. 2016b) all others (Lee et al. 2006)

further communication: Hi Adrienne, Does your Northeast Pacific (Fassbender et al. 2016b) region include the SEAK, GAKOA, and Kodiak stations as well as Papa? Also, do you include Cheeca Rocks in the South Atlantic Bight (Xue et al. 2006) region in addition to Gray's Reef? Thanks again! Elise Good question. No, the Fassbender pub only used data at station Papa. And for the coral reef buoys, there are no good TA-S relationships given the reefs modify TA, so you'll have to stick to using pCO2 and measured pH.

Cullison Gray, S.E., M.D. DeGrandpre, T.S. Moore, T.R. Martz, G.E. Friederich, and K.S. Johnson. 2011. Applications of in situ pH measurements for inorganic carbon calculations. Marine Chemistry 125(1-4):82-90, 10.1016/j.marchem.2011.02.005.

Fassbender, A.J., S.R. Alin, R.A. Feely, A.J. Sutton, J.A. Newton, and R.H. Byrne. 2016. Estimating Total Alkalinity in the Washington State Coastal Zone: Complexities and Surprising Utility for Ocean Acidification Research. Estuaries and Coasts:1-15, 10.1007/s12237-016-0168-z.

    Washington State coast and Puget Sound
    uncertainty 1 sigma=+/-17 umol/kg

Fassbender, A.J., C.L. Sabine, and M.F. Cronin. 2016. Net community production and calcification from seven years of NOAA Station Papa Mooring measurements. Global Biogeochemical Cycles:n/a-n/a, 10.1002/2015GB005205.

Fassbender, A.J., C.L. Sabine, M.F. Cronin, and A.J. Sutton. 2017. Mixed-layer carbon cycling at the Kuroshio Extension Observatory. Global Biogeochemical Cycles:n/a-n/a, 10.1002/2016GB005547.

Lee, K., L.T. Tong, F.J. Millero, C.L. Sabine, A.G. Dickson, C. Goyet, G.-H. Park, R. Wanninkhof, R.A. Feely, and R.M. Key. 2006. Global relationships of total alkalinity with salinity and temperature in surface waters of the world's oceans. Geophys. Res. Lett. 33, 10.1029/2006gl027207.

Xue, L., W.-J. Cai, X. Hu, C. Sabine, S. Jones, A.J. Sutton, L.-Q. Jiang, and J.J. Reimer. 2016. Sea surface carbon dioxide at the Georgia time series site (2006–2007): Air–sea flux and controlling processes. Progress in Oceanography 140:14-26, http://dx.doi.org/10.1016/j.pocean.2015.09.008.
"""

ReefList=['Cheeca Rocks','Chuuk', 'Hog Reef', 'Crescent Reef', 'La Parguera', 
                 'Ala Wai', 'Kilo Nalu','Kaneohe', 'CRIMP2','CRIMP1'] #, 'Fagatele Bay', 'Heron Island'
"""
exclude coral reef moorings: ('shortTitle')
Cheeca Rocks (24.90°N, 80.62°W) 'Cheeca Rocks'
Chuuk (7.46°N,151.90°E) 'Chuuk'
Fagatele Bay (14.36°S, 170.76°W) not included
Hog Reef (32.46°N, 64.83°W) 'Hog Reef'
Crescent Reef (32.40°N, 64.79°W) 'Crescent Reef'
Heron Island (22.46°S, 151.93°E) not included
La Parguera (17.95°N, 67.05°W) 'La Parguera'
Ala Wai (21.28°N, 157.85°W) 'Ala Wai'
Kilo Nalu (21.29°N, 157.87°W) 'Kilo Nalu'
Kaneohe (21.48°N, 157.78°W) 'Kaneohe'
CRIMP2 (21.46°N, 157.80°W) 'CRIMP2'
CRIMP (21.43°N, 157.79°W) 'CRIMP1'
"""

class AlkFit:
    def __init__(self,TAFun,SSS_range=None,SST_range=None,uncertainty=None,citStr=None):
        #TA function should take inputs (SSS,SST,LON) for uniformity
        self.TAFun=TAFun
        self.SSS_range=SSS_range if SSS_range is not None else [-999,999]
        self.SST_range=SST_range if SST_range is not None else [-999,999]
        self.uncertainty=uncertainty if uncertainty is not None else np.nan
        self.citStr=citStr if citStr is not None else 'missing'
    def checkbounds(self,SSS,SST):
        validS=(SSS>=self.SSS_range[0])&(SSS<self.SSS_range[1])
        validT=(SST>=self.SST_range[0])&(SST<self.SST_range[1])
        return validS, validT
    def estimate(self,SSS,SST=None,LON=None,valid=True):
        if SST is None:
            SST=0.0*np.ones(np.shape(SSS)) # 0 so that SST will pass bounds check
        if LON is None:
            LON=np.nan*np.ones(np.shape(SSS)) # Lon not checked, doesn't matter
        TA=self.TAFun(SSS,SST,LON)
        if valid:
            validS,validT=self.checkbounds(SSS,SST)
        return np.where(validS&validT,TA,np.nan) if valid else TA


def LeeLoc(LON,LAT):
    # define primary Lee et al region for Alk-T-S fit
    pts=(geo.Point(np.array([LON,LAT])),
         geo.Point(np.array([LON+360,LAT])),
         geo.Point(np.array([LON-360,LAT]))) # do only once per point not every time inPoly called
    # subfunction to check if Lon, Lat inside polygon with small buffer to include bounding lines
    def inPoly(poly):
        # poly is geo.Polygon
        # covers the anticipated lon ranges by adding and subtracting 360 from each lon
        # returns True if pos in poly else False
        return poly.buffer(1e-10).contains(pts[0])|\
            poly.buffer(1e-10).contains(pts[1])|\
            poly.buffer(1e-10).contains(pts[2])
        
    # define some regions as shapely polygons:
    poly_eqPac=geo.Polygon([[-140,10],[-110,20],[-100,20],[-86.5,13],[-81.6,8.3],[-79,9.3],[-69.5,2],
                    [-69.5,-20],[-110,-20],[-140,-10]])
    poly_GMex=geo.Polygon([[-100,31],[-100,20],[-93,17],[-87,21.2],[-81,25.1],[-82,31],])
    poly_NPac=geo.Polygon([[115,30],[125,60],[175,66],[220,66],[240,50],[245,30]])
    poly_NAtl=geo.Polygon([[-82,30],[-70,70],[-90,80],[20,80,],[20,80],[15,60],[-11,30]])

    # Spatial criteria
    # Equatorial Pacific
    if inPoly(poly_eqPac):
        region='Equatorial Pacific'#2
    # (Sub)tropics: 
    elif ((LAT>-30) & (LAT<30)):# | inPoly(poly_GMex): 
        region='(Sub)tropics'#1
    # North Atlantic
    elif inPoly(poly_NAtl)| inPoly(poly_GMex): 
        region='North Atlantic'#3
    # North Pacific
    elif inPoly(poly_NPac): 
        region='North Pacific'#4
    #Southern Ocean
    elif (LAT<=-30)&(LAT>=-70):
        region='Southern Ocean'#5
    else:
        region=np.nan
    return region
LLocNum={'Equatorial Pacific':2,'(Sub)tropics':1,'North Atlantic':3,
         'North Pacific':4,'Southern Ocean':5,np.nan:np.nan}

def LeeLoc2nd(LON,LAT,baseRegion):
    # assign secondary Lee et al region (neighboring region, to be considered if T out of bounds)
    pts=(geo.Point(np.array([LON,LAT])),
        geo.Point(np.array([LON+360,LAT])),
        geo.Point(np.array([LON-360,LAT]))) # do only once per point not every time inPoly called
    def inPoly(poly):
        # poly is geo.Polygon
        # covers the anticipated lon ranges by adding and subtracting 360 from each lon
        # returns True if pos in poly else False
        return poly.buffer(1e-10).contains(pts[0])|\
            poly.buffer(1e-10).contains(pts[1])|\
            poly.buffer(1e-10).contains(pts[2])
    poly_eqPacCatch=geo.Polygon([[-150,15],[-110,25],[-100,20],[-86.5,13],[-81.6,8.3],[-79,9.3],[-69.5,2],
                [-69.5,-25],[-110,-25],[-150,-15]])
    if baseRegion in ('North Atlantic','North Pacific','Southern Ocean'):
        reg2='(Sub)tropics'
    elif baseRegion=='Equatorial Pacific':
        reg2=np.nan # if T is less than minimum for Eq Pac, adjoining region won't match either since minimum (Sub)tropics T is higher
    elif baseRegion=='(Sub)tropics':
        if inPoly(poly_eqPacCatch):
            reg2='Equatorial Pacific'
        elif LAT<-10:
            reg2='Southern Ocean'
        elif (LAT>10):
            if ((LON<0)&(LON>-100)|(LON<0-360)&(LON>-100-360)|(LON<0+360)&(LON>-100+360))&~inPoly(poly_eqPacCatch):
                reg2='North Atlantic'
            else:
                reg2='North Pacific'
        else:
            reg2=np.nan
    else:
        reg2=np.nan
    return reg2

# define Lee et al fits for each region and store in dictionary
regdict={1:'(Sub)tropics',2:'Equatorial Upwelling Pacific',3:'North Atlantic',4:'North Pacific',5:'Southern Ocean'}
def _lonW(lon):
    lon =-1*(lon-360)
    if lon>360:
        lon=lon-360
    elif lon<0:
        lon=lon+360
    return lon
LeeFits={'(Sub)tropics':AlkFit(lambda SSS,SST,LON: 2305 + 58.66*(SSS - 35) + 2.32*(SSS - 35)**2 - 1.41*(SST - 20) + 0.040*(SST - 20)**2,
                    SST_range=[20,999],
                    SSS_range=[31,38],
                    uncertainty=8.6,citStr='Lee'),
        'Equatorial Pacific':AlkFit(lambda SSS,SST,LON: 2294 + 64.88*(SSS - 35) + 0.39*(SSS - 35)**2 \
                                              - 4.52*(SST - 29) - 0.232*(SST - 29)**2,
                    SST_range=[18,999],
                    SSS_range=[31,36.5],
                    uncertainty=7.5,citStr='Lee'),
        'North Atlantic':AlkFit(lambda SSS,SST,LON: 2305 + 53.97*(SSS - 35) + 2.74*(SSS - 35)**2 - 1.16*(SST - 20) - 0.040*(SST - 20)**2,
                    SST_range=[0,20],
                    SSS_range=[31,37],
                    uncertainty=6.4,citStr='Lee'),
        'North Pacific':AlkFit(lambda SSS,SST,LON: 2305 + 53.23*(SSS - 35) + 1.85*(SSS - 35)**2 - 14.72*(SST - 20) - 0.158*(SST - 20)**2\
                                            + 0.062*(SST - 20)*_lonW(LON), # Alk is supposed to be higher in western Pac, so LON must be degrees W. Constrain to (0,360)
                    SST_range=[-999,20],
                    SSS_range=[31,35],
                    uncertainty=8.7,citStr='Lee'),
        'Southern Ocean':AlkFit(lambda SSS,SST,LON: 2305 + 52.48*(SSS - 35) + 2.85*(SSS - 35)**2 - 0.49*(SST - 20) + 0.086*(SST - 20)**2,
                    SST_range=[-999,20],
                    SSS_range=[33,36],
                    uncertainty=8.4,citStr='Lee'),
        np.nan:np.nan} # handle undefined case


# define fits for other sources, and create a dictionary mapping short names of stations to these fits
AlkFuns={}
# Washington State coast and Puget Sound
Fass2016a=AlkFit(lambda SSS,SST,LON: 47.4*SSS+647,
                 SSS_range=[20,35],uncertainty=17,citStr='Fass2016a')
for st in ['Cha ba','Cape Elizabeth']:
    AlkFuns[st]=Fass2016a
# Gulf of Maine
Hunt=AlkFit(lambda SSS,SST,LON: 52.5*SSS+476,citStr='Hunt')
AlkFuns['Gulf of Maine']=Hunt
# California Current Ecosystem
# NOTE: maybe ask Adrienne Sutton for advice re: CCE1 pH? one sensor section looks high
CullisonGray=AlkFit(lambda SSS,SST,LON:2131+50.8*(SSS-31.25),
                   uncertainty=20,citStr='CullisonGray')
for st in ['CCE1','CCE2','Cape Arago','NH-10']:
    AlkFuns[st]=CullisonGray
# South Atlantic Bight (Gray's Reef)
# NOTE: here a few measurements fall below the reported salinity range. Most were within
#       the valid range for the nearly identical Lee et al fit, and agreement with sensor
#       pH remained strong, so the data were not excluded.
Xue=AlkFit(lambda SSS,SST,LON: 49.66*SSS+573.63,
                 SSS_range=[31.7,36.7],uncertainty=13,citStr='Xue')
AlkFuns["Gray's Reef"]=Xue
# KEO site, south of Kuroshio extension (probably don't use for JKEO, which is north of extension
# Note: Lee et al fit was very similar, contrary to what was reported in Fassbender et al (2017); maybe longitude was applied in 
# deg E instead of West as I initially did, which causes the wrong sign in the longitude-dependent term; not clear in Lee et al
# S range not reported, but I read it off the figure as [34.4,35.2]; however I did not use this to exclude data outside the 
# range; salinity seasonally dipped below 34.4, but agreement with sensor pH remained strong.
Fass2017=AlkFit(lambda SSS,SST,LON: 65.6*SSS+1.9,citStr='Fass2017')
AlkFuns["KEO"]=Fass2017
#Station Papa
Fass2016b=AlkFit(lambda SSS,SST,LON: 37*SSS+988,
                 SSS_range=[32,33.5],uncertainty=3,citStr='Fass2016b')
AlkFuns["Papa"]=Fass2016b

#Not excluded: Iceland: all T and S within valid range
#              BTM: Seasonal dips only slightly below valid temperature range
#              Coastal MS: significant dips below valid salinity range, but good agreement between sensor and estimate; temperature
#                                dips into secondary range were short-lived and the change in Alk estimate betwen the primary and
#                                 secondary fits was very small.
#              Coastal LA: less pH sensor data available, but otherwise very similar to Coastal MS
#              Stratus: mostly within range
#              BOBOA: dip below sal range still agrees with sensor pH estimate
#              Kodiak: only a few points below salinity range, not associated with pH extrema
#              M2: within range
#              SOFS: within range
#              JKEO: crosses border between temperature regions, but difference between fits on TA is negligible
#              MOSEAN/WHOTS: within range, and mostly good agreement with sensor
#              all TAO array locations: within range
ExcludeStations=['First Landing', # below Lee et al valid salinity range for all points; poor agreement between sensor and estimated pH
                 'SEAK', # large sections below salinity range wtih no pH sensor data to compare ***?
                 'GAKOA', # below Lee salinity range for almost all points
                ]

def OAP_Alk(stitle,SSS,SST,LON,LAT,valid=False):
    if stitle in ReefList+ExcludeStations:
        return np.nan*np.ones(np.shape(SSS))
    elif stitle in AlkFuns.keys():
        afit=AlkFuns[stitle]
    else:
        region=LeeLoc(LON,LAT)
        afit=LeeFits[region]
    return afit.estimate(SSS,SST,LON,valid=valid)


def OAP_Alk_cit(stitle):
    if stitle in ReefList+ExcludeStations:
        return 'None'
    elif stitle in AlkFuns.keys():
        afit=AlkFuns[stitle]
        return afit.citStr
    else:
        return 'Lee'
