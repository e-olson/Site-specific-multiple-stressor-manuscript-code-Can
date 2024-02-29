"""
Carbonate chemistry calculations
Wrappers on PyCO2SYS
"""

# import numpy as np
# import gsw
import PyCO2SYS as pyco2

pyco2Args1={'opt_pH_scale':1, #(default) total pH scale
           'opt_k_carbonic':10, # carbonic acid dissociation LDK00: Lueker, T. J., Dickson, A. G., and Keeling, C. D. (2000). 
           'opt_k_bisulfate':1, #(default) bisulfate ion dissociation D90a: Dickson, A. G. (1990).
           'opt_total_borate':1, #(default) boron:salinity relationship U74: Uppström, L. R. (1974).
           'opt_k_fluoride':1, #(default) hydrogen fluoride dissociation DR79: Dickson, A. G., and Riley, J. P. (1979).
          } # all others defaults

# list typically saved values present in output dict:
savelist=['pH_total','pCO2','saturation_calcite','saturation_aragonite',
          'HCO3','CO3','CO2','Hfree','alkalinity','dic']

# total silicate, total phosphate, total ammonia, total hydrogen sulfide in umol/kg
def pyco2_surf_ph_from_TA_DIC(TA,DIC,SSS,SST,press=None,total_silicate=None,total_phosphate=None,
                              total_ammonia=None,total_sulfide=None,temperature_out=None,args=None):
    # 1: Alk in umol/kg; 2: DIC in umol/kg; SSS salinity (psu); SST Temperature (deg C), press pressure (dbar), nuts in umol/kg
    if args is None:
        aargs=pyco2Args1.copy()
    else:
        aargs=args.copy() 
    if press is not None:
        aargs['pressure'] = press
    if total_silicate is not None:
        aargs['total_silicate'] = total_silicate
    if total_phosphate is not None:
        aargs['total_phosphate'] = total_phosphate
    if total_ammonia is not None:
        aargs['total_ammonia'] = total_ammonia
    if total_sulfide is not None:
        aargs['total_sulfide'] = total_sulfide
    results=pyco2.sys(par1=TA,par2=DIC,par1_type=1,par2_type=2,salinity=SSS,temperature=SST,
                        opt_buffers_mode=0,**aargs)
    if temperature_out is not None:
        aargs['temperature_out'] = temperature_out
    return results['pH_total']

def pyco2_surf_all_from_TA_DIC(TA,DIC,SSS,SST,press=None,total_silicate=None,total_phosphate=None,
                               total_ammonia=None,total_sulfide=None,temperature_out=None,args=None):
    # 1: Alk in umol/kg; 2: DIC in umol/kg; SSS salinity (psu); SST Temperature (deg C) 
    if args is None:
        aargs=pyco2Args1.copy()
    else:
        aargs=args.copy()  
    if total_silicate is not None:
        aargs['total_silicate'] = total_silicate
    if total_phosphate is not None:
        aargs['total_phosphate'] = total_phosphate
    if total_ammonia is not None:
        aargs['total_ammonia'] = total_ammonia
    if total_sulfide is not None:
        aargs['total_sulfide'] = total_sulfide
    if temperature_out is not None:
        aargs['temperature_out'] = temperature_out
    resultsdict=pyco2.sys(par1=TA,par2=DIC,par1_type=1,par2_type=2,salinity=SSS,temperature=SST,
                        opt_buffers_mode=0,**aargs)
    return resultsdict
 
def pyco2_surf_all_from_TA_pCO2(TA,pCO2,SSS,SST,press=None,total_silicate=None,total_phosphate=None,
                                total_ammonia=None,total_sulfide=None,temperature_out=None,args=None):
    # 1: Alk in umol/kg; 4: pco2 in uatm
    if args is None:
        aargs=pyco2Args1.copy()
    else:
        aargs=args.copy() 
    if total_silicate is not None:
        aargs['total_silicate'] = total_silicate
    if total_phosphate is not None:
        aargs['total_phosphate'] = total_phosphate
    if total_ammonia is not None:
        aargs['total_ammonia'] = total_ammonia
    if total_sulfide is not None:
        aargs['total_sulfide'] = total_sulfide
    if temperature_out is not None:
        aargs['temperature_out'] = temperature_out
    results=pyco2.sys(par1=TA,par2=pCO2,par1_type=1,par2_type=4,salinity=SSS,temperature=SST,
                        opt_buffers_mode=0,**aargs)
    return results

def pyco2_surf_all_from_pH_pCO2(pH,pCO2,SSS,SST,press=None,total_silicate=None,total_phosphate=None,
                                total_ammonia=None,total_sulfide=None,temperature_out=None,args=None):
    # 1: Alk in umol/kg; 4: pco2 in uatm
    if args is None:
        aargs=pyco2Args1.copy()
    else:
        aargs=args.copy() 
    if total_silicate is not None:
        aargs['total_silicate'] = total_silicate
    if total_phosphate is not None:
        aargs['total_phosphate'] = total_phosphate
    if total_ammonia is not None:
        aargs['total_ammonia'] = total_ammonia
    if total_sulfide is not None:
        aargs['total_sulfide'] = total_sulfide
    if temperature_out is not None:
        aargs['temperature_out'] = temperature_out
    results=pyco2.sys(par1=pH,par2=pCO2,par1_type=3,par2_type=4,salinity=SSS,temperature=SST,
                        opt_buffers_mode=0,**aargs)
    return results
