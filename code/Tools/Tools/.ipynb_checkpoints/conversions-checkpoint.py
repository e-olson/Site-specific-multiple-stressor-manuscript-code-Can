import numpy as np
import gsw

def oxyperc_surf_from_o2umolkg(o2umolkg,T,S):
    o2sat=gsw.O2sol_SP_pt(S,T) # S is psu, T is pt, umol/kg
    return o2umolkg/o2sat*100

    ## fist calculate o2sat:
    #sal = min(42.0,max(0.0,S)))
    #tt = 298.15 - min(40.0,max(0.0,T)))
    #tkb = 273.15 + min(40.0,max(0.0,T)))
    #ts = np.log(tt / tkb)
    #ts2 = ts  * ts
    #ts3 = ts2 * ts
    #ts4 = ts3 * ts
    #ts5 = ts4 * ts
    #
    ##solubilities in units of mol/m3/atm
    #o2sat = (1000.0/22391.6) * np.exp( cobalt%a_0 + cobalt%a_1*ts + cobalt%a_2*ts2 + cobalt%a_3*ts3 + cobalt%a_4*ts4 + cobalt%a_5*ts5 + &
    #     (cobalt%b_0 + cobalt%b_1*ts + cobalt%b_2*ts2 + cobalt%b_3*ts3 + cobalt%c_0*sal)*sal)


