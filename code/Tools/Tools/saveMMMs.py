from Tools import extremes as ex
import datetime as dt

yrspan0=[2061,2100]
for iscen0 in set(ex.scenDefaults.keys())-set(['ESM4_historical_D1']):
    yrspan,dtarget=ex.scenDefaults[iscen0]
    print(iscen0,yrspan,dtarget)
    mod_tdt, mod_tind, mod_val,newvals,lf, seasF, MMM = \
                    ex.calc_MMM_NOAA(iscen0,yrspan,dtarget)
    del mod_tdt, mod_tind, mod_val,newvals,lf, seasF, MMM
print(f'done {dt.datetime.now()}')
