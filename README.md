# Site-specific-multiple-stressor-manuscript-code
Code accompanying the manuscript, "Site-specific multiple stressor assessments based on high frequency surface observations and an Earth system model", by Elise M. B. Olson, Jasmin G. John, John Dunne, Charles Stock, Elizabeth J. Drenkard, and Adrienne J. Sutton

This archive contains the code used for the analysis described in "Site-specific multiple stressor assessments based on high frequency surface obserations and an Earth system model" by Olson, E. M., John, J. G., Dunne, J., Stock, C., Drenkard, E. J., and Sutton, A., for submission to Earth and Space Science. The locations of daily model output time series included in this dataset match the CO2 mooring time series of Sutton et al. (2019).

GENERAL INFORMATION:
Primary point of contact:
Elise Olson
email: eo2651@princeton.edu
Additional points of contact:
Jasmin John
email: jasmin.john@noaa.gov
John Dunne
email: john.dunne@noaa.gov

Data type: Processed model output

Model: GFDL-ESM4.1 (Dunne et al., 2020; Stock et al., 2020)

Original source locations (output years):
/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp (1975-2014)
/archive/Elise.Olson/fre/xanadu_esm4_20190304/ESM4/202208/ESM4_historical_D1_CRT_1975_2014/gfdl.ncrc4-intel18-prod-openmp (1975-2014)
/archive/oar.gfdl.bgrp-account/CMIP6/ESM4/ScenarioMIP/ESM4_ssp245_D1/gfdl.ncrc4-intel16-prod-openmp (2015-2022)
/archive/Elise.Olson/fre/xanadu_esm4_20190304/ESM4/202301/ESM4_ssp245_D1_CRT_2015_2024/gfdl.ncrc4-intel18-prod-openmp (2015-2022)

Date of data generation: 2021-2023

Geographic location of (model) data generation: The Geophysical Fluid Dynamics Laboratory (GFDL) of the National Oceanic and Atmospheric Administration (NOAA).

Information about funding sources that supported the collection of the data: This work was completed under project #21413 funded by the Ocean Acidification Program of the National Oceanic and Atmospheric Administration, U.S. Department of Commerce.

I. Data (To be archived at NCEI)
The subdirectory 'data' contains extracted model output and additional calculated fields. The format of all the model output data files is netCDF4.
Model output was extracted at locations nearest the geographical coordinates of the moorings described in Sutton et al. (2019), "Autonomous seawater pCO2 and pH time series from 40 surface buoys and the emergence of anthropogenic trends". These files have the following naming convention:
GFDL-ESM4.1.1975_2022.{mooringID}.j{j}i{i}_{grid}.nc
Here mooringID is a file identifier taken from filenames in the original Sutton et al. (2019) dataset and recorded in dfInfoBuoy.csv. (i,j) are the model grid coordinates of the extracted time series, and {grid} specifies whether the data were extracted from the 1x1 degree ocean grid (1x1grid) or the half degree ocean grid (HDgrid). The 1x1 degree ocean grid files were used in all analyses, except where comparisons were made to results from the half degree grid. The variables extracted from each grid are as follows:

1x1 degree ocean grid (1x1grid):
tos(time, lat, lon): Sea Surface Temperature (degC) 
sos(time, lat, lon): Sea Surface Salinity(psu) 
spco2(time, lat, lon): Surface Aqueous Partial Pressure of CO2 (Pa) 
dpco2(time, lat, lon): Delta PCO2 (Pa) 
chlos(time, lat, lon): Surface Mass Concentration of Total Phytoplankton expressed as Chlorophyll in sea water (kg m-3) 
phos(time, lat, lon): Surface pH
o2os(time, lat, lon): Surface Dissolved Oxygen Concentration (mol m-3) 
o2satos(time, lat, lon): Surface Dissolved Oxygen Concentration at Saturation (mol m-3) 
co3(time, z_l, lat, lon): Carbonate Ion Concentration (mol m-3)
co3satarag(time, z_l, lat, lon): Mole Concentration of Carbonate Ion in Equilibrium with Pure Aragonite in sea water (mol m-3) 

half degree ocean grid (HDgrid):
tos(time, yh, xh): Sea Surface Temperature (degrees C)
sos(time, yh, xh): Sea Surface Salinity (psu)
chlos(time, yh, xh): Surface Mass Concentration of Total Phytoplankton expressed as Chlorophyll in sea water (kg m-3)

Additionally, all files contain the variable 'time' as "days since 1850-01-01 00:00:00" on a "noleap" calendar. The dimensions lat/yh and lon/xh have length 1 and reflect the coordinates of each time series on the model grid. 

Several additional calculated global surface fields that are necessary to reproduce our figures or analyses are also included. 
bcfitvar.GFDL-ESM4.1.1975_2022.*.nc contain fits (b[0]+b[1]*t(days)+gsmoothseas(year day) ) to global surface data along with the magnitude of variability of the seasonal cycle and remaining detrended, deseasonalized component. These files were created using runSave_cfitvar.csh, which calls the function save_bcfitvar in the module extremes.py (see Code section).
MMM.GFDL-ESM4.1.1975_2022.19880414.nc contains the Mean Monthly Maximum surface temperature (MMM, degrees C) based model output for the period 1975-2022 and referenced to the year 1988 for consistency with the NOAA satellite product applied to observation-based analyses. 

II. Code 
py39dc.yml is a yaml-file that describes the Python 3.9 environment that was used for calculations. The sequence of commands used to install the environment is documented in the comments at the bottom of the file. These steps include installation of a local package called "Tools", which contains much of the code used for data analysis and visualization. Modules within the local "Tools" package, which is archived here in the "Tools" directory, are loaded within various analysis scripts with the syntax "from Tools import [NameOfModule] as [ModuleAlias]". 

shellScripts is a directory containing C shell scripts used to submit jobs (or series of jobs) to a local compute nodes via the Slurm Workload Manager.
The following files were used to extract/calculate the data archived in the data directory (described above): 
- runExtractMoorings_any.csh uses runExtractMoorings_scenj.csh to submit jobs calling code in extractMoorings_scenj.py that was used to extract the netCDF4 time series files
- runSaveMMMs.csh calls code in the module extremes.py via runSaveMMMs.py to calculate and save the Mean Monthly Maxima for surface temperature
- runSave_bcfitvar.csh calls code in the module extremes.py that was used to calculate and save the surface fits and variability mangitudes
The following files carry out preliminary analysis steps based on the time series archived in the data directory and produce local intermediate files:
- run_buoyComp_scen40.csh submits a job corresponding to each mooring location via run_buoyComp_scenj.csh, which in turn calls OAPBuoyComp.py with the argument run1CompScen. This triggers the execution of code which loads the netCDF4 time series files archived here in the data directory and carries out model and observation fits and other calculations. The results are then saved using Python's "pickle" module in a series of intermediate ".pkl" files, which are then reloaded for subsequent analyses. 
- runSaveFreqBands.csh calls code frequencyBands.py via saveFreqBands.py that loads the intermediate ".pkl" files, filters the time series, and saves the results in additional intermediate ".pkl" files.

Tools contains code used for intermediate calculations and modules loaded by other analysis scripts. It was installed in the environment used for all calculations as an editable package using the command conda develop.

analysisScripts contains the code that produces the figures, tables, and remaining calculations included in the manuscript in the form of Jupyter notebooks. The figures and tables are created with the specified file names with code in the following files:
Component:  File Name                               Source Code
Figure 1:   map1deg_rev.pdf                         Map-Paper-Revised201401.ipynb
Figure 2:   longTermTrends.pdf                      trends-longterm-Fig-Paper.ipynb
Figure 4:   varMaps_Rev.pdf                         stDevMaps-Paper.ipynb
Figure 5:   bxf_tos_phos_spco2_apco2_Rev.pdf        boxcarVarSummaryPlot-Paper-Rev.ipynb
Figure 6:   bxf_sos_o2os_AOUos_chlos_Rev.pdf        boxcarVarSummaryPlot-Paper-Rev.ipynb
Figure 7:   modObsExtremesPercentile2Row_Rev.pdf    extremes-Paper-Rev202401.ipynb
Figure 8:   modObsExtremesTOm_Rev.pdf               extremes-Paper-Rev202401.ipynb
Figure 9:   lohi_Rev_Monthly.pdf                    boxcars-bands-highVsLowPlots-Paper-Rev202401-FixPercentLocations-withTable-MonthlyAndLonger.ipynb
Figure S1:  bxf_1D_HD_Rev.png                       boxcarVarSummaryPlot-Paper-HD-Rev.ipynb
Figure S2:  taylor.png                              TaylorTarget-Paper.ipynb
Figure S3:  varMaps_o2_chl_Rev.png                  stDevMaps-Paper-Rev202401.ipynb
Figure S4:  bxf_EqPac_Rev.png                       boxcarVarSummaryPlot-EqPac-Paper-Rev.ipynb
Figure S5:  bxf_tos_phos_spco2_apco2.equiv_Rev.png  boxcarVarSummaryPlot-Paper-equiv-Rev.ipynb
Figure S6:  bxf_sos_o2os_AOUos_chlos.equiv_Rev.png  boxcarVarSummaryPlot-Paper-equiv-Rev.ipynb
Figure S7:  modObsExtremesPercentileSens_Rev.png    extremes-Paper-PercentileSensitivity-Rev.ipynb
Figure S8:  modObsExtremesTOmSens_Rev.png           extremes-Paper-TOmSensitivity-RevThresh.ipynb
Table 1:    TableStats.tex                          variability-calculations-Table-Paper.ipynb
Table S1:   Table1.tex                              table-moorings-paper-New.ipynb
Table S2:   TableAlkFits.tex                        table-alkfits-paper.ipynb 
Table S3:   TableLoHi_Monthly.tex                   boxcars-bands-highVsLowPlots-Paper-Rev202401-FixPercentLocations-withTable-MonthlyAndLonger.ipynb

License
This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

References
Dunne, J.P., Horowitz, L.W., Adcroft, A.J., Ginoux, P., Held, I.M., John, J.G., Krasting, J.P., Malyshev, S., Naik, V., Paulot, F., Shevliakova, E., Stock, C.A., Zadeh, N., Balaji, V., Blanton, C., Dunne, K.A., Dupuis,C., Durachta, J., Dussin, R., Gauthier, P.P.G., Griffies, S.M., Guo, H., Hallberg, R.W., Harrison, M., He, J., Hurlin, W., McHugh, C., Menzel, R., Milly, P.C.D., Nikonov, S., Paynter, D.J., Ploshay, J., Radhakrishnan,A., Rand, K., Reichl, B.G., Robinson, T., Schwarzkopf, D.M., Sentman,L.T., Underwood, S., Vahlenkamp, H., Winton, M., Wittenberg, A.T., Wyman, B., Zeng, Y., Zhao, M., 2020. The GFDL Earth System Model version 4.1 (GFDL-ESM 4.1): Overall coupled model description and simulation characteristics. Journal of Advances in Modeling Earth Systems 12, e2019MS002015. doi:10.1029/2019MS002015.

Olson, E. M., John, J. G., Dunne, J., Stock, C., Drenkard, E. J., and Sutton, A., Site-specific multiple stressor assessments based on high frequency surface obserations and an Earth system model, submitted to Earth and Space Science.

Stock, C.A., Dunne, J.P., Fan, S., Ginoux, P., John, J., Krasting, J.P.,Laufkotter, C., Paulot, F., Zadeh, N., 2020. Ocean biogeochemistry in GFDL's Earth System Model 4.1 and its response to increasing atmospheric CO2. Journal of Advances in Modeling Earth Systems 12, e2019MS002043.1074. doi:10.1029/2019MS002043.

Sutton, A.J., Feely, R.A., Maenner-Jones, S., Musielwicz, S., Osborne, J., Dietrich, C., Monacci, N., Cross, J., Bott, R., Kozyr, A., Andersson, A.J., Bates, N.R., Cai, W.J., Cronin, M.F., De Carlo, E.H., Hales, B., Howden,S.D., Lee, C.M., Manzello, D.P., McPhaden, M.J., Melendez, M., Mickett, J.B., Newton, J.A., Noakes, S.E., Noh, J.H., Olafsdottir, S.R., Salisbury, J.E., Send, U., Trull, T.W., Vandemark, D.C., Weller, R.A., 2019. Autonomous seawater pco2 and ph time series from 40 surface buoys and the emergence of anthropogenic trends. Earth System Science Data 11, 421-439. URL: https://essd.copernicus.org/articles/11/421/2019/, doi:10.5194/essd-11-421-2019.

