#!/bin/csh -fx

cd /home/Elise.Olson/OAPMSE/analysis/Tools/Tools/scripts
source /app/Modules/default/init/csh
module load slurm

# listScenDates = [('ESM4_historical_D1',[1975,2014]), #0
#             ('ESM4_ssp119_D1',yrspan_future), #1
#             ('ESM4_ssp126_D1',yrspan_future), #2
#             ('ESM4_ssp245_D1',yrspan_future), #3
#             ('ESM4_ssp245_D151',yrspan_future), #4
#             ('ESM4_ssp245_D201',yrspan_future), #5
#             ('ESM4_ssp370_D1',yrspan_future), #6
#             ('ESM4_ssp534-over_D1',yrspan_future), #7
#             ('ESM4_ssp585_D1',yrspan_future), #8
#             ('ESM4_piControl_D_0226_0265',[226,265]), #9
#             ('ESM4_piControl_D_0312_0351',[312,351]), #10
#             (('ESM4_historical_D1','ESM4_ssp245_D1_histCont'),[1975,2014,2015,2022]), #11
#             ('ESM4_1pctCO2_D1_start',[1,40]), #12
#             (('ESM4_1pctCO2_D1_peak','ESM4_1pctCO2-cdr_D1_peak'),[121,140,141,160]), #13
#             ('ESM4_1pctCO2-cdr_D1_end',[241,280]),] #14

# submitted so far: 0 1 2 3 4 5 6 7 8 9 10 11 
foreach task ( 11 )
  set jobname = extract-$task
  set output  = output/extract-$task-%j.out
  set echo
  sbatch -J $jobname -o $output --export SCENINDEX=$task runExtractMoorings_scenj.csh 
  unset echo
end
