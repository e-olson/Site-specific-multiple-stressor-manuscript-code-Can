#!/bin/csh -fx

cd /home/Elise.Olson/OAPMSE/analysis/Tools/Tools/scripts/bcTasksPPAN
source /app/Modules/default/init/csh
module load slurm
# submitted: 8, 1, 3, 11
foreach SCENID ( 11 )
  foreach BUOYID ( `seq 0 1 40` ) 
    set jobname = buoyComp-$SCENID-$BUOYID
    set output  = output/buoyComp-$SCENID-$BUOYID-%j.out
    set echo
    sbatch -J $jobname -o $output --export BUOYID=$BUOYID,SCENID=$SCENID run_buoyComp_scenj.csh
    unset echo
  end
end
