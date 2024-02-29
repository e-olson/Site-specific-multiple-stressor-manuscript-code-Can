#!/bin/csh -fx
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=gfdl_b

pwd; hostname; date
echo $BUOYID $SCENID

#set environment
if ( `gfdl_platform` == "hpcs-csc" ) then

    source $MODULESHOME/init/csh
    module purge
    module load conda
    conda activate py39dc
    module list
else
    echo "ERROR: Invalid platform"
    exit 1
endif

python /home/Elise.Olson/OAPMSE/analysis/Tools/Tools/OAPBuoyComp.py recalcStats $BUOYID $SCENID
