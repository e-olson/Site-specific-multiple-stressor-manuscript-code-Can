#!/bin/csh -fx

#------------------------------------
#Slurm batch directives 
#SBATCH --job-name=mooringExtract-%j
#SBATCH --time=23:00:00
#SBATCH -p batch
#SBATCH --account=gfdl_b
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Elise.Olson@noaa.gov
#------------------------------------

pwd; hostname; date
echo $SCENINDEX
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

python /home/Elise.Olson/OAPMSE/analysis/Tools/Tools/extractMooringLocs_scenj.py scenario $SCENINDEX
