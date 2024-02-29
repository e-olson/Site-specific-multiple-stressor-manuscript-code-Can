#!/bin/csh -fx

#------------------------------------
#Slurm batch directives
#SBATCH --job-name=MMMs
#SBATCH --time=23:00:00
#SBATCH --output=MMMs.out
#SBATCH -p batch
#SBATCH --account=gfdl_b
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Elise.Olson@noaa.gov
#------------------------------------

pwd; hostname; date

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

python /home/Elise.Olson/OAPMSE/analysis/Tools/Tools/saveMMMs.py

date

