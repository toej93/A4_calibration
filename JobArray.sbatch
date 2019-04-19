#!/bin/bash

#SBATCH --job-name=arrayJob
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=[0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]
#SBATCH --time=0:45:00
# use --partition=broadwl if you want to submit your jobs to Midway2
#SBATCH --partition=broadwl
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=1000


######################
# Begin work section #
######################
#[1,2,9,10,17,18,25,26]

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Do some work based on the SLURM_ARRAY_TASK_ID
# For example: 
# ./my_process $SLURM_ARRAY_TASK_ID
#python EventsfromAbove.py $SLURM_ARRAY_TASK_ID
#python PedestalCorrector.py 
python TimingCalibration.py $SLURM_ARRAY_TASK_ID 

# 
# where my_process is you executable
