#!/bin/bash
#SBATCH -A jprocyk     # change to your asu username
#SBATCH -n 1                     # number of nodes, only need 1 for 99% of jobs
#SBATCH -c 6                     # number of cores
#SBATCH -t 4-00:00               # wall time (D-HH:MM)
#SBATCH -o slurm_RESNET.%j.out          # STDOUT (%j = JobId)  # Outfile
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)   # Errorfile
#SBATCH -p cidsegpu1     # Partition of ASU HPC Agave cluster to use
#SBATCH -q wildfire      # The queue to submit to, vary by partition
#SBATCH --gres=gpu:1     # Request 1 GPU to run this job
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     # notifications send to my email
#SBATCH --chdir=/scratch/jprocyk/ML/ML_for_aptamers/    # Working directory, in this case make sure it points to wherever pytorch-intro is located

module load cuda/10.2.89    # load cuda
module load anaconda3/4.4.0   # load anaconda

source activate exmachina3     # Activate our conda env

python resnet.py
