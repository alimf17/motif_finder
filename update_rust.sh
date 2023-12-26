#!/bin/bash
  

#SBATCH --account=mia174
#SBATCH --partition=shared
#SBATCH --job-name=proc # job name
#SBATCH --output=%x_%j.log             # stdout fname (will be <job-name>_<job_id??>.log) Not sure about job_id here to be honest. It's numbers... just numbers.
#SBATCH --error=%x_%j.err              # stderr fname
#SBATCH --mail-type=BEGIN,END,FAIL     # will get email at job start, finish, and if it fails
#SBATCH --mail-user=alimf@umich.edu # email (please don't use mine...)
#SBATCH --nodes=1                      # Total number of nodes
#SBATCH --ntasks-per-node=10           # Total # of mpi tasks
#SBATCH --time=48:00:00                 # Run time (hh:mm:ss)
#SBATCH --mem=20G

echo 1\n | curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
