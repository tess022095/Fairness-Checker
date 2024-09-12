#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=3:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --mem=20G   # maximum memory per node
#SBATCH --job-name="ac"
#SBATCH --mail-user=gnguyen@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="ac"


python3 AC2.py
python3 AC3.py
python3 AC4.py
python3 AC5.py
python3 AC6.py
python3 AC7.py
python3 AC8.py
python3 AC9.py
python3 AC10.py
