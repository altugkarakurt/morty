#!/bin/sh
# Name the process
# ----------------
#$ -N compIden_$JOB_ID
#
# Call from the current working directory; no need to cd
# ------------------------------------------------------
#$ -cwd
#$ -q default.q
#
# Max time limits
# ---------------
#$ -l s_rt=5:00:00
#$ -l h_rt=10:00:00
#
# Output/Error Text
# ----------------
#$ -o ./logs/makamRecogTest_$JOB_ID.out
#$ -e ./logs/makamRecogTest_$JOB_ID.err
#
# Create an array job = !!!!!!number of audio in the target folder!!!!!!
# ----------------
#$ -t 1-2:1
#
# Send me a mail when processed and when finished:
# ------------------------------------------------
# -M sertan.senturk@upf.edu
# -m bea
#
# Start script
# --------------------------------

printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME at `date`\n"

# force UTF 8
export LANG="en_US.utf8"

module load python/2.7.5
module load essentia/2.0.1

python testBozkurt_mode.py 5 
% the int job Done
printf "Job $JOB_ID done at `date`\n"
