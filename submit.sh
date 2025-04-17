#!/bin/bash
# Usage:
# ./dependency_launcher.sh -n 5 <file.sub>
# Grab command line options
# n: Number of times to submit the job
# sbatch ptuning/gpt4all_sft8b.sh

# create 10 jobs that sequentially run (once one finishes, the other starts)
# bash submit.sh -n 10 ptuning/gpt4all_sft8b.sh

N_CALLS=10
while getopts "n:J:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done
# Grab the .sub file to run
SUBFILE=${@:$OPTIND:1}
if [[ -z $SUBFILE ]]; then
  echo "Usage: $(basename "$0") [flags] [sub file]"
  exit 1
fi
echo "Calling [$SUBFILE] $N_CALLS times."
# Repeat calls
PREV_JOBID=""
for (( i = 1; i <= $N_CALLS; i++ ))
do
  if [ -z $PREV_JOBID ]; then
    echo "Submitting job ${i}"
    OUTPUT=$(sbatch $SUBFILE)
  else
    echo "Submitting job ${i} w/ dependency on jobid ${PREV_JOBID}"
    OUTPUT=$(sbatch --dependency=afterany:${PREV_JOBID} $SUBFILE)
  fi
  PREV_JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
done
