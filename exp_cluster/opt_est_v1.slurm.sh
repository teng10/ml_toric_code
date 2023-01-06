#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared # Partition to submit to
#SBATCH --mail-user=y_teng@g.harvard.edu #Email to which notifications will be sent
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mem=20000
#SBATCH -c 1
#SBATCH --array=0-2 #iteration index
#SBATCH -o /n/home11/yteng/experiments/optimization/logs/%j.out # Standard out goes to this file
#SBATCH -e /n/home11/yteng/experiments/optimization/logs/%j.err # Standard err goes to this filehostname
# To use this script, make sure the array index (array_task_id) = # iterations
# Modify the filepath to the current dataset we want to estimate for
source activate ml_tc
exp_file_path=/n/home11/yteng/ml_toric_code/exp_cluster/
cd /n/home11/yteng/ml_toric_code/exp_cluster/
FILEPATH="/n/home11/yteng/experiments/optimization/data/34548775/"
mkdir -p ${FILEPATH}logs/
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "filepath is ${FILEPATH}"
python opt_est_v1.py --config="${exp_file_path}configs/config_opt_est_v1.py" --config.job_id=${SLURM_ARRAY_JOB_ID} --config.file_id=${SLURM_ARRAY_TASK_ID} --config.data_dir=$FILEPATH > "${FILEPATH}logs/${SLURM_ARRAY_JOB_ID}_log.txt"
