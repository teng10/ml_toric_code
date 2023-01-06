#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared  # Partition to submit to
#SBATCH --mail-user=y_teng@g.harvard.edu #Email to which notifications will be sent
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mem=50000
#SBATCH -c 1
#SBATCH --array=0-11 # enumerating the sector and iteraction index: e.g. 4 x 3=12
#SBATCH -o /n/home11/yteng/experiments/optimization/logs/%j.out # Standard out goes to this file
#SBATCH -e /n/home11/yteng/experiments/optimization/logs/%j.err # Standard err goes to this filehostname
source activate ml_tc
exp_file_path=/n/home11/yteng/ml_toric_code/exp_cluster/
cd /n/home11/yteng/ml_toric_code/exp_cluster/
FILEPATH="/n/home11/yteng/experiments/optimization/data/${SLURM_ARRAY_JOB_ID}/"
mkdir -p ${FILEPATH}
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "filepath is ${FILEPATH}"
python opt_v1.py --config="${exp_file_path}configs/config_opt.py" --config.job_id=${SLURM_ARRAY_JOB_ID} --config.file_id=${SLURM_ARRAY_TASK_ID} --config.data_dir=${FILEPATH}> "$FILEPATH/${SLURM_ARRAY_JOB_ID}_log.txt"
