#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared  # Partition to submit to
#SBATCH --mail-user=y_teng@g.harvard.edu #Email to which notifications will be sent
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mem=20000
#SBATCH -c 1
#SBATCH --array=0-32
#SBATCH -o /n/home11/yteng/experiments/optimization/logs/%j.out # Standard out goes to this file
#SBATCH -e /n/home11/yteng/experiments/optimization/logs/%j.err # Standard err goes to this filehostname
source activate ml_tc
exp_file_path=/n/home11/yteng/ml_toric_code/exp_cluster/
cd /n/home11/yteng/ml_toric_code/exp_cluster/
FILEPATH="/n/home11/yteng/experiments/optimization/data/10331255/ensemble/11480580/"
mkdir ${FILEPATH}logs/
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "filepath is ${FILEPATH}"
python estimates_ensemble.py --config="${exp_file_path}configs/config_est_ens.py" --config.job_id=${1} --config.file_id=${SLURM_ARRAY_TASK_ID} --config.num_workers=5 --config.data_dir=${FILEPATH}> "${FILEPATH}../../logs/${SLURM_ARRAY_JOB_ID}_log.txt"
#python estimates_ensemble.py --config="${exp_file_path}configs/config_est_ens.py" --config.job_id=${SLURM_ARRAY_TASK_ID} --config.file_id=${1} --config.num_workers=10 --config.data_dir=${FILEPATH}> "${FILEPATH}../../logs/${SLURM_ARRAY_JOB_ID}_log.txt"
