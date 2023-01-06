# ml_toric_code: Machine Learning Toric Code Topological Phase
Authors: Yanting Teng

ml_toric_code is the code base for our paper [Classifying topological neural network quantum states via diffusion maps][arXiv link to be added]
The code base is implemented in JAX.

- To run the variational Monte Carlo component, see the demo notebook [`Optimization_demo.ipynb`](https://github.com/teng10/ml_toric_code/blob/0370ce6f8d35b7920cb662f436f88a6df8718816/Optimization_demo.ipynb).
- For visualizing the dataset linked [here](https://drive.google.com/file/d/1pIdqtE137oLBKq-EmNkpxA_xyH6MxgLB/view?usp=sharing) , see `DM_data.ipynb`. 


## Commands to run the code
The experiments are run on FASRC Cannon cluster at Harvard University. It is set up with `config` files in `exp_cluster/configs/` folder. To generate and characterize the datasets, follow these steps:
1. Optimize for intial ``seeds'' $\{\Lambda^0\}$ at various field valuess $h$ (specified in `config_opt.py`) 
```
sbatch opt_v1.slurm.sh
```
2. Estimate properties of the optimized states  
e.g. for the optimized states with the tracker id 10331255 using `config_opt_est_v1_10331255`: 
```
sbatch opt_est_v1.slurm.sh
```
3. Generate ensembles for each state in ``seeds'' 
e.g. to generate the ensembles the tracker id 10910143 using `config_ens_v2_10910143.py`: 
```
sbatch ens_v2.slurm.sh 
```
4. Estimate properties of the states in ensembles
e.g. use `config_est_ens.py`
```
sbatch est_ens.slurm.sh
```
5. Perform dimensional reduction of the dataset using `diffusion_maps` module. This is done in a colab notebook. Some examples of diffusion map spectra are in `DM_data.ipynb`. 
