## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Setup on cluster](#setup-on-a-cluster)

## General info
Working space for Python scripts implementing a Geometry Matching (GeM) Local Order Metric (LOM) for forward flux sampling (FFS). The algorithm enabling the measurement of order in the neighbourhood of an atomic or molecular site is described in [Martelli2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.064105). The FFS is available in Python Suite for Advanced General Ensemble Simulations ([PySAGES](https://github.com/SSAGESLabs/PySAGES/tree/ffs)). PySAGES is an Python implementation of [SSAGES](https://ssagesproject.github.io/) with support for GPUs.

## Technologies
Project was created with:
* python 3.9.7
* jax 0.3.1
* jaxlib 0.3.0
* jaxlie 1.2.10
* numpy 1.22.2

## Setup
All packages are listed in `pakage-list.txt`. This file may be used to create an environment using:
```
$ conda create --name <env> --file <this file>
```
To execute the code, run
```
python pygem.py
```
The code saves all the results in a pickled file, `gem_results.pickle`. This option should be removed to speed up the execution.
If the analysis is done on GPUs, specify the device before running the code:
```
export CUDA_VISIBLE_DEVICES=0
```
or
```
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
```
This code was tested on MAC OS. In order to use it on a cluster with GPUs, an appropriate environment with CUDA and Jax should be setup. This is discussed in section  [Setup on a cluster](#setup-on-a-cluster).

Input parameters are specified in `params.ini`, together with files for reference sites, coordinates of simulated atoms, and simulation box specification in
* `example/reference.txt`
* `example/positions.txt`
* `example/simulation_box.txt`, respectively.

### Testing
To analyse the output file `gem_results.pickle` and visulise the optimisation process, execute
```
python analyse_pygem.py <atom-site-number> <path-to-log-file>
```
So, e.g.,
```
python analyse_pygem.py 0 ./pygem.log
```
will generate plots that demonstrate the score computation for atom 0.

In order to compute the compilation and execution time run the following:
```
cv = jit(gem(
        np.array(positions),
        reference_positions=reference_positions,
        box=box))
t = time.time()
mean_score, all_results = cv(np.array(positions)).block_until_ready()
print(f'Including just-in-time compilation {time.time()-t}')

t = time.time()
mean_score, all_results = cv(np.array(positions)).block_until_ready()
print(f'Execution {time.time()-t}')
```
If we didn’t include the warm-up call separately, everything would still work, but then the compilation time would be included in the benchmark.(Note the use of `block_until_ready()`, which is required due to JAX’s Asynchronous execution model).
## Setup on a cluster
If you would like to use PySAGES on a cluster, you should follow the instructions below. Among all of these steps, there are also some useful (arguably) suggestions regarding the management of conda environments. I hope you will enjoy this “smooth” read and try the installation yourself.

Specify in `.condarc` where you want all the environment files and packages to be placed:
```
pkgs_dirs:
 - $HCBASE/.conda/pkgs
envs_dirs:
 - $HCBASE/.conda/envs
```
Or type
```
# Anaconda is loaded with an active base or environment

conda config --add pkgs_dirs $HCBASE/.conda/pkgs
conda config --add envs_dirs $HCBASE/.conda/envs
```
`$HCBASE` may be substituted for `/any/path/of/your/choice`, of course.

No we can create our environment in which we define all the rules. Well, most of them.
```
conda create --name pysages python=3.9
conda activate pysages
```
And the fun begins! Time to install stuff…
```
conda install -c conda-forge cudatoolkit=11.2
```
Check if you did the right thing:
```
conda list cudatoolkit
```
If you are satisfied, continue with the fun
```
conda install -c conda-forge cudnn
```
Even though the commands above are supposed to set-up CUDA tools for us, you still might need to load this (don’t ask why!)
```
module load cuda/11.2
```
For the given combination of CUDA (11.2) and cudnn8.2, the following installation of jax is required:
```
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
You can check the jax installation
```
module load anaconda3
conda activate pysages
module load cuda
CUDA_VISIBLE_DEVICES=1 python -c "import jax.numpy as np; print(np.sin(np.pi/2))"
``````
Install CUDA Toolkit
```
conda install -c conda-forge openmm cudatoolkit=11.2
```
Verify your installation by typing the following command:
```
python -m openmm.testInstallation
module load cmake/3.21.1
git clone https://github.com/SSAGESLabs/openmm-dlext.git
``````
GCC versions < 5 do not support certain functions, e.g., std::is_trivially_copyable, from the C++11 standard. So, make sure to load an appropriate gcc, such as
```
module load gcc/6.5.0 
```
Now we are building the openmm-dlext :
```
cd openmm-dlext && mkdir build && cd build && cmake .. && make install
```
Verify your installation by typing the following command:
```
python -c "import openmm_dlext"
```
And the final step!
```
git clone https://github.com/SSAGESLabs/PySAGES.git
cd PySAGES && pip install .
```
### Additional notes
You can see your Conda environment in
```
vi .conda/environments.txt
```
Conda sometimes breaks dependencies. If needed, you can set the environment variables when an environment is activated by editing the `activate.d/env_vars.sh` script. See [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#macos-and-linux).

So, in my case, I would have to follow these steps:
```
cd $HCBASE/.conda/envs/pysages
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```
Edit `./etc/conda/activate.d/env_vars.sh` as follows:
```
#!/bin/sh
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/your/path:${LD_LIBRARY_PATH}
```
And then in `deactivate.d/env_vars.sh`:
```
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
```
