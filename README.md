# Shapley-PC
This repository provides the code for the paper "Shapley-PC: Constraint-based Causal Structure Learning with Shapley Values". 

There are four main python scripts:
- `PC.py` contains the `pc()` function that overrides the one from the causal-learn package to allow for our proposed decision rule. Example usage of this function is given below.
- `spc.py ` contains the `shapley_cs()` functions that applies our proposed v-structure discovery algorithm within the `pc()` function.
- `models.py` contains the `run_method()` function that allows the run of our method as well as all the baselines used in the experiments.
- `main.py` reproduces the experiments. The parameters for the runs are loaded from the config folder (e.g. [bnlearn_data](config/bnlearn_data.yaml)). Example usage of this script is given below.

All the plots included in the paper can be inspected interactively from the [results/figs](results/figs) folder. Just download them and open them in a browser. 

A jupyter notebook collecting the stored results and producing the tables and plots in the paper is provided [here](notebooks/Experiments.ipynb).

### Example usage
To run the Shapley-PC algorithm from python at the root folder, run:
```
# Imports
from PC import pc
from utils.helpers import random_stability
from utils.data_utils import load_bnlearn_data_dag
from utils.graph_utils import DAGMetrics

# Load alarm data
X_s, B_true = load_bnlearn_data_dag(dataset_name='alarm', data_path='datasets', sample_size=10000, seed=2024)

# Run SPC
random_stability(2024)
fitted = pc(data=X_s, alpha=0.01, show_progress=True, verbose=False)
B_est = fitted.G.graph.T

# Evaluate 
mt = DAGMetrics(B_est, B_true, sid=False)
metrics = ['nnz', 'shd', 'skF1', 'arrF1', 'immoral_UT_F1']
print({m:round(v,2) for m,v in mt.metrics.items() if m in metrics})

OUTPUT:
{'nnz': 55, 'arrF1': 0.52, 'skF1': 0.97, 'shd': 59.0, 'immoral_UT_F1': 0.69}
```

### Reproduce Results
To reproduce the experiments, from a terminal at the root folder, run:
```
python main.py <synthetic_data,real_data,bespoke_config>
```
Beware, an end-to-end run of this script with the config provided takes several hours if run on single cpu since it contains all the scenarios. Modify it to run only part of it in one go.

### Datasets
The datasets used for the synthetic data are created within the `main.py` function. Some of the `bnlearn` datasets are provided in `.rds` since no `.bif` are available in the `bnlearn` repository for the gaussian and conditional linear gaussian Bayesian Networks. The data was generated in a R environment, for 10 seeds, using the script provided [here](datasets/bayesian/sample.R). `.pkl` files that are picked up by the `main.py` function to reproduce the experiments are provided for these data, these were produced using [this](datasets/bayesian/import_r.py) script.

### Requirements
The code was tested with Python 3.10.12. `requirements.txt` provides the necessary python packages. Run `pip install -r requirements.txt` from a terminal at the root folder to install all packages in your virtual environment. 

Note that the R dependencies for `CausalDiscoveyToolbox` need separate installation. `install_script.R` installs the necessary R packages but this needs to be ran from R while in the root folder. R-4.1.2 was used for the testing. 
