# Shapley-PC
This repository provides the code for the paper "Shapley-PC: Constraint-based Causal Structure Learning with Shapley Values". 

There are four main python scripts:
- `PC.py` contains the `pc()` function that overrides the one from the causal-learn package to allow for our proposed decision rule. Example usage of this function is given below.
- `spc.py ` contains the `shapley_cs()` functions that applies our proposed v-structure discovery algorithm within the `pc()` function.
- `models.py` contains the `run_method()` function that allows the run of our method as well as all the baselines used in the experiments.
- `main.py` reproduces the experiments. The parameters for the runs are loaded from the config folder (e.g. [real_data](config/real_data.yaml)). Example usage of this script is given below.

All the plots included in the paper can be inspected interactively from the [results/figs](results/figs) folder. Just download them and open them in a browser. 

A jupyter notebook collecting the stored results and producing the plots in the paper is provided [here](notebooks/Experiments.ipynb).

### Example usage
To run the Shapley-PC algorithm from python at the root folder, run:
```
# Imports
import networkx as nx
from sklearn.preprocessing import StandardScaler
from PC import pc
from utils import random_stability, load_bnlearn_data_dag
from castle.metrics import MetricsDAG

# Load sachs data
X_s, B_true = load_bnlearn_data_dag(dataset_name='sachs', data_path='datasets', sample_size=2000, seed=2023)

# Run SPC
random_stability(2023)
fitted = pc(data=X_s, alpha=0.01, test='kci', uc_rule=3, uc_priority=2, 
            selection='bot', show_progress=True, verbose=False)
W_est = fitted.G.graph
B_est = (W_est > 0).astype(int)

# Evaluate 
mt = MetricsDAG(B_est, B_true)
print(mt.metrics)

OUTPUT:
{'fdr': 0.5556, 'tpr': 0.2353, 'fpr': 0.1316, 'shd': 14, 'nnz': 9, 'precision': 0.4444, 'recall': 0.2353, 'F1': 0.3077, 'gscore': 0.0}
```

To reproduce the experiments, from a terminal at the root folder, run:
```
python main.py <synthetic_data,real_data,bespoke_config>
```
Beware, an end-to-end run of this script with the config provided takes several hours if run on single cpu/gpu since it contains all the scenarios. Modify it to run only part of it in one go.

### Requirements
The code was tested with Python 3.9.9 and 3.10.12. `requirements.txt` provides the necessary python packages. Run `pip install -r requirements.txt` from a terminal at the root folder to install all packages in your virtual environment. 

Note that some of the baselines used in the experiments and called by the `run_method()` function, require separate installations. 
`py-causal` needs to be installed separately as detailed in `requirements.txt`. The R dependencies for `CausalDiscoveyToolbox` also need separate installation. `install_script.R` installs the necessary R packages but this needs to be ran from R while in the root folder. R-4.1.2 was used for the testing. 