import sys
import os
import yaml
import argparse
import time
import numpy as np
import networkx as nx
import pandas as pd
import torch
import pydot
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import ParameterGrid
import gc
gc.set_threshold(0,0,0)
from utils import random_stability, save_pickle, load_args, existing_runs_pkl, check_existing_runs, load_bnlearn_data_dag
from models import run_method

try:
    from notears import utils as nt_utils
except:
    print('NOTears not installed, trying local import')
    sys.path.append('../notears/')
    from notears import utils as nt_utils

os.environ['R_HOME'] = '../R/R-4.1.2/bin/'
try:
    import cdt
    cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'
    # print(cdt.SETTINGS.rpath)
except:
    print('CDT not installed, trying local import')
    sys.path.append('../CausalDiscoveryToolbox/') 
    import cdt
from cdt.metrics import SHD, SID, SHD_CPDAG, SID_CPDAG, precision_recall


# @profile
def main():
    args = load_args()
    ##---------CREATE DIRs--------------
    if not os.path.isdir(os.path.join(args.output_dir, args.run_name)):
        os.makedirs(os.path.join(args.output_dir, args.run_name))
    if not os.path.isdir(os.path.join(args.output_dir, args.run_name, "est_matrices")):
        os.makedirs(os.path.join(args.output_dir, args.run_name, "est_matrices"))
    if not os.path.isdir(os.path.join(args.output_dir, args.run_name, "args")):
        os.makedirs(os.path.join(args.output_dir, args.run_name, "args"))
    if not os.path.isdir(os.path.join(args.output_dir, args.data_dir)):
        os.makedirs(os.path.join(args.output_dir, args.data_dir))

    ##---------DUMP ARGS--------------
    if args.dump_args:   
        # Dump args
        arg_file = os.path.join(args.output_dir, args.run_name, "args", f'{args.expt_name}.yaml')
        if os.path.isfile(arg_file) and not args.overwrite_args:
            print('Arg file exists, renaming...')
            arg_file = os.path.join(args.output_dir, args.run_name, "args", f'{args.expt_name}_{"".join([str(a) for a in time.gmtime()[0:6]])}.yaml')
        with open(arg_file, 'w') as f:
            yaml.dump(args.__dict__, f)
    
    if os.path.isdir(os.path.join(args.output_dir, args.run_name)):
        existing_runs_list = existing_runs_pkl(args.output_dir, args.run_name, args.expt_name)   
    else:
        existing_runs_list = []

    param_grid = {a:v for a,v in args.__dict__.items() if type(args.__dict__[a]) == list}
    other_args = {a:v for a,v in args.__dict__.items() if type(args.__dict__[a]) != list}
    
    random_stability(args.main_seed)
    seeds_list = np.random.randint(0, 10000, (args.num_data_per_graph, )).tolist()
    param_grid['seed'] = seeds_list
    all_iter = np.prod([len(v) for k,v in param_grid.items()])
    ## running multiple methods for the same data
    # if len(args.method) > 1:
    param_grid.pop('method')
    method_list = args.method ##methods always a list
    grid = ParameterGrid(param_grid)
    del param_grid

    # perf = defaultdict(list)
    for i, params in enumerate(grid):
        
        ##---------LOAD and PRINT PARAMS--------------
        args = argparse.Namespace(**params,**other_args)
        if i < args.start_iter:
            print(f'Skipping iteration {i+1}/{all_iter}')
            continue

        if args.wandb:
            import wandb
            # reset_wandb_env()
            run = wandb.init(project=args.wdb_proj_name, config=args, 
                            reinit=True, group=args.method, job_type=f'{args.expt_name}_{args.min_abs_unif}_{args.max_abs_unif}_{args.noise_scale}_{args.noise_scale_vec}',
                            name=f'{args.method}_{args.s}_{args.sem_type}_{args.n_nodes}_{args.edge_per_node}_{args.graph_type}_{args.seed}')

        s0 = int(args.n_nodes*args.edge_per_node)
        if s0 > int(args.n_nodes*(args.n_nodes-1)/2):
            print(f'{s0} is too many edges, setting s0 to the max:', int(args.n_nodes*(args.n_nodes-1)/2))
            s0 = int(args.n_nodes*(args.n_nodes-1)/2)

        ##---------CHECK IF DATA IS AVAILABLE --------------
        B_file = os.path.join(args.output_dir, args.data_dir, "DAGs", f'{args.graph_type}_{args.n_nodes}_{args.edge_per_node}_{args.seed}.npy')
        if os.path.isfile(B_file) and not args.overwrite_data:
            B_true = np.load(B_file)
        else:
            ##---------RANDOM DAG-------------- 
            if 'toy' not in args.graph_type:
                random_stability(args.seed)
                B_true = nt_utils.simulate_dag(d=args.n_nodes, s0=s0, graph_type=args.graph_type)
                ##---------TOY DAG-------------- 
            elif args.graph_type == 'toy':
                B_true = np.array( [[ 0,  0,  0,  0,  0],
                                    [ 0,  0,  0,  0,  0],
                                    [ 1,  1,  0,  0,  0],
                                    [ 1,  0,  1,  0,  0],
                                    [ 1,  1,  1,  1,  0]])
                ### toy1: 1->3,4,5; 2->3,5; 3->4,5; 4->5
            elif args.graph_type == 'toy1':
                B_true = np.array( [[ 0,  0,  0,  0,  0],
                                    [ 0,  0,  0,  0,  0],
                                    [ 1,  0,  0,  0,  0],
                                    [ 1,  0,  0,  0,  0],
                                    [ 1,  1,  1,  0,  0]])
                ### toy2: 1->3,4,5; 2->5; 3->5;
            elif args.graph_type == 'toy2':
                B_true = np.array( [[ 0,  0,  0,  0,  0],
                                    [ 1,  0,  1,  1,  0],
                                    [ 1,  0,  0,  1,  0],
                                    [ 1,  0,  0,  0,  0],
                                    [ 1,  1,  1,  1,  0]])
                ### toy3: 1->2,3,4,5; 2->5; 3->2,5; 4->2,3,5
            elif args.sem_type in ['survey', 'earthquake', 'cancer','sachs', 'asia', 
                                    'alarm', 'insurance', 'child', 'hailfinder',  'hepar2']:
                B_true = None ### we load the data later
            else:
                raise NotImplementedError(f'{args.graph_type} not implemented')

        if args.debug:
            print(B_true)
            print("True edges:", {(f'X{a+1}',f'X{b+1}') for a,b in set(nx.DiGraph(B_true.T).edges)})

        data_file = os.path.join(args.output_dir, args.data_dir, f'{args.graph_type}_{args.sem_type}_{args.n_nodes}_{args.edge_per_node}_{args.s}_{args.w_ranges}_{args.noise_scale}_{args.noise_scale_vec}_{args.seed}.npy')
        if os.path.isfile(data_file) and not args.overwrite_data:
            print('Data exists, loading...')
            X = np.load(data_file)
            W_true = np.load(data_file.replace('.npy', '_W.npy'))
        else:
            print("Generating data...")
            if len(args.w_ranges) > 0:
                w_ranges = eval(args.w_ranges)
            else:
                w_ranges = ((-args.max_abs_unif, -args.min_abs_unif), (args.min_abs_unif, args.max_abs_unif))
            if args.noise_scale_vec:
                random_stability(args.seed)
                noise_scale = list(np.random.uniform(0., args.noise_scale, (B_true.shape[1], )))
            else:
                noise_scale = args.noise_scale

            ##--------- Continuous DAG --------------
            random_stability(args.seed)
            W_true = nt_utils.simulate_parameter(B_true, w_ranges=w_ranges)
            n = int(args.s*args.n_nodes)

            ##---------DGP--------------
            if args.sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
                random_stability(args.seed)
                X = nt_utils.simulate_linear_sem(W_true, n, args.sem_type, noise_scale=noise_scale)
            elif args.sem_type in ['mlp', 'mim', 'gp', 'gp-add']: 
                random_stability(args.seed)
                X = nt_utils.simulate_nonlinear_sem(B_true, n, args.sem_type, noise_scale=noise_scale)
            elif args.sem_type in ['survey', 'earthquake', 'cancer','sachs', 'asia', 
                                    'alarm', 'insurance', 'child', 'hailfinder',  'hepar2']:
                X, B_true = load_bnlearn_data_dag(args.sem_type, 'datasets', 2000, seed=2023, standardise=False, print_info=True)
            else:
                raise NotImplementedError(f'{args.sem_type} not implemented')
            
            if args.dump_data:
                if os.path.isfile(data_file) and not args.overwrite_data:
                    print('Data exists, skipping save...')
                else:
                    np.save(data_file, X)
                    np.save(data_file.replace('.npy', '_W.npy'), W_true)
                    np.save(B_file, B_true)

        ##---------PREPROCESSING--------------
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # print("Varsortability =", varsortability(X, B_true, debug=False))
        
        for method in method_list:
            print(f'Running {i+1}/{all_iter} for model: {method} with params: {params}')

            ##---------CHECK IF RUN EXISTS--------------
            check_list = ['method','model','seed','n_nodes','graph_type','sem_type','edge_per_node','s',
                      'noise_scale','noise_scale_vec','w_ranges']
            if 'pc' in method:
                check_list.append('test_alpha')
                check_list.append('test_name')
                check_list.append('priority')
            if 'spc' in method:
                check_list.append('selection')

            run_exists = check_existing_runs({**args.__dict__, 'model':method}, existing_runs_list, check_list=check_list, debug=args.debug)
            # if run_exists == None:
            #     print('Run is Nan, rerunning...')
                # continue
            if run_exists and not args.overwrite_res:
                print('Run exists, skipping...')
                continue

            ##---------MODELS--------------
            W_est, elapsed = run_method(X, method, args.seed, debug=args.debug, device=args.device,
                                        selection=args.selection, priority=args.priority, test_name=args.test_name, test_alpha=args.test_alpha, 
                                        )

            print(f'Time: {".".join([str(a) for a in time.gmtime()[0:6]])}')


            ##---------EVALUATION--------------
            B_est = None
            if W_est is not None:
                B_est = (W_est > 0)
            try:
                acc = nt_utils.count_accuracy(B_true, B_est)
                acc['p_shd'] = acc['shd']/B_true.sum()
                if args.compute_extra_metrics:

                    acc['SHD'] = SHD(B_true, B_est)
                    acc['SID'] = SID(B_true, B_est).item()
                    acc['SHD_CPDAG'] = SHD_CPDAG(B_true, B_est)
                    acc['SID_CPDAG_low'], acc['SID_CPDAG_high'] = [a.item() for a in SID_CPDAG(B_true, B_est)]
                    acc['auc'], _ = precision_recall(B_true, B_est)
            except ValueError:
                print('!!!!! Not a DAG !!!!!')
                acc = {'fdr': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'shd': np.nan, 'nnz': np.nan, 'p_shd': np.nan}
            except AttributeError:
                print('no output from method')
                acc = {'fdr': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'shd': np.nan, 'nnz': np.nan, 'p_shd': np.nan}
            acc['time'] = elapsed
            acc['model'] = method

            if args.debug:
                print(B_true)
                print(B_est)
                print("True edges:", {(f'X{a+1}',f'X{b+1}') for a,b in set(nx.DiGraph(B_true.T).edges)})            
                print("Est edges:", {(f'X{a+1}',f'X{b+1}') for a,b in set(nx.DiGraph(B_est.T).edges)})            
                print(acc)

            ##---------DUMP RESULTS--------------
            if args.dump_results:
                if run_exists == None and acc.get('shd', None) is None:
                    continue
                else:
                    out_file = os.path.join(args.output_dir, args.run_name, f'{args.expt_name}_{"".join([str(a) for a in time.gmtime()[0:6]])}.pkl')
                    if os.path.isfile(out_file) and not args.overwrite_res:
                        print('Result file exists, renaming...')
                        out_file = os.path.join(args.output_dir, args.run_name, f'{args.expt_name}_{int(time.time()*10000)}.pkl')
                    save_pickle({**args.__dict__, **acc, 'n_edges':s0}, out_file, verbose=False)
                    W_out_file = out_file.replace('.pkl', '_W.csv').replace(f'/{args.run_name}/', f'/{args.run_name}/est_matrices/')
                    if W_est is None:
                        W_est = np.zeros((args.n_nodes, args.n_nodes))
                    np.savetxt(W_out_file, W_est, delimiter=",")
                    B_out_file = W_out_file.replace('_W', '_B')
                    np.savetxt(B_out_file, B_true, delimiter=",")
            if args.wandb:
                run.finish()  
            i=i+1
        try:
            del fitted, W_est, acc, args, B_est, B_true, W_true, X, scaler
            gc.collect()
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            jm.stop_vm()
            del jm
        except:
            pass

if __name__ == '__main__':
    main()