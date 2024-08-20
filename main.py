import sys
import os
import yaml
import argparse
import time
import numpy as np
import networkx as nx
import pandas as pd
import torch
import wandb
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import ParameterGrid
import gc
gc.set_threshold(0,0,0)
from utils.helpers import random_stability, logger_setup, save_pickle, load_args, existing_runs_pkl, check_existing_runs, time_limit
from utils.data_utils import bnlearn_nodes_map, bnlearn_arcs_map, load_bnlearn_data_dag, simulate_dag, simulate_parameter, simulate_linear_sem, simulate_nonlinear_sem, simulate_discrete_data
from utils.graph_utils import DAGMetrics, dag2cpdag, get_immoralities, mount_adjacency_list, clgraph2adj
from models import run_method


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
    if not os.path.isdir(os.path.join(args.output_dir, args.data_dir, "DAGs")):
        os.makedirs(os.path.join(args.output_dir, args.data_dir, "DAGs"))

    ##---------DUMP ARGS--------------
    if args.dump_args:   
        # Dump args
        arg_file = os.path.join(args.output_dir, args.run_name, "args", f'{args.expt_name}.yaml')
        if os.path.isfile(arg_file) and not args.overwrite_args:
            logging.info('Arg file exists, renaming...')
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
    i = 1
    # perf = defaultdict(list)
    for params in grid:
        
        ##---------LOAD and PRINT PARAMS--------------
        args = argparse.Namespace(**params,**other_args)
        if i < args.start_iter:
            logging.info(f'Skipping iteration {i}/{all_iter}')
            i += len(method_list)
            if i > args.start_iter:
                i = args.start_iter
            else:
                continue
        
        logger_setup(f'{args.output_dir}/{args.run_name}/log_{args.expt_name}.log', continue_logging=True)
        if args.wandb:
            import wandb
            # reset_wandb_env()
            run = wandb.init(project=args.wdb_proj_name, config=args, 
                            reinit=True, group=args.method, job_type=f'{args.expt_name}_{args.min_abs_unif}_{args.max_abs_unif}_{args.noise_scale}_{args.noise_scale_vec}',
                            name=f'{args.method}_{args.s}_{args.sem_type}_{args.n_nodes}_{args.edge_per_node}_{args.graph_type}_{args.seed}')

        if 'n_nodes' in args:
            s0 = int(args.n_nodes*args.edge_per_node)
            if s0 > int(args.n_nodes*(args.n_nodes-1)/2):
                logging.debug(f'{s0} is too many edges, setting s0 to the max:', int(args.n_nodes*(args.n_nodes-1)/2))
                s0 = int(args.n_nodes*(args.n_nodes-1)/2)
        elif args.sem_type in bnlearn_nodes_map.keys():
            args.n_nodes = bnlearn_nodes_map[args.sem_type]
            n_edges = bnlearn_arcs_map[args.sem_type]
            s0 = round(n_edges/args.n_nodes)
            args.edge_per_node = s0

        ##---------CHECK IF DATA IS AVAILABLE --------------
        if 'real' in args.graph_type:
            B_file = os.path.join(args.output_dir, args.data_dir, "DAGs", f'{args.graph_type}_{args.sem_type}_{args.n_nodes}_{args.edge_per_node}_{args.seed}.npy')
        else:
            B_file = os.path.join(args.output_dir, args.data_dir, "DAGs", f'{args.graph_type}_{args.n_nodes}_{args.edge_per_node}_{args.max_degree}_{args.seed}.npy')

        if os.path.isfile(B_file) and not args.overwrite_data:
            logging.info('DAG exists, loading...')
            B_true = np.load(B_file)
        else:
            ##---------RANDOM DAG-------------- 
            if 'toy' not in args.graph_type and 'real' not in args.graph_type:
                random_stability(args.seed)
                B_true = simulate_dag(d=args.n_nodes, s0=s0, graph_type=args.graph_type, max_degree=args.max_degree)
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
            elif args.graph_type == 'real':
                B_true = None ### we load the data later
            else:
                raise NotImplementedError(f'{args.graph_type} not implemented')

        vars_to_outname = [args.graph_type,args.sem_type,args.n_nodes,args.edge_per_node,args.s,args.seed]
        if 'real' not in args.graph_type:
            vars_to_outname = vars_to_outname + [args.w_ranges, args.noise_scale, args.noise_scale_vec]
        if 'pct_weak' in args:
            vars_to_outname = vars_to_outname + [args.pct_weak]
        else:
            args.pct_weak = 0.0
        data_file = os.path.join(args.output_dir, args.data_dir, "_".join([str(it) for it in vars_to_outname])+'.npy')
        if os.path.isfile(data_file) and not args.overwrite_data:
            logging.info('Data exists, loading...')
            X = np.load(data_file)
            if 'real' not in args.graph_type:
                W_true = np.load(data_file.replace('.npy', '_W.npy'))
        else:
            logging.info("Generating data...")
            n = int(args.s*args.n_nodes)
            if 'real' not in args.graph_type:
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
                W_true = simulate_parameter(B_true, w_ranges=w_ranges, pct_weak=args.pct_weak)

            ##---------DGP--------------
            if args.sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
                random_stability(args.seed)
                X = simulate_linear_sem(W_true, n, args.sem_type, noise_scale=noise_scale)
            elif args.sem_type in ['mlp', 'mim', 'gp', 'gp-add']: 
                random_stability(args.seed)
                X = simulate_nonlinear_sem(B_true, n, args.sem_type, noise_scale=noise_scale)
            elif args.sem_type in bnlearn_nodes_map.keys():
                X, B_true = load_bnlearn_data_dag(args.sem_type, 'datasets', args.s, seed=args.seed, standardise=args.standardise, print_info=True)
            elif args.sem_type in ['discrete', 'BN']:
                G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
                logging.debug(G_true.edges)
                truth_DAG_directed_edges = set([(int(e[0].replace("X",""))-1,int(e[1].replace("X",""))-1)for e in G_true.edges])
                X = simulate_discrete_data(args.n_nodes, n, truth_DAG_directed_edges, w_ranges[1], args.seed)
            else:
                raise NotImplementedError(f'{args.sem_type} not implemented')

        if args.dump_data:
            if os.path.isfile(data_file) and not args.overwrite_data:
                logging.info('Data exists, skipping save...')
            else:
                np.save(data_file, X)
                np.save(B_file, B_true)
                if 'real' not in args.graph_type:
                    np.save(data_file.replace('.npy', '_W.npy'), W_true)

        if args.debug:
            logging.debug(B_true)
            logging.debug("True edges:", {(f'X{a+1}',f'X{b+1}') for a,b in set(nx.DiGraph(B_true.T).edges)})
            logging.debug("True immoralities:", get_immoralities(mount_adjacency_list(B_true)))
        logging.info(f"N true edges: {B_true.sum()}")
        logging.info(f"N True immoralities: {len(get_immoralities(mount_adjacency_list(B_true)))}")


        ##---------PREPROCESSING--------------
        if 'real' not in args.graph_type:
            if args.standardise:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            elif type(X) == pd.DataFrame:
                X = X.values
            elif type(X) == np.ndarray:
                X = X
            else:
                raise NotImplementedError(f'Data of {type(X)} not implemented, use pd.DataFrame or np.ndarray')

        # logging.info("Varsortability =", varsortability(X, B_true, debug=False))
        
        for method in method_list:
            logging.info(f'{((i/all_iter)*100):.1f}% complete - Running {i}/{all_iter}: {args.graph_type} {args.n_nodes} {args.edge_per_node} {args.sem_type} {args.s}')
            logging.debug(f'with params: {params}')

            ##---------CHECK IF RUN EXISTS--------------
            check_list = ['method','model','seed','n_nodes','graph_type','sem_type','edge_per_node','s',
                      'noise_scale','noise_scale_vec','w_ranges', 'pct_weak', 'standardise', 'max_degree']
            if 'pc' in method:
                check_list.append('test_alpha')
                check_list.append('test_name')
                check_list.append('priority')
            if 'spc' in method:
                check_list.append('selection')
                check_list.append('extra_tests')

            run_exists = check_existing_runs({**args.__dict__, 'model':method}, existing_runs_list, check_list=check_list, debug=args.debug)
            # if run_exists == None:
            #     logging.info('Run is Nan, rerunning...')
                # continue
            if run_exists and not args.overwrite_res:
                logging.info('Run exists, skipping...')
                i += 1
                continue

            ##---------MODELS--------------
            try:
                if 'time_limit' in args:
                    time_limit(args.time_limit)
                else:
                    time_limit(900) ## 15' time limit
                W_est, elapsed = run_method(X, method, args.seed, debug=args.debug, device=args.device,
                                            selection=args.selection, priority=args.priority, test_name=args.test_name, test_alpha=args.test_alpha,
                                            extra_tests=args.extra_tests)
                time_limit(0) ## reset time limit
            except Exception as e:
                logging.info(f'Error: {e}')
                W_est = None
                elapsed = np.nan

            ##---------EVALUATION--------------
            B_est = np.empty((args.n_nodes,args.n_nodes,))
            B_est[:] = np.nan
            CPB_est = np.empty((args.n_nodes,args.n_nodes,))
            CPB_est[:] = np.nan
            if W_est is not None:
                if 'Tensor' in str(type(W_est)):
                    W_est = np.asarray([list(t) for t in W_est])
                B_est = (W_est > 0).astype(int)
            try:
                CPB_est = dag2cpdag(clgraph2adj(W_est))
                mt_cpdag = DAGMetrics(CPB_est, B_true).metrics
                mt_dag = DAGMetrics(B_est, B_true).metrics
                mt_dag['p_shd'] = mt_dag['shd']/B_true.sum()
                mt_cpdag['p_shd'] = mt_cpdag['shd']/B_true.sum()

            except (ValueError, AssertionError) as e:
                logging.info(f'!!!!! Not a DAG !!!!!, Error: {e}')
                mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                mt_cpdag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
            except AttributeError as e:
                logging.info(f'no output from method, Error: {e}')
                mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                mt_cpdag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
            
            mt_cpdag['time'] = elapsed
            mt_cpdag['model'] = method
            mt_dag['time'] = elapsed
            mt_dag['model'] = method
            if type(mt_cpdag['sid'])==tuple:
                mt_sid_low = mt_cpdag['sid'][0]
                mt_sid_high = mt_cpdag['sid'][1]
            else:
                mt_sid_low = mt_cpdag['sid']
                mt_sid_high = mt_cpdag['sid']
            mt_cpdag.pop('sid')
            mt_cpdag['sid_low'] = mt_sid_low
            mt_cpdag['sid_high'] = mt_sid_high

            logging.debug(mt_dag)
            logging.debug(mt_cpdag)

            ##---------DUMP RESULTS--------------
            if args.dump_results:
                if run_exists == None and mt_dag.get('shd', None) is None:
                    i += 1
                    continue
                else:
                    out_file = os.path.join(args.output_dir, args.run_name, f'{args.expt_name}_{"".join([str(a) for a in time.gmtime()[0:6]])}.pkl')
                    if os.path.isfile(out_file) and not args.overwrite_res:
                        logging.info('Result file exists, renaming...')
                        out_file = os.path.join(args.output_dir, args.run_name, f'{args.expt_name}_{int(time.time()*10000)}.pkl')
                    out_file_cp = out_file.replace('.pkl', '_cp.pkl')
                    save_pickle({**args.__dict__, **mt_dag, 'n_edges':s0}, out_file, verbose=False)
                    save_pickle({**args.__dict__, **mt_cpdag, 'n_edges':s0}, out_file_cp, verbose=False)
                    W_out_file = out_file.replace('.pkl', '_W.csv').replace(f'/{args.run_name}/', f'/{args.run_name}/est_matrices/')
                    if W_est is None:
                        W_est = np.zeros((args.n_nodes, args.n_nodes))
                    np.savetxt(W_out_file, W_est, delimiter=",")
                    B_out_file = W_out_file.replace('_W', '_B')
                    CPB_out_file = W_out_file.replace('_W', '_CPB')
                    np.savetxt(B_out_file, B_est, delimiter=",")
                    np.savetxt(CPB_out_file, CPB_est, delimiter=",")
            if args.wandb:
                run.finish()  
            i += 1
        try:
            del W_est, args, B_est, B_true, W_true, X, scaler
            gc.collect()
            torch.cuda.empty_cache()
            # logging.info(torch.cuda.memory_summary(device=None, abbreviated=False))
        except:
            pass

if __name__ == '__main__':
    # sys.argv.append('test')
    main()