# create logger
import logging
import os, sys, re
from datetime import datetime
import time
import numpy as np
import pandas as pd
from itertools import chain, combinations
import pickle
import yaml
import argparse
import signal

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def break_point(signum, frame):
    raise Exception(f"Time over!")

signal.signal(signal.SIGALRM, break_point)

def time_limit(seconds:int=1800):
    signal.alarm(seconds)
    

def logger_setup(output_file:str="", continue_logging=False):
    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if output_file == "":
        output_file = f'.temp/{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    elif ".log" not in output_file: ## when one passes only the name of the file
        output_file = f'.temp/{output_file}.log'

    file_mode = 'a' if continue_logging else 'w'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename= output_file,
                        filemode=file_mode, force=True)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-8s %(module)-12s - %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    # format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    if len(logging.getLogger('').handlers) < 2:
        logging.getLogger('').addHandler(console)

# @profile
def get_freer_gpu():
    import torch
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available)>0:
        return np.argmax(memory_available)
    elif torch.cuda.is_available():
        return 0

def random_stability(seed_value=0, deterministic=True, verbose=False):
    '''
        seed_value : int A random seed
        deterministic : negatively effect performance making (parallel) operations deterministic
    '''
    if verbose:
        print('Random seed {} set for:'.format(seed_value))
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if verbose:
            print(' - PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        if verbose:
            print(' - random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        if verbose:
            print(' - NumPy')
    except:
        pass
    # try:
    #     import torch
    #     torch.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     if verbose:
    #         print(' - PyTorch')
    #     if deterministic:
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #         if verbose:
    #             print('   -> deterministic')
    # except:
        pass

def append_value(array, i, j, value):
    """
    Append value to the list at array[i, j]
    """
    if array[i, j] is None:
        array[i, j] = [value]
    elif value in array[i, j]:
        pass
    else:
        array[i, j].append(value)

def powerset(L):
    """
    Return the powerset of L (list)
    """
    s = list(L)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def save_pickle(obj, filename, verbose=True):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=-1)
        if verbose:
            print(f'Dumped PICKLE to {filename}')

def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        if verbose:
            print(f'Loaded PICKLE from {filename}')
        return obj

def load_args():
    """ Load args and run some basic checks.
        Args loaded from:
        - Manual args from .yaml file
    """
    # assert sys.argv[1] in ['train', 'test', 'chen']
    # Load args from file
    with open(f'config/{sys.argv[1]}.yaml', 'r') as f:
    # with open(f'config/ER124_d10_testn_memorytest.yaml', 'r') as f:
        return argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))


def existing_runs_pkl(output_dir, run_name, expt_name):
    """ Create a list of existing runs in the output directory.
    """
    existing_runs = []
    for file in os.listdir(os.path.join(output_dir, run_name)):
        if file.endswith(".pkl") and file.startswith(f'{expt_name}'):
            filepath = os.path.join(output_dir, run_name, file)
            existing_runs.append(load_pickle(filepath, verbose=False))
    return existing_runs

def check_existing_runs(args, existing_runs_list, check_list, debug=False):
    """ Check if the current run exists in the output directory.
    """
    if check_list is None:
        check_list = ['method','model','seed','n_nodes','graph_type','sem_type','edge_per_node','s', 'pct_weak',
                      'noise_scale','noise_scale_vec','w_ranges', 'test_alpha', 'test_name', 'priority', 'selection']
    # existing_runs_list = existing_runs_pkl(args.output_dir, args.run_name, args.expt_name)   
    if isinstance(args, argparse.Namespace):
        n_existing = sum([all((d.get(k) == v for k, v in args.__dict__.items() if k in check_list)) for d in existing_runs_list])
        if debug:
            print(f'Found {n_existing} existing runs')
            print([d for d in existing_runs_list if all((d.get(k) == v for k, v in args.__dict__.items() if k in check_list))] )
        if n_existing>0:
            shd = [d['shd'] for d in existing_runs_list if all((d.get(k) == v for k, v in args.__dict__.items() if k in check_list))][0]
            if np.isnan(shd):
                # print(f'SHD={shd}. Found NaNs')
                return None
            else:
                return n_existing>0
    elif isinstance(args, dict):
        n_existing = sum([all((d.get(k) == v for k, v in args.items() if k in check_list)) for d in existing_runs_list])
        if debug:
            print(f'Found {n_existing} existing runs')
            print([d for d in existing_runs_list if all((d.get(k) == v for k, v in args.items() if k in check_list))] )
        if n_existing>0:
            shd = [d['shd'] for d in existing_runs_list if all((d.get(k) == v for k, v in args.items() if k in check_list))][0]
            if np.isnan(shd):
                # print(f'SHD={shd}. Found NaNs')
                return None
            else:
                return n_existing>0
            

def load_results(results_file_name="stored_results_2024811112612", load_from_stored_results=True, save_res=False, 
                 results_folder=None, expt_name=None, run_name=None, spc_runs=None, debug=False, print_detail=False,
                 cp=True):
    """ Load results from stored results or from runs folder.
    """
    old_runs = False
    if load_from_stored_results:
        res_df = pd.read_pickle(os.path.join("../results/",f'{results_file_name}.pkl'))
        res_nan_df = pd.read_pickle(os.path.join("../results/",f'{results_file_name}_nan.pkl'))
        print(f"Loaded results from {results_file_name}")
    else:
        if old_runs:
            results_folder = ['../runs/','../runs/shappc/']
            expt_name = '(ER124_d51020_allsem|ER124_d51020_nl_gp|ER124_d20_w205|ER124_d51020_nl_w205|ER124_d51020_nl_mlp|ER124_d51020_allbutgp|ER124_d51020_lin|ER124_d51020_nl|ER124_d51020_lin|ER124_d102050_allsem)'
            run_name = '(onlytop_allsem_sel_001|onlytop_gp_pc_sel_001|only_top_sv|onlytop_lin4_cam|onlytop_mlp104_cam|onlytop_gp_pc_sel_500|onlytop_gp_nt|onlytop_gp_pcmax_500|\
                        |onlytop_gp_pcmax|onlytop_lin_1000|ER124_d51020_cam_50|ER124_d51020_nt_50_1|ER124_d51020_nt_50_2|ER124_d51020_nt_50_4|nt_gp_50|onlybot_lin_spc|onlytop_nl_gd|onlytop_lin_gd|\
                            |onlybot_nonlin_spc|onlybot_50_spc|onlytop_lin4_ges|onlytop_nl_ges|onlytop_allsem_mcsl_50|allbgp_50_pcmax|onlytop_lin_fgs|onlytop_nl_fgs|spc_nsvf_allbutgp|spc_nsvf_gp)'
            spc_runs = ["spc_meek_allbutgp","spc_meek_gp"]
        else:
            if results_folder is None:
                results_folder = ['../runs/']
            if expt_name is None:
                expt_name = '(ER124_d1020_linear005|SF124_d1020_linear005|ER124_d50_linear005|SF124_d50_linear005|ER124_d1020_linear01|SF124_d1020_linear01|ER124_d50_linear01|SF124_d50_linear01|ER124_d1020_linear001|SF124_d1020_linear001|ER124_d50_linear001|SF124_d50_linear001|SF124_d50_linear005_opt|ER124_d10_linear01|SF124_d10_linear01|ER124_d20_linear005|SF124_d20_linear005)' 
            if run_name is None:
                run_name = '(all_pc_linear_md|all_pc_linear_md_opt|all_pc_linear_wl|all_pc_linear_wl03|all_pc_linear_wl005)'
            if spc_runs is None:
                spc_runs = ["spc_negsiv_allbutgp_1020_v2","spc_negsiv_gp_1020_v2","spc_negsiv_allbutgp_50_v2","spc_negsiv_gp_50_v2"]
            if spc_runs == 'all':
                spc_runs = ["all_pc_linear_md_opt", "all_pc_linear_wl03", "all_pc_linear_wl005", "all_pc_linear_wl"]
                # spc_runs = run_name.replace('(','').replace(')','').split('|')

        if len(results_folder) > 1:
            for folder in results_folder:
                print(f"Folder:{folder} <- {os.listdir(folder)}")
        else:
            print(os.listdir(results_folder[0]))

        if len(results_folder) >= 1:
            v = '[0-9]+',
            res = []
            res_nan = []
            count_nans = 0
            count_0 = 0
            count_excluded = 0
            for folder in results_folder:
                print(folder)
                for subfolder in os.listdir(folder):
                    print(subfolder)
                    count = 0
                    if subfolder in run_name.replace('(','').replace(')','').split('|'):
                        for filename in os.listdir(os.path.join(folder, subfolder)):
                            if cp:
                                match = re.search(f'^{expt_name}_[0-9]+_cp.pkl$', filename)
                            else:
                                match = re.search(f'^{expt_name}_[0-9]+.pkl$', filename)
                            if match != None:
                                count += 1
                                count_0 += 1
                                if debug:
                                    print(filename)
                                    print(count_0, count_excluded)
                                report = load_pickle(os.path.join(folder, subfolder, filename), verbose=False)
                                # if report['n_nodes'] == 5 or report['sem_type']=="gp'":
                                #     count_excluded += 1
                                #     continue
                                if report['model'] == 'spc' and report['run_name'] not in spc_runs:
                                    count_excluded += 1
                                    continue
                                if np.isnan(report['shd']):
                                    count_nans += 1
                                    res_nan.append(report)
                                res.append(report)
                                # print(res)
                        print(f'  Found {count} (total:{count_0}) files for {expt_name} in {subfolder}, {count_nans} nans, {count_excluded} excluded')
                        
        res_df = pd.DataFrame(res)
        res_nan_df = pd.DataFrame(res_nan)

        if save_res:
            results_name = "stored_results_{}".format("".join([str(a) for a in time.gmtime()[0:6]]))
            res_file = os.path.join("../results/",f"{results_name}.pkl")
            res_df.to_pickle(res_file)
            res_nan_df.to_pickle(res_file.replace('.pkl', '_nan.pkl'))
            print(f"Saved results to {res_file}")

    if print_detail:
        print('Total', res_df.shape)
        print(res_df.groupby(['model']).p_shd.agg(['count']))

    # res_df = res_df[(res_df['graph_type'].isin(['ER']))]
    # res_df = res_df[(res_df['model'].isin(['cam','fgs','nt', 'mcsl','grandag'])) | (res_df['model'].isin(['spc','pc_max'])) & (res_df['test_name'].isin(['fisherz','kci']))  & (res_df['test_alpha'].isin([0.01,0.05,0.1]))]
    if print_detail:
        print('First selection:', res_df.shape)
        print(res_df.groupby(['model']).p_shd.agg(['count']))

    if print_detail:
        print('Drop prec:', res_df.shape)
    res_df.drop_duplicates(inplace=True)
    if print_detail:
        print('First dedup:', res_df.shape)

    # res_df[res_df[['model', 'n_nodes', 'edge_per_node','s','sem_type','seed']].duplicated(keep=False)].sort_values(['model', 'n_nodes', 'edge_per_node','s','sem_type','seed'])

    try:
        res_dedup = res_df.drop_duplicates(subset=['model', 'n_nodes', 'edge_per_node','s','sem_type','seed','test_name','test_alpha','selection','noise_scale','graph_type','pct_weak'])
        if print_detail:
            print('Second dedup:', res_dedup.shape)
            print(res_dedup.groupby(['model']).p_shd.agg(['count']))
    except:
        res_dedup = res_df
        

    if old_runs:
        res_dedup = res_dedup[(res_dedup['model'].isin(['nt']))&((res_dedup['test_name']=='fisherz')|(res_dedup['selection']=='top')|(res_dedup['selection']==''))|(res_dedup['model'].isin(['pc_max','cam','fgs', 'mcsl', 'spc','grandag']))]
        if print_detail:
            print('nt dedup:', res_dedup.shape)
            print(res_dedup.groupby(['model']).p_shd.agg(['count']))

    print_detail2=False
    if print_detail2:
        for model in res_dedup.model.unique(): ##['mcsl']:
            print(res_dedup[res_dedup['model']==model].groupby(['model','graph_type','n_nodes','edge_per_node','sem_type','s','run_name','expt_name']).p_shd.agg(['count']))

    if len(res_nan_df) > 0:
        full_summary2 = res_dedup.query('s<=500 and sem_type !="logistic"').groupby(['model']).p_shd.agg(['count']).merge(res_nan_df.query('s<=500 and sem_type !="logistic"').groupby(['model']).time.agg(['count']), on='model', how='outer').replace(np.nan, 0)
        full_summary2.columns = ['count', 'count_nan']
        full_summary2['total'] = full_summary2['count'] + full_summary2['count_nan']
    
        if print_detail:
            print(full_summary2.reset_index())
            print("Columns Report")
            print(res_dedup.columns)
            print(res_nan_df.columns)
            print(res_dedup.columns.name==res_nan_df.columns.name)

        if len(res_dedup.columns) > len(res_nan_df.columns):
            for col in res_dedup.columns:
                if col not in res_nan_df.columns:
                    res_nan_df[col] = np.nan
    else:
        def df_empty(columns, dtypes, index=None):
            assert len(columns)==len(dtypes)
            df = pd.DataFrame(index=index)
            for c,d in zip(columns, dtypes):
                df[c] = pd.Series(dtype=d)
            return df

        res_nan_df = df_empty(res_dedup.columns, res_dedup.dtypes)

    if print_detail:
        df_to_plot = create_vars(res_dedup)
        df_to_plot_nan = create_vars(res_nan_df)
        # model_det = ['fgs','nt', 'cam', 'pc_max', 'mcsl', 'spc', 'grandag']
        model_det = df_to_plot.model.unique()

        def table_up_by_(df, df_nan, group_list, filter, agg_list=['mean','std','count']):
            print(filter)
            if len(df_nan.query(filter)) > 0:
                summary = df.query(filter).groupby(group_list).shd.agg(agg_list).round(2).merge(
                            df_nan.query(filter).groupby(group_list).time.agg(['count']).round(2), 
                            on=group_list, how='left').replace(np.nan, 0)
                summary['SHD'] = summary['mean'].astype(str) + ' +/-' + summary['std'].astype(str)
                summary = summary.drop(columns=['mean','std'])
                summary.columns = ['count','count_nan','SHD']
                summary['Total'] = (summary['count'] + summary['count_nan'])
                summary = summary.reset_index() 
                summary['TO_RUN'] = 40 - summary['Total'] if summary['sem_broad_type'].unique() == 'non-linear' else 40 - summary['Total']
                # summary.columns = [agg_list+['count_nan']]
            else:
                summary = df.query(filter).groupby(group_list).shd.agg(agg_list).round(2).reset_index()
                summary['SHD'] = summary['mean'].astype(str) + ' +/-' + summary['std'].astype(str)
                summary = summary.drop(columns=['mean','std'])
                summary['Total'] = summary['count']
                summary = summary.reset_index()
                summary['TO_RUN'] = 40 - summary['Total'] if summary['sem_broad_type'].unique() == 'non-linear' else 40 - summary['Total']
                

            return summary

        for model in model_det:# res_dedup.model.unique():
            group_list = ['model_test','sem_broad_type','sem_type','n_nodes','edge_per_node','graph_type']
            general_filter = "sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'mlp', 'mim', 'gp', 'gp-add'] and s<=500"
            for graph_type in df_to_plot.query(general_filter).graph_type.unique():
                for sem_type in df_to_plot.query(general_filter).sem_broad_type.unique():
                    filter = f"sem_broad_type=='{sem_type}' and model == '{model}' and graph_type == '{graph_type}'"
                    print(table_up_by_(df_to_plot.query(general_filter), df_to_plot_nan, group_list, filter)[['model_test','n_nodes','edge_per_node','sem_type','Total','TO_RUN']])

    return res_df, res_nan_df, res_dedup


###Create additional variables
def create_vars(df, sid=False, impute_nan=0, mix_alphas=True, debug=False):
    df_to_plot = df.copy()
    df_to_plot['p_shd'] = df_to_plot['shd']/df_to_plot['n_edges']
    if sid:
        df_to_plot['p_SID'] = df_to_plot['SID']/df_to_plot['n_edges']
    df_to_plot['sparsity'] = round(df_to_plot['n_edges']/(df_to_plot['n_nodes']*(df_to_plot['n_nodes']-1)/2),1)
    ### bin sparsity into 3 categories: <=0.1, 0.1-0.4, >=0.4
    df_to_plot['sparse_g'] = pd.cut(df_to_plot['sparsity'], bins=[0,0.1,0.3,1], include_lowest=True)
    df_to_plot['dset_size'] = round(df_to_plot['n_nodes']*df_to_plot['s'],0)
    df_to_plot['sem_broad_type'] = ['linear' if sem in ['gauss','exp','gumbel','uniform','logistic'] else 'non-linear' for sem in df_to_plot['sem_type'] ]
    cols = ['model', 'test_alpha']
    df_to_plot['model_test'] = df_to_plot[cols].apply(lambda row: '_'.join(row.values.astype(str)).replace(".",""),axis=1) #if row['model']=='spc' else row['model'], axis=1)

    ### Choose the results based on alpha - 0.1 for n_nodes=10, 0.05 for n_nodes=20 and 0.01 for n_nodes=50
    df_to_plot2 = df_to_plot.copy()
    if mix_alphas:
        conditions = [df_to_plot2['n_nodes']==10, df_to_plot2['n_nodes']==20, df_to_plot2['n_nodes']==50]
        choices = [0.1, 0.05, 0.01]
        df_to_plot2['test_alpha2'] = np.select(conditions, choices, default=0.05)
        df_to_plot2['test_alpha2'].value_counts()
        ### filter the results based on the test_alpha2
        df_to_plot2 = df_to_plot2[df_to_plot2['test_alpha']==df_to_plot2['test_alpha2']]

    ### Set all nans to 0 for the metrics
    if impute_nan==0:
        metrics = ['pct_weak', 'immoral_UT_F1', 'immoral_UT_prec', 'immoral_UT_rec', 'arrF1', 'tpr', 'hhr', 'skF1', 'skp', 'skr']
        for m in metrics:
            df_to_plot2[m] = df_to_plot2[m].fillna(0)

    if debug:
        ## pivot table to check the aggregated results
        display(round(df_to_plot2.pivot_table(index=['graph_type','n_nodes'], columns='test_alpha2', values='model', aggfunc='count'),2))

        ## pivot table to check the aggregated results for arrF1 and immmoral_UT_F1
        display(round(df_to_plot2.pivot_table(index=['graph_type','n_nodes','edge_per_node'], columns='model', values='immoral_UT_F1', aggfunc='mean'),2))

        ### count nan for each model
        failed_runs_by_model = df_to_plot2[df_to_plot2['immoral_UT_F1'].isna()]
        failed_runs_by_model['immoral_UT_F1'] = failed_runs_by_model['immoral_UT_F1'].fillna(1)
        display(failed_runs_by_model.pivot_table(index=['graph_type','n_nodes','pct_weak'], columns='model', values='immoral_UT_F1', aggfunc='sum'))

        ### count time over for each model
        failed_runs_by_model = df_to_plot2[df_to_plot2['time'].isna()]
        failed_runs_by_model['time'] = failed_runs_by_model['time'].fillna(1)
        display(failed_runs_by_model.pivot_table(index=['graph_type','n_nodes','pct_weak'], columns='model', values='time', aggfunc='sum'))

    return df_to_plot2