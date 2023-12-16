import pickle
import yaml
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import torch
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cdt
from cdt.metrics import SHD, SID, SHD_CPDAG, SID_CPDAG, precision_recall
cdt.SETTINGS.rpath = '/vol/bitbucket/fr920/R/R-4.1.2/bin/Rscript'


# sys.path.append('../../notears/')
# git config --global url."https://".insteadOf git://
# git config --global url.https://github.com/.insteadOf git://github.com/

try:
    from notears import utils as nt_utils
except:
    print('Notears not installed, trying your local version')
    sys.path.append('../../notears/')

def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        if verbose:
            print(f'Loaded PICKLE from {filename}')
        return obj

def save_pickle(obj, filename, verbose=True):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=-1)
        if verbose:
            print(f'Dumped PICKLE to {filename}')
            
def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

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
    #     import tensorflow as tf
    #     try:
    #         tf.set_random_seed(seed_value)
    #     except:
    #         tf.random.set_seed(seed_value)
    #     if verbose:
    #         print(' - TensorFlow')
    #     from keras import backend as K
    #     if deterministic:
    #         session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #     else:
    #         session_conf = tf.ConfigProto()
    #     session_conf.gpu_options.allow_growth = True
    #     sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #     K.set_session(sess)
    #     if verbose:
    #         print(' - Keras')
    #     if deterministic:
    #         if verbose:
    #             print('   -> deterministic')
    # except:
    #     pass
    try:
        import torch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        if verbose:
            print(' - PyTorch')
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if verbose:
                print('   -> deterministic')
    except:
        pass

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

# @profile
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available)>0:
        return np.argmax(memory_available)
    elif torch.cuda.is_available():
        return 0

# @profile
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
        check_list = ['method','model','seed','n_nodes','graph_type','sem_type','edge_per_node','s',
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

#################### Plots ####################

#################### Define names and parameters ####################
##Colors
main_gray = '#262626'
sec_gray = '#595959'
main_blue = '#005383'
sec_blue = '#0085CA'
main_green = '#379f9f' 
sec_green = '#196363' 
main_purple='#9454c4'
sec_purple='#441469'
main_orange='#8a4500'
sec_orange='#b85c00'


#################### Boxplot ####################

def boxplot_by(df1, x, y, general_filter,
                        names_dict, symbols_dict, colors_dict, share_y=False, y_range=[0,5],
                        save_figs=False, output_name="boxplot.html", debug=False, font_size=20):
    df=df1.query(general_filter)
    fig = go.Figure()
    df.sort_values(by=[x,'model_test'],axis=0, inplace=True)
    if x=='sparsity':
        df.loc[df[x]==0,x] = '<0.1'
    for sem in ['linear','non-linear']:
        for model in df['model_test'].unique():
            line_w, opa = (2,0.8) if model=='spc_bot_kci_001' else (1,0.6)
            # print(model)
            fig.add_trace(go.Box(
                y = df[(df['model_test']==model)&(df['sem_broad_type']==sem)][y],
                x = df[(df['model_test']==model)&(df['sem_broad_type']==sem)][x].astype(str),
                name=names_dict[model],
                marker_color=colors_dict[model],
                alignmentgroup=sem,
                        boxmean='sd', # represent mean
                        boxpoints=False,#'outliers',
                        jitter=0.0, # add some jitter for a better separation between points
                            whiskerwidth=0.2,
                                    marker_size=2,
                                        line_width=line_w,
                                            opacity=opa,
                                                    showlegend=(sem!='linear')
                                                    ))
        if sem=='linear':
            ## Add random baseline
            rand_df = pd.DataFrame()
            for n_nodes in [10,20,50]:
                for edge_per_node in [1,2,4]:
                    unnormalised = y.replace('p_','').replace('n_','')
                    rand = pd.DataFrame(create_rand_baseline(n_nodes, edge_per_node, additional_metric=unnormalised))
                    rand[x] = round((edge_per_node*n_nodes)/(n_nodes*(n_nodes-1)/2),1)
                    if "p_" in y or "n_" in y:
                        unnormalised = y.replace('p_','').replace('n_','')
                        rand[y] = rand[unnormalised]/rand['nnz']
                    rand_df = pd.concat([rand_df,rand])
            if x=='sparsity':
                rand_df.loc[rand_df[x]==0,x] = '<0.1'
            trace_name = "True Graph Size" if y=='nnz' else 'Random'
            fig.add_trace(go.Box(
                y = rand_df[y],
                x = rand_df[x].astype(str),
                name=trace_name,
                marker_color='black',
                        boxmean='sd', # represent mean
                        boxpoints=False,#'outliers',
                        jitter=0.0, # add some jitter for a better separation between points
                            whiskerwidth=0.2,
                                    marker_size=2,
                                        line_width=1,
                                                opacity=0.6,
                                                    showlegend=(sem=='linear')
                                                    ))
            rand_sum = rand_df[[x,y]].groupby(x).agg(['mean', 'std']).reset_index()
            fig.add_trace(
                    go.Scatter(x=rand_sum[x], y=rand_sum[y]['mean'], name= 'Rand', mode='markers', marker_symbol='line-ew',
                            marker=dict(
                            color='grey',
                            size=175,
                            line=dict(
                                color='grey',
                                width=1)
                                ),
                            showlegend=False), 
                )

    # Add figure title
    fig.update_layout(
        # title='',
        legend={
            'y':-0.2,
            'x':0.5,
            'orientation':"h",
            'xanchor': 'center',
            'yanchor': 'bottom',
            'font_size': font_size,},
        font=dict(
            family="Serif",
            size=font_size,
            color="Black"
        ),
        template='plotly_white',
        autosize=False,
        boxmode='group',
        boxgroupgap=0.4, # update
        boxgap=0.1,
        width=1500, height=600, 
        margin=dict(
            l=10,
            r=10,
            b=0,
            t=10,
            pad=0
        )
        # ,hovermode='y unified',
        # font=dict(
        #     family='Serif',#"Courier New, monospace",
        #     size=font_size,
        #     # color="Black"
        # )    
    )        

    if x=='sparsity':
        fig.update_xaxes(showgrid=False,zeroline=False, title={'text':'Saturation = Number of Edges in DAG / Maximum DAG Edges','font':{'size':font_size}
        }#, nticks=13, secondary_y=True
        )
        if 'p_' in y:
            orig_y = y.replace('p_','').upper()
            fig.update_yaxes(range=y_range,title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':font_size}
            }#,, secondary_y=False
            )
        elif y=='shd_norm':
            fig.update_yaxes(title={'text':'Normalised SHD = SHD / Maximum Number of Edges','font':{'size':font_size}
            }#,, secondary_y=False
            ,
            )
        elif y=='nnz':
            fig.update_yaxes(title={'text':'EGS = Estimated Graph Size','font':{'size':font_size}
            }#,, secondary_y=False
            ,
            )
        else:
            fig.update_yaxes(title={'text':y.title(),'font':{'size':font_size}})
    else:
        fig.update_xaxes(showgrid=False,zeroline=False, title={'text':x,'font':{'size':font_size}
        }#, nticks=13, secondary_y=True
        )
        fig.update_yaxes(title={'text':y,'font':{'size':font_size}
        }#,, secondary_y=False
        )


    start_pos = 0.037
    intra_dis = 0.091
    inter_dis = 0.098
    list_of_pos = []
    left=start_pos
    for i in range(5):
            right = left+intra_dis
            list_of_pos.append((left, right))
            left = right+inter_dis

    lin_space=7
    nl_space=3

    for s1,s2 in list_of_pos:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s1,
            y=1.015,
                    text=f"{' '*lin_space}Linear{' '*(lin_space)}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s2,
            y=1.015,
                    text=f"{' '*(nl_space)}Non-Linear{' '*nl_space}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
        
    # changing the orientation to horizontal
    # fig.update_traces(orientation='h')

    if save_figs:
        fig.write_html(output_name)

        
    fig.show()

### Define Line Plots Layout
def fig_update_layout(fig, rows, cols, var_to_plot, font_size=20):

    for r in range(1,rows+1):
        this_yaxis = next(fig.select_yaxes(row = r, col = 1))
        this_yaxis.update(title=var_to_plot.upper(),title_standoff=0)    
        if var_to_plot == 'nnz':
            this_yaxis.update(title='EGS',title_standoff=0)
        elif var_to_plot in ['precision', 'recall']:
            this_yaxis.update(title=var_to_plot.title(),title_standoff=0)
    
    fig.update_xaxes(matches='x', showline=False)

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0.5,
        yanchor="bottom",
        y=-0.12,
        text=f"s= number of samples (N)/ number of nodes (|V|)",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=font_size,
                    color="Black"
                    )
        )

    x_ann = 1.18 if cols==6 else 1.3
    for r,d in zip([1,2,3],[10,20,50]):
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=x_ann,
            y=0.5,
            text=f"         |V|={d}         ",
            showarrow=False,    
            font=dict(
                    family="Serif",
                    size=font_size,
                    color="Black"
                    ),
            row=r,
            col=cols
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.9
        , textangle=90
        )

    if cols==6:
        width, height = 1300, 600
        top_annotations_cols = zip([1,3,5],[1,2,4])
        top_annotation_spaces = 36

        if var_to_plot=='shd':
                    
            for j in [1]:
                for i in [1,2]:
                    fig.update_yaxes(range=[0,30], row=j, col=i, zeroline=False, tick0=0)
                for i in [3,4]:
                    fig.update_yaxes(range=[0,40], row=j, col=i, zeroline=False, tick0=0)
                for i in [5,6]:
                    fig.update_yaxes(range=[0,50], row=j, col=i, dtick=10, zeroline=False, tick0=0)
            for j in [2]:
                for i in [1,2]:
                    fig.update_yaxes(range=[5,90], row=j, col=i, tick0=5
                                    # tickmode='array', #change 1
                                    # tickvals = x, #change 2
                                    # ticktext = [0,5,10,15,20,25], #change 3
                                    )
                for i in [3,4]:
                    fig.update_yaxes(range=[10,100], row=j, col=i,  dtick=20, tick0=10
                                        # dtick=25
                                        )
                for i in [5,6]:
                    fig.update_yaxes(range=[10,160], row=j, col=i, dtick=40, tick0=10)
            for j in [3]:
                for i in [1,2]:
                    fig.update_yaxes(range=[10,230], row=j, col=i, dtick=50, tick0=10)
                for i in [3,4]:
                    fig.update_yaxes(range=[50,350], row=j, col=i, dtick=100, tick0=50)
                for i in [5,6]:
                    fig.update_yaxes(range=[50,810], row=j, col=i, dtick=150, tick0=50)

        elif var_to_plot in ['fdr', 'tpr']:    
            fig.update_yaxes(matches='y', zeroline=False, #tick0=0, 
                            dtick=0.2, showgrid=True)
        
        elif var_to_plot in ['fpr']:
            fig.update_yaxes(zeroline=False, #tick0=0, dtick=0.2, 
                                showgrid=True)

        for i in range(2,rows+1):
            for j in [2,4,6]:
                fig.update_yaxes(row=i, col=j, showticklabels=False, showgrid=True)
        for j in [2,4,6]:
            fig.update_yaxes(row=1, col=j, showticklabels=False, showgrid=True)

        # fig.update_yaxes(matches='y', zeroline=False, tick0=0, dtick=0.25, showgrid=True)
    elif cols==12:
        width, height = 1800, 650
        top_annotations_cols = zip([1,5,9],[1,2,4])
        top_annotation_spaces = 52

        for i in range(2,rows+1):
            for j in [2,3,4,6,7,8,10,11,12]:
                fig.update_yaxes(visible=False, row=i, col=j)
        for j in [2,3,4,6,7,8,10,11,12]:
            fig.update_yaxes(visible=False, row=1, col=j, zeroline=True)   


    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=-0.1),
        template='plotly_white',
        # autosize=True,
        width=width, 
        height=height,
        margin=dict(
            l=40,
            r=40,
            b=10,
            t=70,
            # pad=10
        ),hovermode='x unified',
        font=dict(size=font_size, family="Serif", color="black")
        )

    if cols in [6,12]:
        for c,d in top_annotations_cols:
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=-0.1,
                y=1.4,
                text=f"{' '*top_annotation_spaces}d={d}{' '*top_annotation_spaces}",
                showarrow=False,    
                font=dict(
                    family="Serif",
                    size=font_size,
                    color="Black"
                    ),
                row=1,
                col=c
            , bordercolor='#E5ECF6'
            , borderwidth=2
            , bgcolor="#E5ECF6"
            , opacity=0.9
                    )
    fig.update_annotations(font_size=font_size)

def table_up_by(df1, var, group_list, filter, agg_list=['mean','std','count']):
    # print(filter)
    df = df1.copy()
    df['shd2'] = np.where(np.isnan(df['shd']),df['n_edges'],df['shd'])
    df['fdr2'] = np.where(np.isnan(df['fdr']),1,df['fdr'])
    df['tpr2'] = np.where(np.isnan(df['tpr']),0,df['tpr'])
    summary = df.query(filter).groupby(group_list)[var].agg(agg_list)
    return summary.reset_index()

def plot_runtime(df_to_plot, x_var_list, general_filter, names_dict, symbols_dict, colors_dict,
                         share_y=False, save_figs=False, output_name="runtime.html", debug=False, font_size=20):
    cols = 2
    rows = 1
    fig = make_subplots(rows, cols, vertical_spacing=0.05, horizontal_spacing=0.01, shared_yaxes=True, shared_xaxes=True,)
    i = 1
    j = 1
    var_to_plot = 'time'
    metric = 'median'

    for x_var in x_var_list:
        group_list = [x_var]
        for model in ['cam', 'fgs', 'mcsl', 'nt', 'pc_max', 'spc_bot_kci_001']:#df_to_plot['model_test'].unique():
            filter = f"model_test=='{model}'"
            # df_to_plot['n'] = [human_format(i) for i in df_to_plot['n_nodes']*df_to_plot['s']]
            df_to_plot['n'] =  df_to_plot['n_nodes']*df_to_plot['s']
            tab = table_up_by(df_to_plot.query(general_filter), var_to_plot, group_list, filter, agg_list=[metric])
            # print(tab)
            tab.reset_index().sort_values(by=[x_var], inplace=True)
            fig.add_trace(go.Scatter(x=tab[x_var].astype(str)
                                    ,y=tab[metric]
                                    ,name=names_dict[model], #legendgroup=f'group{i}{j}', 
                                    line=dict(color=colors_dict[model], width=2, simplify=True), mode='lines+markers', 
                                    marker=dict(size=8, symbol=symbols_dict[model], color=colors_dict[model]), 
                                    showlegend=(j==1 and i==1)), j, i)
        i += 1 if i < cols else 0

    fig.update_yaxes(matches='y', type="log")

    for r in range(1,rows+1):
        this_yaxis = next(fig.select_yaxes(row = r, col = 1))
        this_yaxis.update(title='log(elapsed time [s])',title_standoff=0)
    for c,n in enumerate(["Number of Nodes (|V|)",
                        "Proportional Sample Size (s=N/|V|)"
                        # "Sample Size (N)"
                        ]):
        this_xaxis = next(fig.select_xaxes(row = rows, col = c+1))
        this_xaxis.update(title=n,title_standoff=0)

    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=-0.22),
        template='plotly_white',
        # autosize=True,
        width=750, 
        height=300,
        margin=dict(
            l=10,
            r=10,
            b=80,
            t=10,
        ),font=dict(size=font_size, family="Serif", color="black")
                    )

    for dl in range(0,len(fig.data)):
        fig.data[dl].error_y.thickness = 1
    if save_figs:
        fig.write_html(output_name)
    fig.show()

### Define Line Plot Function
def plot_ly_s_n_d_sem(df_to_plot, var_to_plot, group_list, general_filter, sem_broad, 
                        names_dict, symbols_dict, colors_dict, share_y=False, save_figs=False, 
                            output_name="line_plot.html", debug=False):
    
    if sem_broad=='all':
        sem_list = ['linear','non-linear']
    elif sem_broad=='linear':
        sem_list = ['gauss','exp','gumbel','uniform']
    elif sem_broad=='non-linear':
        sem_list = ['mim','mlp','gp','gp-add']

    cols = len(sem_list)*3
    rows = 3

    if sem_broad=='all':
        fig = make_subplots(rows, cols, subplot_titles=(
        'Linear', 'Non-Linear', 'Linear', 'Non-Linear', 'Linear', 'Non-Linear'), 
        vertical_spacing=0.05, horizontal_spacing=0.02, shared_yaxes=share_y, shared_xaxes=True,)
    else:
        fig = make_subplots(rows, cols, subplot_titles=(sem_list*3), 
        vertical_spacing=0.03, horizontal_spacing=0.015, shared_yaxes=False, shared_xaxes=True,)

    i = 1
    j = 1
    
    for n_nodes in [10,20,50]:
        for edge_per_node in [1,2,4]:
            for sem_type in sem_list:
                for model in ['random', 'cam', 'fgs', 'mcsl', 'nt', 'pc_max', 'spc_bot_kci_001']:#df_to_plot['model_test'].unique():
                    filter = f"n_nodes=={n_nodes} and edge_per_node=={edge_per_node} and model_test=='{model}'"
                    if sem_broad=='all':
                        filter += f" and sem_broad_type=='{sem_type}'"
                    else:
                        filter += f" and sem_type=='{sem_type}'"
                    tab = table_up_by(df_to_plot.query(general_filter), var_to_plot, group_list, filter, agg_list=['mean','std'])
                    if debug:
                        print(tab)
                        print(filter)
                        print(j,i)
                    fig.add_trace(go.Scatter(x=tab['s'].astype(str)
                                            ,y=tab['mean']
                                            ,error_y=dict(
                                                type='data', # value of error bar given in data coordinates
                                                array=tab['std'],
                                                visible=True)
                                            ,name=names_dict[model], #legendgroup=f'group{i}{j}', 
                                            line=dict(color=colors_dict[model], width=2, simplify=True), mode='lines+markers', 
                                            marker=dict(size=8, symbol=symbols_dict[model], color=colors_dict[model]), 
                                            showlegend=(j==1 and i==1)), j, i)
                ## Add random baseline
                # additional = False if var_to_plot in ['shd', 'fpr', 'fdr', 'tpr', 'nnz'] else True
                rand = pd.DataFrame(create_rand_baseline(n_nodes, edge_per_node, additional_metric=var_to_plot))
                rand_sum = rand.agg(['mean', 'std']).reset_index()
                rand_name_baseline = names_dict['random'] if var_to_plot!='nnz' else 'True Graph Size'
                fig.add_trace(go.Scatter(x=tab['s'].astype(str)
                                        ,y=np.repeat(rand_sum[rand_sum['index']=='mean'][var_to_plot], len(tab['s']))   
                                        ,error_y=dict(
                                            type='data', # value of error bar given in data coordinates
                                            array=np.repeat(rand_sum[rand_sum['index']=='std'][var_to_plot], len(tab['s']))  ,
                                            visible=True)
                                        ,name=rand_name_baseline, #legendgroup=f'group{i}{j}', 
                                            line=dict(color=colors_dict['random'], width=2, simplify=True, dash='dash'), mode='lines+markers', 
                                            marker=dict(size=4, symbol='x-thin', color=colors_dict['random']),              
                                            showlegend=(j==1 and i==1)), j, i)   
                i += 1 if i < cols else 0
        i=1
        j += 1 if j < rows else 0
    
    fig_update_layout(fig, rows, cols, var_to_plot)
    for dl in range(0,len(fig.data)):
        fig.data[dl].error_y.thickness = 1

    if save_figs:
        fig.write_html(output_name)
    fig.show()   


def bar_chart_plotly(all_sum, var_to_plot, names_dict, colors_dict, save_figs=False, output_name="bar_chart.html", debug=False):
    fig = go.Figure()
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)']:
        trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method
        fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                opacity=0.6,
                            #  width=0.1
                                )
                                )
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.1, # gap between bars of the same location coordinate.)
            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1),
            template='plotly_white',
            # autosize=True,
            width=1600, 
            height=700,
            margin=dict(
                l=40,
                r=40,
                b=80,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=20, family="Serif", color="black")
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0,
        yanchor="bottom",
        y=-0.05,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=20,
                    color="Black"
                    )
        )

    if 'n_' in var_to_plot or 'p_' in var_to_plot:
        orig_y = var_to_plot.replace('n_','').replace('p_','').upper()
        fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':20}})
    elif var_to_plot=='nnz':
        orig_y = 'Number of Edges in DAG'
        fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':20}})
    else:
        fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':20}})

    if save_figs:
        fig.write_html(output_name)

    fig.show()


###Create additional variables
def create_vars(df):
    df_to_plot = df.copy()
    df_to_plot['p_shd'] = df_to_plot['shd']/df_to_plot['n_edges']
    df_to_plot['p_SID'] = df_to_plot['SID']/df_to_plot['n_edges']
    df_to_plot['sparsity'] = round(df_to_plot['n_edges']/(df_to_plot['n_nodes']*(df_to_plot['n_nodes']-1)/2),1)
    df_to_plot['dset_size'] = round(df_to_plot['n_nodes']*df_to_plot['s'],0)
    df_to_plot['sem_broad_type'] = ['linear' if sem in ['gauss','exp','gumbel','uniform','logistic'] else 'non-linear' for sem in df_to_plot['sem_type'] ]
    # df_to_plot['model_test'] = np.where((df_to_plot['test_name'] == 'kci')&(df_to_plot['model']=='spc'), 'spc_kci', df_to_plot['model']) 
    # 'top', 'top_change', 'median', 'top2'
    # create a list of our conditions
    conditions = [
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.1),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.1),
        
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.1),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top2') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.1),
                
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.1),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'median') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.1),

        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.1),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'top_change') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.1),

        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'kci') & (df_to_plot['test_alpha'] == 0.1),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.01),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.05),
        (df_to_plot['model'] == 'spc') & (df_to_plot['selection'] == 'bot') & (df_to_plot['test_name'] == 'fisherz') & (df_to_plot['test_alpha'] == 0.1),

                  ]

    # create a list of the values we want to assign for each condition
    values = [
        'spc_top_kci_001', 'spc_top_kci_005', 'spc_top_kci_01', 'spc_top_z_001', 'spc_top_z_005', 'spc_top_z_01',
        'spc_top2_kci_001', 'spc_top2_kci_005', 'spc_top2_kci_01', 'spc_top2_z_001', 'spc_top2_z_005', 'spc_top2_z_01',
        'spc_median_kci_001', 'spc_median_kci_005', 'spc_median_kci_01', 'spc_median_z_001', 'spc_median_z_005', 'spc_median_z_01',
        'spc_delta_kci_001', 'spc_delta_kci_005', 'spc_delta_kci_01', 'spc_delta_z_001', 'spc_delta_z_005', 'spc_delta_z_01',
        'spc_bot_kci_001', 'spc__kci_bot005', 'spc_bot_kci_01', 'spc_bot_z_001', 'spc_bot_z_005', 'spc_bot_z_01',
    ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_to_plot['model_test'] = np.select(conditions, values, default=df_to_plot['model'])

    # # display updated DataFrame
    # df_to_plot.head()

    return df_to_plot

def format_tab(table, debug=False):
    def compute_top_2(names):
        t = ''
        pcount = round(len(names)/len(np.unique(names)))
        _, idx = np.unique(names, return_index=True)
        names = names[np.sort(idx)]
        if debug:
            print(names)
        for i in range(len(names)):
            t += ' & \multicolumn{' + str(pcount) + '}{c}{' + str(names[i]) + '}'
        return t
                
    top1 = compute_top_2(table.columns.values) + '\\\\ \n\hline '

    print_table = lambda table, top : table.to_latex(escape = False, index = False, index_names = False, header =False,
                        formatters=[(lambda x : '\!' + str(x) + '\!') for col in table.columns.values],
                        column_format = 'r' + 'c' * (len(table.columns.values)),
                        ).replace(r'\toprule', top).replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline').replace('.0%', '%').replace('\!%', '\!0%').replace('%', '\%').replace('_',' ')
    print(print_table(table, top1))

#################### Load Results ####################

def load_results(results_file_name="stored_results_20231016213719", load_from_stored_results=True, save_res=False, results_folder=None, expt_name=None, run_name=None, spc_runs=None, debug=False, print_detail=False):
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
                            |onlybot_nonlin_spc|onlybot_50_spc|onlytop_lin4_ges|onlytop_nl_ges|onlytop_allsem_mcsl_50|allbgp_50_pcmax|onlytop_lin_fgs|onlytop_nl_fgs|spc_meek_allbutgp|spc_meek_gp)'
            spc_runs = ["spc_meek_allbutgp","spc_meek_gp"]
        else:
            results_folder = ['../runs/']
            expt_name = '(ER124_d102050_allsem)'
            run_name = '(spc_nsvf_allbutgp|spc_nsvf_gp|pcmax_nsvf_allbutgp|pcmax_nsvf_gp|fgs_nsvf_allbutgp|fgs_nsvf_gp|cam_nsvf_allbutgp|cam_nsvf_gp|nt_nsvfo_lin|nt_nsvfo_nl|mcsl_nsvfo_lin|mcsl_nsvfo_nl)'
            spc_runs = ["spc_nsvf_allbutgp","spc_nsvf_gp"]

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
                            match = re.search(f'^{expt_name}_[0-9]+.pkl$', filename)
                            if match != None:
                                count += 1
                                count_0 += 1
                                if debug:
                                    print(filename)
                                    print(count_0, count_excluded)
                                report = load_pickle(os.path.join(folder, subfolder, filename), verbose=False)
                                if report['n_nodes'] == 5 or report['sem_type']=="gp'":
                                    count_excluded += 1
                                    continue
                                if report['model'] == 'spc' and report['run_name'] not in spc_runs:
                                    count_excluded += 1
                                    continue
                                if np.isnan(report['shd']):
                                    count_nans += 1
                                    res_nan.append(report)
                                res.append(report)
                                # print(res)
                        print(f'Found {count} (total:{count_0}) files for {expt_name} in {subfolder}, {count_nans} nans, {count_excluded} excluded')
                        
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

    res_df = res_df[(res_df['graph_type'].isin(['ER']))]
    res_df = res_df[(res_df['model'].isin(['cam','fgs','nt', 'mcsl','grandag'])) | (res_df['model'].isin(['spc','pc_max'])) & (res_df['test_name'].isin(['fisherz','kci']))  & (res_df['test_alpha'].isin([0.01,0.05,0.1]))]
    if print_detail:
        print('First selection:', res_df.shape)
        print(res_df.groupby(['model']).p_shd.agg(['count']))

    if print_detail:
        print('Drop prec:', res_df.shape)
    res_df.drop_duplicates(inplace=True)
    if print_detail:
        print('First dedup:', res_df.shape)

    res_df[res_df[['model', 'n_nodes', 'edge_per_node','s','sem_type','seed']].duplicated(keep=False)].sort_values(['model', 'n_nodes', 'edge_per_node','s','sem_type','seed'])

    res_dedup = res_df.drop_duplicates(subset=['model', 'n_nodes', 'edge_per_node','s','sem_type','seed','test_name','test_alpha','selection'])
    if print_detail:
        print('Second dedup:', res_dedup.shape)
        print(res_dedup.groupby(['model']).p_shd.agg(['count']))

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
        model_det = ['fgs','nt', 'cam', 'pc_max', 'mcsl', 'spc', 'grandag']

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
            group_list = ['model_test','sem_broad_type','sem_type','n_nodes','edge_per_node']
            general_filter = "sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'mlp', 'mim', 'gp', 'gp-add'] and s<=500"
            for sem_type in df_to_plot.query(general_filter).sem_broad_type.unique():
                filter = f"sem_broad_type=='{sem_type}' and model == '{model}'"
                print(table_up_by_(df_to_plot.query(general_filter), df_to_plot_nan, group_list, filter)[['model_test','n_nodes','edge_per_node','sem_type','Total','TO_RUN']])

    return res_df, res_nan_df, res_dedup


def create_rand_baseline(V, d, true_path="../runs/generated_data/DAGs", seeds=[357,470,2743,4951,5088,5657,5852,6049,6659,9076], g_type='ER', B_true=None, additional_metric=''):
    """
    Create a baseline of random DAGs with the same characteristics as the true DAGs

    Args:
        V (_type_): _description_
        d (_type_): _description_
        true_path (_type_): _description_
        seeds (list, optional): _description_. Defaults to [357,470,2743,4951,5088,5657,5852,6049,6659,9076].
        g_type (str, optional): _description_. Defaults to 'ER'.
        B_true (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    rand_accs = []
    if seeds == None:
        seeds = range(10)
    for seed in seeds:
        if true_path == None:
            B_rand = nt_utils.simulate_dag(d=V, s0=d*V, graph_type=g_type)
        else:
            B_true = np.load(os.path.join(true_path, f'{g_type}_{V}_{d}_{seed}.npy'))
            B_rand = nt_utils.simulate_dag(d=V, s0=d*V, graph_type=g_type)
        acc = nt_utils.count_accuracy(B_true, B_rand)
        if additional_metric == "SID":
            acc[additional_metric] = SID(B_true, B_rand).flat[0]
        elif additional_metric == "SHD_CPDAG":
            acc[additional_metric] = SHD_CPDAG(B_true, B_rand)
        elif additional_metric==SID_CPDAG:
                acc['SID_CPDAG_low'], acc['SID_CPDAG_high'] = [a.flat[0] for a in SID_CPDAG(B_true, B_rand)]
        elif additional_metric == "precision_recall":
            acc[additional_metric], prre = precision_recall(B_true, B_rand)
            acc[additional_metric] = prre[0]
            acc[additional_metric] = prre[1]
        if additional_metric in ['precision', 'recall', 'F1', 'gscore']:
            add = MetricsDAG(B_rand, B_true)
            acc[additional_metric] = add.metrics[additional_metric]
        rand_accs.append(acc)
    return rand_accs

BIF_FOLDER_MAP = {
    'alarm': 'medium',
    'child': 'medium',
    'insurance': 'medium',
    'asia': 'small',
    'cancer': 'small',
    'sachs': 'small',
    'survey': 'small',
    'earthquake': 'small',
    'hailfinder': 'large',
    'hepar2': 'large'
}

def load_bn_from_BIF(main_data_path, data_folder='bayesian', dataset_name='child', seed=1, verbose=False):
    bif_file = os.path.join(main_data_path, data_folder, BIF_FOLDER_MAP[dataset_name], dataset_name + '.bif', dataset_name + '.bif')
    image_file = os.path.join(main_data_path, data_folder, BIF_FOLDER_MAP[dataset_name], dataset_name + '.png')

    if verbose:
        print(f'Loading graph from {bif_file}')

    random_stability(seed)
    reader = BIFReader(bif_file)
    model = reader.get_model()

    if os.path.exists(image_file):
        if verbose:
            print(f'Loading graph image from {image_file}')
        model.image = Image.open(image_file)

    # Take the leaves as features
    __FEATURES = model.get_leaves()
    __ROOTS = model.get_roots()
    __NODES = model.nodes()
    __NOT_FEATURES = list({node for node in __NODES if node not in __FEATURES and node not in __ROOTS})

    if verbose:
        print(f'Nodes: {__NODES} ({len(__NODES)})')
        print(f'Features/Leaves: {__FEATURES} ({len(__FEATURES)})')
        print(f'Roots: {__ROOTS} ({len(__ROOTS)})')
        print(f'Intermediate (non-roots/non-leaves): {__NOT_FEATURES} ({len(__NOT_FEATURES)})')

    return model

def load_bnlearn_data_dag(dataset_name, data_path, sample_size, seed=1, standardise=True, print_info=False):
    assert dataset_name in BIF_FOLDER_MAP.keys(), "dataset name not recognised"
    ##Load Bayesian Network
    random_stability(seed)
    bn = load_bn_from_BIF(main_data_path=data_path, dataset_name=dataset_name, seed=seed)
    ##Simulate data from BN
    df = bn.simulate(sample_size, seed=seed)
    ##Preprocess categorical data
    df = df[np.sort(df.columns)] ##Sort columns alphabetically to match DAG
    enc = LabelEncoder()
    df_le = df.copy()
    for var in df.columns:
        enc.fit(df[var])
        df_le[var] = enc.transform(df[var])
    if standardise:
        df_le_s = StandardScaler().fit_transform(df_le)
    else:
        df_le_s = df_le

    ##Extract true DAG from Bayesian network
    G = nx.from_edgelist(list(bn.edges()), create_using=nx.DiGraph)
    B_true = nx.adjacency_matrix(G).todense()
    B_pd = pd.DataFrame(B_true, columns=G.nodes(), index=G.nodes())
    ##Sort columns alphabetically to match data
    B_pd = B_pd.reindex(sorted(df.columns), axis=0)
    B_pd = B_pd.reindex(sorted(df.columns), axis=1)
    B_true = B_pd.values

    if print_info:
        print("Data shape:", df_le_s.shape)
        print("Number of true edges: ", len(bn.edges()))
        print("True BN edges:", bn.edges())
        print("DAG?",nx.is_directed_acyclic_graph(G))
        print("True DAG shape:", B_true.shape, "True DAG edges:", B_true.sum())
        print(B_pd)

    return df_le_s, B_true

def is_dag(W):
    # G = ig.Graph.Weighted_Adjacency(W.tolist())
    # return G.is_dag()
    return nx.is_directed_acyclic_graph(nx.from_numpy_array((W > 0).astype(int), create_using=nx.DiGraph))

# Largely from gcastle package
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

class GraphDAG(object):
    '''
    Visualization for causal discovery learning results.

    Parameters
    ----------
    est_dag: np.ndarray
        The DAG matrix to be estimated.
    true_dag: np.ndarray
        The true DAG matrix.
    show: bool
        Select whether to display pictures.
    save_name: str
        The file name of the image to be saved.
    '''

    def __init__(self, est_dag, true_dag=None, show=True, save_name=None, title='est_graph', size=(9, 3)):

        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name
        self.title = title
        self.size = size

        if not isinstance(est_dag, (np.ndarray, pd.DataFrame)):
            raise TypeError("Input est_dag is not numpy.ndarray or pd.DataFrame!")

        if true_dag is not None and not isinstance(true_dag, (np.ndarray, pd.DataFrame)):
            raise TypeError("Input true_dag is not numpy.ndarray or pd.DataFrame!")

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        GraphDAG._plot_dag(self.est_dag, self.true_dag, self.show, self.save_name, self.title, self.size)

    @staticmethod
    def _plot_dag(est_dag, true_dag, show=True, save_name=None, title='est_graph', size=(9, 3)):
        """
        Plot the estimated DAG and the true DAG.

        Parameters
        ----------
        est_dag: np.ndarray
            The DAG matrix to be estimated.
        true_dag: np.ndarray
            The True DAG matrix.
        show: bool
            Select whether to display pictures.
        save_name: str
            The file name of the image to be saved.
        """

        if isinstance(true_dag, (np.ndarray,pd.DataFrame)):
            
            
            if isinstance(est_dag, np.ndarray):
                # trans diagonal element into 0
                for i in range(len(true_dag)):
                    if est_dag[i][i] == 1:
                        est_dag[i][i] = 0
                    if true_dag[i][i] == 1:
                        true_dag[i][i] = 0

            # set plot size
            fig, (ax1, ax2) = plt.subplots(figsize=size, ncols=2)

            ax1.set_title(title)
            map1 = ax1.imshow(est_dag, cmap='viridis', interpolation='none')
            fig.colorbar(map1, ax=ax1)
            
            if isinstance(est_dag, pd.DataFrame):
                plt.xticks(range(est_dag.shape[0]), est_dag.columns, rotation=90)
                plt.yticks(range(est_dag.shape[1]), est_dag.columns)

            ax2.set_title('True Graph')
            map2 = ax2.imshow(true_dag, cmap='viridis', interpolation='none')
            fig.colorbar(map2, ax=ax2)

            if isinstance(true_dag, pd.DataFrame):
                plt.xticks(range(true_dag.shape[0]), true_dag.columns, rotation=90)
                plt.yticks(range(true_dag.shape[1]), true_dag.columns)
            
            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        elif isinstance(est_dag, (np.ndarray,pd.DataFrame)):

            if isinstance(est_dag, np.ndarray):
                # trans diagonal element into 0
                for i in range(len(est_dag)):
                    if est_dag[i][i] == 1:
                        est_dag[i][i] = 0

            # set plot size
            fig, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

            ax1.set_title(title)
            map1 = ax1.imshow(est_dag, cmap='viridis', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            if isinstance(est_dag, pd.DataFrame):
                plt.xticks(range(est_dag.shape[0]), est_dag.columns, rotation=90)
                plt.yticks(range(est_dag.shape[1]), est_dag.columns)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()



class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):
        
        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0
        
        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]
        
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
              'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)
        
        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """
        
        num_true = W_true.sum(axis=1).sum()
        assert num_true!=0
        
        # true_positives
        num_tp =  (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        num_fn_r = (W_p - W_true).applymap(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
        score = np.max((num_tp-num_fn_r,0))/num_true
        
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert(W_p.shape==W_true.shape and W_p.shape[0]==W_p.shape[1])
        TP = (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)
        
        return precision, recall, F1

from itertools import combinations
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

def find_all_d_separations_sets(G):
    no_of_var = len(G.nodes)
    sepset = np.empty((no_of_var, no_of_var), object)  # store the collection of sepsets
    for comb in combinations(range(no_of_var), 2):
        # print(comb)
        if comb[0] != comb[1]:
            x = comb[0]
            y = comb[1]
            # print(i,j)
            depth = 0
            while no_of_var-1 > depth:
                Neigh_x_noy = [f"X{k+1}" for k in range(no_of_var) if k != x and k != y]
                # print(Neigh_x_noy)
                # print(depth)
                for S in combinations(Neigh_x_noy, depth):
                    # print(S)
                    if nx.algorithms.d_separated(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        print(f"X{x+1} and X{y+1} are d-separated by {S}")
                        append_value(sepset, x, y, S)
                depth += 1
    return sepset