from plotly import graph_objects as go
from plotly.subplots import make_subplots
# import kaleido
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.graph_utils import DAGMetrics, dag2cpdag
from utils.data_utils import simulate_dag

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
ocean_blue = '#00C9BC'


dgp_list = ['gauss', 'exp', 'gumbel', 'uniform']

### make model list for all combinations of model, test and alpha
models = ['pc', 'pc_con', 'pc_maj', 'pc_max', 'spc']
# tests = ['fisherz']
alphas = ['005', '001', '01']

model_list = [m+"_"+a for m in models for a in alphas]
simple_name_dict = {'pc':'PC', 'pc_con':'CPC', 'pc_maj':'MPC', 'pc_max':'PC-Max', 'spc':'Shapley-PC'}
model_names = {'pc_con_':'CPC', 'pc_maj_':'MPC', 'pc_max_':'PC-Max', 'spc_':'Shapley-PC'}
markers = {'pc':'circle', 'pc_max':'square', 'pc_con':'triangle-up', 'pc_maj':'triangle-down', 'spc':'diamond'}

names_dict = {}
for m in model_list:
    for key in model_names.keys():
        entry = m.replace(key, model_names[key])
        if "_" not in entry:
            names_dict[m] = entry

names_dict['pc_001'] = 'PC001'
names_dict['pc_005'] = 'PC005'
names_dict['pc_01'] = 'PC01'

symbols_dict = {'pc_001':'circle-open','pc_005':'circle-dot','pc_01':'circle','pc_max_001':'square-open','pc_max_005':'square-dot','pc_max_01':'square','pc_maj_001':'triangle-up-open','pc_maj_005':'triangle-up-dot','pc_maj_01':'triangle-up','pc_con_001':'triangle-down-open','pc_con_005':'triangle-down-dot','pc_con_01':'triangle-down','spc_001':'diamond-open','spc_005':'diamond-dot','spc_01':'diamond'}
colors_dict = {'pc_001':main_blue,'pc_005':main_blue,'pc_01':main_blue,'pc_max_001':ocean_blue,'pc_max_005':ocean_blue,'pc_max_01':ocean_blue,'pc_maj_001':sec_orange,'pc_maj_005':sec_orange,'pc_maj_01':sec_orange,'pc_con_001':main_green,'pc_con_005':main_green,'pc_con_01':main_green,'spc_001':sec_blue,'spc_005':sec_blue,'spc_01':sec_blue}
general_filter = f"s<=1000 and n_nodes<=50 and model_test in {model_list} and sem_type in {dgp_list}"

def bar_chart_plotly(all_sum, var_to_plot, names_dict, colors_dict, methods, font_size, save_figs=False, output_name="bar_chart.html", debug=False):
    fig = go.Figure()
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    # for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
    for method in methods:#['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)']:
        trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method
        if 'log' in var_to_plot:
            var_to_plot = var_to_plot.replace('log_','')
            var_to_plot = "log(Elapsed Time)" if "lapsed" in var_to_plot else "log("+var_to_plot+")"
            trace_name = 'Log '+trace_name
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                opacity=0.6,
                            #  width=0.1
                                ))
            fig.update_yaxes(type="log")
        else:
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                    y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                    error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                    name=trace_name,
                                    marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                    opacity=0.6, #marker = dict(line=dict(width=30, color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]])),
                                #  width=0.1
                                    ))
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
            font=dict(size=font_size, family="Serif", color="black")
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=-0.01,
        yanchor="bottom",
        y=-0.1,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=font_size,
                    color="Black"
                    )
        )

    if 'n_' in var_to_plot or 'p_' in var_to_plot:
        orig_y = var_to_plot.replace('n_','').replace('p_','').upper()
        fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':font_size}})
    elif var_to_plot=='nnz':
        orig_y = 'Number of Edges in DAG'
        fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':font_size}})
    elif var_to_plot=='AH-F1':
        fig.update_yaxes(title={'text':'AH-F1','font':{'size':font_size}})
    else:
        fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':font_size}})

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))
    fig.show()

def double_bar_chart_plotly(all_sum, vars_to_plot, names_dict, colors_dict, 
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)'],
                            range_y1=None, range_y2=None,
                            save_figs=False, output_name="bar_chart.html", debug=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    # for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
    for n, var_to_plot in enumerate(vars_to_plot):
        for m, method in enumerate(methods):
            trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method#+' '+var_to_plot
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                yaxis=f"y{n+1}",
                                offsetgroup=m+len(methods)*n+(1*n),
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                opacity=0.6,
                                #  width=0.1
                                showlegend=n==0
                                ))
        if n==0:
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                    y=np.zeros(len(all_sum[(all_sum.model==method)]['dataset'])), 
                                    name='',
                                    offsetgroup=m+1,
                                    marker_color='white',
                                    opacity=1,
                                    # width=0.1
                                    showlegend=False
                                    )
                                    )
    second_ticks = False if all('SID' in var for var in vars_to_plot) else True
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.1, # gap between bars of the same location coordinate.)

            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
            template='plotly_white',
            # autosize=True,
            width=1600, 
            height=700,
            margin=dict(
                l=40,
                r=00,
                b=70,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=20, family="Serif", color="black"),
            yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=second_ticks, zeroline=True),
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0,
        yanchor="bottom",
        y=-0.08,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=20,
                    color="Black"
                    )
        )
    
    for n, var_to_plot in enumerate(vars_to_plot):
        if vars_to_plot == ['precision', 'recall']:
            if range_y1 is None:
                range_y = [0, 1.3]
            else:
                range_y = range_y1
        elif vars_to_plot == ['fdr', 'tpr']:
            range_y = [0, 1]
        elif 'shd' in var_to_plot or 'SID' in var_to_plot:
            if range_y1 is None and range_y2 is None:
                if 'high' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_high_mean'])+.3]
                elif 'low' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_low_mean'])+.3]
                else:
                    range_y = [0, 2] if n==0 else [0, max(all_sum['p_SID_mean'])+.3]
            else:
                range_y = range_y1 if n==0 else range_y2
        if 'n_' in var_to_plot or 'p_' in var_to_plot:
            orig_y = var_to_plot.replace('n_','').replace('p_','').replace('_low','').replace('_high','').upper()
            fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':20}}, secondary_y=n==1, range=range_y)
            if second_ticks == False:
                fig.update_yaxes(title={'text':'','font':{'size':20}}, secondary_y=True, range=range_y, showticklabels=False)
        elif var_to_plot=='nnz':
            orig_y = 'Number of Edges in DAG'
            fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':20}}, secondary_y=n==1, range=range_y)
        else:
            fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':20}}, secondary_y=n==1, range=range_y)

    start_pos = 0.017
    intra_dis = 0.12
    inter_dis = 0.13

    if vars_to_plot == ['precision', 'recall']:
        name1 = 'Precision'
        name2 = 'Recall'
        lin_space=6
        nl_space=8
        intra_dis = 0.115
        inter_dis = 0.135
    elif vars_to_plot == ['fdr', 'tpr']:
        name1 = 'FDR'
        name2 = 'TPR'
        lin_space=5
        nl_space=7
    elif vars_to_plot == ['p_shd', 'p_SID'] or vars_to_plot == ['p_shd', 'p_SID_low'] or vars_to_plot == ['p_shd', 'p_SID_high']:
        name1 = 'NSHD'
        name2 = 'NSID'
        lin_space=9
        nl_space=9
    elif vars_to_plot == ['p_SID_low', 'p_SID_high']:
        name1 = 'Low'
        name2 = 'High'
        lin_space=11
        nl_space=11
        intra_dis = 0.115
        inter_dis = 0.135
    elif vars_to_plot == ['p_shd', 'F1']:
        name1 = 'NSHD'
        name2 = 'F1'
        lin_space=9
        nl_space=11

    n_x_cat = len(all_sum.dataset.unique())
    list_of_pos = []
    left=start_pos
    for i in range(n_x_cat):
            right = left+intra_dis
            list_of_pos.append((left, right))
            left = right+inter_dis

    for s1,s2 in list_of_pos:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s1,
            y=1.015,
                    text=f"{' '*lin_space}{name1}{' '*(lin_space)}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=20,
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
                    text=f"{' '*(nl_space)}{name2}{' '*nl_space}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=20,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )


    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))

    fig.show()

###Plot Runtime
def plot_runtime(df, x_var_list, general_filter, names_dict, symbols_dict, colors_dict, methods,
                         share_y=False, save_figs=False, output_name="runtime.html", debug=False, font_size=20):
    cols = len(x_var_list)
    rows = 1
    fig = make_subplots(rows, cols, vertical_spacing=0.05, horizontal_spacing=0.01, shared_yaxes=True, shared_xaxes=True,)
    i = 1
    j = 1
    metric = 'elapsed_mean'

    for x_var in x_var_list:
        group_list = [x_var]
        for model in methods:#df_to_plot['model_test'].unique():
            filter = f"model=='{names_dict[model]}'"
            tab = df.query(filter)
            tab_grouped = pd.DataFrame()
            for g in df[group_list].drop_duplicates().values:
                ## aggregate only if more than one obs in the group
                if list(tab.groupby(group_list)[metric].count().loc[g])[0]>1:
                    tab_grouped = pd.concat([tab_grouped, tab.groupby(group_list)[metric].agg(['mean','std']).loc[g].reset_index()], axis=0)
                else:
                    ##append the single obs
                    single_tab = tab.query(f"{x_var}=={g[0]}")[[x_var,metric,metric.replace('mean','std')]]
                    single_tab.columns = [x_var,'mean','std']
                    tab_grouped = pd.concat([tab_grouped, single_tab], axis=0)
                    
            tab_grouped.sort_values(by=[x_var], inplace=True)
            fig.add_trace(go.Scatter(x=tab_grouped[x_var].astype(str)
                                    ,y=tab_grouped[metric.split("_")[1]]
                                    ,error_y=dict(type='data', array=tab_grouped['std'])
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
                        # "Proportional Sample Size (s=N/|V|)"
                        # "Sample Size (N)"
                        ]):
        this_xaxis = next(fig.select_xaxes(row = rows, col = c+1))
        this_xaxis.update(title=n,title_standoff=0)

    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
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
        fig.data[dl].error_y.thickness = 2
    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))
    fig.show()


#################### Boxplot ####################

def boxplot_by(df1, x, y, general_filter,
                        names_dict, symbols_dict, colors_dict, share_y=False, y_range=[0,5],
                        sem_types = ['linear','non-linear'], 
                        save_figs=False, output_name="boxplot.html", debug=False, font_size=20):
    df=df1.query(general_filter)
    fig = go.Figure()
    df.sort_values(by=[x,'model_test'],axis=0, inplace=True)
    if x=='sparsity':
        df.loc[df[x]==0,x] = '<0.1'
    for sem in sem_types:
        for model in df['model_test'].unique():
            line_w, opa = (2,0.8) if 'spc' in model else (1,0.6)
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
                                                    showlegend=(sem=='linear')
                                                    ))
        # if sem=='linear':
        #     ## Add random baseline
        #     rand_df = pd.DataFrame()
        #     idx = 0
        #     for n_nodes in [10,20,50]:
        #         for edge_per_node in [1,2,4]:
        #             idx += 1
        #             # unnormalised = y.replace('p_','').replace('n_','')
        #             rand = pd.DataFrame(create_rand_baseline(n_nodes, edge_per_node), index=[idx])
        #             if x == 'sparsity':
        #                 rand[x] = round((edge_per_node*n_nodes)/(n_nodes*(n_nodes-1)/2),1)
        #             else:
        #                 rand[x] = df[x].unique()[0] ## temp to group all
        #             if "p_" in y or "n_" in y:
        #                 unnormalised = y.replace('p_','').replace('n_','')
        #                 rand[y] = rand[unnormalised]/rand['nnz']
        #             rand_df = pd.concat([rand_df,rand])
        #     if x=='sparsity':
        #         rand_df.loc[rand_df[x]==0,x] = '<0.1'
        #     trace_name = "True Graph Size" if y=='nnz' else 'Random'
        #     fig.add_trace(go.Box(
        #         y = rand_df[y],
        #         x = rand_df[x].astype(str),
        #         name=trace_name,
        #         marker_color='black',
        #                 boxmean='sd', # represent mean
        #                 boxpoints=False,#'outliers',
        #                 jitter=0.0, # add some jitter for a better separation between points
        #                     whiskerwidth=0.2,
        #                             marker_size=2,
        #                                 line_width=1,
        #                                         opacity=0.6,
        #                                             showlegend=(sem=='linear')
        #                                             ))
        #     rand_sum = rand_df[[x,y]].groupby(x).agg(['mean', 'std']).reset_index()
        #     fig.add_trace(
        #             go.Scatter(x=rand_sum[x], y=rand_sum[y]['mean'], name= 'Rand', mode='markers', marker_symbol='line-ew',
        #                     marker=dict(
        #                     color='grey',
        #                     size=175,
        #                     line=dict(
        #                         color='grey',
        #                         width=1)
        #                         ),
        #                     showlegend=False), 
        #         )

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

    if len(sem_types)==2:
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


def create_rand_baseline(V, d, true_path="../datasets/synthetic_data/DAGs", 
                         seeds=[357,470,2743,4951,5088,5657,5852,6049,6659,9076], 
                         g_type='ER', B_true=None, metric='shd', cpdag=False):
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
            B_rand = simulate_dag(d=V, s0=d*V, graph_type=g_type)
        else:
            B_true = np.load(os.path.join(true_path, f'{g_type}_{V}_{d}_{seed}.npy'))
            B_rand = simulate_dag(d=V, s0=d*V, graph_type=g_type)
        calc_sid=False if 'sid' not in metric else True
        rand_accs = DAGMetrics(B_true, B_rand, sid=calc_sid).metrics
        if cpdag:
            rand_accs = DAGMetrics(B_true, dag2cpdag(B_rand)).metrics
    return rand_accs




### Define Line Plot Function
def plot_ly_s_n_d_graph(df_to_plot, var_to_plot, group_list, general_filter, sem_broad, 
                        names_dict, symbols_dict, colors_dict, share_y=False, save_figs=False, 
                            output_name="line_plot.html", debug=False):
    
    sem_list = ['ER','SF']
   
    cols = len(sem_list)*3
    rows = 3

    fig = make_subplots(rows, cols, subplot_titles=(sem_list*3), 
    vertical_spacing=0.03, horizontal_spacing=0.015, shared_yaxes=False, shared_xaxes=True,)

    i = 1
    j = 1
    
    for n_nodes in [10,20,50]:
        for edge_per_node in [1,2,4]:
            for graph_type in sem_list:
                for model in ['pc', 'pc_con', 'pc_maj', 'pc_max', 'spc']:
                    filter = f"n_nodes=={n_nodes} and edge_per_node=={edge_per_node} and model=='{model}' and graph_type=='{graph_type}'"
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
                i += 1 if i < cols else 0
        i=1
        j += 1 if j < rows else 0
    
    fig_update_layout(fig, rows, cols, var_to_plot)
    for dl in range(0,len(fig.data)):
        fig.data[dl].error_y.thickness = 1

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
            text=f"       |V|={d}       ",
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
    else:
        width, height = 1000, 600
        top_annotations_cols = zip([1,3,5],[1,2,4])
        top_annotation_spaces = 36


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
    elif sem_broad=='discrete':
        sem_list = ['BN']

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
                for model in df_to_plot['model'].unique():
                    filter = f"n_nodes=={n_nodes} and edge_per_node=={edge_per_node} and model=='{model}'"
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
                # rand = pd.DataFrame(create_rand_baseline(n_nodes, edge_per_node), index=[1])
                # rand_sum = rand.agg(['mean', 'std']).reset_index()
                # rand_name_baseline = names_dict['random'] if var_to_plot!='nnz' else 'True Graph Size'
                # fig.add_trace(go.Scatter(x=tab['s'].astype(str)
                #                         ,y=np.repeat(rand_sum[rand_sum['index']=='mean'][var_to_plot], len(tab['s']))   
                #                         ,error_y=dict(
                #                             type='data', # value of error bar given in data coordinates
                #                             array=np.repeat(rand_sum[rand_sum['index']=='std'][var_to_plot], len(tab['s']))  ,
                #                             visible=True)
                #                         ,name=rand_name_baseline, #legendgroup=f'group{i}{j}', 
                #                             line=dict(color=colors_dict['random'], width=2, simplify=True, dash='dash'), mode='lines+markers', 
                #                             marker=dict(size=4, symbol='x-thin', color=colors_dict['random']),              
                #                             showlegend=(j==1 and i==1)), j, i)   
                i += 1 if i < cols else 0
        i=1
        j += 1 if j < rows else 0
    
    fig_update_layout(fig, rows, cols, var_to_plot)
    for dl in range(0,len(fig.data)):
        fig.data[dl].error_y.thickness = 1

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
    else:
        width, height = 1000, 600
        top_annotations_cols = zip([1,3,5],[1,2,4])
        top_annotation_spaces = 36


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
