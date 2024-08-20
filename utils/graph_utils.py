import os, sys
import logging
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations, chain
from copy import deepcopy
from utils.helpers import append_value
import warnings
warnings.filterwarnings("ignore")
try:
    import cdt
except:
    sys.path.append('../CausalDiscoveryToolbox/')
    import cdt
from cdt.metrics import SHD, SID, SID_CPDAG

def model_to_adjacency_matrix(model:list, num_of_nodes:int)->np.ndarray:
    adj_mat = np.zeros((num_of_nodes,num_of_nodes))
    for atom in model:
        if atom.name == 'arrow':
            adj_mat[int(atom.arguments[0].number)][int(atom.arguments[1].number)] = 1
    return adj_mat

def model_to_set_of_arrows(model:list)->set:
    arrows = set()
    for atom in model:
        if atom.name == 'arrow':
            arrows.add((atom.arguments[0].number,atom.arguments[1].number))
    return arrows

def model_to_set_of_indep(model:list)->set:
    indeps = set()
    for atom in model:
        if atom.name == 'indep':
            indeps.add((atom.arguments[0].number,atom.arguments[1].number,atom.arguments[2].name))
    return indeps

def set_of_models_to_set_of_graphs(models, n_nodes, mec_check=True):
    MECs = defaultdict(list)
    MEC_set = set()
    model_sets = set()
    # logging.info("   Checking MECs")
    for model in models:
        arrows = model_to_set_of_arrows(model)
        indeps = model_to_set_of_indep(model)
        model_sets.add(frozenset(arrows))        
        if mec_check:
            adj = model_to_adjacency_matrix(model, n_nodes)
            cp_adj = dag2cpdag(adj)
            #cp_adj = get_CPDAG(adj)
            cp_adj_hashable = map(tuple, cp_adj)
            MECs[cp_adj_hashable] = list(adj.flatten())
            MEC_set.add(frozenset(cp_adj_hashable))
    logging.debug(f"   Number of MECs: {len(MEC_set)}")
    return model_sets, MECs

def extract_test_elements_from_symbol(symbol:str)->tuple:
    dep_type, elements = symbol.replace(").","").split("(")
    
    if "dep" in dep_type:
        X, Y, condset = elements.split(",")
        if condset == "empty":
            S = set()
        elif condset[0] == "s":
            S = set([int(e) for e in condset[1:].split("y")])
        else:
            raise ValueError(f"Unknown element {condset}")

        return int(X), S, int(Y), dep_type
    elif dep_type in ["arrow", "edge"]:
        X, Y = elements.split(",")
        return int(X), int(Y), dep_type
    else:
        raise ValueError(f"Unknown element {dep_type}")

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def find_all_d_separations_sets_list(G, verbose=True, debug=False)->list:
    no_of_var = len(G.nodes)
    septests = []
    for comb in combinations(range(no_of_var), 2):
        if comb[0] != comb[1]:
            x = comb[0]
            y = comb[1]
            depth = 0
            while no_of_var-1 > depth:
                Neigh_x_noy = [f"X{k+1}" for k in range(no_of_var) if k != x and k != y]
                for S in combinations(Neigh_x_noy, depth):
                    s = set([int(s.replace('X',''))-1 for s in S])
                    s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in s])
                    if nx.algorithms.d_separated(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        logging.debug(f"X{x+1} and X{y+1} are d-separated by {S}")
                        septests.append(f"indep({x},{y},{s_str}).")
                    else:
                        # logging.info(f"X{x+1} and X{y+1} are not d-separated by {S}")
                        septests.append(f"dep({x},{y},{s_str}).")
                depth += 1
    return septests

def find_all_d_separations_sets(G, verbose=False)->np.ndarray:
    """
    Find all d-separation sets for all pairs of nodes in a graph G
    """
    no_of_var = len(G.nodes)
    sepset = np.empty((no_of_var, no_of_var), object)  # store the collection of sepsets
    for comb in tqdm(combinations(range(no_of_var), 2)):
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
                    S_0 = tuple([int(s[1:]) - 1 for s in S])
                    if nx.algorithms.d_separated(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        append_value(sepset, x, y, (S_0, 1))
                        if verbose:
                            logging.debug(f"X{x+1} and X{y+1} are d-separated by {S}")
                    else:
                        append_value(sepset, x, y, (S_0, 0))
                        if verbose:
                            logging.debug(f"X{x+1} and X{y+1} are not d-separated by {S}")
                depth += 1
    return sepset

def initial_strength(p:float, len_S:int, alpha:float, base_strength:float, num_vars:int, verbose=False)->float:
    w_S = (1-len_S/(num_vars-2))
    # w_S = 1
    if p != None:
        if p < alpha:
            initial_strength = (1-0.5/alpha*p)*w_S
        else:
            initial_strength = ((alpha-0.5*p-0.5)/(alpha-1))*w_S
    else:
        initial_strength = base_strength
    return initial_strength

def is_dag(B):
    """Check if a matrix is a DAG"""
    return ig.Graph.Adjacency(B.tolist()).is_dag()

def mount_adjacency_list(adjacency_matrix):
    """
    Reads an adjacency matrix and returns the corres-
    ponding adjacency list (or adjacency map)
    """
    adjacency_list = {}
    for v1 in range(len(adjacency_matrix)):
        adjacency_list.update({v1: [v2 for v2 in range(len(adjacency_matrix[v1])) if adjacency_matrix[v1][v2] == 1]})
    return adjacency_list

def get_immoralities(adj_list):
    """
    Finds the set of immoralities in the adj_list
    """
    return [(v1, v3, v2) for v1 in adj_list for v2 in adj_list for v3 in adj_list[v1] \
            if v3 in adj_list[v2] and v1 < v2 and v2 not in adj_list[v1] and v1 not in adj_list[v2]]

def get_uts(adj):
    """
    Finds the set of unshielded triples in the adj matrix
    """
    return [(v1, v3, v2) for v1 in range(len(adj)) for v2 in range(len(adj)) for v3 in range(len(adj)) \
            if adj[v1, v3] == 1 and adj[v2, v3] == 1 and adj[v1, v2] == 0 and v1 < v2]

def clgraph2adj(G):
    """
    Convert a Causal-Learn Graph to an adjacency matrix
    """
    assert G.shape[1]==G.shape[0], "Input matrix is not square"

    adj = np.zeros((G.shape[1], G.shape[1]))
    for i in range(G.shape[1]):
        for j in range(G.shape[1]):
            ## Only replace tails of directed edges.
            if (G[i,j] == 1 and G[j,i] == -1):
                adj[i, j] = 1
                adj[j, i] = 0
            elif (G[i,j] == -1 and G[j,i] == 1):
                adj[i, j] = 0
                adj[j, i] = 1
            ## Leave undirected edges as they are [-1, -1]
            else:
                adj[i, j] = G[i,j]
                adj[j, i] = G[j,i]
    return adj

def dag2skel(G, unique=False):
    """Convert a DAG to a skeleton.

    Args:
        B (np.ndarray): [d, d] [0,1,-1] adj matrix of DAG (allowing for undirected edges as -1)

    Returns:
        C (np.ndarray): [d, d] [-1,0] adj matrix of skeleton
    """
    C = np.zeros(G.shape)
    undirected_edges = [(v1, v2) for v1 in range(len(G)) for v2 in range(len(G)) if G[v1, v2] == -1 and G[v2, v1] == -1]
    G1 = mount_adjacency_list(G)
    edges = [(v1, v2) for v1 in range(len(G1)) for v2 in G1[v1]]
    for v1, v2 in edges:
        C[v1, v2] = -1
        if not unique:
            C[v2, v1] = -1
    for v1, v2 in undirected_edges:
        C[v1, v2] = -1
        C[v2, v1] = -1
    return C

def dseps2skel(dseps, n_nodes, alpha=0.05):
    C = nx.complete_graph(n_nodes)
    for x,y in combinations(range(n_nodes), 2):
        if any([test[1]>alpha for test in dseps[x,y]]):
            C.remove_edge(x,y)
    return C 

def dag2cpdag(g, cdt_method=False):
    """Convert a DAG to a CPDAG.

    Args:
        G (np.ndarray): [d, d] binary adj matrix of DAG

    Returns:
        C (np.ndarray): [d, d] binary adj matrix of CPDAG
    """
    ### handle results from PC methods
    G = deepcopy(g)
    if (G == -1).any():  # undirected edges
        if ((G == -1) & (G.T == -1)).any():
            for i in range(len(G)):
                for j in range(len(G[i])):
                    if G[i, j] == G[j, i] == -1:
                        G[i, j] = 1
                        G[j, i] = 1
        if ((G == -1) & (G.T == 1)).any(): ## Causal Learn tails (-1) on directed edges
            for i in range(len(G)):
                for j in range(len(G[i])):
                    if G[i, j] == -1 and G[j, i] == 1:
                        G[i, j] = 0
                        G[j, i] = 1

    edges_removed = False
    if not is_dag(G):
        ### get undirected edges and remove them from G
        undirected_edges = [(v1, v2) for v1 in range(len(G)) for v2 in range(len(G)) if G[v1, v2] == 1 and G[v2, v1] == 1]
        if len(undirected_edges) > 0:
            edges_removed = True
            for v1, v2 in undirected_edges:
                G[v1, v2] = 0
                G[v2, v1] = 0
            
    assert is_dag(G), 'Input graph is not a DAG'
    ###only leave the arrows that are part of a v-structure
    C = dag2skel(G)
    immoralities = get_immoralities(mount_adjacency_list(G))
    for v1, v3, v2 in immoralities:
        C[v1, v3] = 1
        C[v3, v1] = 0
        C[v2, v3] = 1
        C[v3, v2] = 0
    if edges_removed:
        for v1, v2 in undirected_edges:
            ## reintroduce undirected edges
            C[v1, v2] = -1
            C[v2, v1] = -1
    if cdt_method:
        C = (C != 0).astype(int)
    return C


#########################################################################################
#                                   PAG utils                                           #
#########################################################################################

### Largely from TrustworthyAI repo, with some modifications and the addition of SID from cdt.metrics
class DAGMetrics(object):
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

    def __init__(self, B_est, B_true, sid=True):
        
        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = deepcopy(B_est)
        self.B_true = deepcopy(B_true)

        self.metrics = DAGMetrics._count_accuracy(self.B_est, self.B_true, sid)

    @staticmethod
    def _count_accuracy(B_est, B_true, sid=True, output_n=True, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        sid: bool
            If True, compute SID.
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

        ## Do not allow self loops
        if (np.diag(B_est)).any():
            raise ValueError('Graph contains self loops')
        ## Only allow 0, 1, -1
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        
        B_est_unique = deepcopy(B_est)
        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        if ((B_est_unique == 1) & (B_est_unique.T == 1)).any():
            cpdag = True
            for i in range(len(B_est_unique)):
                for j in range(len(B_est_unique[i])):
                    if B_est_unique[i, j] == B_est_unique[j, i] == 1:
                        B_est_unique[i, j] = -1
                        B_est_unique[j, i] = 0
        if (B_est_unique == -1).any():  # cpdag
            cpdag = True
            ## only one entry in the pair of undirected edges should be -1
            if ((B_est_unique == -1) & (B_est_unique.T == -1)).any():
                for i in range(len(B_est_unique)):
                    for j in range(len(B_est_unique[i])):
                        if B_est_unique[i, j] == B_est_unique[j, i] == -1:
                            B_est_unique[i, j] = -1
                            B_est_unique[j, i] = 0
                assert not ((B_est_unique == -1) & (B_est_unique.T == -1)).any()
                assert not ((B_est_unique == -1) & (B_est_unique.T == -1)).any()
        else:  # dag
            cpdag = False
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            if not is_dag(B_est):
                raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]
        
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred_und_unique = np.flatnonzero(B_est_unique == -1)
        pred = np.flatnonzero(B_est_unique == 1)
        pred_reversed = np.flatnonzero(B_est_unique.T == 1)
        pred_skeleton = np.concatenate([pred, pred_reversed, pred_und])
        pos = np.flatnonzero(B_true == 1) if not cpdag else np.flatnonzero(dag2cpdag(B_true) == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, pos, assume_unique=True) if len(pred) > 0 else np.array([])
            ## treat undirected edge favorably
        true_pos_skel = np.intersect1d(pred_skeleton, cond_skeleton, assume_unique=True)
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos_und_unique = np.setdiff1d(pred_und_unique, cond_skeleton, assume_unique=True)
        false_pos_unique = np.concatenate([false_pos_und_unique, false_pos]) # Edge is one false discovery
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True) if len(extra) > 0 else np.array([])
        # compute ratio
        pred_size = len(pred) ## Number of directed edges
        sksize = len(pred_skeleton)/2 ## Number of directed and undirected edges
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos_unique)) / max(sksize, 1) ## False discovery rate, both directed and undirected
        fpr = float(len(reverse) + len(false_pos_unique)) / max(cond_neg_size, 1) ## False positive rate, both directed and undirected
        skp = float(len(true_pos_skel)) / max(sksize*2, 1) ## Skeleton precision
        skr = float(len(true_pos_skel)) / max(len(cond_skeleton), 1) ## Skeleton recall
        tpr = float(len(true_pos)) / max(len(pred), 1) ## Arrowhead precision
        hhr = float(len(true_pos)) / max(len(pos), 1) ## Arrowhead recall
        try:
            skF1 = 2 * (skp * skr) / (skp + skr) ## Skeleton F1
        except ZeroDivisionError:
            skF1 = 2 * (skp * skr) / max((skp + skr),1) ## Skeleton F1
        try:
            arrF1 = 2 * (tpr * hhr) / (tpr + hhr) ## Arrowhead F1
        except ZeroDivisionError:
            arrF1 = 2 * (tpr * hhr) / max((tpr + hhr),1) ## Arrowhead F1

        # structural hamming distance 
        # pred_lower = np.flatnonzero(np.tril(B_est_unique + B_est.T))
        # cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        # extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        # missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        # shd = len(extra_lower) + len(missing_lower) + len(reverse)        
        ### this, although standard in some packages,
        ### treats undirected edge as a present edge in the CPDAG 
        ### Replacing with SHD from cdt
        if cpdag:
            shd = DAGMetrics._cal_SHD_CPDAG(B_est, B_true)
        else:
            shd = SHD(B_true, B_est, False)

        W_p = pd.DataFrame(B_est_unique)
        W_true = pd.DataFrame(B_true)

        # gscore = DAGMetrics._cal_gscore(W_p, W_true)
        precision, recall, F1 = DAGMetrics._cal_precision_recall(W_p, W_true) ## These do not differentiate between directed and undirected edges

        mt = {'nnz': pred_size, 'sksize':sksize, 'fdr': fdr, 'fpr': fpr, 
              'tpr': tpr, 'hhr': hhr, 'arrF1': arrF1,
              'skp': skp, 'skr': skr, 'skF1': skF1, 
              'precision': precision, 'recall': recall, 'F1': F1,#, 'gscore': gscore
              'shd': shd}

        for i in mt:
            mt[i] = round(mt[i], decimal_num)   

        if sid and not cpdag:
            mt['sid'] = DAGMetrics._cal_SID(B_est, B_true)
        elif sid and cpdag:
            mt['sid'] = DAGMetrics._cal_SID_CPDAG(B_est, B_true)

        ### immoralities
        mt['immoral_prec'], mt['immoral_rec'], mt['immoral_F1'], mt['immoral_UT_prec'], mt['immoral_UT_rec'], mt['immoral_UT_F1'],\
          mt['UT_prec'], mt['UT_rec'], mt['UT_F1'], \
            mt['n_imm_est'], mt['n_imm'], mt['n_imm_ut_est'], mt['n_imm_ut'], mt['n_ut_right'], mt['n_ut'], mt['n_ut_est']  \
              = DAGMetrics._cal_immoral_acc(B_est, B_true, output_n)

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
        score = np.nan
        if num_true!=0:
            # true_positives
            num_tp =  (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
            # False Positives + Reversed Edges
            num_fn_r = (W_p - W_true).map(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
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
        if (W_p == -1).any().any():
            W_p = pd.DataFrame((W_p != 0).astype(int))
        if (W_true == -1).any().any():
            W_true = pd.DataFrame((W_true != 0).astype(int))

        TP = (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)
        
        return precision, recall, F1
    
    @staticmethod
    def _cal_SID(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1}.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        SID: float
            Structural Intervention Distance
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        assert is_dag(B_est), 'B_est should be a DAG'
        return SID(B_true, B_est).flat[0]
    
    @staticmethod
    def _cal_SHD_CPDAG(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        SHD: int
            Structural Hamming Distance of CPDAG
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        ### treat undirected edge as a present edge in the CPDAG
        ### the difference will be in the missed immoralities
        return SHD(dag2cpdag(B_true,True), (B_est != 0).astype(int), False)

    @staticmethod
    def _cal_SID_CPDAG(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}. Needs to be a DAG.

        Return
        ------
        SID_CPDAG_low: float
            Lower bound of Structural Intervention Distance
        SID_CPDAG_high: float
            Upper bound of Structural Intervention Distance
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        ### treat undirected edge as a present edge in the CPDAG
        ### the difference will be in the missed immoralities
        SID_CPDAG_low, SID_CPDAG_high = [a.flat[0] for a in SID_CPDAG(B_true, (B_est != 0).astype(int))]
        return SID_CPDAG_low, SID_CPDAG_high

    @staticmethod
    def _cal_immoral_acc(B_est, B_true, output_absolute_n=True):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        immoral_acc: float
            Immorality accuracy
        """
        n_imm = get_immoralities(mount_adjacency_list(B_true))
        n_imm_est = get_immoralities(mount_adjacency_list(B_est))

        C_true = (dag2skel(B_true) != 0).astype(int)
        C_est = (dag2skel(B_est) != 0).astype(int)

        n_ut = get_uts(C_true)
        n_ut_est = get_uts(C_est)
        n_ut_right = set(n_ut).intersection(set(n_ut_est))

        if len(n_ut_est) == 0:
            UT_prec = 0
            UT_rec = 0
            UT_F1 = 0
        elif len(n_ut) == 0:
            UT_prec = 1
            UT_rec = 0
            UT_F1 = 0
        else:
            UT_prec = len(n_ut_right) / len(n_ut_est)
            UT_rec = len(n_ut_right) / len(n_ut)
            if UT_prec + UT_rec == 0:
                UT_F1 = 0
            else:
                UT_F1 = 2 * (UT_prec * UT_rec) / (UT_prec + UT_rec)

        if len(n_imm_est) == 0:
            immoral_prec = 0
            immoral_rec = 0
            immoral_F1 = 0
        elif len(n_imm) == 0:
            immoral_prec = 1
            immoral_rec = 0
            immoral_F1 = 0
        else:
            immoral_prec = len(set(n_imm_est).intersection(set(n_imm))) / len(set(n_imm_est))
            immoral_rec = len(set(n_imm_est).intersection(set(n_imm))) / len(set(n_imm_est))
            if immoral_prec + immoral_rec == 0:
                immoral_F1 = 0
            else:
                immoral_F1 = 2 * (immoral_prec * immoral_rec) / (immoral_prec + immoral_rec)

        ### exclude skeleton effect
        n_imm_ut = set(n_ut_right).intersection(set(n_imm))
        n_imm_ut_est = set(n_ut_right).intersection(set(n_imm_est))

        if len(n_imm_ut_est) == 0:
            immoral_UT_prec = 0
            immoral_UT_rec = 0
            immoral_UT_F1 = 0
        elif len(n_imm_ut) == 0:
            immoral_UT_prec = 1
            immoral_UT_rec = 0
            immoral_UT_F1 = 0
        else:
            immoral_UT_prec = len(n_imm_ut_est.intersection(n_imm_ut)) / len(n_imm_ut_est)
            immoral_UT_rec = len(n_imm_ut_est.intersection(n_imm_ut)) / len(n_imm_ut)
            if immoral_UT_prec + immoral_UT_rec == 0:
                immoral_UT_F1 = 0
            else:
                immoral_UT_F1 = 2 * (immoral_UT_prec * immoral_UT_rec) / (immoral_UT_prec + immoral_UT_rec)

        if output_absolute_n:
            return immoral_prec, immoral_rec, immoral_F1, immoral_UT_prec, immoral_UT_rec, immoral_UT_F1, UT_prec, UT_rec, UT_F1, \
                len(n_imm_est), len(n_imm), len(n_imm_ut_est), len(n_imm_ut), len(n_ut_right), len(n_ut), len(n_ut_est)

        return immoral_prec, immoral_rec, immoral_F1, immoral_UT_prec, immoral_UT_rec, immoral_UT_F1, UT_prec, UT_rec, UT_F1


def trueCov(dag):
    wm = wgtMatrix(dag)
    p = len(dag.nodes)
    return np.dot(np.linalg.inv(np.eye(p) - wm), np.transpose(np.linalg.inv(np.eye(p) - wm)))


def wgtMatrix(g:nx.Graph, transpose=True):
    res = nx.adjacency_matrix(g).toarray()
    if transpose:
        res = res.T
    return res

def cov2cor(V):
    p = V.shape[0]
    if not isinstance(V, np.ndarray) or len(V.shape) != 2 or p != V.shape[1]:
        raise ValueError("'V' is not a square numeric matrix")
    Is = np.sqrt(1 / np.diag(V))
    if np.any(~np.isfinite(Is)):
        warnings.warn("diag(.) had 0 or NA entries; non-finite result is doubtful")
    r = V.copy()
    r *= np.repeat(Is, p).reshape(p, p)
    np.fill_diagonal(r, 1)
    return r


#########################################################################################
#                                   PAG Metrics                                         #
#########################################################################################

### instrumental functions largely from https://github.com/mensxmachina/AutoCD/blob/main/metrics_evaluation/shd_mag_pag.py

# amat[i,j] = 0 iff no edge btw i,j
# amat[i,j] = 1 iff i *-o j
# amat[i,j] = 2 iff i *-> j
# amat[i,j] = 3 iff i *-- j

class PAGMetrics(object):
    """
    Compute various accuracy metrics for P_est.

    Parameters
    ----------
    P_est: np.ndarray
        [d, d] estimate, {0, 1, 2, 3}.
    P_true: np.ndarray
        [d, d] ground truth graph, {0, 1, 2, 3}.
    output_counts: bool
        If True, return counts of True Positive, False Positive and False Negative 
        for each of Undirected (o, 1), Arrow (>, 2), Tail (-, 3), and bidirected edge marks (<>).
    
    Output
    ------
    metrics: dict
        shd: int
            Structural Hamming Distance
        arr_prec: float
            Arrowhead Precision
        arr_rec: float
            Arrowhead Recall
        arr_F1: float
            Arrowhead F1
        tail_prec: float
            Tail Precision
        tail_rec: float
            Tail Recall
        tail_F1: float
            Tail F1
        und_prec: float
            Undirected Precision
        und_rec: float
            Undirected Recall
        und_F1: float
            Undirected F1
        bid_prec: float
            Bidirected Precision
        bid_rec: float
            Bidirected Recall
        bid_F1: float
            Bidirected F1
        adj_prec: float
            Adjacency Precision
        adj_rec: float
            Adjacency Recall
        adj_F1: float
            Adjacency F1
    """

    def __init__(self, P_est, P_true, return_counts=False):
        
        if not isinstance(P_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(P_true, np.ndarray):
            raise TypeError("Input P_true is not numpy.ndarray!")

        self.P_est = deepcopy(P_est)
        self.P_true = deepcopy(P_true)

        self.metrics = PAGMetrics.marks_accuracy(self.P_est, self.P_true, return_counts)

        self.metrics['adj_prec'], self.metrics['adj_rec'], self.metrics['adj_F1'] = \
                    PAGMetrics.adjacency_precision_recall(pd.DataFrame(self.P_est), pd.DataFrame(self.P_true))

    @staticmethod
    def marks_accuracy(G1, G2, return_counts=False):

        '''
        Computes the structural hamming distance as appeared in
        S. Triantafillou and I. Tsamardinos,  UAI 2016
            SHD Author : kbiza@csd.uoc.gr, based on matlab code by striant@csd.uoc.gr
        Computes Broken down precision, recall and F1 score for the different types of edges
            Author : fabrizio@imperial.ac.uk

        Args:
            G1(numpy array): a matrix of a graph (mag or pag)
            G2(numpy array): a matrix of a graph (mag or pag, must be the same type with G1)

        Returns:
            shd(int): the value of the metric
            
        '''

        n_nodes = G1.shape[0]
        shd = 0
        arrTP = 0
        arrFP = 0
        arrFN = 0
        tailTP = 0
        tailFP = 0
        tailFN = 0
        undTP = 0
        undFP = 0
        undFN = 0
        bidTP = 0
        bidFP = 0
        bidFN = 0
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # o-o
                if G1[i,j] == 1 and G1[j,i] == 1:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 0
                        undTP = undTP + 2
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 1
                        undTP = undTP + 1
                        undFN = undFN + 1
                        arrFP = arrFP + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 1
                        undTP = undTP + 1
                        undFN = undFN + 1
                        arrFP = arrFP + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 2
                        undFN = undFN + 2
                        arrFP = arrFP + 2
                        bidFP = bidFP + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 2
                        undFN = undFN + 2
                        tailFP = tailFP + 1
                        arrFP = arrFP + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 2
                        undFN = undFN + 2
                        tailFP = tailFP + 1
                        arrFP = arrFP + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 1
                        undFN = undFN + 2
                        
                # o->
                if G1[i,j] == 2 and G1[j,i] == 1:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 1
                        undTP = undTP + 1
                        undFP = undFP + 1
                        arrFN = arrFN + 1
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 0
                        arrTP = arrTP + 1
                        undTP = undTP + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 2
                        arrFN = arrFN + 1
                        arrFP = arrFP + 1
                        undFP = undFP + 1
                        undFN = undFN + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFP = arrFP + 1
                        undFN = undFN + 1
                        bidFP = bidFP + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        tailFP = tailFP + 1
                        undFN = undFN + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        undFN = undFN + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 2
                        arrFN = arrFN + 1
                        undFN = undFN + 1

                # <-o
                if G1[i,j] == 1 and G1[j,i] == 2:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 1
                        undTP = undTP + 1
                        undFP = undFP + 1
                        arrFN = arrFN + 1
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        undFP = undFP + 1
                        undFN = undFN + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 0
                        arrTP = arrTP + 1
                        undTP = undTP + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFP = arrFP + 1
                        undFN = undFN + 1
                        bidFP = bidFP + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        undFN = undFN + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        tailFP = tailFP + 1
                        undFN = undFN + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 2
                        arrFN = arrFN + 1
                        undFN = undFN + 1

                # <->
                if G1[i,j] == 2 and G1[j,i] == 2:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 2
                        undFP = undFP + 2
                        arrFN = arrFN + 2
                        bidFN = bidFN + 1
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFN = arrFN + 1
                        undFP = undFP + 1
                        bidFN = bidFN + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFN = arrFN + 1
                        undFP = undFP + 1
                        bidFN = bidFN + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 0
                        arrTP = arrTP + 2
                        bidTP = bidTP + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        bidFN = bidFN + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        bidFN = bidFN + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 3
                        arrFN = arrFN + 2
                        bidFN = bidFN + 1

                # -->
                if G1[i,j] == 2 and G1[j,i] == 3:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 2
                        undFP = undFP + 2
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        undFP = undFP + 1
                        tailFN = tailFN + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        undFP = undFP + 1
                        tailFN = tailFN + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFP = arrFP + 1
                        tailFN = tailFN + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 0
                        arrTP = arrTP + 1
                        tailTP = tailTP + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        tailFN = tailFN + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 3
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1

                # <--
                if G1[i,j] == 3 and G1[j,i] == 2:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 2
                        undFP = undFP + 2
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                        undFP = undFP + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        tailFN = tailFN + 1
                        undFP = undFP + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 1
                        arrTP = arrTP + 1
                        arrFP = arrFP + 1
                        tailFN = tailFN + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 2
                        arrFP = arrFP + 1
                        arrFN = arrFN + 1
                        tailFP = tailFP + 1
                        tailFN = tailFN + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 0
                        arrTP = arrTP + 1
                        tailTP = tailTP + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 3
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                        
                # 'empty'
                if G1[i,j] == 0 and G1[j,i] == 0:
                    # o-o
                    if G2[i,j] == 1 and G2[j,i] == 1:
                        shd = shd + 1
                        undFN = undFN + 2
                    # o->
                    if G2[i,j] == 2 and G2[j,i] == 1:
                        shd = shd + 2
                        arrFN = arrFN + 1
                        undFN = undFN + 1
                    # <-o
                    if G2[i,j] == 1 and G2[j,i] == 2:
                        shd = shd + 2
                        arrFN = arrFN + 1
                        undFN = undFN + 1
                    # <->
                    if G2[i,j] == 2 and G2[j,i] == 2:
                        shd = shd + 3
                        arrFN = arrFN + 2
                        bidFP = bidFP + 1
                    # -->
                    if G2[i,j] == 2 and G2[j,i] == 3:
                        shd = shd + 3
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                    # <--
                    if G2[i,j] == 3 and G2[j,i] == 2:
                        shd = shd + 3
                        arrFN = arrFN + 1
                        tailFN = tailFN + 1
                    # 'empty'
                    if G2[i,j] == 0 and G2[j,i] == 0:
                        shd = shd + 0

            ### Arrowhead precision
            if arrTP + arrFP != 0:
                arr_prec = arrTP / (arrTP + arrFP)
            else:
                arr_prec = np.NaN
            ### Arrowhead recall
            if arrTP + arrFN != 0:
                arr_rec = arrTP / (arrTP + arrFN)
            else:
                arr_rec = np.NaN
            ### Tail precision
            if tailTP + tailFP != 0:
                tail_prec = tailTP / (tailTP + tailFP)
            else:
                tail_prec = np.NaN
            ### Tail recall
            if tailTP + tailFN != 0:
                tail_rec = tailTP / (tailTP + tailFN)
            else:
                tail_rec = np.NaN
            ### Undirected precision
            if undTP + undFP != 0:
                und_prec = undTP / (undTP + undFP)
            else:
                und_prec = np.NaN
            ### Undirected recall
            if undTP + undFN != 0:
                und_rec = undTP / (undTP + undFN)
            else:
                und_rec = np.NaN
            ### Bidirected precision
            if bidTP + bidFP != 0:
                bid_prec = bidTP / (bidTP + bidFP)
            else:
                bid_prec = np.NaN
            ### Bidirected recall
            if bidTP + bidFN != 0:
                bid_rec = bidTP / (bidTP + bidFN)
            else:
                bid_rec = np.NaN

            ### F1 scores
            if arr_prec + arr_rec != 0:
                arr_F1 = 2 * (arr_prec * arr_rec) / (arr_prec + arr_rec)
            else:
                arr_F1 = np.NaN
            if tail_prec + tail_rec != 0:
                tail_F1 = 2 * (tail_prec * tail_rec) / (tail_prec + tail_rec)
            else:
                tail_F1 = np.NaN
            if und_prec + und_rec != 0:
                und_F1 = 2 * (und_prec * und_rec) / (und_prec + und_rec)
            else:
                und_F1 = np.NaN
            if bid_prec + bid_rec != 0:
                bid_F1 = 2 * (bid_prec * bid_rec) / (bid_prec + bid_rec)
            else:
                bid_F1 = np.NaN

            metrics = {'shd': shd, 'arr_prec': arr_prec, 'arr_rec': arr_rec, 'arr_F1': arr_F1,
                          'tail_prec': tail_prec, 'tail_rec': tail_rec, 'tail_F1': tail_F1,
                          'und_prec': und_prec, 'und_rec': und_rec, 'und_F1': und_F1,
                          'bid_prec': bid_prec, 'bid_rec': bid_rec, 'bid_F1': bid_F1}
            
            if return_counts:
                metrics['arrTP'] = arrTP
                metrics['arrFP'] = arrFP
                metrics['arrFN'] = arrFN
                metrics['tailTP'] = tailTP
                metrics['tailFP'] = tailFP
                metrics['tailFN'] = tailFN
                metrics['undTP'] = undTP
                metrics['undFP'] = undFP
                metrics['undFN'] = undFN
                metrics['bidTP'] = bidTP
                metrics['bidFP'] = bidFP
                metrics['bidFN'] = bidFN
        
        return metrics  


    @staticmethod
    def adjacency_precision_recall(true_G_pd, est_G_pd):

        '''
        Computes adjacency precision and recall
        Parameters
        ----------
            true_G_pd(pandas Dataframe): the true graph
            est_G_pd(pandas Dataframe): the estimated graph

        Returns
        -------
            adj_prec(float) : the adjacency precision
            adj_rec(float) : the adjacency recall
        '''

        var_names = true_G_pd.columns
        true_G = true_G_pd.to_numpy()
        est_G = est_G_pd.to_numpy()
        n_nodes = true_G.shape[0]

        tp = 0
        fn = 0
        fp = 0
        tn = 0
        true_positive_edges = []
        false_negative_edges = []
        false_positive_edges = []
        true_negative_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):

                # adjacent in true_G
                if true_G[i, j] != 0 and true_G[j, i] != 0:

                    # adjacent in G2
                    if est_G[i, j] != 0 and est_G[j, i] != 0:
                        tp += 1
                        true_positive_edges.append([var_names[i], var_names[j]])

                    # not adjacent in G2
                    else:
                        fn += 1
                        false_negative_edges.append([var_names[i], var_names[j]])

                # not adjacent in true_G
                else:
                    # adjacent in est_G
                    if est_G[i, j] != 0 and est_G[j, i] != 0:
                        fp += 1
                        false_positive_edges.append([var_names[i], var_names[j]])

                    # not adjacent in G2
                    else:
                        tn += 1
                        true_negative_edges.append([var_names[i], var_names[j]])

        if tp + fp != 0:
            adj_prec = tp / (tp + fp)
        else:
            adj_prec = np.NaN

        if tp + fn != 0:
            adj_rec = tp / (tp + fn)
        else:
            adj_rec = np.NaN

        if (adj_prec + adj_rec) != 0:
            adj_F1 = 2 * (adj_prec * adj_rec) / (adj_prec + adj_rec)
        else:
            adj_F1 = np.NaN

        return adj_prec, adj_rec, adj_F1
