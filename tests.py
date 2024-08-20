"""Copyright 2024 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2024 Fabrizio Russo"

import sys
import unittest
import logging
from PC_testing import pc
import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime
from utils.helpers import logger_setup, random_stability, powerset, append_value
from utils.graph_utils import find_all_d_separations_sets, dseps2skel, mount_adjacency_list, get_immoralities, dag2cpdag, DAGMetrics, PAGMetrics
from utils.data_utils import simulate_dag, simulate_parameter, simulate_linear_sem, simulate_nonlinear_sem, simulate_discrete_data
from utils.cit import *
from spc import shapley

import jpype.imports

try:
    jpype.startJVM(classpath=[f"../py-tetrad/pytetrad/resources/tetrad-current.jar"])
except OSError:
    print("JVM already started")

import tools.translate as tr
import edu.cmu.tetrad.search as ts

os.environ['R_HOME'] = '../R/R-4.1.2/bin/'
### To not have the WARNING: ignoring environment value of R_HOME 
### set the verbose to False in the launch_R_script function in:
### CausalDiscoveryToolbox/cdt/utils/R.py#L155
os.environ["CUDA_VISIBLE_DEVICES"]="0"
try:
    import cdt
except:
    sys.path.append('../CausalDiscoveryToolbox/')
    import cdt
from cdt.metrics import SHD, SID, SID_CPDAG
cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'

class TestShapley(unittest.TestCase):

    def collider(self):
        scenario = "collider"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1],
                            [ 0,  0,  1],
                            [ 0,  0,  0]
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = set([
            frozenset({(0, 2), (1, 2)}),
        ])
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, prec_lev=1, verbose=False)
            logging.info(sv_list)
            list_of_svs.append(sv_list[0])

        self.assertEqual(list_of_svs, [(2, -1.0), (1, 0.0), (0, 0.0)])

    def chains_confounder(self):
        scenario = "confounder"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1],
                            [ 0,  0,  0],
                            [ 0,  0,  0]
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = set([
            frozenset({(0, 1), (0, 2)}),
        ])
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append(sv_list[0])

        self.assertEqual(list_of_svs, [(2, 0.0), (1, 0.0), (0, 1.0)])

    def triangle(self):
        scenario = "triangle"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1],
                            [ 0,  0,  1],
                            [ 0,  0,  0]
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = set([
            frozenset({(0, 1), (1, 2), (0, 2)}),
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append(sv_list[0])

        self.assertEqual(list_of_svs, [(2, 0.0), (1, 0.0), (0, 0.0)])

    def disconnected(self):
        scenario = "disconnected"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  0],
                            [ 0,  0,  0],
                            [ 0,  0,  0]
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = set([
            frozenset({(0, 1)}),
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append(sv_list[0])

        self.assertEqual(list_of_svs, [(2, 0.0), (1, 0.0), (0, 0.0)])

    def double_chain_shapPC_example(self):
        scenario = "double_chain_shapPC_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (1, 2), (3, 2), (4, 3)})

        true_svs = [{(0, 1): [(2, 0.0), (3, 0.0)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.5), (3, -0.5)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        self.assertEqual(list_of_svs, true_svs)

    def four_node_shapPC_example(self):
        scenario = "four_node_shapPC_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = set([
            frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        ])
        true_svs = [{(0, 1): [(2, -0.5), (3, -0.5)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        self.assertEqual(list_of_svs, true_svs)

    def five_node_colombo_example(self):
        scenario = "five_node_colombo_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  1,  1],
                            [ 0,  0,  1,  0,  1],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})

        true_svs = [{(0, 1): [(2, -0.333), (3, -0.333), (4, -0.333)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 3): [(1, 0.0), (2, 0.0), (4, 0.0)]}, 
                    {(0, 4): [(1, 0.0), (2, 0.0), (3, 0.0)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(1, 3): [(0, 0.167), (2, 0.167), (4, -0.333)]}, 
                    {(1, 4): [(0, 0.0), (2, 0.0), (3, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                    {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0)]}, 
                    {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        self.assertEqual(list_of_svs, true_svs)

    def five_node_M_example(self):
        scenario = "five_node_M_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (1, 3), (0, 2), (2, 3)})

        true_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 2): [(1, 1.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, -0.5), (4, 0.0)]}, 
                    {(0, 4): [(1, 0.167), (2, -0.333), (3, 0.167)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, -1.0), (4, 0.0)]}, 
                    {(1, 4): [(0, 0.0), (2, -0.5), (3, 0.5)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                    {(2, 4): [(0, 0.0), (1, 0.0), (3, 1.0)]}, 
                    {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        self.assertEqual(list_of_svs, true_svs)

    def five_node_sprinkler_example(self):
        scenario = "five_node_sprinkler_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (1, 3), (0, 2), (2, 3)})

        true_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5), (4, 0.0)]}, 
                    {(0, 4): [(1, 0.167), (2, 0.167), (3, 0.667)]}, 
                    {(1, 2): [(0, 0.333), (3, -0.167), (4, -0.167)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0), (4, 0.0)]}, 
                    {(1, 4): [(0, 0.0), (2, 0.0), (3, 1.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                    {(2, 4): [(0, 0.0), (1, 0.0), (3, 1.0)]}, 
                    {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        true_seplist = find_all_d_separations_sets(G_true)
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        self.assertEqual(list_of_svs, true_svs)


class TestDecisionRule(unittest.TestCase):

    def four_node_shapPC_example(self):
        scenario = "four_node_shapPC_example_colliders"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        true_svs = [{(0, 1): [(2, -0.5), (3, -0.5)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        ## list unshielded triples
        UTs = []
        C = dseps2skel(true_seplist, n_nodes)
        logging.debug(f"Skeleton: {C.edges}")

        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"Unshielded triples: {UTs}")

        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")
                
        self.assertEqual(set(est_immoralities), set(immoralities))

    def five_node_M_example(self):
        scenario = "five_node_M_example"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (0, 2), (3, 2), (3, 4)})
        true_svs = []
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        est_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0)]}, 
                   {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0)]}, 
                   {(0, 3): [(1, 0.0), (2, -1.0), (4, 0.0)]}, 
                   {(0, 4): [(1, 0.0), (2, -0.5), (3, 0.5)]}, 
                   {(1, 2): [(0, 1.0), (3, 0.0), (4, 0.0)]}, 
                   {(1, 3): [(0, 0.5), (2, -0.5), (4, 0.0)]}, 
                   {(1, 4): [(0, 0.167), (2, -0.333), (3, 0.167)]}, 
                   {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                   {(2, 4): [(0, 0.0), (1, 0.0), (3, 1.0)]}, 
                   {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        
        ## list unshielded triples
        UTs = []
        C = dseps2skel(true_seplist, n_nodes)
        logging.debug(f"Skeleton: {C.edges}")

        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"Unshielded triples: {UTs}")

        logging.debug(f"Immoralities: {immoralities}")
        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")
                
        self.assertEqual(set(est_immoralities), set(immoralities))


    def five_node_M_example1(self):
        scenario = "five_node_M_example1"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (1, 3), (0, 2), (2, 3)})
        true_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 2): [(1, 1.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, -0.5), (4, 0.0)]}, 
                    {(0, 4): [(1, 0.167), (2, -0.333), (3, 0.167)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, -1.0), (4, 0.0)]}, 
                    {(1, 4): [(0, 0.0), (2, -0.5), (3, 0.5)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                    {(2, 4): [(0, 0.0), (1, 0.0), (3, 1.0)]}, 
                    {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        ## list unshielded triples
        UTs = []
        C = dseps2skel(true_seplist, n_nodes)
        logging.debug(f"Skeleton: {C.edges}")

        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"Unshielded triples: {UTs}")

        logging.debug(f"Immoralities: {immoralities}")
        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")
                
        self.assertEqual(set(est_immoralities), set(immoralities))

    def five_node_colombo_example(self):
        scenario = "five_node_colombo_example_colliders"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  1,  1],
                            [ 0,  0,  1,  0,  1],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})
        true_svs = [{(0, 1): [(2, -0.333), (3, -0.333), (4, -0.333)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(0, 3): [(1, 0.0), (2, 0.0), (4, 0.0)]}, 
                    {(0, 4): [(1, 0.0), (2, 0.0), (3, 0.0)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0), (4, 0.0)]}, 
                    {(1, 3): [(0, 0.167), (2, 0.167), (4, -0.333)]}, 
                    {(1, 4): [(0, 0.0), (2, 0.0), (3, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, 
                    {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0)]}, 
                    {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        ## list unshielded triples
        UTs = []
        C = dseps2skel(true_seplist, n_nodes)
        logging.debug(f"Skeleton: {C.edges}")

        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"Unshielded triples: {UTs}")

        logging.debug(f"Immoralities: {immoralities}")
        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")
                
        self.assertEqual(set(est_immoralities), set(immoralities))

    def four_node_shapPC_example_incorrect_marginal(self):
        scenario = "four_node_shapPC_example_incorrect_marginal"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        true_svs = [{(0, 1): [(2, -0.5), (3, -0.5)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)

        ## change result of marginal independence test between x=0 and y=1
        true_seplist[(0,1)] = [((), 0), ((2,), 0), ((3,), 0), ((2, 3), 0)]
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        modified_svs = [{(0, 1): [(2, 0.0), (3, 0.0)]}, ## this cannot be deemed as a collider anymore
                        {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                        {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                        {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                        {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                        {(2, 3): [(0, 0.0), (1, 0.0)]}]

        self.assertEqual(list_of_svs, modified_svs)

    def four_node_shapPC_example_incorrect_total(self):
        scenario = "four_node_shapPC_example_incorrect_total"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        true_svs = [{(0, 1): [(2, -0.5), (3, -0.5)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)

        ## change result of marginal independence test between x=0 and y=1
        true_seplist[(0,1)] = [((), 1), ((2,), 0), ((3,), 0), ((2, 3), 1)]
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        modified_svs = [{(0, 1): [(2, 0.0), (3, 0.0)]}, ## this cannot be deemed as a collider anymore
                        {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                        {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                        {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                        {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                        {(2, 3): [(0, 0.0), (1, 0.0)]}]

        self.assertEqual(list_of_svs, modified_svs)

    def four_node_shapPC_example_mocked_tests(self):
        scenario = "four_node_shapPC_example_mocked_tests"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        true_svs = [{(0, 1): [(2, -0.5), (3, -0.5)]}, 
                    {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                    {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                    {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                    {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                    {(2, 3): [(0, 0.0), (1, 0.0)]}]
        
        true_seplist = find_all_d_separations_sets(G_true)

        ## change result of marginal independence test between x=0 and y=1
        true_seplist[(0,1)] = [((), 0.7), ((2,), 0.01), ((3,), 0.1), ((2, 3), 0.75)]
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        modified_svs = [{(0, 1): [(2, -0.02), (3, 0.07)]}, 
                        {(0, 2): [(1, 0.0), (3, 0.0)]}, 
                        {(0, 3): [(1, 0.5), (2, 0.5)]}, 
                        {(1, 2): [(0, 0.0), (3, 0.0)]}, 
                        {(1, 3): [(0, 0.0), (2, 0.0)]}, 
                        {(2, 3): [(0, 0.0), (1, 0.0)]}]

        self.assertEqual(list_of_svs, modified_svs)

    def detect_confounders(self):
        scenario = "detect_confounders"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (0, 2), (2, 3), (2, 4), (4, 3)})
        true_svs = []
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        est_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0)]}, 
                   {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0)]}, 
                   {(0, 3): [(1, 0.0), (2, 1.0), (4, 0.0)]}, 
                   {(0, 4): [(1, 0.0), (2, 1.0), (3, 0.0)]}, 
                   {(1, 2): [(0, 1.0), (3, 0.0), (4, 0.0)]}, ## confounder with no effect between the confounded variables
                   {(1, 3): [(0, 0.5), (2, 0.5), (4, 0.0)]}, ## both 0 and 2 close the path between 1 and 3
                   {(1, 4): [(0, 0.5), (2, 0.5), (3, 0.0)]}, ## both 0 and 2 close the path between 1 and 4
                   {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0)]}, ## direct path between 2 and 3
                   {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0)]}, ## direct path between 2 and 4
                   {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0)]}] ## direct path between 3 and 4, 2 is a confounder but not detected

    def detect_confounders2(self):
        scenario = "detect_confounders"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0, 0],
                            [ 0,  0,  0,  0,  0, 0],
                            [ 0,  0,  0,  1,  1, 0],
                            [ 0,  0,  0,  0,  0, 0],
                            [ 0,  0,  0,  1,  0, 0],
                            [ 0,  0,  1,  0,  1, 0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (0, 2), (2, 3), (2, 4), (4, 3), (5, 2), (5, 4)})
        true_svs = []
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        est_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 0 and 1
                   {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 0 and 2
                   {(0, 3): [(1, 0.0), (2, 0.667), (4, 0.167), (5, 0.167)]}, ## 2 and 4 have direct effects on 3 but 2 is mediator between 0 and 3. 5 is confounder of 2 and 4
                   {(0, 4): [(1, 0.0), (2, 0.5), (3, 0.0), (5, 0.5)]}, ## 2 is mediatior between 0 and 4, 5 is confounder of 2 and 4
                   {(0, 5): [(1, 0.0), (2, -0.333), (3, -0.333), (4, -0.333)]}, ## 2 is collider between 0 and 5, 3 and 4 are descendants of 2 hence colliding
                   {(1, 2): [(0, 1.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## 0 is confounder of 1 and 2, no direct path between 1 and 2
                   {(1, 3): [(0, 0.583), (2, 0.25), (4, 0.083), (5, 0.083)]}, ## 0 is confounder of 1 and 2, 2 mediates to 3, 5 is confounder of 2 and 4
                   {(1, 4): [(0, 0.667), (2, 0.167), (3, 0.0), (5, 0.167)]}, ## 0 is confounder of 1 and 2, 2 is mediator between 0 and 4, 5 is confounder of 2 and 4, 3 is descendant of 4
                   {(1, 5): [(0, 0.75), (2, -0.25), (3, -0.25), (4, -0.25)]}, ## 0 is confounder of 1 and 2, 2 is collider between 1 and 5, 3 and 4 are descendants of 2 hence colliding
                   {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 2 and 3
                   {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0), (5, 0.0)]}, ## direct arrow between 2 and 4
                   {(2, 5): [(0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0)]}, ## direct arrow between 2 and 5
                   {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0), (5, 0.0)]}, ## direct arrow between 3 and 4
                   {(3, 5): [(0, 0.0), (1, 0.0), (2, 0.5), (4, 0.5)]}, ## 2 is mediator between 5 and 3, 4 is mediator between 5 and 3
                   {(4, 5): [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]}] ## direct arrow between 5 and 4


    def detect_confounders2_data(self):
        scenario = "detect_confounders"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0, 0],
                            [ 0,  0,  0,  0,  0, 0],
                            [ 0,  0,  0,  1,  1, 0],
                            [ 0,  0,  0,  0,  0, 0],
                            [ 0,  0,  0,  1,  0, 0],
                            [ 0,  0,  1,  0,  1, 0]])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)
        expected = frozenset({(0, 1), (0, 2), (2, 3), (2, 4), (4, 3), (5, 2), (5, 4)})
        true_svs = []
        
        true_seplist = find_all_d_separations_sets(G_true)
        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})

        true_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 0 and 1
                   {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 0 and 2
                   {(0, 3): [(1, 0.0), (2, 0.667), (4, 0.167), (5, 0.167)]}, ## 2 and 4 have direct effects on 3 but 2 is mediator between 0 and 3. 5 is confounder of 2 and 4
                   {(0, 4): [(1, 0.0), (2, 0.5), (3, 0.0), (5, 0.5)]}, ## 2 is mediatior between 0 and 4, 5 is confounder of 2 and 4, 3 is collider but it's effect is neutralized by 2 and 5
                   {(0, 5): [(1, 0.0), (2, -0.333), (3, -0.333), (4, -0.333)]}, ## 2 is collider between 0 and 5, 3 and 4 are descendants of 2 hence colliding
                   {(1, 2): [(0, 1.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, ## 0 is confounder of 1 and 2, no direct path between 1 and 2
                   {(1, 3): [(0, 0.583), (2, 0.25), (4, 0.083), (5, 0.083)]}, ## 0 is confounder of 1 and 2, 2 mediates to 3, 5 is confounder of 2 and 4
                   {(1, 4): [(0, 0.667), (2, 0.167), (3, 0.0), (5, 0.167)]}, ## 0 is confounder of 1 and 2, 2 is mediator between 0 and 4, 5 is confounder of 2 and 4, 3 is descendant of 4
                   {(1, 5): [(0, 0.75), (2, -0.25), (3, -0.25), (4, -0.25)]}, ## 0 is confounder of 1 and 2, 2 is collider between 0 and 5, 3 and 4 are descendants of 2 hence colliding
                   {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0), (5, 0.0)]}, ## direct arrow between 2 and 3
                   {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0), (5, 0.0)]}, ## direct arrow between 2 and 4
                   {(2, 5): [(0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0)]}, ## direct arrow between 2 and 5
                   {(3, 4): [(0, 0.0), (1, 0.0), (2, 0.0), (5, 0.0)]}, ## direct arrow between 3 and 4
                   {(3, 5): [(0, 0.0), (1, 0.0), (2, 0.5), (4, 0.5)]}, ## 2 is mediator between 5 and 3, 4 is mediator between 5 and 3
                   {(4, 5): [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]}] ## direct arrow between 5 and 4
        
        data = simulate_discrete_data(num_of_nodes=n_nodes, sample_size=1000, truth_DAG_directed_edges=expected, random_seed=2024)

        cg_sk, cg_v, fitted = pc(data=data, alpha=0.05, indep_test='fisherz', uc_rule=3, uc_priority=3, show_progress=False, verbose=True)

        est_B = fitted.G.graph.T

        undirected_edges = fitted.find_undirected()

        est_DAG = (est_B > 0).astype(int)
        est_nx_G = nx.DiGraph(pd.DataFrame(est_DAG, columns=[f"X{i+1}" for i in range(est_DAG.shape[1])], index=[f"X{i+1}" for i in range(est_DAG.shape[1])]))
        est_seplist = find_all_d_separations_sets(est_nx_G)

        list_of_svs_from_G_est = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, est_seplist, verbose=False)
            logging.info(sv_list)
            list_of_svs_from_G_est.append({(x,y):sv_list})        

        est_svs_from_G_est = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, 
                              {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, 
                              {(0, 3): [(1, 0.0), (2, 1.0), (4, 0.0), (5, 0.0)]}, 
                              {(0, 4): [(1, 0.0), (2, 1.0), (3, 0.0), (5, 0.0)]}, 
                              {(0, 5): [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)]}, 
                              {(1, 2): [(0, 1.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, 
                              {(1, 3): [(0, 0.5), (2, 0.5), (4, 0.0), (5, 0.0)]}, 
                              {(1, 4): [(0, 0.5), (2, 0.5), (3, 0.0), (5, 0.0)]}, 
                              {(1, 5): [(0, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)]}, 
                              {(2, 3): [(0, 0.0), (1, 0.0), (4, 0.0), (5, 0.0)]}, 
                              {(2, 4): [(0, 0.0), (1, 0.0), (3, 0.0), (5, 0.0)]}, 
                              {(2, 5): [(0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0)]}, 
                              {(3, 4): [(0, -0.333), (1, -0.333), (2, -0.333), (5, 0.0)]}, 
                              {(3, 5): [(0, 0.0), (1, 0.0), (2, 0.0), (4, 0.0)]}, 
                              {(4, 5): [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]}]


        indep_test = 'fisherz'
        test_indep = CIT(data, indep_test)

        ### all tests
        est_sepset = np.empty((n_nodes, n_nodes), object)  # store the collection of sepsets
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Running independence test between {x} and {y}")
            for S in powerset(set(range(n_nodes)) - {x,y}):
                p = test_indep(x, y, S)
                logging.info(f"Independence test between {x} and {y} given {S} is {p}")
                append_value(est_sepset, x, y, (S, p))

        list_of_svs_est = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, est_sepset, verbose=False)
            logging.info(sv_list)
            list_of_svs_est.append({(x,y):sv_list})

        est_svs = [{(0, 1): [(2, 0.0), (3, 0.0), (4, 0.0), (5, -0.0)]}, 
                   {(0, 2): [(1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]}, 
                   {(0, 3): [(1, 0.033), (2, 0.418), (4, 0.002), (5, 0.006)]}, 
                   {(0, 4): [(1, 0.008), (2, 0.208), (3, -0.008), (5, 0.012)]}, 
                   {(0, 5): [(1, -0.168), (2, -0.04), (3, 0.017), (4, -0.017)]}, 
                   {(1, 2): [(0, 0.136), (3, 0.003), (4, 0.005), (5, -0.001)]}, 
                   {(1, 3): [(0, 0.1), (2, 0.008), (4, 0.01), (5, -0.027)]}, 
                   {(1, 4): [(0, -0.015), (2, -0.008), (3, 0.014), (5, -0.217)]}, 
                   {(1, 5): [(0, -0.011), (2, 0.0), (3, -0.002), (4, -0.01)]}, 
                   {(2, 3): [(0, 0.008), (1, -0.0), (4, -0.002), (5, -0.0)]}, 
                   {(2, 4): [(0, 0.001), (1, 0.0), (3, -0.002), (5, 0.001)]}, 
                   {(2, 5): [(0, -0.026), (1, -0.008), (3, -0.031), (4, 0.221)]}, 
                   {(3, 4): [(0, -0.006), (1, 0.003), (2, -0.1), (5, 0.051)]}, 
                   {(3, 5): [(0, 0.011), (1, -0.019), (2, -0.008), (4, 0.112)]}, 
                   {(4, 5): [(0, 0.0), (1, -0.0), (2, 0.0), (3, 0.0)]}] 

        ## list unshielded triples
        UTs = []
        C = dseps2skel(true_seplist, n_nodes)
        logging.debug(f"Skeleton: {C.edges}")

        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"Unshielded triples: {UTs}")

        logging.debug(f"Immoralities: {immoralities}")
        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")

        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value
        est_immoralities_data = []
        for x,z,y in UTs:
            svs_x_y = [sv for sv in list_of_svs_est if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities_data.append((x,z,y))
                logging.debug(f"Orienting {x} -> {z} <- {y}")

    def randomG_PC_facts(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
        scenario = "randomG_PC_facts"
        alpha = 0.01
        w_ranges = ((-2.0, -0.5), (0.5, 2.0))
        sem_type = "gauss"        
        
        output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
        facts_location_I = f"encodings/test_lps/{output_name}_I.lp"
        logger_setup(output_name)
        logging.info(f"===============Running {scenario}===============")
        logging.info(f"n_nodes={n_nodes}, edge_per_node={edge_per_node}, graph_type={graph_type}, seed={seed}")
        s0 = int(n_nodes*edge_per_node)
        if s0 > int(n_nodes*(n_nodes-1)/2):
            logging.info(f'{s0} is too many edges, setting s0 to the max:', int(n_nodes*(n_nodes-1)/2))
            s0 = int(n_nodes*(n_nodes-1)/2)

        random_stability(2024)
        B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=graph_type)
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
        expected = frozenset(set(G_true1.edges()))
        true_sepset = find_all_d_separations_sets(G_true)
        
        true_seplist = []
        list_of_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, true_sepset, verbose=False)
            logging.info(sv_list)
            list_of_svs.append({(x,y):sv_list})
            for i in true_sepset[(x,y)]:
                true_seplist.append((x,y,i[0],i[1]))

        print("Generating data...")
        ##--------- Continuous DAG --------------
        random_stability(seed)
        W_true = simulate_parameter(B_true, w_ranges=w_ranges)
        n = int(s0*n_nodes)

        ##---------DGP--------------
        if sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
            random_stability(seed)
            X = simulate_linear_sem(W_true, n, sem_type)
        elif sem_type in ['mim', 'mlp', 'gp', 'gp-add']:
            random_stability(seed)
            X = simulate_nonlinear_sem(W_true, n, sem_type)

        ##--------- PC --------------
        random_stability(seed)
        fitted = pc(data=X, alpha=alpha, indep_test='fisherz', uc_rule=3, uc_priority=2, 
                    selection='neg', show_progress=False, verbose=True)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph
        elapsed = fitted.PC_elapsed
        print('Time taken for SPC:', round(elapsed,2), 's')

        facts = []
        count_wrong = 0
        count_missing = 0
        for x,y in combinations(range(n_nodes), 2):
            I_from_data = list(set(fitted.sepset[x,y]))
            I_true = true_sepset[(x,y)]
            S_all = [t[0] for t in I_true]
            S_PC = [t[0] for t in I_from_data]
            S_missing = list(set(S_all) - set(S_PC))
            count_missing += len(S_missing)
            I_true_PC = [t for t in I_true if t[0] in S_PC]
            for i,p in I_from_data:
                PC_dep_type = 'I' if p > alpha else 'D'
                dep_type = 'I' if [test for test in I_true_PC if test[0]==i][0][1] > alpha else 'D'
                if dep_type != PC_dep_type:
                    count_wrong += 1
                facts.append((x,y,i,p, PC_dep_type, dep_type==PC_dep_type))
        
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({count_wrong/len(facts)*100:.2f}%)")

        est_svs = []
        for x,y in combinations(range(n_nodes), 2):
            logging.info(f"Calculating Shapley value for {x} and {y}")
            sv_list = shapley(x, y, n_nodes, fitted.sepset, verbose=False)
            logging.info(sv_list)
            est_svs.append({(x,y):sv_list})

        immoralities = get_immoralities(mount_adjacency_list(B_true))
        
        ## list unshielded true triples
        UTs_true = []
        C_true = dseps2skel(true_sepset, n_nodes)
        logging.debug(f"True Skeleton: {C_true.edges}")
        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C_true.has_edge(x,z) and C_true.has_edge(y,z) and not C_true.has_edge(x,y):
                    UTs_true.append((x,z,y))

        ## list unshielded estimated triples
        UTs = []
        C = dseps2skel(fitted.sepset, n_nodes)
        logging.debug(f"Est Skeleton: {C.edges}")
        for x,y in combinations(range(n_nodes), 2):
            for z in set(range(n_nodes)) - {x,y}:
                if C.has_edge(x,z) and C.has_edge(y,z) and not C.has_edge(x,y):
                    UTs.append((x,z,y))

        logging.debug(f"True Unshielded triples: {UTs_true}")
        logging.debug(f"Est Unshielded triples: {UTs}")

        logging.debug(f"Extra UTs: {set(UTs).difference(set(UTs_true))}")
        logging.debug(f"Missing UTs: {set(UTs_true).difference(set(UTs))}")
        logging.info(f"Number of missing UTs: {len(set(UTs_true).difference(set(UTs)))} ({len(set(UTs_true).difference(set(UTs)))/len(UTs_true)*100:.2f}%)")

        logging.debug(f"Immoralities ({len(immoralities)}): {immoralities}")
        ##apply decision rule: orient v-structure if candidate collider has negative Shapley value

        est_immoralities = []
        ## record how many sivs were wrong
        siv_stats = pd.DataFrame()
        count_wrong_sivs = 0
        count = 0
        for x,z,y in UTs:
            siv_stats_x_z_y = {}
            siv_stats_x_z_y["should_be_UTs"]= (x,z,y) in UTs_true ## take out the effect of a wrong skeleton decision
            ## Do we deem the candidate as collider?
            svs_x_y = [sv for sv in est_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z = [sv[1] for sv in svs_x_y if sv[0]==z][0]
            if sv_z < 0:
                est_immoralities.append((x,z,y))
                logging.debug(f"Oriented {x} -> {z} <- {y}")
            ## True Shapley values
            svs_x_y_true = [sv for sv in list_of_svs if list(sv.keys())[0] == (x,y)][0][(x,y)]
            sv_z_true = [sv[1] for sv in svs_x_y_true if sv[0]==z][0]

            if ((x,z,y) in UTs_true):
                assert ((x,z,y) in immoralities) == (sv_z_true < 0), f"Immoralities and True Shapley values do not match for {x} and {y} and {z}"

            max_len_S = max([len(f[2]) for f in facts if f[0]==x and f[1]==y])
            if sv_z_true*sv_z < 0:
                count_wrong_sivs += 1
                logging.debug(f"Wrong Shapley value for {x} and {y}: {sv_z_true} vs {sv_z}")
            siv_stats_x_z_y["X"]=x
            siv_stats_x_z_y["Y"]=y
            siv_stats_x_z_y["candidate"]=z
            siv_stats_x_z_y["SIV"]=sv_z_true
            siv_stats_x_z_y["SIV_est"]=sv_z
            ## correct if sign is the same
            siv_stats_x_z_y["Correct"]=sv_z_true*sv_z > 0

            siv_stats_x_z_y["deemed_collider"]=sv_z<0
            siv_stats_x_z_y["deemed_collider_pos"]=sv_z>0
            siv_stats_x_z_y["is_collider"]=(x,z,y) in immoralities

            ## how many facts were used in the shapley calculation
            siv_stats_x_z_y["N_facts_used"]= len([f for f in facts if f[0]==x and f[1]==y and z in list(f[2])])+1 ## +1 for the marginal independence test
            ## whether the marginal independence test was true
            siv_stats_x_z_y["marginal_correct"]= [f[5] for f in facts if f[0]==x and f[1]==y and f[2]==()][0]
            ## p-value of the marginal independence test
            siv_stats_x_z_y["marginal_p"]= [f[3] for f in facts if f[0]==x and f[1]==y and f[2]==()][0]
            ## whether the test involving the candidate was carried out
            siv_stats_x_z_y["candidate_tested"]= len([f[5] for f in facts if f[0]==x and f[1]==y and f[2]==(z,)])>0
            ## whether the test involving the candidate was correct
            if siv_stats_x_z_y["candidate_tested"]:
                siv_stats_x_z_y["candidate_tested_correct"]= [f[5] for f in facts if f[0]==x and f[1]==y and f[2]==(z,)][0]
                ## p-value of the test involving the candidate
                siv_stats_x_z_y["candidate_p"]= [f[3] for f in facts if f[0]==x and f[1]==y and f[2]==(z,)][0]
            else:
                siv_stats_x_z_y["candidate_tested_correct"]= None
                siv_stats_x_z_y["candidate_p"]= None
            ### NOTE: SIV is correct even if the test involving the candidate is wrong, as long as the marginal test is correct TODO: check this
            ## whether the biggest conditional independence test was true
            siv_stats_x_z_y["biggest_correct"]= [f[5] for f in facts if f[0]==x and f[1]==y and len(f[2])==max_len_S][0]
            ## p-value of the biggest conditional independence test
            siv_stats_x_z_y["biggest_p"]= [f[3] for f in facts if f[0]==x and f[1]==y and len(f[2])==max_len_S][0]

            ## how many facts were correct
            add_marginal = 1 if siv_stats_x_z_y["marginal_correct"] else 0
            siv_stats_x_z_y["N_facts_correct"]= len([f for f in facts if f[0]==x and f[1]==y and z in list(f[2]) and f[5]])+add_marginal

            ## record facts used
            siv_stats_x_z_y["facts_x_y"] = str([f for f in facts if f[0]==x and f[1]==y])

            ## record neighbor nodes to x and y
            siv_stats_x_z_y["neighbors_x"] = str(fitted.neighbors(x))
            siv_stats_x_z_y["neighbors_y"] = str(fitted.neighbors(y))
            ## record the true neighbors of x and y
            siv_stats_x_z_y["true_neighbors_x"] = str(list(G_true1.neighbors(x)))
            siv_stats_x_z_y["true_neighbors_y"] = str(list(G_true1.neighbors(y)))

            siv_stats = pd.concat([siv_stats, pd.DataFrame(siv_stats_x_z_y, index=[count])], ignore_index=True)
            count += 1

        logging.debug(f"Number of wrong Shapley values: {count_wrong_sivs}")
        ## record how many immoralities were correctly identified
        correct_immoralities = set(immoralities).intersection(set(est_immoralities))
        logging.debug(f"Correctly identified immoralities: {correct_immoralities}")

        # siv_stats.to_csv(f"{output_name}_siv_stats_{seed}.csv", index=False)

        return siv_stats

        # self.assertEqual(set(est_immoralities), set(immoralities))
        

class TestMetricsDAG(unittest.TestCase):

    def test_metrics_perfect(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], n_edges)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0)
        self.assertEqual(metrics['fpr'], 0)
        self.assertEqual(metrics['tpr'], 1)
        self.assertEqual(metrics['hhr'], 1)
        self.assertEqual(metrics['skp'], 1)
        self.assertEqual(metrics['skr'], 1)
        self.assertEqual(metrics['arrF1'], 1)
        self.assertEqual(metrics['skF1'], 1)

        self.assertEqual(metrics['precision'], 1)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['F1'], 1)

        self.assertEqual(metrics['shd'], 0)
        self.assertEqual(metrics['sid'], 0)
        
        ## calculate metrics for CPDAG
        metrics = DAGMetrics(dag2cpdag(B_est), B_true).metrics

        self.assertEqual(metrics['nnz'], 2)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0)
        self.assertEqual(metrics['fpr'], 0)
        self.assertEqual(metrics['tpr'], 1)
        self.assertEqual(metrics['hhr'], 1)
        self.assertEqual(metrics['skp'], 1)
        self.assertEqual(metrics['skr'], 1)
        self.assertEqual(metrics['arrF1'], 1)
        self.assertEqual(metrics['skF1'], 1)

        self.assertEqual(metrics['precision'], 1)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['F1'], 1)

        self.assertEqual(metrics['shd'], 0)
        self.assertEqual(metrics['sid'][0], 0)
        self.assertEqual(metrics['sid'][1], 12)

    def test_metrics_errors(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  1,  1,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  1,  1],
                           [ 0,  0,  0,  0,  1],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], n_edges)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0.2) #1/5
        self.assertEqual(metrics['fpr'], 0.2) #1/5
        self.assertEqual(metrics['tpr'], 0.8) #4/5
        self.assertEqual(metrics['hhr'], 0.8)
        self.assertEqual(metrics['skp'], 0.8)
        self.assertEqual(metrics['skr'], 0.8)
        self.assertEqual(metrics['arrF1'], 0.8)
        self.assertEqual(metrics['skF1'], 0.8)

        self.assertEqual(metrics['precision'], 0.8)
        self.assertEqual(metrics['recall'], 0.8)
        self.assertEqual(metrics['F1'], 0.8)

        self.assertEqual(metrics['shd'], 2)
        self.assertEqual(metrics['sid'], 2)
        
        ## calculate metrics for CPDAG
        metrics = DAGMetrics(dag2cpdag(B_est), B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], 0)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0.2) #1/5 1 extra undirected edges
        self.assertEqual(metrics['fpr'], 0.2) #same as fdr since sksize is correct
        self.assertEqual(metrics['tpr'], 0.0) #No directed edges
        self.assertEqual(metrics['hhr'], 0.0) #No directed edges
        self.assertEqual(metrics['skp'], 0.8) # 4 out of 5 undirected edges are correct
        self.assertEqual(metrics['skr'], 0.8) #same as skp since sksize is correct
        self.assertEqual(metrics['arrF1'], 0.0)
        self.assertEqual(metrics['skF1'], 0.8) 

        self.assertEqual(metrics['precision'], 0.8) #same as skp
        self.assertEqual(metrics['recall'], 0.8) #same as skr
        self.assertEqual(metrics['F1'], 0.8) #same as skF1
        
        self.assertEqual(metrics['shd'], 3) # 2 missing arrows (collider) and 1 extra edge
        self.assertEqual(metrics['sid'][0], 2)
        self.assertEqual(metrics['sid'][1], 15)


    def test_metrics_errors2(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  0,  0,  0,  0],
                           [ 1,  0,  0,  0,  0],
                           [ 1,  0,  0,  1,  1],
                           [ 0,  0,  0,  0,  1],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], n_edges)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0.6) # 2 reverse and 1 extra edge 3/5
        self.assertEqual(metrics['fpr'], 0.6) # same as fdr since sksisze is correct
        self.assertEqual(metrics['tpr'], 0.4) # 2 correct arrows out of 5 predicted
        self.assertEqual(metrics['hhr'], 0.4) # 2 correct arrows out of 5 true arrows
        self.assertEqual(metrics['skp'], 0.8) # 1 extra and 1 missing edge
        self.assertEqual(metrics['skr'], 0.8) # same as skp since sksize is correct
        self.assertEqual(metrics['arrF1'], 0.4)
        self.assertEqual(metrics['skF1'], 0.8)

        self.assertEqual(metrics['precision'], 0.4) ### same as tpr
        self.assertEqual(metrics['recall'], 0.4) ### same as hhr
        self.assertEqual(metrics['F1'], 0.4) ### same as arrF1

        self.assertEqual(metrics['shd'], 4) # 2 reverse, 1 extra and 1 missing edge
        self.assertEqual(metrics['sid'], 14)
        
        ## calculate metrics for CPDAG
        metrics = DAGMetrics(dag2cpdag(B_est), B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], 2)
        self.assertEqual(metrics['sksize'], n_edges)

        self.assertEqual(metrics['fdr'], 0.6) # 2 reverse and 1 extra edge
        self.assertEqual(metrics['fpr'], 0.6) # same as fdr since sksize is correct
        self.assertEqual(metrics['tpr'], 0.0) # 2 directed edges, both reversed
        self.assertEqual(metrics['hhr'], 0.0) # same as tpr since len(pred)==len(pos)
        self.assertEqual(metrics['arrF1'], 0.0)
        self.assertEqual(metrics['skp'], 0.8) # 4 out of 5 undirected edges are correct
        self.assertEqual(metrics['skr'], 0.8) # same as skp since sksize is correct
        self.assertEqual(metrics['skF1'], 0.8)

        self.assertEqual(metrics['precision'], 0.4) # average of tpr and skp
        self.assertEqual(metrics['recall'], 0.4)  # average of hhr and skr
        self.assertEqual(metrics['F1'], 0.4) # average of arrF1 and skF1
        
        self.assertEqual(metrics['shd'], 5) # 2 reversed, 2 missing arrows (collider) and 1 extra edge
        self.assertEqual(metrics['sid'][0], 14)
        self.assertEqual(metrics['sid'][1], 20)

    def test_metrics_errors3(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  0,  0,  0,  0],
                           [ 1,  0,  0,  0,  0],
                           [ 0,  0,  0,  1,  1],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], 3)
        self.assertEqual(metrics['sksize'], 3)

        self.assertEqual(metrics['fdr'], 0.6667) # 1 reverse and 1 extra edge out of the 3 predicted
        self.assertEqual(metrics['fpr'], 0.4) # 1 reverse and 1 extra edge out of the 5 true edges
        self.assertEqual(metrics['tpr'], 0.3333) # 1 correct arrow out of 3 predicted
        self.assertEqual(metrics['hhr'], 0.2) # 1 correct arrow out of 5 true arrows
        self.assertEqual(metrics['arrF1'], 0.25)
        self.assertEqual(metrics['skp'], 0.6667) # 2 correct edges out of 3 predicted
        self.assertEqual(metrics['skr'], 0.4) # 2 correct edges out of 5 true edges
        self.assertEqual(metrics['skF1'], 0.5)

        self.assertEqual(metrics['precision'], 0.3333) ### same as tpr
        self.assertEqual(metrics['recall'], 0.2) ### same as hhr
        self.assertEqual(metrics['F1'], 0.25) ### same as arrF1

        self.assertEqual(metrics['shd'], 5) # 1 reverse, 1 extra and 3 missing edges
        self.assertEqual(metrics['sid'], 16)
        
        ## calculate metrics for CPDAG
        metrics = DAGMetrics(dag2cpdag(B_est), B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], 0) # no directed edges (no colliders)
        self.assertEqual(metrics['sksize'], 3)

        self.assertEqual(metrics['fdr'], 0.3333) # 1 extra edge out of the 3 predicted
        self.assertEqual(metrics['fpr'], 0.2) # 1 extra edge out of the 5 true edges
        self.assertEqual(metrics['tpr'], 0.0) # no directed edges
        self.assertEqual(metrics['hhr'], 0.0) # no directed edges
        self.assertEqual(metrics['arrF1'], 0.0)
        self.assertEqual(metrics['skp'], 0.6667) # 2 out of 3 undirected edges are correct
        self.assertEqual(metrics['skr'], 0.4) # 2 out of 5 undirected edges are correct
        self.assertEqual(metrics['skF1'], 0.5)

        self.assertEqual(metrics['precision'], 0.6667) # same as skp
        self.assertEqual(metrics['recall'], 0.4)  # same as skr
        self.assertEqual(metrics['F1'], 0.5) # same as skF1
        
        self.assertEqual(metrics['shd'], 5) # 2 missing orientation, 2 missing edges and 1 extra edge
        self.assertEqual(metrics['sid'][0], 9)
        self.assertEqual(metrics['sid'][1], 17)

    def test_metrics_cpdag_input(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  -1, -1,  0,  0],
                           [-1,  0,  0,  1,  0],
                           [-1,  0,  0,  1,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['nnz'], 2)
        self.assertEqual(metrics['sksize'], 4)

        self.assertEqual(metrics['fdr'], 0) # no errors
        self.assertEqual(metrics['fpr'], 0) # no errors
        self.assertEqual(metrics['tpr'], 1) # all directed edges are correct
        self.assertEqual(metrics['hhr'], 1) # all directed edges identified in CPDAG
        self.assertEqual(metrics['arrF1'], 1)
        self.assertEqual(metrics['skp'], 1) # 4 correct edges out of 4 predicted
        self.assertEqual(metrics['skr'], 0.8) # 4 correct edges out of 5 true edges
        self.assertEqual(metrics['skF1'], 0.8889)

        self.assertEqual(metrics['precision'], 1) ### same as skp
        self.assertEqual(metrics['recall'], 0.8) ### same as skr
        self.assertEqual(metrics['F1'], 0.8889) ### same as arrF1

        self.assertEqual(metrics['shd'], 1) # 1 missing edge
        self.assertEqual(metrics['sid'][0], 4)
        self.assertEqual(metrics['sid'][1], 11)
        
        ## calculate metrics for CPDAG
        metrics_cp = DAGMetrics(dag2cpdag(B_est), B_true).metrics

        self.assertEqual(metrics_cp, metrics)

    def test_immorality_metrics(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  1,  1,  0,  0],
                           [ 0,  0,  0,  1,  0],
                           [ 0,  0,  0,  1,  1],
                           [ 0,  0,  0,  0,  1],
                           [ 0,  0,  0,  0,  0]])
        
        G_true = nx.DiGraph(B_true)
        true_edges = list(G_true.edges)
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['immoral_prec'], 1.)
        self.assertEqual(metrics['immoral_rec'], 1.)
        self.assertEqual(metrics['immoral_F1'], 1.)
        self.assertEqual(metrics['immoral_UT_prec'], 1.)
        self.assertEqual(metrics['immoral_UT_rec'], 1.)
        self.assertEqual(metrics['immoral_UT_F1'], 1.)

    def test_immorality_metrics_errors(self):
        ## true DAG
        B_true = np.array( [[ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  0,  1,  0,  0],
                           [ 0,  0,  1,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['immoral_prec'], 1.)
        self.assertEqual(round(metrics['immoral_rec'],4), 0.1667) # 1 out of 6 true immoralities
        self.assertEqual(round(metrics['immoral_F1'],4), 0.2857)
        self.assertEqual(metrics['immoral_UT_prec'], 1.)
        self.assertEqual(metrics['immoral_UT_rec'], 1.)
        self.assertEqual(metrics['immoral_UT_F1'], 1.)

    def test_immorality_metrics_errors2(self):
        ## true DAG
        B_true = np.array( [[ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  0,  1,  0,  0],
                           [ 0,  0,  1,  0,  0],
                           [ 0,  0,  0, -1, -1],
                           [ 0,  0, -1,  0,  0],
                           [ 0,  0, -1,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['immoral_prec'], 1.)
        self.assertEqual(round(metrics['immoral_rec'],4), 0.1667) # 1 out of 6 true immoralities
        self.assertEqual(round(metrics['immoral_F1'],4), 0.2857)
        self.assertEqual(metrics['immoral_UT_prec'], 1.)
        self.assertEqual(round(metrics['immoral_UT_rec'],4), 0.1667) # 1 out of 6 correct UTs
        self.assertEqual(round(metrics['immoral_UT_F1'],4), 0.2857)

    def test_immorality_metrics_errors3(self):
        ## true DAG
        B_true = np.array( [[ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  1,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  0, -1,  0,  0],
                           [ 0,  0, -1,  0,  0],
                           [-1, -1,  0, -1, -1],
                           [ 0,  1, -1,  0,  0],
                           [ 0,  1, -1,  0,  0]])
        
        ## calculate metrics
        metrics = DAGMetrics(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['immoral_prec'], 0.)
        self.assertEqual(metrics['immoral_rec'], 0.) # 1 out of 6 true immoralities
        self.assertEqual(metrics['immoral_F1'], 0.)
        self.assertEqual(metrics['immoral_UT_prec'], 0.)
        self.assertEqual(metrics['immoral_UT_rec'], 0.)
        self.assertEqual(metrics['immoral_UT_F1'], 0.)

class TestMetricsPAG(unittest.TestCase):

    def test_dag2pag(self):

        ########################################################
        ##
        ##       Example 2: Zhang (2006), Fig. 5.2, p.198
        ##                  Dissertation
        ##
        ########################################################

        ### From pcalg R package
        ##   A B C D E F
        ## A . . . 1 . .
        ## B . . . 1 1 .
        ## C . . . 1 . .
        ## D . . . . 1 .
        ## E . . . . . .
        ## F . . 1 1 . .

        # corr.pag2 <- rbind(c(.,.,.,2,.),
        #                    c(.,.,.,2,2),
        #                    c(.,.,.,2,.),
        #                    c(1,1,1,.,2),
        #                    c(.,3,.,3,.))

        ### Arrow marks legend
        # amat[i,j] = 0 iff no edge btw i,j
        # amat[i,j] = 1 iff i *-o j
        # amat[i,j] = 2 iff i *-> j
        # amat[i,j] = 3 iff i *-- j

        ## Possible arrow configurations
        ## o-o, o->, -->, or no edge
        ## 1-1, 1-2, 3-2, or 0-0

        B_true = np.array( [[ 0,  0,  0,  1,  0, 0],
                            [ 0,  0,  0,  1,  1, 0],
                            [ 0,  0,  0,  1,  0, 0],
                            [ 0,  0,  0,  0,  1, 0],
                            [ 0,  0,  0,  0,  0, 0],
                            [ 0,  0,  1,  1,  0, 0]])

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])
        
        latent_vars = [5]
        B_true_obs = np.delete(np.delete(B_true, latent_vars, 0), latent_vars, 1)

        tetrad_B_true = tr.adj_matrix_to_graph(B_true_obs)
        
        pag = tr.graph_to_matrix(ts.utils.DagToPag(tetrad_B_true).convert())

        self.assertSequenceEqual(pag.to_numpy().tolist(), P_true.tolist())

    def test_shd_pag(self):

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        shd = PAGMetrics(P_true, P_est).metrics['shd']                                  

        self.assertEqual(shd, 0) ## equal graphs


        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0], 
                            [ 1,  1,  3,  0,  2],##
                            [ 0,  3,  0,  3,  0]])   
                                         
        shd = PAGMetrics(P_true, P_est).metrics['shd'] 

        self.assertEqual(shd, 1) ## tail missed o- instead of --


        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  1],##
                            [ 0,  3,  0,  1,  0]])   
                                         ##
        shd = PAGMetrics(P_true, P_est).metrics['shd'] 

        self.assertEqual(shd, 2) ## both tail and arrow errors

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  1], ##
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   
                                         
        shd = PAGMetrics(P_true, P_est).metrics['shd'] 

        self.assertEqual(shd, 0) ## not possible to have o only on one side, no error


        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0], 
                            [ 1,  1,  1,  0,  3],
                            [ 0,  3,  0,  3,  0]])   
                                         
        shd = PAGMetrics(P_true, P_est).metrics['shd'] 

        self.assertEqual(shd, 0) ## --- is not possible, it would be o-o, no error

    def test_adj_pr_re(self):

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['adj_prec', 'adj_rec', 'adj_F1']]

        self.assertEqual(pr, 1)
        self.assertEqual(re, 1)
        self.assertEqual(F1, 1)

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  1,  0,  2,  0],
                            [ 1,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['adj_prec', 'adj_rec', 'adj_F1']]

        self.assertEqual(pr, 0.8333333333333334) ## 1 extra edge
        self.assertEqual(re, 1)
        self.assertEqual(F1, 0.9090909090909091)


        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  1,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 0,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['adj_prec', 'adj_rec', 'adj_F1']]

        self.assertEqual(pr, 1)
        self.assertEqual(re, 0.8) ## 1 missing edge
        self.assertEqual(F1, 0.888888888888889)

    def test_arr_pr_re(self):
            
        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['arr_prec', 'arr_rec', 'arr_F1']]

        self.assertEqual(pr, 1)
        self.assertEqual(re, 1)
        self.assertEqual(F1, 1)

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  1,  0,  1,  0],
                            [ 1,  0,  0,  1,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  2,  0]])
        
        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['arr_prec', 'arr_rec', 'arr_F1']]

        self.assertEqual(pr, 0.75) ## 1 extra edge not taken into account since undirected, 3 out of 4 arrows are correct
        self.assertEqual(re, 0.6) #two missed arrows out of 5, recall is 3/5
        self.assertEqual(F1, 0.6666666666666665)

    def test_tail_pr_re(self):

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])   

        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['tail_prec', 'tail_rec', 'tail_F1']]

        self.assertEqual(pr, 1)
        self.assertEqual(re, 1)
        self.assertEqual(F1, 1)

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  1,  0,  2,  0],
                            [ 3,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 3,  3,  1,  0,  2],
                            [ 0,  3, 0,  1,  0]])
        
        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['tail_prec', 'tail_rec', 'tail_F1']]

        self.assertEqual(pr, 0.3333333333333333) ## 2 extra tail out of 3 predicted
        self.assertEqual(re, 0.5) ## 1 missed tail out of 2, recall is 2/4
        self.assertEqual(F1, 0.4)

    def bidirected_pr_re(self):

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  1,  0,  2],
                            [ 0,  3,  0,  3,  0]])
        
        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['bid_prec', 'bid_rec', 'bid_F1']]

        self.assertEqual(pr, 1)
        self.assertEqual(re, 1)
        self.assertEqual(F1, 1)

        P_true = np.array( [[ 0,  0,  0,  2,  0],
                            [ 0,  0,  0,  2,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  2,  2,  0,  2],
                            [ 0,  3,  0,  3,  0]])

        P_est = np.array( [ [ 0,  1,  0,  1,  0],
                            [ 1,  0,  0,  1,  2],
                            [ 0,  0,  0,  2,  0],
                            [ 1,  1,  2,  0,  2],
                            [ 0,  3,  0,  2,  0]])
        
        pr, re, F1 = [PAGMetrics(P_true, P_est).metrics.get(key) for key in ['bid_prec', 'bid_rec', 'bid_F1']]

        self.assertEqual(pr, 0.5) ## 1 extra bidirected edge out of 2 predicted
        self.assertEqual(re, 0.5) ## 1 missed bidirected edge out of 2, recall is 1/2
        self.assertEqual(F1, 0.5)


start = datetime.now()

# TestShapley().collider()
# TestShapley().chains_confounder()
# TestShapley().triangle()
# TestShapley().disconnected()

# TestShapley().four_node_shapPC_example()
# TestShapley().double_chain_shapPC_example()
# TestShapley().five_node_colombo_example()
# TestShapley().five_node_M_example()
# TestShapley().five_node_sprinkler_example()

# TestDecisionRule().four_node_shapPC_example()
# TestDecisionRule().five_node_M_example()
# TestDecisionRule().five_node_M_example1()
# TestDecisionRule().five_node_colombo_example()
# TestDecisionRule().four_node_shapPC_example_incorrect_marginal()
# TestDecisionRule().four_node_shapPC_example_incorrect_total()
# TestDecisionRule().four_node_shapPC_example_mocked_tests()


# TestDecisionRule().detect_confounders()
# TestDecisionRule().detect_confounders2()
# TestDecisionRule().detect_confounders2_data()


### Test Metrics
TestMetricsDAG().test_metrics_perfect()
TestMetricsDAG().test_metrics_errors()
TestMetricsDAG().test_metrics_errors2()
TestMetricsDAG().test_metrics_errors3()
TestMetricsDAG().test_metrics_cpdag_input()

TestMetricsDAG().test_immorality_metrics()
TestMetricsDAG().test_immorality_metrics_errors()
TestMetricsDAG().test_immorality_metrics_errors2()
TestMetricsDAG().test_immorality_metrics_errors3()

# TestMetricsPAG().test_dag2pag()
# TestMetricsPAG().test_shd_pag()
# TestMetricsPAG().test_adj_pr_re()
# TestMetricsPAG().test_arr_pr_re()
# TestMetricsPAG().test_tail_pr_re()


logging.info(f"Total time={str(datetime.now()-start)}")

# siv_stats_full = pd.DataFrame()
# random_stability(2023)
# seeds_list = np.random.randint(0, 10000, (10, )).tolist()

# for s in [1,2,4]:
#     for seed in seeds_list:
#         siv_stats = TestDecisionRule().randomG_PC_facts(10, s, "ER", seed)
#         siv_stats["seed"] = np.repeat(seed, len(siv_stats))
#         siv_stats["density"] = np.repeat(s, len(siv_stats))
#         siv_stats_full = pd.concat([siv_stats_full, siv_stats], ignore_index=True)
# siv_stats_full.to_csv(f"siv_stats_full.csv", index=False)
