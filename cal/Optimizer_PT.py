import itertools
import math
import multiprocessing
from collections import defaultdict
from itertools import product, combinations
from colorama import Fore

from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import multiprocessing

import numpy as np
import random
import time
from copy import copy
from scipy.optimize import minimize
from functools import reduce
from operator import mul
from Levenshtein import distance as levenshtein_distance
from pm4py.algo.evaluation.earth_mover_distance.variants import pyemd
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

from model.ProcessTree import Seq, Choice, Parallel, Loop, BinaryChoice, BinaryParallel


class Optimizer_PT:

    def __init__(self, ev, pt):
        self.ev, self.pt = ev, pt
        self.dtmc = pt.to_dtmc()
        self.ev_language = self.ev.get_language_pm4py()

    def emd_norm_lev(self, l1, l2):

        nc1, nc2 = 0, 0
        words = set(l1) & set(l2)
        for w in words:
            if w in l1:
                nc1 = nc1 + l1[w]
            if w in l2:
                nc2 = nc2 + l2[w]

        hist1 = np.zeros(len(words))
        hist2 = np.zeros(len(words))
        distance = np.zeros((len(words), len(words)))
        for x, xw in enumerate(words):
            for y, yw in enumerate(words):
                if xw in l1:
                    hist1[x] = l1[xw] / nc1
                if yw in l2:
                    hist2[y] = l2[yw] / nc2
                if len(xw) == 0 or len(yw) == 0:
                    distance[x, y] = 1.
                else:
                    maxl = float(max(len(xw), len(yw)))
                    distance[x, y] = float(levenshtein_distance(xw, yw)) / maxl

        return pyemd.emd(hist1, hist2, distance)

    def ER1(self, ev_language, pt_language):

        def H0(x):
            if int(x) == 0 or int(x) == 1:
                return 0
            return (-x * math.log2(x) * x) - ((1 - x) * math.log2(x) * (1 - x))

        def J(trace):
            if pt_language[trace] != 0:
                return -math.log2(pt_language[trace])
            else:
                return (1 + len(ev_language)) * math.log2(1 + len(self.ev.activities))

        h = H0(sum([ev_language[trace] for trace in ev_language if pt_language[trace] != 0]))
        j = sum([self.ev_language[trace] * J(trace) for trace in self.ev_language])

        return 1 - (1 / (h + j))

    def KLD(self, ev_language, pt_language):
        lh = 0
        for trace in pt_language:
            lh += ev_language[trace] * np.log(pt_language[trace])
        return -lh

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def get_simulated_slanguage(self, p, n):

        def simulate_rec(node):

            if node.is_leaf():
                if node.value == "":
                    return ()
                else:
                    return (node.value,)

            elif type(node) is Seq:
                selected_perm = random.choices(list(node.O.keys()), weights=list(node.O.values()), k=1)[0]
                tr, c = (), 0
                while c < len(node.children):
                    tr += simulate_rec(node.children[int(selected_perm[c])])
                    c += 1
                return tr

            elif type(node) is Choice:
                selected_child = random.choices(node.children, weights=p[node.wi_start:node.wi_end + 1], k=1)[0]
                return simulate_rec(selected_child)

            elif type(node) is BinaryChoice:
                selected_child = random.choices(node.children, weights=[p[node.wi], 1 - p[node.wi]], k=1)[0]
                return simulate_rec(selected_child)

            elif type(node) is Parallel or type(node) is BinaryParallel:
                traces_children = [simulate_rec(child) for child in node.children]
                if type(node) is BinaryParallel:
                    temp_p = copy([p[node.wi], 1 - p[node.wi]])
                else:
                    temp_p = copy(p[node.wi_start:node.wi_end + 1])
                ind = [0 for _ in range(len(traces_children))]
                lengths = [len(trace_children) for trace_children in traces_children]
                for l in range(len(lengths)):
                    if lengths[l] == 0:
                        temp_p[l] = 0
                        normalized_temp_p = [x / sum(temp_p) for x in temp_p]
                        temp_p = normalized_temp_p
                tr = ()
                while ind != lengths:
                    i = random.choices([i for i in range(len(temp_p))], weights=temp_p, k=1)[0]
                    tr += (traces_children[i][ind[i]],)
                    ind[i] += 1
                    if ind[i] == lengths[i] and ind != lengths:
                        temp_p[i] = 0
                        normalized_temp_p = [x / sum(temp_p) for x in temp_p]
                        temp_p = normalized_temp_p
                return tr

            elif type(node) is Loop:
                m = np.random.geometric(1 - p[node.wi], size=1)[0]
                tr = ()
                for _ in range(m - 1):
                    tr += simulate_rec(node.children[0]) + simulate_rec(node.children[1])
                tr += simulate_rec(node.children[0])
                return tr

            return None

        traces = {}
        for _ in range(n):
            trace = simulate_rec(self.pt.root)
            if trace not in traces:
                traces[trace] = 1
            else:
                traces[trace] += 1
        for trace in self.ev_language:
            if trace not in traces:
                traces[trace] = 0
        return {key: value / n for key, value in traces.items()}

    def compute_sim(self, p, n):
        pt_language = self.get_simulated_slanguage(p, n)
        return self.emd_norm_lev(self.ev_language, pt_language)

    def compute_exact(self, p, d):
        self.dtmc.compute_dtmc(p)
        pt_language = self.dtmc.unfold()
        if d == "KLD":
            return self.KLD(self.ev_language, pt_language)
        elif d == "rEMD":
            return self.emd_norm_lev(self.ev_language, pt_language)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def compute_sim_starting_prob(self, i, n):
        np.random.seed(i)
        p = (1 - 0.001) * np.random.rand(self.pt.size_p) + 0.001
        rEMD = self.compute_sim(p, n)
        return (p, rEMD)

    def compute_exact_starting_prob(self, i):
        np.random.seed(i)
        p = (1 - 0.001) * np.random.rand(self.pt.size_p) + 0.001
        self.dtmc.compute_dtmc(p)
        pt_language = self.dtmc.unfold()
        rEMD = self.emd_norm_lev(self.ev_language, pt_language)
        KLD = self.KLD(self.ev_language, pt_language)
        return (p, KLD, rEMD)

    def estimate_sim(self, obj_function="rEMD", method="Powell", n=10000, starting_prob=None, nw0=10):
        start_time = time.time()
        print("OPT-based SPD on Process tree with simulation | Minimising " + obj_function + " with " + method + " | n: " + str(n) + "\n")

        bnds = ((0.001, 0.999),) * self.pt.size_p
        if starting_prob is None:
            print("####################################################################################################")
            print("Looking for starting points (" + str(nw0) + " vectors) started at " + str(time.time() - start_time) + " seconds")
            (p_before, rEMD_before) = [tup for tup in sorted([self.compute_sim_starting_prob(i, n) for i in range(nw0)], key=lambda x:x[1])[0]]
            print("Looking for starting points (" + str(nw0) + " vectors) ended at " + str(time.time() - start_time) + " seconds")
            print("####################################################################################################\n")
        else:
            p_before = starting_prob

            rEMD_before = self.compute_sim(starting_prob, 10000)

        print("####################################################################################################")
        print("Optimising vectors started at " + str(time.time() - start_time) + " seconds")
        start_time_opt = time.time()
        min = minimize(self.compute_sim, p_before, (n),  method=method, bounds=bnds, options={'maxiter': np.inf, 'disp': False})
        time_opt = time.time() - start_time_opt
        print("Optimising vectors ended at " + str(time.time() - start_time) + " seconds")
        print("####################################################################################################\n")

        print("####################################################################################################")
        print(f"Minimising {obj_function} with {method} | n: {n} in {time_opt} seconds ({min.nit} iterations)")
        print(f"\tStarting values: {list(p_before)}")
        print(f"\t\trEMD value with starting values: {rEMD_before:<6}")
        print(f"\tOptimized values: {list(min.x)}")
        print(f"\t\trEMD value with optimized values: {min.fun:<6}")
        print("####################################################################################################\n")

        print("Execution time: " + str(time.time() - start_time) + " seconds")

        return True

    def estimate_exact(self, obj_function="rEMD", method="Powell", starting_prob=None, nw0=100):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        start_time = time.time()

        print("OPT-based SPD on Process tree | Minimising " + obj_function + " with " + method + " | nw0: " + str(nw0) + "\n")

        bnds = ((0.001, 0.999),) * self.pt.size_p
        if starting_prob is None:
            print("####################################################################################################")
            print("Looking for starting points (" + str(nw0) + " vectors) started at " + str(time.time() - start_time) + " seconds")
            (starting_prob, KLD_before, rEMD_before) = [tup for tup in sorted([self.compute_exact_starting_prob(i) for i in range(nw0)], key=lambda x:x[1 if obj_function == "KLD" else 2])[0]]
            print("Looking for starting points (" + str(nw0) + " vectors) ended at " + str(time.time() - start_time) + " seconds")
            print("####################################################################################################\n")
        else:
            self.dtmc.compute_dtmc(starting_prob)
            pt_language = self.dtmc.unfold()
            KLD_before = self.KLD(self.ev_language, pt_language)
            rEMD_before = self.emd_norm_lev(self.ev_language, pt_language)

        print("####################################################################################################")
        print("Optimising vectors started at " + str(time.time() - start_time) + " seconds")
        start_time_opt = time.time()
        min = minimize(self.compute_exact, starting_prob, obj_function, bounds=bnds, method=method, options={'disp': False})
        time_opt = time.time() - start_time_opt
        self.dtmc.compute_dtmc(min.x)
        pt_language = self.dtmc.unfold()
        KLD_after = self.KLD(self.ev_language, pt_language)
        rEMD_after = self.emd_norm_lev(self.ev_language, pt_language)
        print("Optimising vectors ended at " + str(time.time() - start_time) + " seconds")
        print("####################################################################################################\n")

        print(f"KLD / rEMD value with starting values: {KLD_before:<6} / {rEMD_before:<6}")
        print(f"\t\t\tKLD / rEMD value with optimized values ({method:<8}): {KLD_after:<6} / {rEMD_after:<6} in {time_opt:<10.2f} seconds ({min.nit:<4} iterations)")

        print("Execution time: " + str(time.time() - start_time) + " seconds\n")

        return True
