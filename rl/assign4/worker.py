from dataclasses import dataclass
from typing import Tuple, Dict, List
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from rl.dynamic_programming import policy_iteration, value_iteration
from rl.dynamic_programming import almost_equal_vf_pis, almost_equal_vfs
from rl.iterate import converge
from scipy.stats import poisson
import numpy as np
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt

w = np.array([5,3,7,9,11])
alpha = 0.3
gamma  = 0.9
num_jobs = len(w)
v_u = np.zeros(num_jobs)
v_u_n = np.ones(num_jobs)*(1./(1.-gamma))
while(np.max(np.abs(v_u_n - v_u)) > 0.001):
    v_u = v_u_n
    accept_col = (np.log(w) + alpha*gamma/num_jobs*np.sum(v_u))/(1-gamma*(1-alpha))
    reject_col = (np.log(w[0]) + gamma/num_jobs*np.sum(v_u))*np.ones(num_jobs)
    v_u_n = np.max(np.vstack([accept_col, reject_col]).T, axis = 1)

v_e = (np.log(w)+alpha*gamma/num_jobs*np.sum(v_u))/(1-gamma*(1-alpha))