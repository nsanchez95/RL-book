from dataclasses import dataclass
from typing import Tuple, Dict, TypeVar,Iterable, Iterator, Callable
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping,MarkovDecisionProcess, TransitionStep
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant, Distribution, Choose
from rl.function_approx import FunctionApprox, Tabular
from rl.iterate import last
from itertools import islice
from scipy.stats import poisson
from rl.returns import returns
import rl.markov_decision_process as markov_decision_process
import numpy as np
S = TypeVar('S')
A = TypeVar('A')
X = TypeVar('X')


class LSTD(FunctionApprox[X]):
    A_matrix:np.ndarray
    b_vector:np.ndarray
    w:np.ndarray
    feat_func: Callable[[X],np.ndarray]

    def __init__(
        self,
        num_feats: int,
        feat_func: Callable[[X],np.ndarray],
    ):
        self.feat_func = feat_func
        self.A_matrix: np.ndarray = np.zeros((num_feats,num_feats))
        self.b_vector: np.ndarray = np.zeros(num_feats)
        self.w: np.ndarray = np.zeros(num_feats)
        super().__init__()

    ########### MUST MEET FUNCTIONAPPROX SPEC ###############
    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> FunctionApprox[X]:
        return self

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> FunctionApprox[X]:
        return self

    def representational_gradient(self, x_value: X) -> FunctionApprox[X]:
        return self

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array([np.dot(self.weights, self.feat_func(x)) for x in x_values_seq])

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        return False

    ########### ADDITIONAL FUNCTIONS #######################
    def train_up(self, s1: X, rew: float, s2: X, gamma:float, term: bool) -> None:
        s1_feat = self.feat_func(s1)
        s2_feat = self.feat_func(s2)
        self.A_matrix += s1_feat.T.dot(s1_feat - gamma*s2_feat*(1-float(int(term))))
        self.b_vector+= rew*s1_feat*gamma

    def update_weights(self, normalization = 1.0, reg_val = 0.001) -> None:
        self.A_matrix *=normalization 
        self.b_vector *=normalization
        self.A_matrix += reg_val*np.eye(self.b_vector.shape[0])
        self.w = np.linalg.inv(self.A_matrix).dot(self.b_vector)

    def reset_count(self) -> None:
        self.A_matrix = np.zeros((num_feats,num_feats))
        self.b_vector =np.zeros(num_feats)

    def evaluate_vf(self, s: X) -> float:
        s_feat = feat_func(s)
        return np.dot(self.w, s_feat)


def LSPI(
        replay_memory: Distribution[TransitionStep[S, A]],
        mdp: MarkovDecisionProcess[S,A],
        feat_funct: Callable[Tuple[S,A],np.ndarray],
        dim_feat: int,
        batch_size: int,
        eps: float,
        gamma: float
) -> Iterable[FunctionApprox]:
    lstqd_approx = LSTD(num_feats =dim_feat, feat_func = feat_func, gamma = gamma) 
    while(True):
        curr_policy = markov_decision_process.policy_from_q(lstqd_approx, mdp, eps)
        for tr_step in replay_memory.sample_n(batch_size):
            next_action = curr_policy.act(tr_step.next_state)
            lstsq.train_up((tr.state,tr.action),tr.reward,(tr.next_state, next_action),\
                gamma, mdp.is_terminal(next_state))
        lstsq.update_weights()
        yield lstqd_approx

if __name__ == '__main__':
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )


    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result


    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    print(opt_policy_pi)
    print()

    num_traces = 1000
    start_states = Choose(set(si_mdp.non_terminal_states))



