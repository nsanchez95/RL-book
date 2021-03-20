from dataclasses import dataclass
from typing import Tuple, Dict, TypeVar,Iterable, Iterator, Callable
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping,MarkovDecisionProcess
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



class LSTD:
    A:np.ndarray
    b:np.ndarray
    w:np.ndarray
    update_count : int
    update_freq : int
    gamma: float

    def __init__(
        self,
        num_feats: int,
        update_freq: int,
        gamma: float
    ):
        self.gamma: float = gamma
        self.update_freq: int = update_freq
        self.A: np.ndarray = np.zeros((num_feats,num_feats))
        self.b: np.ndarray = np.zeros(num_feats)
        super().__init__(self.get_action_transition_reward_map())

    def train(self, s1_feat:np.ndarray, rew: float, s2_feat: np.ndarray) -> None:
        self.A += np.outer(s1_feat, s1_feat - self.gamma*s2_feat)
        self.b+= rew*s2_feat

    def update_weights(self) -> None:
        self.weights = np.linalg.lstsq(self.A,self.b)

    def evaluate_vf(self, s_feat1: np.ndarray) -> float:
        return np.dot(self.weights, s_feat1)

def mc_control(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        gamma: float,
        eps: Callable[[int], float],
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    p = markov_decision_process.policy_from_q(q, mdp)
    trace_count = 1
    while True:
        trace: Iterable[markov_decision_process.TransitionStep[S, A]] =\
            mdp.simulate_actions(states, p)

        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, gamma, tolerance)
        )

        p = markov_decision_process.policy_from_q(q, mdp, eps(trace_count))
        trace_count += 1
        yield q


def td_sarsa(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        eps: Callable[[int], float],
        gamma: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    state_distribution = states
    trace_count = 1
    while True:
        p = markov_decision_process.policy_from_q(q, mdp, eps(trace_count))
        state = state_distribution.sample()
        action = p.act(state).sample()
        next_state, reward = mdp.step(state,action).sample()
        next_action = p.act(next_state).sample()
        q = q.update([( (state, action),reward+gamma*q((next_state,next_action)))])
        trace_count+=1
        yield q


def q_learning(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        gamma: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    state_distribution = states

    while True:
        state = state_distribution.sample()
        action = Choose(mdp.actions(state)).sample()
        next_state, reward = mdp.step(state,action).sample()
        next_reward = max(
            q((next_state, a))
            for a in mdp.actions(next_state)
        )
        q = q.update([((state, action), reward+gamma*next_reward)])
        yield q

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


    # approxs_mc = mc_control(
    #     mdp = si_mdp,
    #     states = start_states,
    #     approx_0 = Tabular,
    #     gamma = user_gamma,
    #     eps = lambda x: 1./float(x)
    #     )

    # last_func = last(islice(approxs_mc,num_traces))
    # p = markov_decision_process.policy_from_q(last_func, si_mdp)
    # pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

    # approxs_td = td_sarsa(
    #     mdp = si_mdp,
    #     states = start_states,
    #     approx_0 = Tabular(),
    #     gamma = user_gamma,
    #     eps = lambda x: 1./float(x)
    #     )

    # last_func = last(islice(approxs_td,num_traces))
    # p = markov_decision_process.policy_from_q(last_func, si_mdp)
    # pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

    # approxs_ql = q_learning(
    #     mdp = si_mdp,
    #     states = start_states,
    #     approx_0 = Tabular(),
    #     gamma = user_gamma
    #     )

    # last_func = last(islice(approxs_ql,num_traces))
    # p = markov_decision_process.policy_from_q(last_func, si_mdp)
    # pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

