
from typing import Iterable, Iterator, Tuple, TypeVar, Callable

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns

from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical
import numpy as np
from pprint import pprint
from rl.distribution import Choose
from rl.function_approx import Tabular, FunctionApprox
from rl.iterate import last, accumulate
from itertools import islice
from rl.chapter10.prediction_utils import unit_experiences_from_episodes,fmrp_episodes_stream
from rl.function_approx import learning_rate_schedule
from rl.markov_process import FiniteMarkovProcess

S = TypeVar('S')
A = TypeVar('A')


def mc_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        gamma: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    episodes = (returns(trace, gamma, tolerance) for trace in traces)

    return approx_0.iterate_updates(
        ((step.state, step.return_) for step in episode)
        for episode in episodes
    )




@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMRP(MarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: InventoryState
    ) -> SampledDistribution[Tuple[InventoryState, float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[InventoryState, float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward: float = - self.holding_cost * state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> RewardTransition[InventoryState]:
        d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                base_reward = - self.holding_cost * state.on_hand
                sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
                    {(InventoryState(ip - i, beta1), base_reward):
                     self.poisson_distr.pmf(i) for i in range(ip)}
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                sr_probs_map[(InventoryState(0, beta1), reward)] = probability
                d[state] = Categorical(sr_probs_map)
        return d


def mc_prediction_tabular(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        gamma: float,
        iters_max: int,
        alpha_fn: Callable[[int], float],
        tolerance: float = 1e-6
) -> Dict[S,float]:
    episodes = (returns(trace, gamma, tolerance) for trace in traces)
    iter_count = 0

    count_dict = {}
    MC_approx = {}

    for episode in episodes:
        iter_count +=1
        for step in episode:
            count_dict[step.state] = count_dict.get(step.state, 0) + 1
            MC_approx[step.state] = MC_approx.get(step.state, 0.0) + \
                    (step.return_ - MC_approx.get(step.state, 0.0))*alpha_fn(count_dict[step.state])
        if iter_count > iters_max:
            break

    return MC_approx

def td_prediction_tabular(
        transitions: Iterable[mp.TransitionStep[S]],
        alpha_fn: Callable[[int], float],
        gamma: float,
        iters_max :int
) -> Dict[S,float]:

    iter_count = 0
    counts = {}
    vals = {}

    for tr_step in transitions:
        iter_count +=1
        est_val = tr_step.reward + gamma*vals.get(tr_step.next_state, 0)
        counts[tr_step.state] = counts.get(tr_step.state, 0) + 1
        weight: float = alpha_fn(counts.get(tr_step.state, 0))
        vals[tr_step.state] = weight * est_val + (1 - weight) * vals.get(tr_step.state, 0.)
        if iter_count > iters_max:
            break

    return vals

def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        gamma: float,
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)

    '''
    def step(v, transition):
        return v.update([(transition.state,
                          transition.reward + gamma * v(transition.next_state))])

    return accumulate(transitions, step, initial=approx_0)


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    episode_length: int = 100
    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5
    gamma: float = 0.9

    episodes: Iterable[Iterable[mp.TransitionStep[S]]] = fmrp_episodes_stream(si_mrp)

    td_experiences: Iterable[mp.TransitionStep[S]] = unit_experiences_from_episodes(episodes,episode_length)

    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )

    td_vfs: Iterator[FunctionApprox[S]] = td_prediction(
        transitions=td_experiences,
        approx_0=Tabular(count_to_weight_func=learning_rate_func),
        gamma=gamma
    )

    num_episodes = 10000
    num_iters = num_episodes*100
    td_vf_tabular= td_prediction_tabular(
        transitions=td_experiences,
        alpha_fn = learning_rate_func,
        gamma=gamma,
        iters_max = num_iters
    )

    print("Value Function with Tabular TD")
    print("--------------")
    pprint({s: round(td_vf_tabular[s], 3)
        for s in si_mrp.non_terminal_states})

    print("Value Function with Tabular TD Approx")
    final_td_vf: FunctionApprox[S] = last(islice(td_vfs, episode_length * num_episodes))
    pprint({s: round(final_td_vf(s), 3) for s in si_mrp.non_terminal_states})
    traces : Iterable[Iterable[mp.TransitionStep[S]]] =\
        si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))

    num_traces = 1000

    it1: Dict[InventoryState, float] = mc_prediction_tabular(
        traces = traces,
        gamma = user_gamma,
        alpha_fn = learning_rate_func,
        iters_max = num_traces,
        tolerance = 1e-6
    )
    print("Value Function with Tabular MC")
    print("--------------")
    pprint({s: round(it1[s], 3)
        for s in si_mrp.non_terminal_states})

    it: Iterator[FunctionApprox[InventoryState]] = mc_prediction(
        traces = traces,
        approx_0 = Tabular(),
        gamma = user_gamma,
        tolerance = 1e-6
    )

    last_func = last(islice(it,num_traces))
    print("Value Function with Tabular MC Approx")
    print("--------------")
    pprint({s: round(last_func.evaluate([s])[0], 3)
        for s in si_mrp.non_terminal_states})

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

"""
Value Function with Tabular RD
--------------
{InventoryState(on_hand=0, on_order=1): -27.949,
 InventoryState(on_hand=1, on_order=0): -28.928,
 InventoryState(on_hand=0, on_order=0): -35.494,
 InventoryState(on_hand=0, on_order=2): -28.501,
 InventoryState(on_hand=2, on_order=0): -30.387,
 InventoryState(on_hand=1, on_order=1): -29.334}
Value Function with Tabular TD Approx
{InventoryState(on_hand=0, on_order=1): -27.837,
 InventoryState(on_hand=1, on_order=0): -28.804,
 InventoryState(on_hand=0, on_order=0): -35.433,
 InventoryState(on_hand=0, on_order=2): -28.429,
 InventoryState(on_hand=2, on_order=0): -30.427,
 InventoryState(on_hand=1, on_order=1): -29.309}
Value Function with Tabular MC
--------------
{InventoryState(on_hand=0, on_order=1): -27.923,
 InventoryState(on_hand=1, on_order=0): -28.925,
 InventoryState(on_hand=0, on_order=0): -35.505,
 InventoryState(on_hand=0, on_order=2): -28.339,
 InventoryState(on_hand=2, on_order=0): -30.339,
 InventoryState(on_hand=1, on_order=1): -29.336}
Value Function with Tabular MC Approx
--------------
{InventoryState(on_hand=0, on_order=1): -27.924,
 InventoryState(on_hand=1, on_order=0): -28.922,
 InventoryState(on_hand=0, on_order=0): -35.501,
 InventoryState(on_hand=0, on_order=2): -28.334,
 InventoryState(on_hand=2, on_order=0): -30.337,
 InventoryState(on_hand=1, on_order=1): -29.335}
Value Function
--------------
{InventoryState(on_hand=0, on_order=1): -27.932,
 InventoryState(on_hand=1, on_order=0): -28.932,
 InventoryState(on_hand=0, on_order=0): -35.511,
 InventoryState(on_hand=0, on_order=2): -28.345,
 InventoryState(on_hand=2, on_order=0): -30.345,
 InventoryState(on_hand=1, on_order=1): -29.345}
"""

