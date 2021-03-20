
from typing import Iterable, Iterator, Tuple, TypeVar, Callable

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns
import math
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


def n_step_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        n_steps: int,
        approx_0: FunctionApprox[S],
        gamma: float,
        tolerance: float,
) -> Iterator[FunctionApprox[S]]:
    '''Value Function Prediction using the n-step bootstrapping method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      n_steps -- number of steps for bootstrapping
      approx_0 -- initial approximation of value function
      gamma -- discount rate (0 < γ ≤ 1)
    '''
    func_approx: FunctionApprox[S] = approx_0
    max_steps  = round(math.log(tolerance) / math.log(gamma)) if gamma < 1 else 10000

    for trace in traces:
        # these will be sliding windows to look at n-long windows
        relevant_states = []
        relevant_rewards = []
        
        #these will be fed into function approximation
        predictors: List[S] = []
        responses: Sequence[float] = []

        t = 0
        for tr in trace:
            if t > max_steps:
                break
            if t < n_steps:
                # gather enough step for n-step bootstrapping
                relevant_states.append(tr.state)
                relevant_rewards.append(tr.reward)
            else:

                # record the pair
                predictors.append(relevant_states[0])
                rew = (gamma**n_steps)*func_approx(tr.state)
                for i in range(n_steps):
                    rew += (gamma**i)*relevant_rewards[i]
                responses.append(rew)

                # update the sliding window
                relevant_states.append(tr.state)
                relevant_rewards.append(tr.reward)
                relevant_states = relevant_states[1:]
                relevant_rewards = relevant_rewards[1:]
            t+=1

        func_approx = func_approx.update(zip(predictors, responses))
        yield func_approx

def td_lambda_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        gamma: float,
        lambd: float,
        tolerance:float,
        max_traces: int
) -> Iterator[FunctionApprox[S]]:
    '''Value Function Prediction using the td lambda method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      n_steps -- number of steps for bootstrapping
      approx_0 -- initial approximation of value function
      gamma -- discount rate (0 < γ ≤ 1)
    '''
    func_approx: Tabular[S] = approx_0
    max_steps  = round(math.log(tolerance) / math.log(gamma)) if gamma < 1 else 10000
    num_traces = 0
    for trace in traces:
        trace_eligibility : Dict[S,float] = {}
        #these will be fed into function approximation
        if num_traces > max_traces: break
        # if num_traces % 500 == 0:
        #     print(num_traces)
        #     print(func_approx.values_map)
        num_traces+=1
        t = 0
        for tr in trace:
            if t > max_steps:
                break
            td_error = tr.reward + gamma*func_approx(tr.next_state) - func_approx(tr.state)
            if tr.state not in trace_eligibility:
                trace_eligibility[tr.state] = 0.
            for state in trace_eligibility:
                if state == tr.state:
                    trace_eligibility[tr.state] = gamma*lambd*trace_eligibility[state] + 1.
                else:
                    trace_eligibility[state] = gamma*lambd*trace_eligibility[state]
                
                func_approx = func_approx.update(iter([(state,td_error*trace_eligibility[state]+func_approx(state))]))

            t+=1
        yield func_approx


def td_lambda_prediction_tabular(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        gamma: float,
        lambd: float,
        tolerance:float,
        learning_rate: Callable[[int], float],
        max_traces: int
) -> Iterator[FunctionApprox[S]]:

    values_dict = {}
    max_steps  = round(math.log(tolerance) / math.log(gamma)) if gamma < 1 else 10000
    step_num = 0
    trace_num = 0
    for trace in traces:
        trace_eligibility : Dict[S,float] = {}
        if trace_num > max_traces:
        	break
        #these will be fed into function approximation
        # if step_num % 100 == 0:
        #     print(step_num)
        #     print(values_dict)
        #     print(max_steps)
        t = 0
        for tr in trace:
            if t > max_steps:
                break
            td_error = tr.reward + gamma*values_dict.get(tr.next_state,0) - values_dict.get(tr.state,0)
            
            if tr.state not in trace_eligibility:
                trace_eligibility[tr.state] = 0.
            for state in trace_eligibility:
                if state == tr.state:
                    trace_eligibility[tr.state] = gamma*lambd*trace_eligibility[state] + 1.
                else:
                    trace_eligibility[state] = gamma*lambd*trace_eligibility[state]
                # print(learning_rate(step_num)*td_error*trace_eligibility[state])
                values_dict[state] = values_dict.get(state,0.) + learning_rate(step_num)*td_error*trace_eligibility[state]
            step_num+=1
            t+=1
        trace_num += 1
    return values_dict


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

    
    traces : Iterable[Iterable[mp.TransitionStep[S]]] =\
        si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))

    num_traces = 1000
    n_steps = 5


    it: Iterator[FunctionApprox[InventoryState]] = mc_prediction(
        traces = traces,
        approx_0 = Tabular(),
        gamma = user_gamma,
        tolerance = 1e-6
    )


    # )
    # it= td_lambda_prediction(
    #     traces = traces,
    #     approx_0 = Tabular(),
    #     gamma = user_gamma,
    #     lambd = 0.5,
    #     tolerance = 1e-6,
    #     max_traces = num_traces
    # )

    last_func = last(islice(it,num_traces))
    print("Value Function with Tabular MC Approx")
    print("--------------")
    pprint({s: round(last_func.evaluate([s])[0], 3)
        for s in si_mrp.non_terminal_states})


    it: Iterator[FunctionApprox[InventoryState]] = n_step_prediction(
        traces = traces,
        approx_0 = Tabular(),
        gamma = user_gamma,
        n_steps = n_steps,
        tolerance = 1e-6
    )


    last_func = last(islice(it,num_traces))
    print("Value Function with 5-Step Approx")
    print("--------------")
    pprint({s: round(last_func.evaluate([s])[0], 3)
        for s in si_mrp.non_terminal_states})
    # last_func = last(islice(it,num_traces))
    # print("Value Function with Tabular MC Approx")
    # print("--------------")
    # pprint({s: round(it[s], 3)
    #     for s in si_mrp.non_terminal_states})

    it= td_lambda_prediction_tabular(
        traces = traces,
        approx_0 = Tabular(),
        gamma = user_gamma,
        lambd = 0.5,
        tolerance = 1e-6,
        learning_rate = lambda x: 0.05,
        max_traces = num_traces)


    print("Value Function with TD Lambda Tabular MC Approx")
    print("--------------")
    pprint({s: round(it[s], 3)
        for s in si_mrp.non_terminal_states})
    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

