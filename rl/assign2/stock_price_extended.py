from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovRewardProcess
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes
from typing import (Callable, Dict, Generic, Iterator,
                    Mapping, Set, Sequence, Tuple, TypeVar)
from rl.gen_utils.plot_funcs import plot_list_of_curves

@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMRP1(MarkovRewardProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    reward_f : Callable[[],float]
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def __init__(
        self,
        level_param: int,
        reward_f: Callable[[],float],
        alpha1: float = 0.25,
        ):
        self.reward_f = reward_f
        self.alpha1 = alpha1
        self.level_param = level_param

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition_reward(self, state: StateMP1) -> Categorical[Tuple[StateMP1,float]]:
        up_p = self.up_prob(state)
        return Categorical({
            (StateMP1(state.price + 1), self.reward_f(StateMP1(state.price + 1))): up_p,
            (StateMP1(state.price - 1), self.reward_f(StateMP1(state.price - 1))): 1 - up_p
        })

def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
):
    def rew_f(st : StateMP1):
        return 1. if st.price > level_param else 0.

    mp = StockPriceMRP1(level_param=level_param, reward_f = rew_f, alpha1=alpha1)
    start_state_distribution = Constant(StateMP1(price=start_price))
    price_hist = []
    rew_hist = []
    for s in itertools.islice(
        mp.simulate_reward(start_state_distribution),time_steps + 1):
        price_hist.append(s.next_state.price)
        rew_hist.append(s.reward)
    return np.array(price_hist), np.array(rew_hist)



if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )

    plot_list_of_curves(
        [range((process1_traces[0]).shape[0])],
        [process1_traces[0]],
        ["r"],
        [
            r"Price Evolution MP1"
        ],
        "Time",
        "Price",
        "Price Evolution according to process 1"
    )

    plot_list_of_curves(
        [range((process1_traces[1]).shape[0])],
        [process1_traces[1].cumsum()],
        ["r"],
        [
            r"Reward Evolution MP1"
        ],
        "Time",
        "Cumulative Return",
        "Example Cumulative Return for Time Spent Above Level Param"
    )
