from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List, Iterable
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.distribution import Constant, Categorical, SampledDistribution, Distribution
from rl.finite_horizon import optimal_vf_and_policy
from rl.approximate_dynamic_programming import MDP_FuncApproxV_Distribution, back_opt_vf_and_policy
from rl.function_approx import Tabular, AdamGradient, LinearFunctionApprox

@dataclass(frozen=True)
class MMState:
    price: float
    pnl: float
    inv: int
    t: float

# @dataclass(frozen=True)
class OptimalMarketMaking(MarkovProcess[MMState]):

    c: float
    k: float
    vol: float
    dt: float
    T: float
    bid_spd : Callable[[MMState],float]
    ask_spd : Callable[[MMState],float]


    def __init__(self, c: float, k: float, vol: float, dt: float, T: float, \
                    bid_fn :Callable[[MMState],float], ask_fn :Callable[[MMState],float]):
        self.c: float = c
        self.k: float = k
        self.vol: float = vol
        self.dt: float = dt
        self.T: float = T
        self.bid_spd : Callable[[MMState],float] = bid_fn
        self.ask_spd : Callable[[MMState],float] = ask_fn

        self.inv_chg : Callable[[MMState],Distribution[MMState]] = \
            lambda s: Categorical(\
                    {MMState(s.price,s.pnl-s.price+self.bid_spd(s), s.inv +1): \
                        self.c*np.exp(-self.k*self.bid_spd(s)),\
                     MMState(s.price,s.pnl+s.price+self.ask_spd(s), s.inv -1):
                        self.c*np.exp(-self.k*self.ask_spd(s)),\
                     MMState(s.price,s.pnl, s.inv):
                         1-self.c(np.exp(-self.k*self.bid_spd(s))+np.exp(-self.k*self.ask(s)))})

        self.price_chg : Distribution[float] = \
                Categorical({self.vol*np.exp(dt): 0.5, -self.vol*np.exp(dt): 0.5})

    def transition(self, state: MMState) : SampledDistribution[S]:
        if state.t == self.T:
            return None

        def sample_next_state_reward(
            state=state
        ) -> Tuple[PriceState, float]:
            # first execute inventory/pnl change
            state_next_inv = self.inv_chg(state).sample()
            
            # next update price
            next_price = state.price + self.price_chg.sample()

            next_state = MMState(price=next_price, pnl = state_next_inv.pnl, \
                inv=state_next_inv.inv,t = state.t+self.dt)


        return SampledDistribution(sample_next_state_reward)

if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves

    T = 1.
    dt = 0.005
    gamma = 0.1
    vol = 2.
    k = 1.5
    c = 140

    as_optimal_bid = lambda s : 0.5*(2.*s.inv+1.0)*gamma*(vol**2)*(T-s.t)+np.log(1.+gamma/k)/gamma
    as_optimal_ask = lambda s : 0.5*(1-2.*s.inv)*gamma*(vol**2)*(T-s.t)+np.log(1.+gamma/k)/gamma

    as_mp : OptimalMarketMaking = OptimalMarketMaking(
            c=c,
            k= k,
            vol= vol,
            dt =  dt,
            T = T)




