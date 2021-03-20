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
class PriceState:
    time_step: int
    price: float

# @dataclass(frozen=True)
class OptimalExerciseTree(MarkovDecisionProcess[PriceState, bool]):

    spot_price: float
    payoff: Callable[[float, float], float]
    price_step_fn: Callable[[float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def __init__(self, spot_price: float, payoff: Callable[[float, float], float],\
                expiry: float, rate: float, price_step_fn: Callable[[float], float], num_steps: int):
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, float], float] = payoff
        self.expiry: float = expiry
        self.rate: float = rate
        self.price_step_fn: Callable[[float], float] = price_step_fn
        self.num_steps: int = num_steps

    def dt(self) -> float:
        return self.expiry / self.num_steps

    def step(
        self,
        state: PriceState,
        exercise : bool
    ) -> SampledDistribution[Tuple[PriceState, float]]:
        if state.time_step > self.num_steps:
            return None

        # state that will return None later
        terminal_state = PriceState(self.num_steps+1, 0)

        # expiry of contract, must be unexercised up until now and must now be exercised
        if state.time_step == self.num_steps:
                return Constant((terminal_state, self.payoff(self.expiry,state.price)))

        # can choose to exercise, leads straight to terminal state
        if exercise:
            return Constant((terminal_state, self.payoff(state.time_step*self.dt(),state.price)))

        # can choose to keep, leads to new prices
        def sample_next_state_reward(
            state=state,
            exercise=exercise
        ) -> Tuple[PriceState, float]:
            price_sample: float = self.price_step_fn(state.price)
            next_state = PriceState(state.time_step + 1, price_sample)
            return next_state, 0.

        return SampledDistribution(sample_next_state_reward)

    def actions(self, state: PriceState) -> Iterable[bool]:
        if state.time_step > self.num_steps: return {}
        return [True, False]

class NeverExercisePolicy(Policy[PriceState, bool]):
    def act(self, state: PriceState) -> Distribution[bool]:
        return Constant((False))

class LogNormalPrice(SampledDistribution[Tuple[PriceState]]):
    '''A Gaussian distribution with the given μ and σ.'''
    start_price: float
    rate : float
    vol: float
    time: float
    time_step: int
#self, start_price: float, rate: float, vol : float, time_step:int, time: expectation_samples: int = 1000
    def __init__(self, start_price : float, rate : float, vol : float, time : float, time_step : int, expectation_samples : int = 1000):
        self.start_price = start_price
        self.rate = rate
        self.vol = vol
        self.time = time
        self.time_step = time_step
        super().__init__(
            sampler=lambda: PriceState(self.time_step,self.start_price*np.exp(np.random.normal(loc=self.rate*self.time, scale=self.vol*self.time))) ,
            expectation_samples=expectation_samples
            )


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 5

    # price transition example

    dt_init = expiry_val/num_steps_val
    opt_trans = lambda x : x*(1+np.random.normal(rate_val*dt_init,vol_val*np.sqrt(dt_init)))


    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)



    opt_ex_am_option: OptimalExerciseTree = OptimalExerciseTree(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        rate=rate_val,
        price_step_fn=opt_trans,
        num_steps=num_steps_val
    )


    start_state = Constant(PriceState(0, spot_price_val))
    # pol_hold = NeverExercisePolicy()
    # sim_acts  = opt_ex_am_option.simulate_actions(start_state,pol_hold)
    # # for state,action,next_state, rew in sim_acts:
    # for step in sim_acts:
    #     print("Price: ", step.state.price, "Reward: ", step.reward)

    ag = AdamGradient(
        learning_rate=0.5,
        decay1=0.9,
        decay2=0.999
    )
    ffs = [
        lambda _: 1.,
        lambda x: x.price,
        lambda x: x.price**2,
        lambda x: x.price**3
    ]



    mdp_f0_mu_triples : MDP_FuncApproxV_Distribution = []
    for layer_num in range(num_steps_val+1):
        mdp_inst : OptimalExerciseTree = OptimalExerciseTree(
            spot_price=spot_price_val,
            payoff=opt_payoff,
            expiry=expiry_val-dt_init*layer_num,
            rate=rate_val,
            price_step_fn=opt_trans,
            num_steps=num_steps_val
            )

        lfa = LinearFunctionApprox.create(
             feature_functions=ffs,
             adam_gradient=ag,
             regularization_coeff=0.001,
             direct_solve=True
        )

        state_distr: LogNormalPrice = LogNormalPrice(
                start_price = spot_price_val,
                rate = rate_val,
                vol =  vol_val,
                time = layer_num*dt_init,
                time_step = layer_num,
                expectation_samples = 1000)
        mdp_f0_mu_triples.append((mdp_inst,lfa, state_distr))


    opt_search = back_opt_vf_and_policy(
            mdp_f0_mu_triples = mdp_f0_mu_triples,
            γ = 1.-rate_val,
            num_state_samples = 1000,
            error_tolerance = 1)

    for i, layer in enumerate(opt_search):
        print("Time Step number ", i)
        print("At the money value of ", layer[0].evaluate([PriceState(i,100)])[0])
        for price in range(70*2,99*2):
            print("Exercise at price ", price/2., "? ", "yes" if layer[1].act(PriceState(i,price)).sample() else "no" )





