from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List, Iterable
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.markov_process import MarkovRewardProcess
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
class OptimalMarketMaking(MarkovRewardProcess[MMState]):

    c: float
    k: float
    vol: float
    dt: float
    T: float
    bid_spd : Callable[[MMState],float]
    ask_spd : Callable[[MMState],float]
    gamma: float


    def __init__(self, c: float, k: float, vol: float, dt: float, T: float, \
                    bid_fn :Callable[[MMState],float], ask_fn :Callable[[MMState],float], gamma: float):
        self.c: float = c
        self.k: float = k
        self.vol: float = vol
        self.dt: float = dt
        self.T: float = T
        self.bid_spd : Callable[[MMState],float] = bid_fn
        self.ask_spd : Callable[[MMState],float] = ask_fn
        self.gamma = gamma

        self.inv_chg : Callable[[MMState],Distribution[MMState]] = \
            lambda s: Categorical(\
                    {MMState(s.price,s.pnl-s.price+self.bid_spd(s), s.inv +1, s.t): \
                        self.dt*self.c*np.exp(-self.k*self.bid_spd(s)),\
                     MMState(s.price,s.pnl+s.price+self.ask_spd(s), s.inv -1, s.t):
                        self.dt*self.c*np.exp(-self.k*self.ask_spd(s)),\
                     MMState(s.price,s.pnl, s.inv, s.t):
                         1.-self.dt*self.c*(np.exp(-self.k*self.bid_spd(s))+np.exp(-self.k*self.ask_spd(s)))})

        self.price_chg : Distribution[float] = \
                Categorical({self.vol*np.sqrt(self.dt): 0.5, -self.vol*np.sqrt(self.dt): 0.5})

    def transition_reward(self, state: MMState) -> SampledDistribution[Tuple[MMState,float]]:
        if state.t > self.T:
            return None

        def sample_next_state_reward(
            state=state
        ) -> Tuple[MMState, float]:
            state_next_inv = self.inv_chg(state).sample()
            
            # next update price
            next_price = state.price + self.price_chg.sample()

            next_state = MMState(price=next_price, pnl = state_next_inv.pnl, \
                inv=state_next_inv.inv,t = state.t+self.dt)

            rew = 0 if next_state.t < self.T else \
                    -np.exp(-self.gamma*(next_state.pnl+next_state.inv*next_state.price))
            return next_state, rew


        return SampledDistribution(sample_next_state_reward)

def track_simulation(mrp: OptimalMarketMaking, start_dist: Distribution[MMState],\
                         num_sims:int, sim_length:int, append : str):
    trading_pnl_hist = np.zeros((num_sims,sim_length))
    inventory_hist = np.zeros((num_sims,sim_length))
    ob_bid_hist = np.zeros((num_sims,sim_length))
    ob_ask_hist = np.zeros((num_sims,sim_length))
    num_trades_hist = np.zeros((num_sims,sim_length))
    bid_spd = np.zeros((num_sims,sim_length))
    offer_spd = np.zeros((num_sims,sim_length))
    mark_to_market = np.zeros((num_sims,sim_length))
    print("STARTING ", append)
    for sim_num in range(num_sims):
        for sim_step,sim in enumerate(mrp.simulate_reward(start_dist)):
            ob_bid_hist[sim_num,sim_step] = sim.state.price - as_optimal_bid(sim.state)
            ob_ask_hist[sim_num,sim_step] = sim.state.price + as_optimal_ask(sim.state)
            trading_pnl_hist[sim_num,sim_step] = sim.state.pnl+sim.state.inv*sim.state.price
            inventory_hist[sim_num,sim_step] = sim.state.inv
            bid_spd[sim_num,sim_step] = as_optimal_bid(sim.state)
            offer_spd[sim_num,sim_step] = as_optimal_ask(sim.state)
            mark_to_market[sim_num,sim_step] = sim.state.pnl+sim.state.inv*sim.state.price
        if(sim_num %1000 == 0):
            print("SIM NUMBER", sim_num)
    np.savetxt('trading_pnl_'+append+'.csv', trading_pnl_hist)
    np.savetxt('inventory_'+append+'.csv', inventory_hist)
    np.savetxt('ob_ask_'+append+'.csv', ob_ask_hist)
    np.savetxt('ob_bid_'+append+'.csv', ob_bid_hist)
    return ob_bid_hist, ob_ask_hist, trading_pnl_hist

if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves

    S_0 = 100
    I_0 = 0
    T = 1.
    dt = 0.005
    gamma = 0.1
    vol = 2.
    k = 1.5
    c = 140.

    as_optimal_bid = lambda s : 0.5*(2.*s.inv+1.0)*gamma*(vol**2)*(T-s.t)+np.log(1.+gamma/k)/gamma
    as_optimal_ask = lambda s : 0.5*(1-2.*s.inv)*gamma*(vol**2)*(T-s.t)+np.log(1.+gamma/k)/gamma

    as_mp : OptimalMarketMaking = OptimalMarketMaking(
            c=c,
            k= k,
            vol= vol,
            dt =  dt,
            T = T,
            bid_fn = as_optimal_bid,
            ask_fn = as_optimal_ask,
            gamma = gamma)

    num_sims = 1000
    sim_length = int(T/dt)

    start_dist: Distribution[MMState] = Constant(MMState(S_0,0.,I_0,0.))
    

    bid_hist, offer_hist, trading_pnl_hist = track_simulation(as_mp,start_dist,
                                                 num_sims, sim_length, 'as')

    avg_bid_offer = (offer_hist-bid_hist).mean()
    avg_pnl_as = trading_pnl_hist.mean(axis = 0)
    final_pnl_as = trading_pnl_hist.mean(axis = 1)
    print('AS Optimal', avg_pnl_as, final_pnl_as)





    naive_bid_spd = lambda s : avg_bid_offer/2.
    naive_ask_spd = lambda s : avg_bid_offer/2.

    naive_mp : OptimalMarketMaking = OptimalMarketMaking(
        c=c,
        k= k,
        vol= vol,
        dt =  dt,
        T = T,
        bid_fn = naive_bid_spd,
        ask_fn = naive_ask_spd,
        gamma = gamma)

    mark_to_market2 = np.zeros((num_sims,sim_length))
    bid_hist, offer_hist, trading_pnl_hist = track_simulation(naive_mp,start_dist,
                                                 num_sims, sim_length, 'naive')

    print("Naive Optimal")
    final_pnl_naive = trading_pnl_hist.mean(axis = 0)
    trading_pnl_hist = mark_to_market2.mean(axis = 1)
    print(final_pnl_naive.mean(), np.std(final_pnl_naive))

