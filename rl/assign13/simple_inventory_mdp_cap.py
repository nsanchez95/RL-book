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
S = TypeVar('S')
A = TypeVar('A')

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


InvOrderMapping = StateActionMapping[InventoryState, int]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

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
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(alpha, beta)
                ip: int = state.inventory_position()
                base_reward: float = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                        {(InventoryState(ip - i, order), base_reward):
                         self.poisson_distr.pmf(i) for i in range(ip)}

                    probability: float = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: float = base_reward - self.stockout_cost *\
                        (probability * (self.poisson_lambda - ip) +
                         ip * self.poisson_distr.pmf(ip))
                    sr_probs_dict[(InventoryState(0, order), reward)] = \
                        probability
                    d1[order] = Categorical(sr_probs_dict)

                d[state] = d1
        return d


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
        gamma: float,
        episodes:int  = 1000,
        max_iter:int = 100000
) -> FunctionApprox[Tuple[S, A]]:
    q = approx_0
    iter_count = 0
    for episode_num in range(episodes):
        epsilon: float = 1.0 / (episode_num + 1)
        state: Cell = states.sample()
        
        # go through a full episode

        while(True):
            p = markov_decision_process.policy_from_q(q, mdp, epsilon)
            action = p.act(state).sample()
            next_state, reward = mdp.step(state,action).sample()            
            # end while loop if end of road
            if mdp.is_terminal(next_state):
                q = q.update([((state, action),reward)])
                break

            next_action = p.act(next_state).sample()
            q = q.update([( (state, action),reward+gamma*q((next_state,next_action)))])
            state = next_state
            iter_count += 1
            if iter_count > max_iter: break
        if iter_count > max_iter: break
    return q


def q_learning(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        gamma: float,
        episodes:int  = 1000,
        max_iter:int = 100000
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    q = approx_0
    iter_count = 0
    for episode_num in range(episodes):
        epsilon: float = 1.0 / (episode_num + 1)
        state: Cell = states.sample()
        # go through a full episode
        while(True):
            p = markov_decision_process.policy_from_q(q, mdp, epsilon)
            action = p.act(state).sample()
            next_state, reward = mdp.step(state,action).sample()
            
            # end while loop if end of road
            if mdp.is_terminal(next_state):
                q = q.update([((state, action),reward)])
                break

            next_reward = max(q((next_state, a)) for a in mdp.actions(next_state))
            q = q.update([( (state, action),reward+gamma*next_reward)])
            state = next_state
            iter_count += 1
            if iter_count > max_iter: break
        if iter_count > max_iter: break
    return q

if __name__ == '__main__':
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.8

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


    approxs_mc = mc_control(
        mdp = si_mdp,
        states = start_states,
        approx_0 = Tabular(),
        gamma = user_gamma,
        eps = lambda x: 1./float(x)
        )

    last_func = last(islice(approxs_mc,num_traces))
    print("MONTE CARLO CONTROL")
    p = markov_decision_process.policy_from_q(last_func, si_mdp)
    pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

    approxs_td = td_sarsa(
        mdp = si_mdp,
        states = start_states,
        approx_0 = Tabular(),
        gamma = user_gamma,
        )

    last_func = approxs_td
    print("SARSA CONTROL")
    p = markov_decision_process.policy_from_q(last_func, si_mdp)
    pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

    approxs_td = q_learning(
        mdp = si_mdp,
        states = start_states,
        approx_0 = Tabular(),
        gamma = user_gamma,
        )
    print("Q LEARNING CONTROL")
    last_func = approxs_td
    p = markov_decision_process.policy_from_q(last_func, si_mdp)
    pprint({s:p.act(s).sample() for s in si_mdp.non_terminal_states})

