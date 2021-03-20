from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List, Mapping, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Constant, Categorical
from rl.dynamic_programming import value_iteration_result, value_iteration, almost_equal_vfs
from pprint import pprint
from scipy.stats import poisson

########## MDP CLASS DEFINITIONS #######################
@dataclass(frozen=True)
class CareerState:
    wage: int

CareerLadderMapping = StateActionMapping[CareerState, Tuple[int,int]]

class CareerMDP(FiniteMarkovDecisionProcess[CareerState, Tuple[int,int]]):

    def __init__(
        self,
        daily_hours_H: int,
        max_wage_W: int,
        alpha: float,
        beta: float
    ):
        self.daily_hours_H: int = daily_hours_H
        self.max_wage_W: int = max_wage_W
        self.alpha: float = alpha
        self.beta: float = beta
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> CareerLadderMapping:
        d: Dict[CareerState, Dict[Tuple[int,int], Categorical[Tuple[CareerState,float]]]] = {}
        for curr_wage in range(1,self.max_wage_W+1):
            state_curr: CareerState = CareerState(curr_wage)
            d_s : Dict[Tuple[int,int], Categorical[Tuple[CareerState,float]]] = {}
            
            # create all possible actions
            for learn_hours in range(self.daily_hours_H+1):
                lambda_curr = float(learn_hours)*self.alpha
                for search_hours in range(self.daily_hours_H+1 - learn_hours):
                    p_job_offer = self.beta*float(search_hours)/float(self.daily_hours_H)
                    # reward is only dependent on action and current state so can compute that first
                    reward_daily = float((self.daily_hours_H - learn_hours - search_hours)*curr_wage)
                    
                    # next we need to check probability of all wage increases
                    next_wage_probs_dict : Dict[Tuple[CareerState, float], float] = {}
                    
                    # first case of no incremental wage
                    next_wage_probs_dict[(CareerState(curr_wage), reward_daily)] =\
                     (1.-p_job_offer)*poisson.pmf(0,lambda_curr)

                    # then all incremental wages cases as outlined in pdf
                    for add_wage in range(1,self.max_wage_W+1-curr_wage):
                        prob_trsn = poisson.pmf(add_wage,lambda_curr)
                        if add_wage ==1:
                            prob_trsn +=poisson.pmf(0,lambda_curr)*p_job_offer
                        if curr_wage+add_wage == self.max_wage_W:
                            prob_trsn += 1.-poisson.cdf(add_wage,lambda_curr)
                        next_wage_probs_dict[(CareerState(curr_wage+add_wage), reward_daily)] =prob_trsn

                    # construct transition for given state+action
                    d_s[(learn_hours,search_hours)] = Categorical(next_wage_probs_dict)
            d[state_curr] = d_s
        return d


################# IMPLEMENTATION ########################
if __name__ == '__main__':
    alpha = 0.08
    beta = 0.82
    gamma = 0.95
    H = 10
    W = 30

    # create MDP
    career_mdp: FiniteMarkovDecisionProcess[CareerState, Tuple[int,int]] =\
        CareerMDP(
        daily_hours_H= H,
        max_wage_W=W,
        alpha= alpha,
        beta=beta
        )

    # perform optimization
    opt_vf_vi, opt_policy_vi = value_iteration_result(career_mdp, gamma=gamma)
    
    print("15")
    list_next = career_mdp.step(CareerState(15), (0,10))
    print(list_next)
    s =0
    for k,v in list_next:
        s+= v
    print("sum", s)


    print("10")
    list_next = career_mdp.step(CareerState(10), (0,10))
    print(list_next)
    s =0
    for k,v in list_next:
        s+= v
    print("sum", s)

    #print out result
    print(opt_policy_vi)
    