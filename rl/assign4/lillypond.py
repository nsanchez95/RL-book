from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from rl.dynamic_programming import policy_iteration, value_iteration
from rl.dynamic_programming import almost_equal_vf_pis, almost_equal_vfs
from rl.iterate import converge
from scipy.stats import poisson
import numpy as np
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class LilyPondState:
    pad: int



LilyCroakMapping = StateActionMapping[LilyPondState, int]


class LilyPondMDP(FiniteMarkovDecisionProcess[LilyPondState, int]):

    def __init__(
        self,
        num_pads: int,
    ):
        self.num_pads: int = num_pads
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> LilyCroakMapping:
        d: Dict[LilyPondState, Dict[int, Categorical[Tuple[LilyPondState,
                                                            float]]]] = {}

        rew_map = np.zeros(self.num_pads+1)                             
        rew_map[self.num_pads] = 1                            
        for pad in range(1,self.num_pads):
            cur_state: LilyPondState = LilyPondState(pad)
            d1: Dict[int, Categorical[Tuple[LilyPondState, float]]] = {}

            # frog croaks A
            croakA_probs = np.zeros(self.num_pads + 1)
            croakA_probs[pad-1] = float(pad)/self.num_pads
            croakA_probs[pad+1] = 1. - croakA_probs[pad-1]

            lilyPadCatProbA: Mapping[LilyPondState, float] = \
                    {(LilyPondState(i),rew_map[i]):croakA_probs[i] for i in range(0,self.num_pads+1)}
            d1[0] = Categorical(lilyPadCatProbA)

            # frog croaks B
            croakB_probs = np.ones(self.num_pads + 1)/float(self.num_pads)
            croakB_probs[pad] = 0.

            lilyPadCatProbB: Mapping[LilyPondState, float] = \
                    {(LilyPondState(i),rew_map[i]):croakB_probs[i] for i in range(0,self.num_pads+1)}
            d1[1] = Categorical(lilyPadCatProbB)
            
            d[cur_state] = d1
        return d


def all_policies_llp(num_pads):
    pol_list_arrs = [np.array(i) for i in itertools.product([0, 1],\
        repeat = num_pads)]
    pol_list = [FinitePolicy({LilyPondState(i) : Constant(sing_pol[i]) \
        for i in range(1,num_pads)}) for sing_pol in pol_list_arrs]
    return pol_list


def find_optimal_policy(mrd, pols):
    return max([(pol, mrd.apply_finite_policy(pol).get_value_function_vec(gamma = 1.0),\
        mrd.apply_finite_policy(pol).get_value_function_vec(gamma = 1.0)[0]) for pol in pols],
        key = itemgetter(2))

if __name__ == '__main__':
    from pprint import pprint

    gamma = 1.0
    diff_nums = np.arange(25,50, 5)
    num_tests = len(diff_nums)
    num_iters_pi = np.zeros(num_tests)
    num_iters_vi = np.zeros(num_tests)
    count = 0
    for num_llpads in diff_nums:
        llp_mdp: FiniteMarkovDecisionProcess[LilyPondState, int] =\
            LilyPondMDP(
                num_pads=num_llpads,
            )
        num_iters_pi[count] = sum(1 for _ in converge(policy_iteration(llp_mdp, gamma = gamma), almost_equal_vf_pis))
        num_iters_vi[count] = sum(1 for _ in converge(value_iteration(llp_mdp, gamma = gamma), almost_equal_vfs))
        count+= 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    states = np.arange(1, num_llpads)
    ax.plot(diff_nums, num_iters_pi, color = 'b', label = 'Policy Iteration')
    ax2 = ax.twinx()
    ax2.plot(diff_nums, num_iters_vi, color = 'r', label = 'Value Iteration')
    ax.legend()
    ax2.legend()
    ax.set_ylabel('Policy Iteration Num Iterations')
    ax2.set_ylabel('Value Iteration Num Iterations')
    ax.set_xlabel('Number of Lillypads')
    plt.show()
    # i = 0
    # last_step = None
    # for update_step in policy_iteration(llp_mdp, gamma = gamma):
    #     if(last_step == None):
    #         last_step = update_step
    #         continue
    #     if almost_equal_vf_pis(last_step, update_step):
    #         break
    #     last_step = update_step
    #     i+=1
    # print("Num iterations", i)
    # print(last_step)


    # i = 0
    # last_step = None
    # for update_step in value_iteration(llp_mdp, gamma = gamma):
    #     if(last_step == None):
    #         last_step = update_step
    #         continue
    #     if almost_equal_vfs(last_step, update_step):
    #         break
    #     last_step = update_step
    #     i+=1
    # print("Num iterations", i)
    # print(last_step)
        
    # all_pols_llp = all_policies_llp(num_llpads)
    # pol = all_pols_llp[2]
    # print(pol)
    # print(llp_mdp.apply_finite_policy(pol))
    # print(llp_mdp.apply_finite_policy(pol).get_value_function_vec(gamma = 1.0))
    # opt_pol, opt_val, opt_start_val = find_optimal_policy(llp_mdp, all_pols_llp)
    # opt_pol = [opt_pol.act(k).sample() for k in opt_pol.states()]
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    # states = np.arange(1, num_llpads)
    # ax.bar(states-0.1,opt_val, color = 'b', width = 0.2)
    # ax.bar(states+0.1,opt_pol, color = 'r', width = 0.2)
    # ax.legend(labels=['Escape Prob', 'Croak B Optimal'])
    # ax.set_ylabel('Escape Probability')
    # ax.set_xlabel('Lilypad Number')
    # plt.show()
    # fdp: FinitePolicy[InventoryState, int] = FinitePolicy(
    #     {InventoryState(alpha, beta):
    #      Constant(user_capacity - (alpha + beta)) for alpha in
    #      range(user_capacity + 1) for beta in range(user_capacity + 1 - alpha)}
    # )

    # print("Policy Map")
    # print("----------")
    # print(fdp)

    # implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
    #     si_mdp.apply_finite_policy(fdp)
    # print("Implied MP Transition Map")
    # print("--------------")
    # print(FiniteMarkovProcess(implied_mrp.transition_map))

    # print("Implied MRP Transition Reward Map")
    # print("---------------------")
    # print(implied_mrp)

    # print("Implied MP Stationary Distribution")
    # print("-----------------------")
    # implied_mrp.display_stationary_distribution()
    # print()

    # print("Implied MRP Reward Function")
    # print("---------------")
    # implied_mrp.display_reward_function()
    # print()

    # print("Implied MRP Value Function")
    # print("--------------")
    # implied_mrp.display_value_function(gamma=user_gamma)
    # print()

    # from rl.dynamic_programming import evaluate_mrp_result
    # from rl.dynamic_programming import policy_iteration_result
    # from rl.dynamic_programming import value_iteration_result

    # print("Implied MRP Policy Evaluation Value Function")
    # print("--------------")
    # pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    # print()

    # print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_pi, opt_policy_pi = policy_iteration_result(
    #     si_mdp,
    #     gamma=user_gamma
    # )
    # pprint(opt_vf_pi)
    # print(opt_policy_pi)
    # print()

    # print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    # pprint(opt_vf_vi)
    # print(opt_policy_vi)
    # print()
