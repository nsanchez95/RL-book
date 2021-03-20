from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand1: int
    on_order1: int
    on_hand2: int
    on_order2: int

    def inventory_position1(self) -> int:
        return self.on_hand1 + self.on_order1
    def inventory_position2(self) -> int:
        return self.on_hand2 + self.on_order2

@dataclass(frozen=True)
class OrderAction:
    internal_move: int
    order_c1: int
    order_c2: int


InvOrderMapping = StateActionMapping[InventoryState, OrderAction]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, OrderAction]):

    def __init__(
        self,
        capacity1: int,
        capacity2: int,
        poisson_lambda1: float,
        poisson_lambda2: float,
        holding_cost1: float,
        holding_cost2: float,
        stockout_cost1: float,
        stockout_cost2: float,
        transp_cost_sup: float,
        transp_cost_int: float
    ):
        self.capacity1: int = capacity1
        self.capacity2: int = capacity2
        self.poisson_lambda1: float = poisson_lambda1
        self.poisson_lambda2: float = poisson_lambda2
        self.holding_cost1: float = holding_cost1
        self.holding_cost2: float = holding_cost2
        self.stockout_cost1: float = stockout_cost1
        self.stockout_cost2: float = stockout_cost2
        self.transp_cost_sup: float = transp_cost_sup
        self.transp_cost_int: float = transp_cost_int

        self.poisson_distr1 = poisson(poisson_lambda1)
        self.poisson_distr2 = poisson(poisson_lambda2)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[OrderAction, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha1 in range(self.capacity1 + 1):
            for beta1 in range(self.capacity1 + 1 - alpha1):
                for alpha2 in range(self.capacity2 + 1):
                    for beta2 in range(self.capacity2 + 1 - alpha2):
                        state: InventoryState = InventoryState(alpha1, beta1, alpha2,beta2)
                        ip1: int = state.inventory_position1()
                        ip2: int = state.inventory_position2()
                        d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}
                        for int_move in range(-alpha2,alpha1+1):
                            ip1_m = ip1 -int_move
                            ip2_m = ip2 +int_move
                            for order1 in range(self.capacity1 - ip1_m + 1):

                                for order2 in range(self.capacity2 - ip2_m + 1):
                                    # base reward calc - holding costs + internal transport + supply transport
                                    base_reward: float = - self.holding_cost1 * (alpha1-int_move) \
                                         - self.holding_cost2 * (alpha2 + int_move)
                                    if order1 != 0: base_reward -= self.transp_cost_sup
                                    if order2 != 0: base_reward -= self.transp_cost_sup
                                    if int_move != 0: base_reward -= self.transp_cost_int

                                    ord_act = OrderAction(int_move, order1,order2)
                                    # add in transportation costs:

                                    # neither store runs out of inventory
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                                        {(InventoryState(ip1_m - i1, order1,ip2_m - i2, order2), base_reward):
                                         self.poisson_distr1.pmf(i1)*self.poisson_distr2.pmf(i2)\
                                          for i1 in range(ip1_m) for i2 in range(ip2_m)}

                                    # Run out probabilities
                                    cum_prob1: float = 1 - self.poisson_distr1.cdf(ip1_m - 1)
                                    cum_prob2: float = 1 - self.poisson_distr2.cdf(ip2_m - 1)

                                    # Run out expected costs (independent events and expectation is linear)
                                    stock_out_cost1 = -self.stockout_cost1*(self.poisson_lambda1 - ip1_m)\
                                            +ip1_m*self.poisson_distr1.pmf(ip1_m)
                                    stock_out_cost2 = -self.stockout_cost2*(self.poisson_lambda2 - ip2_m)\
                                            +ip1_m*self.poisson_distr2.pmf(ip2_m)


                                    # Store 1 runs out of inv
                                    for i2 in range(ip2_m):
                                         sr_probs_dict[(InventoryState(0, order1,ip2_m - i2, order2),\
                                                base_reward+stock_out_cost1)] \
                                                        = self.poisson_distr1.pmf(i2)*cum_prob1
                                    # Store 2 runs out of inv
                                    for i1 in range(ip1_m):
                                         sr_probs_dict[(InventoryState(ip1_m - i1, order1,0, order2),\
                                                base_reward+stock_out_cost2)] \
                                                        = self.poisson_distr1.pmf(i1)*cum_prob2

                                    # both stores run out of inventory
                                    sr_probs_dict[(InventoryState(0, order1,0, order2),\
                                                base_reward+stock_out_cost2+stock_out_cost1)] \
                                                        = cum_prob1*cum_prob2

                                    #Voila!
                                    d1[ord_act] = Categorical(sr_probs_dict)
                        d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity1 = 2
    user_capacity2 = 2
    user_poisson_lambda1 = 1.0
    user_poisson_lambda2 = 1.0
    user_holding_cost1 = 1.0
    user_holding_cost2 = 1.0
    user_stockout_cost1 = 10.0
    user_stockout_cost2 = 10.0
    user_transp_cost_sup = 1.0
    user_transp_cost_int  = 1.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity1=user_capacity1,
            capacity2=user_capacity2,
            poisson_lambda1=user_poisson_lambda1,
            poisson_lambda2=user_poisson_lambda2,
            holding_cost1=user_holding_cost1,
            holding_cost2=user_holding_cost2,
            stockout_cost1=user_stockout_cost1,
            stockout_cost2=user_stockout_cost2,
            transp_cost_sup = user_transp_cost_sup,
            transp_cost_int = user_transp_cost_int
        )


    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result


    print("MDP Policy Iteration Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    print(opt_policy_pi)
