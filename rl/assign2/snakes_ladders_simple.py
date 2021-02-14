from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical,SampledDistribution,Constant
from rl.markov_process import Transition, FiniteMarkovProcess, FiniteMarkovRewardProcess, RewardTransition
import numpy as np
from collections import Counter
from operator import itemgetter
from rl.gen_utils.plot_funcs import plot_list_of_curves
import itertools

@dataclass(frozen=True)
class SLPositionState:
    tile_num: int

@dataclass
class SnakeLadderMP1(FiniteMarkovProcess[SLPositionState]):
    ladder_snake_map : Dict[int,int]
    def __init__(self, ladder_snake_map : Dict[int,int]):
        self.ladder_snake_map = ladder_snake_map
        super().__init__(self.get_transition_map())


    def get_transition_map(self) -> Transition[SLPositionState]:
        d : Dict[SLPositionState, Categorical[SLPositionState]] = {}
        for i in range(1,100):
            prob_list = {}
            for j in range(i+1,i+7):
                if j in self.ladder_snake_map.keys():
                    prob_list[SLPositionState(self.ladder_snake_map[j])] = 1./6.
                else:
                    prob_list[SLPositionState(j)] = 1./6.
            d[SLPositionState(i)] = Categorical(prob_list)
        for i in range(100,107):
            d[SLPositionState(i)] = None
        return d


@dataclass
class SnakeLadderMRP1(FiniteMarkovRewardProcess[SLPositionState]):
    ladder_snake_map : Dict[int,int]
    def __init__(self, ladder_snake_map : Dict[int,int]):
        self.ladder_snake_map = ladder_snake_map
        super().__init__(self.get_transition_reward_map())


    def get_transition_reward_map(self) -> RewardTransition[SLPositionState]:
        d : Dict[SLPositionState, Categorical[Tuple[SLPositionState, float]]] = {}
        for i in range(1,100):
            prob_list = {}
            for j in range(i+1,i+7):
                if j in self.ladder_snake_map.keys():
                    prob_list[(SLPositionState(self.ladder_snake_map[j]),1)] = 1./6.
                else:
                    prob_list[(SLPositionState(j),1)] = 1./6.
            d[SLPositionState(i)] = Categorical(prob_list)
        for i in range(100,107):
            d[SLPositionState(i)] = None
        return d


def plot_distr_rolls_to_finsh(si_mp:SnakeLadderMP1, num_traces : int):
    lens = []
    for i in range(num_traces):
        trace = si_mp.simulate(start_state_distribution)
        lens.append(sum(1 for _ in trace))
    hist_toplot = sorted(Counter(lens).items(),key = itemgetter(0))
    lens,counts_finish = [x for x, _ in hist_toplot], [y for _, y in hist_toplot]
    probs_finish = np.array(counts_finish)
    probs_finish = probs_finish/probs_finish.sum()
    plot_list_of_curves(
        [lens],
        [probs_finish],
        ["r"],
        [
            r"Snake and Ladder MP"
        ],
        "# Dice Rolls to Finish",
        "Probability",
        "Probability Distribution of Num Dice to Finish"
    )

if __name__ == '__main__':
    ladder_snake_ladders = {1:38,4:14,9:31,21:24,28:84,36:44,51:67, 71:91, 80:100}
    ladder_snake_map = {98:78, 95:75, 93:73, 87:24, 64:60, 62:19, 56:53, 49:11, 47:26, 16:6}
    ladder_snake_map.update(ladder_snake_ladders)
    start_state_distribution = Constant(SLPositionState(tile_num = 1))
    

    ####### FINITE MARKOV PROCESS
    si_mp = SnakeLadderMP1(ladder_snake_map)

    ## PRINT EXAMPLE TRACE
    trace_ex = [s.tile_num for s in si_mp.simulate(start_state_distribution)]
    print(trace_ex)

    ## PLOT DISTRIBUTION OF NUM ROLLS NEEDED TO FINISH THE GAME
    # plot_distr_rolls_to_finsh(si_mp, 10000)
 

    ####### FINITE MARKOV REWARD PROCESS
    # display value and look for the value for position 1
    si_mrp = SnakeLadderMRP1(ladder_snake_map)
    si_mrp.display_value_function(1)




