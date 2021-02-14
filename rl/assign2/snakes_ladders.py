from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical,SampledDistribution
from rl.markov_process import Transition, FiniteMarkovProcess
from scipy.stats import poisson
import numpy as np

@dataclass(frozen=True)
class SLPositionState:
    tile_num: int

@dataclass
class SnakeLadderMP1(MarkovProcess[PositionState]):
    ladder_snakes_map : dict
    ladder_snake_ladders = {1:38,4:14,9:31,21:24,28:84,36:44,51:67, 71:91, 80:100}
    ladder_snake_map = {98:78, 95:75, 93:73, 87:24, 64:60, 62:19, 56:53, 49:11, 47:26, 16:6}
    ladder_snake_map.update(ladder_snake_ladders)

    def next_tile_sampler(start_state : SLPositionState) -> SLPositionState:
        curr_tile = start_state.tile_num
        while(True):
            dice = np.random.randint(6) + 1
            curr_tile +=  dice
            if curr_tile > 100: curr_tile = 100
            if next_num in ladder_snake_map.keys():
                next_num = ladder_snake_map[next_num]
            if dice < 6: break
        return SLPositionState(tile_num = curr_tile)


    def transition(self, state: PositionState) -> SampledDistribution[PositionState]:
        return SampledDistribution(
            sampler = lambda: self.next_tile_sampler(start_state=state),\
            expectation_samples = 100)


def process_snake_ladder() -> np.ndarray:
    mp = SnakeLadderMP1()
    start_state_distribution = Constant(SLPositionState(tile_num = 0))
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])