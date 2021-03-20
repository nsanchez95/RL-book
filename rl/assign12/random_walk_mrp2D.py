from typing import Mapping, Dict, Optional, Tuple
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovRewardProcess


class RandomWalkMRP2D(FiniteMarkovRewardProcess[Tuple[int,int]]):
    '''
    This MRP's states are {0, 1, 2,...,self.barrier}
    with 0 and self.barrier as the terminal states.
    At each time step, we go from state i to state
    i+1 with probability self.p or to state i-1 with
    probability 1-self.p, for all 0 < i < self.barrier.
    The reward is 0 if we transition to a non-terminal
    state or to terminal state 0, and the reward is 1
    if we transition to terminal state self.barrier
    '''
    barrier_top: int
    barrier_right: int
    p_up: float
    p_down: float
    p_right: float

    def __init__(
        self,
        barrier_top: int,
        barrier_right: int,
        p_up: float,
        p_down: float,
        p_right: float
    ):
        self.barrier_top = barrier_top
        self.barrier_right = barrier_right
        self.p_up = p_up
        self.p_down = p_down
        self.p_right = p_right
        self.p_left = 1. - p_up - p_right - p_down        
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[Tuple[int,int], Optional[Categorical[Tuple[Tuple[int,int], float]]]]:


        d: Dict[Tuple[int,int], Optional[Categorical[Tuple[Tuple[int,int], float]]]] = {
            (i,j): Categorical({
                ((i + 1,j), 0. if i < self.barrier_right - 1 else 1.): self.p_right,
                ((i,j+1), 0. if j < self.barrier_top - 1 else 1.): self.p_up,
                ((i - 1,j), 0.): self.p_left,
                ((i,j-1), 0.): self.p_down
            }) for i in range(1, self.barrier_right) for j in range(1,self.barrier_top)
        }
        for i in range(self.barrier_right):
            d[(i,0)] = None
            d[(i,self.barrier_top)] = None

        for j in range(self.barrier_right):
            d[(0,j)] = None
            d[(self.barrier_right,j)] = None

        return d


if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc

    this_barrier_top: int = 10
    this_barrier_right: int = 10
    this_p_up: float = 0.25
    this_p_right: float = 0.25
    this_p_down: float = 0.25
    random_walk: RandomWalkMRP2D = RandomWalkMRP2D(
        barrier_top=this_barrier_top,
        barrier_right=this_barrier_right,
        p_up=this_p_up,
        p_down=this_p_down,
        p_right=this_p_right
    )

    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=800,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )
