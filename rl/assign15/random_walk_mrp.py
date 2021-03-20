from typing import Mapping, Dict, Optional, Tuple, Sequence, TypeVar, Callable
from rl.distribution import Categorical, Distribution, Choose, Constant
from rl.markov_process import FiniteMarkovRewardProcess,MarkovRewardProcess
from rl.dynamic_programming import almost_equal_np_arrays
import numpy as np

X = TypeVar('X')

class TDC:

    def __init__(
        self,
        feat_funcs: Sequence[Callable[[X],float]],
        alpha: float,
        beta: float
    ):
        self.feat_funcs = feat_funcs
        self.num_feats = len(feat_funcs)
        self.w = np.ones(self.num_feats)*0.2
        self.theta = np.zeros(self.num_feats)
        self.alpha = alpha
        self.beta = beta

    def featurize(self, s: X):
        return np.array([f(s) for f in self.feat_funcs])

    def update_weights(self,s:X,s_next:X,td:float, gamma:float):
        s_f = self.featurize(s)
        s_next_f = self.featurize(s_next)
        self.w = self.w + self.alpha*(td*s_f-gamma*s_next_f*np.dot(theta,s_f))
        self.theta = self.theta + self.beta*(td-np.dot(self.theta,s_f))*s_f

    def evaluate(self, s:X):
        return np.dot(self.w,self.featurize(s))

    def update_weights(self,s:X,s_next:X,reward:float, gamma:float, term:bool):
        s_f = self.featurize(s)
        s_next_f = self.featurize(s_next)
        td = reward+gamma*self.evaluate(s_next)*float(1-int(term)) - self.evaluate(s)
        self.w = self.w + self.alpha*(td*s_f-gamma*s_next_f*np.dot(self.theta,s_f)*float(1-int(term)))
        self.theta = self.theta + self.beta*(td-np.dot(self.theta,s_f))*s_f
        return self.alpha*(td*s_f-gamma*s_next_f*np.dot(self.theta,s_f))
     
    def learn_policy(self,mrp:MarkovRewardProcess[X], state_dist : Distribution[X], gamma:float, tolerance = 1e-5):
        count = 0
        while True:
            state = state_dist.sample()
            while(not(mrp.is_terminal(state))):
                state = state_dist.sample()
                next_state, reward = mrp.transition_reward(state).sample()
                diff = self.update_weights(state, next_state,reward,gamma, mrp.is_terminal(next_state))
                state = next_state
                count+=1
                if np.max(np.abs(diff)) < 10000 and count >20000:
                    return
class RandomWalkMRP(FiniteMarkovRewardProcess[int]):
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
    barrier: int
    p: float

    def __init__(
        self,
        barrier: int,
        p: float
    ):
        self.barrier = barrier
        self.p = p
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[int, Optional[Categorical[Tuple[int, float]]]]:
        d: Dict[int, Optional[Categorical[Tuple[int, float]]]] = {
            i: Categorical({
                (i + 1, 0. if i < self.barrier - 1 else 1.): self.p,
                (i - 1, 0.): 1 - self.p
            }) for i in range(1, self.barrier)
        }
        d[0] = None
        d[self.barrier] = None
        return d


if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc


    this_barrier: int = 10
    this_p: float = 0.5
    random_walk: RandomWalkMRP = RandomWalkMRP(
        barrier=this_barrier,
        p=this_p
    )

    feat_funcs = [lambda x : float(x) if x>0 else 0.0]
    alpha = 0.001
    beta = 0.002
    gamma = 1.0
    tdc_prediction = TDC(feat_funcs, alpha, beta)
    tdc_prediction.learn_policy(random_walk,Choose(random_walk.non_terminal_states), gamma)
    for s in random_walk.non_terminal_states:
        print("For state ", s, " TDC estimates ", tdc_prediction.evaluate(s), " and correct is ", s/float(this_barrier))
