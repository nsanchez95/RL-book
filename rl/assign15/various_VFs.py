from typing import Sequence, Tuple, Mapping

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]
import numpy as np
from rl.function_approx import LinearFunctionApprox
import random

def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    sum_tracker = {}
    count_tracker = {}
    value_tracker ={}
    for s,returns_ in state_return_samples:
        sum_tracker[s] = sum_tracker.get(s,0.)+returns_
        count_tracker[s] = count_tracker.get(s,0)+1
        value_tracker[s] = sum_tracker[s]/float(count_tracker[s])
    return value_tracker


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    prob_tracker = {}
    reward_tracker = {}
    count_tracker = {}
    for s1,r,s2 in srs_samples:
        reward_tracker[s1] = reward_tracker.get(s1,0.) + r
        prob_tracker[s1] = prob_tracker.get(s1,{})
        prob_tracker[s1][s2] = prob_tracker[s1].get(s2,0) + 1
        count_tracker[s1] = count_tracker.get(s1,0) + 1

    for s_from in prob_tracker:
        reward_tracker[s_from] = reward_tracker[s_from]/float(count_tracker[s_from])
        for s_to in prob_tracker[s_from]:
            prob_tracker[s_from][s_to] = float(prob_tracker[s_from][s_to])/float(count_tracker[s_from])

    return prob_tracker, reward_tracker


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states_set = reward_func.keys()
    num_states = len(states_set)
    prob_matrix = np.zeros((num_states,num_states))
    rew_vector = np.zeros((num_states))
    
    for i, s1 in enumerate(states_set):
        rew_vector[i] = reward_func[s1]
        for j, s2 in enumerate(states_set):
            prob_matrix[i,j] = prob_func.get(s1,{}).get(s2,0.)

    val_soln = np.linalg.lstsq(np.identity(num_states) - prob_matrix, rew_vector, rcond =None)[0]

    return {s : val_soln[i] for i,s in enumerate(states_set)}

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    v = {}
    for i in range(num_updates):
        s1,r,s2 = random.choice(srs_samples)
        alpha = learning_rate*(float(i)/learning_rate_decay + 1)**-0.5
        v[s1] =  (1.-alpha)*v.get(s1,0.) + alpha*(r+v.get(s2,0))
    return v

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    non_terminal_states = np.unique([s for s,_,_  in srs_samples])
    num_states = non_terminal_states.shape[0]
    feat_func = lambda x : (x ==non_terminal_states).astype(int)
    A = np.zeros((num_states,num_states))
    b = np.zeros(num_states)
    for s1,r,s2 in srs_samples:
        A += np.outer(feat_func(s1), feat_func(s1) - feat_func(s2))
        b += feat_func(s1)*r

    weights_final = np.linalg.lstsq(A,b, rcond =None)[0]
    return {s:weights_final[i] for i,s in enumerate(non_terminal_states)}

if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)

    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
