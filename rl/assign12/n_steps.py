'''lambda-return and TD(lambda) methods for working with prediction and control

'''

from typing import Iterable, Iterator, TypeVar, List, Sequence,Tuple, Dict
import rl.markov_process as mp
import numpy as np

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns

S = TypeVar('S')

def td_lambda_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        gamma: float,
        lambd: float
) -> Iterator[FunctionApprox[S]]:
    '''Value Function Prediction using the td lambda method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      n_steps -- number of steps for bootstrapping
      approx_0 -- initial approximation of value function
      gamma -- discount rate (0 < γ ≤ 1)
    '''
    func_approx: FunctionApprox[S] = approx_0

    for trace in traces:
        trace_eligibility : Dict[S,float] = {}

        #these will be fed into function approximation

        trace_seq: Sequence[mp.TransitionStep[S]] = list(trace)
        for t, tr in enumerate(trace_seq):
            td_error = tr.reward - gamma*func_approx(tr.next_state) - func_approx(tr.state)
            for state in trace_eligibility:
                if state == t.state:
                    trace_eligibility[t.state] = gamma*lambd*trace_eligibility.get(t.state,0.) + 1
                else:
                    trace_eligibility[state] = gamma*lambd*trace_eligibility[state]
                func_approx=func_approx.update([(state,td_error*trace_eligibility[state]+func_approx(state))])

        yield func_approx



def n_step_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        n_steps: int,
        approx_0: FunctionApprox[S],
        gamma: float,
        lambd: float
) -> Iterator[FunctionApprox[S]]:
    '''Value Function Prediction using the n-step bootstrapping method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      n_steps -- number of steps for bootstrapping
      approx_0 -- initial approximation of value function
      gamma -- discount rate (0 < γ ≤ 1)
    '''
    func_approx: FunctionApprox[S] = approx_0

    for trace in traces:
        # these will be sliding windows to look at n-long windows
        relevant_states = []
        relevant_rewards = []
        
        #these will be fed into function approximation
        predictors: List[S] = []
        responses: Sequence[float] = []

        trace_seq: Sequence[mp.TransitionStep[S]] = list(trace)
        for t, tr in enumerate(trace_seq):
            if t < n_steps:
                # gather enough step for n-step bootstrapping
                relevant_states.append(tr.state)
                relevant_rewards.append(tr.reward)
            else:

                # record the pair
                predictors.append(relevant_states[0])
                rew = (gamma**n_steps)*func_approx(relevant_states[-1].next_state)
                for i in range(n_steps):
                    rew += (gamma**i)*relevant_rewards[i]
                responses.append(rew)

                # update the sliding window
                relevant_states.append(tr.state)
                relevant_rewards.append(tr.reward)
                relevant_states = relevant_states[1:]
                relevant_rewards = relevant_rewards[1:]

        func_approx = func_approx.update(zip(predictors, responses))
        yield func_approx


def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        γ: float,
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)

    '''
    def step(v, transition):
        return v.update([(transition.state,
                          transition.reward + γ * v(transition.next_state))])

    return iterate.accumulate(transitions, step, initial=approx_0)


def lambda_return_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: FunctionApprox[S],
        γ: float,
        lambd: float
) -> Iterator[FunctionApprox[S]]:
    '''Value Function Prediction using the lambda-return method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''
    func_approx: FunctionApprox[S] = approx_0

    for trace in traces:
        gp: List[float] = [1.]
        lp: List[float] = [1.]
        predictors: List[S] = []
        partials: List[List[float]] = []
        weights: List[List[float]] = []
        trace_seq: Sequence[mp.TransitionStep[S]] = list(trace)
        for t, tr in enumerate(trace_seq):
            for i, partial in enumerate(partials):
                partial.append(
                    partial[-1] +
                    gp[t - i] * (tr.reward - func_approx(tr.state)) +
                    (gp[t - i] * γ * func_approx(tr.next_state)
                     if t < len(trace_seq) - 1 else 0.)
                )
                weights[i].append(
                    weights[i][-1] * lambd if t < len(trace_seq)
                    else lp[t - i]
                )
            predictors.append(tr.state)
            partials.append([tr.reward + (γ * func_approx(tr.next_state)
                             if t < len(trace_seq) - 1 else 0.)])
            weights.append([1. - (lambd if t < len(trace_seq) else 0.)])
            gp.append(gp[-1] * γ)
            lp.append(lp[-1] * lambd)
        responses: Sequence[float] = [np.dot(p, w) for p, w in
                                      zip(partials, weights)]
        func_approx = func_approx.update(zip(predictors, responses))
        yield func_approx
