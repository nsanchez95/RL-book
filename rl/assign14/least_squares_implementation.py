'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterable, Iterator, Tuple, TypeVar

from rl.distribution import Distribution, Choose
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns

S = TypeVar('S')
A = TypeVar('A')

def mc_control(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        gamma: float,
        eps: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      ϵ -- the fraction of the actions where we explore rather
      than following the optimal policy
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q = approx_0
    p = markov_decision_process.policy_from_q(q, mdp)

    while True:
        trace: Iterable[markov_decision_process.TransitionStep[S, A]] =\
            mdp.simulate_actions(states, p)
        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, gamma, tolerance)
        )
        p = markov_decision_process.policy_from_q(q, mdp, eps)
        yield q



# def td_sarsa1(
#         transitions: Iterable[markov_decision_process.TransitionStep[S,A]],
#         approx_0: FunctionApprox[S],
#         gamma: float,
# ) -> Iterator[FunctionApprox[S]]:
#     '''Evaluate an MRP using TD(0) using the given sequence of
#     transitions.

#     Each value this function yields represents the approximated value
#     function for the MRP after an additional transition.

#     Arguments:
#       transitions -- a sequence of transitions from an MRP which don't
#                      have to be in order or from the same simulation
#       approx_0 -- initial approximation of value function
#       γ -- discount rate (0 < γ ≤ 1)

#     '''
#     def step(q, transition):
#         p = markov_decision_process.policy_from_q(q, mdp, eps)
#         next_action = p.act(next_action).sample()
#         return v.update([(transition.state,
#                           transition.reward + γ * v(transition.next_state))])

#     return iterate.accumulate(transitions, step, initial=approx_0)

def td_sarsa(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        eps: float,
        gamma: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      gamma -- discount rate (0 < γ ≤ 1)
      eps -- the fraction of the actions where we explore rather
      than following the optimal policy

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q = approx_0
    state_distribution = states

    while True:
        p = markov_decision_process.policy_from_q(q, mdp, eps)
        state = state_distribution.sample()
        action = p.act(state).sample()
        next_state, reward = mdp.step(state,action)
        next_action = p.act(next_action).sample()
        q = q.update([(state, action), step.return_+gamma*q((next_state,next_action) )])
        yield q


def q_learning(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        eps: float,
        gamma: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    state_distribution = states

    while True:
        state = state_distribution.sample()
        action = Choose(mdp.actions(state)).sample()
        next_state, reward = mdp.step(state,action)
        next_reward = max(
            q((next_state, a))
            for a in mdp.actions(next_state)
        )
        q = q.update([(state, action), reward+gamma*next_reward])
        yield q

    def step(q, transition):
        next_reward = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        )
        return q.update([
            ((transition.state, transition.action),
             transition.reward + gamma * next_reward)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)

