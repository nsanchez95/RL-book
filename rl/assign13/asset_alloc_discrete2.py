from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, TypeVar
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian, Constant
from rl.markov_decision_process import MarkovDecisionProcess, Policy, policy_from_q
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox, FunctionApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from rl.approximate_dynamic_programming import back_opt_qvf
from operator import itemgetter
from rl.returns import returns
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

@dataclass(frozen=True)
class AssetAllocState:
    wealth: int
    time: int

@dataclass(frozen=True)
class AssetAllocMDP(MarkovDecisionProcess[AssetAllocState, float]):
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[AssetAllocState, float]], float]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def uniform_actions(self) -> Choose[float]:
        return Choose(set(self.risky_alloc_choices))

    def step(
        self,
        wealth_state: AssetAllocState,
        alloc: float
    ) -> SampledDistribution[Tuple[AssetAllocState, float]]:
        def sr_sampler_func(
            wealth_state=wealth_state,
            alloc=alloc
        ) -> Tuple[float, float]:
            next_wealth: float = alloc * (1 + self.risky_return_distributions[wealth_state.time].sample()) \
                + (wealth_state.wealth - alloc) * (1 + self.riskless_returns[wealth_state.time])
            reward: float = self.utility_func(next_wealth) \
                if wealth_state.time == self.time_steps() - 1 else 0.
            return (AssetAllocState(next_wealth,wealth_state.time+1), reward)

        return SampledDistribution(
            sampler=sr_sampler_func,
            expectation_samples=1000
        )

    def actions(self, wealth_state: AssetAllocState) -> Sequence[float]:
        return self.risky_alloc_choices if wealth_state.time < self.time_steps() else []

    def get_qvf_func_approx(self) -> DNNApprox[Tuple[AssetAllocState, float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return DNNApprox.create(
            feature_functions=self.feature_functions,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

def q_learning(
    mdp: MarkovDecisionProcess[AssetAllocState, float],
    init_states: Distribution[float],
    approx_0: FunctionApprox[Tuple[AssetAllocState, float]],
    gamma: float
) -> Iterator[FunctionApprox[Tuple[AssetAllocState, float]]]:

    q = approx_0
    while True:
        state = AssetAllocState(init_states.sample(),0)
        policy = policy_from_q(q,mdp,0.2)
        while True:
            action = policy.act(state).sample()
            next_state, reward = mdp.step(state,action).sample()
            next_reward = 0. if mdp.is_terminal(next_state) else max(
                q((next_state, a))
                for a in mdp.actions(next_state)
            )
            q = q.update([((state, action), reward+gamma*next_reward)])
            state = next_state
            if mdp.is_terminal(next_state):
                break
        yield q

def td_sarsa(
        mdp: MarkovDecisionProcess[AssetAllocState, float],
        states: Distribution[float],
        approx_0: FunctionApprox[Tuple[AssetAllocState, float]],
        eps: Callable[[int], float],
        gamma: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    state_distribution = states
    trace_count = 1
    while True:
        p = policy_from_q(q, mdp, eps(trace_count))
        state = AssetAllocState(state_distribution.sample(),0)
        action = p.act(state).sample()
        next_state, reward = mdp.step(state,action).sample()
        if mdp.is_terminal(next_state):
            q = q.update([((state, action),reward)])
        else:
            next_action = p.act(next_state).sample()
            q = q.update([( (state, action),reward+gamma*q((next_state,next_action)))])
        trace_count+=1
        yield q

def mc_control(
        mdp: MarkovDecisionProcess[AssetAllocState, float],
        states: Distribution[float],
        approx_0: FunctionApprox[Tuple[AssetAllocState, float]],
        eps: Callable[[int], float],
        gamma: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    p = policy_from_q(q, mdp)
    trace_count = 1
    while True:
        trace: Iterable[markov_decision_process.TransitionStep[S, A]] =\
            mdp.simulate_actions(Constant(AssetAllocState(states.sample(),0)), p)

        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, gamma, tolerance)
        )

        p = policy_from_q(q, mdp, eps(trace_count))
        trace_count += 1
        yield q
if __name__ == '__main__':

    from pprint import pprint

    steps: int = 1
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]

    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[AssetAllocState, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0].wealth,
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_var)

    aad: AssetAllocMDP = AssetAllocMDP(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )
    init_fa: DNNApprox[Tuple[float, float]] = aad.get_qvf_func_approx()
    step_count = 0
    for q_approx in mc_control(aad,init_wealth_distr,init_fa, lambda x: 1./float(x), 1.):
        step_count +=1
        if (step_count%10000 == 0):
            print("step ", step_count)
            # print(f"Time {1:d}")
            print()
            opt_alloc: float = max(
                ((q_approx.evaluate([(AssetAllocState(init_wealth,0), ac)])[0], ac) for ac in alloc_choices),
                key=itemgetter(0)
            )[1]
            val: float = max(q_approx.evaluate([(AssetAllocState(init_wealth,0), ac)])[0]
                             for ac in alloc_choices)
            print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
            print("Optimal Weights below:")
            for wts in q_approx.weights:
                pprint(wts.weights)
            print()


    # for t in range(steps):
    #     print(f"Time {t:d}")
    #     print()
    #     left: int = steps - t
    #     growth: float = (1 + r) ** (left - 1)
    #     alloc: float = base_alloc / growth
    #     val: float = - np.exp(- excess * excess * left / (2 * var)
    #                           - a * growth * (1 + r) * init_wealth) / a
    #     bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
    #         np.log(np.abs(a))
    #     w_t_wt: float = a * growth * (1 + r)
    #     x_t_wt: float = a * excess * growth
    #     x_t2_wt: float = - var * (a * growth) ** 2 / 2

    #     print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {val:.3f}")
    #     print(f"Bias Weight = {bias_wt:.3f}")
    #     print(f"W_t Weight = {w_t_wt:.3f}")
    #     print(f"x_t Weight = {x_t_wt:.3f}")
    #     print(f"x_t^2 Weight = {x_t2_wt:.3f}")
    #     print()

'''

Time 3

Opt Risky Allocation = 1.500, Opt Val = -0.328
Bias Weight = 0.000
W_t Weight = 1.070
x_t Weight = 0.060
x_t^2 Weight = -0.020
'''


'''
Q-Learning 
Opt Risky Allocation = 1.500, Opt Val = -0.248
Optimal Weights below:
array([[-0.06196071,  1.19015488,  0.34865245, -0.11426409]])
'''
