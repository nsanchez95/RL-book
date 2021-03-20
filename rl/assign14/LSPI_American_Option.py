from dataclasses import dataclass, replace
from typing import Callable, Sequence, Tuple, List, TypeVar, Iterable, Optional
import numpy as np
from scipy.stats import norm
from rl.function_approx import \
    DNNApprox, LinearFunctionApprox, FunctionApprox, DNNSpec, AdamGradient
from random import randrange
from numpy.polynomial.laguerre import lagval

TrainingDataType = Tuple[int, float, float]
X = TypeVar('X')
class LSTD(FunctionApprox[X]):
    A_matrix:np.ndarray
    b_vector:np.ndarray
    w:np.ndarray
    feat_func: Callable[[X],np.ndarray]

    def __init__(
        self,
        num_feats: int,
        feat_func: Callable[[X],np.ndarray],
    ):
        self.feat_func = feat_func
        self.A_matrix: np.ndarray = np.zeros((num_feats,num_feats))
        self.b_vector: np.ndarray = np.zeros(num_feats)
        self.w: np.ndarray = np.zeros(num_feats)
        super().__init__()

    ########### MUST MEET FUNCTIONAPPROX SPEC ###############
    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> FunctionApprox[X]:
        return self

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> FunctionApprox[X]:
        return self

    def representational_gradient(self, x_value: X) -> FunctionApprox[X]:
        return self

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array([np.dot(self.weights, self.feat_func(x)) for x in x_values_seq])

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        return False

    ########### ADDITIONAL FUNCTIONS #######################
    def train_up(self, s1: X, rew: float, s2: X, gamma:float, term: bool) -> None:
        s1_feat = self.feat_func(s1)
        s2_feat = self.feat_func(s2)
        self.A_matrix += s1_feat.T.dot(s1_feat - gamma*s2_feat*(1-float(int(term))))
        self.b_vector+= rew*s1_feat*gamma

    def update_weights(self, normalization = 1.0, reg_val = 0.001) -> None:
        self.A_matrix *=normalization 
        self.b_vector *=normalization
        self.A_matrix += reg_val*np.eye(self.b_vector.shape[0])
        self.w = np.linalg.inv(self.A_matrix).dot(self.b_vector)

    def reset_count(self) -> None:
        self.A_matrix = np.zeros((num_feats,num_feats))
        self.b_vector =np.zeros(num_feats)

    def evaluate_vf(self, s: X) -> float:
        s_feat = feat_func(s)
        return np.dot(self.w, s_feat)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 200

    dt: float = expiry_val / num_steps_val

    num_training_paths: int = 5000
    spot_price_frac_val: float = 0.0

    dql_training_iters: int = 1000000
    lspi_training_iters: int = 8

    split_val: int = 1000

    def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)


    #### create LSPI object
    num_feats = 7
    num_laguerre: int = 3
    ident: np.ndarray = np.eye(num_laguerre)
    feat_func = lambda ts : np.array([\
            1.,\
            np.exp(-ts[1] / (2 * strike)) *\
                  lagval(ts[1] / strike, ident[0]),\
            np.exp(-ts[1] / (2 * strike)) *\
                  lagval(ts[1] / strike, ident[1]),\
            np.exp(-ts[1] / (2 * strike)) *
                  lagval(ts[1] / strike, ident[2]),\
            np.cos(-ts[0] * np.pi / (2 * expiry_val)),
            np.log(expiry_val - ts[0]) if ts[0] != expiry_val else 0.,
            (ts[0] / expiry_val) ** 2 ])
    lstd_approx = LSTD(num_feats, feat_func)


    ### run simulations and updates
    ret: List[TrainingDataType] = []
    spot: float = spot_price_val
    vol2: float = vol_val * vol_val

    mean2: float = spot * spot
    var: float = mean2 * spot_price_frac_val * spot_price_frac_val
    log_mean: float = np.log(mean2 / np.sqrt(var + mean2))
    log_stdev: float = np.sqrt(np.log(var / mean2 + 1))
    gamma: float = np.exp(-rate_val * dt)
    for train_iter in range(lspi_training_iters):
        print("Training Iteration number ", train_iter )
        for i in range(num_training_paths):
            price: float = np.random.lognormal(log_mean, log_stdev)
            for step in range(num_steps_val):
                m: float = np.log(price) + (rate_val - vol2 / 2) * dt
                v: float = vol2 * dt
                next_price: float = np.exp(np.random.normal(m, np.sqrt(v)))
                state = (step*dt,price)
                next_state = ((step+1)*dt,next_price)
                ex: bool = lstd_approx.evaluate_vf(next_state) > payoff_func(*next_state)
                term: bool = step+1 == num_steps_val
                keep:bool = not( ex or term)
                lstd_approx.train_up(
                    state,\
                    0.0 if keep else payoff_func(*next_state),\
                    next_state,\
                    gamma,\
                    not keep)
                price = next_price

        lstd_approx.update_weights(normalization = 1./float(num_training_paths*num_steps_val))
        print(lstd_approx.w)
        lstd_approx.reset_count()


    # num_scoring_paths: int = 10000

    

    # opt_ex_rl: OptimalExerciseRL = OptimalExerciseRL(
    #     spot_price=spot_price_val,
    #     payoff=payoff_func,
    #     expiry=expiry_val,
    #     rate=rate_val,
    #     vol=vol_val,
    #     num_steps=num_steps_val
    # )

    # training_data: Sequence[TrainingDataType] = opt_ex_rl.training_sim_data(
    #     num_paths=num_training_paths,
    #     spot_price_frac=spot_price_frac_val
    # )
    # print("Generated Training Data")

    # flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
    #     obj=opt_ex_rl,
    #     strike=strike,
    #     expiry=expiry_val,
    #     training_data=training_data,
    #     training_iters=lspi_training_iters,
    #     split=split_val
    # )
    # print("Generated Training Data")
