'''An interface for different kinds of function approximations
(tabular, linear, DNN... etc), with several implementations.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
import itertools
import numpy as np
from operator import itemgetter
from scipy.interpolate import splrep, BSpline
import scipy
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)
import matplotlib.pyplot as plt
import rl.iterate as iterate

X = TypeVar('X')
SMALL_NUM = 1e-6


class FunctionApprox(ABC, Generic[X]):
    '''Interface for function approximations.
    An object of this class approximates some function X ↦ ℝ in a way
    that can be evaluated at specific points in X and updated with
    additional (X, ℝ) points.
    '''

    @abstractmethod
    def representational_gradient(self, x_value: X) -> FunctionApprox[X]:
        '''Computes the gradient of the self FunctionApprox with respect
        to the parameters in the internal representation of the
        FunctionApprox, i.e., computes Gradient with respect to internal
        parameters of expected value of y for the input x, where the
        expectation is with respect tp the FunctionApprox's model of
        the probability distribution of y|x. The gradient is output
        in the form of a FunctionApprox whose internal parameters are
        equal to the gradient values.
        '''

    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Computes expected value of y for each x in
        x_values_seq (with the probability distribution
        function of y|x estimated as FunctionApprox)
        '''

    def __call__(self, x_value: X) -> float:
        return self.evaluate([x_value]).item()

    @abstractmethod
    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> FunctionApprox[X]:

        '''Update the internal parameters of the FunctionApprox
        based on incremental data provided in the form of (x,y)
        pairs as a xy_vals_seq data structure
        '''

    @abstractmethod
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> FunctionApprox[X]:
        '''Assuming the entire data set of (x,y) pairs is available
        in the form of the given input xy_vals_seq data structure,
        solve for the internal parameters of the FunctionApprox
        such that the internal parameters are fitted to xy_vals_seq.
        Since this is a best-fit, the internal parameters are fitted
        to within the input error_tolerance (where applicable, since
        some methods involve a direct solve for the fit that don't
        require an error_tolerance)
        '''

    @abstractmethod
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        '''Is this function approximation within a given tolerance of
        another function approximation of the same type?
        '''

    def argmax(self, xs: Iterable[X]) -> X:
        '''Return the input X that maximizes the function being approximated.
        Arguments:
          xs -- list of inputs to evaluate and maximize, cannot be empty
        Returns the X that maximizes the function this approximates.
        '''
        # return list(xs)[np.argmax(self.evaluate(xs))] # OLD VERSION - used up iterator
        keep_xs = list(xs)
        return keep_xs[np.argmax(self.evaluate(keep_xs))]


    def rmse(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> float:
        '''The Root-Mean-Squared-Error between FunctionApprox's
        predictions (from evaluate) and the associated (supervisory)
        y values
        '''
        x_seq, y_seq = zip(*xy_vals_seq)
        errors: np.ndarray = self.evaluate(x_seq) - np.array(y_seq)
        return np.sqrt(np.mean(errors * errors))

    def iterate_updates(
        self,
        xy_seq_stream: Iterator[Iterable[Tuple[X, float]]]
    ) -> Iterator[FunctionApprox[X]]:
        '''Given a stream (Iterator) of data sets of (x,y) pairs,
        perform a series of incremental updates to the internal
        parameters (using update method), with each internal
        parameter update done for each data set of (x,y) pairs in the
        input stream of xy_seq_stream
        '''
        return iterate.accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )

    def representational_gradient_stream(
        self,
        x_values_seq: Iterable[X]
    ) -> Iterator[FunctionApprox[X]]:
        for x_val in x_values_seq:
            yield self.representational_gradient(x_val)

@dataclass(frozen=True)
class UnivariateBSpline(FunctionApprox[X]):
    '''A FunctionApprox that works exactly the same as exact dynamic
    programming. Each update for a value in X replaces the previous
    value at X altogether.

    Fields:
    latest_x -- latest_x passed in for the Univariate fitting
    latest_y -- latest_y passed in for the Univariate fitting
    '''
    degree: int
    feature_function: Callable[[X], float]
    spl_facs : np.ndarray = field(default_factory=lambda: np.array([]))
    spl_knots : np.ndarray = field(default_factory=lambda: np.array([]))
    latest_x: np.ndarray = field(default_factory=lambda: np.array([]))
    latest_y: np.ndarray = field(default_factory=lambda: np.array([]))




    def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
        return [self.feature_function(x) for x in x_values_seq]

    def representational_gradient(self, x_value: X) -> UnivariateBSpline[X]:
        feature_val: float = self.feature_function(x_value)
        spl = scipy.interpolate.Univariate(self.latest_x, self.latest_y, self.degree)
        self.spl_facs = spl.get_coeffs()
        self.spl_knots = spl.get_knots()
        dx = 1e-6
        coeff_derivs = []
        for i in range(self.spl_facs.shape[0]):
            dx_vec = (np.arange(self.spl_facs.shape[0]) == i).astype(float)*dx
            dv = scipy.interpolate.BSpline(self.knots, self.coeffs+dx_vec, self.degree)\
                    - scipy.interpolate.BSpline(self.knots, self.coeffs-dx_vec, self.degree)
            coeff_derivs.append(dv/2*dx)
        coeff_derivs = np.array(coeff_derivs)
        return replace(
            self,
            spl_facs = coeff_derivs,
        )

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Evaluate the function approximation by looking up the value in the
        mapping for each state.

        Will raise an error if an X value has not been seen before and
        was not initialized.

        '''
        spl = scipy.interpolate.UnivariateSpline(self.latest_x, self.latest_y, k = self.degree)
        return np.array([spl(x) for x in x_values_seq])

    def update(self, xy_vals_seq: Iterable[Tuple[X, float]]) -> UnivariateBSpline[X]:
        '''Update each X value by replacing its saved Y with a new one. Pairs
        later in the list take precedence over pairs earlier in the
        list.

        '''
        x,y = zip(*xy_vals_seq) 
        new_x = np.append(self.latest_x, np.array(self.get_feature_values(x)))
        new_y = np.append(self.latest_y, np.array(y))
        ind_sort = np.argsort(new_x)
        return replace(
            self,
            latest_x=new_x[ind_sort],
            latest_y=new_y[ind_sort]
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> UnivariateBSpline[X]:
        x,y = zip(*xy_vals_seq)
        x = np.array(self.get_feature_values(x))
        y = np.array(y)
        ind_sort = np.argsort(x)
        self.latest_x = x[ind_sort]
        self.latest_y = y[ind_sort]
        return replace(
            self,
            latest_x=x[ind_sort],
            latest_y=y[ind_sort]
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        '''This approximation is within a tolerance of another if the value
        for each X in both approximations is within the given
        tolerance.

        Raises an error if the other approximation is missing states
        that this approximation has.

        '''
        if not isinstance(other, UnivariateBSpline):
            return False

        return np.all(self.evaluate(latest_x) - other.evaluate(latest_x) <= tolerance).item()



@dataclass(frozen=True)
class BSplineApprox(FunctionApprox[X]):
    feature_function: Callable[[X], float]
    degree: int
    knots: np.ndarray = field(default_factory=lambda: np.array([]))
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
        return [self.feature_function(x) for x in x_values_seq]

    def representational_gradient(self, x_value: X) -> BSplineApprox[X]:
        feature_val: float = self.feature_function(x_value)
        eps: float = 1e-6
        one_hots: np.array = np.eye(len(self.coeffs))
        return replace(
            self,
            coeffs=np.array([(
                BSpline(
                    self.knots,
                    c + one_hots[i] * eps,
                    self.degree
                )(feature_val) -
                BSpline(
                    self.knots,
                    c - one_hots[i] * eps,
                    self.degree
                )(feature_val)
            ) / (2 * eps) for i, c in enumerate(self.coeffs)]))

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        spline_func: Callable[[Sequence[float]], np.ndarray] = \
            BSpline(self.knots, self.coeffs, self.degree)
        return spline_func(self.get_feature_values(x_values_seq))

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> BSplineApprox[X]:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: Sequence[float] = self.get_feature_values(x_vals)
        sorted_pairs: Sequence[Tuple[float, float]] = \
            sorted(zip(feature_vals, y_vals), key=itemgetter(0))
        new_knots, new_coeffs, _ = splrep(
            [f for f, _ in sorted_pairs],
            [y for _, y in sorted_pairs],
            k=self.degree
        )
        return replace(
            self,
            knots=new_knots,
            coeffs=new_coeffs
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> BSplineApprox[X]:
        return self.update(xy_vals_seq)

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, BSplineApprox):
            return \
                np.all(np.abs(self.knots - other.knots) <= tolerance).item() \
                and \
                np.all(np.abs(self.coeffs - other.coeffs) <= tolerance).item()

        return False



x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
trialN = UnivariateBSpline(feature_function = lambda x : x, degree = 3)
trialN = trialN.update(zip(x,y)) 
print(trialN.evaluate([-2.9]).item())
plt.plot(x, y, 'ro', ms=5)
spl = scipy.interpolate.UnivariateSpline(x, y, k = 3)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, trialN.evaluate(xs), 'g', lw=3, label = 'Nico')
plt.plot(xs, spl(xs), 'b', lw=3, label = 'Scipy')
plt.show()