# Re-implement qiskit.algorithm.optimizer.Optimizer to use SPSA instead numerical gradient
# Based on https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/algorithms/optimizers/optimizer.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
import logging
from typing import Any, Union, Protocol
from typing import Any, Optional, Callable, Dict, Tuple, List
import numpy as np
import scipy


from qiskit.algorithms.optimizers import OptimizerSupportLevel, OptimizerResult

logger = logging.getLogger(__name__)

POINT = Union[float, np.ndarray]

class OptimizerSPSAGrad(ABC):
    """Base class for optimization algorithm."""

    @abstractmethod
    def __init__(self):
        """
        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.
        """
        self._gradient_support_level = self.get_support_level()["gradient"]
        self._bounds_support_level = self.get_support_level()["bounds"]
        self._initial_point_support_level = self.get_support_level()["initial_point"]
        self._options = {}
        self._max_evals_grouped = None

    @abstractmethod
    def get_support_level(self):
        """Return support level dictionary"""
        raise NotImplementedError

    def set_options(self, **kwargs):
        """
        Sets or updates values in the options dictionary.

        The options dictionary may be used internally by a given optimizer to
        pass additional optional values for the underlying optimizer/optimization
        function used. The options dictionary may be initially populated with
        a set of key/values when the given optimizer is constructed.

        Args:
            kwargs (dict): options, given as name=value.
        """
        for name, value in kwargs.items():
            self._options[name] = value
        logger.debug("options: %s", self._options)


    @staticmethod
    def gradient_spsa(x_center,k, f, c=0.2, alpha=0.602, gamma=0.101, A=None, a=None, maxiter=None):
        if not maxiter and not A:
            raise TypeError("One of the parameters maxiter or A must be provided.")
        if not A:
            A = maxiter*0.1
        if not a:
            a = 0.05*(A+1)**alpha
        ck = c/k**gamma
        delta = np.random.choice([-1, 1], size=x_center.shape)
        #print(delta)
        multiplier = ck * delta
        #print(multiplier)
        thetaplus = np.array(x_center)+multiplier
        thetaminus = np.array(x_center)-multiplier
        #print(thetaplus)
        #print(thetaminus)
        # the output of the cost function should be a scalar
        yplus = f(thetaplus)
        yminus = f(thetaminus)

        grad = [(yplus - yminus) / (2 * ck * di) for di in delta]

        return np.array(grad)

    @abstractmethod
    def optimize(
            self,
            num_vars,
            objective_function,
            gradient_function=None,
            variable_bounds=None,
            initial_point=None,
            verbose: bool = False,
    ):
        if initial_point is not None and len(initial_point) != num_vars:
            raise ValueError("Initial point does not match dimension")
        if variable_bounds is not None and len(variable_bounds) != num_vars:
            raise ValueError("Variable bounds not match dimension")

        has_bounds = False
        if variable_bounds is not None:
            # If *any* value is *equal* in bounds array to None then the does *not* have bounds
            has_bounds = not np.any(np.equal(variable_bounds, None))

        if gradient_function is None and self.is_gradient_required:
            raise ValueError("Gradient is required but None given")
        if not has_bounds and self.is_bounds_required:
            raise ValueError("Variable bounds is required but None given")
        if initial_point is None and self.is_initial_point_required:
            raise ValueError("Initial point is required but None given")

        if gradient_function is not None and self.is_gradient_ignored:
            logger.debug(
                "WARNING: %s does not support gradient function. It will be ignored.",
                self.__class__.__name__,
            )
        if has_bounds and self.is_bounds_ignored:
            logger.debug(
                "WARNING: %s does not support bounds. It will be ignored.", self.__class__.__name__
            )
        if initial_point is not None and self.is_initial_point_ignored:
            logger.debug(
                "WARNING: %s does not support initial point. It will be ignored.",
                self.__class__.__name__,
            )
        pass

    @staticmethod
    def wrap_function(function, args):
        """
        Wrap the function to implicitly inject the args at the call of the function.

        Args:
            function (func): the target function
            args (tuple): the args to be injected
        Returns:
            function_wrapper: wrapper
        """

        def function_wrapper(*wrapper_args):
            return function(*(wrapper_args + args))

        return function_wrapper

    @property
    def setting(self):
        """Return setting"""
        ret = f"Optimizer: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    @property
    def settings(self) -> dict[str, Any]:
        """The optimizer settings in a dictionary format.

        The settings can for instance be used for JSON-serialization (if all settings are
        serializable, which e.g. doesn't hold per default for callables), such that the
        optimizer object can be reconstructed as

        .. code-block::

            settings = optimizer.settings
            # JSON serialize and send to another server
            optimizer = OptimizerClass(**settings)

        """
        raise NotImplementedError("The settings method is not implemented per default.")

    @abstractmethod
    def minimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        gradient_function: Callable[[np.ndarray, int], float],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, int]:

        raise NotImplementedError()

    @property
    def gradient_support_level(self):
        """Returns gradient support level"""
        return self._gradient_support_level

    @property
    def is_gradient_ignored(self):
        """Returns is gradient ignored"""
        return self._gradient_support_level == OptimizerSupportLevel.ignored

    @property
    def is_gradient_supported(self):
        """Returns is gradient supported"""
        return self._gradient_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_gradient_required(self):
        """Returns is gradient required"""
        return self._gradient_support_level == OptimizerSupportLevel.required

    @property
    def bounds_support_level(self):
        """Returns bounds support level"""
        return self._bounds_support_level

    @property
    def is_bounds_ignored(self):
        """Returns is bounds ignored"""
        return self._bounds_support_level == OptimizerSupportLevel.ignored

    @property
    def is_bounds_supported(self):
        """Returns is bounds supported"""
        return self._bounds_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_bounds_required(self):
        """Returns is bounds required"""
        return self._bounds_support_level == OptimizerSupportLevel.required

    @property
    def initial_point_support_level(self):
        """Returns initial point support level"""
        return self._initial_point_support_level

    @property
    def is_initial_point_ignored(self):
        """Returns is initial point ignored"""
        return self._initial_point_support_level == OptimizerSupportLevel.ignored

    @property
    def is_initial_point_supported(self):
        """Returns is initial point supported"""
        return self._initial_point_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_initial_point_required(self):
        """Returns is initial point required"""
        return self._initial_point_support_level == OptimizerSupportLevel.required

    def print_options(self):
        """Print algorithm-specific options."""
        for name in sorted(self._options):
            logger.debug("%s = %s", name, str(self._options[name]))

    def set_max_evals_grouped(self, limit):
        """Set max evals grouped"""
        self._max_evals_grouped = limit