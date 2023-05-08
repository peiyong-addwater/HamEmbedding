# Based on https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/algorithms/optimizers/adam_amsgrad.py

from typing import Any, Optional, Callable, Dict, Tuple, List
import os

import csv
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizer import OptimizerSupportLevel, OptimizerResult
from Optimiser import OptimizerSPSAGrad as Optimizer

class ADAM(Optimizer):
    """Adam and AMSGRAD optimizers, with SPSA gradient."""


    _OPTIONS = [
        "maxiter",
        "tol",
        "lr",
        "beta_1",
        "beta_2",
        "noise_factor",
        "eps",
        "amsgrad",
        "snapshot_dir",
    ]

    def __init__(
        self,
        maxiter: int = 10000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
        snapshot_dir: str | None = None,
        c=0.2,
        alpha=0.602,
        gamma=0.101,
        A=None,
        a=None
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations
            tol: Tolerance for termination
            lr: Value >= 0, Learning rate.
            beta_1: Value in range 0 to 1, Generally close to 1.
            beta_2: Value in range 0 to 1, Generally close to 1.
            noise_factor: Value >= 0, Noise factor
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
            amsgrad: True to use AMSGRAD, False if not
            snapshot_dir: If not None save the optimizer's parameter
                after every step to the given directory
        """
        super().__init__()
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                self._options[k] = v
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

        # SPSA parameters
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._A = A
        self._a = a
        if not A:
            self._A = maxiter*0.1
        if not a:
            self_a = 0.05*(self._A+1)**self._alpha


        # runtime variables
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:

            with open(os.path.join(self._snapshot_dir, "adam_params.csv"), mode="w") as csv_file:
                if self._amsgrad:
                    fieldnames = ["v", "v_eff", "m", "t"]
                else:
                    fieldnames = ["v", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "maxiter": self._maxiter,
            "tol": self._tol,
            "lr": self._lr,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "noise_factor": self._noise_factor,
            "eps": self._eps,
            "amsgrad": self._amsgrad,
            "snapshot_dir": self._snapshot_dir,
        }

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def save_params(self, snapshot_dir: str) -> None:
        """Save the current iteration parameters to a file called ``adam_params.csv``.

        Note:

            The current parameters are appended to the file, if it exists already.
            The file is not overwritten.

        Args:
            snapshot_dir: The directory to store the file in.
        """
        if self._amsgrad:
            with open(os.path.join(snapshot_dir, "adam_spsa_params.csv"), mode="a") as csv_file:
                fieldnames = ["v", "v_eff", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({"v": self._v, "v_eff": self._v_eff, "m": self._m, "t": self._t})
        else:
            with open(os.path.join(snapshot_dir, "adam_spsa_params.csv"), mode="a") as csv_file:
                fieldnames = ["v", "m", "t"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({"v": self._v, "m": self._m, "t": self._t})

    def load_params(self, load_dir: str) -> None:
        """Load iteration parameters for a file called ``adam_params.csv``.

        Args:
            load_dir: The directory containing ``adam_params.csv``.
        """
        with open(os.path.join(load_dir, "adam_spsa_params.csv")) as csv_file:
            if self._amsgrad:
                fieldnames = ["v", "v_eff", "m", "t"]
            else:
                fieldnames = ["v", "m", "t"]
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line["v"]
                if self._amsgrad:
                    v_eff = line["v_eff"]
                m = line["m"]
                t = line["t"]

        v = v[1:-1]
        self._v = np.fromstring(v, dtype=float, sep=" ")
        if self._amsgrad:
            v_eff = v_eff[1:-1]
            self._v_eff = np.fromstring(v_eff, dtype=float, sep=" ")
        m = m[1:-1]
        self._m = np.fromstring(m, dtype=float, sep=" ")
        t = t[1:-1]
        self._t = np.fromstring(t, dtype=int, sep=" ")

    @deprecate_arg("objective_function", new_alias="fun", since="0.19.0")
    @deprecate_arg("initial_point", new_alias="fun", since="0.19.0")
    @deprecate_arg("gradient_function", new_alias="jac", since="0.19.0")
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        # pylint:disable=unused-argument
        objective_function: Callable[[np.ndarray], float] | None = None,
        initial_point: np.ndarray | None = None,
        gradient_function: Callable[[np.ndarray], float] | None = None,
        # ) -> Tuple[np.ndarray, float, int]:
    ) -> OptimizerResult:  # TODO find proper way to deprecate return type
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.
            objective_function: DEPRECATED. A function handle to the objective function.
            initial_point: DEPRECATED. The initial iteration point.
            gradient_function: DEPRECATED. A function handle to the gradient of the objective
                function.

        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        if jac is None:
            jac = Optimizer.wrap_function(Optimizer.gradient_spsa, (fun, self._eps))

        derivative = jac(x0)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = params_new = x0
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2**self._t) / (1 - self._beta_1**self._t)
            if not self._amsgrad:
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v.flatten()) + self._noise_factor
                )
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)

            # check termination
            if np.linalg.norm(params - params_new) < self._tol:
                break

            params = params_new

        result = OptimizerResult()
        result.x = params_new
        result.fun = fun(params_new)
        result.nfev = self._t
        return result