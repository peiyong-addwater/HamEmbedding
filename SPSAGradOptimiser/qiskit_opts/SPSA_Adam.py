# Based on https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/algorithms/optimizers/adam_amsgrad.py

from typing import Any, Optional, Callable, Dict, Tuple, List
import os

import csv
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import OptimizerSupportLevel, OptimizerResult
from .Optimiser import OptimizerSPSAGrad as Optimizer
import time

class ADAMSPSA(Optimizer):
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
        "c",
        "alpha",
        "gamma",
        "A",
        "a"
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
            snapshot_dir: str = None,
            c=0.2,
            alpha=0.602,
            gamma=0.101,
            A=None,
            a=None
    ) -> None:
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
            self._A = maxiter * 0.1
        if not a:
            self_a = 0.05 * (self._A + 1) ** self._alpha

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
            "c":self._c,
            "alpha":self._alpha,
            "gamma":self._gamma,
            "A":self._A,
            "a":self._a
        }

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def save_params(self, snapshot_dir: str) -> None:
        """Save the current iteration parameters to a file called ``adam_spsa_params.csv``.

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
        """Load iteration parameters for a file called ``adam_spsa_params.csv``.

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

    def minimize(
            self,
            objective_function: Callable[[np.ndarray], float],
            initial_point: np.ndarray,
            gradient_function: Callable[[np.ndarray, int], float],
            verbose: bool = True,
    ) -> Tuple[np.ndarray, float, int, list[float]]:
        start = time.time()
        self._loss_list = []
        self._t = 1
        #derivative = gradient_function(initial_point, self._t)
        #print(derivative)
        self._m = np.zeros(np.shape(initial_point))
        self._v = np.zeros(np.shape(initial_point))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(initial_point))
        params = params_new = initial_point
        while self._t <= self._maxiter:
            self._current_obj = objective_function(params)
            self._loss_list.append(self._current_obj)
            derivative = gradient_function(params, self._t)
            self._t+=1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = params - lr_eff * self._m.flatten() / (
                        np.sqrt(self._v.flatten()) + self._noise_factor
                )
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (np.sqrt(self._v_eff.flatten()) + self._noise_factor)

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)
            if verbose:
                current_time_lapse = time.time()-start
                epoch_time = current_time_lapse/(self._t-1)
                print(f"Training at {self._t-1}/{self._maxiter}, Objective = {np.round(self._current_obj, 4)}, Avg Iter Time = {np.round(epoch_time, 4)}, Total Time = {np.round(current_time_lapse, 4)}")
            if np.linalg.norm(params - params_new) < self._tol:
                return params_new, objective_function(params_new), self._t, self._loss_list
            else:
                params = params_new
        return params_new, objective_function(params_new), self._t, self._loss_list

    def optimize(
            self,
            num_vars:int,
            objective_function: Callable[[np.ndarray], float],
            gradient_function: Callable[[np.ndarray, int], float],
            variable_bounds: Optional[List[Tuple[float, float]]] = None,
            initial_point: Optional[np.ndarray] = None,
            verbose: bool = False,
    )->Tuple[np.ndarray, float, int, list[float]]:
        super().optimize(
            num_vars,objective_function, gradient_function, variable_bounds, initial_point
        )
        if initial_point is None:
            initial_point = algorithm_globals.random.random(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(
                Optimizer.gradient_spsa, (objective_function, self._c, self._alpha, self._gamma, self._A, self._a, self._maxiter)
            )

        point, value, nfev, losses = self.minimize(objective_function, initial_point, gradient_function, verbose)
        return point, value, nfev, losses




