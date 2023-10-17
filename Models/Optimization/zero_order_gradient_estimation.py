from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from qiskit_algorithms.gradients.base.base_sampler_gradient import BaseSamplerGradient
from qiskit_algorithms.gradients.base.sampler_gradient_result import SamplerGradientResult

from qiskit_algorithms.exceptions import AlgorithmError

class RSGFSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the RSGF method mentioned in Eqn. (5) of [1]:

    g(theta) = u*(f(theta+c*u)-f(theta))/c

    [1] Z. Leng, P. Mundada, S. Ghadimi, and A. Houck, “Efficient Algorithms for High-Dimensional Quantum Optimal
    Control of a Transmon Qubit,” Phys. Rev. Appl., vol. 19, no. 4, p. 044034,
    Apr. 2023, doi: 10.1103/PhysRevApplied.19.044034.
    """

    def __init__(
        self,
        sampler: BaseSampler,
        c: float,
        batch_size: int = 1,
        seed: int | None = None,
        options: Options | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            c: The offset size for the RSGF gradients.
            batch_size: number of gradients to average.
            seed: The seed for a random Gaussian vector.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if c <= 0:
            raise ValueError(f"epsilon ({c}) should be positive.")
        self._batch_size = batch_size
        self._epsilon = c
        self._seed = np.random.default_rng(seed)

        super().__init__(sampler, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        job_circuits, job_param_values, metadata, offsets = [], [], [], []
        all_n = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # Indices of parameters to be differentiated.
            indices = [circuit.parameters.data.index(p) for p in parameters_]

            metadata.append({"parameters": parameters_})
            mean = np.zeros(len(circuit.parameters))
            cov = np.eye(len(circuit.parameters))

            offset = np.array(
                [
                    self._seed.multivariate_normal(mean, cov)
                    for _ in range(self._batch_size)
                ]
            )

            plus = [parameter_values_ + self._epsilon * offset_ for offset_ in offset]
            minus = [parameter_values_  for _ in offset]
            offsets.append(offset)

            # Combine inputs into a single job to reduce overhead.
            n = 2 * self._batch_size
            job_circuits.extend([circuit] * n)
            job_param_values.extend(plus + minus)
            all_n.append(n)

        # Run the single job with all circuits.
        job = self._sampler.run(job_circuits, job_param_values, **options)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for i, n in enumerate(all_n):
            dist_diffs = {}
            result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
            for j, (dist_plus, dist_minus) in enumerate(zip(result[: n // 2], result[n // 2 :])):
                dist_diff: dict[int, float] = defaultdict(float)
                for key, value in dist_plus.items():
                    dist_diff[key] += value / (2 * self._epsilon)
                for key, value in dist_minus.items():
                    dist_diff[key] -= value / (2 * self._epsilon)
                dist_diffs[j] = dist_diff
            gradient = []
            indices = [circuits[i].parameters.data.index(p) for p in metadata[i]["parameters"]]
            for j in indices:
                gradient_j: dict[int, float] = defaultdict(float)
                for k in range(self._batch_size):
                    for key, value in dist_diffs[k].items():
                        gradient_j[key] += value * offsets[i][k][j]
                gradient_j = {key: value / self._batch_size for key, value in gradient_j.items()}
                gradient.append(gradient_j)
            gradients.append(gradient)
            partial_sum_n += n

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)