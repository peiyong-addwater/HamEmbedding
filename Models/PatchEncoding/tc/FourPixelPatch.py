from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

import tensorcircuit as tc

Circuit = Any  # we don't use the real circuit class as too many mypy complains emerge
Tensor = Any
Graph = Any

