from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.trial._frozen import FrozenTrial


def _filter_nonfinite(
    trials: List[FrozenTrial],
    target: Optional[Callable[[FrozenTrial], float]] = None,
    with_message: bool = True,
    distributions: Optional[Dict[str, BaseDistribution]] = None,
) -> List[FrozenTrial]:

    # For multi-objective optimization target must be specified to select
    # one of objective values to filter trials by (and plot by later on).
    # This function is not raising when target is missing, sice we're
    # assuming plot args have been sanitized before.
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    filtered_trials: List[FrozenTrial] = []
    for trial in trials:
        # Not a Number, positive infinity and negative infinity are considered to be non-finite.
        if not np.isfinite(target(trial)):
            continue
        elif distributions is not None and any(
            name not in trial.params for name in distributions.keys()
        ):
            continue
        else:
            filtered_trials.append(trial)

    return filtered_trials
