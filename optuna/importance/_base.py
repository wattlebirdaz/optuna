import abc
from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import intersection_search_space
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.trial._filter_nonfinite import _filter_nonfinite



def _check_evaluate_args(completed_trials: List[FrozenTrial], params: Optional[List[str]]) -> None:
    if len(completed_trials) == 0:
        raise ValueError("Cannot evaluate parameter importances without completed trials.")
    if len(completed_trials) == 1:
        raise ValueError("Cannot evaluate parameter importances with only a single trial.")

    if params is not None:
        if not isinstance(params, (list, tuple)):
            raise TypeError(
                "Parameters must be specified as a list. Actual parameters: {}.".format(params)
            )
        if any(not isinstance(p, str) for p in params):
            raise TypeError(
                "Parameters must be specified by their names with strings. Actual parameters: "
                "{}.".format(params)
            )

        if len(params) > 0:
            at_least_one_trial = False
            for trial in completed_trials:
                if all(p in trial.distributions for p in params):
                    at_least_one_trial = True
                    break
            if not at_least_one_trial:
                raise ValueError(
                    "Study must contain completed trials with all specified parameters. "
                    "Specified parameters: {}.".format(params)
                )


def _get_distributions(study: Study, params: Optional[List[str]]) -> Dict[str, BaseDistribution]:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    _check_evaluate_args(completed_trials, params)

    if params is None:
        return intersection_search_space(study, ordered_dict=True)

    # New temporary required to pass mypy. Seems like a bug.
    params_not_none = params
    assert params_not_none is not None

    # Compute the search space based on the subset of trials containing all parameters.
    distributions = None
    for trial in completed_trials:
        trial_distributions = trial.distributions
        if not all(name in trial_distributions for name in params_not_none):
            continue

        if distributions is None:
            distributions = dict(
                filter(
                    lambda name_and_distribution: name_and_distribution[0] in params_not_none,
                    trial_distributions.items(),
                )
            )
            continue

        if any(
            trial_distributions[name] != distribution
            for name, distribution in distributions.items()
        ):
            raise ValueError(
                "Parameters importances cannot be assessed with dynamic search spaces if "
                "parameters are specified. Specified parameters: {}.".format(params)
            )

    assert distributions is not None  # Required to pass mypy.
    distributions = OrderedDict(
        sorted(distributions.items(), key=lambda name_and_distribution: name_and_distribution[0])
    )
    return distributions

def _get_trans_params_values(
    trans: _SearchSpaceTransform,
    trials: Sequence[FrozenTrial],
    target: Optional[Callable[[FrozenTrial], float]],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    n_trials = len(trials)

    trans_params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
    trans_values = numpy.empty(n_trials, dtype=numpy.float64)

    for trial_idx, trial in enumerate(trials):
        trans_params[trial_idx] = trans.transform(trial.params)
        trans_values[trial_idx] = trial.value if target is None else target(trial)

    return trans_params, trans_values


class BaseImportanceEvaluator(object, metaclass=abc.ABCMeta):
    """Abstract parameter importance evaluator."""

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )
        distributions = _get_distributions(study, params)
        if len(distributions) == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        # A parameter whose domain is a single value has undefined importance.
        # We return zero for such parameters.
        single_distributions = {
            name: dist for name, dist in distributions.items() if dist.single()
        }
        non_single_distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }

        trials = _filter_nonfinite(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
            target=target,
            distributions=non_single_distributions,
        )
        trans = _SearchSpaceTransform(
            non_single_distributions, transform_log=False, transform_step=False
        )
        trans_params, values = _get_trans_params_values(trans, trials, target)

        non_single_importance_values = self.evaluate_core(trans_params, values, trans)
        non_single_importances = {
            name: value
            for name, value in zip(non_single_distributions.keys(), non_single_importance_values)
        }
        single_importances = {name: 0.0 for name in single_distributions.keys()}
        importances = {**non_single_importances, **single_importances}
        sorted_importances = OrderedDict(
            reversed(
                sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
            )
        )
        return sorted_importances

    @abc.abstractmethod
    def evaluate_core(
        self, trans_params: numpy.ndarray, values: numpy.ndarray, trans: _SearchSpaceTransform
    ) -> numpy.ndarray:
        """Evaluate parameter importances based on completed trials in the given study.

        .. note::

            This method is not meant to be called by library users.

        .. seealso::

            Please refer to :func:`~optuna.importance.get_param_importances` for how a concrete
            evaluator should implement this method.

        Args:
            trans_params:
                A ``numpy.ndarray`` of shape ``(T, F)`` containing transformed parameter values
                (feature values) for each `trial`, where ``T`` is the number of trials and ``F``
                is the number of feature values.

                .. note::
                    For `CategoricalDistribution`, the transformed parameter value
                    is the one-hot vector describing the categorical value. For other
                    distributions, the transformed parameter value is the value itself.
            values:
                A :class:`numpy.ndarray` of shape ``(T,)`` containing objective values for each
                `trial`, where ``T`` is the number of trials.
            trans:
                A :class:`_SearchSpaceTransform` describing how the original parameters are
                transformed into feature values.
        Returns:
            A :class:`numpy.ndarray` of shape ``(P,)`` containing importance values for each
            parameter, where ``P`` is the number of parameters.
        """
        raise NotImplementedError
