from typing import Optional

import numpy

from optuna._transform import _SearchSpaceTransform
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova


class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA importance evaluator.

    Implements the fANOVA hyperparameter importance evaluation algorithm in
    `An Efficient Approach for Assessing Hyperparameter Importance
    <http://proceedings.mlr.press/v32/hutter14.html>`_.

    Given a study, fANOVA fits a random forest regression model that predicts the objective value
    given a parameter configuration. The more accurate this model is, the more reliable the
    importances assessed by this class are.

    .. note::

        Requires the `sklearn <https://github.com/scikit-learn/scikit-learn>`_ Python package.

    .. note::

        Pairwise and higher order importances are not supported through this class. They can be
        computed using :class:`~optuna.importance._fanova._fanova._Fanova` directly but is not
        recommended as interfaces may change without prior notice.

    .. note::

        The performance of fANOVA depends on the prediction performance of the underlying
        random forest model. In order to obtain high prediction performance, it is necessary to
        cover a wide range of the hyperparameter search space. It is recommended to use an
        exploration-oriented sampler such as :class:`~optuna.samplers.RandomSampler`.

    .. note::

        For how to cite the original work, please refer to
        https://automl.github.io/fanova/cite.html.

    Args:
        n_trees:
            The number of trees in the forest.
        max_depth:
            The maximum depth of the trees in the forest.
        seed:
            Controls the randomness of the forest. For deterministic behavior, specify a value
            other than :obj:`None`.

    """

    def __init__(
        self, *, n_trees: int = 64, max_depth: int = 64, seed: Optional[int] = None
    ) -> None:
        self._evaluator = _Fanova(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=seed,
        )

    def evaluate(
        self, features: numpy.ndarray, values: numpy.ndarray, trans: _SearchSpaceTransform
    ) -> numpy.ndarray:
        evaluator = self._evaluator
        evaluator.fit(
            X=features,
            y=values,
            search_spaces=trans.bounds,
            column_to_encoded_columns=trans.column_to_encoded_columns,
        )

        param_importances = numpy.array(
            [evaluator.get_importance((i,))[0] for i in range(trans.num_params)]
        )

        total_importance = numpy.sum(param_importances)
        param_importances /= total_importance
        return param_importances
