"""
Scikit-learn metaoptimizers
=====================================
E.g.: for use with RandomizedSearchCV.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from neuraxle.base import MetaStepMixin, BaseStep


class MetaSKLearnWrapper(MetaStepMixin, BaseStep):
    """
    Wrap a Sklearn meta estimator with a Neuraxle base step.
    """

    def __init__(self, wrapped: 'MetaEstimatorMixin'):
        self.wrapped_sklearn_metaestimator = wrapped
        # sklearn.model_selection.RandomizedSearchCV
        super().__init__()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        """
        Fit the wrapped SkLearn meta estimator with the received data inputs and expected outputs

        :param data_inputs:
        :param expected_outputs:
        :return: fitted self
        """
        self.wrapped_sklearn_metaestimator = self.wrapped_sklearn_metaestimator.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        """
        Transform the data inputs with the wrapped SKLearn meta estimator `transform` method.

        :param data_inputs:
        :return: processed outputs
        """
        return self.wrapped_sklearn_metaestimator.transform(data_inputs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_ = self.__class__
        module = type_.__module__
        qualname = type_.__qualname__
        wrappedname = str(self.wrapped_sklearn_metaestimator.__class__.__name__)
        return "<{}.{}({}(...)) object {}>".format(module, qualname, wrappedname, hex(id(self)))
