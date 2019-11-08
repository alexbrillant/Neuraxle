"""
Random
====================================
Meta steps for hyperparameter tuning, such as random search.

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

import copy
import math
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

import numpy as np
from sklearn.metrics import r2_score

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.steps.loop import StepClonerForEachDataInput
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch, NumpyConcatenateOnCustomAxis


class BaseValidation(MetaStepMixin, BaseStep, ABC):
    """
    Base class For validation wrappers.
    It has a scoring function to calculate the score for the validation split.

    .. seealso::
        :class`ValidationSplitWrapper`,
        :class`ValidationSplitWrapper`,
        :class`KFoldCrossValidationWrapper`,
        :class`AnchoredWalkForwardTimeSeriesCrossValidationWrapper`,
        :class`WalkForwardTimeSeriesCrossValidationWrapper`

    """

    def __init__(self, scoring_function: Callable = r2_score):
        """
        Base class For validation wrappers.
        It has a scoring function to calculate the score for the validation split.

        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        :type scoring_function: Callable
        """
        MetaStepMixin.__init__(self)
        BaseStep.__init__(self)
        self.scoring_function = scoring_function


class ValidationSplitWrapper(BaseValidation):
    """
    Wrapper for validation split that calculates the score for the validation split.

    .. code-block:: python

        random_search = Pipeline([
            RandomSearch(
                ValidationSplitWrapper(
                    Identity(),
                    test_size=0.1
                )
            )
        ])

    .. seealso::
        :class`BaseValidation`
        :class`BaseCrossValidationWrapper`
        :class`RandomSearch`

    """

    def __init__(
            self,
            wrapped: BaseStep,
            test_size: float,
            scoring_function=r2_score
    ):
        """
        :param wrapped: wrapped step
        :param test_size: ratio for test size between 0 and 1
        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        """
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.test_size = test_size
        self.scoring_function = scoring_function

    def transform(self, data_inputs):
        """
        Transform given data inputs without splitting.

        :param data_inputs: data inputs
        :return: outputs
        """
        return self.wrapped.transform(data_inputs)

    def fit(self, data_inputs, expected_outputs=None) -> 'ValidationSplitWrapper':
        """
        Fit using the training split.
        Calculate the scores using the validation split.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs
        :return: fitted self
        """
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_inputs, expected_outputs)

        self.wrapped = self.wrapped.fit(train_data_inputs, train_expected_outputs)

        results = self.wrapped.transform(validation_data_inputs)
        self.scores = [self.scoring_function(a, b) for a, b in zip(results, validation_expected_outputs)]
        self.scores_mean = np.mean(self.scores)
        self.scores_std = np.std(self.scores)

        return self

    def split(self, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
        """
        Split data inputs, and expected outputs into a training set, and a validation set.

        :param data_inputs: data inputs to split
        :param expected_outputs: expected outputs to split
        :return: train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs
        """
        validation_data_inputs, validation_expected_outputs = self.validation_split(data_inputs, expected_outputs)
        train_data_inputs, train_expected_outputs = self.train_split(data_inputs, expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, data_inputs, expected_outputs) -> (List, List):
        """
        Split training set.

        :param data_inputs: data inputs to split
        :param expected_outputs: expected outputs to split
        :return: train_data_inputs, train_expected_outputs
        """
        index_split = math.floor(len(data_inputs) * (1 - self.test_size))
        train_data_inputs = data_inputs[0:index_split]

        train_expected_outputs = None
        if expected_outputs is not None:
            train_expected_outputs = expected_outputs[0:index_split]

        return train_data_inputs, train_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> (List, List):
        """
        Split validation set.

        :param data_inputs: data inputs to split
        :param expected_outputs: expected outputs to split
        :return: validation_data_inputs, validation_expected_outputs, validation_data_inputs, validation_expected_outputs
        """
        index_split = math.floor(len(data_inputs) * self.test_size)
        validation_data_inputs = data_inputs[:index_split]
        validation_expected_outputs = None
        if expected_outputs is not None:
            validation_expected_outputs = expected_outputs[:index_split]

        return validation_data_inputs, validation_expected_outputs


class BaseCrossValidationWrapper(BaseValidation, ABC):
    # TODO: assert that set_step was called.
    # TODO: change default argument of scoring_function...
    def __init__(self, scoring_function=r2_score, joiner=NumpyConcatenateOuterBatch()):
        BaseValidation.__init__(self, scoring_function)
        self.joiner = joiner

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCrossValidationWrapper':
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_inputs, expected_outputs)

        step = StepClonerForEachDataInput(self.wrapped)
        step = step.fit(train_data_inputs, train_expected_outputs)

        results = step.transform(validation_data_inputs)
        self.scores = [self.scoring_function(a, b) for a, b in zip(results, validation_expected_outputs)]
        self.scores_mean = np.mean(self.scores)
        self.scores_std = np.std(self.scores)

        return self

    @abstractmethod
    def split(self, data_inputs, expected_outputs):
        raise NotImplementedError("TODO")

    def transform(self, data_inputs):
        # TODO: use the splits and average the results?? instead of picking best model...
        raise NotImplementedError("TODO: code this method in Neuraxle.")
        data_inputs = self.split(data_inputs)
        predicted_outputs_splitted = self.wrapped.transform(data_inputs)
        return self.joiner.transform(predicted_outputs_splitted)


class KFoldCrossValidationWrapper(BaseCrossValidationWrapper):

    def __init__(self, scoring_function=r2_score, k_fold=3, joiner=NumpyConcatenateOuterBatch()):
        self.k_fold = k_fold
        BaseCrossValidationWrapper.__init__(self, scoring_function=scoring_function, joiner=joiner)

    def split(self, data_inputs, expected_outputs):
        validation_data_inputs, validation_expected_outputs = self.validation_split(
            data_inputs, expected_outputs)

        train_data_inputs, train_expected_outputs = self.train_split(
            validation_data_inputs, validation_expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, validation_data_inputs, validation_expected_outputs) -> (List, List):
        train_data_inputs = []
        train_expected_outputs = []
        for i in range(len(validation_data_inputs)):
            inputs = validation_data_inputs[:i] + validation_data_inputs[i + 1:]
            outputs = validation_expected_outputs[:i] + validation_expected_outputs[i + 1:]

            inputs = self.joiner.transform(inputs)
            outputs = self.joiner.transform(outputs)

            train_data_inputs.append(inputs)
            train_expected_outputs.append(outputs)

        return train_data_inputs, train_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> List:
        splitted_data_inputs = self._split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs
        return splitted_data_inputs

    def _split(self, data_inputs):
        splitted_data_inputs = []
        step = len(data_inputs) / float(self.k_fold)
        for i in range(self.k_fold):
            a = int(step * i)
            b = int(step * (i + 1))
            if b > len(data_inputs):
                b = len(data_inputs)

            slice = data_inputs[a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs


class AnchoredWalkForwardTimeSeriesCrossValidationWrapper(BaseCrossValidationWrapper):
    """
    Prform an anchored walk forward cross validation by performing a forward rolling split.
    All training splits start at the beginning of the time series, but finish at different time. The finish time
    increase toward the end at each forward split.

    For the validation split it will start after a certain time delay (if padding is set)
    after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(self, minimum_training_size, validation_window_size=None, padding_between_training_and_validation=0,
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateOnCustomAxis(axis=1)):
        """
        Create a anchored walk forward time series cross validation object.

        The size of the validation split is defined by `validation_window_size`.
        The difference in start position between two consecutive validation split is also equal to
        `validation_window_size`.

        :param minimum_training_size: size of the smallest training split.
        :param validation_window_size: size of each validation split and also the time step taken between each
            forward roll, by default None. If None : It takes the value `minimum_training_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        :param scoring_function: scoring function use to validate performance if it is not None, by default r2_score,
        :param joiner the joiner callable that can join the different result together.
        :return: WalkForwardTimeSeriesCrossValidation instance.
        """
        BaseCrossValidationWrapper.__init__(self, scoring_function=scoring_function, joiner=joiner)
        self.minimum_training_size = minimum_training_size
        # If validation_window_size is None, we give the same value as training_window_size.
        self.validation_window_size = validation_window_size or self.minimum_training_size
        self.padding_between_training_and_validation = padding_between_training_and_validation
        self.drop_remainder = drop_remainder
        self._validation_initial_start = self.minimum_training_size + self.padding_between_training_and_validation

    def split(self, data_inputs, expected_outputs):
        """
        Split the data into train inputs, train expected outputs, validation inputs, validation expected outputs.

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target/label that will be used during walk forward cross validation.
        :return: train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs
        """
        validation_data_inputs, validation_expected_outputs = self.validation_split(
            data_inputs, expected_outputs)

        train_data_inputs, train_expected_outputs = self.train_split(
            data_inputs, expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, data_inputs, expected_outputs=None) -> (List, List):
        """
        Split the data into train inputs, train expected outputs

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target*label that will be used during walk forward cross validation.
        :return: train_data_inputs, train_expected_outputs
        """
        splitted_data_inputs = self._train_split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._train_split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> List:
        """
        Split the data into validation inputs, validation expected outputs.

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target*label that will be used during walk forward cross validation.
        :return: validation_data_inputs, validation_expected_outputs
        """
        splitted_data_inputs = self._validation_split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._validation_split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs
        return splitted_data_inputs

    def _train_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)

        for i in range(number_step):
            # first slice index is always 0 for anchored walk forward cross validation.
            a = 0
            b = int(self.minimum_training_size + i * self.validation_window_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs

    def _validation_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)
        for i in range(number_step):
            a = int(self._validation_initial_start + i * self.validation_window_size)
            b = int(a + self.validation_window_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs

    def _get_number_fold(self, data_inputs):
        if self.drop_remainder:
            number_step = math.floor(
                (data_inputs.shape[1] - self._validation_initial_start) / float(self.validation_window_size)
            )
        else:
            number_step = math.ceil(
                (data_inputs.shape[1] - self._validation_initial_start) / float(self.validation_window_size)
            )
        return number_step


class WalkForwardTimeSeriesCrossValidationWrapper(AnchoredWalkForwardTimeSeriesCrossValidationWrapper):
    """
    Perform a classic walk forward cross validation by performing a forward rolling split.

    All the training split have the same `validation_window_size` size. The start time and end time of each training
    split will increase identically toward the end at each forward split. Same principle apply with the validation
    split, where the start and end will increase in the same manner toward the end. Each validation split start after
    a certain time delay (if padding is set) after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(self, training_window_size, validation_window_size=None, padding_between_training_and_validation=0,
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateOnCustomAxis(axis=1)):
        """
        Create a classic walk forward time series cross validation object.

        The difference in start position between two consecutive validation split are equal to one
        `validation_window_size`.

        :param training_window_size: the window size of training split.
        :param validation_window_size: the window size of each validation split and also the time step taken between
            each forward roll, by default None. If None : It takes the value `training_window_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        :param scoring_function: scoring function use to validate performance if it is not None, by default r2_score,
        :param joiner the joiner callable that can join the different result together.
        :return: WalkForwardTimeSeriesCrossValidation instance.
        """
        AnchoredWalkForwardTimeSeriesCrossValidationWrapper.__init__(
            self,
            training_window_size,
            validation_window_size=validation_window_size,
            padding_between_training_and_validation=padding_between_training_and_validation,
            drop_remainder=drop_remainder, scoring_function=scoring_function, joiner=joiner
        )

    def _train_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)

        for i in range(number_step):
            a = int(i * self.validation_window_size)
            # Here minimum_training_size = training_size, since each training split has the same length.
            b = int(a + self.minimum_training_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs


class RandomSearch(MetaStepMixin, BaseStep):
    """Perform a random hyperparameter search."""

    # TODO: CV and rename to RandomSearchCV.

    def __init__(
            self,
            wrapped=None,
            n_iter: int = 10,
            higher_score_is_better: bool = True,
            validation_technique: BaseCrossValidationWrapper = KFoldCrossValidationWrapper(),
            refit=True,
    ):
        if wrapped is not None:
            MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.n_iter = n_iter
        self.higher_score_is_better = higher_score_is_better
        self.validation_technique: BaseCrossValidationWrapper = validation_technique
        self.refit = refit

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        started = False
        best_hyperparams = None

        for _ in range(self.n_iter):

            step = copy.copy(self.wrapped)

            new_hyperparams = step.get_hyperparams_space().rvs()
            step.set_hyperparams(new_hyperparams)

            step: BaseCrossValidationWrapper = copy.copy(self.validation_technique).set_step(step)

            step = step.fit(data_inputs, expected_outputs)
            score = step.scores_mean

            if not started or self.higher_score_is_better == (score > self.score):
                started = True
                self.score = score
                self.best_validation_wrapper_of_model = copy.copy(step)
                best_hyperparams = new_hyperparams

        self.best_validation_wrapper_of_model.wrapped.set_hyperparams(best_hyperparams)

        if self.refit:
            self.best_model = self.best_validation_wrapper_of_model.wrapped.fit(
                data_inputs,
                expected_outputs
            )

        return self

    def get_best_model(self):
        return self.best_model

    def transform(self, data_inputs):
        if self.best_validation_wrapper_of_model is None:
            raise Exception('Cannot transform RandomSearch before fit')
        return self.best_validation_wrapper_of_model.wrapped.transform(data_inputs)
