"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

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
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Tuple

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin, DataContainer, Context
from neuraxle.checkpoints import BaseCheckpointStep


class BasePipeline(TruncableSteps, ABC):
    def __init__(self, steps: NamedTupleList):
        TruncableSteps.__init__(self, steps_as_tuple=steps)

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipeline':
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BasePipeline', Any):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform_processed_outputs(self, data_inputs) -> Any:
        raise NotImplementedError()

    def inverse_transform(self, processed_outputs):
        if self.transform != self.inverse_transform:
            raise BrokenPipeError("Don't call inverse_transform on a pipeline before having mutated it inversely or "
                                  "before having called the `.reverse()` or `reversed(.)` on it.")

        return self.inverse_transform_processed_outputs(processed_outputs)


class Pipeline(BasePipeline):
    """
    Fits and transform steps
    """

    def __init__(self, steps: NamedTupleList):
        BasePipeline.__init__(self, steps=steps)

    def transform(self, data_inputs: Any):
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            step_source_code='',
            data_inputs=data_inputs
        )

        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs
        )

        context = Context(
            current_path=self.name,
            parent_path_stack=[self.name],
            parent_step_stack=[self]
        )

        data_container = self._transform_core(data_container, context)

        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            step_source_code='',
            data_inputs=data_inputs
        )

        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )

        context = Context(
            current_path=self.name,
            parent_path_stack=[self.name],
            parent_step_stack=[self]
        )

        new_self, data_container = self._fit_transform_core(data_container, context)

        return new_self, data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            step_source_code='',
            data_inputs=data_inputs
        )

        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )

        context = Context(
            current_path=self.name,
            parent_path_stack=[self.name],
            parent_step_stack=[self]
        )

        new_self, _ = self._fit_transform_core(data_container, context)

        return new_self

    def inverse_transform_processed_outputs(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs

        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        for step_name, step in list(reversed(self.items())):
            processed_outputs = step.transform(processed_outputs)
        return processed_outputs

    def handle_fit_transform(self, data_container: DataContainer, context: Context) -> ('BaseStep', DataContainer):
        """
        Fit transform then rehash ids with hyperparams and transformed data inputs

        :param context: pipeline execution context
        :param data_container: data container to fit transform
        :return: tuple(fitted pipeline, transformed data container)
        """
        context.push(self.name, self)

        new_self, data_container = self._fit_transform_core(data_container, context)
        ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(ids)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: Context) -> DataContainer:
        """
        Transform then rehash ids with hyperparams and transformed data inputs

        :param context: pipeline execution context
        :param data_container: data container to transform
        :return: tuple(fitted pipeline, transformed data container)
        """
        context.push(self.name, self)

        data_container = self._transform_core(data_container, context)
        ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(ids)

        return data_container

    def _fit_transform_core(self, data_container: DataContainer, context: Context) -> ('Pipeline', DataContainer):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_container: the data container to fit transform on
        :param context: the pipeline execution context

        :return: tuple(pipeline, data_container)
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)

        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_left_to_do:
            step, data_container = step.handle_fit_transform(data_container, context)
            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[:len(self.steps_as_tuple) - len(steps_left_to_do)] + \
                              new_steps_as_tuple

        return self, data_container

    def _transform_core(self, data_container: DataContainer, context: Context) -> DataContainer:
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_container: the data container to transform
        :param context: the pipeline execution context

        :return: transformed data container
        """
        steps_left_to_do, data_container = self._load_checkpoint(data_container, context)

        for step_name, step in steps_left_to_do:
            data_container = step.handle_transform(data_container, context)

        return data_container

    def _load_checkpoint(self, data_container: DataContainer, context: Context) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :param context: the pipeline execution context

        :return: tuple(steps left to do, last checkpoint data container)
        """
        return self.steps_as_tuple, data_container


class ResumablePipeline(Pipeline, ResumableStepMixin):
    """
    Fits and transform steps after latest checkpoint
    """

    def __init__(self, steps: NamedTupleList):
        Pipeline.__init__(self, steps=steps)

    def _load_checkpoint(self, data_container: DataContainer, context: Context) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :param context: the pipeline execution context
        :return: tuple(steps left to do, last checkpoint data container)
        """
        new_starting_step_index, starting_step_data_container, starting_step_context = \
            self._get_starting_step_info(data_container, context)

        step = self[new_starting_step_index]
        if isinstance(step, BaseCheckpointStep):
            starting_step_data_container = step.read_checkpoint(starting_step_data_container)

        return self[new_starting_step_index:], starting_step_data_container

    def _get_starting_step_info(self, data_container: DataContainer, context: Context) -> \
            Tuple[int, DataContainer, Context]:
        """
        Find the index of the latest step that can be resumed

        :param data_container: the data container to resume
        :return: index of the latest resumable step, data container at starting step
        """
        starting_step_data_container = copy(data_container)
        starting_step_context = copy(context)
        current_data_container = copy(data_container)
        index_latest_checkpoint = 0

        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, ResumableStepMixin):
                if step.should_resume(current_data_container, context):
                    index_latest_checkpoint = index
                    starting_step_data_container = copy(current_data_container)

            current_ids = step.hash(
                current_ids=current_data_container.current_ids,
                hyperparameters=step.hyperparams,
                data_inputs=current_data_container.data_inputs
            )

            current_data_container.set_current_ids(current_ids)

        return index_latest_checkpoint, starting_step_data_container, starting_step_context

    def should_resume(self, data_container: DataContainer, context: Context) -> bool:
        """
        Return True if the pipeline has a saved checkpoint that it can resume from

        :param context: pipeline execution context
        :param data_container: the data container to resume
        :return: bool
        """
        for index, (step_name, step) in enumerate(reversed(self.items())):
            if isinstance(step, ResumableStepMixin):
                if step.should_resume(
                        data_container=data_container,
                        context=context.push(step.name, step)
                ):
                    return True

        return False
