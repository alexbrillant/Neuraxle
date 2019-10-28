"""
Pipeline Steps For Looping
=====================================

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
from typing import List, Any

from neuraxle.base import MetaStepMixin, BaseStep, DataContainer, ExecutionContext, ListDataContainer, \
    NonTransformableMixin, NonFittableMixin
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class ForEachDataInput(NonFittableMixin, NonTransformableMixin, MetaStepMixin, BaseStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.
    """

    def __init__(
            self,
            wrapped: BaseStep
    ):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        output_data_container = ListDataContainer.empty()

        for current_id, di, eo in data_container:
            self.wrapped, output = self.wrapped.handle_fit(
                DataContainer(current_ids=range(len(di)), data_inputs=di, expected_outputs=eo),
                context
            )

            # TODO: summary hash for new current id
            output_data_container.append(current_id, output.data_inputs, output.expected_outputs)

        return self, output_data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        output_data_container = ListDataContainer.empty()

        for current_id, di, eo in data_container:
            output = self.wrapped.handle_transform(
                DataContainer(current_ids=range(len(di)), data_inputs=di, expected_outputs=eo),
                context
            )

            # TODO: summary hash for new current id
            output_data_container.append(current_id, output.data_inputs, output.expected_outputs)

        return output_data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        output_data_container = ListDataContainer.empty()

        for current_id, di, eo in data_container:
            self.wrapped, output = self.wrapped.handle_fit_transform(
                DataContainer(current_ids=range(len(di)), data_inputs=di, expected_outputs=eo),
                context
            )

            # TODO: summary hash for new current id
            output_data_container.append(current_id, output.data_inputs, output.expected_outputs)

        return self, output_data_container

class StepClonerForEachDataInput(MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, copy_op=copy.deepcopy):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        MetaStepMixin.set_hyperparams(self, hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.get_hyperparams()) for s in self.steps]
        return self

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        MetaStepMixin.set_hyperparams_space(self, hyperparams_space)
        self.steps = [s.set_hyperparams_space(self.wrapped.get_hyperparams_space()) for s in self.steps]
        return self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        fit_transform_result = [self.steps[i].fit_transform(di, eo) for i, (di, eo) in
                                enumerate(zip(data_inputs, expected_outputs))]
        self.steps = [step for step, di in fit_transform_result]
        data_inputs = [di for step, di in fit_transform_result]

        return self, data_inputs

    def fit(self, data_inputs: List, expected_outputs: List = None) -> 'StepClonerForEachDataInput':
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        # Fit them all.
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        self.steps = [self.steps[i].fit(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]

        return self

    def transform(self, data_inputs: List) -> List:
        return [self.steps[i].transform(di) for i, di in enumerate(data_inputs)]

    def inverse_transform(self, data_output):
        return [self.steps[i].inverse_transform(di) for i, di in enumerate(data_output)]