"""
Neuraxle's Checkpoint Classes
====================================
The checkpoint classes used by the checkpoint pipeline runner

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
import os
import pickle
from abc import abstractmethod
from typing import List, Tuple

from build.lib.neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.base import ResumableStepMixin, BaseStep, DataContainer, ListDataContainer, DEFAULT_CACHE_FOLDER, \
    ExecutionContext, NonTransformableMixin, NonFittableMixin


class BaseCheckpoint(BaseStep):
    def __init__(self, cache_folder: str = DEFAULT_CACHE_FOLDER):
        BaseStep.__init__(self)
        self.cache_folder = cache_folder

    @abstractmethod
    def is_fit(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_fit_transform(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_transform(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        raise NotImplementedError()


class ReadableCheckpointMixin:
    @abstractmethod
    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        raise NotImplementedError()

    def set_checkpoint_path(self, path):
        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)


class StepCheckpoint(NonFittableMixin, NonTransformableMixin, BaseCheckpoint):
    def is_fit(self) -> bool:
        return True

    def is_fit_transform(self) -> bool:
        return True

    def is_transform(self) -> bool:
        return False

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        context.save_all_unsaved()


class BaseDataInputCheckpoint(NonFittableMixin, NonTransformableMixin, ReadableCheckpointMixin, BaseCheckpoint):
    def is_fit(self) -> bool:
        return True

    def is_fit_transform(self) -> bool:
        return True

    def is_transform(self) -> bool:
        return True

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        self.set_checkpoint_path(context.get_path())
        for current_id, data_input, expected_output in data_container:
            self.save_current_id_checkpoint(current_id, data_input, expected_output)

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        list_data_container = ListDataContainer.empty()

        for current_id, data_input, expected_output in data_container:
            checkpoint_current_id, checkpoint_data_input, checkpoint_expected_outputs = self.read_current_id_checkpoint(
                current_id)
            list_data_container.append(
                checkpoint_current_id,
                checkpoint_data_input,
                checkpoint_expected_outputs
            )

        return list_data_container

    @abstractmethod
    def save_current_id_checkpoint(self, current_id, data_input, expected_output) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def read_current_id_checkpoint(self, current_id) -> DataContainer:
        raise NotImplementedError()


class PickleDataInputCheckpoint(BaseDataInputCheckpoint):
    def save_current_id_checkpoint(self, current_id, data_input, expected_output) -> bool:
        with open(self.get_checkpoint_file_path(current_id), 'wb') as file:
            pickle.dump(
                (current_id, data_input, expected_output),
                file
            )
        return True

    def read_current_id_checkpoint(self, current_id) -> Tuple:
        with open(self.get_checkpoint_file_path(current_id), 'rb') as file:
            (checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output) = \
                pickle.load(file)
            return checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output

    def get_checkpoint_file_path(self, current_id) -> str:
        """
        Returns the checkpoint file path for a data input id

        :param current_id:
        :return:
        """
        return os.path.join(
            self.checkpoint_path,
            '{0}.pickle'.format(current_id)
        )


class BaseExpectedOutputsCheckpoint(NonFittableMixin, NonTransformableMixin, ReadableCheckpointMixin, BaseCheckpoint):
    def is_fit(self) -> bool:
        return True

    def is_fit_transform(self) -> bool:
        return True

    def is_transform(self) -> bool:
        return False

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        pass

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        pass


class Checkpoint(BaseStep):
    def __init__(self, checkpoint_savers: List[BaseCheckpoint] = None):
        BaseStep.__init__(self)
        if checkpoint_savers is None:
            checkpoint_savers = [StepCheckpoint(), BaseDataInputCheckpoint(), BaseExpectedOutputsCheckpoint()]
        self.checkpoint_savers: List[BaseCheckpoint] = checkpoint_savers

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        data_container = self._handle_any(context, data_container)
        return self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_container = self._handle_any(context, data_container)
        return data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        data_container = self._handle_any(context, data_container)
        return self, data_container

    def _handle_any(self, context, data_container):
        self.set_checkpoint_path(context.get_path())
        data_container: DataContainer = self.save_checkpoint(data_container)

        self.save_checkpoint(data_container)
        context.save_all_unsaved()

        return data_container


class BaseCheckpointStep(ResumableStepMixin, BaseStep):
    """
    Base class for a checkpoint step that can persists the received data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name: str = None):
        ResumableStepMixin.__init__(self)
        BaseStep.__init__(self)
        self.force_checkpoint_name = force_checkpoint_name

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        data_container = self._handle_any(context, data_container)
        return self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_container = self._handle_any(context, data_container)
        return data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        data_container = self._handle_any(context, data_container)
        return self, data_container

    def _handle_any(self, context, data_container):
        self.set_checkpoint_path(context.get_path())
        data_container: DataContainer = self.save_checkpoint(data_container)

        self.save_checkpoint(data_container)
        context.save_all_unsaved()

        return data_container

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCheckpointStep':
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run

        :param expected_outputs: initial expected outputs of pipeline to load checkpoint from
        :param data_inputs: data inputs to save
        :return: self
        """
        return self

    def transform(self, data_inputs):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run

        :param data_inputs: data inputs to save
        :return: data_inputs
        """
        return data_inputs

    @abstractmethod
    def set_checkpoint_path(self, path):
        """
        Set checkpoint Path

        :param path: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Read checkpoint data to get the data inputs and expected output.

        :param data_container: data inputs to save
        :return: checkpoint data container
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run

        :param data_container: data inputs to save
        :return: saved data container
        """
        raise NotImplementedError()


class PickleCheckpointStep(BaseCheckpointStep):
    """
    Create pickles for a checkpoint step data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, cache_folder: str = DEFAULT_CACHE_FOLDER):
        super().__init__()
        self.cache_folder = cache_folder

    def read_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Read pickle files for data inputs and expected outputs checkpoint

        :return: tuple(data_inputs, expected_outputs
        """
        list_data_container = ListDataContainer.empty()

        for current_id, data_input, expected_output in data_container:
            with open(self.get_checkpoint_file_path(current_id), 'rb') as file:
                (checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output) = \
                    pickle.load(file)
                list_data_container.append(
                    current_id=checkpoint_current_id,
                    data_input=checkpoint_data_input,
                    expected_output=checkpoint_expected_output
                )

        return list_data_container

    def save_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Save pickle files for data inputs and expected output to create a checkpoint

        :param data_container: data to resume
        :return:
        """
        for current_id, data_input, expected_output in data_container:
            with open(self.get_checkpoint_file_path(current_id), 'wb') as file:
                pickle.dump(
                    (current_id, data_input, expected_output),
                    file
                )

        return data_container

    def set_checkpoint_path(self, path):
        """
        Set checkpoint path inside the cache folder (ex: cache_folder/pipeline/step_a/current_id.pickle)

        :param path: checkpoint path
        """
        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Whether or not we should resume the pipeline (if the checkpoint exists)

        :param context: execution context
        :param data_container: data to resume
        :return:
        """
        self.set_checkpoint_path(context.get_path())
        return self._checkpoint_exists(data_container)

    def _checkpoint_exists(self, data_container: DataContainer) -> bool:
        """
        Returns True if the checkpoints for each data input id exists
        :param data_container:
        :return:
        """
        for current_id in data_container.current_ids:
            if not os.path.exists(self.get_checkpoint_file_path(current_id)):
                return False

        return True

    def get_checkpoint_file_path(self, current_id) -> str:
        """
        Returns the checkpoint file path for a data input id

        :param current_id:
        :return:
        """
        return os.path.join(
            self.checkpoint_path,
            '{0}.pickle'.format(current_id)
        )
