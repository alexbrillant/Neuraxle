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
from abc import abstractmethod
import os
import pickle
from typing import Tuple, Any

from neuraxle.pipeline import ResumableStep

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


class BaseCheckpointStep(ResumableStep):
    """
    Base class for a checkpoint step that can persists the received data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name: str = None):
        ResumableStep.__init__(self)
        self.force_checkpoint_name = force_checkpoint_name

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCheckpointStep':
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param expected_outputs: initial expected outputs of pipeline to load checkpoint from
        :param data_inputs: data inputs to save
        :return: self
        """
        self.save_checkpoint(data_inputs)

        return self

    def transform(self, data_inputs):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :return: data_inputs
        """
        self.save_checkpoint(data_inputs)

        return data_inputs

    def load_checkpoint(self, data_inputs) -> Tuple[list, Any]:
        """
        Load the checkpoint step data inputs.
        There is no steps left to do when we are loading an actual checkpoint
        instead of a pipeline that can contain checkpoints.
        :param data_inputs:
        :return: steps_left_to_do, data_inputs_checkpoint
        """
        return [], self.read_checkpoint()

    @abstractmethod
    def set_checkpoint_path(self, path):
        """
        Set checkpoint Path
        :param path: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self):
        """
        Read checkpoint data to get the data inputs and expected output.
        :return: tuple(data_inputs, expected_outputs)
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_inputs):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :return:
        """
        raise NotImplementedError()


class PickleCheckpointStep(BaseCheckpointStep):
    """
    Create pickles for a checkpoint step data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name: str = None, cache_folder: str = DEFAULT_CACHE_FOLDER):
        super().__init__(force_checkpoint_name)
        self.cache_folder = cache_folder
        self.force_checkpoint_name = force_checkpoint_name

    def read_checkpoint(self):
        """
        Read pickle files for data inputs and expected outputs checkpoint
        :return: tuple(data_inputs, expected_outputs
        """
        data_inputs_checkpoint_file_name = self.checkpoint_path
        with open(self.get_checkpoint_file_path(data_inputs_checkpoint_file_name), 'rb') as file:
            data_inputs = pickle.load(file)

        return data_inputs

    def save_checkpoint(self, data_inputs):
        """
        Save pickle files for data inputs and expected output
        to create a checkpoint
        :param data_inputs: data inputs to be saved in a pickle file
        :return:
        """
        # TODO: don't force the user to set the checkpoint name (use step name instead).
        self.set_checkpoint_path(self.force_checkpoint_name)
        with open(self.get_checkpoint_file_path(data_inputs), 'wb') as file:
            pickle.dump(data_inputs, file)

    def set_checkpoint_path(self, path):
        """
        Set checkpoint path inside the cache folder (ex: cache_folder/pipeline_name/force_checkpoint_name/data_inputs.pickle)
        :param path: checkpoint path
        """
        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def should_resume(self, data_inputs) -> bool:
        return self.checkpoint_exists(data_inputs)

    def checkpoint_exists(self, data_inputs) -> bool:
        self.set_checkpoint_path(self.force_checkpoint_name)
        return os.path.exists(self.get_checkpoint_file_path(data_inputs))

    def get_checkpoint_file_path(self, data_inputs):
        return os.path.join(self.checkpoint_path, 'data_inputs.pickle')
