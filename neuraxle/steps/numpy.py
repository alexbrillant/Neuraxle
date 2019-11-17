"""
Pipeline Steps Based on NumPy
=====================================
Those steps works with NumPy (np) arrays.

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

import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples


class NumpyFlattenDatum(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs.reshape(data_inputs.shape[0], -1)


class NumpyConcatenateOnCustomAxis(NonFittableMixin, BaseStep):
    """
    Numpy concetenation step where the concatenation is performed along the specified custom axis.
    """

    def __init__(self, axis):
        """
        Create a numpy concatenate on custom axis object.
        :param axis: the axis where the concatenation is performed.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        self.axis = axis
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis.
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def _concat(self, data_inputs):
        return np.concatenate(data_inputs, axis=self.axis)


class NumpyConcatenateInnerFeatures(NumpyConcatenateOnCustomAxis):
    """
    Numpy concetenation step where the concatenation is performed along `axis=-1`.
    """

    def __init__(self):
        """
        Create a numpy concatenate inner features object.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        # The concatenate is on the inner features so axis = -1.
        NumpyConcatenateOnCustomAxis.__init__(self, axis=-1)


class NumpyConcatenateOuterBatch(NumpyConcatenateOnCustomAxis):
    """
    Numpy concetenation step where the concatenation is performed along `axis=0`.
    """

    def __init__(self):
        """
        Create a numpy concetenate outer batch object.
        :return: NumpyConcatenateOnCustomAxis instance which is inherited by base step.
        """
        NumpyConcatenateOnCustomAxis.__init__(self, axis=0)


class NumpyTranspose(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return self._transpose(data_inputs)

    def inverse_transform(self, data_inputs):
        return self._transpose(data_inputs)

    def _transpose(self, data_inputs):
        return np.array(data_inputs).transpose()


class NumpyShapePrinter(NonFittableMixin, BaseStep):

    def __init__(self, custom_message: str = ""):
        self.custom_message = custom_message
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        self._print(data_inputs)
        return data_inputs

    def inverse_transform(self, processed_outputs):
        self._print(processed_outputs)
        return processed_outputs

    def _print(self, data_inputs):
        print(self.__class__.__name__ + ":", data_inputs.shape, self.custom_message)

    def _print_one(self, data_input):
        print(self.__class__.__name__ + " (one):", data_input.shape, self.custom_message)


class MultiplyByN(NonFittableMixin, BaseStep):
    def __init__(self, multiply_by=1):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                'multiply_by': multiply_by
            })
        )

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.hyperparams['multiply_by']

    def inverse_transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs / self.hyperparams['multiply_by']


class OneHotEncoder(NonFittableMixin, BaseStep):
    """
    Step to one hot a set of columns.
    Accepts Integer Columns and converts it ot a one_hot.
    Rounds floats  to integer for safety in the transform.
    """

    def __init__(self, nb_columns, name, npdtype=np.int32):
        super().__init__(name=name)
        self.npdtype = npdtype
        self.nb_columns = nb_columns

    def transform(self, data_inputs):
        """
        Transform data inputs using one hot encoding, adding no_columns to the -1 axis.
        :param data_inputs: data inputs to encode
        :return: one hot encoded data inputs
        """
        # validate enum values
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        # treats invalid values as having no columns activated. create a temporary column for invalid values
        data_inputs[data_inputs == None] = self.nb_columns
        data_inputs[data_inputs >= self.nb_columns] = self.nb_columns
        data_inputs[data_inputs < 0] = self.nb_columns

        # finally, one hot encode data inputs
        outputs_ = np.eye(self.nb_columns + 1)[np.array(data_inputs, dtype=self.npdtype)]

        # delete the invalid values column, and zero hot the invalid values
        outputs_ = np.delete(outputs_, self.nb_columns, axis=-1)

        return np.squeeze(outputs_)
