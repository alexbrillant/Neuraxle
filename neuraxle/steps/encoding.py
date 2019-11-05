"""
Encoding Pipeline Steps
====================================
You can find here encoder pipeline steps, for example, one hot encoder.

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


class OneHotEncoder(NonFittableMixin, BaseStep):
    """
    Step to one hot a set of columns.
    Accepts Integer Columns and converts it ot a one_hot.
    Rounds floats  to integer for safety in the transform.
    """

    def __init__(self, nb_columns, name):
        super().__init__(name=name)
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
        outputs_ = np.eye(self.nb_columns + 1)[np.array(data_inputs, dtype=np.int32)]

        # delete the invalid values column, and zero hot the invalid values
        outputs_ = np.delete(outputs_, self.nb_columns, axis=-1)

        return outputs_

