"""
Neuraxle's Flow Steps
====================================
Pipeline wrapper steps that only implement the handle methods, and don't apply any transformation to the data.

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
..

    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext, \
    ForceHandleMixin, ResumableStepMixin
from neuraxle.data_container import DataContainer, ExpandedDataContainer


class ExpandDim(
    ForceHandleMixin,
    MetaStepMixin,
    ResumableStepMixin,
    BaseStep
):
    """
    Similar to numpys expand_dim function, ExpandDim step expands the dimension of all the data inside the data container.
    ExpandDim sends the expanded data container to the wrapped step.
    ExpandDim returns the transformed expanded dim reduced to its original shape (see :func:`~neuraxle.steps.loop.ExpandedDataContainer.reduce_dim`).

    The wrapped step will receive a single current_id, data_input, and expected output:
        - The current_id is a list of one element that contains a single summary hash for all of the current ids.
        - The data_inputs is a list of one element that contains the original expected outputs list.
        - The expected_outputs is a list of one element that contains the original expected outputs list.

    .. seealso::
        :class:`ForceHandleMixin`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
        :class:`BaseHasher`
        :class:`ExpandedDataContainer`
    """

    def __init__(self, wrapped: BaseStep):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_transform method, and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        expanded_data_container = self.wrapped.handle_transform(
            expanded_data_container,
            context.push(self.wrapped)
        )

        expanded_data_container = self.handle_after_any(expanded_data_container)

        return expanded_data_container.reduce_dim()

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_fit_transform method,
        and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to fit_transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        self.wrapped, expanded_data_container = self.wrapped.handle_fit_transform(
            expanded_data_container,
            context.push(self.wrapped)
        )

        data_container = expanded_data_container.reduce_dim()
        data_container = self.handle_after_any(data_container)

        return self, data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_fit method,
        and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to fit_transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        self.wrapped, expanded_data_container = self.wrapped.handle_fit(
            expanded_data_container,
            context.push(self.wrapped)
        )

        data_container = expanded_data_container.reduce_dim()
        data_container = self.handle_after_any(data_container)

        return self, data_container

    def _create_expanded_data_container(self, data_container: DataContainer) -> ExpandedDataContainer:
        """
        Create expanded data container.

        :param data_container: data container to expand
        :type data_container: DataContainer
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        data_container = self.hash_data_container(data_container)

        expanded_data_container = ExpandedDataContainer.create_from(data_container)

        return expanded_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        expanded_data_container = self._create_expanded_data_container(data_container)
        print('should_resume summary_id {0}'.format(expanded_data_container.summary_id))

        if isinstance(self.wrapped, ResumableStepMixin) and \
                self.wrapped.should_resume(expanded_data_container, context.push(self.wrapped)):
            return True
        return False
