"""
Neuraxle's Base Classes
====================================
This is the core of Neuraxle. Most pipeline steps derive (inherit) from those classes. They are worth noticing.

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

import hashlib
import inspect
import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Tuple, List, Union, Any

from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples

DEFAULT_CACHE_FOLDER = 'cache'


class Context:
    def __init__(self, current_path, parent_path_stack, parent_step_stack, is_parent_saved_stack=None):
        self.current_path = current_path

        if parent_path_stack is None:
            parent_path_stack = []

        if parent_step_stack is None:
            parent_step_stack = []

        self.parent_path_stack = parent_path_stack
        self.parent_step_stack = parent_step_stack

        if is_parent_saved_stack is None:
            self.is_parent_saved_stack = [False] * len(parent_path_stack)
        else:
            self.is_parent_saved_stack = is_parent_saved_stack

    @staticmethod
    def from_step(step: 'BaseStep'):
        return Context(current_path=step.name, parent_path_stack=[step.name], parent_step_stack=[step])

    def pop(self) -> 'Context':
        return Context(
            current_path=self.parent_path_stack[-1],
            parent_path_stack=self.parent_path_stack[:-1],
            parent_step_stack=self.parent_step_stack[:-1],
            is_parent_saved_stack=self.is_parent_saved_stack[:-1]
        )

    def push(self, path, step) -> 'Context':
        new_current_path = os.path.join(self.current_path, path)
        return Context(
            current_path=path,
            parent_path_stack=self.parent_path_stack.append(new_current_path),
            parent_step_stack=self.parent_step_stack.append(step),
            is_parent_saved_stack=self.is_parent_saved_stack.append(False)
        )


class BaseHasher(ABC):
    @abstractmethod
    def hash(self, current_ids, hyperparameters: HyperparameterSamples, data_inputs: Any):
        raise NotImplementedError()

    def _hash_hyperparameters(self, hyperparams: HyperparameterSamples):
        hyperperams_dict = hyperparams.to_flat_as_dict_primitive()
        return hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()


class RangeHasher(BaseHasher):
    def hash(self, current_ids, hyperparameters, data_inputs: Any = None):
        if current_ids is None:
            current_ids = [str(i) for i in range(len(data_inputs))]

        if len(hyperparameters) == 0:
            return current_ids

        current_hyperparameters_hash = self._hash_hyperparameters(hyperparameters)

        new_current_ids = []
        for current_id in current_ids:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids


class DataContainer:
    def __init__(self,
                 current_ids,
                 data_inputs: Any,
                 expected_outputs: Any = None
                 ):
        self.current_ids = current_ids
        self.data_inputs = data_inputs
        if expected_outputs is None:
            self.expected_outputs = [None] * len(current_ids)
        else:
            self.expected_outputs = expected_outputs

    def set_data_inputs(self, data_inputs: Any):
        self.data_inputs = data_inputs

    def set_expected_outputs(self, expected_outputs: Any):
        self.expected_outputs = expected_outputs

    def set_current_ids(self, current_ids: Any):
        self.current_ids = current_ids

    def __iter__(self):
        current_ids = self.current_ids
        if self.current_ids is None:
            current_ids = [None] * len(self.data_inputs)

        expected_outputs = self.expected_outputs
        if self.expected_outputs is None:
            expected_outputs = [None] * len(self.data_inputs)

        return zip(current_ids, self.data_inputs, expected_outputs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + "(current_ids=" + repr(list(self.current_ids)) + ", ...)"

    def __len__(self):
        return len(self.data_inputs)


class ListDataContainer(DataContainer):
    @staticmethod
    def empty() -> 'ListDataContainer':
        return ListDataContainer([], [], [])

    def append(self, current_id, data_input, expected_output):
        self.current_ids.append(current_id)
        self.data_inputs.append(data_input)
        self.expected_outputs.append(expected_output)


class BaseStep(ABC):
    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            saver: 'StepSaver' = None
    ):

        if hyperparams is None:
            hyperparams = dict()
        if hyperparams_space is None:
            hyperparams_space = dict()
        if name is None:
            name = self.__class__.__name__

        self.name: str = name

        self.hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams)
        self.hyperparams = self.hyperparams.to_flat()

        self.hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)
        self.hyperparams_space = self.hyperparams_space.to_flat()

        self.pending_mutate: ('BaseStep', str, str) = (None, None, None)
        self.is_initialized = False

        if saver is None:
            self.saver = JoblibStepSaver(DEFAULT_CACHE_FOLDER)
        else:
            self.saver = saver

    def hash(self, current_ids, hyperparameters, step_source_code: str, data_inputs: Any = None):
        """
        :param current_ids:
        :param hyperparameters:
        :param step_source_code:
        :param data_inputs:
        :return:
        """
        if current_ids is None:
            current_ids = [str(i) for i in range(len(data_inputs))]

        if len(hyperparameters) == 0:
            return current_ids

        current_hyperparameters_hash = self._hash_hyperparameters(hyperparameters)

        new_current_ids = []
        for current_id in current_ids:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(step_source_code))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids

    def _hash_hyperparameters(self, hyperparams: HyperparameterSamples):
        hyperperams_dict = hyperparams.to_flat_as_dict_primitive()
        return hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()

    def save(self, data_container: DataContainer, context: Context):
        self.saver.save(self, data_container, context)

    def setup(self, step_path: str, setup_arguments: dict) -> 'BaseStep':
        """
        Initialize step before it runs

        :param step_path: pipeline step path ex: pipeline/step_name/
        :param setup_arguments: any setup arguments that need to be passed to the setup method of one of the pipeline step
        :return: self
        """
        self.is_initialized = True
        return self

    def teardown(self):
        """
        Teardown step after program execution

        :return:
        """
        self.is_initialized = False
        return self

    def set_name(self, name: str):
        """
        Set the name of the pipeline step.

        :param name: a string.
        :return: self
        """
        self.name = name
        return self

    def get_name(self) -> str:
        """
        Get the name of the pipeline step.

        :return: the name, a string.
        """
        return self.name

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseStep':
        self.hyperparams = HyperparameterSamples(hyperparams).to_flat()
        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        return self.hyperparams

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        self.hyperparams_space = HyperparameterSpace(hyperparams_space).to_flat()
        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        return self.hyperparams_space

    def handle_fit_transform(self, data_container: DataContainer, context: Context) -> ('BaseStep', DataContainer):
        """
        Update the data inputs inside DataContainer after fit transform,
        and update its current_ids.

        :param context: pipeline execution context
        :param data_container: the data container to transform
        :return: tuple(fitted pipeline, data_container)
        """
        new_self, out = self.fit_transform(data_container.data_inputs, data_container.expected_outputs)
        data_container.set_data_inputs(out)

        current_ids = self.hash(
            current_ids=data_container.current_ids,
            hyperparameters=self.hyperparams,
            step_source_code=inspect.getsource(self.__class__),
            data_inputs=out
        )
        data_container.set_current_ids(current_ids)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: Context) -> DataContainer:
        """
        Update the data inputs inside DataContainer after transform.

        :param context: pipeline execution context
        :param data_container: the data container to transform
        :return: transformed data container
        """
        out = self.transform(data_container.data_inputs)
        data_container.set_data_inputs(out)

        current_ids = self.hash(
            current_ids=data_container.current_ids,
            hyperparameters=self.hyperparams,
            step_source_code=inspect.getsource(self.__class__),
            data_inputs=out
        )
        data_container.set_current_ids(current_ids)

        return data_container

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self.transform(data_inputs)

        return new_self, out

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        for data_input, expected_output in zip(data_inputs, expected_outputs):
            self.fit_one(data_input, expected_output)

        return self

    def transform(self, data_inputs):
        processed_outputs = [self.transform_one(data_input) for data_input in data_inputs]
        return processed_outputs

    def inverse_transform(self, processed_outputs):
        data_inputs = [self.inverse_transform_one(data_output) for data_output in processed_outputs]
        return data_inputs

    def fit_one(self, data_input, expected_output=None) -> 'BaseStep':
        # return self
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def transform_one(self, data_input):
        # return processed_output
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def inverse_transform_one(self, data_output):
        # return data_input
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def predict(self, data_input):
        return self.transform(data_input)

    def meta_fit(self, X_train, y_train, metastep: 'MetaStepMixin'):
        """
        Uses a meta optimization technique (AutoML) to find the best hyperparameters in the given
        hyperparameter space.

        Usage: ``p = p.meta_fit(X_train, y_train, metastep=RandomSearch(n_iter=10, scoring_function=r2_score, higher_score_is_better=True))``

        Call ``.mutate(new_method="inverse_transform", method_to_assign_to="transform")``, and the
        current estimator will become

        :param X_train: data_inputs.
        :param y_train: expected_outputs.
        :param metastep: a metastep, that is, a step that can sift through the hyperparameter space of another estimator.
        :return: your best self.
        """
        metastep.set_step(self)
        metastep = metastep.fit(X_train, y_train)
        best_step = metastep.get_best_model()
        return best_step

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Replace the "method_to_assign_to" method by the "new_method" method, IF the present object has no pending calls to
        ``.will_mutate_to()`` waiting to be applied. If there is a pending call, the pending call will override the
        methods specified in the present call. If the change fails (such as if the new_method doesn't exist), then
        a warning is printed (optional). By default, there is no pending ``will_mutate_to`` call.

        This could for example be useful within a pipeline to apply ``inverse_transform`` to every pipeline steps, or
        to assign ``predict_probas`` to ``predict``, or to assign "inverse_transform" to "transform" to a reversed pipeline.

        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        pending_new_base_step, pending_new_method, pending_method_to_assign_to = self.pending_mutate

        # Use everything that is pending if they are not none (ternaries).
        new_base_step = pending_new_base_step if pending_new_base_step is not None else copy(self)
        new_method = pending_new_method if pending_new_method is not None else new_method
        method_to_assign_to = pending_method_to_assign_to if pending_method_to_assign_to is not None else method_to_assign_to

        # We set "new_method" in place of "method_to_affect" to a copy of self:
        try:
            # 1. get new method's reference
            new_method = getattr(new_base_step, new_method)

            # 2. delete old method
            try:
                delattr(new_base_step, method_to_assign_to)
            except AttributeError as e:
                pass

            # 3. assign new method to old method
            setattr(new_base_step, method_to_assign_to, new_method)

        except AttributeError as e:
            if warn:
                import warnings
                warnings.warn(e)

        return new_base_step

    def will_mutate_to(
            self, new_base_step: 'BaseStep' = None, new_method: str = None, method_to_assign_to: str = None
    ) -> 'BaseStep':
        """
        This will change the behavior of ``self.mutate(<...>)`` such that when mutating, it will return the
        presently provided new_base_step BaseStep (can be left to None for self), and the ``.mutate`` method
        will also apply the ``new_method`` and the  ``method_to_affect``, if they are not None, and after changing
        the object to new_base_step.

        This can be useful if your pipeline requires unsupervised pretraining. For example:

        .. code-block:: python

            X_pretrain = ...
            X_train = ...

            p = Pipeline(
                SomePreprocessing(),
                SomePretrainingStep().will_mutate_to(new_base_step=SomeStepThatWillUseThePretrainingStep),
                Identity().will_mutate_to(new_base_step=ClassifierThatWillBeUsedOnlyAfterThePretraining)
            )
            # Pre-train the pipeline
            p = p.fit(X_pretrain, y=None)

            # This will leave `SomePreprocessing()` untouched and will affect the two other steps.
            p = p.mutate(new_method="transform", method_to_affect="transform")

            # Pre-train the pipeline
            p = p.fit(X_train, y_train)  # Then fit the classifier and other new things

        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self

    def tosklearn(self) -> 'NeuraxleToSKLearnPipelineWrapper':
        from sklearn.base import BaseEstimator

        class NeuraxleToSKLearnPipelineWrapper(BaseEstimator):
            def __init__(self, neuraxle_step):
                self.p: Union[BaseStep, TruncableSteps] = neuraxle_step

            def set_params(self, **params) -> BaseEstimator:
                self.p.set_hyperparams(HyperparameterSpace(params))
                return self

            def get_params(self, deep=True):
                neuraxle_params = HyperparameterSamples(self.p.get_hyperparams()).to_flat_as_dict_primitive()
                return neuraxle_params

            def get_params_space(self, deep=True):
                neuraxle_params = HyperparameterSpace(self.p.get_hyperparams_space()).to_flat_as_dict_primitive()
                return neuraxle_params

            def fit(self, **args) -> BaseEstimator:
                self.p = self.p.fit(**args)
                return self

            def transform(self, **args):
                return self.p.transform(**args)

            def fit_transform(self, **args) -> Any:
                self.p, out = self.p.fit_transform(**args)
                # Careful: 1 return value.
                return out

            def inverse_transform(self, **args):
                return self.p.reverse().transform(**args)

            def predict(self, **args):
                return self.p.transform(**args)

        return NeuraxleToSKLearnPipelineWrapper(self)

    def reverse(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.mutate(new_method="inverse_transform", method_to_assign_to="transform")

    def __reversed__(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.reverse()


class MetaStepMixin:
    """A class to represent a meta step which is used to optimize another step."""

    # TODO: remove equal None, and fix random search at the same time ?
    def __init__(
            self,
            wrapped: BaseStep = None
    ):
        self.wrapped: BaseStep = wrapped

    def setup(self, step_path: str, setup_arguments: dict = None) -> BaseStep:
        self.wrapped.setup(
            step_path=os.path.join(step_path, self.name),
            setup_arguments=setup_arguments
        )

        self.is_initialized = True

        return self

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Set meta step and wrapped step hyperparams using the given hyperparams.

        :param hyperparams: ordered dict containing all hyperparameters
        :type hyperparams: HyperparameterSamples

        :return:
        """
        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get meta step and wrapped step hyperparams as a flat hyperparameter samples.

        :return: hyperparameters
        """
        return HyperparameterSamples({
            **self.hyperparams.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.hyperparams.to_flat_as_dict_primitive()
        }).to_flat()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        """
        Set meta step and wrapped step hyperparams space using the given hyperparams space.

        :param hyperparams_space: ordered dict containing all hyperparameter spaces
        :type hyperparams_space: HyperparameterSpace

        :return: self
        """
        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space.to_nested_dict())

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams_space = HyperparameterSpace(remainders)

        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get meta step and wrapped step hyperparams as a flat hyperparameter space

        :return: hyperparameters_space
        """
        return HyperparameterSpace({
            **self.hyperparams_space.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.hyperparams_space.to_flat_as_dict_primitive()
        }).to_flat()

    def set_step(self, step: BaseStep) -> BaseStep:
        self.wrapped: BaseStep = step
        return self

    def get_best_model(self) -> BaseStep:
        return self.best_model


NamedTupleList = List[Union[Tuple[str, 'BaseStep'], 'BaseStep']]


class NonFittableMixin:
    """A pipeline step that requires no fitting: fitting just returns self when called to do no action.

    Note: fit methods are not implemented"""

    def fit(self, data_inputs, expected_outputs=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
        :return: self
        """
        return self

    def fit_one(self, data_input, expected_output=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_input: the data that would normally be fitted on.
        :param expected_output: the data that would normally be fitted on.
        :return: self
        """
        return self


class NonTransformableMixin:
    """A pipeline step that has no effect at all but to return the same data without changes.

    Note: fit methods are not implemented"""

    def transform(self, data_inputs):
        """
        Do nothing - return the same data.

        :param data_inputs: the data to process
        :return: the ``data_inputs``, unchanged.
        """
        return data_inputs

    def inverse_transform(self, processed_outputs):
        """
        Do nothing - return the same data.

        :param processed_outputs: the data to process
        :return: the ``processed_outputs``, unchanged.
        """
        return processed_outputs


class TruncableSteps(BaseStep, ABC):

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict()
    ):
        super().__init__(hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self._set_steps(steps_as_tuple)

        assert isinstance(self, BaseStep), "Classes that inherit from TruncableMixin must also inherit from BaseStep."

    def are_steps_before_index_the_same(self, other: 'TruncableSteps', index: int):
        """
        Returns true if self.steps before index are the same as other.steps before index.

        :param other: other truncable steps to compare
        :param index: max step index to compare

        :return: bool
        """
        for current_index, (step_name, step) in enumerate(self[:index]):
            source_current_step = inspect.getsource(step.__class__)
            source_cached_step = inspect.getsource(other[current_index].__class__)

            if source_current_step != source_cached_step:
                return False

        return True

    def _load_saved_pipeline_steps_before_index(self, saved_pipeline: 'TruncableSteps', index: int):
        """
        Load the cached pipeline steps
        before the index into the current steps

        :param saved_pipeline:
        :param index:
        :return:
        """
        self.set_hyperparams(saved_pipeline.get_hyperparams())
        self.set_hyperparams_space(saved_pipeline.get_hyperparams_space())

        new_truncable_steps = saved_pipeline[:index] + self[index:]
        self._set_steps(new_truncable_steps.steps_as_tuple)

    def _set_steps(self, steps_as_tuple: NamedTupleList):
        """
        Set pipeline steps

        :param steps_as_tuple: List[Tuple[str, BaseStep]] list of tuple containing step name and step
        :return:
        """
        self.steps_as_tuple = self.patch_missing_names(steps_as_tuple)
        self._refresh_steps()

    def setup(self, step_path: str = None, setup_arguments: dict = None) -> 'BaseStep':
        if self.is_initialized:
            return self

        if step_path is None:
            step_path = self.name

        if setup_arguments is None:
            setup_arguments = {}

        for step_name, step in self.steps_as_tuple:
            step.setup(
                step_path=os.path.join(step_path, step_name),
                setup_arguments=setup_arguments
            )

        self.is_initialized = True

        return self

    def teardown(self) -> 'BaseStep':
        for step_name, step in self.steps_as_tuple:
            step.teardown()

        return self

    def patch_missing_names(self, steps_as_tuple: List) -> NamedTupleList:
        names_yet = set()
        patched = []
        for step in steps_as_tuple:

            if isinstance(step, tuple):
                class_name = step[0]
                step = step[1]
            else:
                class_name = step.get_name()

            _name = class_name
            if class_name in names_yet:
                warnings.warn(
                    "Named pipeline tuples must be unique. "
                    "Will rename '{}' because it already exists.".format(class_name))

                _name = self._rename_step(step_name=_name, class_name=class_name, names_yet=names_yet)
                step.set_name(_name)

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        return patched

    def _rename_step(self, step_name, class_name, names_yet):
        """
        Rename step by adding a number suffix after the class name.
        Ensure uniqueness with the names yet parameter.
        :param step_name:
        :param class_name:
        :param names_yet:
        :return:
        """
        # Add suffix number to name if it is already used to ensure name uniqueness.
        i = 1
        while step_name in names_yet:
            step_name = class_name + str(i)
            i += 1
        return step_name

    def _refresh_steps(self):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)

    def get_hyperparams(self) -> HyperparameterSamples:
        hyperparams = dict()

        for k, v in self.steps.items():
            hparams = v.get_hyperparams()  # TODO: oop diamond problem?
            if hasattr(v, "hyperparams"):
                hparams.update(v.hyperparams)
            if len(hparams) > 0:
                hyperparams[k] = hparams

        hyperparams = HyperparameterSamples(hyperparams)

        return hyperparams.to_flat()

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def set_hyperparams_space(self, hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]) -> BaseStep:
        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = HyperparameterSpace(remainders)

        return self

    def get_hyperparams_space(self):
        all_hyperparams = HyperparameterSpace()
        for step_name, step in self.steps_as_tuple:
            hspace = step.get_hyperparams_space()
            all_hyperparams.update({
                step_name: hspace
            })
        all_hyperparams.update(
            super().get_hyperparams_space()
        )

        return all_hyperparams.to_flat()

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Call mutate on every steps the the present truncable step contains.

        :param new_method: the method to replace transform with.
        :param method_to_assign_to: the method to which the new method will be assigned to.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        if self.pending_mutate[0] is None:
            new_base_step = self
            self.pending_mutate = (new_base_step, self.pending_mutate[1], self.pending_mutate[2])

            new_base_step.steps_as_tuple = [
                (
                    k,
                    v.mutate(new_method, method_to_assign_to, warn)
                )
                for k, v in new_base_step.steps_as_tuple
            ]
            new_base_step._refresh_steps()
            return super().mutate(new_method, method_to_assign_to, warn)
        else:
            return super().mutate(new_method, method_to_assign_to, warn)

    def _step_name_to_index(self, step_name):
        for index, (current_step_name, step) in self.steps_as_tuple:
            if current_step_name == step_name:
                return index

    def _step_index_to_name(self, step_index):
        name, _ = self.steps_as_tuple[step_index]
        return name

    def __getitem__(self, key):
        if isinstance(key, slice):
            self_shallow_copy = copy(self)

            if isinstance(key.start, int):
                start = self._step_index_to_name(key.start)
            else:
                start = key.start

            if isinstance(key.stop, int):
                stop = self._step_index_to_name(key.stop)
            else:
                stop = key.stop

            step = key.step
            if step is not None or (start is None and stop is None):
                raise KeyError("Invalid range: '{}'.".format(key))
            new_steps_as_tuple = []
            if start is None:
                if stop not in self.steps.keys():
                    raise KeyError("Stop '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if stop == key:
                        break
                    new_steps_as_tuple.append((key, val))
            elif stop is None:
                if start not in self.steps.keys():
                    raise KeyError("Start '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in reversed(self.steps_as_tuple):
                    new_steps_as_tuple.append((key, val))
                    if start == key:
                        break
                new_steps_as_tuple = list(reversed(new_steps_as_tuple))
            else:
                started = False
                if stop not in self.steps.keys() or start not in self.steps.keys():
                    raise KeyError(
                        "Start or stop ('{}' or '{}') not found in '{}'.".format(start, stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if stop == key:
                        break
                    if not started and start == key:
                        started = True
                    if started:
                        new_steps_as_tuple.append((key, val))
            self_shallow_copy.steps_as_tuple = new_steps_as_tuple
            self_shallow_copy.steps = OrderedDict(new_steps_as_tuple)
            return self_shallow_copy
        else:
            if isinstance(key, int):
                key = self._step_index_to_name(key)

            return self.steps[key]

    def __add__(self, other: 'TruncableSteps') -> 'TruncableSteps':
        self._set_steps(self.steps_as_tuple + other.steps_as_tuple)
        return self

    def items(self):
        return self.steps.items()

    def keys(self):
        return self.steps.keys()

    def values(self):
        return self.steps.values()

    def append(self, item: Tuple[str, 'BaseStep']):
        self.steps_as_tuple.append(item)
        self._refresh_steps()

    def pop(self) -> 'BaseStep':
        return self.popitem()[-1]

    def popitem(self, key=None) -> Tuple[str, 'BaseStep']:
        if key is None:
            item = self.steps_as_tuple.pop()
            self._refresh_steps()
        else:
            item = key, self.steps.pop(key)
            self.steps_as_tuple = list(self.steps.items())
        return item

    def popfront(self) -> 'BaseStep':
        return self.popfrontitem()[-1]

    def popfrontitem(self) -> Tuple[str, 'BaseStep']:
        item = self.steps_as_tuple.pop(0)
        self._refresh_steps()
        return item

    def __contains__(self, item):
        """
        Check wheter the ``item`` key or value (or key value tuple pair) is found in self.

        :param item: The key or value to check if is in self's keys or values.
        :return: True or False
        """
        return item in self.steps.keys() or item in self.steps.values() or item in self.items()

    def __iter__(self):
        """
        Iterate through the steps.

        :return: iter(self.steps_as_tuple)
        """
        return iter(self.steps_as_tuple)

    def __len__(self):
        """
        Return the number of contained steps.

        :return: len(self.steps_as_tuple)
        """
        return len(self.steps_as_tuple)


class ResumableStepMixin:
    """
    A step that can be resumed, for example a checkpoint on disk.
    """

    @abstractmethod
    def should_resume(self, data_container: DataContainer, context: Context) -> bool:
        """
        To know if we can resume or not a resumable step.

        :param context: pipeline execution context
        :param data_container: data container
        :type data_container: DataContainer
        :return bool
        """
        raise NotImplementedError()


class StepSaver(ABC):
    @abstractmethod
    def save(self, step: 'BaseStep', data_container: DataContainer, context: Context) -> 'Pipeline':
        """
        Persist pipeline for current data container

        :param context:
        :param step:
        :param data_container:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def has_saved(self, step: 'BaseStep', data_container: DataContainer, context: Context) -> bool:
        """
        Returns True if the pipeline can be loaded with the passed data container

        :param context:
        :param step:
        :param data_container:
        :return: pipeline
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, step: 'BaseStep', data_container: DataContainer, context: Context) -> 'Pipeline':
        """
        Load pipeline for current data container

        :param context:
        :param step:
        :param data_container:
        :return: pipeline
        """
        raise NotImplementedError()


class JoblibStepSaver(StepSaver):
    """
    Pipeline Repository to persist and load pipeline based on a data container
    """

    def __init__(self, cache_folder, pipeline_cache_list_file_name='pipeline_cache_list.txt'):
        self.pipeline_cache_list_file_name = pipeline_cache_list_file_name
        self.cache_folder = cache_folder

    def save(self, step: 'Pipeline', data_container: DataContainer, context: Context) -> 'Pipeline':
        """
        Persist pipeline for current data container

        :param context:
        :param step:
        :param data_container:
        :return:
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, step.name)

        if not os.path.exists(pipeline_cache_folder):
            os.makedirs(pipeline_cache_folder)

        next_cached_pipeline_path = self._create_cached_pipeline_path(pipeline_cache_folder, data_container)
        dump(step, next_cached_pipeline_path)

        return step

    def has_saved(self, step: 'Pipeline', data_container: DataContainer, context: Context) -> bool:
        """
        Returns True if the pipeline can be loaded with the passed data container

        :param context:
        :param step:
        :param data_container:
        :return: pipeline
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, step.name)

        return os.path.exists(
            self._create_cached_pipeline_path(
                pipeline_cache_folder,
                data_container
            )
        )

    def load(self, step: 'Pipeline', data_container: DataContainer, context: Context) -> 'Pipeline':
        """
        Load pipeline for current data container.

        :param context:
        :param step:
        :param data_container:
        :return: pipeline
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, step.name)

        return load(
            self._create_cached_pipeline_path(
                pipeline_cache_folder,
                data_container
            )
        )

    def _create_cached_pipeline_path(self, pipeline_cache_folder, data_container: 'DataContainer') -> str:
        """
        Create cached pipeline path with data container, and pipeline cache folder.

        :type data_container: DataContainer
        :param pipeline_cache_folder: str

        :return: path string
        """
        all_current_ids_hash = self._hash_data_container(data_container)
        return os.path.join(pipeline_cache_folder, '{0}.joblib'.format(str(all_current_ids_hash)))

    def _hash_data_container(self, data_container):
        """
        Hash data container current ids with md5.

        :param data_container: data container
        :type data_container: DataContainer

        :return: str hexdigest of all of the current ids hashed together.
        """
        all_current_ids_hash = None
        for current_id, *_ in data_container:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            if all_current_ids_hash is not None:
                m.update(str.encode(all_current_ids_hash))
            all_current_ids_hash = m.hexdigest()
        return all_current_ids_hash
