import copy
import glob
import hashlib
import json
import os
import time
import traceback
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.callbacks import BaseCallback, CallbackList, MetricCallback
from neuraxle.metaopt.random import BaseCrossValidationWrapper
from neuraxle.metaopt.trial import Trial, TRIAL_STATUS, Trials
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class HyperparamsRepository(ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`HyperparameterSamples`,
        :class:`Trainer`

    .. seealso::
        :class:`AutoMLSequentialWrapper`
        :class:`AutoMLAlgorithm`,
        :class:`BaseValidation`,
        :class:`RandomSearchBaseAutoMLStrategy`,
        :class:`HyperparameterSpace`,
        :class:`HyperparameterSamples`
    """

    def __init__(self, hyperparameter_optimizer=None, cache_folder=None, best_retrained_model_folder=None):
        if best_retrained_model_folder is None:
            best_retrained_model_folder = os.path.join(cache_folder, 'best')
        self.best_retrained_model_folder = best_retrained_model_folder

        self.hyperparameter_optimizer = hyperparameter_optimizer
        self.cache_folder = cache_folder

    def set_optimizer(self, hyperparameter_optimizer: 'BaseHyperparameterSelectionStrategy'):
        """
        Set optimizer.

        :param hyperparameter_optimizer: hyperparameter optimizer
        :return:
        """
        self.hyperparameter_optimizer = hyperparameter_optimizer

    @abstractmethod
    def load_all_trials(self, status: 'TRIAL_STATUS') -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.

        :return: Trials (hyperparams, scores)
        """
        pass

    @abstractmethod
    def save_trial(self, trial: 'Trial'):
        """
        Save trial.

        :param trial: trial to save.
        :return:
        """
        pass

    def get_best_hyperparams(self) -> HyperparameterSamples:
        trials = self.load_all_trials(status=TRIAL_STATUS.SUCCESS)
        best_hyperparams = HyperparameterSamples(trials.get_best_hyperparams())
        return best_hyperparams

    def get_best_model(self):
        hyperparams: HyperparameterSamples = self.get_best_hyperparams()
        trial_hash: str = self._get_trial_hash(HyperparameterSamples(hyperparams).to_flat_as_dict_primitive())
        p: BaseStep = ExecutionContext(str(self.cache_folder)).load(trial_hash)

        return p

    def save_best_model(self, step: BaseStep):
        hyperparams = step.get_hyperparams().to_flat_as_dict_primitive()
        trial_hash = self._get_trial_hash(hyperparams)
        step.set_name(trial_hash).save(ExecutionContext(self.best_retrained_model_folder), full_dump=True)

        return step


    @abstractmethod
    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Save hyperparams, and score for a failed trial.

        :return: (hyperparams, scores)
        """
        pass

    def _get_trial_hash(self, hp_dict):
        """
        Hash hyperparams with md5 to create a trial hash.

        :param hp_dict:
        :return:
        """
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.

    Example usage :

    .. code-block:: python

        InMemoryHyperparamsRepository(
            hyperparameter_optimizer=RandomSearchHyperparameterOptimizer(),
            print_func=print,
            cache_folder='cache'
        )

    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`HyperparamsRepository`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`HyperparameterSamples`,
        :class:`Trainer`
    """

    def __init__(self, hyperparameter_optimizer=None, print_func: Callable = None, cache_folder: str = None):
        HyperparamsRepository.__init__(
            self,
            hyperparameter_optimizer=hyperparameter_optimizer,
            cache_folder=cache_folder
        )
        if print_func is None:
            print_func = print
        self.print_func = print_func
        self.cache_folder = cache_folder

        self.trials = Trials()

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def save_trial(self, trial: 'Trial'):
        """
        Save trial.

        :param trial: trial to save
        :return:
        """
        self.print_func(trial)
        self.trials.append(trial)

    def new_trial(self, auto_ml_container: 'AutoMLContainer') -> 'Trial':
        """
        Create a new trial with the best next hyperparams.

        :param auto_ml_container: auto ml data container
        :return: trial
        """
        hyperparams = self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)
        self.print_func('new trial:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        return Trial(hyperparams)


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`HyperparamsRepository`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`HyperparameterSamples`,
        :class:`Trainer`
    """

    def __init__(self, hyperparameter_optimizer: 'BaseHyperparameterSelectionStrategy' = None, cache_folder=None, best_retrained_model_folder=None):
        HyperparamsRepository.__init__(self, hyperparameter_optimizer=hyperparameter_optimizer,
                                       cache_folder=cache_folder, best_retrained_model_folder=best_retrained_model_folder)
        self.best_retrained_model_folder = best_retrained_model_folder

    def save_trial(self, trial: 'Trial'):
        """
        Save trial json.

        :param trial: trial to save
        :return:
        """
        hp_dict = trial.hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        if trial.status == TRIAL_STATUS.SUCCESS:
            trial_file_path = self._get_successful_trial_json_file_path(trial)
        else:
            trial_file_path = self._get_failed_trial_json_file_path(trial)

        with open(trial_file_path, 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        hyperparams = self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)
        trial = Trial(hyperparams, cache_folder=self.cache_folder)
        self._create_trial_json(trial=trial)

        return trial

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :return: (hyperparams, scores)
        """
        trials = Trials()

        files = glob.glob(os.path.join(self.cache_folder, '*.json'))

        # sort by created date:
        def getmtimens(filename):
            return os.stat(filename).st_mtime_ns

        files.sort(key=getmtimens)

        for base_path in files:
            with open(base_path) as f:
                trial_json = json.load(f)

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(trial_json))

        return trials

    def _create_trial_json(self, trial: 'Trial'):
        """
        Save new trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = trial.hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        with open(os.path.join(self._get_new_trial_json_path(current_hyperparameters_hash)), 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

    def _get_successful_trial_json_file_path(self, trial: 'Trial'):
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_as_dict_primitive())
        return os.path.join(self.cache_folder,
                            str(float(trial.validation_score)).replace('.', ',') + "_" + trial_hash) + '.json'

    def _get_failed_trial_json_file_path(self, trial: 'Trial'):
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_as_dict_primitive())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _remove_new_trial_json(self, current_hyperparameters_hash):
        new_trial_json = self._get_new_trial_json_path(current_hyperparameters_hash)
        if os.path.exists(new_trial_json):
            os.remove(new_trial_json)

    def _get_new_trial_json_path(self, current_hyperparameters_hash):
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'


class BaseHyperparameterSelectionStrategy(ABC):
    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :type auto_ml_container: neuraxle.metaopt.new_automl.Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        raise NotImplementedError()


class Trainer:
    """

    Example usage :

    .. code-block:: python

        trainer = Trainer(
            metrics=self.metrics,
            callbacks=self.callbacks,
            score=self.scoring_function,
            epochs=self.epochs
        )

        trial = trainer.fit(
            p=p,
            trial_repository=repo_trial,
            train_data_container=training_data_container,
            validation_data_container=validation_data_container,
            context=context
        )

        pipeline = trainer.refit(repo_trial.pipeline, data_container, context)

    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`HyperparamsRepository`,
        :class:`HyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            epochs,
            metrics=None,
            callbacks=None,
            print_metrics=True,
            print_func=None
    ):
        self.epochs = epochs
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        self._initialize_metrics(metrics)

        self.callbacks = CallbackList(callbacks)

        if print_func is None:
            print_func = print

        self.print_func = print_func
        self.print_metrics = print_metrics

    def fit(self, p, train_data_container: DataContainer, validation_data_container: DataContainer, trial: Trial,
            context: ExecutionContext) -> Trial:
        """
        Train pipeline using the training data container.
        Track training, and validation metrics for each epoch.

        :param p: pipeline to train on
        :param train_data_container: train data container
        :param validation_data_container: validation data container
        :param trial: trial to execute
        :param context: execution context

        :return: executed trial
        """
        early_stopping = False

        for i in range(self.epochs):
            self.print_func('epoch {}/{}'.format(i, self.epochs))
            p = p.handle_fit(train_data_container, context)

            y_pred_train = p.handle_predict(train_data_container, context)
            y_pred_val = p.handle_predict(validation_data_container, context)

            trial.set_fitted_pipeline(pipeline=p)

            if self.callbacks.call(
                    trial=trial,
                    epoch_number=i,
                    total_epochs=self.epochs,
                    input_train=train_data_container,
                    pred_train=y_pred_train,
                    input_val=validation_data_container,
                    pred_val=y_pred_val,
                    is_finished_and_fitted=early_stopping
            ):
                break

        return trial

    def refit(self, p: BaseStep, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param p: trial to refit
        :param data_container: data container
        :param context: execution context

        :return: fitted pipeline
        """
        early_stopping = False

        train_scores = []

        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)
            pred = p.handle_predict(data_container, context)

            train_score = self.refit_score(pred.data_inputs, pred.expected_outputs)
            train_scores.append(train_score)

            for callback in self.refit_callbacks:
                if callback.call(train_scores):
                    early_stopping = True

            if early_stopping:
                break

        return p

    def _initialize_metrics(self, metrics):
        """
        Initialize metrics results dict for train, and validation using the metrics function dict.

        :param metrics: metrics function dict
        :type metrics: dict

        :return:
        """
        self.metrics_results_train = {}
        self.metrics_results_validation = {}

        for m in metrics:
            self.metrics_results_train[m] = []
            self.metrics_results_validation[m] = []



class AutoML(ForceHandleOnlyMixin, BaseStep):
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml = AutoML(
            pipeline=Pipeline([
                MultiplyByN(2),
                NumpyReshape(shape=(-1, 1)),
                linear_model.LinearRegression()
            ]),
            validation_technique=KFoldCrossValidationWrapper(
                k_fold=2,
                scoring_function=average_kfold_scores(mean_squared_error),
                split_data_container_during_fit=False,
                predict_after_fit=False
            ),
            hyperparams_optimizer=RandomSearchHyperparameterOptimizer(),
            hyperparams_repository=InMemoryHyperparamsRepository(),
            scoring_function=average_kfold_scores(mean_squared_error),
            n_trial=1,
            metrics={'mse': average_kfold_scores(mean_squared_error)},
            epochs=2
        )

        auto_ml = auto_ml.fit(data_inputs, expected_outputs)

    .. seealso::
        :class:`BaseValidation`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`HyperparamsRepository`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`ForceHandleOnlyMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            pipeline,
            validation_split_function: Callable,
            refit_trial,
            scoring_callback: MetricCallback,
            hyperparams_optimizer: BaseHyperparameterSelectionStrategy = None,
            hyperparams_repository: HyperparamsRepository = None,
            n_trials: int = 10,
            epochs: int = 10,
            callbacks: List[BaseCallback] = None,
            refit_scoring_function: Callable = None,
            print_func: Callable = None,
            print_metrics=True,
            cache_folder_when_no_handle=None
    ):
        BaseStep.__init__(self)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.scoring_callback = scoring_callback
        self.validation_split_function = create_split_data_container_function(validation_split_function)

        self.print_metrics = print_metrics
        if print_func is None:
            print_func = print

        if hyperparams_optimizer is None:
            hyperparams_optimizer = RandomSearchHyperparameterSelectionStrategy()
        self.hyperparameter_optimizer = hyperparams_optimizer

        if hyperparams_repository is None:
            hyperparams_repository = HyperparamsJSONRepository(hyperparams_optimizer, cache_folder_when_no_handle)
        else:
            hyperparams_repository.set_optimizer(hyperparams_optimizer)

        self.hyperparams_repository = hyperparams_repository

        self.pipeline = pipeline
        self.print_func = print_func

        self.n_trial = n_trials
        self.hyperparams_repository = hyperparams_repository
        self.hyperparameter_optimizer = hyperparams_optimizer

        self.refit_scoring_function = refit_scoring_function

        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

        self.epochs = epochs
        self.refit_trial = refit_trial

        self.trainer = Trainer(
            callbacks=self.callbacks,
            epochs=self.epochs,
            print_func=self.print_func
        )

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        training_data_container, validation_data_container = self.validation_split_function(data_container)

        for trial_number in range(self.n_trial):
            self.print_func('trial {}/{}'.format(trial_number, self.n_trial))

            auto_ml_data = self._load_auto_ml_data(trial_number)
            p = copy.deepcopy(self.pipeline)

            with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                try:
                    p.update_hyperparams(repo_trial.hyperparams)

                    repo_trial = self.trainer.fit(
                        p=p,
                        train_data_container=training_data_container,
                        validation_data_container=validation_data_container,
                        trial=repo_trial,
                        context=context
                    )

                    repo_trial.set_success()
                except Exception as error:
                    track = traceback.format_exc()
                    self.print_func(track)
                    repo_trial.set_failed(error)

            self.hyperparams_repository.save_trial(repo_trial)

        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()
        p: BaseStep = self._load_virgin_model(hyperparams=best_hyperparams)
        if self.refit_trial:
            p = self.trainer.refit(
                p=p,
                data_container=data_container,
                context=context
            )

            self.hyperparams_repository.save_best_model(p)

        return self

    def get_best_model(self):
        return self.hyperparams_repository.get_best_model()

    def _load_virgin_best_model(self) -> BaseStep:
        """
        Get the best model from all of the previous trials.
        :return: best model step
        :rtype: BaseStep
        """
        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()
        p: Union[BaseCrossValidationWrapper, BaseStep] = copy.copy(self.pipeline)
        p = p.update_hyperparams(best_hyperparams)

        best_model = p.get_step()
        return copy.deepcopy(best_model)

    def _load_virgin_model(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Load virigin model with the given hyperparams.

        :return: best model step
        :rtype: BaseStep
        """
        return copy.deepcopy(self.pipeline).update_hyperparams(hyperparams)

    def _load_auto_ml_data(self, trial_number: int) -> 'AutoMLContainer':
        """
        Load data for all trials.

        :param trial_number: trial number
        :type trial_number: int
        :return: auto ml data container
        :rtype: Trials
        """
        trials = self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS)
        hyperparams_space = self.pipeline.get_hyperparams_space()

        return AutoMLContainer(
            trial_number=trial_number,
            trials=trials,
            hyperparameter_space=hyperparams_space,
        )


class AutoMLContainer:
    """
    Data object for auto ml.

    .. seealso::
        :class:`AutoMLSequentialWrapper`,
        :class:`RandomSearch`,
        :class:`HyperparamsRepository`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            trials: 'Trials',
            hyperparameter_space: HyperparameterSpace,
            trial_number: int
    ):
        self.trials = trials
        self.hyperparameter_space = hyperparameter_space
        self.trial_number = trial_number


class RandomSearchHyperparameterSelectionStrategy(BaseHyperparameterSelectionStrategy):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.
    Please refer to :class:`AutoMLSequentialWrapper` for a usage example.
    .. seealso::
        :class:`AutoMLAlgorithm`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`AutoMLSequentialWrapper`,
        :class:`TrialsContainer`,
        :class:`HyperparameterSamples`,
        :class:`HyperparameterSpace`
    """

    def __init__(self):
        BaseHyperparameterSelectionStrategy.__init__(self)

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.
        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        return auto_ml_container.hyperparameter_space.rvs()


ValidationSplitter = Callable

def create_split_data_container_function(validation_splitter_function: Callable):
    def split_data_container(data_container: DataContainer) -> Tuple[DataContainer, DataContainer]:
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = \
            validation_splitter_function(data_container.data_inputs, data_container.expected_outputs)

        train_data_container = DataContainer(data_inputs=train_data_inputs, expected_outputs=train_expected_outputs)
        validation_data_container = DataContainer(data_inputs=validation_data_inputs, expected_outputs=validation_expected_outputs)

        return train_data_container, validation_data_container

    return split_data_container


def kfold_cross_validation_split(k_fold):
    def split(data_inputs, expected_outputs):
        validation_data_inputs, validation_expected_outputs = \
            kfold_cross_validation_validation_split(data_inputs, expected_outputs)
        train_data_inputs, train_expected_outputs = \
            kfold_cross_validation_train_split(data_inputs, expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def kfold_cross_validation_train_split(data_inputs, expected_outputs) -> (List, List):
        train_data_inputs = []
        train_expected_outputs = []
        data_inputs = np.array(data_inputs)
        expected_outputs = np.array(expected_outputs)
        joiner = NumpyConcatenateOuterBatch()

        for i in range(len(data_inputs)):
            before_di = data_inputs[:i]
            after_di = data_inputs[i + 1:]
            inputs = (before_di, after_di)

            before_eo = expected_outputs[:i]
            after_eo = expected_outputs[i + 1:]
            outputs = (before_eo, after_eo)

            inputs = joiner.transform(inputs)
            outputs = joiner.transform(outputs)

            train_data_inputs.append(inputs)
            train_expected_outputs.append(outputs)

        return train_data_inputs, train_expected_outputs

    def kfold_cross_validation_validation_split(data_inputs, expected_outputs=None) -> (List, List):
        splitted_data_inputs = _kfold_cross_validation_split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = _kfold_cross_validation_split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs

        return splitted_data_inputs, [None] * len(splitted_data_inputs)

    def _kfold_cross_validation_split(data_inputs):
        splitted_data_inputs = []
        step = len(data_inputs) / float(k_fold)
        for i in range(k_fold):
            a = int(step * i)
            b = int(step * (i + 1))
            if b > len(data_inputs):
                b = len(data_inputs)

            slice = data_inputs[a:b]
            splitted_data_inputs.append(slice)

        return splitted_data_inputs

    return split
