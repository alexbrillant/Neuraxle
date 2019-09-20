import shutil
import time
from typing import Any

import numpy as np

from neuraxle.base import BaseStep
from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline


class Multiplication(BaseStep):
    def __init__(self, sleep_time=0.10, hyperparams=None, hyperparams_space=None):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self.sleep_time = sleep_time

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        return self, self.transform(data_inputs)

    def transform(self, data_inputs):
        time.sleep(self.sleep_time)
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.hyperparams['hp_mul']


class AutoMLSequentialWrapper:
    def __init__(
            self,
            pipeline: Pipeline,
            hyperparameters_space: HyperparameterSpace,
            objective_function,
            n_iters=100
    ):
        self.objective_function = objective_function
        self.hyperparameters_space = hyperparameters_space
        self.pipeline = pipeline
        self.n_iters = n_iters

    def fit(self, data_inputs, expected_outputs) -> Pipeline:
        best_score = None
        best_hp = None
        for _ in range(self.n_iters):
            next_hp: HyperparameterSamples = self.hyperparameters_space.rvs()
            self.pipeline.set_hyperparams(next_hp)

            self.pipeline, actual_outputs = self.pipeline.fit_transform(data_inputs, expected_outputs)

            score = self.objective_function(actual_outputs, expected_outputs)
            best_score = best_score if best_score is not None else score

            if score < best_score:
                best_score = score
                best_hp = next_hp

        return self.pipeline.set_hyperparams(best_hp)


def main():
    hyperparams_space = HyperparameterSpace(
        {
            'multiplication_2__hp_mul': RandInt(0, 3),
            'multiplication_3__hp_mul': RandInt(0, 3),
            'multiplication_4__hp_mul': RandInt(0, 3),
            'multiplication_5__hp_mul': RandInt(0, 3),
            'multiplication_6__hp_mul': RandInt(0, 3)
        }
    )

    pipeline = Pipeline([
        ('multiplication_2', Multiplication(sleep_time=0.1)),
        ('multiplication_3', Multiplication(sleep_time=0.1)),
        ('checkpoint_1', PickleCheckpointStep()),
        ('multiplication_4', Multiplication(sleep_time=0.1)),
        ('multiplication_5', Multiplication(sleep_time=0.1)),
        ('checkpoint_2', PickleCheckpointStep()),
        ('multiplication_6', Multiplication(sleep_time=0.1)),
    ])

    resumable_pipeline = ResumablePipeline([
        ('multiplication_2', Multiplication(sleep_time=0.1)),
        ('multiplication_3', Multiplication(sleep_time=0.1)),
        ('checkpoint_1', PickleCheckpointStep()),
        ('multiplication_4', Multiplication(sleep_time=0.1)),
        ('multiplication_5', Multiplication(sleep_time=0.1)),
        ('checkpoint_2', PickleCheckpointStep()),
        ('multiplication_6', Multiplication(sleep_time=0.1)),
    ])

    data_inputs = np.array(range(10))
    expected_outputs = np.array(range(10, 20))

    print('Classic Pipeline')
    test_automl_pipeline(pipeline, hyperparams_space, data_inputs, expected_outputs)

    print('\n')

    print('Resumable Pipeline')
    test_automl_pipeline(resumable_pipeline, hyperparams_space, data_inputs, expected_outputs)

    shutil.rmtree(DEFAULT_CACHE_FOLDER)


def test_automl_pipeline(pipeline, hyperparams_space, data_inputs, expected_outputs):
    time_a = time.time()

    pipeline = AutoMLSequentialWrapper(
        pipeline=pipeline,
        n_iters=100,
        hyperparameters_space=hyperparams_space,
        objective_function=mean_squared_error
    ).fit(data_inputs, expected_outputs)

    pipeline, outputs = pipeline.fit_transform(data_inputs, expected_outputs)

    time_b = time.time()

    actual_score = mean_squared_error(outputs, expected_outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest error: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))


def mean_squared_error(actual_outputs, expected_outputs):
    mses = [(a - b) ** 2 for a, b in zip(actual_outputs, expected_outputs)]
    return sum(mses) / float(len(mses))


if __name__ == '__main__':
    main()
