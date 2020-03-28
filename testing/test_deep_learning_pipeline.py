import math

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from neuraxle.api import DeepLearningPipeline
from neuraxle.metaopt.deprecated import RandomSearch
from neuraxle.steps.sklearn import SKLearnWrapper

N_ITER = 1

TIMESTEPS = 10

VALIDATION_SIZE = 0.1
BATCH_SIZE = 32
N_EPOCHS = 15

DATA_INPUTS_PAST_SHAPE = (BATCH_SIZE, TIMESTEPS)


def test_deep_learning_pipeline():
    # Given
    data_inputs, expected_outputs = create_2d_data()

    p = DeepLearningPipeline(
        SKLearnWrapper(linear_model.LinearRegression()),
        validation_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=N_EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error),
    )

    # When
    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    metrics = p.apply('get_metrics')

    # Then
    batch_mse_train = metrics['DeepLearningPipeline__EpochRepeater__validation_split_wrapper__epoch_metrics']['train']['mse']
    epoch_mse_train = metrics['DeepLearningPipeline__EpochRepeater__validation_split_wrapper__epoch_metrics__TrainShuffled__MiniBatchSequentialPipeline__batch_metrics']['train']['mse']

    batch_mse_validation = metrics['DeepLearningPipeline__EpochRepeater__validation_split_wrapper__epoch_metrics__TrainShuffled__MiniBatchSequentialPipeline__batch_metrics']['validation']['mse']
    epoch_mse_validation = metrics['DeepLearningPipeline__EpochRepeater__validation_split_wrapper__epoch_metrics']['validation']['mse']

    assert len(epoch_mse_train) == N_EPOCHS
    assert len(epoch_mse_validation) == N_EPOCHS

    expected_len_batch_mse = math.ceil((len(data_inputs) / BATCH_SIZE) * (1 - VALIDATION_SIZE)) * N_EPOCHS

    assert len(batch_mse_train) == expected_len_batch_mse
    assert len(batch_mse_validation) == expected_len_batch_mse


def test_deep_learning_pipeline_with_random_search():
    # Given
    data_inputs, expected_outputs = create_2d_data()

    p = RandomSearch(DeepLearningPipeline(
        SKLearnWrapper(linear_model.LinearRegression()),
        batch_size=BATCH_SIZE,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=N_EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error),
        validation_size=0.15
    ), n_iter=N_ITER)

    # When
    p, outputs = p.fit_transform(data_inputs, expected_outputs)
    best_model = p.get_best_model()
    best_model.set_train(False)
    best_model.apply('disable_metrics')

    # Then
    outputs = best_model.transform(data_inputs)

    mse = ((outputs - expected_outputs) ** 2).mean()
    assert mse < 1.5


def create_2d_data():
    i = 0
    data_inputs = []
    for batch_index in range(BATCH_SIZE):
        batch = []
        for _ in range(TIMESTEPS):
            batch.append(i)
            i += 1
        data_inputs.append(batch)

    data_inputs = np.array(data_inputs)
    random_noise = np.random.random(DATA_INPUTS_PAST_SHAPE)

    expected_outputs = 3 * data_inputs + 4 * random_noise
    expected_outputs = expected_outputs.astype(np.float32)

    data_inputs = data_inputs.astype(np.float32)

    return data_inputs, expected_outputs


def to_numpy_metric_wrapper(metric_fun):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs), np.array(expected_outputs))

    return metric
