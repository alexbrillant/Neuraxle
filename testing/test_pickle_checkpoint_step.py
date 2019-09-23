import os
import pickle

import numpy as np
from py._path.local import LocalPath

from neuraxle.base import DataContainer
from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep, OutputTransformerWrapper
from testing.steps.test_output_transformer_wrapper import MultiplyBy2BaseOutputTransformer

EXPECTED_TAPE_AFTER_CHECKPOINT = ["2", "3"]

data_inputs = np.array([1, 2])
expected_outputs = np.array([2, 3])
expected_rehashed_data_inputs = ['44f9d6dd8b6ccae571ca04525c3eaffa', '898a67b2f5eeae6393ca4b3162ba8e3d']


def create_pipeline(pickle_checkpoint_step, tape, hyperparameters=None):
    pipeline = Pipeline(
        steps=[
            ('a',
             TransformCallbackStep(tape.callback, ["1"], hyperparams=hyperparameters)),
            ('pickle_checkpoint', pickle_checkpoint_step),
            ('c', TransformCallbackStep(tape.callback, ["2"])),
            ('d', TransformCallbackStep(tape.callback, ["3"]))
        ]
    )
    return pipeline


def test_when_no_hyperparams_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    pipeline = create_pipeline(pickle_checkpoint_step, tape)

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(0))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(1))


def test_when_hyperparams_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    pipeline = create_pipeline(pickle_checkpoint_step, tape, HyperparameterSamples({"a__learning_rate": 1}))

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[0]))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[1]))


def test_when_no_hyperparams_should_load_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    setup_pickle_checkpoint(
        current_id=0,
        data_input=data_inputs[0],
        expected_output=expected_outputs[0],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    setup_pickle_checkpoint(
        current_id=1,
        data_input=data_inputs[1],
        expected_output=expected_outputs[1],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    pipeline = create_pipeline(pickle_checkpoint_step, tape)

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def test_when_hyperparams_should_load_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    setup_pickle_checkpoint(
        current_id=expected_rehashed_data_inputs[0],
        data_input=data_inputs[0],
        expected_output=expected_outputs[0],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    setup_pickle_checkpoint(
        current_id=expected_rehashed_data_inputs[1],
        data_input=data_inputs[1],
        expected_output=expected_outputs[1],
        pickle_checkpoint_step=pickle_checkpoint_step
    )

    pipeline = create_pipeline(pickle_checkpoint_step, tape, HyperparameterSamples({"a__learning_rate": 1}))

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def setup_pickle_checkpoint(current_id, data_input, expected_output, pickle_checkpoint_step):
    with open(pickle_checkpoint_step.get_checkpoint_file_path(current_id), 'wb') as file:
        pickle.dump((current_id, data_input, expected_output), file)


def create_pickle_checkpoint_step(tmpdir):
    pickle_checkpoint_step = PickleCheckpointStep(cache_folder=tmpdir)
    pickle_checkpoint_step.set_checkpoint_path(os.path.join('Pipeline', 'pickle_checkpoint'))

    return pickle_checkpoint_step


def test_pickle_checkpoint_step_should_load_data_container(tmpdir: LocalPath):
    initial_data_inputs = [1, 2]
    initial_expected_outputs = [2, 3]

    create_pipeline_output_transformer = lambda: Pipeline(
        [
            ('output_transformer', OutputTransformerWrapper(MultiplyBy2BaseOutputTransformer())),
            ('pickle_checkpoint', create_pickle_checkpoint_step(tmpdir)),
            ('output_transformer', OutputTransformerWrapper(MultiplyBy2BaseOutputTransformer())),
        ]
    )

    create_pipeline_output_transformer().fit_transform(
        data_inputs=initial_data_inputs, expected_outputs=initial_expected_outputs
    )
    actual_data_container = create_pipeline_output_transformer().handle_transform(
        DataContainer(current_ids=[0, 1], data_inputs=initial_data_inputs, expected_outputs=initial_expected_outputs)
    )

    assert np.array_equal(actual_data_container.data_inputs, [4, 8])
    assert np.array_equal(actual_data_container.expected_outputs, [8, 12])
