# Copyright 2019, The NeurAxle Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from neuraxle.base import BaseStep
from neuraxle.pipeline import Pipeline, BlockPipelineRunner

AN_INPUT = "I am an input"
AN_EXPECTED_OUTPUT = "I am an expected output"


class SomeStep(BaseStep):
    def get_default_hyperparams(self):
        return dict()

    def fit_one(self, data_input, expected_output=None):
        return self

    def transform_one(self, data_input):
        return AN_EXPECTED_OUTPUT


steps_lists = [
    [("just_one_step", SomeStep())],
    [
        ("some_step_1", SomeStep()),
        ("some_step_2", SomeStep()),
        ("some_step_3", SomeStep())
    ]
]

pipeline_runners = [BlockPipelineRunner]  # TODO: streaming pipeline runner


@pytest.mark.parametrize("steps_list", steps_lists)
@pytest.mark.parametrize("pipeline_runner", pipeline_runners)
def test_pipeline_fit_transform(steps_list, pipeline_runner):
    data_input_ = [AN_INPUT]
    expected_output_ = [AN_EXPECTED_OUTPUT]
    p = Pipeline(steps_list, pipeline_runner=pipeline_runner())

    result = p.fit_transform(data_input_, expected_output_)

    assert tuple(result) == tuple(expected_output_)


@pytest.mark.parametrize("steps_list", steps_lists)
@pytest.mark.parametrize("pipeline_runner", pipeline_runners)
def test_pipeline_fit_then_transform(steps_list, pipeline_runner):
    data_input_ = [AN_INPUT]
    expected_output_ = [AN_EXPECTED_OUTPUT]
    p = Pipeline(steps_list, pipeline_runner=pipeline_runner())

    p.fit(data_input_, expected_output_)
    result = p.transform(data_input_)

    assert tuple(result) == tuple(expected_output_)
