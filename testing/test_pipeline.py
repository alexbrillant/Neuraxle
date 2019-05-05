# Copyright 2019, The Neuraxle Authors
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
from neuraxle.steps.sklearn import SKLearnWrapper

AN_INPUT = "I am an input"
AN_EXPECTED_OUTPUT = "I am an expected output"


class SomeStep(BaseStep):
    def get_hyperparams_space(self):
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


def test_pipeline_slicing_before():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p["b":]

    assert "a" not in r
    assert "b" in r
    assert "c" in r


def test_pipeline_slicing_after():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p[:"c"]

    assert "a" in r
    assert "b" in r
    assert "c" not in r


def test_pipeline_slicing_both():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p["b":"c"]

    assert "a" not in r
    assert "b" in r
    assert "c" not in r


def test_pipeline_set_one_hyperparam_level_one_flat():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "a__learning_rate": 7
    })

    assert p["a"].hyperparams["learning_rate"] == 7
    assert p["b"].hyperparams == dict()
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_one_dict():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b": {
            "learning_rate": 7
        }
    })

    assert p["a"].hyperparams == dict()
    assert p["b"].hyperparams["learning_rate"] == 7
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_two_flat():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", Pipeline([
            ("a", SomeStep()),
            ("b", SomeStep()),
            ("c", SomeStep())
        ])),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b__a__learning_rate": 7
    })
    print(p.get_hyperparams())

    assert p["b"]["a"].hyperparams["learning_rate"] == 7
    assert p["b"]["c"].hyperparams == dict()
    assert p["b"].hyperparams == dict()
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_two_dict():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", Pipeline([
            ("a", SomeStep()),
            ("b", SomeStep()),
            ("c", SomeStep())
        ])),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b": {
            "a": {
                "learning_rate": 7
            },
            "learning_rate": 9
        }
    })
    print(p.get_hyperparams())

    assert p["b"]["a"].hyperparams["learning_rate"] == 7
    assert p["b"]["c"].hyperparams == dict()
    assert p["b"].hyperparams["learning_rate"] == 9
    assert p["c"].hyperparams == dict()


def test_pipeline_tosklearn():
    import sklearn.pipeline
    the_step = SomeStep()
    step_to_check = the_step.tosklearn()

    p = Pipeline([
        ("a", SomeStep()),
        ("b", SKLearnWrapper(sklearn.pipeline.Pipeline([
            ("a", sklearn.pipeline.Pipeline([
                ('z', step_to_check)
            ])),
            ("b", SomeStep().tosklearn()),
            ("c", SomeStep().tosklearn())
        ]), return_all_sklearn_default_params_on_get=True)),
        ("c", SomeStep())
    ])

    # assert False
    p.set_hyperparams({
        "b": {
            "a__z__learning_rate": 7,
            "b__learning_rate": 9
        }
    })
    assert the_step.get_hyperparams()["learning_rate"] == 7

    p = p.tosklearn()
    p = sklearn.pipeline.Pipeline([('sk', p)])

    p.set_params(**{"sk__b__a__z__learning_rate": 11})
    assert p.named_steps["sk"].p["b"].wrapped_sklearn_predictor.named_steps["a"]["z"]["learning_rate"] == 11

    # p.set_params(**dict_to_flat({
    #     "sk__b": {
    #         "a__z__learning_rate": 12,
    #         "b__learning_rate": 9
    #     }
    # }))
    p.set_params(**{"sk__b__a__z__learning_rate": 12})
    assert p.named_steps["sk"].p["b"].wrapped_sklearn_predictor.named_steps["a"]["z"]["learning_rate"] == 12
    # assert the_step.get_hyperparams()["learning_rate"] == 12  # TODO: debug why wouldn't this work
