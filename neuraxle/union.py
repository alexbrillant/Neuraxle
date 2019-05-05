from neuraxle.base import BaseStep, TruncableSteps, NonFittableMixin, NamedTupleList
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class ModelStacking(TruncableSteps):
    def __init__(self, brothers: NamedTupleList, judge: BaseStep, joiner: BaseStep):
        super().__init__(brothers)
        self.joiner: BaseStep = joiner
        self.judge: BaseStep = judge

    def fit(self, data_inputs, expected_outputs=None):
        # TODO: FeatureUnion?
        results = []
        for _, bro in self.steps_as_tuple:
            res = bro.fit_transform(data_inputs, expected_outputs)
            results.append(res)

        results = self.joiner.fit_transform(results)

        self.judge.fit(results, expected_outputs)
        return self

    def transform(self, data_inputs):
        results = []
        for name, bro in self.steps_as_tuple:
            res = bro.transform(data_inputs)
            results.append(res)

        results = self.joiner.transform(results)

        return self.judge.transform(results)


class FeatureUnion(TruncableSteps):
    # TODO: parallel.
    def __init__(self, steps_as_tuple: NamedTupleList, joiner: NonFittableMixin = NumpyConcatenateInnerFeatures()):
        super().__init__(steps_as_tuple)
        self.joiner = joiner  # TODO: add "other" types of step(s) to TuncableSteps or to another intermediate class.

    def fit(self, data_inputs, expected_outputs=None):
        for _, bro in self.steps_as_tuple:
            bro.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        results = []
        for _, bro in self.steps_as_tuple:
            res = bro.transform(data_inputs)
            results.append(res)
        results = self.joiner.transform(results)
        return results


class Identity(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        return data_inputs

    def transform_one(self, data_input):
        return data_input


class AddFeatures(FeatureUnion):
    def __init__(self, steps_as_tuple: NamedTupleList, **kwargs):
        super().__init__([Identity()] + steps_as_tuple, **kwargs)
