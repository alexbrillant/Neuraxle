from neuraxle.base import MetaStepMixin, BaseStep


class MetaSKLearnWrapper(MetaStepMixin, BaseStep):

    def __init__(self, wrapped: 'MetaEstimatorMixin'):
        self.wrapped_sklearn_metaestimator = wrapped
        # sklearn.model_selection.RandomizedSearchCV
        super().__init__()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        self.wrapped_sklearn_metaestimator.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        return self.wrapped_sklearn_metaestimator.transform(data_inputs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_ = self.__class__
        module = type_.__module__
        qualname = type_.__qualname__
        wrappedname = str(self.wrapped_sklearn_metaestimator.__class__.__name__)
        return "<{}.{}({}(...)) object {}>".format(module, qualname, wrappedname, hex(id(self)))
